import { BasalError, errorFromResponse } from "./errors.js";
import { guard, route } from "./guard.js";
import { PRESETS } from "./presets.js";
import type {
  Answer,
  AnswersFor,
  ChoiceAnswer,
  Criteria,
  DecisionResponse,
  Health,
  ImageInput,
  GuardDecision,
  GuardPolicy,
  ModelInfo,
  NoulAnswer,
  Questions,
  RequestOptions,
  RouteDecision,
  ScoreAnswer,
} from "./types.js";

export interface BasalOptions {
  /** Default "http://127.0.0.1:8000". */
  url?: string;
  /** Sends "Authorization: Bearer <key>" (servers behind a proxy, other /v1/systemone servers). */
  apiKey?: string;
  /** Seconds, default 60: the whole call, the response body included (a number > 0). */
  timeout?: number;
  /** Default early_exit of every request (--mode fast-exit servers, basal-1.0). */
  earlyExit?: RequestOptions["early_exit"];
  /** Default facts of every request: "auto" appends computed calendar / arithmetic facts to a Polish state. */
  facts?: RequestOptions["facts"];
  /** Extra headers. */
  headers?: Record<string, string>;
  /** A fetch implementation (default: the global fetch of Node >= 18, browsers, Deno, Bun). */
  fetch?: typeof fetch;
}

/** A noul answer that carries only "noul" (another server's) gets "probabilities" and "confidence", as basal-serve
 *  computes them; every other answer (basal-serve's, or null for a question another server skipped) is returned as
 *  it is. */
export function complete<A>(answer: A): A {
  const a = answer as unknown as Partial<NoulAnswer> | null;
  if (a && typeof a === "object" && a.type === "noul" && a.probabilities === undefined && a.noul !== undefined) {
    const p = Number(a.noul);
    return { ...a, probabilities: { true: p, false: 1 - p }, confidence: Math.max(p, 1 - p) } as unknown as A;
  }
  return answer;
}

const PL = /[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]|(?<![\p{L}\p{N}_])(?:i|w|z|nie|się|jest|na|do|że|od)(?![\p{L}\p{N}_])/gu;

/** "pl" when the text looks Polish (diacritics, common words), else "en" (as basal/presets.py). */
export function langOf(text: string): "pl" | "en" {
  return [...(text ?? "").matchAll(PL)].length >= 2 ? "pl" : "en";
}

/** Names of the ready question sets. */
export function presetNames(): string[] {
  return Object.keys(PRESETS).sort();
}

/** The questions of preset `name` in `lang` (default: the language of `state`), as a fresh object. */
export function presetQuestions(name: string, lang?: "pl" | "en", state = "", maxError?: number): Questions {
  const preset = PRESETS[name];
  if (!preset) throw new Error(`unknown preset ${JSON.stringify(name)} (presets: ${presetNames().join(", ")})`);
  const l = lang ?? langOf(state);
  const qs = preset[l];
  if (!qs) throw new Error(`preset ${JSON.stringify(name)} has no ${JSON.stringify(l)} wording (pl, en)`);
  const out = structuredClone(qs);
  if (maxError !== undefined)
    for (const q of Object.values(out)) if (q.type === "choice" || q.type === "noul" || q.type === "score") q.max_error = maxError;
  return out;
}

/** An image from bytes (Uint8Array) as a "data:image/...;base64," URL, for `images`. */
export function imageFromBytes(bytes: Uint8Array, contentType = "image/png"): ImageInput {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${contentType};base64,${btoa(bin)}`;
}

type CallOptions = RequestOptions & { signal?: AbortSignal };

export class Basal {
  readonly url: string;
  readonly origin: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;
  private readonly defaults: RequestOptions;

  constructor(options: BasalOptions = {}) {
    this.origin = (options.url ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
    this.url = `${this.origin}/v1/systemone`;
    const timeout = options.timeout ?? 60;
    if (typeof timeout !== "number" || !Number.isFinite(timeout) || timeout <= 0 || timeout * 1000 > 2 ** 32 - 1)
      throw new RangeError(`timeout: a number of seconds > 0 (49 days at most), not ${String(timeout)}`);
    this.timeoutMs = Math.max(1, Math.round(timeout * 1000));  // AbortSignal.timeout takes whole milliseconds
    this.headers = { "content-type": "application/json", ...(options.headers ?? {}) };
    if (options.apiKey) this.headers["authorization"] = `Bearer ${options.apiKey}`;
    this.fetchImpl = options.fetch ?? ((...a) => fetch(...a));
    this.defaults = {};
    if (options.earlyExit) this.defaults.early_exit = options.earlyExit;
    if (options.facts) this.defaults.facts = options.facts;
  }

  private async call(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<unknown> {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const sig = signal ? AbortSignal.any([signal, timeout]) : timeout;
    let r: Response;
    let text: string;
    try {  // the body too: it can stall or break after the headers came
      r = await this.fetchImpl(`${this.origin}${path}`, {
        method,
        headers: this.headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: sig,
      });
      text = await r.text();
    } catch (e) {
      if (timeout.aborted) throw new BasalError("timeout", `no answer in ${this.timeoutMs / 1000} s`, { cause: e });
      if (signal?.aborted) throw e;
      throw new BasalError("network", e instanceof Error ? e.message : String(e), { cause: e });
    }
    if (r.status !== 200) throw errorFromResponse(r.status, text);
    try {
      return JSON.parse(text);
    } catch (e) {
      throw new BasalError("server", "the response is not JSON", { status: r.status, body: text, cause: e });
    }
  }

  /** The whole response of POST /v1/systemone: model, answers and usage. */
  async request<const Q extends Questions>(
    state: unknown,
    questions: Q,
    options: CallOptions = {},
  ): Promise<DecisionResponse<AnswersFor<Q>>> {
    const { signal, ...fields } = options;
    const body = { state, questions, ...this.defaults, ...fields };
    const res = (await this.call("POST", "/v1/systemone", body, signal)) as DecisionResponse;
    const answers = Object.fromEntries(Object.entries(res.answers ?? {}).map(([k, a]) => [k, complete(a)]));
    return { ...res, answers } as DecisionResponse<AnswersFor<Q>>;
  }

  /** The answers to named questions about `state` (text or JSON). A question skipped by ask_if has the answer null. */
  async decide<const Q extends Questions>(state: unknown, questions: Q, options: CallOptions = {}): Promise<AnswersFor<Q>> {
    return (await this.request(state, questions, options)).answers;
  }

  /** One choice question. `criteria`: {key: description}, or a list of keys. */
  async choice(
    state: unknown,
    question: string,
    criteria: Criteria,
    extra: { coverage?: number; max_error?: number; option_keys?: "show" | "hide" } & CallOptions = {},
  ): Promise<ChoiceAnswer> {
    const { coverage, max_error, option_keys, ...options } = extra;
    // basal-serve reads a list of keys as the map {key: null}; other servers need the map
    const crit = Array.isArray(criteria) ? Object.fromEntries(criteria.map((k) => [k, null])) : criteria;
    const q = defined({ type: "choice", instructions: question, criteria: crit, coverage, max_error, option_keys });
    return (await this.decide(state, { q } as Questions, options))["q"] as ChoiceAnswer;
  }

  /** One yes/no question; `sides.yes` and `sides.no` describe the two outcomes (optional). */
  async yesNo(
    state: unknown,
    question: string,
    sides: { yes?: string; no?: string } = {},
    extra: { coverage?: number; max_error?: number } & CallOptions = {},
  ): Promise<NoulAnswer> {
    const { coverage, max_error, ...options } = extra;
    const criteria: Record<string, string> = {};
    if (sides.yes) criteria["true"] = sides.yes;
    if (sides.no) criteria["false"] = sides.no;
    const q = defined({ type: "noul", instructions: question, criteria, coverage, max_error });
    return (await this.decide(state, { q } as Questions, options))["q"] as NoulAnswer;
  }

  /** One score question; `levels` lowest first. */
  async score(
    state: unknown,
    question: string,
    levels: string[],
    extra: { coverage?: number; max_error?: number } & CallOptions = {},
  ): Promise<ScoreAnswer> {
    const { coverage, max_error, ...options } = extra;
    const q = defined({ type: "score", instructions: question, criteria: levels, coverage, max_error });
    return (await this.decide(state, { q } as Questions, options))["q"] as ScoreAnswer;
  }

  /** The answers to a ready question set ("triage", "moderation", "compliance") about `state`. */
  async preset(state: string, name: string, lang?: "pl" | "en", maxError?: number): Promise<Record<string, Answer | null>> {
    return this.decide(state, presetQuestions(name, lang, state, maxError));
  }

  /** Runs the "guard_input" / "guard_output" battery on `state` and decides pass / review / block / support
   *  (see guard() in basal/presets.py). `maxError` takes the thresholds from the model's calibration. */
  async guard(
    state: string,
    opts: { kind?: "input" | "output"; policy?: string | GuardPolicy; lang?: "pl" | "en"; maxError?: number } = {},
  ): Promise<GuardDecision & { answers: Record<string, Answer | null> }> {
    const answers = await this.preset(state, `guard_${opts.kind ?? "input"}`, opts.lang, opts.maxError);
    return { ...guard(answers, opts.policy ?? "strict", opts.maxError, opts.kind ?? "input"), answers };
  }

  /** Runs "route_intent" on `state`: the intent, or "human" on low confidence or abstain. */
  async route(
    state: string,
    opts: { lang?: "pl" | "en"; minConfidence?: number; maxError?: number } = {},
  ): Promise<RouteDecision & { answers: Record<string, Answer | null> }> {
    const answers = await this.preset(state, "route_intent", opts.lang, opts.maxError);
    return { ...route(answers, opts.minConfidence, opts.maxError), answers };
  }

  /** GET /v1/models. */
  async models(signal?: AbortSignal): Promise<ModelInfo[]> {
    return ((await this.call("GET", "/v1/models", undefined, signal)) as { models: ModelInfo[] }).models;
  }

  /** GET /health: how many decision requests run and wait; both 0: the server is idle. */
  async health(signal?: AbortSignal): Promise<Health> {
    return (await this.call("GET", "/health", undefined, signal)) as Health;
  }
}

function defined<T extends Record<string, unknown>>(o: T): T {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as T;
}
