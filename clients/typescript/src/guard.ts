// Decision steps of the guard_input / guard_output / route_intent presets; the same logic as guard() and route() in
// basal/presets.py (test/presets_fixture.json holds shared cases).
import { POLICIES, PRECEDENCE, PRESETS } from "./presets.js";
import type { GuardAction, GuardCheck, GuardDecision, GuardPolicy, RouteDecision } from "./types.js";

type Loose = Record<string, any> | null | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any

/** A probability, or null when it is not a finite number in [0, 1]. */
function prob(x: unknown): number | null {
  return typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1 ? x : null;
}

/** The most probable level of a score answer (0 = the first), null when its probabilities are not a non-empty
 *  {level index: probability} map. */
function level(ans: Record<string, any>): number | null { // eslint-disable-line @typescript-eslint/no-explicit-any
  const probs = ans.probabilities;
  if (!probs || typeof probs !== "object" || Array.isArray(probs)) return null;
  const keys = Object.keys(probs);
  if (!keys.length || keys.some((k) => !/^\d+$/.test(k) || prob(probs[k]) === null)) return null;
  return Number(keys.reduce((a, b) => (probs[b] > probs[a] ? b : a)));
}

/** The question names of the guard_input ("input") or guard_output ("output") battery. */
export function battery(kind: "input" | "output"): string[] {
  const set = (PRESETS as Record<string, Record<string, Record<string, unknown>>>)[`guard_${kind}`]?.en;
  if (!set) throw new Error(`unknown battery ${JSON.stringify(kind)} (input, output, or a list of question names)`);
  return Object.keys(set);
}

/** Answers of a guard battery -> pass / review / block / support (precedence support > block > review > pass).
 *  Every question of the battery must be answered: `expect` is "input" / "output", a list of question names (your own
 *  battery, or an intentional part of one), or undefined (the built-in battery of the answers: output when they hold
 *  policy_break and no jailbreak, else input). A missing, null or malformed answer (a yes/no probability that is not a
 *  finite number in [0, 1], a severity without a {level: probability} map) is a review. With `maxError` (ask the
 *  questions with the same max_error) a hazard is hit at p >= 0.5 and an answer marked `abstain`, or without a
 *  true / false abstain flag, is a review: the thresholds are those of the model's calibration. */
export function guard(
  answers: Record<string, Loose>,
  policy: string | GuardPolicy = "strict",
  maxError?: number,
  expect?: "input" | "output" | string[],
): GuardDecision {
  const pol = typeof policy === "string" ? POLICIES[policy] : policy;
  if (!pol) throw new Error(`unknown policy ${JSON.stringify(policy)} (policies: ${Object.keys(POLICIES).sort().join(", ")})`);
  if (!answers || typeof answers !== "object") throw new Error("answers must be an object of question name -> answer");
  const certified = maxError !== undefined && maxError !== null;
  const kind = expect ?? ("policy_break" in answers && !("jailbreak" in answers) ? "output" : "input");
  const names = typeof kind === "string" ? battery(kind as "input" | "output") : kind.map(String);
  const checks: Record<string, GuardCheck> = {};
  let worst: GuardAction = "pass";
  const put = (name: string, action: GuardAction, p: number | null, why?: string) => {
    checks[name] = why ? { action, p, why } : { action, p };
    if (PRECEDENCE.indexOf(action) < PRECEDENCE.indexOf(worst)) worst = action;
  };
  const all: [string, Loose][] = [...Object.entries(answers), ...names.filter((n) => !(n in answers)).map((n) => [n, null] as [string, Loose])];
  for (const [name, ans] of all) {
    if (!ans || typeof ans !== "object") { put(name, "review", null, "no answer"); continue; }
    if (certified && typeof ans.abstain !== "boolean") { put(name, "review", null, "no abstain flag: ask with max_error"); continue; }
    if (name === "severity") {
      const lvl = level(ans);
      if (lvl === null) { put(name, "review", null, "malformed answer"); continue; }
      let act: GuardAction = lvl >= pol.severity.block ? "block" : lvl >= pol.severity.review ? "review" : "pass";
      if (certified && ans.abstain && act === "pass") act = "review";
      put(name, act, lvl);
      continue;
    }
    const p = prob(ans.noul);
    if (p === null) { put(name, "review", null, "malformed answer"); continue; }
    const hazard = pol.hazards[name];
    if (hazard === undefined || hazard === "pass") { put(name, "pass", p); continue; }
    const act: GuardAction = certified
      ? ans.abstain ? "review" : p >= 0.5 ? hazard : "pass"
      : p >= pol.act_at ? hazard : p >= pol.review_at ? "review" : "pass";
    put(name, act, p);
  }
  const reason = worst === "pass" ? null : (Object.entries(checks).find(([, c]) => c.action === worst)?.[0] ?? null);
  return { action: worst, reason, checks, mode: certified ? "certified" : "fixed" };
}

/** Answers of route_intent -> the intent, or "human" on low confidence (below `minConfidence`), abstain (with
 *  `maxError`; an answer without a true / false abstain flag too), a missing or a malformed answer. */
export function route(answers: Record<string, Loose>, minConfidence = 0.6, maxError?: number): RouteDecision {
  const intent = answers.intent, comp = answers.complexity;
  if (!intent || !comp || typeof intent !== "object" || typeof comp !== "object") return { route: "human", why: "no answer" };
  const lvl = level(comp), conf = prob(intent.confidence);
  if (lvl === null || conf === null || typeof intent.choice !== "string") return { route: "human", why: "malformed answer" };
  const out: RouteDecision = {
    route: intent.choice, complexity: (comp.legend ?? {})[String(lvl)], level: lvl, confidence: conf,
  };
  if (maxError !== undefined && maxError !== null) {
    if (typeof intent.abstain !== "boolean" || typeof comp.abstain !== "boolean") return { route: "human", why: "no abstain flag: ask with max_error" };
    if (intent.abstain || comp.abstain) return { ...out, route: "human", why: "abstain" };
  } else if (conf < minConfidence) return { ...out, route: "human", why: "low confidence" };
  return out;
}
