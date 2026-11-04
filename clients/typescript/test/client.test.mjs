// The TypeScript client against a real basal-serve (the repo's stub engine; see stub_server.py).
import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";
import { readFileSync } from "node:fs";

import { Basal, BasalError, POLICIES, complete, guard, imageFromBytes, langOf, presetNames, presetQuestions, route } from "../dist/index.js";
import { startServer } from "./helpers.mjs";

let server, auth, openai, b;
before(async () => {
  [server, auth, openai] = await Promise.all([
    startServer(),
    startServer(["--api-key", "k1"]),
    startServer(["--error-format", "openai"]),
  ]);
  b = new Basal({ url: server.url });
});
after(() => [server, auth, openai].forEach((s) => s?.stop()));

const CHOICE = { a: "first ~0.6", b: "second ~0.3", c: "third ~0.1" };  // the stub reads the weights at the end

describe("decide against basal-serve", () => {
  test("choice answers with probabilities and a confidence; a list of keys is the same options", async () => {
    const a = await b.choice("Faktura nie zapłacona", "Which?", CHOICE);
    assert.equal(a.type, "choice");
    assert.equal(a.choice, "a");
    assert.ok(Math.abs(a.probabilities.a - 0.6) < 1e-6 && Math.abs(a.confidence - 0.6) < 1e-6);
    assert.ok(Math.abs(Object.values(a.probabilities).reduce((x, y) => x + y) - 1) < 1e-6);
    const keys = await b.choice("x", "Which?", ["a", "b"]);  // no weights: the stub shares them equally
    assert.deepEqual(Object.keys(keys.probabilities), ["a", "b"]);
  });

  test("yesNo and score", async () => {
    const y = await b.yesNo("x", "Urgent?", { yes: "yes ~0.97", no: "no ~0.03" });
    assert.equal(y.type, "noul");
    assert.ok(Math.abs(y.noul - 0.97) < 1e-6 && Math.abs(y.probabilities.false - 0.03) < 1e-6);
    const s = await b.score("x", "How much?", ["low ~0.2", "mid ~0.5", "high ~0.3"]);
    assert.equal(s.type, "score");
    assert.deepEqual(s.legend, { 0: "low ~0.2", 1: "mid ~0.5", 2: "high ~0.3" });
    assert.ok(Math.abs(s.score - (0.5 + 0.6)) < 1e-6);  // expected level 0*0.2 + 1*0.5 + 2*0.3
  });

  test("several questions over one state, multi, act and tool", async () => {
    const r = await b.request("one state", {
      c: { type: "choice", instructions: "Which?", criteria: CHOICE },
      m: { type: "multi", instructions: "Which apply?", criteria: { a: "A ~0.9", b: "B ~0.2" } },
      act: {
        type: "act",
        instructions: "Approve?",
        criteria: { true: "yes ~0.8", false: "no ~0.2" },
        costs: { approve: { true: 0, false: 10 }, human: { true: 1, false: 1 } },
      },
      t: {
        type: "tool",
        tools: [{ type: "function", function: { name: "lookup", description: "d ~0.7", parameters: { type: "object", properties: {} } } }],
        extra: ["none"],
      },
    });
    assert.equal(r.model, "basal-stub");
    assert.equal(r.usage.questions, 4);
    assert.deepEqual(r.answers.m.selected, ["a"]);
    assert.ok(Math.abs(r.answers.m.probabilities.a - 0.9) < 1e-6);
    assert.equal(r.answers.act.action, "human");  // expected cost of approve 2.0 > human 1.0
    assert.equal(r.answers.t.tool, "lookup");
  });

  test("answer sets (coverage) and abstain (max_error)", async () => {
    const a = await b.choice("x", "Which?", CHOICE, { coverage: 0.9, max_error: 0.05 });
    assert.deepEqual(a.answer_set, ["a", "b"]);
    assert.equal(a.coverage, 0.9);
    assert.equal(a.abstain, true);  // 0.6 < the certified 0.7
    const sure = await b.choice("x", "Which?", { a: "first ~0.9", b: "second ~0.1" }, { max_error: 0.05 });
    assert.equal(sure.abstain, false);
    const none = await b.choice("x", "Which?", { a: "first ~0.9", b: "second ~0.1" }, { max_error: 0.01 });
    assert.equal(none.abstain, true);
    assert.match(none.max_error, /certified|threshold/i);  // a note, not a number
    const noul = await b.yesNo("x", "Yes?", { yes: "yes ~0.97", no: "no ~0.03" }, { coverage: 0.9 });
    assert.deepEqual(noul.answer_set, ["true"]);
    const levels = await b.score("x", "How much?", ["low ~0.2", "mid ~0.5", "high ~0.3"], { coverage: 0.9 });
    assert.equal(levels.answer_set.length, 3);  // no threshold for score: every level
    assert.ok(levels.coverage_note);
  });

  test("abstain on act: deferred", async () => {
    const r = await b.decide("x", {
      a: {
        type: "act",
        instructions: "Approve?",
        criteria: { true: "yes ~0.8", false: "no ~0.2" },
        costs: { approve: { true: 0, false: 1 }, human: { true: 1, false: 1 } },
        max_error: 0.05,
      },
    });
    assert.equal(r.a.deferred, true);
  });

  test("ask_if skips a question: the answer is null and usage.skipped names it", async () => {
    const r = await b.request("x", {
      c: { type: "choice", instructions: "Which?", criteria: { a: "A ~0.9", b: "B ~0.1" } },
      d: { type: "noul", instructions: "Then?", criteria: { true: "y ~0.7", false: "n ~0.3" }, ask_if: { c: ["b"] }, depends_on: ["c"] },
      e: { type: "noul", instructions: "Then?", criteria: { true: "y ~0.7", false: "n ~0.3" }, ask_if: { c: ["a"] } },
    });
    assert.equal(r.answers.d, null);
    assert.deepEqual(r.usage.skipped, ["d"]);
    assert.equal(r.answers.e.type, "noul");
  });

  test("models and health", async () => {
    const [m] = await b.models();
    assert.equal(m.name, "basal-stub");
    assert.equal(m.mode, "eager");
    assert.deepEqual(await b.health(), { status: "ok", running: 0, queued: 0 });
  });
});

describe("errors", () => {

  test("422: a request that must change", async () => {
    await assert.rejects(b.request("x", undefined), (e) => {
      assert.ok(e instanceof BasalError);
      assert.equal(e.kind, "validation");
      assert.equal(e.status, 422);
      assert.match(e.detail, /questions/);
      assert.equal(e.retryable, false);
      return true;
    });
    await assert.rejects(b.choice("x", "Which?", { a: "only one" }), (e) => e.kind === "validation");
  });

  test("a body the server cannot use through a raw field", async () => {
    await assert.rejects(b.decide("x", { q: { type: "choice", instructions: "w", criteria: CHOICE } }, { facts: "sometimes" }), (e) => e.status === 422 && /facts/.test(e.detail));
  });

  test("--error-format openai bodies are read as well", async () => {
    const o = new Basal({ url: openai.url });
    await assert.rejects(o.request("x", undefined), (e) => {
      assert.equal(e.status, 422);
      assert.equal(e.type, "validation_error");
      assert.match(e.detail, /questions/);
      return true;
    });
  });

  test("the bearer token is sent", async () => {
    await assert.rejects(new Basal({ url: auth.url }).choice("x", "Which?", CHOICE), (e) => e.status === 401);
    const a = await new Basal({ url: auth.url, apiKey: "k1" }).choice("x", "Which?", CHOICE);
    assert.equal(a.choice, "a");
  });

  test("images: a server without --media on refuses them (422); network and timeout errors", async () => {
    await assert.rejects(
      b.decide("x", { q: { type: "choice", instructions: "w", criteria: CHOICE } }, { images: ["data:image/png;base64,AAAA"] }),
      (e) => e.kind === "validation" && /media/.test(e.detail),
    );
    await assert.rejects(new Basal({ url: "http://127.0.0.1:1" }).health(), (e) => e.kind === "network" && e.retryable);
    const hang = (_u, init) => new Promise((_, rej) => init.signal.addEventListener("abort", () => rej(new Error("aborted"))));
    await assert.rejects(new Basal({ fetch: hang, timeout: 0.05 }).health(), (e) => e.kind === "timeout" && e.retryable);
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(b.health(ac.signal), (e) => !(e instanceof BasalError));  // the caller's own abort passes through
  });
});

describe("errors through the whole response (a fetch of the test's own)", () => {
  const respond = (body, status = 200) => async () => new Response(body, { status });
  const stream = (start) => new ReadableStream({ start });

  test("a body that stalls after the headers is a timeout, one that breaks is a network error", async () => {
    const stall = async (_u, init) => new Response(stream((c) => init.signal.addEventListener("abort", () => c.error(init.signal.reason))));
    await assert.rejects(new Basal({ fetch: stall, timeout: 0.05 }).health(), (e) => e instanceof BasalError && e.kind === "timeout" && e.retryable);
    const broken = respond(stream((c) => { c.enqueue(new TextEncoder().encode('{"sta')); c.error(new TypeError("socket hang up")); }));
    await assert.rejects(new Basal({ fetch: broken }).health(), (e) => e instanceof BasalError && e.kind === "network" && /socket hang up/.test(e.detail));
  });

  test("the caller's abort while the body is read passes through", async () => {
    const ac = new AbortController();
    const slow = async (_u, init) => new Response(stream((c) => {
      init.signal.addEventListener("abort", () => c.error(init.signal.reason));
      setTimeout(() => ac.abort(new Error("mine")), 10);
    }));
    await assert.rejects(new Basal({ fetch: slow }).health(ac.signal), (e) => !(e instanceof BasalError) && e.message === "mine");
  });

  test("statuses and malformed JSON keep their kinds; only 503 is retryable", async () => {
    await assert.rejects(new Basal({ fetch: respond("not json") }).health(), (e) => e.kind === "server" && e.status === 200 && !e.retryable);
    for (const [status, kind, retryable] of [[422, "validation", false], [500, "server", false], [503, "unavailable", true],
      [429, "server", false], [502, "server", false], [504, "server", false]])
      await assert.rejects(new Basal({ fetch: respond('{"error": "x"}', status) }).health(),
        (e) => e.status === status && e.kind === kind && e.retryable === retryable && e.detail === "x");
  });

  test("a timeout that is not a number of seconds > 0 is refused; a fraction of a millisecond is rounded", async () => {
    for (const timeout of [0, -1, NaN, Infinity, "5", 5e9]) assert.throws(() => new Basal({ timeout }), RangeError, String(timeout));
    const ok = respond('{"status": "ok", "running": 0, "queued": 0}');
    assert.equal((await new Basal({ fetch: ok, timeout: 0.0015 }).health()).status, "ok");
  });
});

describe("request bodies (a recording fetch)", () => {
  const recorder = (answers) => {
    const seen = [];
    const fetch = async (url, init) => {
      seen.push({ url, init, body: init.body ? JSON.parse(init.body) : undefined });
      return new Response(JSON.stringify({ model: "m", answers }), { status: 200 });
    };
    return { seen, fetch };
  };

  test("the same shapes as the Python client", async () => {
    const { seen, fetch } = recorder({ q: { type: "noul", noul: 0.8, probabilities: { true: 0.8, false: 0.2 }, confidence: 0.8 } });
    const c = new Basal({ url: "http://server/", fetch, earlyExit: "0.99", facts: "auto", apiKey: "k" });
    await c.yesNo("x", "Urgent?");
    assert.equal(seen[0].url, "http://server/v1/systemone");
    assert.deepEqual(seen[0].body, { state: "x", questions: { q: { type: "noul", instructions: "Urgent?", criteria: {} } }, early_exit: "0.99", facts: "auto" });
    assert.equal(seen[0].init.headers.authorization, "Bearer k");
    await c.choice("x", "Which team?", ["billing", "tech"]);
    assert.deepEqual(seen[1].body.questions.q.criteria, { billing: null, tech: null });
  });

  test("images reach the body untouched; imageFromBytes makes a data URL", async () => {
    const { seen, fetch } = recorder({});
    const c = new Basal({ fetch });
    const png = imageFromBytes(new Uint8Array([137, 80, 78, 71]), "image/png");
    assert.equal(png, "data:image/png;base64,iVBORw==");
    await c.decide("x", {}, { images: [png, { content_type: "image/jpeg", base64: "AAAA" }], detail: "high", lang: "en" });
    assert.deepEqual(seen[0].body.images, [png, { content_type: "image/jpeg", base64: "AAAA" }]);
    assert.equal(seen[0].body.detail, "high");
  });

  test("a noul answer with only p gets probabilities and a confidence (other servers)", async () => {
    const { fetch } = recorder({ q: { type: "noul", noul: 0.3 }, s: null });
    const a = await new Basal({ fetch }).decide("x", {});
    assert.deepEqual(a.q, { type: "noul", noul: 0.3, probabilities: { true: 0.3, false: 0.7 }, confidence: 0.7 });
    assert.equal(a.s, null);
    assert.equal(complete(null), null);
    const full = { type: "choice", choice: "a" };
    assert.equal(complete(full), full);
  });
});

describe("presets", () => {
  const fixture = JSON.parse(readFileSync(new URL("./presets_fixture.json", import.meta.url), "utf8"));

  test("the question sets and the language guess are those of basal/presets.py", () => {
    assert.deepEqual(presetNames(), Object.keys(fixture.presets).sort());
    for (const name of presetNames())
      for (const lang of ["pl", "en"]) assert.deepEqual(presetQuestions(name, lang), fixture.presets[name][lang]);
    for (const [text, lang] of fixture.langs) assert.equal(langOf(text), lang, text);
    assert.throws(() => presetQuestions("nope"), /unknown preset/);
  });

  test("guard() and route() decide as the Python functions on the shared cases", () => {
    for (const c of fixture.guard)
      assert.deepEqual(guard(c.answers, c.policy, c.max_error ?? undefined, c.battery ?? undefined), c.expect,
        JSON.stringify(c).slice(0, 80));
    assert.deepEqual(POLICIES, fixture.policies);
    for (const c of fixture.route) assert.deepEqual(route(c.answers, 0.6, c.max_error ?? undefined), c.expect);
    assert.throws(() => guard({}, "lax"), /unknown policy/);
  });

  test("guard() and route() never pass a non-finite or out-of-range probability", () => {
    const noul = (p) => ({ type: "noul", noul: p, probabilities: { true: p, false: 1 - p }, confidence: 0.9 });
    const base = () => ({ jailbreak: noul(0), harmful: noul(0), medical: noul(0), self_harm: noul(0),
      severity: { type: "score", probabilities: { 0: 0.97, 1: 0.01, 2: 0.01, 3: 0.01 }, confidence: 0.97 } });
    assert.equal(guard(base()).action, "pass");
    for (const bad of [NaN, Infinity, -Infinity, -0.1, 1.1]) {
      const a = base();
      a.harmful = noul(bad);
      assert.equal(guard(a).action, "review", String(bad));
      assert.equal(guard(a).checks.harmful.why, "malformed answer");
      const s = base();
      s.severity.probabilities = { 0: bad, 1: 0.1 };
      assert.equal(guard(s).action, "review", String(bad));
      const comp = { probabilities: { 0: 0.2, 1: 0.8 } };
      assert.equal(route({ intent: { choice: "billing", confidence: bad }, complexity: comp }).route, "human");
    }
    assert.equal(guard({}).action, "review");
    assert.equal(guard({ harmful: noul(0) }, "strict", undefined, ["harmful"]).action, "pass");
  });

  test("guard() asks the battery with max_error and returns the decision with the answers", async () => {
    const seen = [];
    const fetch = async (_u, init) => {
      const body = JSON.parse(init.body);
      seen.push(body);
      const answers = {};
      for (const [k, v] of Object.entries(body.questions))
        answers[k] = v.type === "noul"
          ? { type: "noul", noul: 0.01, probabilities: { true: 0.01, false: 0.99 }, confidence: 0.99, abstain: false }
          : { type: "score", score: 0, probabilities: { 0: 0.97, 1: 0.01, 2: 0.01, 3: 0.01 }, confidence: 0.97, abstain: false };
      return new Response(JSON.stringify({ model: "m", answers }), { status: 200 });
    };
    const r = await new Basal({ fetch }).guard("Hello", { kind: "output", lang: "en", maxError: 0.05 });
    assert.equal(r.action, "pass");
    assert.equal(seen[0].questions.policy_break.max_error, 0.05);
    assert.deepEqual(Object.keys(r.answers), ["policy_break", "harmful", "medical", "self_harm", "severity"]);
  });

  test("preset() asks the questions in the language of the state", async () => {
    const seen = [];
    const fetch = async (_u, init) => {
      seen.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ model: "m", answers: {} }), { status: 200 });
    };
    const c = new Basal({ fetch });
    await c.preset("Klient nie zapłacił faktury w terminie.", "triage");
    await c.preset("The customer did not pay.", "triage");
    assert.equal(seen[0].questions.urgency.instructions, "Jak pilna jest sprawa?");
    assert.equal(seen[1].questions.urgency.instructions, "How urgent is this case?");
  });
});
