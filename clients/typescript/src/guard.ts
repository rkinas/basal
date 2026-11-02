// Decision steps of the guard_input / guard_output / route_intent presets; the same logic as guard() and route() in
// basal/presets.py (test/presets_fixture.json holds shared cases).
import { POLICIES, PRECEDENCE } from "./presets.js";
import type { GuardAction, GuardCheck, GuardDecision, GuardPolicy, RouteDecision } from "./types.js";

type Loose = Record<string, any> | null | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any

function level(ans: Record<string, any>): number { // eslint-disable-line @typescript-eslint/no-explicit-any
  const probs = ans.probabilities as Record<string, number>;
  return Number(Object.keys(probs).reduce((a, b) => ((probs[b] ?? 0) > (probs[a] ?? 0) ? b : a)));
}

/** Answers of a guard battery -> pass / review / block / support (precedence support > block > review > pass).
 *  A missing answer is a review. With `maxError` (ask the questions with the same max_error) a hazard is hit at
 *  p >= 0.5 and an answer marked `abstain` is a review: the thresholds are those of the model's calibration. */
export function guard(
  answers: Record<string, Loose>,
  policy: string | GuardPolicy = "strict",
  maxError?: number,
): GuardDecision {
  const pol = typeof policy === "string" ? POLICIES[policy] : policy;
  if (!pol) throw new Error(`unknown policy ${JSON.stringify(policy)} (policies: ${Object.keys(POLICIES).sort().join(", ")})`);
  const certified = maxError !== undefined && maxError !== null;
  const checks: Record<string, GuardCheck> = {};
  let worst: GuardAction = "pass";
  const put = (name: string, action: GuardAction, p: number | null, why?: string) => {
    checks[name] = why ? { action, p, why } : { action, p };
    if (PRECEDENCE.indexOf(action) < PRECEDENCE.indexOf(worst)) worst = action;
  };
  for (const [name, ans] of Object.entries(answers)) {
    if (!ans || typeof ans !== "object") { put(name, "review", null, "no answer"); continue; }
    if (certified && ans.abstain === undefined) { put(name, "review", null, "no abstain flag: ask with max_error"); continue; }
    if (name === "severity") {
      const lvl = level(ans);
      let act: GuardAction = lvl >= pol.severity.block ? "block" : lvl >= pol.severity.review ? "review" : "pass";
      if (certified && ans.abstain && act === "pass") act = "review";
      put(name, act, lvl);
      continue;
    }
    const hazard = pol.hazards[name];
    const p = Number(ans.noul);
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
 *  `maxError`) or a missing answer. */
export function route(answers: Record<string, Loose>, minConfidence = 0.6, maxError?: number): RouteDecision {
  const intent = answers.intent, comp = answers.complexity;
  if (!intent || !comp) return { route: "human", why: "no answer" };
  const lvl = level(comp);
  const out: RouteDecision = {
    route: intent.choice, complexity: (comp.legend ?? {})[String(lvl)], level: lvl, confidence: intent.confidence,
  };
  if (maxError !== undefined && maxError !== null) {
    if (intent.abstain === undefined || comp.abstain === undefined) return { route: "human", why: "no abstain flag: ask with max_error" };
    if (intent.abstain || comp.abstain) return { ...out, route: "human", why: "abstain" };
  } else if (intent.confidence < minConfidence) return { ...out, route: "human", why: "low confidence" };
  return out;
}
