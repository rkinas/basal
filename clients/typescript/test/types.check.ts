// Compile-time checks (tsc --noEmit -p test/tsconfig.json): answers are typed by the question's type.
import {
  Basal,
  type ActAnswer,
  type Answer,
  type ChoiceAnswer,
  type DecisionResponse,
  type FallbackChoiceAnswer,
  type MultiAnswer,
  type NoulAnswer,
  type RankAnswer,
} from "../src/index.js";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
const ok = <T extends true>(_?: T) => undefined;

async function check(b: Basal) {
  const r = await b.decide("state", {
    c: { type: "choice", instructions: "?", criteria: ["a", "b"], coverage: 0.9 },
    y: { type: "noul", instructions: "?" },
    m: { type: "multi", instructions: "?", criteria: { a: "A" }, objective: "f1" },
    a: { type: "act", instructions: "?", criteria: { true: "yes" }, costs: { wrong: 99, defer: 1 }, max_error: 0.05 },
  });
  ok<Equal<typeof r.c, ChoiceAnswer | FallbackChoiceAnswer | null>>();
  ok<Equal<typeof r.y, NoulAnswer | null>>();
  ok<Equal<typeof r.m, MultiAnswer | null>>();
  ok<Equal<typeof r.a, ActAnswer | null>>();
  const choice: string | undefined = r.c?.choice;
  const set: string[] | undefined = r.c?.answer_set;
  const p: number | undefined = r.y?.probabilities.true;
  void [choice, set, p];
  if (r.c && r.c.answered_by !== "fallback") {  // basal's own answer: calibrated probabilities
    const conf: number = r.c.confidence;
    const probs: Record<string, number> = r.c.probabilities;
    void [conf, probs];
  }
  // @ts-expect-error a fallback answer may have no probabilities (the LLM gave no log-probabilities)
  const fp: number = (r.c as FallbackChoiceAnswer).confidence;
  void fp;

  // rank questions (basal/rank.py) and their answers
  const ranked = await b.decide("query", {
    hits: { type: "rank", instructions: "Does {candidate} answer the question?", candidates: ["a", { key: "k", text: "b", group: "g" }],
      top: 5, choice_context: "auto", prior: { k: 2 } },
  });
  ok<Equal<typeof ranked.hits, RankAnswer | null>>();
  const best: string | undefined = ranked.hits?.ranking[0]?.key;
  void best;
  // @ts-expect-error a rank question takes no ask_if
  await b.decide("s", { r: { type: "rank", instructions: "{candidate}?", candidates: ["a"], ask_if: { x: ["y"] } } });

  // the README: the whole response, usage always there
  const full = await b.request("ticket", { q: { type: "noul", instructions: "Is it a receipt?" } }, { detail: "high" });
  const skipped: string[] | undefined = full.usage.skipped;
  const cut: number | undefined = full.usage.truncated?.dropped_tokens;
  void [skipped, cut];

  // @ts-expect-error a choice question needs criteria
  await b.decide("s", { q: { type: "choice", instructions: "?" } });
  // @ts-expect-error an unknown question type is not accepted by name (task types have their own list)
  await b.decide("s", { q: { instructions: "?" } });
}
void check;

// representative responses: skipped answers, media (calibration "text"), the router's fallback, a rank answer
const responses: DecisionResponse[] = [
  { model: "m", answers: { q: null }, usage: { input_tokens: 1, output_tokens: 0, questions: 1, branches: 0, latency_ms: 1, skipped: ["q"] } },
  { model: "m", answers: { q: { type: "noul", noul: 0.9, probabilities: { true: 0.9, false: 0.1 }, confidence: 0.9, calibration: "text" } },
    usage: { input_tokens: 9, output_tokens: 0, questions: 1, branches: 1, latency_ms: 2,
      media: { images: 1, audio: 0, media_tokens: 280, detail: "default" } } },
  { model: "m", fallback: { below: 0.5, asked: ["c"], answered: ["c"] },
    answers: { c: { type: "choice", choice: "b", probabilities: null, confidence: null, answered_by: "fallback", model: "llm",
      calibrated: false, basal: { choice: "a", confidence: 0.4, model: "m" } } },
    usage: { input_tokens: 9, output_tokens: 0, questions: 1, branches: 2, latency_ms: 3 } },
  { model: "m", answers: { r: { type: "rank", ranking: [{ key: "7", score: 0.9, choice: 0.8 }], top: "7", candidates: 40, requests: 2,
    experimental: true } }, usage: { input_tokens: 9, output_tokens: 0, questions: 1, branches: 41, latency_ms: 9 } },
];
const answer: Answer | null | undefined = responses[0]?.answers["q"];
void answer;
