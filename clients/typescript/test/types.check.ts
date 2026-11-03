// Compile-time checks (tsc --noEmit -p test/tsconfig.json): answers are typed by the question's type.
import { Basal, type ActAnswer, type ChoiceAnswer, type MultiAnswer, type NoulAnswer } from "../src/index.js";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
const ok = <T extends true>(_?: T) => undefined;

async function check(b: Basal) {
  const r = await b.decide("state", {
    c: { type: "choice", instructions: "?", criteria: ["a", "b"], coverage: 0.9 },
    y: { type: "noul", instructions: "?" },
    m: { type: "multi", instructions: "?", criteria: { a: "A" }, objective: "f1" },
    a: { type: "act", instructions: "?", criteria: { true: "yes" }, costs: { wrong: 99, defer: 1 }, max_error: 0.05 },
  });
  ok<Equal<typeof r.c, ChoiceAnswer | null>>();
  ok<Equal<typeof r.y, NoulAnswer | null>>();
  ok<Equal<typeof r.m, MultiAnswer | null>>();
  ok<Equal<typeof r.a, ActAnswer | null>>();
  const ocr = await b.decide("state", {
    x: { type: "choice", instructions: "Amount?", __other__: "No amount", generated_criteria:
      { preprocessor: "ocr", source: "invoice.png", filter: "\\d+" } },
    y: { type: "multi", instructions: "Which dates?", generated_criteria:
      { preprocessor: "ocr", source: "invoice.png", filter: "\\d+" } },
  }, { media: [{ type: "image", name: "invoice.png", ocr: "on", context: "none", data: "base64" }] });
  ok<Equal<typeof ocr.x, ChoiceAnswer | null>>();
  ok<Equal<typeof ocr.y, MultiAnswer | null>>();
  const selectedOther: boolean | undefined = ocr.x?.__other__;
  const otherProbability: number | undefined = ocr.x?.probabilities.__other__;
  void [selectedOther, otherProbability];
  const choice: string | null | undefined = r.c?.choice;
  const set: string[] | undefined = r.c?.answer_set;
  const p: number | undefined = r.y?.probabilities.true;
  void [choice, set, p];

  // @ts-expect-error a choice question needs criteria
  await b.decide("s", { q: { type: "choice", instructions: "?" } });
  // @ts-expect-error an unknown question type is not accepted by name (task types have their own list)
  await b.decide("s", { q: { instructions: "?" } });
}
void check;
