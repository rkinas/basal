---
name: basal
description: Make typed decisions (route, triage, classify, moderate, guard, pick a tool) with a local basal server. Use when code or an agent must choose among options, answer yes/no, or rate something, and needs calibrated probabilities and an abstain-to-human rule instead of free text. Covers writing the state and questions, question types, presets, coverage / max_error, calibrating on your own data, and when to escalate.
license: Apache-2.0
---

# basal: typed decisions with calibrated confidence

basal does not generate text. You send one **state** (the data) and named **questions**; it answers every question in
one pass with a typed answer and calibrated probabilities. Use it where the output is a choice, not prose.

## Call it

A server is assumed running (`basal-serve --model <model> --port 8000`; see docs/SERVING.md). Then:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 4471 arrived damaged. Customer wants money back, today.",
  "questions": {
    "intent":  {"type": "choice", "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "replace": "a new item", "info": "information only"}},
    "urgent":  {"type": "noul", "instructions": "Is this urgent?"},
    "tone":    {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "angry"]}}}'
```

Python: `Basal("http://127.0.0.1:8000").decide(state, questions)` (`from basal.client import Basal`). TypeScript:
`new Basal({url}).decide(state, questions)` (`clients/typescript`).

## Write a good state

- Put **everything the decision needs and nothing else** in the state: the message, the relevant record fields, the
  rule text. JSON objects are fine. Cut boilerplate, signatures and unrelated history.
- Do not put the question in the state, and do not tell the model how to answer. One state, many questions: the
  state is read once, so ask everything you need about it in the same request.
- Facts the model cannot know (today's date, a price list, a policy) belong in the state. Polish and English are both
  supported; the language follows the text (or `"lang": "pl" | "en"`).

## Write good questions

- `instructions` is one clear sentence about the state. Ask about one thing.
- Options (`criteria`) must be **mutually exclusive and exhaustive**. Always include an `other` option when the
  options do not cover everything. Describe each option in a few words: the model sees `key: description`.
- Yes/no questions state the positive case ("Does the customer ask for a refund?"), not a double negative.
- Prefer `choice` over many yes/no questions when exactly one thing is true; prefer `multi` when several can be.

## Question types

| type | use it for | read |
|---|---|---|
| `choice` | exactly one of 2-255 options | `choice`, `probabilities` |
| `noul` | yes / no | `noul` = P(yes) |
| `score` | an ordered scale (urgency, severity, complexity) | `score` (expected level, 0 = first) |
| `multi` | any number of labels apply | `selected`, a probability per label |
| `act` | pick the action with the lowest expected cost, or defer | `action`, `deferred` |
| `tool` | the next tool of an agent from tool definitions | `tool`, `call` |
| `classify`, `verify`, `compress`, `extract` | rubric labels, claim checking, context compression, JSON fill | docs/QUESTION_TYPES.md |

`ask_if` / `depends_on` skip a question unless an earlier answer matches (the skipped answer is `null`).

## Presets

Ready question sets (Polish and English): `triage`, `moderation`, `compliance`, `guard_input`, `guard_output`,
`route_intent`. `basal-run --list-presets`, `Basal(url).preset(state, "triage")`.

- `b.guard(text, kind="input" | "output", policy="strict" | "permissive")` returns `action`: `pass | review | block |
  support` (fixed precedence support > block > review > pass; a missing answer is a review).
- `b.route(text)` returns the intent, or `"human"` on low confidence or abstain.
- A preset is a starting point: copy its questions and edit them to your own rules and routes.

## Trust the confidence only as far as it is calibrated

- Every answer has `confidence` (the top probability). It is calibrated by question type and option count.
- `"max_error": 0.05` on a question adds `abstain: true` when the confidence is below the threshold **certified** for
  that error rate and type. Abstain means "send to a person", not "the answer is wrong". Without a certified threshold
  every answer abstains (the answer says so): pick a target the model's calibration file certifies (the keys of its `thresholds`, docs/CALIBRATION.md)
  or calibrate yourself.
- `"coverage": 0.9` adds `answer_set`: the options kept so that the right one is inside for about 90% of requests like
  the calibration data. A one-element set is a confident answer; a larger set is honest uncertainty.
- Neither field changes the answer. They are about what you do with it.

## Calibrate on your own data

Shipped thresholds were fitted on other data. If your texts, labels or options differ (and for any use that acts
automatically), calibrate:

1. Collect labelled requests (about 1,000 per question type, at least 200): the same state and questions you send, plus the
   correct answer. Keep them out of anything you tune on.
2. `basal-calibrate labelled.jsonl --model <model> --mode fast` writes `CALIBRATION.custom.json` with the leave-one-family-out
   ECE and the share automated at each certified threshold. Format and options: docs/CALIBRATION.md.
3. `basal-serve ... --calibration CALIBRATION.custom.json`. A target the file cannot certify is marked, and those
   questions abstain.
4. Re-check on fresh labelled data when the model, the options or the traffic changes.

## When to escalate

- **Abstain or low confidence:** a person decides (`review`, `human`). That is the design, not a failure.
- **Safety, money, legal or irreversible actions:** use `act` or `max_error` so automatic action needs a certified
  threshold, and keep a person in the loop for the rest.
- **Many answers near the threshold:** use a bigger model for those questions only: `basal-router --replicas <small>
  --escalate <big> --escalate-below 0.8` asks the bigger replica again where the small one is unsure (docs/SERVING.md).
- **The task needs prose** (a summary, an explanation, a draft): basal does not write it. Decide with basal, generate with
  a language model.
- Never raise `max_error` to make the abstain rate look better; the share that goes to a person is the cost of the
  guarantee. Report the error rate and the automated share together.
