# Question types

Every basal request has one **state** (the data: text, JSON, and with basal-omni also images and audio) and any number
of named **questions** about it. Each question has a type; the model answers all of them in one pass over the shared
state ([SOAM](SOAM.md)) and returns a typed answer with **calibrated probabilities** — no text is generated.

```json
POST /v1/basal
{"state": "Order 4471, 2 × kettle, delivered damaged on 3 Oct. Customer asks for a refund.",
 "questions": {
   "intent":  {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "money back", "replace": "a new item", "info": "information only"}},
   "damaged": {"type": "noul", "instructions": "Was the item damaged?"},
   "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}}}
```

| type | asks | answer | status |
|---|---|---|---|
| [`choice`](#choice) | pick one of 2–255 options | `choice`, a probability per option | stable |
| [`noul`](#noul) | is this statement true? | `noul` = P(true) | stable |
| [`score`](#score) | rate on ordered levels | `score` (expected level), a probability per level | stable |
| [`multi`](#multi) | which labels apply (any number) | `selected`, an independent probability per label | stable |
| [`act`](#act) | which action has the lowest expected cost | `action`, `expected_costs`, defer to a person | experimental |
| [`tool`](#tool) | the next tool of an agent | `tool`, `call`, a probability per tool | experimental |
| [`classify`](#classify) | labels with a rubric each | `label` / `labels`, `scores` | experimental (2.1) |
| [`verify`](#verify) | are these claims supported by the state? | per claim: supported / contradicted / not stated | experimental (2.1) |
| [`compress`](#compress) | which parts of the state are needed for a question? | `compressed` text, a probability per unit | experimental (2.1) |
| [`extract`](#extract) | fill a JSON schema from the state | `data`, per field: value, offsets, confidence | experimental (2.1) |
| [`rank`](#rank) | order thousands of candidates by a yes / no question | `ranking` (best first), `top` | experimental (2.5) |

*Stable* types keep their request and answer shape within a major version. *Experimental* types may change in a minor
release and their measured quality is below what we call production-ready: check them on your own data.

## Common to every type

- **Structured input:** the state, `instructions`, option descriptions, labels and claims may be JSON objects or arrays;
  they are written into the prompt as compact JSON.
- **Language:** Polish and English prompts, chosen from the text (or `"lang": "pl" | "en"`).
- **Calibration:** probabilities are calibrated per question type and number of options (`CALIBRATION.json`). With
  `max_error`, an answer whose confidence is below the threshold certified for that error rate is deferred to a person
  ([API.md](API.md#using-the-confidence)).
- **Options:** `option_keys: "hide"` shows the model only the descriptions (`"show"`, the default, shows `key: description`).
- **Media** (basal-omni with `--media on`): images and audio in `"media"`; every decision type reads them.
- **Optional extras:** `"facts": "auto"` (computed calendar and money facts for Polish states), `"evidence": true`
  (supporting spans, models with an evidence head) — see [FEATURES.md](FEATURES.md).

## choice
```json
{"type": "choice", "instructions": "What does the customer want?",
 "criteria": {"refund": "money back", "replace": "a new item", "info": "information only"}}
→ {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.94, "replace": 0.04, "info": 0.02}, "confidence": 0.94}
```
Exactly one option; probabilities sum to 1. Up to 255 options (prompt format 2). `criteria` may be a list of keys.

## noul
```json
{"type": "noul", "instructions": "Was the item damaged?"}
→ {"type": "noul", "noul": 0.97, "probabilities": {"true": 0.97, "false": 0.03}, "confidence": 0.97}
```
Optional `criteria: {"true": "...", "false": "..."}` words the two answers.

## score
```json
{"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}
→ {"type": "score", "score": 1.62, "legend": {"0": "low", "1": "medium", "2": "high"},
   "probabilities": {"0": 0.05, "1": 0.28, "2": 0.67}, "confidence": 0.67}
```
Levels lowest first; `score` is the expected level.

## multi
```json
{"type": "multi", "instructions": "Which issues does the message raise?",
 "criteria": {"damage": "damaged goods", "delay": "late delivery", "billing": "wrong charge"}, "threshold": 0.5}
→ {"type": "multi", "selected": ["damage"], "probabilities": {"damage": 0.96, "delay": 0.08, "billing": 0.03},
   "threshold": 0.5, "set_confidence": 0.86}
```
One yes/no per label; select by `threshold` (with `min` / `max`), or by the set with the best expected `"objective":
"iou" | "f1" | "exact"`; `exclusive` labels and a `fallback` label are optional. Details: [FEATURES.md](FEATURES.md).

## act
```json
{"type": "act", "instructions": "Is the refund due?", "criteria": {"true": "due", "false": "not due"},
 "costs": {"approve": {"true": 0, "false": 1240}, "reject": {"true": 300, "false": 0}, "human": {"true": 15, "false": 15}},
 "max_error": 0.05}
→ {"type": "act", "action": "human", "expected_costs": {"approve": 47.1, "reject": 288.6, "human": 15.0},
   "answer": "true", "probabilities": {...}, "confidence": 0.96, "deferred": true}
```
The action with the lowest expected cost under the calibrated probabilities; with `max_error`, automatic actions
below the certified threshold go to the defer action. Details: [FEATURES.md](FEATURES.md).

## tool
```json
{"type": "tool", "instructions": "What should the agent do next?", "tools": [<OpenAI, MCP or Anthropic tool definitions>],
 "extra": ["answer_directly", "ask_user", "none"]}
→ {"type": "tool", "tool": "refund_order", "call": true, "probabilities": {...}, "confidence": 0.91}
```
Each tool becomes an option (name, parameters, description); `multi: true` returns the set of tools; `rank: k` the top
k. Details: [FEATURES.md](FEATURES.md#tool-which-tool-next).

## classify
```json
{"type": "classify", "labels": [{"name": "refund", "rubric": "asks for money back"},
                                {"name": "track", "rubric": "asks where the parcel is"}], "none": true}
→ {"type": "classify", "label": "refund", "top_label": "refund", "scores": {"refund": 0.93, "track": 0.02, "none": 0.05},
   "confidence": 0.93, "automated": {"0.01": null, "0.05": null}}
```
`choice` (one label) or, with `multi_label: true`, `multi` (any number), with a classification wording, an optional "none
of these" label and `top_label` / `scores` fields. Details: [TASKS.md](TASKS.md#classify).

## verify
```json
{"type": "verify", "claims": {"total": "The total is 1,230 PLN.", "terms": "Payment is due in 30 days."}}
→ {"type": "verify", "all_supported": false, "not_supported": ["terms"],
   "claims": {"total": {"verdict": "supported", "probabilities": {...}, "confidence": 0.98},
              "terms": {"verdict": "contradicted", "probabilities": {...}, "confidence": 0.95}}}
```
Each claim is checked against the state: supported, contradicted or not stated (`binary: true`: supported or not).
Use it to check extracted fields, an LLM's answer or summary, or a document against stated facts.
Details: [TASKS.md](TASKS.md#verify).

## compress
```json
{"type": "compress", "instructions": "What are the payment terms?", "unit": "sentence", "keep_ratio": 0.3}
→ {"type": "compress", "compressed": "...", "kept": [3, 4, 17], "units": [{"start": 0, "end": 84, "probability": 0.03}, ...],
   "ratio": 0.29}
```
One yes/no per sentence, line or paragraph; `compressed` keeps the needed parts in document order, as a shorter input
for another model. Details: [TASKS.md](TASKS.md#compress).

## extract
```json
{"type": "extract", "schema": {"type": "object", "properties": {
   "invoice_number": {"type": "string"}, "gross_total": {"type": "number"},
   "payment_method": {"enum": ["transfer", "card", "cash"]}, "paid": {"type": "boolean"},
   "seller": {"type": "object", "properties": {"name": {"type": "string"}, "tax_id": {"type": "string"}}}}}}
→ {"type": "extract", "data": {"invoice_number": "FV/309/02/2026", "gross_total": 1230.0, "payment_method": "transfer",
                               "paid": null, "seller": {"name": "Med Energia sp. k.", "tax_id": "5251234567"}},
   "fields": {"invoice_number": {"value": "FV/309/02/2026", "start": 15, "end": 29, "confidence": 0.96, ...}, ...}}
```
JSON schema: nested objects, `string`, `number`, `integer`, `boolean`, `enum` / `oneOf` with `const`, arrays of enum
values. Text fields are two decisions (the line, then the exact span), so values are always exact text of the document
with character offsets; any field can be `null` (not stated). Not yet: arrays of strings or objects.
Details: [TASKS.md](TASKS.md#extract).

## rank
```json
{"type": "rank", "instructions": "Does {candidate} answer the question?", "label": "Passage", "top": 20,
 "candidates": [{"key": "visa#3", "text": "Visas take two weeks.", "group": "Visa guide"}, ...]}
→ {"type": "rank", "ranking": [{"key": "visa#3", "score": 0.97, "choice": 0.81}, ...], "top": "visa#3",
   "candidates": 1000, "requests": 21}
```
One yes / no per candidate in shared-context requests of 50, then one choice over the top 20; beyond the 255 options of
a choice question, up to 10,000 candidates per request. `basal-navigate` uses it to search a document tree. Details:
[NAVIGATE.md](NAVIGATE.md).

## What basal does not do

basal does not generate text. For a summary, put `compress` in front of a generative model and `verify` behind it
([TASKS.md](TASKS.md#summaries-with-another-model)).
