# API

The HTTP API implements the *System One* JSON interface (`POST /v1/systemone`, `GET /v1/models`); responses validate
against the official OpenAPI schema (`tests/test_server.py`). Clients that use this subset work by changing the base
URL (the official SDK was not tested). `POST /v1/basal` is the same endpoint; the extensions below (`multi`, `act`,
`tool`, `evidence`, `facts`, `option_keys`, `think`, `early_exit`) are accepted on both.

## Request

`POST /v1/systemone`

```jsonc
{
  "state": "text or JSON",                         // JSON values are serialised compactly into the prompt
  "questions": {
    "<name>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<description>", ...}},   // or a list of keys
    "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},  // criteria optional
    "<name>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]},      // lowest first
    "<name>": {"type": "multi",  "instructions": "...", "criteria": {...}, "threshold": 0.5, "min": 1, "max": 3},
    "<name>": {"type": "multi",  "instructions": "...", "criteria": {...}, "objective": "iou", "max": 3,  // or "f1", "exact"
               "exclusive": ["<label>"], "fallback": "<label>", "fallback_threshold": 0.4},              // optional
    "<name>": {"type": "act",    "instructions": "...", "criteria": {...}, "costs": {...}, "max_error": 0.05},
    "<name>": {"type": "tool",   "instructions": "...", "tools": [<OpenAI or MCP tool definitions>],      // instructions optional
               "extra": ["answer_directly", "ask_user", "none"], "rank": 3, "max_error": 0.05,        // all optional
               "description_chars": 300},
    "<name>": {"type": "tool",   "tools": [...], "multi": true, "objective": "f1"},                   // the set of tools
    // optional on every question:
    //   "option_keys": "show" | "hide"   (default "show": options are shown as "key: description")
    //   "evidence": true                 (models with an evidence head, basal-1.5; not for multi; experimental)
    // optional on choice, noul and score questions:
    //   "coverage": 0.9                  (the answer set at this coverage; basal-calibrate --conformal)
    //   "max_error": 0.05                ("abstain" below the threshold certified for this target)
  },
  "facts": "off" | "auto",                          // optional, default "off" (Polish calendar / money facts)
  "think": "auto" | "off" | "force",                // optional, --think auto servers only (basal-2.0 4.5B)
  "think_text": true,                               // optional, returns the reasoning note
  "early_exit": "off" | "0.999" | "0.995" | "0.99" | "0.98",  // optional, --mode fast-exit only (basal-1.0)
  "media": [{"type": "image" | "audio", "data": "<base64>"}],  // optional, --media on servers only (basal-omni)
  "detail": "default" | "high",                     // optional, image tokens 280 / 1,120 (with media)
  "lang": "pl" | "en"                               // optional, the prompt language when the state text does not show it
}
```

2–255 options per question for prompt-format-2 models (basal-2.0), 2–10 for prompt-format-1 models (basal-1.0 to 1.5);
`ollama` mode: at most 20 (see [FORMATS.md](FORMATS.md)). Several questions per request are answered over
one shared state (SOAM); asking them in one request is faster than separate requests and gives the same answers.

## Response

```jsonc
{
  "model": "basal-2.0-4.5B",
  "answers": {
    "<choice>": {"type": "choice", "choice": "<key>", "probabilities": {"<key>": p, ...}, "confidence": p},
    "<noul>":   {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p},
    "<score>":  {"type": "score", "score": expected_level, "legend": {"0": "level 0", ...},
                 "probabilities": {"0": p, ...}, "confidence": p},
    "<multi>":  {"type": "multi", "selected": ["<key>", ...], "probabilities": {"<key>": p_applies, ...},
                 "threshold": 0.5, "set_confidence": p},          // with "objective": "objective" and "expected_score"
    "<act>":    {"type": "act", "action": "<action>", "expected_costs": {"<action>": c, ...}, "answer": "<outcome>",
                 "probabilities": {...}, "confidence": p, "calibration": "validated" | "unvalidated"},
                 // with max_error: "deferred" (true: the defer action, "refused" names the automatic one)
    "<tool>":   {"type": "tool", "tool": "<tool name or extra option>", "call": true, "probabilities": {...},
                 "confidence": p, "ranking": [{"tool": "<name>", "probability": p}, ...]},  // ranking with "rank"
                 // with max_error: "deferred" (true: "tool" is null, "refused" names the model's choice)
                 // with "multi": true: "tools": [...] instead of "tool", and the fields of a multi answer
    // with "coverage": L, a choice / noul / score answer also has "answer_set": ["<key>", ...] (most probable
    //   first) and "coverage": L; with "max_error": "abstain": true | false
    // with "evidence": true, any of the above (except multi) also has
    //   "evidence": [{"text": "...", "start": 102, "end": 153, "probability": 0.93}, ...]
  },
  "usage": {"input_tokens": n, "output_tokens": 0, "questions": n, "branches": n, "latency_ms": t}
  // "truncated" when the state was cut to the context; with --orders auto also "second_order_branches": n
}
```

- `probabilities` are calibrated (temperatures per question type and option-count bucket from the calibration file);
  `confidence` is the probability of the chosen answer.
- `score` is the expected level (0 = first level); `legend` maps level keys to their texts.
- `branches` counts the question branches of the request: one per question, one per label of a `multi` and one per
  tool of a `tool` question with `"multi": true` (times the option orders).
- Errors return `{"error": "..."}` with a status code (2.3.1; before, every error was 422):
  - **422**: the request is invalid and must be changed (a body that is not a JSON object, a missing `state` or
    `questions`, a wrong field value, too many options, a prompt longer than the context, a field not available in
    the server's mode or model);
  - **503**: the GPU ran out of memory (CUDA, MPS or Metal). The server frees the failed request's memory, so a retry,
    later or on another server, may succeed;
  - **500**: any other server error.
  Every error body also has a `code`: `INVALID_REQUEST` (422), `OUT_OF_MEMORY` (503) or `INTERNAL_ERROR` (500).
  A 422 adds `detail`, a one-item body-location validation error; `error` retains the message.

  `basal-serve --error-format openai` answers `{"error": {"message": "...", "type": "validation_error" |
  "server_error", "code": 422}}` instead, with the same status codes, for clients of OpenAI-style servers.
- `GET /v1/models` returns `{"models": [{"name", "description", "release_date", "mode", "early_exit",
  "temperature_per_prim", "certified", "context"}]}`. `certified` maps each `max_error` target to certified confidence
  thresholds by question type (omitted when the calibration does not match); `context` reports the limit, overflow policy
  and truncation head share. `release_date` is the engine's release date, not the model's.
- `GET /health` returns `{"status": "ok", "running": R, "queued": Q}` (2.3.1). The counts cover the decision requests
  in flight (`/v1/systemone`, `/v1/basal` and MCP tool calls), from the moment the server has read a request until its
  answer or error. `queued` counts those waiting for the engine: all their queue entries are still in the server's
  queue (the main queue, the `--scheduler` backlog or the `--long-lane` queue). `running` counts the others: in an
  engine call, or in the server's own work before or after one (a request with a second pass, such as `--orders auto`,
  can be queued again). `running` and `queued` both 0 means that the server is idle.

## Requests from other `/v1/systemone` servers' clients

The request shape of vLLM's structured-diffusion example (`examples/features/structured_diffusion`) is accepted as
well:

- `"ask_if": {"<question>": ["<answer>", ...]}` on a question: it is answered only when every named question's answer
  is among the values, otherwise its answer is `null` and `usage.skipped` names it. The named question must be a
  `choice` (an option key), `noul` (`yes` / `no`, `true` / `false`) or `score` question (a level key or its text);
  a question named by a skipped one is skipped too.
- `"depends_on": ["<question>", ...]` is checked (known names, no cycle with `ask_if`). basal-serve answers all
  questions over the state in one pass, so an earlier answer is not added to a later question's prompt.
- `"images": ["data:image/png;base64,...", {"content_type": "image/jpeg", "base64": "..."}]` is the same as
  `"media"` with image items (servers with `--media on`); a request gives one of the two.
- `"model"`, `"seed"`, `"alone"` and the example's read options (`samples`, `steps`, ...) are ignored.

A request without these fields is answered as before. The example's own answers differ from basal-serve's in one way
a client sees: its `noul` answer has only `noul`; `basal.client` fills in `probabilities` and `confidence`.

## OpenAI's Decisions API (`POST /v1/decisions`)

`POST /v1/decisions` takes the request shape of OpenAI's Decisions API, so `client.decisions.create` of the OpenAI
Python SDK (checked with 3.40) works against basal-serve with `base_url="http://127.0.0.1:8000/v1"` (any `api_key`).
Each request is decided as the System One request it stands for, with the same probabilities as `/v1/systemone`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
decision = client.decisions.create(model="basal", input="I was charged twice.", questions=[
    {"type": "predicate", "name": "angry", "instructions": "Is the customer angry?"},
    {"type": "choice", "name": "team", "instructions": "Which team?",
     "choices": [{"value": "billing", "description": "payments and invoices"}, {"value": "shipping"}]},
    {"type": "score", "name": "urgency", "instructions": "How urgent?",
     "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high", "description": "today"}]}])
print(decision.answers[0].probability, decision.answers[1].choice)
```

| Decisions API | System One question | answer |
|---|---|---|
| `input` | `state`: the text, or the list of messages / items as JSON (images are refused) | |
| `predicate` | `noul` | `probability`: P(true) |
| `choice` with `choices` `[{"value", "description"}]` | `choice` with `{value: description}` (a boolean value is shown as `true` / `false`) | `choice` and `probabilities` with the typed values, in the order of the choices; `confidence` |
| `score` with `levels` `[{"label", "description"}]` | `score` with the levels `label` or `label: description` | `score` (expected level index), `probabilities` per level (`label`, `value` = index); `confidence` |

Answers come in question order with the question's `name` (null without one); `usage` counts the prompt tokens.
`model` and the prompt-cache fields are accepted and ignored; the served model answers. The engine-level types
(`multi`, `act`, `tool`), `coverage`, `max_error` and `facts` stay on `/v1/systemone`. Errors have OpenAI's shape,
`{"error": {"message", "type", "param", "code"}}`: HTTP 400 `invalid_request_error` for a request to change (the SDK
raises `BadRequestError`), 5xx `server_error` otherwise.

## Question types

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |
| `rank` | the candidates in order, any number of them (experimental) | the passages for a query, the best of 5,000 products |

**`multi`**: each label is asked as a yes/no branch over the shared state ("does label X apply?"). `selected` holds
the labels with probability ≥ `threshold` (default 0.5), at most `max` and at least `min`; `set_confidence` estimates
that the whole set is exactly right. With `"objective": "iou"`, `"f1"` or `"exact"` the engine instead returns the set
with the highest expected IoU, F1 or exact-match probability under the label probabilities (`min` / `max` bound its
size) and its `expected_score`. An `exclusive` label is selected only alone; a `fallback` label ("none of these") is
selected only when no other label is. See [FEATURES.md](FEATURES.md#choosing-the-set-objective).

**`act`**: `criteria` are the possible outcomes (`{"true": ..., "false": ...}` for a yes/no question, or named options),
`costs` gives the cost of every action for every outcome (`{"approve": {"true": 0, "false": 1240}, "reject":
{"true": 300, "false": 0}, "human": {"true": 15, "false": 15}}`), or the short form `{"wrong": 99, "defer": 1}`
(answer with any outcome, wrong costs 99; deferring costs 1). The engine returns the action with the minimum expected
cost under the calibrated probabilities. With `"max_error"` it also refuses automatic actions below the confidence
threshold certified for that error and question type (the response then names the `refused` action and says
`"deferred": true`) and needs a `defer` / `human` action (or `"defer_action": "<name>"`). Without a threshold certified
for the type and target, every automatic action is refused. See [FEATURES.md](FEATURES.md#act-cost-aware-actions).

**`tool`** (*experimental*): `tools` takes OpenAI (Chat Completions or Responses), MCP (`inputSchema`) or Anthropic
(`input_schema`) tool definitions. Each tool becomes an option `name(a, b, c?) — description`, where `?` marks an
optional parameter; descriptions are cut at 300 characters (`"description_chars"`, 0 = no cap). The extra options
(`answer_directly`, `ask_user`, `none`; by default all three) follow the tools. `"rank": k` returns the k most probable
options. `"multi": true` returns the set of tools to call, using the `multi` objectives and constraints. `max_error`
defers to a person, as in `act`. See [FEATURES.md](FEATURES.md#tool-which-tool-next), which also shows a
[step of an agent loop](FEATURES.md#a-step-of-an-agent-loop) in one request.

**Answer sets and abstain** (`choice`, `noul`, `score`). `"coverage": 0.9` adds `answer_set`: the keys whose
calibrated probability reaches the type's threshold for that coverage level, the most probable first (and always in).
The set holds the correct answer for at least 90% of requests like the labelled ones the threshold was fitted on
(split conformal prediction). The levels come from a calibration fitted with `basal-calibrate --conformal 0.8,0.9,0.95`.
For a level or type the file does not have, the set is every option and `coverage_note` says why. `"max_error": 0.05`
adds `"abstain": true` when the confidence is below the threshold certified for that error and the question's type,
as `act` defers. Without a certified threshold every answer abstains, and `max_error` in the answer says so. Neither
field changes the answer itself. See [CALIBRATION.md](CALIBRATION.md#answer-sets-coverage).

**Option keys.** By default every described option is shown to the model as `key: description`
(`{"B": "Dispatch office: Krakow"}` → `B: Dispatch office: Krakow`), because only you know whether a key is meaningful
(a service code, a decision name); the server never guesses. If your keys are placeholders (`option_1`, `0`, …) and
the descriptions alone define the options, send `"option_keys": "hide"`: the model then sees the descriptions only.
A key without a description (`"criteria": {"approve": null}` or a list of keys) is shown by itself.

**Evidence** (*experimental*): `"evidence": true` on a question returns up to three spans of the state that support the
answer, with character offsets and a probability. It needs a model that ships `evidence_head.pt` (basal-1.5); the
basal-2.0 models do not ship one, and the server refuses evidence requests for them. See
[FEATURES.md](FEATURES.md#evidence-spans).

**Task types** (*experimental*, 2.1): `classify`, `verify`, `compress` and `extract` (a JSON schema) are accepted as
question types next to the others in the same request. They are expanded into `choice` / `noul` / `multi` decisions over
the shared state and answered in the shapes described in [TASKS.md](TASKS.md).

**Rank** (*experimental*, 2.5.1): `{"type": "rank", "instructions": "Does {candidate} answer the question?",
"candidates": {...} | [...]}` ranks any number of candidates, more than the 255 options of a choice question: yes / no
screening in shared-context requests of 50 candidates, then one choice question over the top 20. The answer is
`{"type": "rank", "ranking": [{"key", "score", "choice"?}, ...], "top": "<key>", "candidates": n, "requests": n}`. See
[NAVIGATE.md](NAVIGATE.md#rank-any-number-of-candidates), which also covers `basal-navigate`, retrieval by navigating a
document tree with these questions.

**Facts**: `"facts": "auto"` appends a block of computed facts to a Polish state: the weekday of every date and
whether it is a Saturday or a statutory day off, gaps between dates, date + stated durations, net/gross amounts for
a stated VAT rate, euro × a stated exchange rate, budget remainders, quantity × price and list totals, and
comparisons of amounts (as symbols only, never a verdict). Only arithmetic and calendar facts are added, never a rule
or an answer.

## Using the confidence

The server applies the per-type temperatures of the calibration file (for basal-2.0 also per option-count bucket:
2–10, 11–26, 27–255 options). The file also contains confidence thresholds chosen on calibration data, before testing,
for a target error of 1% and 5% among accepted decisions (`thresholds` in `CALIBRATION.json`): accept decisions above
the threshold automatically and route the rest to a person, either in your code or with an `act` / `tool` question and
`"max_error": 0.05`, which defers below the certified threshold of the question's type.

The basal-2.0 calibration was fitted on question families the models were not trained on, so it holds on new tasks
but is cautious. Held out by question family (leave one family out):

| | mini | 4.5B | max |
|---|---|---|---|
| calibration error (ECE) | 0.017 | 0.015 | 0.022 |
| 5% target threshold: share automated (observed error) | 10.5% (2.94%) | 33.2% (2.39%) | 41.9% (2.51%) |
| 1% target threshold in the shipped file | not certified | not certified | not certified |

No 1% threshold is certified in the shipped files, except in `CALIBRATION.ollama.json` of `basal-2.0-max-GGUF`,
which certifies one for `choice` (confidence 0.989), so `"max_error": 0.01` defers every other decision: use 0.05, or
certify your own thresholds on your data. The ports use per-engine files where the card lists one
(`CALIBRATION.<engine>.json`), else the bf16 file, which is not validated for them.

**Calibrate on your own data.** On your own domain, refit the temperatures and thresholds on your labelled requests
with `basal-calibrate`. It runs the engine you serve, refits by the same procedure, and prints coverage and error at
the shipped and the refitted thresholds on a held-out part of your items. A few hundred labelled items usually certify
the 5% target; the 1% target needs at least 299 accepted decisions without an error in the fitting part, per
question type, so in practice about 1,000 items of each type or more. A target that cannot be certified is marked in the file, and `max_error` then
defers every `act` / `tool` decision of that type. Input format and method: [CALIBRATION.md](CALIBRATION.md).

```bash
# labelled.jsonl: one request per line, each question with its correct answer in "gold"
basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B --mode fast     # writes CALIBRATION.custom.json
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --calibration CALIBRATION.custom.json
```

## How to ask

- **One condition per yes/no question.** "Does the message mention health, finances or home address?" is harder than
  three questions or one `multi` with three labels.
- **Prefer `choice` with described sides** for a binary decision whose two outcomes have names ("approve: the invoice
  matches the order" / "hold: amounts or items differ").
- **Ask in the language of the state.** The prompt language is chosen from the text: Polish when the state or the
  question contains Polish diacritics, English otherwise.
- **Put the facts and rules in the state.** The model decides from what it reads; a small model's world knowledge is
  limited, and legal rules change.
- **Let code compute, let basal decide.** Exact arithmetic and thresholds are a weak spot of one-pass models: compute
  numbers in code (or send `"facts": "auto"` for Polish dates and amounts) and ask the typed question on top.
- **Offer "cannot be determined"** (or "none of the options") when the state may not contain the answer.
- **Use `criteria` descriptions** that say what each option means, not only a label; keep keys stable and send
  `"option_keys": "hide"` when keys are placeholders.

## MCP

Agents can call the same decisions over the Model Context Protocol: [MCP.md](MCP.md).
