# API

The HTTP API implements the *System One* JSON interface (`POST /v1/systemone`, `GET /v1/models`); responses validate
against the official OpenAPI schema (`tests/test_server.py`). Clients that use this subset work by changing the base
URL; verified TypeSafe and OpenAI SDK versions are listed under [Client setups](#client-setups). `POST /v1/basal` is
the same endpoint; the extensions below (`multi`, `act`, `tool`, `evidence`, `facts`, `option_keys`, `think`,
`early_exit`) are accepted on both.

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
- Errors are described in [Errors and request IDs](#errors-and-request-ids).
- `GET /v1/models` returns `{"models": [{"name", "description", "release_date", "mode", "early_exit",
  "temperature_per_prim"}]}` (`release_date` is the engine's release date, not the model's), plus `"certified"`
  (`{max_error target: {type: confidence threshold}}` of the loaded calibration, the thresholds `max_error` applies to
  `act` / `tool` decisions; absent without certified thresholds) and `"context"` (`{"limit": tokens, "overflow":
  "truncate" | "refuse", "truncate_head"}`, the longest prompt and `--context-overflow` / `--truncate-head`);
  `GET /health` returns status and the running/queued counters described below.

- `GET /health` returns `{"status": "ok", "running": R, "queued": Q}` (2.3.1). The counts cover the decision requests
  in flight (`/v1/systemone`, `/v1/basal` and MCP tool calls), from the moment the server has read a request until its
  answer or error. `queued` counts those waiting for the engine: all their queue entries are still in the server's
  queue (the main queue, the `--scheduler` backlog or the `--long-lane` queue). `running` counts the others: in an
  engine call, or in the server's own work before or after one (a request with a second pass, such as `--orders auto`,
  can be queued again). `running` and `queued` both 0 means that the server is idle.

## Question types

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |

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

**Facts**: `"facts": "auto"` appends a block of computed facts to a Polish state: the weekday of every date and
whether it is a Saturday or a statutory day off, gaps between dates, date + stated durations, net/gross amounts for a
stated VAT rate, euro × a stated exchange rate, budget remainders, quantity × price and list totals, and
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

## HTTP API reference

### Endpoints

The default base URL in examples is `http://127.0.0.1:8000`. Authentication is open unless `--api-keys PATH` is set.

| Method and path | Purpose | Authentication |
|---|---|---|
| `POST /v1/systemone` | Native TypeSafe System One request | Configured key, if enabled |
| `POST /v1/basal` | System One alias, including basal extensions | Configured key, if enabled |
| `POST /v1/decisions`, `POST /decisions` | OpenAI Decisions translation | Configured key, if enabled |
| `GET /v1/models` | Served model and capabilities | Configured key, if enabled |
| `GET /health` | Liveness (`{"status":"ok"}`) | Always open |
| `GET /v1/keys` | Key names and in-memory use times | Open when auth is disabled; admin key when enabled |
| `/mcp` | Streamable HTTP MCP, when started with `--mcp on` | Configured key, if enabled |

The response `model` is the served name, not a routing instruction. basal-serve does not validate or route by the
request's `model` value (basal-ui's API server does: [below](#several-models-behind-basal-ui)). Set the served name with `--name`; basal-ui passes the variant repository id (for example
`Remek/basal-2.0-mini-MLX-8bit`). If omitted, the server reports the last component of its `--model` value.

For example, a request may name any `model`; the server returns its configured name:

```json
{"model":"anything","state":"Klient pisze: paczka nie dotarła od dwóch tygodni, chcę zwrot pieniędzy.",
 "questions":{
  "intent":{"type":"choice","instructions":"Czego chce klient?","criteria":{"refund":"zwrot pieniędzy","info":"informacja o statusie"}},
  "angry":{"type":"noul","instructions":"Czy klient jest zdenerwowany?"},
  "urgency":{"type":"score","instructions":"Jak pilne?","criteria":["może poczekać","w tym tygodniu","dziś"]}}}
```

```json
{"model":"Remek/basal-2.0-mini-MLX-8bit",
 "answers":{
  "intent":{"type":"choice","choice":"refund","probabilities":{"refund":0.9874,"info":0.0126},"confidence":0.9874},
  "angry":{"type":"noul","noul":0.4834,"probabilities":{"true":0.4834,"false":0.5166},"confidence":0.5166},
  "urgency":{"type":"score","score":1.0605,"legend":{"0":"może poczekać","1":"w tym tygodniu","2":"dziś"},
             "probabilities":{"0":0.2679,"1":0.4037,"2":0.3284},"confidence":0.4037}},
 "usage":{"input_tokens":279,"output_tokens":0,"questions":3,"branches":3,"latency_ms":59.3,"longest_prompt_tokens":96}}
```

The live response contains full-precision floats; values are rounded here to four decimal places.

### Several models behind basal-ui

basal-ui's API server ([SERVING.md](SERVING.md#api-server)) serves every model loaded in the app on one port and
routes by the request's `model`, which is therefore **required** there, on `/v1/systemone`, `/v1/basal`,
`/v1/decisions` and `/decisions`. Use a model id from `GET /v1/models` (the variant repository id, for example
`Remek/basal-2.0-mini-MLX-8bit`). The body is otherwise the one above; the answer is the model's own response.

| Case | Status | `code` |
|---|---|---|
| no `model` (or not a string) | 400 | `MODEL_REQUIRED` — the message lists the loaded models |
| `model` not loaded or not ready | 404 | `MODEL_NOT_FOUND` — the message lists the loaded models |
| the model was unloaded while the request arrived | 409 | `MODEL_NOT_READY` |
| its `basal-serve` cannot be reached | 502 | `UPSTREAM_ERROR` |

Errors use the basal envelope (`{"error", "code"}`) on the System One paths and the OpenAI envelope on the Decisions
paths, like basal-serve's own errors.

`GET /v1/models` lists the ready models in both shapes: `models` holds each model's basal-serve entry (as above) and
`data` the OpenAI list (`{"id", "object": "model", "created", "owned_by": "basal"}`), with `"object": "list"`.
`GET /health` answers `{"status": "ok", "models": [ids]}`.

`GET /v1/stats` returns basal-ui's statistics: `{"models": {id: {"requests", "questions", "errors", "input_tokens",
"avg_latency_ms", "p50_ms", "p95_ms", "avg_best_prob", "first_used", "last_used", "by_type": {type: {"questions",
"avg_best_prob"}}}}, "benchmarks": {id: [results]}}`. With **Require an API key** it needs an admin key (only
basal-ui's internal key is one; user keys get 403), otherwise it is open like `/v1/keys`.

MCP on this port: `decide` and `decide_many` take a required `model` argument and a third tool, `list_models`, returns
`{"models": [ids]}`. The stdio bridge passes the model with `basal-mcp --url http://127.0.0.1:8000 --model <id>`;
without `--model` the gateway answers 400 and the tool call fails with that message.

### Media input

Native System One requests can send `media` items:

```json
"media": [{"type":"image","data":"<base64>"},{"type":"audio","data":"<base64>"}],
"detail": "high",
"lang": "en"
```

A request can contain at most 8 media items, each at most 20 MiB after base64 decoding. A media server/model must
support each kind; `--media on` enables media in supported serving modes. `detail` is `default` or `high` for images;
`lang` can be `pl` or `en`.

For vLLM-style clients, System One also accepts `images` and `audio` arrays of base64 data URLs such as
`data:image/png;base64,iVBORw0KGgo...` and `data:audio/wav;base64,UklGR...`. They are converted to native media and
appended after any `media` entries. Remote `http://` and `https://` URLs are rejected: basal does not fetch media
URLs. The aliases share the same item and byte limits.

### Errors and request IDs

System One and `/v1/basal` use an envelope with a string `error` and machine-readable `code`. Validation failures also
include `detail`:

```json
{"error":"question 'q': 1 options (supported: 2..255)","code":"INVALID_REQUEST",
 "detail":[{"loc":["body"],"msg":"question 'q': 1 options (supported: 2..255)","type":"value_error"}]}
```

| Status | Code | Route / meaning |
|---|---|---|
| 400 | `INVALID_JSON` | System One or Decisions: malformed JSON or body is not an object |
| 400 | `INVALID_REQUEST` | Decisions: invalid input or question shape / limits |
| 400 | `unsupported_parameter` | Decisions: unsupported top-level parameter (`param` names it) |
| 401 | `UNAUTHORIZED` | Any protected route: missing or invalid API key |
| 403 | `FORBIDDEN` | `/v1/keys`: a non-admin key requested key usage |
| 503 | `OUT_OF_MEMORY` | CUDA, MPS, or Metal GPU memory exhaustion |
| 422 | `INVALID_REQUEST` | System One validation / media failure, or translated Decisions inference validation |
| 500 | `INTERNAL_ERROR` | Unexpected server failure; details are logged to stderr |

A malformed body example is `400 {"error":"malformed JSON","code":"INVALID_JSON"}`. On `/v1/decisions` and
`/decisions`, errors use the OpenAI envelope:

```json
{"error":{"message":"unsupported parameter: temperature","type":"invalid_request_error",
 "param":"temperature","code":"unsupported_parameter"}}
```

Decisions input/parameter errors return 400; translated inference validation returns 422; authentication failures
return 401; unexpected server failures are 500. Auth errors use `authentication_error`, server errors use
`server_error`, and request errors use `invalid_request_error`. The `error` message remains useful for display. Before
this API change every failure returned 422 with `{"error":"..."}`. Now malformed JSON is 400, GPU memory exhaustion is 503, unexpected internal
failures are 500, and responses include `code`; the existing `error` string is unchanged for the same failure.

Every HTTP response, including errors and `/health`, has `x-request-id` and `x-typesafe-request-id` with the same
value. A valid incoming `x-request-id` (1–128 ASCII letters, digits, `.`, `_`, `:`, or `-`) is echoed in both; an
invalid or absent value is replaced with `req_<uuid4 hex>`. CORS exposes both headers.

### Authentication and CORS

By default the API has no key requirement. To require keys, pass `--api-keys PATH` with this JSON format:

```json
{"keys":[{"name":"laptop","sha256":"<64 lowercase hexadecimal SHA-256 characters>","admin":false}]}
```

Create a random secret and its stored digest without basal-ui:

```bash
python -c 'import hashlib,secrets; s="bsl_"+secrets.token_urlsafe(32); print("secret:",s); print("sha256:",hashlib.sha256(s.encode()).hexdigest())'
```

Put the printed secret in your secret store and only the digest in the key file. Send the secret as
`Authorization: Bearer <secret>` or `X-API-Key: <secret>`. `/health` is always open. The key file is checked again
when its modification time or size changes, so removing a key revokes it without restarting basal-serve.
`GET /v1/keys` requires an `admin: true` key and reports each key's name and in-memory `last_used` timestamp (null
until used); it does not return secrets or hashes.

`--cors-origins` takes comma-separated browser origins, or `*`. CORS permits GET, POST and OPTIONS and the headers
`Accept`, `Accept-Language`, `Authorization`, `Content-Language`, `Content-Type`, `Mcp-Protocol-Version`,
`Mcp-Session-Id` and `X-API-Key`. It exposes `Mcp-Session-Id`, `x-request-id` and `x-typesafe-request-id`. Preflight
requests are handled before authentication; allowed origins receive CORS headers, and a foreign origin is rejected
with HTTP 400 and no allow-origin header.

### OpenAI Decisions compatibility

`POST /v1/decisions` and `POST /decisions` translate the OpenAI Decisions request to System One and translate the
answer back. The served model name is used in the response.

| Decisions request | System One translation | Decisions answer |
|---|---|---|
| `input` string or user messages | `state`; `input_text` parts joined with two newlines | — |
| `input_image` data URL | Image item in `media` | — |
| `predicate` | `noul` | `{type, name, probability}` |
| `choice` | `choice` with criteria keyed by `str(value)` | `{type, name, choice, probabilities, confidence}` |
| `score` | `score` with ordered levels | `{type, name, score, probabilities, confidence}` |

The request accepts `model`, `input`, `questions`, `safety_identifier`, `metadata` and `store`. The last three are
accepted but do not affect inference. Other top-level parameters and unknown question fields are rejected as
`unsupported_parameter` or invalid request. Messages must have role `user` or omit role. Content may be a string or an
array of `input_text` and `input_image` parts. Images must be base64 `data:` URLs; `detail` is ignored and remote URLs
are not fetched. Empty text with image input is accepted if the underlying model accepts it.

There must be 1–200 questions. Each requires `type` and non-empty `instructions`; `name` is optional and must be
unique when provided. Omitted names appear as JSON `null`. A `choice` needs 2–255 values, each a string or boolean;
values must be unique after converting to strings, and original values (including booleans) are restored in answers.
A `score` needs 2–10 levels. These Decisions limits may exceed basal's own per-prompt-format option limit: prompt format
1 supports 2–10 options and format 2 supports 2–255 (the `ollama` mode limit is 20). The serving format must still
accept the translated question.

Full request and verified response:

```json
{
  "model":"basal",
  "input":[{"role":"user","content":[{"type":"input_text","text":"Customer wrote: I was charged twice for order #4411 and want one charge refunded today."}]}],
  "questions":[
    {"type":"predicate","name":"is_refund_request","instructions":"Is the customer asking for a refund?"},
    {"type":"choice","name":"needs_human","instructions":"Should a human agent take over?","choices":[{"value":true,"description":"billing dispute that needs a person"},{"value":false}]},
    {"type":"score","instructions":"How frustrated is the customer?","levels":[{"label":"calm"},{"label":"mildly annoyed"},{"label":"angry","description":"uses hostile language"}]}
  ]
}
```

```json
{
  "id":"dec_e7849d58…","object":"decision","created":1792523317,"model":"Remek/basal-2.0-mini-MLX-8bit",
  "answers":[
    {"type":"predicate","name":"is_refund_request","probability":0.8828},
    {"type":"choice","name":"needs_human","choice":true,"probabilities":[{"value":true,"probability":0.9199},{"value":false,"probability":0.0801}],"confidence":0.9199},
    {"type":"score","name":null,"score":1.0627,"probabilities":[{"value":0,"label":"calm","probability":0.1620},{"value":1,"label":"mildly annoyed","probability":0.6132},{"value":2,"label":"angry","probability":0.2248}],"confidence":0.6132}
  ],
  "usage":{"input_tokens":395,"input_tokens_details":{"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":395}
}
```

### Client setups

#### TypeSafe Python SDK

Install `typesafe-sdk` and import `typesafe_sdk`. The verified run used version 0.7.5:

```bash
python -m pip install typesafe-sdk==0.7.5
export TYPESAFE_BASE_URL=http://127.0.0.1:8765
export TYPESAFE_API_KEY=<secret>
export TYPESAFE_DEFAULT_MODEL=Remek/basal-2.0-mini-MLX-8bit
export NO_PROXY=localhost,127.0.0.1
```

`TypeSafeClient().system_one(...)` returned `SystemOneResponse` with typed Choice/Noul/Score answers; the response
request id matched `x-typesafe-request-id`. `client.models.list()` returned the served model. Wrong keys raised
`TypeSafeAuthenticationError` (401); an invalid one-option choice raised `TypeSafeUnprocessableEntityError` (422)
with the server message and request id. When basal is open, the SDK still requires a non-empty `TYPESAFE_API_KEY`;
use a placeholder such as `local`.

#### OpenAI Python SDK

Verified with OpenAI Python SDK 3.35.1 (the Decisions API is available in SDK versions >= 3.26):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="<secret>")
result = client.decisions.create(
    model="Remek/basal-2.0-mini-MLX-8bit",
    input="Export fails in Safari but works in Chrome.",
    questions=[
        {"type": "score", "name": "severity", "instructions": "How severe is the issue?",
         "levels": [{"label": "minor"}, {"label": "major"}, {"label": "blocking"}]},
        {"type": "choice", "name": "team", "instructions": "Which team should investigate?",
         "choices": [{"value": "frontend"}, {"value": "backend"}]},
    ],
)
```

This returned score and choice answers; a wrong key raised `openai.AuthenticationError` (401).

#### LiteLLM

LiteLLM's self-hosted vLLM System One provider reaches basal-serve as `hosted_vllm/<served name>` with `api_base` (the
server root or `/v1`; both work). Verified with LiteLLM 1.106.1 against a basal-serve started with `--api-keys`.

Python SDK, both request formats (LiteLLM translates the OpenAI shape to System One for this provider):

```python
import litellm

r = litellm.systemone(model="hosted_vllm/basal-2.0-mini", api_base="http://127.0.0.1:8000", api_key="<secret>",
                      state="Customer: charged twice, wants a refund today.",
                      questions={"refund": {"type": "noul", "instructions": "Is this a refund request?"}})
print(r.answers["refund"].noul)

d = litellm.decisions(model="hosted_vllm/basal-2.0-mini", api_base="http://127.0.0.1:8000", api_key="<secret>",
                      input="Customer: charged twice, wants a refund today.",
                      questions=[{"type": "predicate", "name": "refund", "instructions": "Is this a refund request?"}])
print(d.answers[0].probability)
```

A wrong basal key surfaces as `litellm.AuthenticationError` carrying basal's `{"error": ..., "code": "UNAUTHORIZED"}`.

Proxy: the entry below (the same one the basal-ui Connect panel shows) served both `/v1/systemone` and
`/v1/decisions` through `litellm --config config.yaml`. The proxy refuses to start without its own master key
(`LITELLM_MASTER_KEY`); clients then authenticate to the proxy with that key, and the proxy uses `api_key` below
towards basal.

```yaml
model_list:
  - model_name: basal-2.0-mini
    litellm_params:
      model: hosted_vllm/basal-2.0-mini
      api_base: http://127.0.0.1:8000
      api_key: <secret>          # any non-empty value when basal-serve runs without --api-keys
```

#### `basal.client.Basal`

```python
from basal.client import Basal

client = Basal("http://127.0.0.1:8000", api_key="<secret>")
answer = client.choice("A user asks for a refund.", "Which team?", {"billing": "Billing", "support": "Support"})
```

#### curl

```bash
curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <secret>' \
  -d '{"state":"Choose a team","questions":{"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Billing","support":"Support"}}}}'

curl http://127.0.0.1:8000/v1/decisions -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <secret>' \
  -d '{"input":"Choose a team","questions":[{"type":"choice","instructions":"Which team?","choices":[{"value":"billing"},{"value":"support"}]}]}'
```

#### Verified compatibility

The transcript-backed checks used basal-serve 2.3.0 with basal-2.0-mini MLX 8-bit on Apple M4 Max. TypeSafe Python
SDK `typesafe-sdk` 0.7.5, OpenAI SDK 3.35.1 and LiteLLM 1.106.1 (SDK and proxy) worked against that server.

| Ecosystem | Matches | Differences and notes |
|---|---|---|
| TypeSafe System One | `/v1/systemone`, Bearer auth, named questions and answers, request-id header | `model` is optional and not validated; basal reports the served name. Basal adds question types, usage fields and calibration metadata. |
| vLLM 0.32 | System One route and vLLM-style `images`/`audio` data URL aliases | Remote media URLs are rejected; basal limits requests to 8 items and 20 MiB each; the model/backend must support media. |
| Ollama / llama.cpp | System One request and answer shape through basal's server integrations | Basal-specific question types and extensions may not be supported by those servers; option limits depend on the selected backend and prompt format. |
| Ollaya | System One question/answer wire shape | Basal-specific types/fields and response usage may differ. |
| OpenAI Decisions | `/v1/decisions` and `/decisions` request/answer translation | Only predicate, choice and score are translated; see limits above. Safety/metadata/store fields are accepted but ignored. Errors use route-specific OpenAI envelopes. |
