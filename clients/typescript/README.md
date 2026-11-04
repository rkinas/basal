# basal-client (TypeScript)

A client for `basal-serve`'s `POST /v1/systemone` ([docs/API.md](../../docs/API.md)). It sends the same requests and
reads the same answers as the Python client (`basal/client.py`), with types for every question type and answer.

- `fetch` only: Node >= 18.17 (`AbortSignal.any`), Deno, Bun and browsers; no runtime dependencies.
- Types for questions, answers (`choice`, `noul`, `score`, `multi`, `act`, `tool`), answer sets, abstain, `ask_if`,
  images, usage and errors. `decide()` returns answers typed by the question's `type`.
- Not published to npm yet: build it from this directory (below).

```bash
cd clients/typescript && npm install && npm run build      # dist/
```

## Use

```ts
import { Basal, imageFromBytes } from "basal-client";

const basal = new Basal({ url: "http://127.0.0.1:8000" });   // apiKey, timeout (s), earlyExit, facts, headers, fetch

// one question
const a = await basal.choice(ticket, "Which department should handle this?", {
  billing: "Billing", tech: "Technical support",
});
a.choice; a.confidence; a.probabilities;                      // typed: ChoiceAnswer

// several questions over one state (one pass on the server)
const r = await basal.decide(ticket, {
  urgency: { type: "score", instructions: "How urgent is this?", criteria: ["low", "medium", "high"] },
  refund: { type: "noul", instructions: "Does the customer ask for a refund?", max_error: 0.05 },
});
r.urgency?.score; r.refund?.abstain;                          // null when the question was skipped (ask_if)

// answer sets and abstain: certified thresholds come from `basal-calibrate` (docs/CALIBRATION.md)
const c = await basal.choice(ticket, "Which team?", ["billing", "tech"], { coverage: 0.9, max_error: 0.05 });
c.answer_set; c.abstain;

// a ready question set, the whole response (usage, skipped questions), images
await basal.preset(ticket, "triage");                         // "triage" | "moderation" | "compliance" | "guard_input" | ...; pl / en
await basal.guard(userText, { kind: "input", policy: "strict", maxError: 0.05 });  // pass | review | block | support
await basal.route(ticket);                                    // the intent, or "human" on low confidence / abstain
const full = await basal.request(ticket, { q: { type: "noul", instructions: "Is it a receipt?" } },
  { images: [imageFromBytes(bytes, "image/png")], detail: "high" });   // servers started with --media on
full.usage.skipped;
```

Per-request fields (`facts`, `think`, `early_exit`, `lang`, `detail`, `media`, `images`, and any other field, sent
as it is) go in the last argument, with an optional `signal` (`AbortSignal`) to cancel. Question fields `ask_if`,
`depends_on`, `coverage`, `max_error`, `option_keys` and `evidence` are in the types. `basal.models()` and
`basal.health()` read `GET /v1/models` and `GET /health`.

## Errors

Every failure throws a `BasalError` with `kind`, `status`, `detail` (the server's message) and `retryable`:

| kind | when | retryable |
|---|---|---|
| `validation` | HTTP 422: the request must change | no |
| `unavailable` | HTTP 503: the GPU ran out of memory, or the server is still starting | yes |
| `server` | HTTP 500 and any other status | no |
| `timeout`, `network` | no answer, or the response broke off (`timeout` covers the body too) | yes |

Both error formats (`{"error": "..."}` and `basal-serve --error-format openai`) are read. A request you cancel with
your own `signal` rethrows the abort error.

## Tests

`npm test` builds, type-checks `test/types.check.ts` and runs `node --test` against a real `basal-serve` (its HTTP app
and request handling) on the repo's stub engine, the one the Python tests use (`test/stub_server.py`), so no model or
GPU is needed. Give it a Python environment that has the repo's dependencies:

```bash
BASAL_PYTHON=/path/to/env/bin/python npm test
```

The stub reads the probabilities from weights written at the end of each option (`"first ~0.6"`); the tests use that
to know the answer they expect. `test/presets_fixture.json` holds the Python presets, to check that
`src/presets.ts` (generated from `basal/presets.py`) has not drifted.
