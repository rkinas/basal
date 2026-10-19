# Features beyond choice, yes/no and score

basal-1.5 adds request features on top of the three System One types. They are accepted on `POST /v1/systemone` and
`POST /v1/basal`; clients that use only the official fields are not affected.

| feature | what it does | status |
|---|---|---|
| [SOAM](SOAM.md) | all questions of a request over one shared state | stable, on by default |
| [`facts`](#facts-auto) | computed calendar and money facts appended to a Polish state | stable (the models are trained with it) |
| [`multi`](#multi-which-labels-apply) | independent probability per label and the selected set (threshold, or the set with the best expected IoU / F1 / exact match) | stable API; labels are answered as yes/no questions |
| [`act`](#act-cost-aware-actions) | the action with the lowest expected cost, or defer to a person | **experimental** |
| [`tool`](#tool-which-tool-next) | the next tool of an agent from its OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | **experimental** |
| [`evidence`](#evidence-spans) | spans of the state that support the answer | **experimental** |
| [Agents and tool use](#agents-and-tool-use) | next step, tool, ask the user, confirm; [one request per loop step](#a-step-of-an-agent-loop) | uses `tool`, `choice`, `noul`, `multi`, `act` |

*Experimental* means: the API may change in a minor release, and the measured quality is below what we would call
production-ready; use it with a person in the loop and check it on your own data.

## `facts: "auto"`

Exact arithmetic is a weak spot of one-pass decisions: a model that answers in one forward pass cannot reliably add
14 days to a date or compare two gross amounts. With `"facts": "auto"` the server computes such facts from a Polish
state and appends them, under the header *Fakty pomocnicze (wyliczone automatycznie, bez oceny prawnej)*:

- every date: its weekday, and whether it is a Saturday or a statutory day off (with the holiday's name);
- gaps in days between consecutive dates, and "today minus N days" when the text states today's date;
- date + duration for the durations the text mentions ("14 dni", "2 tygodnie", "6 miesięcy");
- net / gross amounts for a VAT rate the text states (23%, 8%, 5% with "VAT" in the sentence);
- euro × a stated exchange rate, budget minus what was spent, quantity × unit price and list totals, the tier of a
  quantity-tiered price list and a stated delivery fee;
- comparisons of the amounts, as symbols only (`>`, `<`, `=`).

Only arithmetic and calendar facts are added, never a rule ("the deadline is shifted to Monday") or a verdict ("the
amount exceeds the limit"): which fact matters is the model's decision. The models were trained with and without the
block, so they work either way; in development tests on Polish date and amount decisions the block raised accuracy by
about 5 points. Evidence spans never point into the facts block.

```json
{"state": "Towar odebrano 3 marca 2025 r. Oświadczenie o odstąpieniu wysłano 20 marca 2025 r. ...",
 "facts": "auto",
 "questions": {"in_time": {"type": "noul", "instructions": "Czy odstąpienie od umowy złożono w terminie?"}}}
```

Polish only (dates in Polish formats, amounts in zł / PLN / euro). The same code is importable:
`from basal.facts import inject`.

The facts are computed from the state only. Dates, durations and amounts written in a question's `instructions` or
`criteria`, such as a rule "zwrot w ciągu 14 dni od doręczenia" placed in the question, are not seen, so the block
cannot relate them to the dates of the state. Put such a rule in the state, after the facts of the case: the block
reads the first dates of the text.

## `multi`: which labels apply?

```json
{"state": "Klient: Laptop zamówiony 3 tygodnie temu przyszedł z pękniętym ekranem, a kurier rzucił paczką pod drzwi. Nie chcę naprawy, chcę zwrotu pieniędzy.",
 "questions": {"tags": {"type": "multi", "instructions": "Które kategorie dotyczą zgłoszenia?",
   "criteria": {"damage": "Towar uszkodzony", "delivery": "Problem z dostawą lub kurierem",
                "refund": "Żądanie zwrotu pieniędzy", "invoice": "Faktura", "repair": "Naprawa gwarancyjna"},
   "max": 3}}}
```

Response fields: `probabilities` (one independent P(applies) per label; they do not sum to 1), `selected` (labels with
probability ≥ `threshold`, default 0.5, at most `max`, at least `min`), `threshold` and `set_confidence` (the
probability that every selected label applies and every other label does not, assuming independence).

### Choosing the set: `objective`

A fixed threshold is rarely the best set for the score you are measured on. With `"objective"` the engine chooses
the set that maximises the expected value of that score, computed from the label probabilities (no retraining):

| `objective` | the selected set | best for |
|---|---|---|
| `threshold` (default) | labels with p ≥ `threshold` (default 0.5), at most `max`, at least `min` | the behaviour of earlier releases |
| `exact` | labels with p ≥ 0.5: the most probable set | exact-set accuracy (all labels right) |
| `iou` | the set with the highest expected IoU (Jaccard index) with the true set | IoU of selected sets |
| `f1` | the set with the highest expected F1 with the true set | F1 over labels |

```json
{"state": "...", "questions": {"tags": {"type": "multi", "instructions": "Które kategorie dotyczą zgłoszenia?",
  "criteria": {"damage": "Towar uszkodzony", "delivery": "Problem z dostawą", "refund": "Żądanie zwrotu pieniędzy"},
  "objective": "iou", "min": 1, "max": 3}}}
```

With an objective other than `threshold` the response has `objective` and `expected_score` (the expected IoU, F1 or
exact-match probability of the returned set) instead of `threshold`; `min` and `max` bound the size of the set
(`threshold` is refused with these objectives). F1 and IoU reward a label whose probability is somewhat below 0.5
when the set is otherwise small: with probabilities 0.9, 0.45, 0.4 and 0.1, `exact` returns one label and `f1`
three.

- **How it works.** Under independence the best set of each size is the top-k labels by probability, so the engine
  scores every prefix of the labels sorted by probability (k = `min` … `max`) and returns the best. The expected
  scores are exact: IoU = A / (k + B) and F1 = 2A / (k + A + B), where A and B are the positives inside and outside
  the prefix (two independent Poisson-binomial variables), and their expectations are computed through the
  generating functions of A and B by Gauss–Legendre quadrature with enough nodes to be exact for these polynomials.
  All prefixes of 255 labels take about 0.2–0.3 ms on a laptop CPU. The tests compare every prefix with brute force
  over all true sets, and the chosen set with all subsets within the size bounds.
- **Independence.** The probabilities are per-label answers, so the expectations assume independent labels; labels
  that imply each other (a child category and its parent) make the expected scores approximate.
- **The empty set** is a candidate whenever `min` is 0 or absent (nothing applies, e.g. an out-of-domain request): its
  expected score is the probability that no label applies, since an empty answer to an empty truth counts as a perfect
  match under every objective.

### Label constraints: `exclusive` and `fallback`

Some label sets have a label that must stand alone: a "general / no narrowing" category, or "other" for requests that
fit none of the specific labels.

```json
{"type": "multi", "instructions": "Które kategorie dotyczą zgłoszenia?", "objective": "iou",
 "criteria": {"damage": "Uszkodzenie towaru w transporcie (pęknięcia, wgniecenia, zalanie) zgłoszone w ciągu 14 dni",
              "refund": "Żądanie zwrotu pieniędzy", "general": "Sprawa ogólna, bez wskazania konkretnej kategorii",
              "other": "Żadna z kategorii"},
 "exclusive": ["general"], "fallback": "other"}
```

- `"exclusive": [labels]`: each of them is selected only alone. The best set of the other labels competes with each
  exclusive label alone under the same objective, and the better one is returned (`threshold`: an exclusive label with
  p ≥ `threshold` wins when it is more probable than every label of the thresholded set).
- `"fallback": label`: selected only when no other label is, so implicitly exclusive. Under `threshold` it needs p ≥
  `fallback_threshold` (default: `threshold`). Under `iou`, `f1` and `exact` it is selected when it scores higher alone
  than the empty set. An explicit `"fallback_threshold"` decides under every objective; this helps when a model
  under-selects its general label.
- The response then has `constraints_applied`: the exclusive labels, the fallback (and its threshold), and `chosen`:
  `labels` (a set of the other labels won), `exclusive`, `fallback` or `empty`. With a fallback,
  `fallback_threshold_source` says where its threshold came from: `request`, `calibration` or `default`.
- `basal-calibrate` fits `fallback_threshold` per label set from your labelled questions and stores it in the
  calibration file ([CALIBRATION.md](CALIBRATION.md#multi-the-fallback-threshold-of-a-label-set)). A request with the
  same labels and fallback and no `fallback_threshold` of its own uses that value.
- Labels may be described at length (a scope of a few hundred characters each): the description goes into the label's
  yes/no question as it is.

- **How it works.** Each label becomes a yes/no branch over the shared state ("Czy dotyczy: „Towar uszkodzony”?" /
  "Does this apply: ..."), answered with the yes/no calibration. With SOAM the cost is one state plus a short branch per
  label.
- **When to use it instead of `choice`.** When several answers can be right at once. A `choice` would have to split
  its probability between two correct labels, which looks like uncertainty but is not.
- **Limits.** The labels are independent questions: the engine does not enforce consistency between them (for example
  mutually exclusive labels). Not available with `evidence`.

## `act`: cost-aware actions

**Experimental.** A plain decision returns probabilities, and the caller still has to turn them into an action,
usually with a hand-picked threshold. `act` does that step with the caller's own costs.

```json
{"state": "Zamówienie #88213, 1 240 zł, dostarczone 6 dni temu. Klientka pisze: odstępuję od umowy, towar nieużywany, w oryginalnym opakowaniu. Regulamin: 14 dni na odstąpienie od umowy od dnia doręczenia.",
 "questions": {"refund": {"type": "act",
   "instructions": "Czy zwrot przysługuje zgodnie z regulaminem?",
   "criteria": {"true": "Przysługuje", "false": "Nie przysługuje"},
   "costs": {"approve": {"true": 0,   "false": 1240},
             "reject":  {"true": 300, "false": 0},
             "human":   {"true": 15,  "false": 15}}}}}
```

For every action the engine computes the expected cost under the calibrated probabilities,
E[cost(a)] = Σ_y p(y) · C[a, y], and returns the cheapest one, with all expected costs, the most likely outcome, the
probabilities and a `calibration` flag. A person (or `defer`) is just another action with a fixed cost, so the engine
automates exactly when automating is cheaper on average than asking someone. In this example an automatic approval
needs P(true) > 1 − 15 / 1,240 = 0.988 and an automatic rejection P(false) > 1 − 15 / 300 = 0.95; anything less
confident goes to the consultant.

- **Short form.** `"costs": {"wrong": 50, "defer": 2}`: answering with any outcome costs 50 when wrong, deferring
  costs 2 (the action names are then the outcome keys and `defer`).
- **`max_error`.** `"max_error": 0.01` also refuses automatic actions below the confidence threshold certified for 1%
  error for the question's type in the calibration file (`noul` for a yes/no question, `choice` for named outcomes);
  the response then carries `refused` (the action that was refused), the defer action (`defer`, `human` or the one
  named in `"defer_action"`) and `"deferred": true`. When the type has no certified threshold for the target
  (`--no-calibration`, a target other than those in the file, a type without a threshold of its own, or
  `"confidence": null` from `basal-calibrate`), every automatic action is refused and the response says
  `"max_error": "no certified threshold for this type at this max_error"`. The threshold pooled over all types is never applied (a
  basal-1.5 file, which has no per-type thresholds, keeps its pooled one).
- **`calibration`.** `validated` when the calibration file has temperatures and the question is a yes/no question or
  uses `"option_keys": "hide"` (the formats whose calibration we measured); otherwise `unvalidated`, and the expected
  costs are only as good as the probabilities.
- **Measured.** On yes/no test decisions with three cost matrices fixed in advance, `act` was cheaper than a
  confidence threshold chosen on separate data on every matrix of the calibration test split, and on two of three on
  the test split (one tie). It was *not* better than a threshold tuned in hindsight on the evaluation items
  themselves, because the highest confidences are slightly overconfident. That is why it is experimental.

## `tool`: which tool next

**Experimental.** The tool definitions an agent already has (OpenAI function tools, or the result of an MCP
`tools/list`) become the options of a choice question. basal returns the tool to call, the probability of every
option and, on request, a ranking.

```json
{"state": "User: Where is my order 5512? I paid for it last week.",
 "questions": {"next": {"type": "tool", "rank": 3,
   "tools": [
     {"type": "function", "function": {"name": "get_order", "description": "Look up an order by its number.",
       "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "fields": {"type": "array"}},
                      "required": ["order_id"]}}},
     {"name": "refund", "description": "Refund a paid order.",
      "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"},
                                                       "reason": {"type": "string"}},
                      "required": ["order_id", "amount"]}},
     {"name": "send_message", "description": "Send a message to the customer."}]}}}
```

The model sees six options:

```text
A. get_order(order_id, fields?) — Look up an order by its number.
B. refund(order_id, amount, reason?) — Refund a paid order.
C. send_message() — Send a message to the customer.
D. Answer directly without calling a tool
E. Ask the user for the missing information
F. No suitable tool — tell the user
```

The answer has this shape:

```jsonc
{"type": "tool", "tool": "get_order", "call": true,
 "probabilities": {"get_order": p, "refund": p, "send_message": p, "answer_directly": p, "ask_user": p, "none": p},
 "confidence": p,
 "ranking": [{"tool": "get_order", "probability": p}, {"tool": "...", "probability": p}, {"tool": "...", "probability": p}]}
```

- **Tool definitions.** `tools` accepts four forms, and they can be mixed:
  - OpenAI Chat Completions: `{"type": "function", "function": {...}}`;
  - OpenAI Responses: `{"type": "function", "name", ...}`;
  - MCP: `{"name", "description", "inputSchema"}`;
  - Anthropic: `input_schema`.

  Each tool becomes one option, `name(a, b, c?) — description`. The top-level parameters appear in schema order,
  optional ones are marked `?`, and the description is put on one line. The model does not see the schemas. To check
  arguments, ask a separate question (see [below](#a-step-of-an-agent-loop)).
- **Long descriptions.** A description is cut at 300 characters, on a word boundary, and ends with `…`. Real MCP
  descriptions can run to several hundred characters, and 150 of them would make a prompt too long for the model.
  `"description_chars": n` sets another cap, and `0` turns the cap off. The default is `basal.tools.DESCRIPTION_CHARS`.
- **`extra`.** These are the options besides the tools, listed after them in the language of the request:
  - `answer_directly`: answer without a tool;
  - `ask_user`: ask for missing information;
  - `none`: no suitable tool, so tell the user.

  The default is all three; `"extra": []` leaves only the tools. A tool whose name matches an extra option in use is
  refused.
- **Response.** `tool` is the chosen key: a tool name or an extra option. `call` says whether it is a tool.
  `probabilities` covers every option, and `confidence` is the probability of the chosen one. `"rank": k` adds the k
  most probable options in order.
- **`max_error`.** It works as in `act`, with the threshold certified for `choice` questions. If the confidence is
  below it, or the file has no such threshold, the decision goes to a person: `tool` is `null`, `deferred` is `true`,
  and `refused` names the option the model would have chosen. With `max_error` the answer always has `deferred`.
- **`"multi": true`.** This returns the set of tools to call now, asking one yes/no question per tool. It takes the
  `multi` [objectives](#choosing-the-set-objective) and [constraints](#label-constraints-exclusive-and-fallback):
  `objective`, `threshold`, `min`, `max`, `exclusive`, `fallback`. The answer has `tools` (the selected set) instead
  of `tool`; an empty set means there is nothing to call. Without `extra`, the set contains only tools. Extra options
  that are given can only be selected alone. `max_error` is not available with `multi`.
- **How many tools.** A question can have up to 255 options (tools plus extra options) with prompt-format-2 models,
  10 with format 1 and 20 in the `ollama` mode (see [FORMATS.md](FORMATS.md)). The context usually binds first; see
  [how many tools fit](#how-many-tools-fit-and-shortlists). The options follow the state in the prompt, so long
  descriptions make the question longer, and unlike the state they are not shared between questions. A prompt
  longer than the context is refused with HTTP 422, which gives its token count and the limit.
- **Errors.** These return HTTP 422:
  - a missing or empty `tools`;
  - duplicate tool names;
  - a name with spaces;
  - a tool type other than `function`;
  - a parameter schema that is not a JSON-schema object: `type` other than `"object"`, `properties` that is not an
    object of schemas, or `required` naming a parameter that is not among the properties;
  - an unknown extra option;
  - a `rank` outside 1 to the number of options, or a negative `description_chars`.
- **Calibration.** The probabilities use the calibration of `choice`; with `multi` they use that of yes/no
  questions. Tool catalogues have no calibration of their own yet, so measure the coverage of `max_error` on your
  own traffic.

### How many tools fit, and shortlists

Every tool costs prompt tokens:
- about 35 tokens for its name and parameters;
- plus its description, at about 3 characters per token for English text in these tokenizers.

With MCP-style descriptions (150 tools averaging 600 characters, cut at the cap) we measured:

| `description_chars` | tokens per tool | tools that fit basal-1.5-mini (8,192-token context) | tools that fit basal-1.5 and basal-1.5-max (32,768) |
|---|---|---|---|
| 300 (default) | about 143 | about 55 | about 225 |
| 150 | about 87 | about 90 | 252 (the option limit) |
| 80 | about 60 | about 130 | 252 |
| 50 | about 49 | about 165 | 252 |

The state and the question come on top. The engine checks the rendered prompt (state, question, options and the
answer prefix) against the model's context, or against the engine's own limit when that is lower: `--max-len` for
vLLM and SGLang, 4,096 tokens for Ollama (sent as `num_ctx` with every request; it overrides the Modelfile), and
llama-server's context per slot (`-c`). With several questions over one state,
the state counts once plus the longest question.

Prompt length also sets the latency, because every option is read in the prefill. basal-1.5-mini on an Apple M4 Pro
(MLX) takes about 4.9 s at 7,500 tokens and 22 s at 22,000 tokens.

**Shortlist large catalogues.** Above about 50 tools for basal-1.5-mini, or whenever prompts get long on any model,
prefilter to about 30–50 tools before asking:

1. Index the tools once: an embedding of `name + description` (any embedding model), or a keyword index (BM25) over
   the same text.
2. Per step, query the index with the user's request (and the last tool result, if the agent is mid-task). Take the
   top 30–50 tools, plus any tool the agent used in the last steps.
3. Ask `tool` over the shortlist. Keep the extra options: `none` then also catches a shortlist that missed the right
   tool, and the client can retry with a wider shortlist.

At the default cap, 40 tools are about 5,800 tokens, which fits every model.

### A step of an agent loop

Use one request per step, over one state: the task, the operating policy, and the steps so far with their results.
The agent's LLM drafts the next call (a tool and its arguments), and basal checks the step with four questions:

```jsonc
{"state": {"policy": "Refunds above 500 EUR need the user's confirmation.",
           "task": "user: Please refund the 740 EUR I paid for order 5512.",
           "steps": [{"call": "get_order", "arguments": {"order_id": "5512"},
                      "result": {"id": "5512", "paid": 740, "status": "delivered"}}]},
 "questions": {
   "done": {"type": "noul", "instructions": "Do the results so far complete the user's request?"},
   "next": {"type": "tool", "tools": [...], "rank": 3, "max_error": 0.05},
   "args": {"type": "noul", "instructions": "Proposed call: refund({\"order_id\": \"5512\", \"amount\": 740}). Its schema: {...}. Are all required arguments present, of the right type, and taken from the conversation rather than invented?"},
   "control": {"type": "choice", "instructions": "How should the agent continue?",
               "criteria": {"continue": "Continue with the next call", "retry": "Retry the last call (a temporary error)",
                            "confirm": "Ask the user to confirm first", "escalate": "Hand over to a person",
                            "stop": "Stop and report to the user"}}}}
```

The client then acts on the first rule that applies:

1. `done` is above your threshold: answer the user and stop.
2. `next` is `deferred`, or `control` is `escalate`: hand over to a person. If `control` is `confirm`, ask the user.
3. `next.tool` is the drafted tool and `args` is above your threshold: execute the call, add the result to the
   state, and start the next step.
4. `next.tool` is a different tool: let the LLM draft arguments for that tool, and check them in the next request.
5. `next.tool` is `ask_user`, `answer_directly` or `none`: do that.
6. `args` is below your threshold: let the LLM correct the arguments, or ask the user for a missing value.

- **Put the draft only in the question.** The draft call, and the schema of its tool, go into the instructions of
  `args`, not into the state. The instructions belong to that question alone, so `next` is decided without seeing the
  draft.
- **One state, four decisions.** The questions share the state (SOAM), which is computed once. A step costs about as
  much as one decision plus the short question suffixes.
- **Results are data.** Keep tool results in the state and out of the instructions. The models treat the state as
  data, not as instructions, which limits instructions injected through tool results.

## Evidence spans

**Experimental.** `"evidence": true` on a question returns the spans of the state that support the model's answer:

```json
{"state": "Umowę najmu zawarto 1 marca 2024 r. na czas nieokreślony. § 7. Każda ze stron może wypowiedzieć umowę z zachowaniem trzymiesięcznego okresu wypowiedzenia. § 9. Czynsz jest płatny do 10. dnia miesiąca.",
 "questions": {"notice": {"type": "noul",
   "instructions": "Czy okres wypowiedzenia umowy jest dłuższy niż jeden miesiąc?",
   "evidence": true}}}
```

The answer gets an `evidence` list of up to three non-overlapping spans, sorted by probability, each
`{"text", "start", "end", "probability"}`. `start` / `end` are character offsets into the request's state (into the
compact JSON serialisation when the state is a JSON value), so `state[start:end] == text`: the quote is always
verbatim and can be highlighted in a UI or stored in an audit log.

- **How it works.** A small pointer head, shipped as `evidence_head.pt` next to the weights, scores the start and end
  of a span over the state's tokens, conditioned on the model's state at the answer position, so the span depends on
  the question. It was trained after the decision model on its frozen states, so it never changes a decision. It runs
  on one extra pass per question that asks for evidence (original option order).
- **Cost.** One extra forward pass over the prompt: with basal-1.5 on real documents on one H100 the request latency
  rises from 40 to 85 ms (p50, served) with evidence on (basal-1.5-mini: 25 to 51 ms; basal-1.5-max: 80 to 158 ms).
- **Measured quality** (basal-1.5 served, 200 held-out real documents): every returned span is verbatim; token-F1 of the top
  span against a verified quote is 0.53 (basal-1.5-mini: 0.50; basal-1.5-max: 0.58), so the span often covers only part of the quote or a neighbouring
  sentence.
  Documents also state the same fact several times, so removing one span frequently leaves the information in place;
  treat the span as *where the model looked*, not as proof that it is the only reason.
- **Use in RAG and document QA.** Put the retrieved passages into the state (numbered text or a JSON list), ask the
  typed question ("Do the passages support the claim?", "Which passage answers the question?", "Can it be answered from
  the passages?") with `"evidence": true`, and show or log the returned spans with the decision.
- **Limits.** Trained mostly on Polish documents; English spans are less tested. Spans are at most 80 tokens. Not
  available for `multi`, and only in the modes that run the model in process (`fast`, `fast-nocompile`, `fp8`,
  `nvfp4`, `fast-exit`, `eager`; not `vllm`, `sglang` or the ports). A model without `evidence_head.pt` answers HTTP 422.

## Agents and tool use

basal decides **what** an agent does next; the LLM or code writes free-text arguments. Typical questions over one
state (the user request, the conversation so far, tool results, the tool catalogue or a UI tree):

| question | type |
|---|---|
| Which tool next, from the agent's tool definitions (plus "answer directly", "ask the user", "no suitable tool")? | [`tool`](#tool-which-tool-next) |
| Which step next, when the options are not tools (plan steps, sub-agents, workflow states)? | `choice` |
| Which element of the page to click or fill next? | `choice` (options may be JSON objects) |
| Is a required argument missing? Does the call need human confirmation? Did the tool result complete the task? | `noul` |
| Which tools will the task need? | `tool` with `"multi": true`, or `multi` |
| Execute, ask for confirmation or hand over, given the cost of a wrong call | `act` |
| How risky is the call? | `score` |

The 1.5 models are trained for action selection: the next step of an agent, a user interface or a workflow under an
explicit operating policy. Werdykt's *action* category measures it: basal-1.5 0.630 (best other system: GPT-6 Astra 1.000).

```json
{"state": {"policy": "Refunds above 500 EUR need the user's confirmation before the refund tool is called.",
           "conversation": ["user: Please refund the 740 EUR I paid for order 5512.", "tool order_lookup: {\"id\": 5512, \"paid\": 740, \"status\": \"delivered\"}"],
           "tools": ["order_lookup", "refund", "send_message"]},
 "questions": {"next": {"type": "choice", "instructions": "What should the agent do next?",
   "criteria": {"refund": "Call refund(5512, 740)", "confirm": "Ask the user to confirm the refund of 740 EUR",
                "lookup": "Call order_lookup again", "answer": "Answer the user without a tool call"}}}}
```

When the options are the agent's tools, the [`tool`](#tool-which-tool-next) type builds them from the tool
definitions. [A step of an agent loop](#a-step-of-an-agent-loop) shows a whole step in one request. A question can
have up to 10 options with the basal-1.5 models and up to 255 with prompt-format-2 models (see
[FORMATS.md](FORMATS.md)). For larger tool catalogues, shortlist first (for example with an embedding search) and
ask basal to choose from the shortlist.

## Availability per engine

All modes run behind the same `basal-serve` HTTP API, so the server-level features work everywhere; what differs is
how the shared state is reused, the calibration file and evidence.

| | basal engine (`fast`, `fp8`, …) | `eager` (CUDA / MPS / CPU) | `vllm` | `sglang` | `mlx` | `ollama` / `llamacpp` |
|---|---|---|---|---|---|---|
| choice / noul / score, `multi`, `act`, `tool`, `facts` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| two option orders | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| calibration file | `CALIBRATION.json` (validated) | `CALIBRATION.json` | `CALIBRATION.vllm.json` if shipped, else `CALIBRATION.json` (1.5: agreement 0.997) | `CALIBRATION.sglang.json` if shipped, else `CALIBRATION.json` | `CALIBRATION.mlx.json` if shipped, else the bf16 file (not validated) | `CALIBRATION.<mode>.json` if shipped, else the bf16 file (not validated) |
| shared state (SOAM) | one packed row | – (same answers, no speed-up) | prefix caching | RadixAttention | shared-prefix KV cache | the runtime's prompt cache (sequential requests) |
| `evidence` | ✓ | ✓ | – | – | – | – |
| options per question (format 2) | 255 | 255 | 255 | 255 | 255 | `llamacpp` 255; `ollama` 20 |
| `--deterministic` | ✓ | ✓ | – | – | ✓ | – |
| longest prompt (longer: HTTP 422 with the token count) | model context | model context | 4,096 tokens by default (`--max-len`) | 4,096 tokens by default (`--max-len`) | model context | Ollama 4,096 (fixed) / llama.cpp `-c` |
