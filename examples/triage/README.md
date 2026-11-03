# Helpdesk triage with basal

[Polska wersja](README.pl.md)

## The use case

An online shop's helpdesk gets hundreds of messages a day: complaints, invoice questions, outages, password resets,
partnership offers. Today a person reads each one to decide three things:

- the team it goes to: complaints, billing, technical, account or other;
- how urgent it is;
- whether a person has to decide the case, or a ready answer is enough.

basal answers all three in one request and gives each answer a calibrated confidence. With an error target
(`max_error`), a confident answer is routed automatically. An answer below the certified threshold abstains, and that
ticket goes to a person. You choose the error you accept, for example 5%. Calibration on your own labelled tickets
then tells you what share of the traffic basal can route at that error.

```
helpdesk / e-mail / form ──> webhook (app.py) ──> basal-serve ──> answers + confidence
                                    │
                                    ├─ category certified        -> team queue (automatic)
                                    ├─ category abstains         -> "human" queue (a person routes it)
                                    ├─ top urgency level         -> priority "high"
                                    └─ needs_human yes / abstain -> a person decides the case
```

## Files

| file | what it does |
|---|---|
| `tickets.jsonl` | Polish helpdesk tickets with gold labels: `{"id", "state", "gold": {"category", "urgency", "needs_human"}}` |
| `run.py` | runs the `triage` preset on the tickets and scores the answers. Options: `--rules` (this helpdesk's rules), `--max-error` (automation), `--calibration-input` (the input of `basal-calibrate`) |
| `app.py` | a webhook: `POST /ticket {"id", "text"}` returns the queue, the priority and the answers |
| `../../tests/test_example_triage.py` | offline tests of `run.py` and `app.py`, which need no model |

## 1. Start a basal server

Install basal ([docs/INSTALL.md](../../docs/INSTALL.md)), then start a server with the command for your hardware:

```bash
# Mac (Apple Silicon), extra basal[mlx]
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000

# NVIDIA GPU
basal-serve --model Remek/basal-2.0-4.5B --mode fast --port 8000

# a small machine: the mini model (about 1.8 GB on a Mac)
basal-serve --model Remek/basal-2.0-mini-MLX-8bit --mode mlx --port 8000
```

Check that it is up:

```bash
curl -s localhost:8000/health       # {"status":"ok",...}
curl -s localhost:8000/v1/models    # the model, its temperatures and its "certified" thresholds
```

## 2. Run the example

```bash
python examples/triage/run.py --lang pl                         # the preset as it is
python examples/triage/run.py --lang pl --rules                 # the preset with this helpdesk's rules
python examples/triage/run.py --lang pl --rules --max-error 0.05
python examples/triage/run.py --lang pl --rules --output answers.jsonl   # also save every answer
```

| option | meaning |
|---|---|
| `--url` | the basal server (default `http://127.0.0.1:8000`) |
| `--lang pl\|en` | the wording of the questions. Pass `pl` for Polish tickets (see "Limits") |
| `--rules` | the preset's questions with `RULES` from `run.py` in the options |
| `--max-error 0.05` | the error target: answers below the certified threshold abstain |
| `--concurrency 8` | requests in flight; the server batches them |
| `--input` | your own tickets in the same format |
| `--calibration-input FILE` | write labelled requests for `basal-calibrate` and exit; needs no server |

What the output means:

- **accuracy**: the share of answers equal to the gold label;
- **automated**: the share of answers that did not abstain at `--max-error`;
- **error when automated**: the share of those answers that were wrong. Calibration keeps this at or below the
  target;
- **automated end to end**: tickets on which no question abstained;
- **abstained ... no certified threshold**: the calibration has no threshold for this question type at this error,
  so every such answer goes to a person.

## 3. Test it

```bash
# offline: the questions, the calibration input and the webhook, with a mock basal
uv run --extra test python -m pytest -q tests/test_example_triage.py

# live: the whole flow on a running server
python examples/triage/run.py --lang pl --rules --max-error 0.05

# the webhook on a running server
BASAL_URL=http://127.0.0.1:8000 uvicorn app:app --app-dir examples/triage --port 9000
curl -s localhost:9000/ticket -H 'content-type: application/json' \
     -d '{"id": "123", "text": "Pobrano ode mnie dwa razy 129 zł za abonament. Proszę o zwrot."}'
```

The webhook answers like this (`answers` cut short):

```json
{"id": "123", "queue": "billing", "priority": "normal", "category": "billing", "urgency": "średnia",
 "needs_human": true, "automated": false, "answers": {...}}
```

## 4. Write your rules into the questions

A preset is a starting point. `RULES` in `run.py` gives each option a description of what it covers at this
helpdesk. For example, `account` becomes "login, password, verification codes, data, address, users, closing the
account". On the 100 tickets this raised category accuracy from 70% to 91% (see "Results"). Do this before you
calibrate, because the calibration belongs to the exact questions you serve:

- write the descriptions in the language of the tickets;
- name the borderline cases explicitly, for example "a wrong size delivered is a complaint";
- keep the option keys (`billing`, `account`, ...) stable: your routing uses them.

## 5. Calibrate on your own tickets

The model ships a calibration fitted on basal's own data. On a helpdesk the model can be accurate and still be
confident in a different way. Then the shipped threshold accepts too few answers: here 0%. `basal-calibrate` fits the
temperatures and the thresholds on your labelled tickets.

**Step 1. Label tickets.** Take real tickets and record the right answers in `gold`, in the format of
`tickets.jsonl`. How many you need:

- **59** accepted decisions without an error per question type, on the 70% FIT part, for a 5% target;
- **299** for a 1% target;
- about as many again on the 30% CERT part, which confirms the threshold.

In practice that is 300 or more tickets for 5% and 1500 or more for 1%. Label honestly, and take tickets from the
whole year, not one week.

**Step 2. Write the calibration input and fit.**

```bash
python examples/triage/run.py --lang pl --rules --calibration-input labelled.jsonl
basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --out CALIBRATION.triage.json
# or against a running server started with --no-calibration:
basal-calibrate labelled.jsonl --url http://127.0.0.1:8000 --model Remek/basal-2.0-4.5B-MLX-8bit
```

Use the same model, mode (`mlx`, `fast`, `gguf`, ...), quantisation and questions as in production: the thresholds
belong to them.

**Step 3. Read the report.**

| line | what to check |
|---|---|
| `accuracy` per type | how often the answer is right, before any threshold |
| `ECE on CERT shipped -> fitted` | the calibration error. Lower is better; the fit should lower it |
| `automated on CERT`, `max_error 0.05` | **the main number**: the share basal routes automatically, and its error, on tickets the fit did not see |
| `0.05 per type: choice 0.93 (73%)` | the threshold of each question type; `none (defers)` means that type always goes to a person |
| `warning: ... too few to confirm on CERT` | the threshold is fitted, but not yet confirmed. Label more tickets |

`--report-only` prints the report without writing a file. `--save-rows rows.jsonl` stores the probabilities, and
`--from-rows` refits from them without the model, for example with other `--targets`.

**Step 4. Serve with the calibration.**

```bash
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --calibration CALIBRATION.triage.json --port 8000
curl -s localhost:8000/v1/models | python -m json.tool    # "certified" now shows your thresholds
python examples/triage/run.py --lang pl --rules --max-error 0.05
```

**Calibrate again when** the model, mode, quantisation or question wording changes, when a new kind of ticket
appears, or every few months on fresh labelled tickets.

## 6. Connect it to other applications

**Python (your backend)**, with the client `basal.client`:

```python
from basal.client import Basal
import run   # examples/triage/run.py: triage_questions, predicted

b = Basal("http://127.0.0.1:8000", retries=2)          # api_key="..." when the server uses --api-keys
text = "Ktoś zalogował się na moje konto z obcego kraju."
answers = b.decide(text, run.triage_questions(text, "pl", max_error=0.05, rules=True))
cat = answers["category"]
queue = "human" if cat.get("abstain") else cat["choice"]
```

`AsyncBasal` has the same methods for asyncio (FastAPI, Starlette, aiohttp).

**Any language over HTTP**: POST the questions to `/v1/systemone`. The body is the one in `labelled.jsonl` without
`gold`, plus `max_error` on each question ([docs/API.md](../../docs/API.md)):

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Pobrano dwa razy 129 zł.",
  "questions": {"category": {"type": "choice", "instructions": "Czego dotyczy zgłoszenie?",
                             "criteria": {"billing": "płatności, faktury, zwroty", "other": "coś innego"},
                             "max_error": 0.05}}}'
```

**Helpdesk systems (Zendesk, Freshdesk, Jira Service Management, a mail gateway)**: point their webhook or
automation at `app.py` (`POST /ticket` with `{"id", "text"}`). Write the result back through their API: `queue` as
the group, `priority` as the priority, and `automated: false` as a "needs review" tag. Run `app.py` next to
basal-serve, on the internal network.

**n8n, Make, Zapier**: an HTTP Request node, `POST http://<host>:9000/ticket` with JSON `{"id": "{{id}}", "text":
"{{body}}"}`, then a Switch node on `queue`. Or call `/v1/systemone` directly with the body above.

**Batch: a night run over an export**:

```bash
basal-run --preset triage --lang pl --input export.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
```

`basal-run --preset` sends the preset without `RULES`. For your own rules, write request lines (`run.py
--calibration-input` writes them; drop `gold`), or use `run.py --input export.jsonl --output answers.jsonl`.

**Agents (LangChain, LlamaIndex, CrewAI, MCP)**:

```python
from basal.adapters import langchain_tool
triage = langchain_tool("http://127.0.0.1:8000", preset="triage", lang="pl")   # the agent passes the ticket only
```

For MCP, `basal-serve --mcp on` serves `/mcp`, and `basal-mcp` bridges to hosts that start local processes
([docs/MCP.md](../../docs/MCP.md)).

**Security.** The server is open by default:

- `--api-keys keys.json` requires a key on every route but `/health` ([docs/SERVING.md](../../docs/SERVING.md#api-access));
- bind `--host 127.0.0.1` when only `app.py` on the same machine calls basal;
- tickets hold personal data, so keep basal and the webhook inside your network.

## 7. In production: close the loop

1. **Start in shadow mode.** basal answers, people still route every ticket, and you compare the two.
2. **Then turn on `max_error`.** Route the certified answers automatically, and send the rest to the `human` queue.
3. **Keep labelling.** Record each person's final decision as a new `gold`, and audit a sample of the automated
   tickets as well. Without that sample you only see the errors in the cases basal was unsure about.
4. **Recalibrate on the new labels** every few months, or when the share automated or the error drifts.

`basal-serve --log-decisions decisions.jsonl` writes a log of the decisions. `GET /metrics` gives Prometheus metrics.

## Results

Measured on an Apple M1 Max with basal 2.5.2. A run over 100 tickets takes about 40 s on the 4.5B model.

### Accuracy

The 4.5B rows were run with `--lang pl`. The mini rows were run earlier, before `--lang pl` was added, so their
wording followed each ticket's language.

| model | tickets | questions | category | urgency | needs_human |
|---|---|---|---|---|---|
| basal-2.0-mini-MLX-8bit | 1-40 | preset | 82% | 55% | 52% |
| basal-2.0-mini-MLX-8bit | 1-40 | `--rules` | 95% | 55% | 72% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | preset | 70% | 66% | 80% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | `--rules` | 91% | 80% | 81% |

### Automation at 5% error, basal-2.0-4.5B-MLX-8bit, 100 tickets

| calibration | category automated | error when automated | urgency, needs_human |
|---|---|---|---|
| shipped (`CALIBRATION.json`) | 0% (threshold 0.977) | - | no threshold: every answer abstains |
| fitted on the 100 tickets | 83% (threshold 0.929) | 1 of 83 | no threshold: every answer abstains |
| the same, on its CERT split only | 73% | 1 of 22 | - |

- **The 83% row is optimistic:** it is measured on the same tickets the thresholds were fitted on.
- **The honest number is CERT:** 73% automated with 1 error in 22.
- **The 5% threshold is not yet confirmed.** `basal-calibrate` reports that CERT has too few decisions.
- **`urgency` and `needs_human` get no threshold.** Too few of their answers are both confident and right, so those
  questions stay with people.

The fit lowers the calibration error (ECE on CERT):

| question | shipped | fitted |
|---|---|---|
| category | 0.122 | 0.087 |
| urgency | 0.140 | 0.071 |
| needs_human | 0.126 | 0.121 |

## Limits

- **Pass `--lang pl` for Polish tickets.** Without it, the wording follows each ticket. `presets.lang_of` reads t33,
  "Aplikacja na iPhone wylogowuje mnie co kilka minut.", as English: it has no diacritics and only one common Polish
  word.
- **The tickets are written for this example.** Real tickets are longer and messier. Measure on your own.
- **`needs_human` depends on your policy.** The descriptions in `RULES` encode one helpdesk's policy. Write yours.
