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
                                    ├─ danger p >= 0.2           -> "alarm" queue, priority urgent (checked first)
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
| `holdout.jsonl` | 97 held-out tickets from four other kinds of business, for an honest test of the rules and the calibration (see "Rules v2 and a held-out test") |
| `safety.jsonl` | 40 tickets for a blind test of the `danger` question: 12 risky, 28 tricky but safe |
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
account". On the 300 tickets this raised category accuracy from 68% to 87% (see "Results"). Do this before you
calibrate, because the calibration belongs to the exact questions you serve:

- write the descriptions in the language of the tickets;
- name the borderline cases explicitly, for example "a wrong size delivered is a complaint";
- keep the option keys (`billing`, `account`, ...) stable: your routing uses them.

## 5. Calibrate on your own tickets

The model ships a calibration fitted on basal's own data. On a helpdesk the model can be accurate and still be
confident in a different way. Then the shipped threshold accepts too few answers: here 1%. `basal-calibrate` fits the
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

**Both wordings in one file.** If you serve the questions in Polish and in English, add `--lang both`:

```bash
python examples/triage/run.py --rules --lang both --calibration-input labelled.jsonl   # 2 requests per ticket
```

This writes every ticket twice, once with the Polish and once with the English questions:

- both copies share one `group`, so they land on the same side of the FIT/CERT split and the CERT numbers stay
  honest;
- each wording gets its own `family`, so the report gives each language its own calibration error
  (`LOFO ECE`).

**Step 3. Read the report.**

| line | what to check |
|---|---|
| `accuracy` per type | how often the answer is right, before any threshold |
| `ECE on CERT shipped -> fitted` | the calibration error. Lower is better; the fit should lower it |
| `automated on CERT`, `max_error 0.05` | **the main number**: the share basal routes automatically, and its error, on tickets the fit did not see |
| `0.05 per type: choice 0.93 (35%)` | the threshold of each question type; `none (defers)` means that type always goes to a person |
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

`basal-run --preset` needs basal 2.5.3 or later (in 2.5.2 it fails on every line with `KeyError: 'q'`). It sends the preset without `RULES`. For your own rules, write request lines (`run.py
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

> **Versions.** Every number in this section was measured on 3 November 2026 with basal 2.5.2 (the GB10 runs on
> the same day). Later versions fix bugs, and a fix can move some answers (each such change is listed in
> [WHATS_NEW.md](../../docs/WHATS_NEW.md)); the contact data of four tickets (e-mail addresses, a phone number) was
> replaced afterwards. Read the tables as what this setup gave then, not as a promise: `run.py` measures your basal
> version on your own tickets.

Measured on an Apple M1 Max with basal 2.5.2 and basal-2.0-4.5B-MLX-8bit, `--lang pl`.

- **The tickets:** 300 in all. Tickets 1-100 are short and clear. Tickets 101-300 come from four kinds of business:
  an online shop, B2B software, a telecom, and mixed messy e-mails (forwarded threads, capitals, no diacritics, two
  issues in one message).
- **The run:** about 7.5 minutes over the 300 tickets, about 1.5 s a ticket, with 8 requests in flight.
- **Calibration:** about 12 minutes with `basal-calibrate`.

### Accuracy

The `--rules` rows here and in the automation tables up to "Rules v2" use the first rules (v1).

| tickets | questions | category | urgency | needs_human |
|---|---|---|---|---|
| 1-300 | preset | 68% | 65% | 77% |
| 1-300 | `--rules` | 87% | 85% | 79% |
| 1-100 | preset | 70% | 66% | 80% |
| 1-100 | `--rules` | 91% | 80% | 81% |

The mini model (basal-2.0-mini-MLX-8bit), on tickets 1-40 before `--lang pl` was added, scored:

| questions | category | urgency | needs_human |
|---|---|---|---|
| preset | 82% | 55% | 52% |
| `--rules` | 95% | 55% | 72% |

### Automation at 5% error (`--rules --max-error 0.05`)

| calibration | category | urgency | needs_human |
|---|---|---|---|
| shipped (`CALIBRATION.json`) | 1% automated | no threshold | no threshold |
| fitted on 300 tickets, CERT split (88 tickets the fit did not see) | **35% automated, 0 errors in 31** (threshold 0.934) | **42% automated, 1 error in 37** (threshold 0.907) | no threshold |
| the same file, served, all 300 tickets | 38%, 0 errors in 114 | 45%, 1% error in 135 | 0% |
| fitted on tickets 1-100 only, CERT split | 73%, 1 error in 22 (threshold 0.929) | no threshold | no threshold |

What the numbers say:

- **With your own calibration, basal routes about a third of the tickets to the right team with no errors on held-out
  tickets.** It also sets the urgency of four in ten. The shipped calibration automates nothing on this helpdesk.
- **The CERT rows are the honest ones.** The "all 300" row includes the tickets the thresholds were fitted on.
- **More data, a smaller but truer share.** On the 100 easy tickets the CERT share was 73%. On 300, with the messy
  ones, it is 35%.
- **The thresholds are fitted, but not yet confirmed at 5%.** `basal-calibrate` warns that CERT has 31 and 37
  accepted decisions. Confirming 5% with 0 errors needs 59. That is about 500 labelled tickets in all.
- **`needs_human` gets no threshold at any data size here.** Who has to decide is a policy question, and the model
  is not confident enough about it. Keep it with people, or split it into concrete yes/no questions, for example
  "does the customer ask for money back?".
- **The fit lowers the calibration error (ECE on CERT)** most for urgency:

  | question | shipped | fitted |
  |---|---|---|
  | urgency | 0.184 | 0.027 |
  | category | 0.089 | 0.046 |
  | needs_human | 0.086 | 0.055 |

- **Most category mistakes are tickets sent to `other`.** In the served run, 12 complaints and 11 technical tickets
  went to `other`. A sharper description of `other` in `RULES` is the next thing to try. After changing it,
  calibrate again.

### NVIDIA GB10, `--mode fast`, Polish and English questions in one calibration

basal-2.0-4.5B in bf16 with `--rules --max-error 0.05`, on the same 300 Polish tickets. The calibration was fitted
with `--lang both`: 600 requests, about 9 minutes.

Calibration report, CERT split (176 requests the fit did not see):

| question | accuracy | automated | errors when automated | ECE shipped -> fitted |
|---|---|---|---|---|
| category | 88% | 37.5% (threshold 0.938) | 1 in 66 | 0.093 -> 0.042 |
| urgency | 83% | 52.8% (threshold 0.863) | **5 in 93 (5.4%)** | 0.169 -> 0.019 |
| needs_human | 80% | no threshold | - | 0.081 -> 0.062 |

The calibration error per language (`LOFO ECE`) is 0.059-0.065 for the category and about 0.045 for the urgency.
One file serves both wordings.

The served runs, all 300 tickets:

| calibration | `--lang` | category: accuracy, automated, errors | urgency | needs_human |
|---|---|---|---|---|
| shipped | pl | 87%, 24%, 0 in 73 | 84%, no threshold | 79%, 9%, 0 in 27 |
| shipped | en | 89%, 17%, 0 in 51 | 81%, no threshold | 80%, 8%, 1 in 23 |
| fitted (both) | pl | 87%, 40%, 2% of 121 | 84%, 54%, 4% of 163 | 79%, no threshold |
| fitted (both) | en | 89%, 41%, 2% of 122 | 81%, 57%, 3% of 171 | 80%, no threshold |

What changes on CUDA:

- **The shipped calibration of the bf16 model already automates some tickets.** It routes 17-24% of the categories,
  and 8-9% of the `needs_human` answers, with no or one error. On the Mac, the MLX port's shipped file automates
  about 1%. Each engine and port has its own calibration file, so measure on the one you serve.
- **The fitted file doubles the category share to about 40%,** in both languages.
- **The fitted urgency threshold does not hold.** `basal-calibrate` warns that on CERT it accepts 93 answers with 5
  errors (5.4%), above the 5% target: "the threshold does not hold on CERT; label more data before relying on it".
  Do not route on urgency with this file.
- **The fitted file drops the `needs_human` threshold the shipped file had.** On these tickets no confidence level
  met the bound. For that question, keep the shipped file's behaviour or send it to people.
- **The English wording is as good as the Polish one** on Polish tickets: 89% against 87% for the category. Use the
  wording your team reads. One calibration with `--lang both` covers both.

### Rules v2 and a held-out test: what calibration does not promise

**The change.** The first `RULES` (v1) left about 30 of their 38 category mistakes in `other`. The v2 `RULES` in
`run.py` say more about what each team covers: internet, TV, routers and setup questions under technical; order
status, returns and cancelled services under complaint; plans, instalments and payment methods under billing; GDPR
requests and owner changes under account. They also narrow `other` to "only when none of the above fits".

**The test set.** `holdout.jsonl` holds 97 tickets from four new kinds of business: a bank, a clinic, an energy
supplier and a courier. They were written after v2, and neither the rules nor any calibration was fitted on them.
Three near copies of the 300 tickets were removed. They were labelled with the same policy, and the policy's
wording of `other` is close to v2's, which may favour v2 a little.

Category accuracy, `--lang pl`, Mac (basal-2.0-4.5B-MLX-8bit):

| rules | the 300 tickets | the 97 held-out tickets |
|---|---|---|
| v1 | 87% | 72% |
| v2 | 95% | **85%** |

The 95% is optimistic, because v2 was written from the mistakes on those tickets. The held-out gain, 72% to 85%, is
the real one.

**Automation on the held-out tickets.** The calibration was fitted again with the v2 rules on the 300 tickets
(`--lang both`) on a GB10 (`--mode fast`, bf16), with `--max-error 0.05`:

| calibration | `--lang` | category: automated, errors | urgency: automated, errors |
|---|---|---|---|
| shipped | pl | 18%, 0 in 17 | no threshold |
| shipped | en | 23%, 0 in 22 | no threshold |
| fitted on the 300 tickets | pl | 89%, **9 in 86 (10%)** | 46%, 1 in 45 |
| fitted on the 300 tickets | en | 92%, **11 in 89 (12%)** | 51%, 1 in 49 |

**The fitted category threshold fails on new kinds of business.** On its own CERT split it accepted 94% with 3%
error, so it looked safe there. On the held-out tickets the model is sure and wrong:

- a smell of gas (0.992) and a power cut with a patient on an oxygen concentrator (0.983) go to `other`, because the
  rules describe internet and routers, not gas or electricity;
- a wrong dose on a prescription goes to `other` (0.995);
- price questions such as "how much is an ultrasound" go to `other`.

The urgency threshold holds: 1 error in 45 and 1 in 49. The shipped file stays cautious, with 0 errors.

What this means for your helpdesk:

- **A calibration certifies the error on traffic like the one it was fitted on.** New products, a new market or a
  new kind of message are new traffic. Calibrate on it.
- **Write the rules on the FIT part only, or calibrate on fresh labelled tickets after you change them.** Here the
  rules were written from all 300 tickets, so the CERT split was not independent of them.
- **Keep a held-out set**, labelled tickets that neither the rules nor the calibration saw. Measure on it before
  you turn on `max_error`, and audit a sample of the automated tickets every week.
- **Do not let the category router decide safety.** A gas smell or a medical emergency needs its own yes/no
  question with its own route, checked before the router.

### Safety first: the danger question

The held-out test showed the category router sending a gas smell and a power cut with a patient on oxygen to `other`,
with confidence above 0.98. A router should not decide safety, so `--rules` also asks a yes/no question:

- **`danger`**: "Is there a risk to people's health, life or safety in the ticket?" The options list the risks:
  smoke, fire, gas, poisoning, injury, a medicine problem, someone without power they need, and also a risk mentioned
  in passing.
- **It is a filter, not an automated decision.** From p >= `SAFETY_AT` = 0.2, `app.py` sends the ticket to the
  `alarm` queue with priority `urgent`, whatever the category says. The threshold sits far below 0.5 because a
  missed risk costs far more than a false alarm.
- **It carries no `max_error`, and `--calibration-input` leaves it out,** so no calibration can raise its threshold
  or turn it off.

Results on basal-2.0-4.5B-MLX-8bit, `--lang pl`:

| set | risky tickets caught | false alarms |
|---|---|---|
| `safety.jsonl`: 40 new tickets from ten businesses, 12 risky (4 of them with the risk mentioned in passing), 28 tricky but safe ("gaz" bills, urgent outages, data leaks) | **12 of 12** | 2 of 28 |
| `holdout.jsonl`: 97 tickets, 4 risky (gas, oxygen concentrator, wrong dose, a child's medicine) | **4 of 4** | 3 of 93 |

How `safety.jsonl` was kept blind:

- an agent wrote it after the question;
- it was not read before the measurement;
- the question was not changed after it.

The 4 risks in `holdout.jsonl` were seen before the question was written, so that row is less independent.

How much margin the threshold has:

- **On `safety.jsonl` the least confident risky ticket scored 0.58.** Every threshold from 0.1 to 0.5 catches all
  12; the false alarms fall from 6 (at 0.1) to 1 (at 0.5).
- **The false alarms are near misses**, all but one just above the threshold: carrying power banks on a plane
  (0.80), a fridge that stopped and spoiled the food (0.28), a parcel with a cracked coffee machine (0.23).

The gas ticket through the webhook:

```json
{"queue": "alarm", "priority": "urgent", "category": "other", "urgency": "wysoka", "danger": 0.867, ...}
```

**Twelve and four risky tickets prove little about rare events.** Before you rely on this filter, collect every
real risky ticket you have, measure the recall on them, and keep a person on the `alarm` queue.

## Limits

- **Pass `--lang pl` for Polish tickets.** Without it, the wording follows each ticket (`presets.lang_of`). Since
  basal 2.5.3 it reads short Polish text without diacritics as Polish, but a line with no common Polish word still
  gets the English wording: 4 of the 437 tickets here, e.g. h033, "Ile kosztuje USG jamy brzusznej bez skierowania?".
  In 2.5.2 it was wrong much more often (t33, "Aplikacja na iPhone wylogowuje mnie co kilka minut.").
- **The tickets are written for this example.** Real tickets are longer and messier. Measure on your own.
- **`needs_human` depends on your policy.** The descriptions in `RULES` encode one helpdesk's policy. Write yours.
