# Helpdesk triage

Each ticket gets three answers from the `triage` preset (`basal/presets.py`) in one request:

- `category`: a choice of complaint, billing, technical, account or other;
- `urgency`: a score of low, medium or high;
- `needs_human`: yes or no.

`run.py` scores the answers against the gold labels in `tickets.jsonl`, which holds 100 Polish tickets.
`--max-error` adds the certified error target: an answer whose confidence is below the threshold abstains, and the ticket
goes to the human queue.

```bash
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000   # any model and mode; a Mac here
python examples/triage/run.py                       # the preset as it is
python examples/triage/run.py --rules               # the preset with this helpdesk's rules in the options
python examples/triage/run.py --rules --max-error 0.05
python examples/triage/run.py --rules --calibration-input labelled.jsonl   # the input of basal-calibrate
```

## Results

Measured on an Apple M1 Max with basal 2.5.2. A run of the 100 tickets takes about 40 s on the 4.5B model.

### Accuracy

The 4.5B rows were run with `--lang pl`. The mini rows were run before it was added, so their wording followed each
ticket's language.

| model | tickets | questions | category | urgency | needs_human |
|---|---|---|---|---|---|
| basal-2.0-mini-MLX-8bit | 1-40 | preset | 82% | 55% | 52% |
| basal-2.0-mini-MLX-8bit | 1-40 | `--rules` | 95% | 55% | 72% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | preset | 70% | 66% | 80% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | `--rules` | 91% | 80% | 81% |

**Write your rules into the options.** The preset describes the account option as "konto lub dane klienta" (the
account or the customer's data). With that wording the model sends a change of address, a password question or a new
user to `other`. `RULES` in `run.py` says what each option covers at this helpdesk, which raises category accuracy
from 70% to 91%. A preset is a starting point.

### Automation at 5% error (`--max-error 0.05`), basal-2.0-4.5B-MLX-8bit

| calibration | category automated | error when automated | urgency, needs_human |
|---|---|---|---|
| shipped (`CALIBRATION.json`) | 0% (threshold 0.977) | - | no threshold: every answer abstains |
| fitted on the 100 tickets | 83% (threshold 0.929) | 1 of 83 | no threshold: every answer abstains |
| the same, on its CERT split only | 73% | 1 of 22 | - |

The shipped thresholds are fitted on basal's own data. Fitted on these tickets:

```bash
python examples/triage/run.py --rules --lang pl --calibration-input labelled.jsonl
basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --out CALIBRATION.triage.json  # ~4 min
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --calibration CALIBRATION.triage.json
python examples/triage/run.py --rules --lang pl --max-error 0.05
```

How to read the three rows:

- **The 83% row is measured on the tickets the thresholds were fitted on**, so it is optimistic.
- **The honest number is the CERT split, which the fit did not see:** 73% of the category answers automated, with
  1 error in 22 accepted.
- **That threshold is not yet proven at 5%.** `basal-calibrate` certifies it on FIT, but says CERT has too few
  decisions to confirm it: 0 errors would need at least 59 accepted CERT decisions. That is about 200 more labelled
  tickets.
- **`urgency` (80%) and `needs_human` (81%) get no threshold at 5% at all.** Too few of their answers are both
  confident and right. Those questions stay with a person, or need more labelled tickets and sharper rules.

The fit also lowers the calibration error (ECE on CERT):

| question | shipped | fitted |
|---|---|---|
| category | 0.122 | 0.087 |
| urgency | 0.140 | 0.071 |
| needs_human | 0.126 | 0.121 |

**Pass `--lang pl` for Polish tickets.** Without it, the wording follows each ticket, and `presets.lang_of` reads
t33, "Aplikacja na iPhone wylogowuje mnie co kilka minut.", as English: it has no diacritics and only one common
Polish word.

Replace `tickets.jsonl` with your own tickets, in the same shape:
`{"id", "state", "gold": {"category", "urgency", "needs_human"}}`.
