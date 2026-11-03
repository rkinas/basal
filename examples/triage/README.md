# Helpdesk triage

Each ticket gets three answers from the `triage` preset (`basal/presets.py`) in one request:

- `category`: a choice of complaint, billing, technical, account or other;
- `urgency`: a score of low, medium or high;
- `needs_human`: yes or no.

`run.py` scores the answers against the gold labels in `tickets.jsonl`, which holds 40 Polish tickets.
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

Measured on an Apple M1 Max with basal 2.5.2. A run of the 40 tickets takes about 15 s with the mini model.

| model | questions | category | urgency | needs_human |
|---|---|---|---|---|
| basal-2.0-mini-MLX-8bit | preset | 82% | 55% | 52% |
| basal-2.0-mini-MLX-8bit | `--rules` | 95% | 55% | 72% |
| basal-2.0-4.5B-MLX-8bit | preset | 78% | 57% | 75% |
| basal-2.0-4.5B-MLX-8bit | `--rules` | 98% | 70% | 80% |

- **Write your rules into the options.** The preset describes the account option as "konto lub dane klienta" (the
  account or the customer's data). With that wording the model sends a change of address, a password question or a
  new user to `other`. `RULES` in `run.py` says what each option covers at this helpdesk. That raises category
  accuracy from 78% to 98% on the 4.5B model, and `needs_human` from 52% to 72% on the mini model. A preset is a
  starting point.
- **`--max-error 0.05` automates nothing here, and that is the point of calibration.** The shipped calibration
  certifies only `choice` at 5% error, with a threshold of 0.977. No ticket reaches it, and `score` and `noul` have no
  threshold, so every answer abstains. The thresholds are fitted on basal's own data. To automate a share of your
  tickets at a known error, fit them on your labelled tickets:

  ```bash
  python examples/triage/run.py --rules --calibration-input labelled.jsonl
  basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx     # writes CALIBRATION.custom.json
  basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --calibration CALIBRATION.custom.json
  ```

  On these 40 tickets, `basal-calibrate --report-only` reports what is missing. A 5% target needs at least 59
  accepted FIT decisions per question type, and 299 for 1%. The FIT/CERT split is 70/30, so that is about 85 or 430
  labelled tickets.

- **The preset's language follows the state.** With no `--lang`, the wording follows each ticket.
  `presets.lang_of` reads t33, "Aplikacja na iPhone wylogowuje mnie co kilka minut.", as English: it has no
  diacritics and one common Polish word. So t33 gets the English questions. `--lang pl` fixes the wording. On the
  4.5B model with `--rules` the results are then 95%, 70% and 80%.

Replace `tickets.jsonl` with your own tickets, in the same shape:
`{"id", "state", "gold": {"category", "urgency", "needs_human"}}`.
