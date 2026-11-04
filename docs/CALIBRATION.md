# Calibration on your own data

Every model ships a `CALIBRATION.json`: one temperature per question type and the confidence thresholds certified
for a target error of 1% and 5% among accepted decisions (`act` with `"max_error"`). Both are fitted on our data. On
your own domain the model can be as accurate or more accurate and still be confident in a different way, so a
threshold certified on our data can accept too few (or too many) of your decisions. `basal-calibrate` fits both on
your labelled requests, by the same procedure, and writes a file for `basal-serve --calibration`. The file also
reports the calibration gates on your data: the leave-one-family-out ECE and the share of decisions automated at each
certified threshold. From labelled `multi` questions with a fallback label, it fits the `fallback_threshold` of each
label set.

## Usage

```bash
# the engine in process (the same options as basal-serve: --model, --mode, --quant, --gguf, ...)
basal-calibrate labelled.jsonl --model Remek/basal-1.5-4.5B --mode fast     # writes CALIBRATION.custom.json

# or a running server started with --no-calibration (--model gives the shipped file to compare with)
basal-serve --model Remek/basal-1.5-4.5B --mode vllm --no-calibration --port 8000
basal-calibrate labelled.jsonl --url http://127.0.0.1:8000 --model Remek/basal-1.5-4.5B

basal-serve --model Remek/basal-1.5-4.5B --mode fast --calibration CALIBRATION.custom.json
```

Fit with the engine, port, quantisation and option orders you serve: the probabilities, and therefore the
thresholds, belong to them. `basal-calibrate` uses basal-serve's default orders (two for prompt-format-1 models, one
for prompt-format-2 models; `--orders` changes it, and with `--url` the server's orders are read from
`GET /v1/models`) and records them as `fitted_on.orders`. `basal-serve` applies a file only to those orders: serving
other orders, it loads `CALIBRATION.o<N>.json` (or `CALIBRATION.<engine>.o<N>.json`) when the model ships one, and
otherwise keeps the file's temperatures without its thresholds (`max_error` defers), warns at start-up and marks every
answer `"calibration": "unvalidated"`. A format-2 model may ship an optional `CALIBRATION.o2.json` for `--orders 2`.
`--url` refuses a server that applies temperatures (it reports them in `GET /v1/models`). Other options:

- `--targets 0.01,0.05`: any `max_error` values.
- `--min-per-type 50`.
- `--min-blend 10`: a type (or option-count bucket) with fewer FIT decisions than `--min-per-type` but at least this
  many gets its fitted temperature blended with the shipped one (below). `0` keeps the shipped one, as before 2.2.
- `--report-only`: print the report on your labelled requests (accuracy, ECE, the thresholds, the share automated,
  and how the compared file does on them) without writing a file. Use it to evaluate a model, or a calibration file
  given with `--calibration`, on your own data.
- `--out`.
- `--calibration FILE`: the file to compare with. By default it is the one `basal-serve` would load.
- `--no-calibration`.
- `--reference [basal-1.5 | FILE]`: what the leave-one-family-out ECE is compared with. The flag alone means the
  basal-1.5 values; a FILE is a calibration file whose `gates.ece_lofo` holds the means.
- `--strict`: exit with status 1 when a type's leave-one-family-out ECE is above the reference. The file is still
  written. Use it in CI. A type with no family of `--min-family` CERT decisions is not tested and does not fail;
  2.3.1 says so on stderr ("0 families tested") and in the report.
- `--min-family 30`: the CERT decisions of a type that a family needs for its own ECE.
- `--min-per-set 30`: the FIT questions that a label set needs before its `fallback_threshold` is fitted.
- `--save-rows FILE`: also write the uncalibrated probabilities of every decision (decision questions only: task
  answers depend on the served temperatures, so their thresholds are fitted with the engine).
- `--from-rows`: the input is such a file, and the fit runs again without a model, for example after a change of
  `--targets`, `--min-family` or `--reference`.

## Input

One `/v1/systemone` request per line; every question carries its correct answer in `"gold"`:

```jsonc
{"id": "t-17", "group": "customer-17", "state": "...",
 "questions": {
   "dept":  {"type": "choice", "instructions": "...", "criteria": {"returns": "...", "it": "..."}, "gold": "returns"},
   "late":  {"type": "noul",   "instructions": "...", "gold": "yes"},
   "level": {"type": "score",  "instructions": "...", "criteria": ["low", "mid", "high"], "gold": 2}}}
```

| field | value |
|---|---|
| `gold` (choice) | an option key, or its description |
| `gold` (noul) | `"yes"` / `"no"` (or `true` / `false`) |
| `gold` (score) | the level index (0 = first level), its key, or its text; an integer is always the index, and is refused when a numeric key names another level (`{"1": ..., "2": ..., "3": ...}` with `"gold": 2`: give `"2"`) |
| `gold` (act) | an outcome, as for choice (`"yes"` / `"no"` for `true` / `false` outcomes); `act` counts as its base type |
| `gold` (multi) | the list of labels that apply: keys or descriptions, with `[]` for none (see [below](#multi-the-fallback-threshold-of-a-label-set)) |
| `group` | optional, on the line or on a question: decisions of one group (twins, duplicates, one customer) stay on the same side of the split; default: one group per line, since the questions of a request share its state (`null` counts as not given, as for `id`). The bound of a threshold counts every decision as independent: when a type or task has more FIT decisions than groups (the claims of one document, the fields of one form), errors shared within a group make it optimistic, and the report warns with the decisions per group |
| `family` | optional, on the line or on a question: the task family of the leave-one-family-out ECE (one kind of decision: a routing question, a document check); default: the question name |
| `id` | optional, used in messages |

A question's `ask_if` / `depends_on` is dropped: every labelled question is asked, whatever the answers of the others.
Rows whose probabilities are not finite numbers in [0, 1], or whose gold is not an option, are left out of the fit
and reported (also from `--from-rows` files).

Task questions (`classify`, `verify`, `extract`, see [TASKS.md](TASKS.md)) carry `"gold"` too, and certify the
`automated` flag of their answers ([below](#tasks-classify-verify-extract)):

| task | `gold` |
|---|---|
| `classify` (one label) | a label key; `"none"` when the question has `"none": true` |
| `verify` | `{claim key: verdict}`, or a list in the order of the claims; `supported` / `contradicted` / `not_stated` (`binary: true`: `supported` / `not_supported`, or `true` / `false`); `null` = not labelled |
| `extract` | the field values, nested as in the schema; `null` = not stated in the document; a field left out is not labelled |

Send the requests exactly as in production: the same `option_keys`, `facts` and language of the state. The simple
items of `basal-run` and `basal-bench` (`{"state", "question", "options", "type", "gold": <index>}`) are accepted
too. Every problem in the file is reported with its line before any model is loaded.

## Method

The shipped files were fitted with the same steps:

1. The decisions are split by a hash of their group into FIT (70%) and CERT (30%).
2. One temperature per type (`choice`, `noul`, `score`) minimises the negative log-likelihood of the served
   prediction (as the server answers: both option orders averaged, or one order) on FIT. A type with fewer than
   `--min-per-type` FIT decisions but at least `--min-blend` (10) gets the fitted temperature blended with the shipped
   one in log space, in proportion *n* / `--min-per-type` (20 FIT decisions of 50: 40% fitted, 60% shipped), so a few
   dozen labels already move it and the full fit takes over smoothly at `--min-per-type`. With fewer it keeps the
   shipped temperature. The table says which (`(blended)`, `(kept)`).
   For prompt-format-2 models (see [FORMATS.md](FORMATS.md)) `choice` also gets a temperature per option-count bucket
   (2–10, 11–26, 27–255 options), fitted the same way where FIT has at least `--min-per-type` decisions in the bucket;
   a bucket with fewer keeps the shipped bucket temperature, else the type temperature applies to it.
3. For each target error *r*, the threshold comes from a **fixed-sequence test** on FIT. The candidates are the
   observed confidences, never a fixed grid; equal confidences are accepted or refused together. They are tested from
   the most confident down, each with the one-sided 95% Clopper–Pearson upper bound of the decisions it accepts
   ≤ *r*, and the test stops at the first level that fails: the threshold is the last level that passed. The first
   level tested is set by the counts alone, before any outcome is read: the most confident level that accepts enough
   decisions to meet the bound with one error (93 at 5%, 473 at 1%) when the type has at least twice that many FIT
   decisions, else enough to meet it without an error (59 at 5%, 299 at 1%; at least 50). If it fails, the type has no
   threshold at that target, even when a lower level would pass on its own: scanning on past a failure until some level passes
   tests many times and certifies a type whose error is just above the target far more often than 5% of the time
   (about 20% with 1,000 decisions at 5.1% error and a 5% target); stopping at the first failure keeps it at most 5%.
   The cost is that errors among the very top decisions end the test there (one is allowed at the start when the
   type has enough decisions); label more data, or check the CERT report, when a type you expected to certify does
   not (the reason names the first level and its errors).
   Each certificate is one test of one question type at one target error: its 95% holds for that type and target,
   which is what a request's `max_error` on a question of that type relies on. It is not a joint statement over all
   types and targets at once (with three types and two targets, the chance that at least one of the six is wrong can
   be up to six times 5%).
   A looser target never keeps a stricter threshold than a tighter one: a confidence certified at 1% is certified at
   5% too, so where the 5% test stops above the 1% threshold (it starts on fewer, very confident decisions and can
   stop on an error just below them), the 5% threshold is the 1% one, marked `floored_from`.
   - It is certified per question type (`thresholds[r].per_type`), on the confidences as served. A temperature keeps
     the order of a type's yes/no confidences, but not of its choice and score confidences: with three or more
     options it can swap two decisions (logits [0, −3, −3] and [0, −2, −20]: 0.909 above 0.881 at T = 1, 0.403
     below 0.512 at T = 10), and the option-count buckets of format 2 have their own temperatures. The temperatures
     are fitted on the same FIT decisions, so for choice and score the order of the levels, and therefore the
     certified coverage, can depend on them.
   - One threshold pooled over all types (`thresholds[r].confidence`, as in earlier files) is still written, marked
     `"pooled": "informational: ..."`, and never applied. It moves with the temperatures (a yes/no temperature of 1.7
     pushes yes/no confidences below the choice ones, and the pooled threshold then refuses yes/no decisions that are
     certifiable on their own), and it can let a type that is not certifiable on its own run above the target.
   - `basal-serve` applies only a type's own threshold. A type without one (too few decisions, or none meeting the
     bound) has every automatic action with that `max_error` deferred: `"deferred": true` and
     `"max_error": "no certified threshold for this type at this max_error"`. Only a format-1 file without
     `per_type` (the basal-1.5 files) keeps its pooled threshold, the only certificate it has. That certificate holds
     for the mix of question types it was certified on, not for each type; callers who need a per-type guarantee
     should use a format-2 model.
4. CERT, not used for fitting, reports coverage, error and the same bound at that threshold, next to the shipped
   threshold on the same decisions.
5. The gates. They are measured, not fitted.
   - `ece_lofo`, per type: for every family, the temperatures are fitted again on the FIT decisions of the other
     families and applied to that family's CERT decisions. The ECE is then reported per family, as the mean and the
     max (with the worst family) over the families with at least `--min-family` CERT decisions of the type, and
     pooled over all of them. `mean_floor` is the mean that a perfectly calibrated predictor with the same confidences
     would show. Small families make the per-family ECE noisy, so read the mean next to its floor.
   - `--reference` adds the reference and `regression` (mean above the reference). The basal-1.5 values are 0.082
     (choice), 0.062 (noul) and 0.109 (score), the same statistic on our calibration split.
   - `automated_at`: the share of CERT decisions that `basal-serve` would automate at each target, each type at its
     own threshold and a type without one not at all. `automated_error` is their error on CERT, and
     `automated_at_pooled` is the share the single pooled threshold would automate, for comparison only. A target that
     is certified for no type gets 0.

The thresholds hold for the mix of question types and prompts you calibrated on. The leave-one-family-out ECE shows
how well one temperature carries over to a kind of decision it was not fitted on, such as a new question in
production.

## Answer sets (coverage)

`--conformal 0.8,0.9,0.95` (off by default) also fits answer sets, by split conformal prediction:

1. For each type and coverage level *L*, the CERT decisions (not used for the temperatures) are scored with the
   probabilities as served: the score of a decision is 1 − p(gold).
2. `p_min` is the *k*-th largest p(gold) of the type's *n* CERT decisions, *k* = ⌈(*n* + 1) *L*⌉. The answer set of a
   question is every option with p ≥ `p_min`, the most probable first and always in.
3. For a new request like the labelled ones, the set then holds the correct answer with probability ≥ *L*. This holds
   on average over requests, not for each one, and only for decisions like those you calibrated on.
4. With fewer than *L* / (1 − *L*) CERT decisions of a type (9 at 0.9, 19 at 0.95) there is no `p_min`, and the set
   is every option.

The FIT decisions, held out from `p_min`, report the coverage, mean set size and singleton share of the sets (the
`answer sets` table, and `conformal.levels[L].per_type[type]` in the file). A question asks for its set with
`"coverage": L`, a level the file has. A file fitted on other option orders or another layout keeps no levels, as it
keeps no thresholds.

```jsonc
"conformal": {"method": "split conformal on CERT ...",
              "levels": {"0.9": {"per_type": {"choice": {"p_min": 0.142, "n": 512, "cert_coverage": 0.902,
                                                         "fit_coverage": 0.897, "fit_mean_size": 1.61, ...}}}}}
```

A question with `"max_error": r` on a `choice`, `noul` or `score` question gets `"abstain": true` when its
confidence is below the type's certified threshold for *r* (step 3 of the method), or when the type has none.

## `multi`: the fallback threshold of a label set

A `multi` question with a `fallback` label (see [FEATURES.md](FEATURES.md#label-constraints-exclusive-and-fallback))
often needs its own `fallback_threshold`: a model can under-select a "general" label, so the default rule (p ≥
`threshold`) leaves those questions empty. Label such questions with the labels that apply:

```jsonc
{"state": "...", "questions": {"tags": {"type": "multi", "instructions": "...", "fallback": "general",
   "criteria": {"damage": "...", "refund": "...", "general": "..."}, "gold": ["general"]}}}   // [] = none applies
```

Send them with the objective, `threshold`, `min`, `max` and `exclusive` you use in production. For every label set (its
keys, descriptions and fallback), `basal-calibrate` takes the label probabilities as they would be served with the new
yes/no temperature. It then picks the `fallback_threshold` with the best mean IoU on FIT from a log grid of
0.001–0.95. The default rule is a candidate too, and it wins ties. CERT reports IoU, empty-set accuracy (no label
applies and none is selected) and the false-empty rate (labels apply and none is selected), next to the default:

```
multi fallback     label set             n  threshold   IoU default -> fitted empty acc.       false-empty
general            01452b992bae2e25    750     0.2647   0.587 -> 0.871        100.0% -> 64.3%  50.8% -> 7.7%
```

A lower fallback threshold turns empty answers into the fallback, so expect the IoU to rise and the empty-set accuracy
to fall. The fit is written to the `multi` section of the file under the label set's id (`basal.sets.label_set_id`, a
hash of the sorted keys with their descriptions and the fallback). `basal-serve` uses it for a request with the same
label set and fallback and no `fallback_threshold` of its own. An explicit value in the request always wins, and
`constraints_applied.fallback_threshold_source` says which one was used (`request`, `calibration` or `default`).
A label set with fewer than `--min-per-set` FIT questions is reported, not written. `multi` questions do not enter
the temperatures: to calibrate the labels themselves, ask each one as a `noul` question with its gold.

## Example: a new domain, measured

basal-omni-1.0-mini (`--mode eager`, bf16, one option order) on
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0: synthetic
decisions over JSON records in four workflows, gold labels from a teacher model), a domain the model was not trained or
calibrated on. Calibrated on the train split (3,958 labelled decisions in 800 requests, plus 596 `classify` and 1,600
`verify` task decisions), measured on the CERT part of the test split (172 choice, 184 yes/no and 237 score decisions),
engine 2.2, 16 Oct 2026:

| on the test split | shipped `CALIBRATION.json` | fitted on the train split |
|---|---|---|
| ECE, choice / yes-no / score | 0.268 / 0.106 / 0.104 | **0.083 / 0.073 / 0.052** |
| `max_error: 0.01`: automated, and its error | 12.6%, **8.0% wrong** | 0% (nothing certifiable) |
| `max_error: 0.05`: automated, and its error | 54.3%, **21.4% wrong** | 3.7%, 0.0% wrong |
| `verify` claims at `max_error: 0.01` | no task threshold | 54.8% automated, 0 wrong (bound 1.1%) |

The shipped file's certificate holds on our calibration data, not on this domain: here its 1% setting would automate one
decision in eight at 8% error. Fitted on the user's labels, the temperatures rise (choice 1.14 → 2.52, yes / no 1.21 →
2.03, score 1.41 → 1.78: the model is overconfident on these records), the calibration error falls by type (choice 0.268
→ 0.083, about 3.2 times; score 0.104 → 0.052, half; yes / no 0.106 → 0.073, about 30% lower), and the thresholds
certify only what the model gets right here (accuracy 0.57 / 0.80 / 0.67, against noisy teacher labels): almost nothing
is automated, which is the right answer for a model this inaccurate on this data. On `verify`, where it is 95% accurate,
the 1% threshold fitted on the train split automates over half of the test claims (54.8%) with none wrong; the one-sided
95% bound on its error is 1.1%. A hosted decision API offers no such step: its confidence is what it is on your data.

## Tasks: classify, verify, extract

A task answer reports `automated[max_error]` only with a threshold certified for that task (`task_thresholds` in the
calibration file): the thresholds of the decision types do not hold on new label sets. `basal-calibrate` fits them
from labelled task questions, one threshold per task and target, by the same bound as the decision types (step 3
above): every `classify` answer, every `verify` claim and every `extract` field is one decision, split into FIT and
CERT by group. The table:

```
task          n   FIT  CERT  accuracy   max_error  threshold  automated   error   bound   (automated share, error and bound on CERT)
classify    ...
```

- The thresholds are certified on the task answers **as served with the new file's temperatures**, so task questions
  need the engine in process (`--model`, `--mode`, ...; not `--url`). The task requests run after the decision fit,
  with its temperatures.
- A file with **task questions only** keeps the decision calibration of the file it compares with (temperatures,
  thresholds, `multi`) and adds the task thresholds. It also keeps that file's thresholds of other tasks, which hold
  because the temperatures are unchanged; the report then measures that file's task thresholds on your CERT decisions,
  next to the fitted ones.
- A file that refits the temperatures (decision questions in the input) leaves out the compared file's thresholds of
  tasks without labelled questions, and warns: they were certified with other temperatures.
- Not certified: `compress` (no `automated` flag), multi-label `classify` (it reports `set_confidence`), arrays of enum
  values in `extract`. An `extract` number that does not parse reports `automated: false`.

## Which model a file belongs to

The probabilities, and so the temperatures and thresholds, belong to one model. `fitted_on` records the model as given
to `basal-calibrate`, except that a Hugging Face cache snapshot path is recorded as its repository ID in
`fitted_on.model` and its commit in `fitted_on.revision`. A repository ID likewise records its resolved commit;
plain local directories keep their path without a revision. `basal-serve --calibration FILE` compares
them with the served model: on another model, or another revision, it warns at start-up, reports
`"calibration_warning"` in `GET /v1/models` and keeps the file for its temperatures only, as a file of other option
orders: its thresholds, task thresholds, conformal levels and fitted fallback thresholds are not applied (`max_error`
defers, tasks report `automated: null`, answer sets hold every option) and every answer is marked
`"calibration": "unvalidated"`; with `--strict-calibration` it refuses to start. When both are Hugging Face snapshots
(a revision on each side) the whole repository ID is compared, so `OrgA/basal-x` is not `OrgB/basal-x`; a local copy
of a model matches its Hugging Face name by the last part of the path (`/data/basal-omni-1.0-max` =
`Remek/basal-omni-1.0-max`).
The files that ship in a model's directory are not checked.

With `--adapter`, `basal-calibrate` records the adapter as `fitted_on.adapter` (`<name>@<sha256 of its
adapter_config.json and weights>[:12]`, so a rescaled adapter is another one)
and basal-serve compares it like the model: a file fitted with another adapter, or with one when none is served (or
the reverse), is kept for its temperatures only in the same way or, with `--strict-calibration`, refused. Without `--calibration`, a `CALIBRATION[.<engine>].json`
in the adapter directory is loaded first (and checked); the model's own file is the base model's, so it is loaded for
its temperatures only, without thresholds, and the answers are marked `"calibration": "unvalidated"`.

## How many items

For the **temperatures**, a few dozen labelled decisions of a type already help (`--min-blend`, above); 50 FIT
decisions, about 72 labelled ones, fit it fully. For the **thresholds**, a few hundred labelled decisions usually certify the 5% target. The 1% target needs the 299 most confident FIT
decisions of a type without an error (the first level the fixed-sequence test checks; with at least 946 FIT decisions
of the type it checks the top 473 and allows one error there; further down, 628 accepted with two errors pass too), so
in practice about 1,000 labelled decisions of a type or more. The thresholds
are certified per question type, so these numbers hold for each type on its own: 299 accepted `noul` decisions without
an error, and again 299 `choice` decisions, not 299 in total. An error made with very high confidence, among the
first level tested, leaves its type without a threshold at that target. On one external set
of 1,254 Polish business decisions (FIT: 462 `noul`, 371 `choice`) no type of five basal-2.0 and basal-omni models
was certified at 1%: the lowest bound of any type was 1.2%, and for the basal-omni models a few errors at confidence
0.997–0.999 set that floor. With fewer, use the refitted temperatures and the 5% threshold, or route uncertain cases
to a person. When no confidence meets a bound, the type's entry has `"confidence": null` and a `reason` for that target; `act` and `tool` questions of that type
with that `max_error` then defer every decision (the defer action, the `refused` one named, `"deferred": true` and
`"max_error": "no certified threshold for this type at this max_error"`), and `basal-calibrate` warns about it. With fewer than 200
decisions it warns that the bounds are loose.

`basal-calibrate` checks this before it runs the model (2.3.1). For each question type and task and each target, it
counts the FIT decisions in your file. If they could not meet the bound even if all of them were accepted and correct,
it says so on stderr and again at the end of the run, for example "verify: max_error 0.01 needs >= 299 accepted FIT
decisions without an error (...); this file gives 192 FIT decisions of 275; label about 153 more verify claims".

A threshold certified on FIT is written even when its CERT decisions are too few to confirm it. 2.3.1 then says so,
in the report and on stderr: "noul at max_error 0.01: certified on FIT; CERT has 150 accepted, 1 error, upper bound
3.12% > 1%, too few to confirm on CERT (0 errors would need >= 299 accepted CERT decisions)". If the error observed on
CERT is above the target, it says that the threshold does not hold on CERT. The threshold and the exit status do not
change (`--strict` checks only the leave-one-family-out ECE).

## Output

`CALIBRATION.custom.json` has the shipped schema (`temperature_per_prim`, `thresholds[<target>].per_type`, and the
informational pooled `thresholds[<target>].confidence`) plus
`temperature_source`, the FIT and CERT figures of each threshold, `engine` (the engine it was fitted on: `fast`,
`fp8`, `vllm`, ...; serve it with that engine, or ship it as `CALIBRATION.<engine>.json`) and `fitted_on` (file,
number of decisions and requests, FIT / CERT sizes, date, model and its revision, engine; with task questions the
task decisions per task). It also has `gates` (`ece_lofo`, `automated_at`, `ece_reference`), from labelled `multi`
questions `multi`, and from task questions `task_thresholds` (what basal-serve applies) and `task_gates` (their FIT
and CERT figures). A file fitted on task questions only says where its decision calibration comes from in
`decision_calibration`. For a prompt-format-2 model it is
calibration file 2: `"prompt": 2` and `temperature_per_bucket` (the table then has one row per bucket under `choice`);
basal-serve refuses a file fitted for another prompt format than the model's. The table printed at the end:

```
type        n   FIT  CERT  accuracy   temperature shipped -> fitted   ECE on CERT shipped -> fitted
choice    ...

max_error on CERT   shipped: threshold  coverage   error   fitted: threshold  coverage   error   bound
0.01      ...
0.05      ...
(one threshold pooled over the types, informational: basal-serve applies each type's own, as in "automated on CERT" below)

LOFO ECE on CERT    families    mean   floor   max (family)                pooled  reference
choice    ...

automated on CERT   per-type thresholds   error   pooled (not applied)   compared file, as served
max_error 0.01      ...
max_error 0.05      ...

task          n   FIT  CERT  accuracy   max_error  threshold  automated   error   bound
classify  ...
```

"compared file, as served" is the share of your CERT decisions the compared file (the shipped one by default, or
`--calibration FILE`) automates at each target and their error, each type at the threshold basal-serve applies to it:
how the file you would otherwise serve does on your data.

*ECE* is the expected calibration error of the top answer (10 bins), with the temperatures as basal-serve applies them
(in calibration file 2, a `choice` question gets its option-count bucket temperature, so the `choice` row covers all
its buckets); *bound* is the one-sided 95% upper bound of the error on CERT.
The temperatures in the table are also the ones basal-serve applies: a bucket row shows the bucket's temperature, or
the `choice` type temperature marked `(type)` when the file has no such bucket, and when the buckets cover every option
count (2–255) the `choice` row marks its type temperature `(not served: buckets)`.
