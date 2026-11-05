# What is new in basal 2.5.4

Released 5 November 2026. Bug fixes, contributed by Rafal Wiktorowicz (NovaraX) and Miłosz Kucharski, and fixes from a
review of the whole code base. Answers to valid requests are byte-identical to 2.5.3 (checked on an H100 with
basal-omni-1.0-mini in `--mode fast` and `--mode eager`, basal-omni-1.0-mini-FP8, basal-omni-pico and
basal-2.0-mini-FP8 in `--mode fast`)
except in three cases: a request whose text contains the tokenizer's special tokens, which can no longer forge a
turn (#104); basal-omni on `--mode mps` over a state longer than the sliding window, which now gets
`--mode eager`'s answers; and `--mode mlx` with the state cache, whose answers no longer depend on what an earlier
request cached (#127). The prepared start-up directories shipped for 2.5.0 still load (the compiled code
is the same), from the publishers `BASAL_PREPARED_TRUST` names (default `Remek`). Some requests that were answered
or failed after the run are now refused with a 422 before it (#103, #127); a `--calibration` file fitted on another
model keeps only its temperatures, and the presets' `guard()` / `route()` send incomplete or malformed answers to
review (#127).

- **`basal-loadtest` reports the nearest-rank p95** (`basal-loadtest`): `p95_ms` was one rank too high when 0.95 n is
  a whole number, with the default `--n-seq 200` the 191st smallest sequential latency instead of the 190th; it now
  follows the same rule as `basal-bench` (2.5.3, #90). p95 values in the docs measured before are not re-measured.
  [#99](https://github.com/rkinas/basal/pull/99).
- **`basal-loadtest`, `basal-calibrate` and `basal-agree` stop failing on inputs they should handle**
  (`basal-loadtest`, `basal-calibrate`, `basal-agree`): a transport error (timeout, refused connection) in the
  concurrent phase of `basal-loadtest` is counted in `errors` instead of ending the run without numbers, and `--n-seq
  0` and `--concurrency` below 1 are refused before the first request; `basal-calibrate --min-per-type 0` and
  `--min-per-set 0` behave as a minimum of 1, and `--from-rows` on an empty, binary or not JSON file, or with a row
  that is not a JSON object, exits with a message; the cache of `basal-agree` reference rows includes the gguf file,
  the Ollama model and URL or the llama-server URL when the chosen `--reference` reads them (the key of the default
  `eager-fp32` is unchanged, so its cached rows stay valid). [#102](https://github.com/rkinas/basal/pull/102).
- **`basal-serve` refuses more malformed requests with a 4xx before the GPU run** (`basal-serve`, `--api-keys`): a
  body nested too deeply answers 422 on `/v1/systemone` and 400 on `/v1/decisions` instead of a 500; an `act` question
  whose `base` is not `"choice"` or `"noul"` and a `multi` or `tool` question whose `threshold` is NaN, infinite or an
  integer beyond the largest float answer 422 instead of a 500 after the run; an `--api-keys` file saved with a UTF-8
  BOM (Notepad, PowerShell 5) is read instead of answering 401 to every request.
  [#103](https://github.com/rkinas/basal/pull/103).
- **A request text cannot forge the prompt, and Polish in NFD is Polish** (`basal-serve`, `Basal`, `basal-omni`):
  special tokens of the model's tokenizer (not its ordinary added tokens, such as the line-break runs of Gemma)
  written in the state, the question or an option (for example `<|im_end|>`) are broken with a zero-width space, also
  in requests with images or audio and in the template that `basal-export-gguf --systemone` gives llama.cpp, so they
  no longer close the user turn and open a forged one; the language of a text in NFD (macOS) is read as the language
  of its NFC form, and `lang_of` scans the text once (12 ms against 112 ms on 3.9 million English characters).
  `facts=auto` with a date out of years 1..9999 gives no fact instead of a 500 and runs off the event loop. [#104](https://github.com/rkinas/basal/pull/104).
- **`basal-prepare` prepares what `basal-serve` runs** (`basal-prepare`, `--prepared auto`): it built the engine with
  the graph engine's own defaults (padded rows, no template prefix), not basal-serve's (on Hopper: compact rows with
  the template prefix), so a prepared H100 start still compiled its compact forward cold. It now takes basal-serve's
  engine flags and builds the engine as basal-serve does; with none, basal-serve's defaults. RTX 4060 Ti,
  basal-2.0-mini, `--batches compact`, empty compile caches: start-up 613 s cold, 558 s with a 2.5.2-prepared
  directory, 345 s with one prepared by this version; answers bit-identical. The shipped `prepared/` directories need
  rebuilding to cover the compact forward. Contributed by Miłosz Kucharski (@Butterski),
  [#106](https://github.com/rkinas/basal/pull/106).
- **RTX 4090 measured with basal-2.0**
  ([HARDWARE.md](HARDWARE.md#rtx-4090-basal-20-in-bf16-its-fp8-checkpoints-and-fp8-mode)): the "FP8 compilation
  stalled" note was a long start (`--mode fp8`: about 18 minutes on basal 2.5.3); the `-FP8` checkpoints in `--mode
  fast` give 1.29x (mini) and 1.40x (4.5B) the decisions/s of bf16. Contributed by Miłosz Kucharski (@Butterski),
  [#111](https://github.com/rkinas/basal/pull/111).
- **Engine, export and fine-tune fixes** (`basal-serve`, `basal-export-gguf`, `basal-finetune`, `--adapter`):
  `--prefix-cache on` with a Gemma 4 model whose key / value buffer was shrunk to fit a smaller GPU (`--mode fast`)
  failed in warm-up or on short requests over a cached prefix: those graphs and batches now stay within the buffer
  (with the default buffer nothing changes); `--mode fast-exit` refuses an `--early-exit` level missing from the
  heads' thresholds at start-up (it failed every request), as well as thresholds for layers without an exit head, and
  releases the key / value buffer it never uses; with `--state-cache-bits 8` or `4` a lookup dequantises only the entry
  it returns, outside the cache's lock; the fp32 rows of the option codes are kept for the recently used option sets
  within 64 MB (every set was kept for good); an unreadable `inductor.bin` in a prepared directory no longer skips its
  fla kernel configs; evidence spans are scored over the band of span ends only (a 32k-token state took 4 GB per score
  matrix; the same spans), and a state of a few tokens returns no inverted or empty span. `basal-finetune` trains the
  readout `basal-serve` serves (Gemma's final soft cap once, not twice), refuses models served with other attention
  masks (two-way, isolated options), trains the decoder's linear layers only (as `--adapter` takes them) and leaves the
  base model's NVFP4 activation scales out of the merged model; `--adapter` loads bf16 adapters and refuses a merge
  into integer (quantised) weights instead of writing damaged ones; `basal-export-gguf` writes under a temporary name
  and renames when complete, so an interrupted export leaves no truncated file; the graph engine reads an NVFP4 amax
  file wrapped or plain, also from a path. Answers to valid requests do not change.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-attach` refuses an `--out` that overlaps the base model** (`basal-attach`): with `--force`, an `--out` equal
  to the base's own directory (also through a symlink) replaced the base's files with links to themselves; such an
  `--out`, one inside the base's directory or one containing it is now refused before anything is written.
  [#127](https://github.com/rkinas/basal/pull/127).
- **JSONL lines with U+2028, U+2029 or U+0085 in a string are read whole** (`basal-run`, `basal-bench`,
  `basal-calibrate`, `basal-agree`, `basal-finetune`, `basal-viz`): the readers split lines with `str.splitlines`, which
  also splits at these separators, so a valid line written with `ensure_ascii=False` (text pasted from a word processor
  or a PDF) was cut in two; they now split at `\n` only, as `scripts/check_jsonl.py` does.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-run` survives a bad line and streams its output** (`basal-run`): a line that is not JSON, a simple item
  without options or an answer that cannot be read gets its own `error` instead of ending the run without output;
  lines are read and answers written as they go (at most a few times `--concurrency` lines held), through
  `<output>.part`, which an interrupted run leaves with its answers so far; `--output` equal to `--input` (also
  through a link) is refused instead of replacing the requests. [#127](https://github.com/rkinas/basal/pull/127).
- **`--concurrency 0` is refused** (`basal-run`, `basal-calibrate`): it made a semaphore that admits nothing, so the
  run waited for ever. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-loadtest` exits on a question file without simple items** (`basal-loadtest`): a `--questions` file of full
  requests only, or an empty one, gave no request bodies and the tool spun one CPU at 100% for ever (also with
  `--determinism`); it now exits with a message before any request. `--n-conc` and `--determinism` below 0 are
  refused. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-bench`: valid JSON without a memory reading, a message for too few items, the `--identity` status with
  `python -m`** (`basal-bench`): `--out` wrote `NaN` for `mem_gb` (GGUF, Ollama, llama.cpp and the CPU have no
  memory tracking), which is not JSON; it is now `null`. Fewer than 6 usable items (5 warm up the latency runs), or
  `--n` / `--lat-n` below 1, crashed in the timing code; they now exit with a message before an engine is loaded.
  `python -m basal.bench --identity` (as `make bench` runs it) exited 0 when the gate failed; it now exits 1, as
  `basal-bench --identity` does. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-navigate build`: Polish k-means topics, and inputs whose files have the same name** (`basal-navigate`):
  `--clusters K --lang pl` failed in scikit-learn (the Polish stop words were passed as a tuple); two inputs with a
  file of the same name (`docs1/index.md`, `docs2/index.md`) gave the same page id and the build failed, and now
  their page ids get the input's folder in front (ids of a build without such a collision are unchanged); the tree
  written to `-o` inside an input folder is no longer read back as a document by the next build.
  [#127](https://github.com/rkinas/basal/pull/127).
- **UTF-8 text in the tools, and the format-2 vectors replaced only when complete** (`basal-interpolate`, tools):
  the interpolation panel builders and run comparison, `basal-interpolate` (tokenizer, config and basal.json) and the
  figure script opened text without an encoding, which on Windows is the locale's (cp1252: Polish text cannot be
  written, a tokenizer vocabulary is misread); they open it as UTF-8, and a test checks every text open in these
  files. `docs/format2_vectors.py` wrote its output in place, so a run that failed left a truncated vectors file; it
  writes `<out>.part` and replaces the file at the end. [#127](https://github.com/rkinas/basal/pull/127).
- **TypeScript client: errors while the body is read are `BasalError`s; Node >= 18.17** (`basal-client`): a body
  that stalled after the headers ended in a raw `DOMException` instead of a `timeout`, and a body that broke off in a
  generic `Error` instead of a `network` error; the read is now inside the same error boundary (the caller's own
  abort still passes through). A `timeout` that is not a number of seconds > 0 is refused when the client is made,
  and a fraction of a millisecond no longer fails in `AbortSignal.timeout`. `engines` says Node >= 18.17, the first
  18.x with `AbortSignal.any`. [#127](https://github.com/rkinas/basal/pull/127).
- **The plugin and the TypeScript client carry the release version** (packaging): `.claude-plugin/plugin.json` and
  `clients/typescript/package.json` still said 2.5.0 in 2.5.3; they now carry the package's version, and a test
  fails when a release bumps `basal/__init__.py` and `pyproject.toml` without them.
  [#127](https://github.com/rkinas/basal/pull/127).
- **CI runs the TypeScript client's tests and one Ruff over the whole repository** (CI): a job builds and tests
  `clients/typescript` (`npm ci`, `npm test` against `basal-serve` on the stub engine), so an API or preset change
  that breaks the client no longer merges green. pre-commit and CI use the same Ruff (0.15.16) with the rules of
  `pyproject.toml`, over every Python file and notebook (pre-commit checked 4 rule groups with another version, CI
  `basal` and `tests` only). [#127](https://github.com/rkinas/basal/pull/127).
- **`--ui full` with `--api-keys` shows a request only to its own key** (`basal-serve --ui full`): any valid key could
  read every kept request at `GET /ui/request/<id>` (the ids count up), the text and answer of another client's
  request included; it now opens for the key that sent the request or an admin key, and another key gets 403. Without
  `--api-keys` the page stays open, as documented. [#127](https://github.com/rkinas/basal/pull/127).
- **Think mode answers every request it takes** (`basal-serve --think auto`, experimental): a request with a question
  that `ask_if` skipped (a `null` answer) failed with a 500, also with `"think": "off"`, and a request with a task
  (`classify`, `verify`, `compress`, `extract`) or `rank` question was refused with a 422 after the first readout; such
  questions now keep their answers (tasks and rank never think, as multi questions do not), and the vLLM log file the
  think server writes is closed when it stops. [#127](https://github.com/rkinas/basal/pull/127).
- **Audio is read up to the model's window, media work leaves the event loop** (`basal-serve --media on`,
  basal-omni): a clip was decoded whole before its length was checked, so a few hundred kB of FLAC silence could
  decode to gigabytes of float32; a clip is now read up to the window (30 s on Gemma 4; one frame more shows it is
  longer) and refused with the same 422 as before, and a clip of a model without a known window up to 67 million
  samples (`MAX_AUDIO_SAMPLES`, 256 MB of float32). Images and audio are decoded in a worker thread instead of the
  event loop, and the engine's media calls run in a thread of their own, so media requests waiting for the engine no
  longer take the threads text batches need. Answers are unchanged. [#127](https://github.com/rkinas/basal/pull/127).
- **Cancelled requests cost the engine nothing** (`basal-serve`): a request cancelled while its work waited in the queue
  (an MCP call cancelled, a caller's timeout) left its queue entries, which the engine still ran; they are now dropped
  before the engine call (the main lane, `--scheduler` and `--long-lane`), and a request whose entry failed drops its
  other queued entries too. Work already in the engine finishes as before; answers are unchanged.
  [#127](https://github.com/rkinas/basal/pull/127).
- **More bad numbers answer 422 before the run** (`basal-serve`): an `act` cost that is NaN, infinite or an integer
  beyond the largest float (also as text, `"NaN"`, `"1e400"`) gave a 500 after the engine's run, or answers with
  non-finite expected costs; a `multi` question with more than 256 labels under `objective` `iou` or `f1` gave a 500
  after the run. Both are now refused with a 422 before it; numbers given as text (`"9"`) are read as before.
  [#127](https://github.com/rkinas/basal/pull/127).
- **One request, one id and one count** (`basal-serve`, `basal-router`): a body that is not JSON is counted in
  `basal_requests_total` and `basal_requests_failed_total` (and the `--ui` traffic totals) on `/v1/systemone` and
  `/v1/basal` too, as it was on `/v1/decisions`; basal-router answers with `x-request-id` and passes the client's
  `X-Request-Id` (or one it makes) to every replica the request reaches, so the replicas' logs name the request the
  client knows. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-router --blend` with a weight of 0 on `--replicas`** (`basal-router --blend`): when only one model with a
  weight above 0 answered, the answer was the first replica's even if its weight was 0 (`--blend-weights 0 1` gave the
  first model's choice under the second model's name); it is now that model's own answer, and a blend of several
  models leaves the zero-weight ones out of the answer as well as out of the mean.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-router --route` and `--fallback` read every request as the replicas do** (`basal-router`): `--route`
  answered a 500 for a state that is a JSON number, list or object (it now reads the state's JSON text, as
  basal-serve shows it to the model, so such a state routes like the same text would), and `--fallback` sent a
  question of a request with `images` (the alias of `media`) to the text-only LLM, which never saw the images; such
  requests now keep the basal answer, as requests with `media` did. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-router`'s `/health` and `/v1/models` follow the pool of ordinary requests** (`basal-router`): `/health`
  answered 200 while only an `--escalate` or `--blend` replica was ready, though every ordinary request then got a 503,
  and `/v1/models` gave the first ready replica in URL order, which could be the escalation model; both now read the
  replicas of `--replicas` / `--gpus` only, and `/health` also gives the ready replicas of every pool (`"pools"`).
  [#127](https://github.com/rkinas/basal/pull/127).
- **`"facts": "auto"` reads a long state in linear time** (`basal-serve`): the fact extraction rescanned the text
  per match (a long run of space-separated digit groups, many zł amounts, many percentages without VAT, many exchange
  rates), so a 40 kB state could take 14 s and a 1 MB state minutes, with the server answering nothing else meanwhile;
  its time now grows linearly with the state, and a state over 1,000,000 characters with `"facts": "auto"` answers
  422 (send `"facts": "off"` or a shorter state). The facts themselves are unchanged.
  [#127](https://github.com/rkinas/basal/pull/127).
- **Gemma 4 on `--mode mps` keeps the sliding window** (`--mode mps`, basal-omni): one mask went through the
  transformers forward to every layer, so the sliding-window layers read whole states; the engine's own Gemma 4
  forward now runs there with its masks per layer type, as in `--mode eager`. Answers over states longer than the
  window change to eager's, and a `compress` request over such a state, which failed with HTTP 500 (the mask did not
  fit the sliding layers), is answered.
  [#127](https://github.com/rkinas/basal/pull/127).
- **Prepared kernels from the Hub only for trusted publishers** (`--prepared auto`): compiled kernels are code the
  compiler cache runs, so the ones a Hugging Face repository ships load only for the owners in
  `BASAL_PREPARED_TRUST` (comma-separated, `*` for every one; default `Remek`); a local model directory,
  `--prepared <dir>` and what `basal-prepare` wrote on this machine load as before.
  [#127](https://github.com/rkinas/basal/pull/127).
- **The Gated DeltaNet convolution's kernel is said and checked** (`--mode fast`, Qwen3.5 models): without
  flash-linear-attention's `causal_conv1d` the convolution silently ran on torch's `conv1d` (cuDNN), which rounds
  differently from the fla kernel `--mode fast` was checked with; start-up now warns, and the ready line and `/ui`
  name the kernel in use. `--batches compact`, which needs fla's kernel, is refused at start-up without it instead of
  failing at the first request. [#127](https://github.com/rkinas/basal/pull/127).
- **fla kernels outside the recorded table autotune again** (`--mode fast`, Qwen3.5 models): with the recorded fla
  configs in use, a kernel the recording never launched was pinned to its first candidate config without timing, a
  config never measured on that GPU (it may not fit there); only the recorded kernels are pinned now, the others pick
  their config as fla does without a table. The recorded kernels keep their configs.
  [#127](https://github.com/rkinas/basal/pull/127).
- **One launch rule for the block-scaled FP8 and int8 GEMMs** (`--mode fp8 --fp8-scaling block`, `--mode int8`): both
  kernels carried a tile-width override that could never apply, under a comment describing a rule they did not follow;
  one grid helper now serves both and refuses an output size that is not a multiple of 128 (whose last outputs would
  have stayed unwritten) instead of launching. Every basal-omni layer is such a multiple: answers are unchanged.
  [#127](https://github.com/rkinas/basal/pull/127).
- **Less attention work on long TPU prompts** (`--mode tpu`, experimental): the blocked attention of prompts longer
  than the largest bucket scanned every key block for every query block, also the blocks after the query block's last
  token, which causality hides from all its queries (about half of the attention work of a long prompt). Each query
  block now stops at the key block of its last query; the result is bit-identical (JAX on the CPU, fp32 and bf16).
  [#127](https://github.com/rkinas/basal/pull/127).
- **basal-finetune's training steps read the codes in fp32** (`basal-finetune`, bf16 / fp16 training): under the
  training steps' autocast the code readout (the output head's input times the code rows) was a bf16 product, so the
  loss saw logits rounded to bf16 steps (0.125 at 16-32); it is fp32 now, as `basal-serve` reads them. The accuracy
  before and after training was already read in fp32. [#127](https://github.com/rkinas/basal/pull/127).
- **Fused and interpolated models leave out what was fitted to the source weights** (`basal-fuse-adapter`,
  `basal-interpolate`): a fused checkpoint kept the base model's evidence head, trained on the base model's hidden
  states (`--adapter` already answered evidence requests with HTTP 422 for that reason), and an interpolated one kept
  `--meta-from`'s `NVFP4_AMAX.json`, whose activation scales `--mode nvfp4` then read for other weights (values above
  a stale scale are clipped). Neither is copied now: evidence and `--mode nvfp4` need the model's own files.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`guard()` and `route()` fail closed on incomplete or malformed answers** (Python `basal.presets` and the
  TypeScript client): both started from pass and looked only at the answers present, so an empty object or a battery
  without its hazardous checks passed, and a NaN probability passed every threshold comparison. Every question of the
  battery must now be answered (`expect`: `"input"`, `"output"`, a list of names for a battery of your own, or the
  built-in battery of the answers by default); a missing, null or malformed answer (a probability that is not a finite
  number in [0, 1], a severity without level probabilities, with `max_error` an `abstain` that is not true / false) is a
  review, and `route()` sends such answers to a human. The shared test fixture holds the new cases for both languages.
  A call that passes a battery of its own without `expect` now gets a review.
  [#127](https://github.com/rkinas/basal/pull/127).
- **A calibration file of another model no longer certifies decisions** (`basal-serve --calibration`, all modes): a
  file fitted on another model, revision or adapter only warned at start-up and kept its thresholds, so `max_error`,
  tasks and answer sets acted on certificates of that model while `GET /v1/models` showed none. It is now kept for its
  temperatures only, as a file of other option orders: `max_error` defers, tasks report `automated: null`, answer sets
  hold every option and every answer is marked `"calibration": "unvalidated"` (`--strict-calibration` still refuses
  it). Two Hugging Face snapshots are compared by their whole repository ID, so `OrgA/basal-x` no longer matches
  `OrgB/basal-x`; a local copy still matches by its directory name. `basal-calibrate --from-rows` now applies the
  same orders and layout rule as basal-serve to the file it compares with (it ignored the layout).
  [#127](https://github.com/rkinas/basal/pull/127).
- **Certified thresholds come from a true fixed-sequence test** (`basal-calibrate`, decision types and tasks): the scan
  went on past failing confidence levels until one passed, so a type whose error was just above the target was
  certified far more often than 5% of the time (seeded simulation, 800 decisions at 5.1% error and a 5% target: 19.6%;
  at 1.05% and 1%: 9.5%). The levels are now tested from the most confident down and the test stops at the first
  level that fails (4.3% and 4.5% in the same simulation). The first level tested is set by the counts alone: enough
  decisions to meet the bound with one error (93 at 5%, 473 at 1%) when the type has at least twice that many, else
  without an error (59 / 299). A type with errors among its most confident decisions can now get no threshold where it
  got one before; the reason names the first level tested. The docs no longer say that a temperature keeps the order
  of choice and score confidences (it does for yes/no only). The scan is fast on large sets: the bound is a binomial
  tail instead of a bisection, the log-factorial table is no longer copied for every level, and a level that adds no
  error needs no bound (a type or task with 50,000 decisions took about a minute, now under a second). Each
  certificate holds for one type at one target error, as the docs now say. Shipped calibration files are unchanged; a
  refit can certify less. Applied to the basal-omni-1.0 models' own calibration data, the corrected test would not
  issue 4 of their 18 per-type certificates (basal-omni-1.0-max: yes/no at 1% and score at 5%; basal-omni-1.0-mini:
  yes/no at 1%; basal-omni-1.0-ultra: score at 5%). The shipped files keep these thresholds: all four still hold on
  the held-out test data (error rates at or below their targets). [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-calibrate` warns when decisions share a group** (`basal-calibrate`): the Clopper-Pearson bound of a threshold
  counts every decision as independent, so when a type or task has more FIT decisions than groups (the claims of one
  document, the fields of one form, several questions about one state) and they share their errors, the certified
  threshold is optimistic. The report now names the decisions per group of each type and task (`fit_groups` in the
  report and in `task_gates`), and the docs state the assumption. No threshold changes.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-calibrate` input and rows are checked before the fit** (`basal-calibrate`): a labelled question with
  `ask_if` / `depends_on` could be answered null and crash the run after the whole inference; such conditions are now
  dropped (every labelled question is asked) and a missing answer is reported as a failed row. A `"group": null` or
  `"id": null` no longer puts every such line into one group "None" (one huge FIT or CERT cluster). An integer `gold` of
  a score question is always the level index, and is refused when a numeric criteria key names another level (it
  silently picked the key before). Rows with non-finite probabilities or a gold out of range, from the engine or a
  `--from-rows` file, are left out of the fit and reported; one NaN row made every temperature 0.25 before.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-finetune` keeps a shared group on one side of the split** (`basal-finetune`): a line went with the group of
  its first question only, so a per-question `group` shared by lines with other first groups (one customer asked about
  in several records) could land in both the training part and the held-out part, and the calibration fitted on
  `heldout.jsonl` then saw decisions the model trained on. Lines that share a group on any question now go to the same
  part; a line that shares no group keeps the side it had. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-selfcheck` sees bucket temperatures** (`basal-selfcheck`): a server was taken as uncalibrated when it
  reported no per-type temperatures, even with format-2 option-count bucket temperatures in use, so
  `--make-reference` could write tempered probabilities as the reference. It now checks every temperature, as
  `basal-calibrate --url` does (all 1.0, or none, is raw). [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-agree` scores decisions in the orders the server answers with** (`basal-agree`, `basal-agree --identity`):
  the decision agreement and the accuracy averaged both option orders for every model, while `basal-serve` answers
  prompt-format-2 models in the given order alone by default. Format-2 models are now scored on that order; format-1
  models keep the average of both. The readout rows and the gate are unchanged.
  [#127](https://github.com/rkinas/basal/pull/127).
- **A rank question's `prior` must be finite** (`basal-serve`): an infinite weight (`1e999` in JSON, or an integer
  beyond the largest float) passed the check and the answer failed with HTTP 500 (`inf` scores are no JSON); it now
  answers 422 as a negative weight does. [#127](https://github.com/rkinas/basal/pull/127).
- **`/v1/decisions` takes a deeply nested input** (`basal-serve`): an `input` nested a few hundred levels deep (valid
  JSON) failed with HTTP 500 from a recursive image check; it is now decided like the same state on
  `/v1/systemone`, and an image part at any depth is still refused with 400.
  [#127](https://github.com/rkinas/basal/pull/127).
- **An option key that is NaN or infinite answers 422** (`basal-serve`): a `criteria` list of a choice, multi or act
  question with `NaN` or `1e999` among its keys was decided, then the answer (with that key) failed with HTTP 500
  because it is no JSON; it is now refused before the engine runs. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-router` passes client errors on instead of failing** (`basal-router`): a state with a lone surrogate
  (`\ud800`) or a body nested too deeply for the JSON parser made the router answer 500 (hashing the routing key,
  reading the state), and an API key header with surrounding whitespace (`Authorization: Bearer ` without a token) 502;
  such requests now reach the replica and get its answer. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-interpolate` takes sharded checkpoints from the Hugging Face cache** (`basal-interpolate`): every shard of a
  downloaded repository is a link into the cache's `blobs/` folder, and the check that the index names no file outside
  the model folder followed those links, so any sharded checkpoint given as a repository id (or a snapshot folder) was
  refused as "outside the model folder". The index's paths are now checked as written (still no absolute path and no
  `..`). [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-interpolate --meta-from` another checkpoint is checked** (`basal-interpolate`): the merged model takes its
  config.json and tokenizer.json from `--meta-from`, but only the two interpolated checkpoints were compared, so a third
  checkpoint with another vocabulary or architecture gave a merged model served with the wrong tokenizer or config. It
  is now refused before anything is written unless its architecture and vocabulary are the checkpoints'.
  [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-navigate build` checks the title of a JSONL document** (`basal-navigate`): a `"title"` that was not a
  string (a number, a list) ended the build with a `TypeError` from the parser; the line is now refused with its file
  and line number, as a document without text is. [#127](https://github.com/rkinas/basal/pull/127).
- **`basal-router --ui on` no longer shows the credentials of `--route` URLs** (`basal-router`): the page's option
  list redacts URLs (no user, password, query or fragment), but a `--route SCRIPT=URL[,URL]` value did not parse as a
  URL, so `user:password@` of its first URL was shown on `/ui/state` (and its other URLs were cut off). Each URL of the
  value is now redacted on its own. [#127](https://github.com/rkinas/basal/pull/127).
- **A long first request no longer stalls basal-serve** (`basal-serve`, prompt format 2 most): the token ids of the
  option codes, computed once per server on its first prompt, were read by tokenizing that whole prompt once per code
  (255 codes with prompt format 2, 10 with format 1), so a first request with a long state held the event loop: 0.6 s
  with a 16 KB state, 3-6 s with 64 KB, 1.5-2.5 minutes with 1 MB on basal-2.0 and basal-omni tokenizers, with every
  other client and `/health` waiting. The codes are now read at the prompt's end, where its tokens are those of the
  whole prompt (the same ids on every basal tokenizer). Answers are unchanged.
  [#127](https://github.com/rkinas/basal/pull/127).
- **With the state cache on, MLX answers no longer depend on what an earlier request cached** (`--mode mlx
  --state-cache-gb`): a state computed for one request was reused by another request about the same state although
  it was computed at that request's row counts, so the second request's answers depended on the order of the requests
  (a 20-option question changed its top answer); such a request now computes its state afresh, and answers match the
  cache off within MLX's normal batch variation. [#127](https://github.com/rkinas/basal/pull/127).

# What is new in basal 2.5.3

Released 4 November 2026. Bug fixes, contributed by Dariusz Koryto and Rafal Wiktorowicz (NovaraX), and the fix of
an issue Dariusz reported. Answers to valid requests are byte-identical to 2.5.2 (checked on an H100 with
basal-omni-1.0-mini in `--mode fast`), and the prepared start-up directories shipped for 2.5.0 still load (the
compiled code is the same). Only the preset wording (#97) and `basal-bench`'s timing of llama.cpp and Ollama (#90)
change what a run gives.

- **Windows: `basal-attach` without symlink rights, portable tree ids and served names** (`basal-attach`,
  `basal-navigate build`, `basal-serve`): where symlinks are not allowed (Windows without developer mode),
  `basal-attach` copies directories and hard-links or copies files instead of failing; navigate-tree ids and groups
  use `/` on every platform; the served name and the calibration's model name are read with `pathlib`, so a Windows
  path gives the model's name, and `--model .` is served under the directory's name (it was `.`). The tests are
  portable to Windows. Contributed by Rafal Wiktorowicz (NovaraX), [#85](https://github.com/rkinas/basal/pull/85).
- **`basal-bench` times llama-server and Ollama on prompts they have to compute** (`basal-bench --modes llamacpp
  ollama`): the warm-up and the single-order call ran on the same prompt just before the timed call, so the prompt
  cache of an HTTP engine answered it almost for free, `lat1_ms` and `lat2_ms` were too low and `dec_s` too high.
  Each question of a timed call now carries a random prefix (both option orders share it, so the second order still
  reuses the first); vLLM and SGLang are unchanged, as `cold()` already clears them. Latencies of llama.cpp and Ollama
  runs rise and their throughput falls, so they are no longer comparable with earlier ones. `lat2_p95_ms` is the
  nearest-rank 95th percentile: with a multiple of 20 samples (100 with the default `--lat-n` and 105 or more
  questions) it was one rank higher. Contributed by Rafal Wiktorowicz (NovaraX),
  [#90](https://github.com/rkinas/basal/pull/90).
- **`basal-run --preset` works again** (`basal-run`): in 2.5.0-2.5.2 every line failed with `KeyError: 'q'`, because
  the summary read a preset line (a state only) as a simple line. Contributed by Dariusz Koryto (@dkoryto),
  [#91](https://github.com/rkinas/basal/pull/91).
- **Batch flags measured on the DGX Spark (GB10)** ([HARDWARE.md](HARDWARE.md#dgx-spark-gb10-batch-flags-with-basal-20)):
  bf16 basal-2.0 with PyTorch 2.11 and 2.13; the defaults are the best choice there, and `--batches compact` and the
  format-1 cascade need PyTorch 2.13 on the GB10. Contributed by Dariusz Koryto (@dkoryto),
  [#94](https://github.com/rkinas/basal/pull/94).
- **Presets pick the Polish wording for short Polish text without diacritics** (`basal-run --preset`,
  `Basal.preset`, `guard`, `route`, the framework adapters): the language of the state is now read from more signs of
  Polish (diacritics, common words, also typed without diacritics) against common English words, so "nie dziala
  logowanie" gets the Polish questions; `Basal(lang=...)` words a preset in the client's language when the call gives
  none. Answers to requests without a preset are unchanged. Reported by Dariusz Koryto (@dkoryto),
  [#92](https://github.com/rkinas/basal/issues/92); [#97](https://github.com/rkinas/basal/pull/97).
- **Tests and CI**: the tests no longer depend on the order they run in (the SDPA backend switches are restored after
  every test, [#87](https://github.com/rkinas/basal/pull/87)), on the GPU class and torch version
  ([#86](https://github.com/rkinas/basal/pull/86)), on the runner's speed
  ([#88](https://github.com/rkinas/basal/pull/88)) or on flash-linear-attention being installed
  ([#98](https://github.com/rkinas/basal/pull/98)); the CI runs the suite on two workers, macOS in about 5 minutes
  instead of 13 ([#89](https://github.com/rkinas/basal/pull/89)). Contributed by Dariusz Koryto (@dkoryto) (#86-#89).

# What is new in basal 2.5.2

Released 3 November 2026. Bug fixes, all contributed by Rafal Wiktorowicz (NovaraX). Answers to valid requests are
byte-identical to 2.5.1 (checked on an H100 with basal-omni-1.0-mini in `--mode fast`), and the prepared start-up
directories shipped for 2.5.0 still load (the compiled code is the same). **One change on the default path:** a
question with an unknown `type` was answered as `choice` and is now refused with HTTP 422 (#84).

- **The start-up 503 answers every HTTP method** (`basal-serve`): while the model loads, every method (PUT, DELETE,
  OPTIONS, PATCH, TRACE, CONNECT, any other) gets the same `503 {"status": "starting"}` with `Retry-After: 5` as GET
  and POST, not the 501 HTML page; HEAD gets the status and the headers without a body.
  Contributed by Rafal Wiktorowicz (NovaraX), [#80](https://github.com/rkinas/basal/pull/80).
- **The config probes of the default mode say why they fell back** (`basal-serve`): when a model's config.json exists
  but cannot be read (permissions, a full disk) or parsed (damaged JSON), `_gemma4`, `_recurrent` and `_gdn` print a
  `basal: warning:` line with the model and the exception instead of silently returning their default; a model
  directory without a config.json stays silent. Contributed by Rafal Wiktorowicz (NovaraX),
  [#81](https://github.com/rkinas/basal/pull/81).
- **`basal-serve` starts on native Windows, and `--quant int8` works** (`basal-serve`): the 503 start-up server called
  `os.register_at_fork`, which Windows does not have, so the server died with `AttributeError` before loading the
  model; the call is now guarded. `--quant int8` was offered by the parser and `--help` for the CUDA graph modes but
  rejected by `validate_quant`; it is accepted now, as `--mode int8` already was. Contributed by Rafal Wiktorowicz
  (NovaraX), [#82](https://github.com/rkinas/basal/pull/82).
- **Calibration files keep their provenance and the flags are honoured** (`basal-serve`, `basal-calibrate`): the
  `CALIBRATION.oN.json` file picked for the served option orders is checked for the layout like the base file;
  `GET /v1/models` reports `temperature_per_bucket` and `basal-calibrate --url` refuses a server that applies bucket
  temperatures; `--from-rows --no-calibration` ignores the baseline; `--save-rows` keeps layout, adapter and revision
  and `--from-rows` writes them to `fitted_on`. Contributed by Rafal Wiktorowicz (NovaraX),
  [#83](https://github.com/rkinas/basal/pull/83).
- **An unknown question `type` answers 422** (`basal-serve`): a question whose `type` is not choice, noul, score, act,
  multi or tool (a typo, for example) was read as `choice` and answered with 200; it is now refused with
  `INVALID_REQUEST`, as `basal-calibrate` already does for labelled data. Requests that relied on the fallback must
  send a valid type. Contributed by Rafal Wiktorowicz (NovaraX), [#84](https://github.com/rkinas/basal/pull/84).

# What is new in basal 2.5.1

Released 2 November 2026. A bug fix; defaults and answers are unchanged.

- **`--prepared auto` loads the start-up kernels shipped in the model repositories:** in 2.5.0 the server's first,
  metadata-only download brought a repository's `prepared/<key>/manifest.json` into the model's snapshot without its
  `inductor.bin`; the server took that directory as found, never fetched the kernels and compiled as on a cold start.
  Now no download takes `prepared/` along, and a prepared directory counts only when every file its manifest lists is
  there, so the server fetches the missing ones (also into snapshots 2.5.0 left incomplete). The directories shipped
  for 2.5.0 load in 2.5.1 (the compiled code is the same). On 2.5.0, download the directory once before the first
  start: `hf download Remek/basal-omni-1.0-mini --include "prepared/*"`. Answers never changed: without the kernels
  the server compiles them as before. Directories written by `basal-prepare` on your machine were not affected.

# What is new in basal 2.5

Released 2 November 2026. Default answers to decision requests are byte-identical to 2.4.1 (checked on an H100:
basal-omni-1.0-mini, basal-omni-1.0-max and basal-omni-pico with `--mode fast`, mini and pico with `--mode eager`); what
is new is opt-in unless a line says otherwise.

- **`rank`: thousands of candidates in one request (experimental):** yes / no screening in shared-context requests of
  50 candidates, then one choice question over the top 20, beyond the 255 options of a choice question (at most
  10,000 candidates per request, `--rank-max-candidates`). The choice reads the screening state again, a state-cache
  hit with `--state-cache-gb`. [NAVIGATE.md](NAVIGATE.md)
- **`basal-navigate`: retrieval by navigating a document tree (experimental):** topics, pages and passages judged level
  by level with `rank` questions, no index; a tree builder for Markdown, HTML, text and JSONL documents, optional
  k-means topics (`basal[navigate]`), budgets, pinned pages and extra candidates from your own retriever. Measured
  with basal-omni-1.0-max on one H100 against a dense 8B embedder: MultiDoc2Dial R@1 44.27 vs 34.70, ConditionalQA
  Hit@1 63.47 vs 44.28; the final choice helped on the first and lowered Hit@1 by 1.48 on the second (`top` 0 turns
  it off); 24-64 s per query (median, 8 concurrent clients). [NAVIGATE.md](NAVIGATE.md#measured-results)
- **`basal-navigate search --split-at N`: very large collections (experimental, off by default):** a level with more
  than N candidates goes out as several requests of at most N, so a tree whose levels exceed basal-serve's
  `--rank-max-candidates` (10,000) is searched without refused requests and without raising the limit; `--per-page`
  sets the last stage's passages per page from the command line. Not faster: on Natural Questions (2.68 million
  paragraphs, page levels of ~11,700 candidates, one H100) 51.6 s per question against 51.4 s with one request per
  level. Scores can differ slightly from one-request runs (at most 0.045); the first passage and the top 10 were the
  same on every measured question. [NAVIGATE.md](NAVIGATE.md#very-large-collections-experimental)
- **Tokenize once (default):** the prompts of a request share the template and the state; the server now tokenizes
  that shared part once and each prompt's own tail after it, cut after a newline where whole-prompt tokenization
  always ends a token, so the engine reads the same ids as before. For the tokenizers of the basal models (Gemma 4,
  Qwen3.5, Bielik v3), checked token for token against whole-prompt tokenization on the release request sets, `rank`
  and `basal-navigate` requests and 12,000 fuzzed state / question splits each; other tokenizers tokenize each prompt
  whole as before. A `rank` request of 300 candidates over a 12,514-token state takes 0.14 s of CPU instead of 5.77 s
  on one thread of an H100 server (0.08 s instead of 3.4 s on an Apple M4 Pro).
- **basal-viz reads Gemma 4 basal-omni text decisions:** the logit lens runs over basal-omni's text decoder with the
  output head's bias and the final logit softcap, so every layer's code distribution ends at the probabilities the
  eager engine serves. Requests with media are refused with a message (basal-viz explains text only). Gemma 4
  inspection is limited to 2,048 tokens per rendered prompt; concurrent HTTP inspections are serialized together
  with their served comparisons so layer hooks cannot capture another request's activations.
  [SERVING.md](SERVING.md#see-how-a-decision-is-made)
  Contributed by Paweł Kiszczak (@pawelkiszczak), [#75](https://github.com/rkinas/basal/pull/75).
- **basal-viz with `"facts": "auto"`:** the explained state carries the computed facts block (dates, gaps, VAT), as
  basal-serve reads it, so the page explains the prompt the served answer came from and the comparison with
  basal-serve's probabilities holds for such requests.
  Contributed by Paweł Kiszczak (@pawelkiszczak), [#73](https://github.com/rkinas/basal/pull/73).
- **basal-viz on the MLX engine (`--mode mlx` / `mlx-q8`):** on a Mac the page reads the layers of the engine basal-serve
  runs there (the `-MLX-8bit` / `-MLX-fp4` exports, bf16 weights, or 8-bit weights made at load with `mlx-q8`), with the
  codes read as that engine reads them, so it shows where a quantised model's decision forms or moves (Llama-architecture
  models; Gemma 4 basal-omni stays on the PyTorch modes). A prompt over the context is refused with HTTP 422 in every mode
  instead of running untruncated. [SERVING.md](SERVING.md#see-how-a-decision-is-made)
  Contributed by Grzegorz Brzezinka (@agentGreg), [#78](https://github.com/rkinas/basal/pull/78).
- **basal-calibrate on a Hugging Face cache snapshot:** `--model <cache>/models--<org>--<name>/snapshots/<commit>`
  records `fitted_on.model` as the repo id and `fitted_on.revision` as the commit, as basal-serve names the model, so
  the file is applied when that snapshot is served (before, the path was recorded and `--strict-calibration` refused
  the file as fitted on another model). The same holds with `--url` and `--model`; repo ids and plain directories are
  recorded as before. Contributed by Paweł Kiszczak (@pawelkiszczak), [#71](https://github.com/rkinas/basal/pull/71).
- **LoRA adapters at load** (`--adapter DIR` on basal-serve, basal-calibrate and basal-bench): an mlx-lm or PEFT LoRA
  adapter (rsLoRA too) is merged into the weights at load, with the arithmetic of PEFT's `merge_and_unload`;
  `--mode vllm` and `sglang` serve a merged copy, and `basal-fuse-adapter` writes the merged checkpoint (for GGUF,
  Ollama or llama.cpp, or to ship it). The answers equal the fused checkpoint's and PEFT's merge's bit for bit
  (Gemma 4 E4B and 31B adapters on an H100 in `fast`, `eager`, `fp8` and `vllm`, E4B in `mlx` on a Mac, and a
  basal-finetune adapter of basal-omni-pico). Calibration files record the adapter
  (`fitted_on.adapter`) and the adapter directory's own file comes first. Refused with a message: GGUF, Ollama and
  llama.cpp, FP8 checkpoints, `fast-exit`, `nvfp4`, `int8`, `vllm` for Qwen3.5 models, DoRA.
  [SERVING.md](SERVING.md#serving-modes),
  [CALIBRATION.md](CALIBRATION.md#which-model-a-file-belongs-to). Contributed by Paweł Kiszczak (@pawelkiszczak),
  [#52](https://github.com/rkinas/basal/pull/52).
- **Tests on Apple M5**: the float32 MLX tests set `MLX_ENABLE_TF32=0` (`tests/conftest.py`), since MLX runs float32
  matmuls in TF32 on an M5's GPU by default; served bf16 models are not affected. Contributed by Grzegorz Brzezinka
  (@agentGreg), [#76](https://github.com/rkinas/basal/pull/76).
- **basal-omni-pico keeps float32 on an Apple M5** (`--mode mlx`): pico's float32 compute (the default; not
  `--mlx-compute bf16`) sets `MLX_ENABLE_TF32=0` at load unless the variable is already set, so an M5's GPU does not
  run its matmuls in TF32. Not yet verified on M5 hardware; on an Apple M4 Pro the answers are byte-identical to before
  (200 prompts, and 2,000 readouts of five questions in two orders over one state).
  [HARDWARE.md](HARDWARE.md#apple-silicon)
- **basal-router: several replicas behind one port.** `basal-router --replicas URL ...`, or `--gpus 0,1` to start one
  basal-serve per GPU (stopped with the router, also when the router is killed). Requests over the same state go to the
  same replica (state-hash routing with bounded load), so its state cache keeps working; a replica that answers 503 or
  not at all gets no requests until it is ready again. On two H100s the router served 1.58x the requests of one GPU
  (twice the clients); scaling is not linear yet. [SERVING.md](SERVING.md#several-gpus-metrics-and-health-25).
- **Cascades in basal-router** (`--escalate URL --escalate-below P`): questions whose top probability is under P are
  asked again of a bigger model, and every answer says which model answered (`answered_by`). Repeat both for ordered
  levels (for example basal-omni-pico -> mini -> ultra), each with its own threshold; with two or more levels every
  answer names its level and model, /metrics counts answers per level, and a level that is down leaves the previous
  answer. On a held-out half, mini -> ultra at 0.95 served about 1.5x the requests of ultra alone on one H100 at the
  same accuracy (within 0.5 pt); at 16 clients its p95 latency is higher than ultra's. `--route SCRIPT=URL` sends states
  mostly in one script (such as Cyrillic) to their own replicas. [RESULTS.md](RESULTS.md#cascade-pico-mini-ultra).
- **`--blend` in basal-router**: the weighted mean of several models' probabilities (`--blend URL ... --blend-weights
  ...`). 0.6 ultra + 0.2 mini + 0.2 pico scored +0.45 pt above ultra alone on a held-out half (95% interval +0.09 to
  +0.81). A quality option, not a speed option: every question goes to every model. Blended answers are not calibrated;
  refit thresholds with basal-calibrate before automating them.
- **LLM fallback in basal-router** (`--fallback URL --fallback-model NAME --fallback-below T`): after the last level, a
  choice question whose top probability is still under T goes to a chat model you serve behind an OpenAI-compatible
  endpoint, which must pick one of the options; its answer (`"answered_by": "fallback"`, `"calibrated": false`) replaces
  the basal one, and the basal one stays when the endpoint fails or is late. With basal-omni-1.0-ultra and, as an
  example, Gemma 4 31B-it with thinking at T = 0.8, on a held-out half: +1.40 pt over ultra alone (95% interval +0.90 to
  +1.89; 8.4% of the questions sent), when no call is cut by the timeout. One call took 54 s at the median and 308 s at
  p95 (one H100, one request at a time); with the default `--fallback-timeout 60`, 46 of 100 calls kept the basal
  answer, about three quarters of the gain on that sample (`--fallback-timeout 600` for offline or review use). Fallback
  answers repeat only with the endpoint settings in [SERVING.md](SERVING.md#several-gpus-metrics-and-health-25)
  (batch-invariant vLLM, no prefix cache, no chunked prefill).
- **A live web page for basal-serve** (`--ui on`, default off): at `/ui`, the server's status, traffic since start
  (requests, failures, rates, p50 / p95 / p99, per endpoint), the state cache, GPU and host telemetry, the configuration
  as parsed and resolved (secrets redacted) and the last 200 requests' metadata, never their text; `--ui full` also
  shows each of those requests and its answer on a click (kept in memory; use `--api-keys` on a shared host). A request
  pays one append to a ring in memory; the rest is computed when a page polls, at most once a second. Open one page at a
  time: on one H100 (basal-omni-1.0-mini, `--mode fast`) the throughput stayed within 1% of `--ui off` with 0, 1 or 2
  pages open, and the p50 latency at 8 clients rose 0.4-2.2% with 1-2 pages and 5.2% with `--ui full` and one page
  (`--ui off`'s own p50 varied by up to 9% between passes); at 1 client the latency stayed within 0.9% and the answers
  byte-identical. A fresh server's first pass at 8 clients is slower with or without the UI. With `--api-keys` the page
  asks for a key. [SERVING.md](SERVING.md#live-web-page-25)
- **One page for basal-router and all its replicas** (`basal-router --ui on`, default off): at the router's `/ui`,
  the replicas, readiness, routing by state and by load, requests in flight, failures, the cascade / blend / fallback
  counters, traffic through the router since start and its last 200 requests (with the replica that answered), next to
  one card per replica with its own status, traffic, state cache and GPU telemetry, read from that replica's `/ui/state`
  (start the replicas with `--ui on`; one without it shows what the router knows). A request pays one append to a ring
  in memory; the replicas are read only while a page polls, at most once a second. On 2 H100 (basal-omni-1.0-mini,
  `--mode fast`, one page open) the answers at 1 client stayed byte-identical to `--ui off`, and the p50 latency at 8
  clients rose 1.3%, within `--ui off`'s own spread between passes.
  [SERVING.md](SERVING.md#one-page-for-basal-router-and-its-replicas)
- **`GET /metrics` (Prometheus) and start-up states**: basal-serve and basal-router export their counters and gauges;
  basal-serve answers HTTP 503 `{"status": "starting"}` while the model loads and warms up, and a failed start-up exits
  instead of hanging. [SERVING.md](SERVING.md#prometheus-metrics).
- **Prepared start-up** (`basal-prepare`, `--prepared auto`): the kernels tuned for this GPU and engine version are
  saved once and loaded at start-up, and the answers are byte-identical to a cold start. The model repositories carry
  `prepared/<key>/` for the H100 (basal-2.0-mini, basal-omni-pico, basal-omni-1.0-mini, basal-omni-1.0-max); `--prepared
  auto` fetches only the directory for your GPU and versions. First start on an H100 with empty caches:
  basal-omni-1.0-mini 290 -> 172 s, basal-omni-1.0-max 259 -> 164 s, basal-omni-pico 105 -> 81 s.
- **basal-omni-pico starts the same every time** on the H100 (default path): fla's kernel configs are pinned, so empty,
  warm and long-used caches give bit-identical answers. 2.4.1's pico fast answers could differ between starts (kernel
  autotuning); 2.5's do not.
- **basal-omni-1.0-ultra on a 48 GB card**: `--quant-load cpu` quantises one layer at a time on the GPU, so ultra in
  `--mode fp8` or `int8` fits a 48 GB card, and `--graph-memory-gb N` sizes the CUDA graphs for N GiB, so every GPU
  given the same N answers the same: on an H100 with 32 GB taken, 1,700 of 1,700 answers bit-identical to the unloaded
  card ([QUANTIZATION.md](QUANTIZATION.md)). Also fixed: with little free GPU memory, a prompt just past the largest
  compact graph no longer fails (HTTP 500).
- **Answer sets and abstain** (per question): `"coverage": 0.9` returns the set of options that holds the right one with
  that probability; with `"max_error"`, a choice, yes/no or score answer gets `"abstain": true` below the certified
  threshold for its type. [API.md](API.md#using-the-confidence).
- **Second read** (`--reread T`): answers below T are read again (the state twice) and the two reads combined;
  `--reread-image` reads an image again at more soft tokens.
- **Two-way models** (trained to read the state in both directions) in the fast engine, and **isolated options**
  (basal.json `"option_isolation"`) for models trained to read the options as a set.
- **Compact rows for basal-omni on Blackwell workstation GPUs** (`--batches compact`; opt-in, off by default: faster,
  but the answers differ from the default rows'): with the flex kernels (sm_120, such as the RTX PRO 6000) compact rows
  run with a flex block mask per prompt in every layer; before, they needed FlashAttention 3. One RTX PRO 6000
  Blackwell Max-Q, 16 clients, one question per request, against the default rows: basal-omni-1.0-mini 1.17-1.20x the
  questions per second, the same top answer on 98.67-99.60% of the readouts and 0-1 changed where the default's margin
  was at least 0.05; basal-omni-1.0-max 1.21-1.27x, 98.00-99.75%, 2-4 such changes (600-2,000 readouts per workload).
  Five questions over a long state are not faster. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **basal-omni-pico in the fast engine with the caches**: the state cache, several questions over one state,
  `--prefix-cache on` and `--batches compact` (with `--readout-layer on`), with answers as close to fp32 as `--mode
  eager`'s (1,700 held-out decisions, H100); compact rows serve 1.11-1.16x the short single questions per second at 16
  clients (no speed-up on long states).
- **`--long-lane` for basal-omni-pico** (`--long-lane-gdn on`; opt-in, off by default: its answers are a little further
  from float32 than `--mode eager`'s): without the option `--long-lane` stays refused for Qwen3.5 models, and the
  message names it. One H100, 1,700 held-out decisions against eager float32: with five questions per state the same
  top answer on 99.39% of 8,500 readouts against eager bf16's 99.51% (0.12 pt fewer); single questions, two option
  orders and both within 0.1 pt of eager bf16's (99.71%, 99.65%, 99.46%). No speed was measured for it.
  [SERVING.md](SERVING.md#serving-modes)
- **basal-omni-pico on a Mac**: `--mode mlx` serves it from its own Hugging Face repository (bf16 weights, computed in
  float32); on 1,700 held-out prompts its answers match CUDA eager float32 (top-1 100%). `--mode mps` stays refused for
  it.
- **basal-omni-pico on a Mac in bf16** (`--mode mlx --mlx-compute bf16`; opt-in, off by default: faster, but further
  from float32 than the default float32 compute): Apple M4 Pro, 1,700 held-out single prompts: 0.136 s per prompt
  against 0.171 s for the GGUF Q8_0 path on the same Mac (1.25x); the same top answer as CUDA eager float32 on 99.41%,
  1 changed where float32's margin was at least 0.05 (the default: 100%, 0); five questions in two option orders
  (17,000 readouts): 99.33%, 8 changed (the default: 99.98%, 0). [QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)
- **basal-omni-1.1-max images and audio through `--mode vllm --media on`**: the same top answer as the eager media path
  on 97.4% of images and 95.8% of clips (every one where eager's margin is at least 0.2), 2.7x / 2.9x the requests per
  second at 8 in flight; `"detail": "high"` stays on `--mode eager`. [OMNI.md](OMNI.md#vllm---mode-vllm).
- **`--mode vllm` against `--mode eager`**: vLLM reads in its own kernels, so its answers are close to eager's, not
  equal: the same top answer on 98.3% of 361 readouts (basal-omni-1.0-mini), 98.3% (basal-omni-pico), 98.6%
  (basal-omni-1.0-max) and 97.6% (basal-omni-1.1-max) on an H100; answers may differ on near-ties, and vLLM's default
  preset varies from run to run. `--vllm-preset deterministic` gives reproducible answers, bit-identical to 2.4.1's on
  four models; it is refused for basal-omni-pico before vLLM starts (vLLM has no batch-invariant kernel for its Gated
  DeltaNet layers).
- **`--vllm-overlap on`** (`--mode vllm`; opt-in, off by default: a little faster, and the answers differ from off's):
  up to four engine calls in flight through vLLM's async engine, batched together by its scheduler. One RTX PRO 6000
  Blackwell Max-Q, 16 clients, basal-omni-1.0-mini and 1.0-max: 1.01-1.07x the questions per second (1.21x for mini on
  2,000 development decisions, one per request), the same top answer as off on 98.17-99.65% of the readouts, 2-6
  changed where off's margin was at least 0.05 (600-2,000 readouts per workload); two runs with off differed about as
  much. With `--media on` it batches requests with media too (off: one call per request): basal-omni-1.1-max on one
  H100 at 8 clients, new images 1.18-1.49x the requests per second; audio clips about the same as off (0.89-1.23x);
  the same requests again 1.35-1.50x. The same top answer as off on 99.68% of 310 images and 99.65-100% of 285 clips
  (no change where off's margin was at least 0.05), accuracy -0.32 / 0.00 to -0.35 points.
  [OMNI.md](OMNI.md#vllm---mode-vllm)
- Experimental `--deterministic --lanes N` (still behind `BASAL_EXPERIMENTAL_LANES`): now exact, not yet faster.
- **Ready question sets and agent frameworks**: presets (triage, moderation, compliance) in `basal-run --preset` and
  `Basal.preset`; `guard_input`, `guard_output` and `route_intent` (Python and TypeScript), strict or permissive, whose
  default thresholds are hand-picked, not certified (with `max_error` they use the served model's own calibration per
  question type); thin LangChain, LlamaIndex and CrewAI tools over the HTTP API (`basal.adapters`).
  [SERVING.md](SERVING.md#agent-frameworks-and-ready-question-sets-25).
- **TypeScript client** (`clients/typescript`, no runtime dependencies): typed questions and answers, answer sets,
  abstain, `ask_if` / `depends_on`, images and errors; build it from the repository (not on npm yet).
- **Other /v1/systemone servers and clients**: the Python client talks to other servers of the same request shape (a
  bearer key, list criteria, noul answers that carry only `noul`); basal-serve takes `ask_if` / `depends_on`, an
  `images` alias and `--error-format openai`. [API.md](API.md#requests-from-other-v1systemone-servers-clients).
- **basal-attach**: serve a chat model that was never trained on basal (Gemma 4, Qwen, ...) as a decision model; it
  starts uncalibrated and basal-calibrate fits it on your labelled requests.
  [SERVING.md](SERVING.md#attach-a-model-that-was-not-trained-on-basal).
- **basal-agree**: any backend against the native engine: top-answer agreement with a 95% interval, |Δp|, paired
  accuracy, and whether each question gets the same answer alone as in a request.
- **basal-finetune**: LoRA on your own labelled decisions ([FINETUNE.md](FINETUNE.md); tested on Apple Silicon with
  basal-omni-pico, other GPUs not yet). **basal-selfcheck** compares your setup with shipped reference answers, and
  [OLD_GPUS.md](OLD_GPUS.md) runs basal-omni-pico on older GPUs (GTX 10xx, RX 480 / 580, Intel Iris Xe) through
  llama.cpp's Vulkan backend.
- **basal as the local backend of your decision code** ([BACKEND.md](BACKEND.md), a one-function example) and a basal
  skill for coding agents (`skills/basal`).
- **Accuracy by document length**: [RESULTS.md](RESULTS.md#accuracy-by-document-length) and each model card show
  accuracy per document-length bucket, with the number of items in each. The calibration report shows the temperatures
  as served.
- **The state cache keeps branches**: cached states that share a history are all kept, and a request continues from the
  longest one it can extend. On branching agent histories (H100) it reused 1.6-1.7x the prefix tokens of the cache
  before it, with answers as close to the uncached ones. Contributed by Marcin Brzeżański (@marcinbrzezanski),
  [#56](https://github.com/rkinas/basal/pull/56).
- **Continuous integration**: every pull request runs ruff and the CPU tests on Linux and macOS. Contributed by Dariusz
  Koryto (@dkoryto), [#53](https://github.com/rkinas/basal/pull/53).
- **`basal-interpolate`: linear weight interpolation of two checkpoints** that share a base model (WiSE-FT, model
  soups), streamed tensor by tensor; quantized or mismatched pairs (tensors, config, tokenizer) are refused before
  anything is written. [INTERPOLATION.md](INTERPOLATION.md). Contributed by Dariusz Koryto (@dkoryto),
  [#60](https://github.com/rkinas/basal/pull/60).
- **API keys, CORS and request IDs for `basal-serve`:** opt-in `--api-keys` protects HTTP and MCP routes (except
  `GET /health`), `--cors-origins` enables browser access from listed origins, and every response carries an
  `x-request-id` header (also without these flags). Contributed by Paweł Kiszczak (@pawelkiszczak),
  [#68](https://github.com/rkinas/basal/pull/68).
- **basal-router passes API keys on:** a client's `Authorization` / `X-API-Key` header goes, as it came, to every
  replica its request reaches (never to `--fallback`), so replicas started with `--api-keys` (the same key file on
  each) check it as when called directly. The router checks no key itself. Requests without a key are sent as before.
  [SERVING.md](SERVING.md#api-access)
- **A Python client for every question type** (`basal.client`): methods for `multi`, `act`, `tool`, `classify`,
  `verify`, `compress` and `extract` next to `choice`, `yes_no` and `score`; `decide` with question builders for several
  questions in one request; `AsyncBasal` for asyncio; the server's status codes as `InvalidRequest` (422),
  `OutOfMemory` (503, retried with `retries=`) and `ServerError` (500), all still a `RuntimeError`; `models()`,
  `health()` and your own httpx client (`http=`). Existing calls are unchanged.
  [INSTALL.md](INSTALL.md#nvidia-gpu-the-basal-engine-recommended). Contributed by Dariusz Koryto (@dkoryto),
  [#58](https://github.com/rkinas/basal/pull/58).
- **MCP model selection:** servers configured with a model list expose `list_models` and optional `model` arguments on
  `decide` and `decide_many`; `basal-mcp --model NAME` forwards that model on each request.
  Contributed by Paweł Kiszczak (@pawelkiszczak), [#66](https://github.com/rkinas/basal/pull/66).
- **`GET /v1/models` reports certified thresholds and context.** Calibration thresholds are listed per question type when
  the calibration matches the served model; context metadata includes its limit, overflow policy and truncation head share.
  Contributed by Paweł Kiszczak (@pawelkiszczak), [#64](https://github.com/rkinas/basal/pull/64).
- **Serving errors:** invalid `max_error` and set bounds are rejected before inference; error responses include a `code`
  (`INVALID_REQUEST`, `OUT_OF_MEMORY` or `INTERNAL_ERROR`) and 422 responses include `detail`.
  Contributed by Paweł Kiszczak (@pawelkiszczak), [#62](https://github.com/rkinas/basal/pull/62).
- **Fix: task questions next to a question named `<task>::<part>`**: such a name is one of the task's expanded
  questions, so the task read that question's answer as its own (a `classify` named `task` answered with the label of a
  `choice` named `task::labels`) and the other answer was dropped. In a request with task questions a name with `::`
  is now refused (HTTP 422), as a task question's own name already was. Reported by Sourcery's review of #58.
  Contributed by Dariusz Koryto (@dkoryto), [#59](https://github.com/rkinas/basal/pull/59).
- **Fix: replicas that basal-router starts listen on 127.0.0.1** (`--gpus`): they took basal-serve's default host,
  0.0.0.0, so the network could reach them past the router (and past its `--host`). A `--host` after `--` still
  applies.
- **Fix: a request with an empty `"questions"` object** is refused with HTTP 422 (`INVALID_REQUEST`, "must name at
  least one question"); before, it reached the engine and failed with HTTP 500 (2.4.1 too).
- **`POST /v1/decisions`: OpenAI's Decisions API on basal-serve.** `client.decisions.create` of the OpenAI Python SDK
  works against basal-serve: predicate, choice and score questions are decided as the System One request they stand
  for (predicate as noul), with the same probabilities as `/v1/systemone`, and answered in OpenAI's shape; errors in
  OpenAI's error shape. [API.md](API.md#openais-decisions-api-post-v1decisions).
- **LM Studio serves basal as a decision model** (GGUF, prompt-format-2 models): `basal-export-gguf --systemone`
  writes llama.cpp's decision metadata into a copy of a `-GGUF` file: basal's prompt as a template and the calibrated
  temperatures; the weights are not touched. LM Studio 0.4.26 or newer then answers choice, yes/no and score questions
  on its own `/v1/systemone` and `/v1/decisions`, up to 26 options, without `basal-serve` in front (`llama-server`
  too). Against `basal-serve --mode gguf` on the same files (Apple M4 Pro, 620 questions per model): the same prompts
  token for token and the same top answer on every question answered, probabilities within 0.0041; LM Studio stops a
  request after 30 s, which 5 long requests of the 4.5B hit (split long states or send fewer questions per request; for
  long documents `basal-serve --mode gguf` has no such limit). `act`, `multi`, `tool`, the certified thresholds and more
  than 26 options stay with `basal-serve`. [INSTALL.md](INSTALL.md#lm-studio-gguf-as-a-decision-model).
- **`--mlx-forward basal` on Apple Silicon** (opt-in; the default `mlx-lm` is unchanged): basal's own forward over
  mlx-lm's layers for Llama and Gemma 4. The layers from the last one with its own keys and values on run at the
  readout rows only, the GPU starts early, and the projections that read one input become one matrix where a check at
  load finds the same bits. basal-omni-1.0-mini-MLX on an Apple M4 Pro: one prompt 225 -> 131 ms, one question in two
  orders 222 -> 179 ms, one prompt over 3,000 tokens 5.3 -> 3.0 s; basal-2.0-mini-MLX-8bit 2-4% faster on short
  prompts. Top answers equal the default path on 184 of 184 readouts per model. Details in
  [QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon).
- **basal-omni-1.0-ultra in int8 for GPUs without FP8** (opt-in, ultra only): `--mode int8` uses block FP8's layout
  with int8 values (128 x 128 weight blocks, 128-feature activation tiles, exact int32 sums per block) for Ampere and
  newer, with its own `CALIBRATION.int8.json`. 99.4% of top answers as bf16; the answers accepted at the 1% target
  0.52% wrong (bf16 0.47%). Max failed the same check. Details in
  [QUANTIZATION.md](QUANTIZATION.md#int8-for-gpus-without-fp8-25).
- **`--state-cache-bits 8`** (opt-in, with `--state-cache-gb` / `--prefix-cache`): cached states kept as 8-bit integers
  with a scale per group (keys per channel, values per token), about 1.7x as many states per GB. On
  basal-omni-1.0-max, answers over cached states agree with uncached ones on 99.45% of top answers. The accepted
  answers' error at the 1% target is 0.85%, against 0.81% with the 16-bit cache. 4 bits failed the same check.
- **basal-omni-1.0-ultra in block FP8 or NVFP4** (opt-in, ultra only): `--mode fp8` quantises ultra's bf16 weights as
  they load (128 x 128 block scales, the first and last 2 layers in bf16): one 80 GB H100 at 1.35x the decisions/s of
  bf16 and 53 instead of 74 GB at 16 clients; the state cache is off in block FP8 until 2.5.1 (`--state-cache-gb` is
  not used, with a warning at start-up). `--mode nvfp4` with ultra's calibrated input scales and its own calibration
  file: 2.3x bf16's scoring speed on Blackwell, 2.4% of top answers change. Other basal-omni models are refused in both
  modes; they failed the accuracy check. Details in
  [QUANTIZATION.md](QUANTIZATION.md#basal-omni-10-ultra-quantised-at-load-time-25).
- **Fix: FP8 checkpoints load without a bf16 copy on the GPU** (`-FP8` repositories, `--mode fast` and `--mode fp8`):
  the loader built the whole bf16 model on the GPU before it read the checkpoint, so basal-omni-1.0-ultra-FP8 needed
  about 95 GB and ran out of CUDA memory on an 80 GB H100. The model is now built without its weights and the
  checkpoint's tensors become its parameters: ultra-FP8 serves on one 80 GB H100 in both modes (44.8 GB after
  start-up). Checkpoints that fitted before load the same tensors bit for bit and give the same answers.
- **Fix: edited chat states with the state cache** (`--mode fast --state-cache-gb`): a state whose last turn was
  replaced, continued from a cached one inside a sliding layer's window, could read GPU memory that was never zeroed,
  and the request failed with HTTP 500 ("Out of range float values": NaN answers). On basal-omni-1.0-ultra (one H100,
  `--graph-memory-gb 4 --state-cache-gb 5`) 27 of 1,248 edited-chat requests failed before the fix and 0 after; 2.4.1
  had the same bug (2 of 624). Those buffers are zeros now; at 1 client every other answer is byte-identical to before.
- **Option-order check and averaging** (opt-in): `basal-bench --order-check` measures how often the top answer flips
  when the options are reversed or rotated, and what averaging gains; `basal-serve --order-average K` serves the mean
  over K option orders, every order reading the request's state once. Off by default: answers are unchanged.
  [SERVING.md](SERVING.md#serving-modes).
- **No first-request stalls in `--mode strix`** (AMD Strix Halo): start-up compiles every Triton kernel copy a request
  can need. Triton compiles a kernel once per tile config and per class of each length argument (1, a multiple of 16,
  other); the start-up of 2.4.1 and earlier ran two lengths only, so the first request that needed a new copy waited
  0.2-0.3 s for each (in the request matrix, with an empty Triton cache, the first S request with 1 question took
  1.82 s instead of 0.18 s). basal-2.0-4.5B on the contributor's Radeon 8060S: no kernel compiled during 152 requests of 90-16,000
  tokens (before: 13 copies in 4 requests, 2.7 s), the same probabilities to the bit on 2,362 prompts; start-up about
  5 s longer at the first start on a machine, about the same once Triton's disk cache holds the copies. Details in
  [STRIX_HALO.md](benchmarks/STRIX_HALO.md#what---mode-strix-does). Contributed by Michał Kardyś (@kardysm),
  [#51](https://github.com/rkinas/basal/pull/51); the stall was first seen by @piotrkowski in
  [#21](https://github.com/rkinas/basal/pull/21). `--mode strix` stays community-maintained, not yet verified on our
  hardware ([HARDWARE.md](HARDWARE.md#amd-strix-halo-experimental-community-maintained)).
- **Starts after a reboot reuse the compiled code** (`--mode fast`): torch.compile's cache moves from Inductor's default
  `/tmp/torchinductor_<user>`, which a reboot empties on most systems, to `~/.cache/basal/torchinductor`
  (`TORCHINDUCTOR_CACHE_DIR` still takes precedence), as `--mode tpu` keeps JAX's in `~/.cache/basal/jax`. RTX 4060 Ti,
  the first start after a reboot: basal-2.0-mini 417 -> 238 s, basal-2.0-4.5B 823 -> 519 s; every probability
  bit-identical to a cold start; the cache takes about 55 MB for both models. Contributed by Miłosz Kucharski
  (@Butterski), [#47](https://github.com/rkinas/basal/pull/47).
- **Less host work in the CUDA cascade** (`--mode fast`, basal-2.0): the pass 2 without a CUDA graph runs kernel by
  kernel and waits on the CPU; the cascade (2.3.5) now merges its two attentions in one compiled kernel per layer
  instead of ~15 small ones and calls FlashAttention's op directly, not through torch's `varlen_attn` (a Python
  custom op that also allocated a random-number state per call). A100, basal-2.0-mini, one client, against the dense
  pass: 40 questions over a 1.5k-token state 1.07x -> 1.11x, compress 1.15-1.37x -> 1.19-1.44x; pass 2 alone for 40
  questions 62.4 -> 48.6 ms (basal-2.0-4.5B: 117 -> 99 ms); every probability bit-identical to 2.3.5.
  `fast-nocompile` keeps the merge op by op. Contributed by Miłosz Kucharski (@Butterski),
  [#44](https://github.com/rkinas/basal/pull/44).
- **`--compile-scope layer`** (`--mode fast`, the Llama-based basal-2.0 models; opt-in, off by default: a much shorter
  start-up, but the answers differ from the whole-model compile's): torch.compile compiles one decoder layer, called
  for every layer, instead of the whole model, so start-up compiles a layer instead of 32-60. The same transformers
  layers as the whole-model compile; kernels fuse within a layer only, so probabilities move by bf16 rounding. One
  H100, basal-2.0-mini, cold compile caches, 840 readouts: start-up 47 s against 228 s; the same top answer as a
  whole-model cold start on 98.69% (829) and as another on 98.33% (826), the changed ones near-ties (margins up to
  0.044 against the first, one at 0.058 against the second). Two whole-model cold starts agreed with each other on
  98.3-98.5% there. RTX 4060 Ti (the contributor's measurement): graph warm-up basal-2.0-mini 235 -> 95 s,
  basal-2.0-4.5B 515 -> 251 s, single questions 2-3% slower on mini. bf16 Llama weights only (FP8 / NVFP4, Gemma 4,
  Qwen3.5 and --lean are refused). Contributed by Miłosz Kucharski (@Butterski),
  [#49](https://github.com/rkinas/basal/pull/49).
- **State cache on Apple Silicon** (`--mode mlx`, opt-in): `--state-cache-gb N` and `--prefix-cache on` work on the
  Mac as on the CUDA graph engine, with both `--mlx-forward` paths. A request whose state (or template prefix) is
  cached computes only its questions. The probabilities equal the server without the cache bit for bit: MLX picks its
  matmul and attention kernels by the shape of the call, so the cached path runs its forwards at the uncached
  forward's row and key counts (zero rows and keys added). A state that begins with a cached one (a growing agent
  trace or chat) computes only its new tokens where that keeps the bits (basal-2.0-mini: from about 1,024 tokens on;
  basal-omni: never, it is computed afresh). Apple M4 Pro, basal-2.0-mini-MLX-8bit and basal-omni-1.0-mini-MLX, both
  forwards, 1,032 readouts per run: every probability equal to the cache off. p50 on basal-2.0-mini: five questions
  about a ~1.5k-token state 1.10 -> 0.33 s, one question 0.51 -> 0.09 s.
  [HARDWARE.md](HARDWARE.md#apple-m4-state-cache-on---mode-mlx). Contributed by Grzegorz Stark (@gstark0),
  [#50](https://github.com/rkinas/basal/pull/50).
- **`GET /metrics` additions:** the engine version (`basal_build_info{version,model,mode,layout}`), the state cache's
  evictions and prefix hits, and a cache read that never waits for the engine's lock (the last values and their time
  instead). [SERVING.md](SERVING.md#prometheus-metrics). Contributed by Marcin Brzeżański (@marcinbrzezanski),
  [#48](https://github.com/rkinas/basal/pull/48).
- **Development commands:** `make help` lists them: `make setup`, `make test`, `make serve MODE=...`, `make bench`,
  `make run`, `make loadtest`, and `make check`, read-only pre-commit checks with pinned versions (Python syntax and
  undefined names, merge markers, private keys, TOML / YAML / JSON / JSONL syntax), no Git hook installed.
  [CONTRIBUTING.md](../CONTRIBUTING.md#development-commands). Contributed by Paweł Kiszczak (@pawelkiszczak),
  [#46](https://github.com/rkinas/basal/pull/46).
- **A Triton kernel for basal-omni's 512-wide global layers** (`--global-attn triton`, Hopper; opt-in, off by default:
  faster on long states, but the answers differ from the default FlexAttention path's): single prompts and the states
  of multi-question requests run basal's own causal kernel in the global layers. One H100, five questions over 24k-32k-
  token states at 4 clients, against the default: basal-omni-1.0-mini 1.10x the questions per second, the same top
  answer on 99.67% of 600 readouts, none changed where the default's margin was at least 0.05; basal-omni-1.1-max
  1.16x, 98.33%, 4 changed; basal-omni-1.0-ultra 1.03x, 99.0%, 2 changed (15k-17k tokens: 1.07-1.10x).
  [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **Known limit:** repeated requests over long states: with `--mode fast` on one H100, basal-omni-1.0-ultra gives
  bit-identical probabilities for the same 24k-32k-token request sent again (one at a time) in 95 of 100 cases, as 2.4.1
  did; basal-omni-1.0-mini in 100 of 100. Ultra's answers on long states are not guaranteed to be bit-stable.
- **Known limit:** with `--mode fast` and `--state-cache-gb`, a rare race can fail one request with HTTP 500 (an error
  about FX symbolic tracing): about 1 request in 8,000 in our tests; the server recovered and kept serving (retry the
  request). Present since 2.3.1. `--state-cache-lock on` (opt-in) removes it at a cost: on one H100
  (basal-omni-1.0-max, 8 clients, 8,740 requests) 0 failed instead of 1, but 8.19 instead of 8.88 requests/s (-8%) and
  a p50 of 630 instead of 577 ms; the top answers agreed with the uncached ones as without it (99.24% vs 99.25%). A fix
  without that cost is planned for 2.5.1.

# What is new in basal 2.4.1

- **Shorter start-up of the graph engine** (`--mode fast`): every CUDA graph is warmed up once before its capture
  instead of three times (pass-2 graphs: twice). torch.compile compiles and cuBLAS and Triton set up at the first
  call; the further calls ran the same kernels again. RTX 4060 Ti 16 GB, the graph warm-up: basal-2.0-mini 468 ->
  323 s, basal-2.0-4.5B 1,094 -> 680 s (with the pass-2 graph budget it needs there); the same GPU memory and speed,
  every probability bit-identical (420 readouts per model). On an H100 with a warm compile cache start-up is 15-40%
  shorter (basal-omni-1.0-max 201 -> 121 s). Contributed by Miłosz Kucharski (@Butterski),
  [#43](https://github.com/rkinas/basal/pull/43).
- **Prompt layouts** (`--layout`, prompt-format-2 models trained on them): besides the
  canonical layout (the state first, shared by a request's questions), `state_twice` reads the state twice and
  `qfirst` puts the question before the state too, one pass per question, for clients who want the state read knowing
  the question rather than the shared state cache. The codes, the prefill and the readout tokens are the same; the
  canonical prompt is bit-identical to 2.3.5. Each layout loads its own `CALIBRATION.<layout>.json` (basal-calibrate
  `--layout` fits one); `--state-cache-gb` and `--prefix-cache` need the canonical layout.

# What is new in basal 2.4

- **Qwen3.5 models in the fast engine** (`--mode fast`, `fast-nocompile`), such as basal-omni-pico (0.8B: 18 Gated
  DeltaNet layers and 6 attention layers). 2.3.2 refused them there: the engine shared a request's state between its
  questions as attention keys and values only, and the Gated DeltaNet layers lost their recurrent state. The port
  (`basal/gdn_graph.py`) keeps the CUDA graphs, batching and readout of the fast engine and carries the state:
  - the state pass writes, for every request state, the recurrent state of each Gated DeltaNet layer (16 heads x 128 x
    128 in fp32, 1 MB per layer) and the last inputs of its short convolution, at the state's last token (rows are
    padded on the right; padding gets no update and no decay, so the padded row ends in the same state);
  - the question pass runs one row per question and option order, each from that state and over the state's keys and
    values: a question's text under its two option orders runs once per order, since one recurrent state cannot
    continue into two branches of one packed row;
  - the Gated DeltaNet layers run fla's kernels (flash-linear-attention, as in the eager reference), their short
    convolution included, at fixed shapes inside the CUDA graphs; `--mode fast` compiles the layer code between them
    (norms, projections, gates, rotary embedding, MLP), and the projections that read one input are one GEMM.
  Checked on one H100 with basal-omni-pico against `--mode eager`, on 1,700 held-out decisions as single questions,
  five questions over one state, two option orders, and both (17,000 prompts): in fp32 the top answer agrees on
  99.94-99.99% (max |dp| 0.009), so the port computes what eager computes. In bf16, as served, `fast` is about as
  close to the fp32 answers as `eager` is (99.44-99.65% against 99.48-99.71%, by shape). `--deterministic` against batched
  serving on the 10,606 decisions of the dev set: the same top answer on 99.83% (`eager`: 99.42%), accuracy as
  `eager`'s (+0.00 points). Speed over HTTP (`eager` in brackets): one question over a 1.5k-token state 13.5 ms at
  one client (47.0), 135 decisions/s at 16 clients (31.7); five questions 26.6 ms (111.4), 400 questions/s (41.1);
  five questions over a 3k-10k-token state 91 ms (926), 94 questions/s (6.4). One question over a long state is
  slower than `--mode vllm` (46 against 39 ms). Details in
  [HARDWARE.md](HARDWARE.md#basal-omni-pico-qwen35-gated-deltanet-on-one-h100).
- **The same probabilities in every start** for basal-omni-pico, with fresh or reused compile caches: its compiled
  layer code runs in Inductor's deterministic mode, as basal-omni's since 2.3.4, so no reduction config is picked by
  timing. On one H100, three starts (empty caches, warm caches, empty again) gave bit-identical probabilities on 300
  single questions and 100 five-question requests with both option orders; the speed is the same within run-to-run
  noise. fla's own Triton kernels still pick their launch configs by timing at their first launch; Triton keeps those
  picks across starts only with `TRITON_CACHE_AUTOTUNING=1`, and such a long-used cache can hold other picks than a
  clean start (in our runs: other probabilities on 282 of 300 single questions with no top answer changed, and on all
  1,000 five-question rows with 9 top answers changed). Recorded fla configs per GPU are planned for 2.5.
- **Without `--mode`** a Qwen3.5 model now runs `fast` on Hopper (H100, where the port passed its gates) when fla is
  installed (`pip install flash-linear-attention`), and `eager` elsewhere.
- **Still refused for Qwen3.5 models** (before loading, each with what to use instead): `fp8`, `nvfp4`, `fast-exit`,
  `mps`, `--batches compact`, `--state-cache-gb`, `--prefix-cache`, `--template-prefix on`, `--readout-layer on`,
  `--lean`, `--long-lane`. Other recurrent models (Qwen3-Next, Mamba and other hybrids) stay refused by every
  shared-prefix engine and run `eager`.
- **Other models are unchanged**: basal-2.0-mini (default and compact rows), basal-2.0-4.5B (compact rows) and
  basal-omni-1.0-mini in `--mode fast` give bit-identical probabilities to
  2.3.5 (4,960 of 4,960 readouts on an H100).

# What is new in basal 2.3.5

- **Compress and many-question requests faster on CUDA** (`--mode fast`, the Llama-based basal-2.0 models, bf16 on
  sm80+ GPUs where torch's varlen attention runs with per-sequence key counts): a pass 2 without a CUDA graph whose
  rows hold 1,024 or more tokens (every `compress` request, dozens of questions over a state) attends to the state's
  keys and values as pass 1 wrote them and, per trie block, to its own ancestors (torch's varlen attention), merged by
  log-sum-exp, instead of one masked call over concatenated and repeated keys; elsewhere the dense pass runs as before.
  Compress over 1.5k-6k-token documents 1.14-1.39x faster (basal-2.0-mini and basal-2.0-4.5B, RTX 4090 and A100), 40
  questions 1.07-1.11x, 2.1-2.5 GB less GPU memory for the largest of them; smaller requests run as before,
  bit-identical. Answers within bf16 rounding: against fp32, mean |Δp| 6.9e-3 (the dense bf16 pass: 7.0e-3).
  Contributed by Milosz (@Butterski), [#39](https://github.com/rkinas/basal/pull/39).
- **basal-2.0-4.5B on 16 GB cards** (`--mode fast`; RTX 4060 Ti 16 GB, 4080, 5060 Ti): the graph engine captures its
  pass-2 graphs (the questions over a state) only while 1.5 GiB of GPU memory stays free after each capture, the
  smallest (commonest) shapes first; every plain graph is captured as before, and a pass 2 without its graph runs
  uncompiled, as for shapes beyond the grid. Each captured graph keeps device memory for its kernels (about 19.5 MB
  per pass-2 graph on the 4.5B), so the 264 of them kept 2.3.3 from starting on 16 GB. RTX 4060 Ti 16 GB: 80 of 264
  pass-2 graphs, 14.4 GB after start-up, 22.6 single questions per second, 9.7 questions per second five at a time
  over 1.5k-token states, no errors; very large `compress` requests (6k-token documents) can still run out of memory
  there without the CUDA cascade. On 24 GB and larger cards every graph is captured as before, the probabilities
  bit-identical (420 of 420, RTX 4090); where graphs are left out, the requests that would have used them change by
  bf16 rounding only (same top answer on 200 of 200, max |Δp| 0.04). Contributed by Miłosz Kucharski (@Butterski),
  [#41](https://github.com/rkinas/basal/pull/41).
- **Checked on one H100** (basal v2.3.4 against 2.3.5, `--mode fast`): every probability bit-identical on the usual
  paths (basal-2.0-mini and 4.5B, default and compact rows, basal-omni-1.0-mini; 5,800 identity readouts), the same
  graphs, GPU memory and start-up time. With torch 2.13 (#39 needs its varlen attention; torch 2.11 keeps the dense
  pass): compress 1.18-1.30x faster on mini and 1.27-1.49x on the 4.5B, 40 questions 0.90x / 1.11x, the same top
  answer on 240 of 240 decisions per model (max |Δp| 0.031). With a 16 GB budget simulated on the H100 the 4.5B kept
  200 of 264 pass-2 graphs (compact rows) and 2.0 GB free, its identity readouts the same as with every graph.

# What is new in basal 2.3.4

- **Less GPU memory for basal-omni in the graph engine** (`--mode fast`): basal-omni's CUDA graphs are now warmed up in
  the usual order (so torch.compile compiles the same kernels) and captured largest first, as basal-2.0's are since
  2.3.3 (#37). The pass-2 graphs over states beyond the largest root graph come last, each captured as it is warmed
  up, so their graph-pool limit still holds. H100 after start-up: basal-omni-1.0-mini 27.8 -> 18.9 GB (graph pool
  10.6 -> 1.7 GiB), basal-omni-1.0-max 45.3 -> 31.3 GB (16.2 -> 2.3 GiB); the same 520 graphs, every probability
  bit-identical (840 identity readouts per model), the same start-up time and throughput (mini: 197-198 against 2.3.3's 201-216 single questions per second
  at 32 clients, which moves this much between runs, 185-189 against 186-187 questions per second five at a time;
  max: 183 against 186, 52.5 against 52.8). Milosz (@Butterski) wrote the same change in
  [#40](https://github.com/rkinas/basal/pull/40), in parallel with ours and 17 minutes earlier, following his own #37;
  his A100 numbers (graph pool 4.5 -> 3.2 GiB with default rows, 420 readouts bit-identical) and his test of the exact
  set of captured graphs add to ours.
- **The same probabilities in every run** (basal-omni, `--mode fast`): Inductor timed some RMSNorm reduction kernels
  at their first launch and kept the faster reduction block (4,096 or 2,048 values per step, timings within 1% of
  each other). The block sets the order of the sum, so two processes could answer differently: on an H100, two runs
  of basal-omni-1.0-mini on fresh compile caches differed in 485 of 840 identity readouts (top answer in 7), a cold
  and a warm cache in 400 (top answer in 4; 2.3.2 and 2.3.3). The Gemma 4 forward now compiles in Inductor's
  deterministic mode, where such a kernel has one config, chosen without timing: two cold runs and a warm one give
  840 of 840 bit-identical readouts. `TORCHINDUCTOR_DETERMINISTIC=0` turns it off. Against 2.3.3 the top answer
  moved in 2-13 of 840 identity readouts, about as much as 2.3.3 moved between its own runs (4-7), with no loss of
  accuracy on 1,700 QD items: basal-omni-1.0-mini 82.47% -> 82.76% (7 newly right, 2 newly wrong, McNemar p 0.18),
  1.0-max 85.12% -> 85.18% (1 newly right). basal-2.0 is unchanged (its compiled kernels have no timed choice that
  changes the numbers): 4,120 of 4,120 readouts bit-identical to 2.3.3.
- **`--deterministic` with basal-omni** still runs one request per forward. Batch-invariant kernels (fixed-tile GEMMs,
  a row-wise RMSNorm, FlashAttention 3 with the window kept) made basal-omni-1.0-mini and 1.0-max bit-identical alone
  and in any batch (840 of 840 readouts each), but batched they served fewer decisions per second than one request
  per forward (H100, mini: 154 instead of 163 single questions per second at 32 clients, p50 15 instead of 9 ms;
  136 instead of 140 questions per second five at a time; start-up 3 minutes longer), so they are not in this
  release; a faster batch-invariant GEMM is future work. `--deterministic` itself serves as in 2.3.3, with the
  smaller graph pool (mini 164 / 138, max 88 / 56; 100 of 100 answers identical alone and under load).

# What is new in basal 2.3.3

- **Less GPU memory for basal-2.0 in the graph engine** (`--mode fast`): the captured CUDA graphs of a family write
  their outputs into one shared buffer (each graph kept its own hidden states in the graph pool: 169 plain graphs on
  basal-2.0-mini), the pass-2 graphs share one mask buffer, the 12 pass-2 graphs that pass 1 can never fill (8 rows over
  roots of 2,304-3,072 tokens) are no longer captured, so the key / value buffer holds 16,384 tokens instead of 24,576,
  and the captures warm up on one stream (each new stream kept its own cuBLAS workspace). RTX 4090 after start-up:
  basal-2.0-mini 12.7 -> 10.0 GB (compact rows 18.7 -> 17.6 GB), basal-2.0-4.5B 24.1 -> 20.5 GB of the card's 24.6;
  every probability bit-identical (420 readouts per model and batching mode), the same decisions per second (128.7 /
  128.4 single questions, 53.4 / 54.5 questions per second five at a time over 1.5k-token states) and start-up time.
  Contributed by Milosz (@Butterski), [#35](https://github.com/rkinas/basal/pull/35).
- **Compact rows on 16 GB and 24 GB cards** (`--batches compact`): the graph engine warms every CUDA graph up in the
  usual order (so torch.compile compiles the same kernels) and then captures them largest first. Captured smallest
  first, each larger graph took new memory in the shared graph pool instead of reusing what the smaller ones had freed
  (basal-2.0-mini with compact rows: 9.0 GiB of graph pool, now 1.0). The compact graphs also share one output buffer.
  basal-2.0-mini with compact rows now starts on an RTX 4060 Ti 16 GB (9.1 GB; out of memory before), with 117.6
  instead of 57.4 decisions per second on single questions; basal-2.0-4.5B with compact rows starts on an RTX 4090
  (18.4 GB; out of memory before), 131.8 instead of 70.0 decisions per second. RTX 4090 after start-up: mini with
  compact rows 17.6 -> 9.4 GB, without 10.0 -> 9.0 GB, 4.5B 20.5 -> 19.3 GB; every probability bit-identical (420
  readouts per model and batching mode), the same decisions per second as before where both start.
  Contributed by Milosz (@Butterski), [#37](https://github.com/rkinas/basal/pull/37).
- **Checked on one H100** (basal v2.3.2 against 2.3.3, `--mode fast`): GPU memory after start-up basal-2.0-mini 10.8 -> 7.1 GB (compact rows 16.6 -> 7.3 GB), basal-2.0-4.5B 20.1 -> 15.2 GB (compact rows 27.3 -> 15.4 GB), so the 4.5B with compact rows now fits a 24 GB card ([#33](https://github.com/rkinas/basal/issues/33)); 10,872 of 10,872 readouts bit-identical (default and compact rows, `--readout-layer on`, `--prefix-cache` with `--state-cache-gb`, basal-omni-1.0-mini), decisions per second within run-to-run noise.
- **Long states on the CPU without dense masks** (`--mode mps --device cpu`): the branch pass attends to the state's
  keys and values as they are and, per trie block, to its own ancestors, merged by log-sum-exp (the softmax over all
  keys, summed in another order), so it needs no `[N, P + N]` mask and no copied keys, and a request with many
  questions is no longer split with the state recomputed per part. basal-2.0-mini fp32, a 3,278-token state, 6 CPU
  cores: 40 questions 105 -> 92 s, 120 questions 313 -> 207 s (the state once instead of 3 times), peak memory -0.5 /
  -0.75 GB (about half the peak memory on many-question requests), the same top answer on 160 of 160 (max |Δp|
  9e-6). On 32 threads of a Xeon 8460Y+ the 40-question case runs at about the same speed (0.94-1.04x), 150 questions
  1.24x faster. Contributed by Milosz (@Butterski),
  [#34](https://github.com/rkinas/basal/pull/34).
- **Windows:** `--batches compact` and `--gemma-attn flex` / `fa3` are refused with the reason where torch is built
  without FlashAttention or Triton is not installed (the Windows wheels), instead of failing at the first request.
  basal reads and writes its text files (calibrations, question and input files, decision logs) as UTF-8 on every
  system; on Windows it used the locale code page before. A file basal wrote on Windows with non-ASCII text (such as
  Polish questions) does not load as UTF-8: convert it once, on the machine that wrote it,
  `python -c "import locale, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text(p.read_text(encoding=locale.getpreferredencoding(False)), encoding='utf-8')" FILE`,
  or write it again ([#38](https://github.com/rkinas/basal/pull/38), @lukaszpodgorski-pl).
- **Docs:** Apple M5 Pro results for basal-2.0 with `--mlx-packed` and the first basal-omni numbers on an M5 (HARDWARE.md, OMNI.md; Kamil, @kamilrosa, [#36](https://github.com/rkinas/basal/issues/36)); basal-omni-pico (0.8B, Qwen3.5) in the README model table; the basal-omni-1.0-ultra `-GGUF` link.

# What is new in basal 2.3.2

- **Models with Gated DeltaNet layers** (Qwen3.5, such as the 0.8B pico): `--mode fast`, `fast-nocompile`,
  `fast-exit`, `fp8`, `nvfp4` and `mps` refuse them before their weights are downloaded or loaded, and without
  `--mode` they run in `eager` (CUDA and Apple Silicon). Those engines share a request's state as attention keys and
  values only, so the recurrent state of the Gated DeltaNet layers was lost: 2.3.1 served the pico in `fast` (the
  CUDA default) with wrong answers and no error. The pico served without `--mode` on an H100: `eager`, top answer as
  the trainer's readout on 99.65% of 1,700 held-out decisions in bf16 (99.94% in fp32, max |dp| 0.005). The fast engine for
  these models comes in 2.4.
- **`--mode gguf` on hybrid and recurrent models** (GGUF architecture `qwen35`, Mamba, RWKV, the others llama.cpp
  flags): one prompt per `llama_decode`. On the CPU, llama.cpp returned non-finite probabilities whenever a decode held
  several prompts of such a model (300 of 300 held-out prompts at 16 per call, every request with several questions). The
  pico's Q8_0 now: no non-finite probability, top answer as eager on 99.0% of 300 held-out decisions on 16 CPU threads,
  99.0% of 1,700 on an H100, 99.0% of 300 on an Apple M4 Pro (Metal); five-question requests answered on all
  three. A request's state is computed once per question and option order for these models.
- **`--mode fp8` below compute capability 8.9** (Ampere: RTX 3090, A100) stops before loading the model and suggests
  `--mode fast`; torchao failed after the load before (#33, @krzysztofkaminski).
- **Facts block** (@agentGreg): amounts with a dot thousands separator ("10.000,00 zł", #31); "roku" after a date is
  not a one-year duration (#32).
- Other models are unchanged: a Qwen3 model (attention only) and basal-omni-1.0-mini still default to `fast`, with the same answers
  as 2.3.1 on 200 held-out decisions (the same probabilities on every item, max |dp| 0).

# What is new in basal 2.3.1

- **basal-omni's fast mode on Blackwell workstation GPUs** (sm_120, such as the RTX PRO 6000): `--gemma-attn auto` is
  FlexAttention there, with tiles for their 99 KB of shared memory per block, and basal-omni without `--mode` runs
  `fast` on them as on Hopper. Gate on an RTX PRO 6000: every path of the fp32 identity check 1.000 (mini, 1.0-max),
  dev v2 port gates pass, automated share at 1% with the shipped calibration within 0.07 points of eager's (mini
  -0.07, 1.0-max -0.05), 2.2-4.1x eager on single questions and 14-22x with five over a long state. vLLM stays faster
  for single 1.0-max questions there (8.9 against 5.0 decisions per second).

- **Incremental states** (`--mode fast` with `--state-cache-gb N`): a request whose state begins with a cached state - a
  growing agent trace or chat history, or an edited one up to the edit - computes only its new tokens over the cached
  keys and values, and the grown state replaces the old entry. Same answers as computing the whole state (fp32: every
  top answer, max |dp| 9e-6; bf16: 1.000 at margin >= 0.2; mini, 1.1-max, basal-2.0-4.5B, edited traces included).
  One H100, an agent loop growing to ~10k tokens: step 12 in 0.19 s instead of 0.39 s on basal-2.0-4.5B and 0.27 s
  instead of 0.79 s on basal-omni-1.1-max (16 traces at once: 1.2-1.3x).
- **Long states in chunks:** a state too long for one forward (half the free GPU memory after start-up) is prefilled in
  2,048-token chunks over the keys and values before them, so basal-omni-1.0-ultra answers 76k- and 82k-token states on
  one 80 GB H100 (2.3.0: out of memory beyond ~32k; 82k with 5 questions: 46 s), also while short requests are being
  served; a forward that runs out of memory anyway is redone in chunks, and the server frees the GPU memory of a
  failed request (before, the next requests failed too).
- **Long states and busy servers in `--mode eager`:** basal-omni-1.0-ultra answers 32k- to 82k-token states in eager
  mode too, on one 80 GB H100 (2.3.0: out of memory from 32k on; 82k with 5 questions: 99 s, also while 16 short
  requests at a time are being served). Above 16k tokens the attention runs in smaller pieces; up to 16k nothing
  changes (dev v2: the same probabilities, bit for bit, on mini, 1.1-max and ultra). Ultra in eager also serves 16
  requests at a time now: 2.04 decisions per second without errors, where 2.3.0 failed 452 of 480 requests. It ran out
  of memory holding keys and values nothing read, and a failed request kept its GPU memory until a garbage collection.
- **Faster `--mode eager` for basal-2.0 and other non-Gemma models on CUDA:** no cuDNN attention in eager, as in the
  fast engine. It built a new plan for every new prompt length (0.5 s a request on an H100). basal-2.0-4.5B: one
  request at a time 52 ms instead of 591 ms (median), 16 at a time 7.35 instead of 4.95 decisions per second; dev v2
  as served: the same top answer on 0.9998 of decisions at margin >= 0.2, accuracy 70.24 -> 70.29%. basal-omni
  (Gemma 4) ran without it already.

- **Questions over long states: no first-use stalls, and faster** (`--mode fast`, basal-omni). The second pass (the
  questions over a state) compiled its attention for a new shape class the first time one came up under load, 1.8-2.4
  s with every request in flight waiting (4-5 times in a first run of five questions over 3k-10k-token states); the
  classes are now compiled at start-up. Questions over 3.5k-8k-token states run as captured CUDA graphs instead of
  kernel by kernel (at most 1 GB more GPU memory; 70 MB on mini). One H100, five questions over 3k-10k tokens: mini
  21.9 -> 33.7 questions per second at 16 in flight in a first run, +5-6% at 1 and 32; 1.1-max 10.3 -> 11.8, +3-4%;
  ultra, single questions and basal-2.0 as before (within run-to-run noise); dev v2 as two runs of 2.3.0 differ
  (0.9994-1.0000 of top answers at margin >= 0.2). The server also keeps its start-up objects out of full garbage
  collections (up to 1.4 s each by the end of start-up; one stalled a request for 2.9 s on basal-2.0-4.5B).
- **Less GPU memory for basal-2.0 in the graph engine:** 8 MB cuBLAS workspaces per captured CUDA graph for every model
  (2.3.0: basal-omni only; `CUBLAS_WORKSPACE_CONFIG` unless you set it): 0.6-0.9 GB less GPU memory, 100 s faster
  start-up on basal-2.0-4.5B (452 s instead of 554 s on an H100), the same speed; dev v2 top answers as two runs of
  2.3.0 differ.
- **Faster sentence splitting for `compress`.** With `unit: "sentence"`, the word before each sentence end was searched
  from the start of the text (and retried at every position of a long token), O(n²) per document; it is now read
  backwards from the sentence end (a 98k-character document: 13 s on the event loop before the 600-unit refusal, now
  4 ms). Same units as before. Contributed by Milosz (@Butterski), [#16](https://github.com/rkinas/basal/pull/16).
- **`--mode gguf` uses every physical core.** llama.cpp in process ran its own default of 4 threads; it now takes
  PyTorch's thread count (the physical cores within the process's CPU affinity, or `OMP_NUM_THREADS` /
  `torch.set_num_threads`). basal-2.0-mini Q8_0 on a 6-core CPU: 1.34× faster on 16 recorded requests, probabilities
  bit-identical (each logit is still one dot product in one thread). Builds with every layer on Metal or CUDA are not
  affected. Contributed by Milosz (@Butterski), [#20](https://github.com/rkinas/basal/pull/20).
- **`--mode mps` on the CPU** (`--device cpu`, the default without Apple Silicon): the packed engine (SOAM) computes a
  request's state once for all its questions and option orders, where `--mode eager` computes it once per prompt.
  basal-2.0-mini, `--dtype float32`, a 6-core Zen 2: three task requests (41, 24 and 10 prompts) 838 s -> 88 s, 9.6x,
  the same top answer on 75 of 75 readouts (max |Δp| 4e-6); single questions as fast as in `eager`. `--device cuda` is
  refused (that is `--mode fast`). Contributed by Milosz (@Butterski), [#26](https://github.com/rkinas/basal/pull/26).
- **Status codes and load for process managers.** basal-serve answered every failed request with HTTP 422; now 422
  means an invalid request, 503 that the GPU ran out of memory (retry later or elsewhere) and 500 any other server
  error, with the same `{"error": ...}` body. `GET /health` also returns the requests in flight as `running` and
  `queued` (waiting for the engine), so a manager can tell when the server is idle ([API.md](API.md)). Suggested by
  @edurdias, [#22](https://github.com/rkinas/basal/issues/22).
- **`basal-calibrate --strict` says when it tested nothing.** The family check needs 30 CERT decisions of a type in a
  family (`--min-family`); with fewer in every family it passed without a test. It still passes, but now prints
  "0 families tested" for each such type on stderr and in the report. Reported by @agentGreg,
  [#19](https://github.com/rkinas/basal/issues/19).
- **`basal-calibrate` says up front when a file is too small for a target.** Before any model readout, it checks
  every question type and task against each `--targets` value. When the FIT decisions could not certify the target
  even if all of them were accepted and correct (1% needs 299 without an error, 5% needs 59), it prints a warning with
  the count, the gap and how many more items to label, and repeats it at the end of the run. A threshold certified on
  FIT whose CERT decisions are too few to confirm it (CERT upper bound above the target) is now flagged in the report
  and on stderr; the threshold is kept. Feedback from BRAUN SI / NovaraX.
- **Apple M5 Pro in HARDWARE.md.** `basal-bench` tables for basal-1.0-4.5B and -1.5B on an M5 Pro (20-core GPU): bf16
  is 2.0-2.5x faster than on the 40-core M4 Max and 4.2-5.5x faster than fp32, so two sentences written for the M4
  Max (latency scaling with the core count, fp32 close to bf16) now say that they do not hold for the M5 chips.
  Measured by Kamil (@kamilrosa), [#7](https://github.com/rkinas/basal/issues/7).
- **`--mlx-packed`: two option orders in one forward on Apple Silicon** (opt-in, `--mode mlx`, basal-1.x and
  basal-2.0). Since 1.5.0 the MLX backend computed the prefix shared by a question's option orders in one forward and
  the option blocks in a second, and every forward reads all the weights. With `--mlx-packed` a group of at most 512
  tokens is one packed row with explicit positions and a block mask, and the rows of several requests share a forward
  (at most 2,048 tokens each). Two orders: basal-1.0-4.5B 118 -> 68 ms on an M5 Max, basal-1.0-1.5B 158 -> 130-136 ms
  on an M4 Pro; the same decisions as the default path at margin >= 0.2. It stays opt-in because the automated share
  at 5% of basal-2.0-4.5B-MLX-8bit's shipped calibration moved by -0.20 pt [-0.60, +0.20] (QUANTIZATION.md).
  Root cause in [#27](https://github.com/rkinas/basal/issues/27), contributed by Grzegorz Brzezinka (@agentGreg),
  [#28](https://github.com/rkinas/basal/pull/28).
- **`--lean on --tree flex` is refused before the model is resolved** also on CUDA GPUs without FlashAttention 3,
  where choosing the default mode resolved the model first (offline: a download error instead of the refusal;
  `tests/test_lean.py` failed on those GPUs). Contributed by Milosz (@Butterski),
  [#30](https://github.com/rkinas/basal/pull/30).
- **Compact rows on any sm80+ GPU** (`--batches compact`; `auto` unchanged): single prompts packed back to back without
  padding ran only with FlashAttention 3 (Hopper); elsewhere they now run through torch's varlen attention (its
  FlashAttention 2), with the template prefix. basal-2.0-mini on an RTX 4060 Ti 16 GB: single questions 66.8 -> 109.4
  decisions/s (+64%), the same top answer on 58 of 58 answers; about 2.4 GB more GPU memory, and five questions over
  ~1.5k tokens (unchanged path) 5.6% slower there with the card full. On an H100 with the kernel forced
  (`--attn fa2`): single questions +16% (basal-2.0-mini and 4.5B), five questions unchanged, 5.8-7.2 GB more GPU
  memory, the same top answer on every dev v2 decision with margin >= 0.2 ([SERVING.md](SERVING.md)); `auto` gave
  bit-identical dev v2 answers there. On a Blackwell RTX PRO 6000 (sm_120, no FlashAttention 3, so `auto` runs no
  compact rows): single questions +18% (mini) and +13% (4.5B), five questions unchanged, the same top answer on 7,632
  of 7,632 and 9,047 of 9,049 dev v2 decisions with margin >= 0.2. The start-up line names the compact kernel.
  Contributed by Milosz (@Butterski), [#29](https://github.com/rkinas/basal/pull/29).
- **Also from the community:** the calibration report's per-type ECE as `basal-serve` applies the temperatures
  ([#24](https://github.com/rkinas/basal/pull/24)), the per-type label count for the 1% target in CALIBRATION.md
  ([#25](https://github.com/rkinas/basal/pull/25)) and the Apple M5 Max benchmark tables in HARDWARE.md
  ([#18](https://github.com/rkinas/basal/pull/18)), by Grzegorz Brzezinka (@agentGreg); `--mode strix` reproduced on a
  second Strix Halo (ROG Flow Z13), bit-identical edge cases and a similar speed-up, by @piotrkowski
  ([#21](https://github.com/rkinas/basal/pull/21)).

# What is new in basal 2.3

- **vLLM: KV-sharing fast prefill for basal-omni-1.0-mini, 1.43x / 1.60x** (`--mode vllm`, on by default for Gemma 4
  E2B / E4B; `--vllm-fast-prefill off` restores 2.2). Its last 18 of 42 layers reuse earlier layers' keys and values,
  so during prefill they run at the answer position only. One H100, against 2.2's vLLM path: 60.8 vs 41.5 decisions/s
  for one question over a ~1.5k-token state, 39.0 vs 24.2 questions/s for five questions over 3k-10k tokens; the same
  top answer as eager on 0.9994 of dev v2 decisions with margin >= 0.2, automation at 1% with the shipped calibration
  within 0.2 points of eager. [OMNI.md](OMNI.md#vllm---mode-vllm)
- **vLLM: images and audio for basal-omni-1.0-mini** (`--mode vllm --media on`): the same prompts and media tokens as
  `--mode eager --media on`, all prompts of a request in one vLLM call. On the omni media check set: 96.5% / 98.3% the
  same top answer as the eager media path (all where eager's margin >= 0.2), 6.0x / 7.9x the requests per second
  (images / audio, 8 in flight). basal-omni-1.1-max is refused (its images and audio did not match the eager path).
- **vLLM: `--vllm-preset deterministic`** for repeatable answers: batch-invariant kernels without the prefix cache and
  chunked prefill; two runs of basal-omni-1.1-max at 32 in flight gave bit-identical probabilities. Costs throughput (1.1-max 0.64x single questions, 0.22x five questions over a long state); automation
  at 1% with each model's shipped calibration within 0.2 points of eager. Presets load `CALIBRATION.vllm.<preset>.json`
  when a model ships one.
- **`basal-loadtest --seed`:** synthetic states from another seed, so a repeated load test does not answer from the
  prefix cache of an earlier one.

- **basal-omni in `--mode eager` computes a request's state once for all its questions.** The eager backend runs
  Gemma 4 through the engine's own forward (`basal/gemma4.py`, transformers' modules and weights): a request's state
  and media run once and every question and option order runs over them, the image and audio encoders once per
  request, and mini's 18 KV-sharing layers only where the answer is read. On one H100 at 16 requests in flight, five
  questions over a 3k-10k-token state: mini 1.6 -> 15.3 questions/s, max 0.7 -> 4.5; one question: mini 10.0 -> 17.2
  decisions/s. Agreement with the 2.2 forward on dev v2: 1.000 (mini) and 0.9997 (max) on decisions with margin >=
  0.2. `--eager-forward hf` keeps the 2.2 path. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **`--eager-attn auto` (opt-in, basal-omni in `--mode eager` on Hopper):** FlashAttention 3 in the sliding-window
  layers and the memory-efficient kernel (keys and values copied per head) in the 512-wide global layers. One H100 at
  16 in flight, against the default math kernel: mini 17.2 -> 24.4 decisions/s and 15.3 -> 19.0 questions/s over long
  states, max 5.0 -> 8.7 and 4.5 -> 7.6. Top answers equal to the 2.2 forward on 0.9995 (mini) and 0.9991 (max) of the
  dev v2 decisions with margin >= 0.2.
- **`--ple-offload on` (basal-omni mini, `--mode eager`):** the 5.6 GB per-layer embedding table stays in host memory
  and only the rows of each forward's tokens go to the GPU: 5.2 GB less GPU memory at about 4% fewer decisions per
  second, the same answers. For mini on smaller GPUs.
- **`--mode eager` avoids a torch attention bug.** Before torch 2.14.1, SDPA's memory-efficient kernel returns wrong
  attention for keys and values expanded from one key / value head at some query lengths (pytorch/pytorch#191937):
  transformers passes such a view with an attention mask, and basal-omni-1.0-max was off by 6.2 in code log-prob on a
  1,857-token prompt. The eager backend now runs every model's transformers forward without that kernel on CUDA with
  older torch (2.2 did so for Gemma 4 only).

- **basal-omni in the CUDA graph engine, 3-6x 2.2's eager** (`--mode fast`; without `--mode`, the default for basal-omni on
  Hopper GPUs with FlashAttention 3; text). Gemma 4 runs
  through the engine's own layer code: SOAM's packed row and two passes over its sliding-window and global layers (the
  window is a condition on positions inside the trie mask), key / value buffers per layer type, E4B's KV-shared layers
  and the last own-key layer at the answer rows only, the fp32 readout with the softcap. On Hopper: FlashAttention 3 in
  the sliding layers, FlexAttention in the 512-wide global ones, merged projections, compact rows, compiled; elsewhere
  the math kernel of `--mode eager` (`--gemma-attn`). One H100, against 2.2's `--mode eager`: 1.1-max 16.2 vs 4.8 decisions
  per second on single questions (vLLM bf16: 16.7) and 10.6 vs 0.72 questions per second with five questions over a long
  state; mini 61.2 vs 10.0 and 21.4 vs 1.6; ultra 6.1 vs 2.0 and 4.8 vs 0.28. Same answers within bf16 rounding (dev
  v2, 10,606 decisions: 0.998-0.999 of the top answers at margin >= 0.2; with the shipped calibration files the
  automated share at 1% within 0.1 points). Without `--mode`, basal-omni runs `eager` on other GPUs (in 2.2 it stopped
  at start-up). On an RTX PRO 6000 Blackwell (`--mode fast --gemma-attn flex`, opt-in until its calibration
  check): mini 23.2 vs 5.6 and 18.4 vs 0.85, 1.0-max 5.0 vs 2.3 and 5.3 vs 0.37 (vLLM is faster there for single max
  questions). [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **Less GPU memory for basal-omni in the graph engine:** 8 MB cuBLAS workspaces per captured graph
  (`CUBLAS_WORKSPACE_CONFIG` unless set; 32 MB each by default), no image / audio towers (text only), and long states
  keep only the sliding layers' last window of keys and values (ultra's 10k-token states ran out of memory on 80 GB).

- **Faster preparation of large requests.** The prompt language is detected from the state once per request in task
  expansion and once in question preparation, not once per question: an English state was re-scanned for Polish
  diacritics for every question (a 40k-character `compress` request with 326 questions: 509 ms, now 2 ms; 50 task
  questions: 79 ms, now 2 ms in task expansion). Same languages as before. Contributed by Milosz (@Butterski),
  [#17](https://github.com/rkinas/basal/pull/17).

Release notes: [v2.3.0](https://github.com/rkinas/basal/releases/tag/v2.3.0).

# What is new in basal 2.2

**Engine v2.2.0: calibration on your own data, for every basal model**

Label a few hundred of your own requests, run `basal-calibrate`, and serve the file it writes. The probabilities, and
the `max_error` thresholds that decide what may be automated, are then certified on your data, not ours. A hosted
decision API cannot be recalibrated on your data; a basal model can.

```bash
basal-calibrate my_labelled.jsonl --model Remek/basal-omni-1.0-max --mode eager    # writes CALIBRATION.custom.json
basal-serve --model Remek/basal-omni-1.0-max --mode eager --calibration CALIBRATION.custom.json
```

- **Task thresholds from your labels.** `basal-calibrate` now takes labelled `classify`, `verify` and `extract`
  questions and certifies the `automated` flag of their answers (`task_thresholds`), by the same bound as the decision
  types, on the answers as served. A file with task questions only keeps the decision calibration and adds the task
  thresholds. [CALIBRATION.md](CALIBRATION.md#tasks-classify-verify-extract)
- **`extract` fields report `automated`**, like `classify` and each `verify` claim.
- **A few dozen labels are enough to start.** Below `--min-per-type` decisions, a type's temperature is blended with
  the shipped one in proportion to the labels you have (`--min-blend`, from 10), instead of being left as shipped.
- **Your data, measured as served.** The report shows how the file you would otherwise serve (the shipped one, or
  `--calibration FILE`) does on your held-out decisions: the share it automates at each target and its error, each
  type at the threshold basal-serve applies (2.1 compared the pooled threshold, which basal-serve does not apply to
  prompt-format-2 models). `--report-only` prints the report without writing a file.
- **A file knows its model.** `fitted_on` records the model and its Hugging Face revision; `basal-serve --calibration`
  warns when a file was fitted on another model or revision (and reports it in `GET /v1/models`), and
  `--strict-calibration` refuses it.
- **Measured on a new domain** (basal-omni-1.0-mini on typed-decisions, test split): the shipped file's 1% setting
  would automate 12.6% of decisions at 8.0% error there; calibrated on the train split, the calibration error falls
  from 0.268 / 0.106 / 0.104 to 0.083 / 0.073 / 0.052 (choice / yes-no / score) and nothing is automated above its
  target. [The example](CALIBRATION.md#example-a-new-domain-measured)
- **Fix:** a looser `max_error` target no longer keeps a stricter threshold than a tighter one (the scan could stop
  early at 5% and automate fewer decisions than at 1%); also in the task thresholds.
- **basal-omni on GGUF and MLX (text only).** `--mode gguf --gguf <file>` runs a GGUF conversion of basal-omni-1.0
  through llama.cpp (Metal, CUDA or CPU; llama-cpp-python 0.3.36 or newer) and `--mode mlx` / `mlx-q8` the bf16
  weights through mlx-lm on Apple Silicon. Q8_0 and Q6_K pass the port gate against bf16; Q5_K_M and Q4_K_M do not.
  `--mode gguf` loads `CALIBRATION.gguf.json` when the model ships one. [OMNI.md](OMNI.md#gguf-llamacpp-ollama-and-mlx-text)
- **New model repositories:** `-GGUF` (Q8_0, Q6_K) and `-FP8` (`--mode vllm`) of basal-omni-1.0-mini and -max, each with
  a calibration fitted on the port's own readouts (GGUF: on Q8_0); 8-bit `-MLX` ports (`--mode mlx`,
  `CALIBRATION.mlx.json`) of 1.0-mini, 1.0-max and 1.1-max. `VLLM_BATCH_INVARIANT=1` gives repeatable answers
  in `--mode vllm` ([OMNI.md](OMNI.md#serve)).
- **From the community.** `basal-export-gguf` (Piotr Durlej, #8; extra `basal[gguf-export]`) converts a basal model
  to GGUF, or repairs an existing GGUF file, with the tokenizer fix of the `-GGUF` files, so that llama.cpp and Ollama
  tokenize like the Hugging Face tokenizer ([GGUF.md](GGUF.md#convert-yourself)). Two experimental,
  community-maintained engines for the Llama-architecture models (basal-1.5, basal-2.0), both opt-in: `--mode tpu`
  (Kuba Sołtys, #15; extra `basal[tpu]`) in JAX / XLA on a Google TPU, and `--mode strix` (Michał Kardyś, #14) with
  Triton kernels for AMD Strix Halo. Their numbers in [HARDWARE.md](HARDWARE.md) are the contributors' measurements.
  `basal-viz` (@lukaspoloki, #2) opens a local page that shows every step of a decision, next to the served answer
  ([SERVING.md](SERVING.md#see-how-a-decision-is-made)).
- Every other request takes the same code path as in v2.1.0.

Release notes: [v2.2.0](https://github.com/rkinas/basal/releases/tag/v2.2.0).

# What is new in basal 2.1

**Engine v2.1.0: task types (experimental)**
- Four task types on the same API, for the basal models you already serve (no new model): **`classify`** (labels with
  a rubric each), **`verify`** (is each statement supported, contradicted or not stated by the state), **`compress`**
  (the sentences, lines or paragraphs needed for a question, as shorter text for another model) and **`extract`** (a
  JSON schema filled with values taken exactly from the document, with character offsets).
- Each task is expanded into the decision types over the shared state, so it costs about as much as the decisions it
  asks: no text is generated, and `extract` values are always exact text of the document.
- Measured on public sets the models were not trained on, with basal-omni-1.0-max: see
  [TASKS.md](TASKS.md#results). Experimental: the API may change, and the `automated` flag of a task answer needs
  thresholds fitted for that task.
- Every other request takes the same code path as in v2.0.0 (on basal-omni-1.0-max, 23,276 of 23,278 served-check
  answers were the same; the two others were near-ties).

**New models (Hugging Face, Sat 17 Oct, 09:00 CEST)**
- **basal-omni-1.0-ultra** (31B, Gemma 4 31B; text and images): Werdykt 0.879, the highest of the basal models; and
  **basal-omni-1.0-ultra-FP8** for `--mode vllm` (text only).
- **basal-omni-1.1-max** (*experimental*): basal-omni-1.0-max with extra training on the task formats, better on the
  tasks and slightly behind 1.0-max on general decisions; 1.0-max stays the default ([TASKS.md](TASKS.md#results)).

Release notes: [v2.1.0](https://github.com/rkinas/basal/releases/tag/v2.1.0).

# What is new in basal-2.0


**Models**
- A family of three: **basal-2.0-mini** (1.5B, fastest), **basal-2.0** (4.5B, main model, with an experimental
  thinking mode) and **basal-2.0-max** (11B, most accurate). Weights in bf16, plus **FP8** (NVIDIA Hopper and
  Blackwell), **GGUF** (Ollama, llama.cpp) and **MLX 8-bit** (Apple Silicon).
- **Prompt format 2**: up to **255 options per question** (basal-1.5: 10), read in one forward pass with **one option
  order** by default (basal-1.5 read two orders). See [FORMATS.md](FORMATS.md).
- On **Werdykt**, the hidden Polish–English decision benchmark, every tier is above its basal-1.5 counterpart (+1.2 /
  +0.7 / +1.1 points for mini / 4.5B / max) with a lower calibration error (0.035 / 0.060 / 0.044 against
  0.086 / 0.068 / 0.074), and faster. Not everything improved: on a panel of 10 public English decision tasks all three
  are below basal-1.5 (−2.1 / −3.6 / −1.8 points). All numbers: [results against basal-1.5](RESULTS.md#results-against-basal-15).
- **Calibration** with a temperature per question type and option-count bucket, fitted on question families the model
  was not trained on. It is cautious on new tasks: the shipped files certify a 5% error target but no 1% target (one
  port file excepted), so `max_error: 0.01` defers every decision ([using the confidence](API.md#using-the-confidence)).
- **Thinking mode** (4.5B, *experimental, off by default*): `--think auto` writes a short reasoning note for uncertain
  questions before deciding. On Werdykt it lowered accuracy and calibration and cut throughput; we recommend leaving it
  off ([thinking mode](THINKING.md#thinking-mode-experimental)).

**basal-omni-1.0 (experimental): images and audio**
- Two models on Google Gemma 4: **basal-omni-1.0-mini** (4B, Gemma 4 E4B) and **basal-omni-1.0-max** (12B). They answer
  the same typed questions over the same API, and with `--media on` a request may also carry **images and audio**
  (`"media"`).
- On Werdykt: 0.769 (mini) and 0.823 (max), against 0.721 for basal-1.5 (4.5B) and 0.773 for basal-1.5-max.
- Trained on text decisions only: Gemma 4's own encoders read the images and audio zero-shot, and media answers use
  the text calibration. `--mode eager` only for now. See [images and audio](INSTALL.md#images-and-audio-basal-omni-experimental)
  and [OMNI.md](OMNI.md).

**Engine v2.0.0**
- **Faster serving on one GPU**: compact batching (single prompts packed into one row with FlashAttention 3 varlen),
  the prompt template computed once at start-up, a scheduler that keeps long requests from delaying short ones,
  merged projections and FlashAttention 3 on Hopper, native ModelOpt FP8 checkpoints in `--mode fp8`.
- **One engine for text and media**: the eager backend serves the basal-omni models (Gemma 4 architecture) and, with `--media on` and
  the `basal[media]` extra, requests with images and audio (`"media"`, `"detail"`, `"lang"`). In every model it now
  reads option codes in fp32, reads left-padded prompts at their own positions and prefills long prompts in
  2,048-token chunks.
- **New question type `tool`**: the next tool (or answer directly / ask the user / no suitable tool) from OpenAI, MCP
  or Anthropic tool definitions, also as a set of tools (*experimental*).
- **MCP for agents** (*experimental*): `basal-serve --mcp on` also serves the decisions as an MCP server at `/mcp`
  (tools `decide` and `decide_many`), and `basal-mcp` connects a running server to agent hosts over stdio
  ([MCP](MCP.md#mcp-basal-as-a-tool-for-agents)).
- **`multi`** gains set objectives (`iou`, `f1`, `exact`) and `exclusive` / `fallback` labels.
- **`basal-calibrate`**: fit the temperatures and certified confidence thresholds on your own labelled requests.
- **`--deterministic`** serving (answers independent of load), **`--orders 1 | 2 | auto`**, long states truncated in
  the middle instead of refused (`--context-overflow`), option-code logits in fp32 in every backend.
- The engine still serves basal-1.5 and basal-1.0 models (prompt format 1) with the same API.

Full list: [release notes](https://github.com/rkinas/basal/releases/tag/v2.0.0). Details of the experimental features:
[FEATURES.md](FEATURES.md).

# Previous releases: basal-1.5 and basal-1.0

**basal-1.5** (October 2026; mini 1.5B, 4.5B, max 11B, with FP8 / NVFP4, MLX 4-bit / 8-bit and GGUF ports) remains
available on Hugging Face ([collection](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20))
and is served unchanged by this engine (`--model Remek/basal-1.5-4.5B`; two option orders by default). Its release
README, with the full Werdykt board of that release: [v1.5.0](https://github.com/rkinas/basal/tree/v1.5.0).

**basal-1.0** (September 2026; 4.5B and 1.5B, plus FP8 / NVFP4 ModelOpt checkpoints) remains available on Hugging Face
(`Remek/basal-1.0-4.5B`, `Remek/basal-1.0-1.5B`). Its full tables, methods and limitations are in the
[basal-1.0 technical report](basal-1.pdf) and the [v1.0.1 README](https://github.com/rkinas/basal/tree/v1.0.1).
The engine serves basal-1.0 models unchanged; `fast-exit` and the `vllm` mode with the ModelOpt checkpoints work as
before. The Apple Silicon engine modes (`mps`, `gguf`, `mlx-q8`) and the Apple Silicon engine study were contributed
by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)).
