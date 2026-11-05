# Contributing to basal

How to work on basal: the development commands and checks, how a pull request adds its entry to the release notes,
and who contributed what, by pull request.

- [Development commands](#development-commands) · [Development checks](#development-checks) ·
  [Release notes](#release-notes) · [Contributors, by pull request](#contributors-by-pull-request)

## Development commands

Run `make` or `make help` to list commands. The Makefile uses `.venv/bin/python` by default; override `PYTHON` to
use another prepared environment. Create it with `uv venv --python 3.12 .venv`, then install the hardware-specific
PyTorch build described in [Quick start](README.md#quick-start) before installing the app. For example:

```bash
make setup                         # editable app plus test dependencies
make setup EXTRAS=test,mlx,gguf     # Apple Silicon; install only the backends you need
make check                         # read-only pre-commit checks, no Git hook installation
make test                          # app tests; requires dependencies and applicable backend hardware
```

`setup` uses `uv pip install` in that environment and may install required dependencies. Launch and test targets
call Python directly: they never sync the environment, remove extras, or replace your selected PyTorch build.
Use `EXTRAS=test,fp8` or `EXTRAS=test,nvfp4` for the matching CUDA modes; keep vLLM in its own environment because
it brings its own PyTorch requirements. MLX requires Apple Silicon; MPS requires a supported Mac; CUDA graph and
torchao modes require compatible NVIDIA hardware. See [Serving modes](docs/SERVING.md#serving-modes) for backend limitations.

```bash
make serve                                      # automatic backend selection
make serve MODE=mlx MODEL=Remek/basal-2.0-mini-MLX-8bit
make serve MODE=fast-nocompile PORT=8001
make serve MODE=fast-exit ARGS='--early-exit 0.99'
make serve MODE=gguf GGUF='/path with spaces/model.gguf'
make serve MODE=ollama OLLAMA_MODEL=basal         # existing Ollama safetensors import
make run INPUT=basal/examples/questions.jsonl OUTPUT=answers.jsonl
make bench                                      # CLI-selected reference and serving modes
make bench ARGS='--modes eager-fp32 mps mlx --n 44'
make loadtest INPUT=basal/examples/questions.jsonl ARGS='--n-seq 20 --n-conc 100'
```

`MODE` selects one backend for `serve` or `bench`; omit it to retain the CLI's automatic defaults, as an empty `MODEL`
keeps their default model. The benchmark
also accepts its native modes such as `eager-fp32` and `fast-exit@0.99`. For multiple benchmark modes, use `ARGS`
as above; provide `GGUF` and `OLLAMA_MODEL` when the list includes those backends. GGUF still uses `MODEL` for
tokenizer/calibration, and Ollama requires a running service and a compatible import; see [hardware docs](docs/HARDWARE.md).
Required GGUF/Ollama values are checked early when selected through `MODE`; flags supplied through `ARGS` are
handled by the application. Serving or benchmarking can download and load models.

`INPUT` and `OUTPUT` are required for `run`; they are optional for `bench` and `loadtest` (which otherwise use
bundled simple questions). Output files are overwritten by the CLI; an `OUTPUT` that is the `INPUT` file is refused
before anything runs. HTTP clients use
`URL=http://127.0.0.1:8000/v1/systemone` by default, follow `PORT` overrides, and require a running server.
`make serve` listens on `HOST=127.0.0.1` (this machine only); `HOST=0.0.0.0` exposes the unauthenticated server on
every interface.
Quote variable values containing spaces. `ARGS` is appended as shell command text for additional CLI options, so
pass only trusted text (as with any Make variable, it runs in the shell); use shell quoting inside it when an option
value contains spaces. Inspect commands without launching with `make -n`.

## Development checks

The [pre-commit configuration](.pre-commit-config.yaml) provides fast, read-only checks: Ruff with the rules of
`pyproject.toml` (`E`, `F`, `W`), the same version CI runs over the whole repository (0.15.16), merge conflicts,
private keys, TOML, YAML, JSON, and JSONL. It covers the engine, server, clients, benchmarks, examples, tests, tools,
notebooks, and figure scripts. JSON checks include the
checked-in OpenAPI fixture; JSONL checks parse each nonblank line, matching the batch runner. These are syntax
checks, not request or response schema validation; the latter remains in `tests/test_server.py`.

Use [uv's isolated tools](https://docs.astral.sh/uv/guides/tools/) without installing basal or its ML dependencies:

```bash
uvx pre-commit==4.7.0 validate-config
uvx pre-commit==4.7.0 run --all-files
python3 -m unittest discover -s tests -p test_check_jsonl.py
```

Hook versions are pinned in the configuration. The first run downloads the hook tools into isolated environments;
subsequent runs reuse them. The hooks do not format files, start a server, download models, or run inference tests.
Run the application tests separately when changing application behavior.

To opt into checks on staged files before each commit, install the tool persistently and then install the Git hook:

```bash
uv tool install pre-commit==4.7.0
pre-commit install
```

Git worktrees share their hooks directory: installing the hook affects sibling worktrees too. Use manual commands
above if that is unwanted. To remove the installed Git hook, run `pre-commit uninstall`.

## Release notes

A pull request adds its entry for [docs/WHATS_NEW.md](docs/WHATS_NEW.md) as a fragment, `docs/changes/<PR number>.md`,
instead of editing WHATS_NEW.md itself; at release, `python scripts/changelog.py <version>` puts the fragments at the
top of WHATS_NEW.md. How to write one: [docs/changes/README.md](docs/changes/README.md). The README keeps only the
latest release's "What is new" section (at most 8 bullets, with a link to WHATS_NEW.md); the previous one is replaced,
not kept.

## Contributors, by pull request

Thank you to the people whose work is in this repository, by pull request:

- Paweł Kiszczak ([@pawelkiszczak](https://github.com/pawelkiszczak)): the Apple Silicon engines `--mode mlx` / `mlx-q8`
  and `--mode mps`, the in-process GGUF engine `--mode gguf` (llama-cpp-python) and `--mode ollama`, and the comparison
  of Apple Silicon engines and quantised formats (MLX, oMLX oQ, GGUF) with its figures in
  [HARDWARE.md](docs/HARDWARE.md#apple-silicon) ([#4](https://github.com/rkinas/basal/pull/4)); the development commands
  (`make`) and read-only pre-commit checks ([#46](https://github.com/rkinas/basal/pull/46)); LoRA adapters merged at
  load (`--adapter`, [#52](https://github.com/rkinas/basal/pull/52)), a `code` in error responses
  ([#62](https://github.com/rkinas/basal/pull/62)), the certified thresholds and the context in `GET /v1/models`
  ([#64](https://github.com/rkinas/basal/pull/64)), MCP model selection
  ([#66](https://github.com/rkinas/basal/pull/66)), `--api-keys`, `--cors-origins` and `x-request-id`
  ([#68](https://github.com/rkinas/basal/pull/68)), `basal-calibrate` on a Hugging Face cache snapshot
  ([#71](https://github.com/rkinas/basal/pull/71)), and `basal-viz` with `"facts": "auto"`
  ([#73](https://github.com/rkinas/basal/pull/73)) and on Gemma 4 basal-omni text decisions
  ([#75](https://github.com/rkinas/basal/pull/75)).
- Piotr Durlej ([@pdurlej](https://github.com/pdurlej)): `basal-export-gguf`, which converts a basal model to GGUF, or
  repairs an existing GGUF file, so that its tokenizer matches the Hugging Face one, as in the official `-GGUF` files
  ([#8](https://github.com/rkinas/basal/pull/8)).
- Kuba Sołtys ([@qooba](https://github.com/qooba)): the experimental Google TPU engine `--mode tpu` in JAX / XLA, with
  its Colab notebook ([#15](https://github.com/rkinas/basal/pull/15)).
- Michał Kardyś ([@kardysm](https://github.com/kardysm)): the experimental `--mode strix`, a basal-1.5 / basal-2.0
  forward with Triton kernels for AMD Strix Halo, and its benchmarks
  ([#14](https://github.com/rkinas/basal/pull/14)); a start-up that compiles every Triton kernel copy, so no request
  waits for one ([#51](https://github.com/rkinas/basal/pull/51)).
- [@lukaspoloki](https://github.com/lukaspoloki): `basal-viz`, a local page that shows every step of a decision, from
  the prompt to the per-layer readout, the calibration and the decision
  ([#2](https://github.com/rkinas/basal/pull/2)).
- Miłosz Kucharski ([@Butterski](https://github.com/Butterski)): the prompt language is detected once per request
  instead of once per question, so a large task request on a long English state is prepared in milliseconds instead
  of half a second ([#17](https://github.com/rkinas/basal/pull/17)); sentence splitting for `compress` in linear time
  ([#16](https://github.com/rkinas/basal/pull/16)); `--mode gguf` on every physical core
  ([#20](https://github.com/rkinas/basal/pull/20)); `--mode mps` on the CPU, so a request's state is computed once
  there too ([#26](https://github.com/rkinas/basal/pull/26)), and its branch pass without dense masks
  ([#34](https://github.com/rkinas/basal/pull/34)); compact rows on any sm80+ GPU through torch varlen attention
  ([#29](https://github.com/rkinas/basal/pull/29)); `--lean` refused on `--tree flex` before the model loads
  ([#30](https://github.com/rkinas/basal/pull/30)); and less GPU memory in the graph engine: shared graph outputs
  ([#35](https://github.com/rkinas/basal/pull/35)) and graphs captured largest first, so compact rows fit 16 GB and
  24 GB cards ([#37](https://github.com/rkinas/basal/pull/37)), for basal-omni too
  ([#40](https://github.com/rkinas/basal/pull/40), the same change as 2.3.4's, written in parallel); the questions over
  a long state on CUDA without dense masks, so `compress` runs up to 1.5x faster
  ([#39](https://github.com/rkinas/basal/pull/39)); and basal-2.0-4.5B on 16 GB cards, with pass-2 graphs captured only
  while GPU memory stays free ([#41](https://github.com/rkinas/basal/pull/41)); torch.compile's cache kept across
  reboots ([#47](https://github.com/rkinas/basal/pull/47)), less host work in the CUDA cascade
  ([#44](https://github.com/rkinas/basal/pull/44)) and `--compile-scope layer`
  ([#49](https://github.com/rkinas/basal/pull/49)); in 2.5.4 `basal-prepare` building the engine `basal-serve`
  runs ([#106](https://github.com/rkinas/basal/pull/106)) and the RTX 4090 measured with basal-2.0
  ([#111](https://github.com/rkinas/basal/pull/111)).
- Grzegorz Brzezinka ([@agentGreg](https://github.com/agentGreg)): the calibration report's per-type ECE as
  `basal-serve` applies the temperatures ([#24](https://github.com/rkinas/basal/pull/24)), the per-type label count for
  the 1% target in CALIBRATION.md ([#25](https://github.com/rkinas/basal/pull/25)), the Apple M5 Max benchmark tables in
  [HARDWARE.md](docs/HARDWARE.md) ([#18](https://github.com/rkinas/basal/pull/18)), and `--mlx-packed`, two option
  orders in one forward on Apple Silicon, with its root cause ([#27](https://github.com/rkinas/basal/issues/27),
  [#28](https://github.com/rkinas/basal/pull/28)), and two fixes to the facts block: amounts with a dot thousands
  separator ([#31](https://github.com/rkinas/basal/pull/31)) and "roku" after a date, which is not a one-year duration
  ([#32](https://github.com/rkinas/basal/pull/32)); the float32 MLX tests without TF32 on Apple M5
  ([#76](https://github.com/rkinas/basal/pull/76)) and `basal-viz` on the MLX engine
  ([#78](https://github.com/rkinas/basal/pull/78)).
- [@piotrkowski](https://github.com/piotrkowski): `--mode strix` reproduced on a second Strix Halo (ROG Flow Z13),
  bit-identical edge cases and a similar speed-up ([#21](https://github.com/rkinas/basal/pull/21)).
- [@lukaszpodgorski-pl](https://github.com/lukaszpodgorski-pl): basal on Windows: kernel checks for torch builds without
  FlashAttention or Triton, and UTF-8 for every text file ([#38](https://github.com/rkinas/basal/pull/38)).
- Marcin Brzeżański ([@marcinbrzezanski](https://github.com/marcinbrzezanski)): Prometheus metrics: the engine version,
  the state cache's evictions and prefix hits, and a cache read that never waits for the engine
  ([#48](https://github.com/rkinas/basal/pull/48)); a state cache that keeps the branches of shared histories
  ([#56](https://github.com/rkinas/basal/pull/56)).
- Grzegorz Stark ([@gstark0](https://github.com/gstark0)): `--state-cache-gb` and `--prefix-cache` on Apple Silicon
  (`--mode mlx`), with the guard against MLX's fused attention losing precision for a few queries over a cache
  ([#50](https://github.com/rkinas/basal/pull/50)).
- Dariusz Koryto ([@dkoryto](https://github.com/dkoryto)): continuous integration, ruff and the CPU tests on Linux and
  macOS for every pull request ([#53](https://github.com/rkinas/basal/pull/53)); a Python client for every question
  type, with `AsyncBasal` ([#58](https://github.com/rkinas/basal/pull/58)); a question named `<task>::<part>` in a
  request with task questions refused instead of misread ([#59](https://github.com/rkinas/basal/pull/59));
  `basal-interpolate` ([#60](https://github.com/rkinas/basal/pull/60)); changelog fragments in `docs/changes`
  ([#55](https://github.com/rkinas/basal/pull/55)); and in 2.5.3 `basal-run --preset` fixed
  ([#91](https://github.com/rkinas/basal/pull/91)), batch flags measured on the DGX Spark
  ([#94](https://github.com/rkinas/basal/pull/94)), tests independent of their order, GPU, torch version and runner
  speed ([#86](https://github.com/rkinas/basal/pull/86), [#87](https://github.com/rkinas/basal/pull/87),
  [#88](https://github.com/rkinas/basal/pull/88)) and the CI on two workers
  ([#89](https://github.com/rkinas/basal/pull/89)).
- Rafal Wiktorowicz ([@onlypositivetravelvibes-dotcom](https://github.com/onlypositivetravelvibes-dotcom)): the 2.5.2
  fixes: the start-up 503 for every HTTP method ([#80](https://github.com/rkinas/basal/pull/80)), config probes that
  say why they fell back ([#81](https://github.com/rkinas/basal/pull/81)), `basal-serve` on native Windows and
  `--quant int8` ([#82](https://github.com/rkinas/basal/pull/82)), the provenance and flags of calibration files
  ([#83](https://github.com/rkinas/basal/pull/83)) and HTTP 422 for an unknown question type
  ([#84](https://github.com/rkinas/basal/pull/84)); in 2.5.3 Windows paths and `basal-attach` without symlink rights
  ([#85](https://github.com/rkinas/basal/pull/85)) and `basal-bench` timing HTTP engines on prompts their cache has
  not seen ([#90](https://github.com/rkinas/basal/pull/90)); in 2.5.4 `basal-loadtest`, `basal-calibrate` and
  `basal-agree` on inputs they failed on ([#102](https://github.com/rkinas/basal/pull/102)), malformed requests
  refused before the run ([#103](https://github.com/rkinas/basal/pull/103)) and request text that cannot forge a
  turn ([#104](https://github.com/rkinas/basal/pull/104)).

Measurements in the docs, from issues: Kamil ([@kamilrosa](https://github.com/kamilrosa)), Apple M5 Pro numbers for
basal-1.0, basal-2.0 and basal-omni ([#7](https://github.com/rkinas/basal/issues/7),
[#36](https://github.com/rkinas/basal/issues/36)), and [@krzysztofkaminski](https://github.com/krzysztofkaminski), the
first Ampere numbers on an RTX 3090 ([#33](https://github.com/rkinas/basal/issues/33)).

`--mode tpu` and `--mode strix` are community-maintained: their numbers in [HARDWARE.md](docs/HARDWARE.md) are the
contributors' own measurements.
