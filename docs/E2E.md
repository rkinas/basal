# End-to-end tests of basal-ui

The release gate of basal-ui. `tests/e2e/` starts a real basal-ui (`python -m basal.ui.app` on a free port, its own
settings directory under pytest's temporary directory), loads real models from your models directory into real
`basal-serve` children, and drives everything a user can do: basal-ui's HTTP API, its API server (keys, both request
shapes, MCP) and the pages in Chromium. The unit tests (`tests/test_ui_*.py`) cover the same routes with stubbed
children; these tests are what catches a broken model start, a page script error or an engine that disagrees.

They are skipped unless `BASAL_E2E=1`, so `pytest` stays fast.

```bash
uv sync --extra ui --extra test --extra e2e      # plus mlx / vllm / media as on the machine under test
uv run playwright install chromium               # once: the page tests drive Chromium
BASAL_E2E=1 uv run pytest tests/e2e -v -rs       # minutes; -rs lists what this machine skipped and why
```

Run it on every hardware class you ship for (Apple Silicon, an NVIDIA GPU, a CPU-only machine): each run tests the
engines that machine has and lists the rest as skipped.

## Models

| Fixture | Picked | Used by |
|---|---|---|
| `text_model` | the smallest downloaded, runnable text model, basal-2.0 first (`BASAL_E2E_MODEL`) | almost everything |
| `second_model` | another one (`BASAL_E2E_SECOND_MODEL`) | several models on one API port, memory budget, restart |
| `bf16_model` | bf16 PyTorch weights of basal-2.0 / 1.5 / 1.0 (`BASAL_E2E_BF16_MODEL`) | Inspect, mode comparison |

A fixture with no model on disk skips its tests; `BASAL_E2E_DOWNLOAD=1` downloads `Remek/basal-2.0-mini` (bf16,
3.2 GB: runnable on CUDA, Apple Silicon and the CPU) through basal-ui's Download button instead. Other variables:
`BASAL_E2E_MODELS_DIR` (the models directory to use; default: basal-ui's default, the Hugging Face cache) and
`BASAL_E2E_LOAD_TIMEOUT` (seconds for a model start or a job; default 1800, since `--mode fast` compiles for minutes).
On a fresh Apple Silicon machine, `Remek/basal-2.0-mini-MLX-8bit` and `Remek/basal-2.0-mini` cover every test that
the hardware allows.

## What is tested

| File | Feature group | Checks |
|---|---|---|
| `test_lifecycle.py` | start, auto-load, shutdown | a load with options is remembered; a new basal-ui on the same settings loads it again with them; Ctrl+C (SIGINT) leaves no `basal-serve` child |
| `test_models.py` | several models, API server | two models behind one port with a required key: `/v1/models`, `MODEL_REQUIRED`, 401 / 404, request ids, OpenAI Decisions, `/v1/stats` closed to user keys, MCP `list_models` / `decide` when MCP is installed; statistics, history (API and MCP), key last-use; the memory budget refusing a load; restart onto a new child, logs, Download refusing a model on disk, unload |
| `test_safety.py` | decision safety | the entry's `certified` thresholds and `context`; `act` with `max_error` deferred below the certified threshold (or without one); a state over the context truncated with consistent token counts; History's `deferred` / `truncated` flags |
| `test_question_types.py` | every question type | each `advanced.jsonl` request (multi, act, tool, classify, verify, compress, extract) answers in its own shape: keys, sets, costs, verdicts, kept units and their offsets, extracted values and their spans; per-type statistics; `facts` and `lang`; evidence spans or its refusal |
| `test_calibration.py` | custom calibration | task questions (classify) fitted with the engine in basal-ui's process and its memory released; a file of another model imported with basal-serve's warning and its report; a fit against a temporary uncalibrated child, its report and listing, the child's memory released; selecting the file: Restart to apply, `--calibration`, no calibration warning; back to the shipped file; cancel; refused inputs (with the problem line) |
| `test_engine231.py` | 2.3.1 integration | real CPU packed and MLX packed loads, inference and restart; child health idle counters; Chromium with explicitly simulated 422/503/500 responses, missing/unreachable counters, warning rendering, packed and cache help |
| `test_load_options.py` | load options | the 16 modes with reasons for the unavailable ones and the default available; options in the command; `--context-overflow refuse` returning 422; invalid / unavailable options refused; basal-serve's own refusal (prefix cache with `--deterministic`) on the model's row; remembered options used by a plain load |
| `test_batch.py` | batch | every line answered in order, the summary, paging, the download; not counted in the statistics; cancel without an output file; refused files |
| `test_measurements.py` | benchmarks | the preset (44 items), a load test (questions per request), a determinism check on a model loaded with `deterministic` (identical answers), two engine modes in process agreeing on 90% or more with the memory released |
| `test_inspect.py` | Inspect | basal-viz at `/viz` for a bf16 model: info, an explanation that matches basal-serve's probabilities, memory counted then released; other ports refused |
| `test_experimental.py` | thinking, early exit | refused where they cannot run; with basal-2.0 4.5B and vLLM: forced reasoning with its text, and `think: off`; with basal-1.0 bf16 on CUDA: every early-exit level |
| `test_pages.py` | the pages (Chromium) | every view without a script error; the Playground building each advanced request exactly as written, sending it and rendering its type's view (source highlighting, claim tables); the builder keeping what its fields cannot write (rubric labels, `option_keys`) until edited; no automation claim on media or unvalidated answers; the builder's type switch, cost matrix, multi-tool set options and inline errors; the safety strip; the load options dialog with each switch scoped to its backend, load-option presets and the revision chip; Batch; History and Logs; importing a calibration in the Calibration block; Inspect embedding basal-viz and explaining a Playground decision with what differs, and the handoff's notes and refusals; the model card, the theme switch, keyboard shortcuts and the usage line's branches / lane |

A test that needs hardware or a model this machine does not have is skipped with the reason (for example the CUDA
graph modes on Apple Silicon, thinking without vLLM, early exit without a CUDA GPU).

## Writing more

Use the session `ui` fixture and its helpers (`tests/e2e/support.py`): `ui.load(id, options)` brings a model to ready
with exactly those options (restarting it if needed), `ui.run_job(path, body, kind)` starts a background job and waits
for it, `ui.detail(method, path, body, status)` returns the message of a refused request. Tests share one basal-ui,
so each one loads what it needs and puts back what it changes (options, calibration selection, memory settings).

## Focused 2.3.1 gate

Run `uv run pytest -q tests/test_ui_engine231.py` for simulated capability, option lifecycle (including auto-load),
health caching/missing fields, both HTTP error envelopes, gateway/Playground forwarding, Batch error records, and
calibration warning tests. Run the hardware/browser gate with existing local models only:

```sh
BASAL_E2E=1 HF_HUB_OFFLINE=1 \
BASAL_E2E_MODEL=Remek/basal-2.0-mini-MLX-8bit \
BASAL_E2E_BF16_MODEL=Remek/basal-2.0-mini \
uv run pytest -q tests/e2e/test_engine231.py
```

The example selects Apple Silicon models; change the selections for another host. MLX checks skip with a reason
when unavailable. CPU packed runs on the host CPU, even on an Apple machine. The browser error/health-warning edge
cases use simulated responses, so their success does not prove a physical GPU OOM or NVIDIA graph acceptance.
NVIDIA compact/fine and sm_120 routing still require hardware-specific acceptance; capability tests alone do not
prove the kernels. Calibration jobs may finish as `done_with_warnings`; the release helper treats that as completed
work and the tests must inspect its warnings/thresholds instead of treating completion as certification.
