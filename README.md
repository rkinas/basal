# basal 2.5

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--2.0%20models-yellow)](https://huggingface.co/collections/Remek/basal-20-6acd1712a8e443c86cd576fc)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--omni--1.0%20models-yellow)](https://huggingface.co/collections/Remek/basal-omni-10-6acea28e8b5baed2db475c1f)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![basal-1.0 technical report](https://img.shields.io/badge/basal--1.0%20technical%20report-PDF-b31b1b.svg)](docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)

Inference engine and models for **basal-2.0** and **basal-omni-1.0**: small, fast, calibrated *typed-decision* models
for Polish and English. The basal-omni models also take **images and audio** in a request (experimental).

**What it is.** basal is inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of a chatbot
that writes an answer, the model reads a **state** (a message, a document, a case file, retrieved passages, an agent
trace, a web page as JSON) and answers a **typed question** about it by returning a **probability for each allowed
answer**, in one forward pass, without generating text. The answer can never fall outside the options you gave, and
the probability says how sure the model is.

**What it is for: a dynamic classifier.** You describe the classes *in the request*, in plain language, per call,
instead of training a classifier for them: routing tickets and documents (up to 255 options), rule and policy
checks, questions about long documents, RAG decisions, scoring on ordered scales, the next step of an agent, judging
answers, and triage with a confidence threshold that sends uncertain cases to a person.

> The name comes from the *basal ganglia*, the part of the brain that selects one action among competing options.

## Contents

- [What is new in basal 2.5](#what-is-new-in-basal-25) · [Models](#models) · [Results](#results) · [Quick start](#quick-start) · [API](#api)
- [Images and audio: basal-omni](#images-and-audio-basal-omni-experimental) · [MCP for agents](#mcp-basal-as-a-tool-for-agents) · [Documentation](#documentation)
- [Limitations](#limitations) · [Previous releases](#previous-releases) · [Citation](#citation) · [Contributors](#contributors) · [License](#license)

## What is new in basal 2.5

Released 2 November 2026. Default answers to decision requests are byte-identical to 2.4.1; what is new is opt-in
unless a line says otherwise. All changes and the known limits: [WHATS_NEW](docs/WHATS_NEW.md).

2.5.1 (2 November 2026): `--prepared auto` now loads the start-up kernels shipped in the model repositories.

2.5.2 (3 November 2026): bug fixes; a question with an unknown `type` is now refused (HTTP 422) instead of
answered as `choice`. [WHATS_NEW](docs/WHATS_NEW.md#what-is-new-in-basal-252).

2.5.3 (4 November 2026): bug fixes; `basal-run --preset` works again, presets pick the Polish wording for short
Polish text without diacritics, and `basal-attach` works on Windows without symlink rights.
[WHATS_NEW](docs/WHATS_NEW.md#what-is-new-in-basal-253).

2.5.4 (5 November 2026): bug fixes; malformed requests are refused before the run, request text cannot forge a turn,
`basal-prepare` prepares the engine `basal-serve` runs, and the CLIs, the router and `--ui` handle inputs they
failed on. [WHATS_NEW](docs/WHATS_NEW.md#what-is-new-in-basal-254).

- **Changes on the default path**: in a request with task questions, a question named `<task>::<part>` is refused (HTTP
  422, #59); error bodies gain `code` (#62); `GET /v1/models` lists the certified thresholds and the context (#64);
  every response carries an `x-request-id` header (#68).
- **basal-router**: replicas behind one port, routed by state (`--gpus 0,1`), cascades from a small to a bigger model
  (`--escalate`), `--blend` and an LLM `--fallback`. [SERVING](docs/SERVING.md).
- **Retrieval without an index** (*experimental*): the `rank` question type ranks thousands of candidates in one
  request, and `basal-navigate` searches a document tree with it. [NAVIGATE](docs/NAVIGATE.md).
- **Operations and start-up**: `GET /metrics` (Prometheus), a live web page (`--ui on`), HTTP 503 while the model warms
  up, `--api-keys`, `--cors-origins`; `basal-prepare` and `--prepared auto` (basal-omni-1.0-mini 290 -> 172 s on an
  H100 with empty caches, the same answers); torch.compile's cache survives a reboot.
- **Answers you can rely on**: answer sets (`"coverage"`), an abstain flag with `max_error`, a second read (`--reread`),
  and an option-order check and average (`basal-bench --order-check`, `--order-average`).
- **More GPUs and engines**: basal-omni-1.0-ultra in `--mode fp8`, `nvfp4` and `int8` and on a 48 GB card
  (`--quant-load cpu`); basal-omni-pico in the fast engine and on `--mode mlx`; basal-omni-1.1-max images and audio
  through `--mode vllm`; LM Studio as a host for basal decision models (`basal-export-gguf --systemone`).
- **APIs, clients and tools**: OpenAI's Decisions API (`POST /v1/decisions`), `AsyncBasal`, a TypeScript client,
  presets, [LangChain / LlamaIndex / CrewAI tools](docs/SERVING.md#agent-frameworks-and-ready-question-sets-25);
  `basal-attach`, `basal-agree`, `basal-finetune`, `basal-selfcheck` and `basal-interpolate`.
- Thank you to Miłosz Kucharski, Paweł Kiszczak, Marcin Brzeżański, Dariusz Koryto, Grzegorz Stark, Michał Kardyś and
  Grzegorz Brzezinka for their pull requests.

## Models

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-2.0-mini** | 1.5B | fastest; high-volume routing, laptops and small GPUs; context 8,192 tokens | [`Remek/basal-2.0-mini`](https://huggingface.co/Remek/basal-2.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-mini-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-mini-MLX-8bit) |
| **basal-2.0** | 4.5B | main model: best balance of accuracy and speed; experimental thinking mode | [`Remek/basal-2.0-4.5B`](https://huggingface.co/Remek/basal-2.0-4.5B) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-4.5B-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-4.5B-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-4.5B-MLX-8bit) |
| **basal-2.0-max** | 11B | most accurate; long documents, judging, hard English decisions | [`Remek/basal-2.0-max`](https://huggingface.co/Remek/basal-2.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-max-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-max-MLX-8bit) |

**basal-omni** (experimental; text, images and audio; built on Google Gemma 4, pico on Qwen3.5):

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-omni-pico** | 0.8B (Qwen3.5-0.8B) | the smallest basal model: text decisions on a CPU or a small GPU (Werdykt 0.660); `--mode fast` on CUDA from basal 2.4 (with flash-linear-attention; 2.3.2-2.3.x: `--mode eager`), `--mode gguf` on the CPU | [`Remek/basal-omni-pico`](https://huggingface.co/Remek/basal-omni-pico) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-pico-GGUF) |
| **basal-omni-1.0-mini** | 4B (Gemma 4 E4B) | decisions about text, images and audio; about 16 GB of GPU memory | [`Remek/basal-omni-1.0-mini`](https://huggingface.co/Remek/basal-omni-1.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX) |
| **basal-omni-1.0-max** | 12B | the default 12B model (Werdykt 0.823); text, images and audio; about 24 GB of GPU memory | [`Remek/basal-omni-1.0-max`](https://huggingface.co/Remek/basal-omni-1.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX) |
| **basal-omni-1.1-max** | 12B | *experimental task model*: better on `classify`, `verify`, `compress` and `extract`, slightly behind 1.0-max on general decisions (Werdykt 0.821) | [`Remek/basal-omni-1.1-max`](https://huggingface.co/Remek/basal-omni-1.1-max) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX) |
| **basal-omni-1.0-ultra** | 31B (Gemma 4 31B) | highest Werdykt score of the basal models (0.879); text and images (no audio); 59 GB of bf16 weights | [`Remek/basal-omni-1.0-ultra`](https://huggingface.co/Remek/basal-omni-1.0-ultra) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-ultra-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-ultra-GGUF) |

Each repository ships its prompt format (`basal.json`) and calibration files, and every port is checked against its
bf16 model ([files in detail](docs/INSTALL.md#what-the-model-repositories-contain)): `-FP8` for the basal engine
(sm89+; basal-omni: `--mode vllm`, text only), `-GGUF` for Ollama and llama.cpp, `-MLX` for Apple Silicon. All models
are fine-tuned from Apache-2.0 base models (Bielik v3 for basal-2.0, Gemma 4 for basal-omni; see [NOTICE](NOTICE)) and
released under Apache-2.0. Start with **basal-2.0 (4.5B)**: **max** when accuracy on long or hard inputs matters more
than latency, **mini** for high volumes, laptops and small GPUs, **basal-omni** when a request carries images or audio.

## Results

[Werdykt](https://basal.si5.pl/benchmark/): 10 categories in Polish and English, 5,000 decisions whose items are
not published, one protocol for every model. Each basal-2.0 model against the basal-1.5 model of the same size:

| | mini: 1.5 → 2.0 | 4.5B: 1.5 → 2.0 | max: 1.5 → 2.0 |
|---|---|---|---|
| Werdykt macro accuracy | 0.607 → **0.618** | 0.721 → **0.729** | 0.773 → **0.784** |
| calibration error (ECE, lower is better) | 0.086 → **0.035** | 0.068 → **0.060** | 0.074 → **0.044** |
| decisions/s at 16 clients, bf16, one H100 | 20.5 → **106.4** | 23.0 → **32.2** | 10.2 → **16.2** |
| English panel (10 public tasks), points | −2.1 | −3.6 | −1.8 |

basal-1.5 (4.5B) and basal-1.5-max are measured on SGLang, their fastest configuration. basal-omni-1.0 scores 0.769
(mini) and 0.823 (max). Hosted API models and larger open decision systems score above basal-2.0 (the
[board](https://basal.si5.pl/benchmark/)). Intervals, calibration, other benchmarks, speed: [RESULTS](docs/RESULTS.md).

## Quick start

NVIDIA GPU, with [uv](https://docs.astral.sh/uv/) in a fresh environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

- **`--template-prefix off` for the 4.5B model** (bf16 and `-FP8`); mini and max run with the default.
- **Install torch first, from the CUDA 12.8 index**: a plain `pip install basal` may take a torch built for a newer
  CUDA than your driver. DGX Spark and B300: the cu130 build ([docs/HARDWARE.md](docs/HARDWARE.md)).
- The first start in `--mode fast` compiles the model (several minutes); `--mode fast-nocompile` starts in seconds.

Apple Silicon, with the native arm64 Python (`--python 3.12` can take an x86_64 one under Rosetta) and no torch step:

```bash
uv python install cpython-3.12-macos-aarch64-none
uv venv --python cpython-3.12-macos-aarch64-none --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000
```

`uname -m` prints `arm64` in a terminal that does not run under Rosetta (`x86_64` under Rosetta).

Ask questions with curl. Several questions about one state go into one request and share the state; each answer has a
probability for every option, the chosen key and the confidence ([response](docs/API.md#response)):

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

Other ways to run: [FP8 checkpoints](docs/INSTALL.md#fp8-checkpoints-basal-engine), [vLLM or SGLang](docs/INSTALL.md#vllm-or-sglang-with-bf16-weights),
[Ollama and llama.cpp](docs/INSTALL.md#ollama-and-llamacpp-gguf), [LM Studio](docs/INSTALL.md#lm-studio-gguf-as-a-decision-model), [Apple Silicon in detail, CPU](docs/INSTALL.md#apple-silicon-mlx),
[all seven engines compared](docs/INSTALL.md#engines), the [Python client](docs/INSTALL.md#nvidia-gpu-the-basal-engine-recommended); a JSONL file, serving flags and
benchmarking your GPU: [SERVING](docs/SERVING.md).

## API

`POST /v1/systemone` with a `state` (text or JSON) and named, typed `questions`; every answer has a calibrated
probability for each option and a `confidence`. Up to 255 options per question (basal-1.x models: 10); several
questions over one state are answered in one pass. `POST /v1/decisions` takes OpenAI's Decisions API
([docs/API.md](docs/API.md#openais-decisions-api-post-v1decisions)).

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |

To send uncertain decisions to a person, use `act` or `tool` with `"max_error": 0.05`: the answer says
`"deferred": true` below the certified confidence threshold. Every field and question type, using the confidence and
how to ask: [docs/API.md](docs/API.md). TypeScript client: [clients/typescript](clients/typescript/README.md).

## Images and audio: basal-omni (experimental)

The basal-omni models answer the same questions; with `--media on` a request may also carry images and audio (`"media"`):

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000   # or Remek/basal-omni-1.0-max
```

They were trained on text decisions only: Gemma 4's encoders read the media zero-shot and the answers use the text
calibration, so validate them on your own media. Request format and limits:
[INSTALL](docs/INSTALL.md#images-and-audio-basal-omni-experimental); memory and engine details: [OMNI](docs/OMNI.md).

## MCP: basal as a tool for agents

*Experimental.* With the `basal[mcp]` extra, `basal-serve --mcp on` also serves the decisions over the
[Model Context Protocol](https://modelcontextprotocol.io) at `/mcp` (tools `decide` and `decide_many`); `basal-mcp` is
a stdio bridge for hosts that start local processes. Protect HTTP and MCP requests with `--api-keys PATH` or bind to
`127.0.0.1` ([SERVING](docs/SERVING.md#api-access), [MCP](docs/MCP.md)).

```bash
uv pip install "basal[fp8,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

## Documentation

| | |
|---|---|
| [INSTALL](docs/INSTALL.md) · [SERVING](docs/SERVING.md) | engines, model files and every install path; serving modes and flags, a JSONL file, benchmarking your GPU |
| [API](docs/API.md) · [QUESTION_TYPES](docs/QUESTION_TYPES.md) · [FEATURES](docs/FEATURES.md) | request, response, using the confidence, how to ask; every question type on one page; `multi`, `act`, `tool` and evidence in detail |
| [CALIBRATION](docs/CALIBRATION.md) · [FINETUNE](docs/FINETUNE.md) | calibrating on your own data; LoRA on your own labelled decisions (`basal-finetune`) |
| [TASKS](docs/TASKS.md) · [NAVIGATE](docs/NAVIGATE.md) | *experimental*: `classify`, `verify`, `compress`, `extract`; `rank` and `basal-navigate` (retrieval by navigating a document tree) |
| [BACKEND](docs/BACKEND.md) · [MCP](docs/MCP.md) | basal-serve as the local backend of code that asked a hosted LLM for decisions; the MCP server and bridge |
| [RESULTS](docs/RESULTS.md) · [THINKING](docs/THINKING.md) | results against basal-1.5: Werdykt, other benchmarks, speed; the experimental thinking mode |
| [OMNI](docs/OMNI.md) · [HARDWARE](docs/HARDWARE.md) · [OLD_GPUS](docs/OLD_GPUS.md) | basal-omni on the engine; which mode on which GPU; GTX 10xx, RX 480 / 580, Iris Xe with llama.cpp's Vulkan backend |
| [FORMATS](docs/FORMATS.md) · [FORMAT2](docs/FORMAT2.md) · [SOAM](docs/SOAM.md) | prompt formats, option orders and calibration files; the runtime contract; shared-state packing |
| [PORTS](docs/PORTS.md) · [GGUF](docs/GGUF.md) · [QUANTIZATION](docs/QUANTIZATION.md) | MLX and GGUF ports; FP8 and other low precision |
| [WHATS_NEW](docs/WHATS_NEW.md) · [CONTRIBUTING](CONTRIBUTING.md) · [basal-1.0 report](docs/basal-1.pdf) | every release's changes; development commands, checks and release-note fragments; the basal-1.0 technical report |

**Community projects:** [basal-lite](https://github.com/lukaszpodgorski-pl/basal-lite) by @lukaszpodgorski-pl serves basal
models from a stock `vllm serve` through a thin client (community-maintained, not verified by us).

## Limitations

- Evaluated on held-out and hidden test sets and on public benchmarks; validate claims about a specific document
  collection on your own data. Decisions with serious consequences for people should be reviewed by a person.
- On a panel of 10 public English decision tasks all three basal-2.0 models are below basal-1.5 (−2.1 / −3.6 / −1.8
  points for mini / 4.5B / max); basal-2.0-mini is also below basal-1.5-mini on Polish decisions (−1.4 points).
- The calibration is cautious on unseen tasks: `max_error: 0.01` defers every decision with the shipped files (except
  `choice` with `basal-2.0-max-GGUF`'s `CALIBRATION.ollama.json`, confidence 0.989); use 0.05 or fit thresholds on your
  data. At the 1% target on Werdykt, basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.
- A small model's Polish world knowledge is limited, and it does not know rules introduced after its training: supply
  the facts and rules in the state. Compute exact arithmetic and thresholds in code, or use `"facts": "auto"` for
  Polish dates and amounts. basal-2.0-mini has an 8,192-token context; longer states are cut in the middle.
- Experimental: the thinking mode (less accurate, less calibrated and slower on Werdykt); `act`, `tool`, `multi` over
  many labels and evidence spans; the task types (`automated` needs task thresholds: only basal-omni-1.1-max ships one,
  `classify` at 5%); basal-omni's images and audio, read zero-shot by Gemma 4's encoders with the text calibration.

## Previous releases

basal-1.5 and basal-1.0 remain on Hugging Face and are served unchanged by this engine: [previous releases](docs/WHATS_NEW.md#previous-releases-basal-15-and-basal-10).

## Citation

```bibtex
@software{kinas2026basal20,
  title   = {basal-2.0: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/rkinas/basal}
}

@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}

@software{kinas2026basalomni10,
  title   = {basal-omni-1.0: typed-decision models for text, images and audio in Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {1.0},
  url     = {https://huggingface.co/collections/Remek/basal-omni-10-6acea28e8b5baed2db475c1f}
}
```

## Contributors

Thank you to [Paweł Kiszczak](https://github.com/pawelkiszczak), [Piotr Durlej](https://github.com/pdurlej), [Kuba Sołtys](https://github.com/qooba),
[Michał Kardyś](https://github.com/kardysm), [@lukaspoloki](https://github.com/lukaspoloki), [Miłosz Kucharski](https://github.com/Butterski),
[Grzegorz Brzezinka](https://github.com/agentGreg), [@piotrkowski](https://github.com/piotrkowski), [@lukaszpodgorski-pl](https://github.com/lukaszpodgorski-pl),
[Marcin Brzeżański](https://github.com/marcinbrzezanski), [Grzegorz Stark](https://github.com/gstark0), [Dariusz Koryto](https://github.com/dkoryto) and
[Rafal Wiktorowicz](https://github.com/onlypositivetravelvibes-dotcom), whose pull requests are in this repository, and to [Kamil](https://github.com/kamilrosa) and [@krzysztofkaminski](https://github.com/krzysztofkaminski) for their measurements.
What each one contributed, by pull request: [CONTRIBUTING](CONTRIBUTING.md#contributors-by-pull-request). `--mode tpu`
and `--mode strix` are community-maintained: their numbers in [HARDWARE](docs/HARDWARE.md) are the contributors' own.

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models (Bielik v3, SpeakLeash and ACK Cyfronet AGH, for
basal-2.0; Gemma 4, Google, for basal-omni); see [NOTICE](NOTICE).
