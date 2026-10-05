# basal 1.5

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--1.5%20collection-yellow)](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![basal-1.0 technical report](https://img.shields.io/badge/basal--1.0%20technical%20report-PDF-b31b1b.svg)](docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)

Inference engine and models for **basal-1.5**: small, fast, calibrated *typed-decision* models for Polish and English.

**What it is.** basal is inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of a chatbot
that writes an answer, the model reads a **state** (a message, a document, a case file, retrieved passages, an agent
trace, a web page as JSON) and answers a **typed question** about it by returning a **probability for each allowed
answer**, in one forward pass, without generating text. The answer can never fall outside the options you gave, and
the probability says how sure the model is.

**What it is for: a dynamic classifier.** You describe the classes *in the request*, in plain language, per call,
instead of training a classifier for them. The same model routes tickets today, checks a filing deadline tomorrow and
scores the urgency of an incident report next week, each time with new options and no retraining. Typical uses:

- ticket, e-mail and document **routing** with categories that change often;
- **rule and policy checks** on a document ("is the claim covered?", "was the appeal filed in time?");
- **questions about long real documents** (judgments, contracts, statutes, reports), with the supporting passage
  returned as **evidence**;
- **RAG decisions**: is a passage relevant, do the passages suffice to answer, do sources conflict, is a claim
  supported, or should the system say "cannot be determined";
- **scoring** on ordered scales (urgency, risk, severity, quality of an answer);
- **agent decisions**: the next step or tool, whether to ask the user, whether a call needs confirmation, with
  **cost-aware actions** that defer to a person when automation is too risky;
- **judging**: is a candidate answer correct and complete, which of two answers is better;
- **triage with a confidence threshold**: accept confident decisions automatically and send the rest to a person.

> The name comes from the *basal ganglia*, the part of the brain that selects one action among competing options.

## What is new in basal-1.5

**Models**
- A family of three: **basal-1.5** (4.5B, main model), **basal-1.5-max** (11B, the most accurate in the family) and
  **basal-1.5-mini** (1.5B, fastest, distilled from max). Weights in bf16, plus **FP8 / NVFP4** checkpoints for vLLM on
  NVIDIA GPUs, **MLX 4-bit / 8-bit** for Apple Silicon and **GGUF** for Ollama.
- Stronger on **long real documents** and **retrieval (RAG) decisions**, more reliable **"cannot be determined"**
  (abstain) answers, harder **English decisions** (judging answers, trade-offs, traps, policies, ambiguous requests),
  **action selection** for agents and **ordinal scores**. On a sealed Polish test, created after the 1.5 weights were
  frozen and scored once, basal-1.5 reaches 0.931 against 0.874 for basal-1.0 (+5.7 points, paired 95% interval [+4.5, +6.8]). On a
  panel of 10 public English decision tasks it is above basal-1.0 (0.688 vs 0.671). The training recipe now includes a reinforcement-learning (RL) stage.
- A new benchmark, **[Werdykt](#werdykt-hidden-polishenglish-decision-benchmark)**: hidden, Polish and English,
  10 categories, one protocol for every model (accuracy, latency and cost).

**Engine**
- **SOAM, "state once, ask many"**: all questions of a request, in both option orders, are answered over one shared
  state: about **3× faster at 5 questions** and 2.8× at 12, with the same answers as separate requests (exact in
  fp32). See [docs/SOAM.md](docs/SOAM.md).
- New question types: **`multi`** (which labels apply) and **`act`** (the cheapest action under your costs, or defer to
  a person).
- **Evidence** (*experimental*): `"evidence": true` returns the spans of the state that support the answer, as
  character offsets, for RAG and document QA.
- **Facts** (`"facts": "auto"`): computed calendar and money facts for Polish states (weekdays, days off, gaps, date +
  duration, net/gross, sums, comparisons) are appended to the state, so the model reads arithmetic instead of doing
  it.
- **More engines**: vLLM and SGLang backends (agreement with the fp32 engine, offline: vLLM 0.994, SGLang 0.995; SGLang also gives the same answer as the basal engine on 98.3% of Werdykt items), Apple Silicon
  (MLX, `mps`, `--device mps`), Ollama and llama.cpp (GGUF, also in process with `--mode gguf`). The `mps`, `gguf` and
  `mlx-q8` modes and the Apple Silicon engine study were contributed by Paweł Kiszczak
  ([#4](https://github.com/rkinas/basal/pull/4)).

Details of the experimental features: [docs/FEATURES.md](docs/FEATURES.md). Full list of changes:
[release notes](https://github.com/rkinas/basal/releases/tag/v1.5.0).

## Models

Available on Hugging Face from **2026-10-05**.

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-1.5** | 4.5B | main model: best balance of accuracy and speed | `Remek/basal-1.5-4.5B` (bf16) · `-FP8` · `-NVFP4` · `-MLX-fp4` · `-MLX-8bit` · `-GGUF` |
| **basal-1.5-max** | 11B | most accurate in the family; long documents, judging, hard English decisions | `Remek/basal-1.5-max` (bf16) · `-FP8` · `-NVFP4` · `-MLX-fp4` · `-MLX-8bit` · `-GGUF` |
| **basal-1.5-mini** | 1.5B | fastest; high-volume routing, laptops and small GPUs (distilled from max) | `Remek/basal-1.5-mini` (bf16) · `-FP8` · `-MLX-fp4` · `-MLX-8bit` · `-GGUF` |

- **bf16** weights run in this engine (`fast`, `fp8`, `eager`), in vLLM and in SGLang. Each repository ships
  `CALIBRATION.json` (per-type temperatures and confidence thresholds), `basal.json` (prompt format and readout) and
  `evidence_head.pt` (the experimental evidence head). The 1.5 models do not ship early-exit heads.
- **MLX 4-bit / 8-bit** for Apple Silicon and **GGUF** for Ollama / llama.cpp are ports of the same weights; their
  agreement with the bf16 model is listed in [docs/QUANTIZATION.md](docs/QUANTIZATION.md).
- **FP8 / NVFP4** (`-FP8`, `-NVFP4`) are NVIDIA ModelOpt checkpoints for vLLM (`--mode vllm`): FP8 for Hopper, Ada
  and Blackwell, NVFP4 for Blackwell only; agreement and speed in [docs/QUANTIZATION.md](docs/QUANTIZATION.md#basal-15-the-modelopt-checkpoints).
  In the basal engine itself, `--mode fp8` still quantises the bf16 weights on the fly.
- All models are fine-tuned from Apache-2.0 base models (see [NOTICE](NOTICE)) and released under Apache-2.0.

Which one to choose: start with **basal-1.5**. Use **max** when accuracy on long or hard inputs matters more than
latency, and **mini** for high volumes, laptops, small GPUs and CPUs. All three use the same API, prompt format and
calibration procedure.

## Engines

basal runs on six engines, all behind the same `basal-serve` HTTP API (`POST /v1/systemone`). Commands below use
basal-1.5; replace the repository name for max or mini.

<!-- engines -->
| engine | `basal-serve` mode | hardware | what it is for | quick start | status (basal-1.5, 4.5B) |
|---|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`, `fp8`, `nvfp4`); `mps`; `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon with `mps` (packed requests on the Mac's GPU, no CUDA graphs) or `eager`; CPU with `eager` | the primary server: packed requests (SOAM), evidence spans, `fp8` on Hopper and Blackwell | `basal-serve --model Remek/basal-1.5-4.5B --mode fast` | **verified**: served check on one H100, 13.7 ms p50 per question over HTTP, 64.4 decisions/s with 32 clients |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server | `basal-serve --model Remek/basal-1.5-4.5B --mode vllm` (extra `basal[vllm]`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.994 with the fp32 engine, accuracy 0.928 vs 0.930, 147 decisions/s |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (RadixAttention prefix sharing) | `basal-serve --model Remek/basal-1.5-4.5B --mode sglang` (own environment: `sglang[srt]==0.5.21`, then basal with `--no-deps`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.995 with the fp32 engine, accuracy 0.925 vs 0.930, 104.6 decisions/s; on Werdykt the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens) |
| **MLX** | `mlx` (default on a Mac with `basal[mlx]`); `mlx-q8` | Apple Silicon | Macs: `-MLX-8bit` (recommended) or `-MLX-fp4` (half the memory); `mlx-q8` makes 8-bit weights from bf16 at load time | `basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.992 (`-MLX-8bit`) / 0.970 (`-MLX-fp4`), 587 / 581 ms per question |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with Ollama (`-GGUF`) | 1. `hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF` 2. in that folder: `ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0` 3. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 518 / 542 ms per question |
| **llama.cpp** | `llamacpp`; `gguf` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files; the engine's token ids are sent as is. `--mode gguf --gguf <file>` runs the file in process instead (extra `basal[gguf]`, [GGUF.md](docs/GGUF.md)) | 1. `llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080` 2. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 427 / 446 ms per question |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp run the weights; the typed answers (a calibrated probability for each of your options, both option orders, the state shared by several questions, `multi`, `act` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API in every mode. A plain `ollama run` or `llama-cli` gives free text only. Evidence spans need the PyTorch engine (`fast`, `eager`); vLLM and SGLang start with a 4,096-token context (longer states are refused); raise it with `--max-len` (e.g. 32768 for long documents). Install and first request for each engine: [quick start](#quick-start).
<!-- /engines -->

> [!WARNING]
> **Quality warning.** The 4-bit MLX ports of basal-1.5-mini and basal-1.5-max lose accuracy, and their confidence thresholds are not certified: mini's about 3.6 points against bf16 (0.882 vs 0.918), max's about 2.0 points against its GGUF Q8_0 reference (0.928 vs 0.948). For mini, use `-MLX-8bit` or GGUF (`Q8_0` / `Q4_K_M`), whose accuracy stays within noise of bf16; for max, use `-MLX-8bit` (agreement 0.996, accuracy 0.948) or, with Ollama / llama.cpp, GGUF `Q4_K_M` (0.994, 0.942). For vLLM, the NVFP4 checkpoint of basal-1.5 (4.5B) loses about 2.8 accuracy points (0.902 vs 0.930; agreement 0.937); use `-FP8` unless memory or throughput matters more.

4-bit ports trade accuracy for memory: on basal-1.5 the 4-bit MLX port costs about 1.4 points (4.5B), 2.0 points (max, against its GGUF Q8_0) and 3.6 points (mini); the 8-bit ports and GGUF Q8_0 stay within noise. Use 8-bit when accuracy matters.

## Performance

### Werdykt: hidden Polish–English decision benchmark

**Werdykt** is our hidden benchmark of typed decisions, built to compare decision models and general LLMs on the same
footing:

- **10 categories × Polish and English**: rules (deadlines, amounts, eligibility), real documents, long documents
  (with injected instructions), retrieval / RAG, contracts, routing, score, judge (answer grading), action (the next
  step of an agent) and abstain ("cannot be determined" vs a concrete answer).
- **Closed dataset.** The items are not published, so they cannot end up in training data; only a few samples per
  category are public ([basal.si5.pl](https://basal.si5.pl/)). No Werdykt item or source document is in basal's
  training data (checked by exact and shared-phrase matching).
- **One protocol for every model**: one call per item, the same pinned prompt (state, question, lettered options,
  JSON answer), unparsed answers count as wrong. Open models are served on **one H100**; API models are called through
  OpenRouter with providers that retain data excluded; the reasoning setting of each model is recorded. basal is served
  by `basal-serve` (`/v1/systemone`, both option orders, calibrated) on its fastest engine: SGLang where measured (the
  leaderboard shows the median of the basal engine's `fast` mode alongside), otherwise the `fast` mode.
- **Metrics**: accuracy macro-averaged over the 20 category × language cells (with a 95% bootstrap interval), the
  Polish and English macros, median latency per decision (one request at a time, on a fixed speed subset) and cost per
  1,000 decisions (open models: one H100 at $3.95 per hour divided by the throughput measured with 16
  concurrent requests; API
  models: the cost reported by the provider).

| # | system | type | params | Werdykt macro [95% CI] | PL | EN | latency p50 | cost / 1k decisions | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gemini 3.8 Flash | API | – | 0.996 [0.994, 0.997] | 0.996 | 0.996 | 3.3 s | $2.53 | – |
| 2 | Claude Opus 5.5 † | API | – | 0.994 [0.992, 0.996] | 0.992 | 0.996 | 5.5 s | $15.17 | – |
| 3 | GPT-6.1 Sol | API | – | 0.994 [0.992, 0.996] | 0.992 | 0.995 | 2.2 s | $3.18 | – |
| 4 | GPT-6 Astra | API | – | 0.993 [0.991, 0.996] | 0.992 | 0.995 | 2.0 s | $15.82 | family of a label voter |
| 5 | GPT-6 Luna | API | – | 0.983 [0.979, 0.986] | 0.978 | 0.988 | 2.4 s | $0.188 | – |
| 6 | Gemini 3.1 Pro (preview) † | API | – | 0.979 [0.975, 0.982] | 0.979 | 0.978 | 5.5 s | $12.30 | – |
| 7 | Jev 1.13.0 | API | – | 0.816 [0.807, 0.826] | 0.820 | 0.812 | 286 ms | $0.071 | – |
| 8 | Cygnet | open | 12B | 0.785 [0.775, 0.795] | 0.790 | 0.780 | 39.6 ms | $0.052 | protocol deviation |
| 9 | Winnow-12B Q8 | open | 12B | 0.785 [0.774, 0.794] | 0.786 | 0.784 | 126 ms | $0.260 | protocol deviation |
| 10 | **basal-1.5-max** | open | 11B | 0.773 [0.762, 0.783] | 0.777 | 0.769 | 67.1 ms (44.7 ms in fast mode) | $0.108 | SGLang engine, in-domain decision types |
| 11 | Jev-Omni | open | 12B | 0.753 [0.742, 0.764] | 0.763 | 0.744 | 1.1 s | $1.24 | protocol deviation |
| 12 | **basal-1.5-4.5B** | open | 4.5B | 0.721 [0.710, 0.733] | 0.724 | 0.718 | 33.8 ms (31.2 ms in fast mode) | $0.048 | SGLang engine, in-domain decision types |
| 13 | Clef-Flash | open | 9B | 0.716 [0.704, 0.727] | 0.701 | 0.731 | 1.7 s | $1.90 | vendor of a build model, protocol deviation |
| 14 | jqv (Qwen3-32B zero-shot) | open | 32B | 0.705 [0.694, 0.717] | 0.695 | 0.716 | 130 ms | $0.230 | vendor of a build model, protocol deviation |
| 15 | Hopper | open | 4B | 0.703 [0.692, 0.714] | 0.694 | 0.712 | 60.4 ms | $0.138 | vendor of a build model |
| 16 | Decision 4B v1.2 | open | 4B | 0.702 [0.691, 0.713] | 0.698 | 0.705 | 28.7 ms | $0.071 | vendor of a build model, protocol deviation |
| 17 | decider-4b v2 | open | 4B | 0.700 [0.689, 0.711] | 0.697 | 0.703 | 27.8 ms | $0.051 | protocol deviation |
| 18 | djev | open | 26B (4B active) | 0.698 [0.687, 0.710] | 0.702 | 0.694 | 66.6 ms | $0.053 | protocol deviation, failed calls counted wrong |
| 19 | Plumb-4B | open | 4B | 0.690 [0.679, 0.703] | 0.689 | 0.692 | 27.7 ms | $0.055 | protocol deviation |
| 20 | Decision 4B v1.1 | open | 4B | 0.690 [0.679, 0.702] | 0.688 | 0.693 | 28.5 ms | $0.124 | vendor of a build model, protocol deviation |
| 21 | JevK5 v0.3 (4B) | open | 4B | 0.689 [0.677, 0.701] | 0.683 | 0.694 | 28.4 ms | $0.057 | protocol deviation |
| 22 | Imajev-4B | open | 4B | 0.682 [0.671, 0.692] | 0.666 | 0.698 | 113 ms | $0.135 | vendor of a build model, protocol deviation, failed calls counted wrong |
| 23 | basal-1.0-4.5B | open | 4.5B | 0.660 [0.647, 0.672] | 0.662 | 0.659 | 26.5 ms | $0.135 | in-domain decision types |
| 24 | Malkuth-4B | open | 4B | 0.634 [0.621, 0.647] | 0.622 | 0.645 | 153 ms | $0.172 | vendor of a build model, protocol deviation, failed calls counted wrong |
| 25 | Manchego v2.1 | open | 4B | 0.630 [0.618, 0.643] | 0.622 | 0.639 | 73.5 ms | $0.115 | – |
| 26 | lev | open | 4B | 0.629 [0.616, 0.641] | 0.617 | 0.640 | 158 ms | $0.201 | vendor of a build model |
| 27 | metask-jev-4b | open | 4B | 0.612 [0.599, 0.623] | 0.607 | 0.616 | 108 ms | $0.340 | vendor of a build model, protocol deviation, failed calls counted wrong |
| 28 | **basal-1.5-mini** | open | 1.5B | 0.607 [0.594, 0.619] | 0.619 | 0.594 | 14.2 ms | $0.054 | in-domain decision types |
| 29 | SemIf (formerly OpenJev) | open | 4B | 0.588 [0.576, 0.599] | 0.575 | 0.601 | 66.0 ms | $0.080 | vendor of a build model, protocol deviation, failed calls counted wrong |
| 30 | basal-1.0-1.5B | open | 1.5B | 0.522 [0.509, 0.536] | 0.527 | 0.518 | 14.6 ms | $0.055 | in-domain decision types |

*Werdykt v1, 2026-10-03. Systems marked † took part in building or checking Werdykt items; their scores are
shown for completeness and are not comparable. Full leaderboard and the public samples: [basal.si5.pl](https://basal.si5.pl/).*

**By category** (macro over Polish and English):

| category | basal-1.5 | basal-1.5-max | basal-1.5-mini | basal-1.0 | best open model | best API model |
|---|---|---|---|---|---|---|
| rules | 0.344 | 0.346 | 0.304 | 0.340 | 0.376 (Cygnet) | 1.000 (Gemini 3.8 Flash) |
| real documents | 0.926 | 0.972 | 0.832 | 0.890 | 0.970 (Cygnet) | 0.996 (Gemini 3.8 Flash) |
| long documents | 0.612 | 0.682 | 0.536 | 0.588 | 0.660 (Clef-Flash) | 1.000 (GPT-6.1 Sol) |
| retrieval (RAG) | 0.956 | 0.978 | 0.862 | 0.912 | 0.988 (Cygnet) | 1.000 (GPT-6.1 Sol) |
| contracts | 0.654 | 0.710 | 0.544 | 0.598 | 0.740 (Cygnet) | 0.994 (Gemini 3.8 Flash) |
| routing | 0.952 | 0.982 | 0.876 | 0.880 | 0.978 (Winnow-12B Q8) | 0.992 (Gemini 3.8 Flash) |
| score | 0.680 | 0.762 | 0.538 | 0.602 | 0.780 (Winnow-12B Q8) | 1.000 (Gemini 3.8 Flash) |
| judge | 0.784 | 0.828 | 0.666 | 0.604 | 0.880 (Winnow-12B Q8) | 0.990 (Gemini 3.8 Flash) |
| action | 0.630 | 0.734 | 0.414 | 0.606 | 0.850 (Cygnet) | 1.000 (GPT-6 Astra) |
| abstain | 0.674 | 0.734 | 0.494 | 0.584 | 0.762 (Imajev-4B) | 1.000 (Gemini 3.8 Flash) |

In short: the frontier API models answer almost every item (0.983–0.996 without the † systems) at 2.0–5.5 s and $0.19–$15.82 per 1,000 decisions. Among the decision systems, the hosted Jev API (0.816) and the 12B open models Cygnet and Winnow-12B (0.785) score above basal; basal-1.5-max (11B, 0.773) comes next, ahead of the 12B Jev-Omni (0.753), and basal-1.5 (4.5B, 0.721) is above every 4B open system and level with or above the 9B Clef-Flash (0.716) and the 32B zero-shot Qwen3 (0.705). basal-1.5 and basal-1.5-max improve on basal-1.0 by 6.1 and 11.2 points and run locally on one H100, at 34 and 67 ms per decision with SGLang (31 and 45 ms in the basal engine's fast mode) and $0.048 and $0.108 per 1,000 decisions in H100 time. basal-1.5-mini (1.5B, 0.607) is 8.4 points above basal-1.0-1.5B at the same speed and cost (14 ms, $0.054), the option for constrained hardware.

### Other results

Accuracy, both option orders averaged, all models served with this engine on one H100 (`--mode fast`, bf16).

| model | PL sealed test | PL decisions | PL general | EN external panel | Public EN bench. | coverage at 1% error |
|---|---|---|---|---|---|---|
| **basal-1.5** (4.5B) | 0.931 | 0.923 | 0.741 | **0.688** | 0.762 | 51.0% |
| basal-1.5-max (11B) | 0.933 | 0.940 | 0.794 | 0.737 | 0.848 | 62.2% |
| basal-1.5-mini (1.5B) | 0.910 | 0.877 | 0.667 | 0.581 | 0.753 | 46.5% |
| basal-1.0-4.5B | 0.874 | 0.886 | 0.737 | 0.671 | 0.740 | 55.2% |
| basal-1.0-1.5B | 0.889 | 0.851 | 0.656 | 0.584 | 0.675 | – |

basal-1.5 against basal-1.0 (4.5B, paired 95% intervals): sealed Polish test +5.7 points [+4.5, +6.8], Polish decisions +3.7 [+3.0, +4.4], Polish general knowledge +0.4 (not significant), public English benchmark +2.2 (0.762 vs 0.740). On the English external panel it is
above basal-1.0: 0.688 vs 0.671 (+1.6 [+0.1, +3.0]), with the largest gains on Financial PhraseBank and BIG-Bench Hard
and a loss on Circa (per task below). At the 1% error target it decides fewer test decisions automatically than
basal-1.0 (51.0% vs 55.2%), at a lower observed error (0.55% vs 0.89%). basal-1.5-max reaches 0.737 on the English
external panel (+6.9 [+5.3, +8.5] over basal-1.0-4.5B), higher than basal-1.0 on Circa as well. basal-1.5-mini scores 0.581 there, level with basal-1.0-1.5B (0.584).

basal-1.5-max against basal-1.0-4.5B (paired 95% intervals): sealed Polish test +5.9 points [+4.7, +7.1], Polish decisions +5.4 [+4.7, +6.2], Polish general knowledge +5.7 [+4.3, +7.2], public English benchmark +10.8 (0.848 vs 0.740). Against basal-1.5 it is level on the sealed Polish test (0.933 vs 0.931, not significant) and higher on Polish decisions (+1.7 [+1.1, +2.3]) and Polish general knowledge (0.794 vs 0.741).

basal-1.5-mini against basal-1.0-1.5B (paired 95% intervals): sealed Polish test +2.1 points [+1.1, +3.2], Polish decisions +2.5 [+1.8, +3.3], Polish general knowledge +1.0 (not significant), public English benchmark +7.8 (0.753 vs 0.675).

- *PL sealed test*: 3,000 new Polish decisions, created after the 1.5 weights were frozen and scored once.
- *PL decisions*: the held-out Polish decision test of basal-1.0 (7,081 items, unseen templates, statutes and
  domains), kept for continuity, scored with the corrected labels. The basal-1.0 rows are re-served with this
  release's engine (the basal-1.0 report gives 0.884 / 0.849).
- *PL general*: Polish knowledge, exams and reading comprehension (3,079 items).
- *EN external panel*: unweighted mean accuracy over 10 public English decision tasks (WinoGrande, Financial
  PhraseBank, RAGTruth, JudgeBench, BIG-Bench Hard, TabFact, ContractNLI, Circa, Belebele, TruthfulQA as yes/no
  decisions), a seeded sample of each task's evaluation split, 3,842 items in all. This is our adaptation of the task
  list of a public decision-model card, not a reproduction of its protocol.
- *Public EN bench.*: the 231-item public English decision benchmark of basal-1.0's table (official harness). For 1.5
  it is in-domain: the training covers the same kinds of decisions (no benchmark item was used).
- *Coverage at 1% error*: the share of the 8,560 test decisions accepted automatically with the threshold shipped in
  `CALIBRATION.json` (temperatures and threshold fitted on one calibration split and certified on another with a
  one-sided 95% Clopper–Pearson bound for a target error of 1%, before testing), at 0.55% observed error for
  basal-1.5. Same procedure and test set for every row.

**The family on the release readouts** (fp32 readouts of the release checkpoints, the same procedure for every
model):

- **Held-out test:** basal-1.0-4.5B 0.858 → basal-1.5 0.875 (+1.7, 95% interval [+1.0, +2.4]); basal-1.5-max 0.893
  (+3.5 over basal-1.0-4.5B); basal-1.5-mini 0.832 against 0.829 for basal-1.0-1.5B (+0.4, not significant).
- **basal-1.5-max against basal-1.5:** Polish general knowledge test 0.799 vs 0.745; decided automatically
  at the 1% / 5% error targets 62.2% / 84.5% (at 0.58% / 4.1% observed error) vs 51.0% / 76.6% (0.55% / 3.7%).
- **basal-1.5-mini against basal-1.0-1.5B:** its gains are on the decision tasks: public decision tasks (Polish and
  English) 0.839 vs 0.707, Polish decisions dev 0.901 vs 0.869; Polish general knowledge test 0.671 vs 0.658. It
  decides 46.5% automatically at the 1% target (0.90% observed error) and 70.4% at the 5% target, where the observed
  error, 5.3%, is above the target.

The served numbers of max and mini follow in the table above.

On held-out development sets (fp32 readouts; not public benchmarks), basal-1.0 → basal-1.5: action selection
(held-out dev) 0.642 → 0.845, ordinal scores (held-out dev) 0.419 → 0.785, hard English decisions (held-out dev)
0.803 → 0.945; basal-1.5-max: 0.903, 0.867 and 0.972.

<details>
<summary>English external panel per task (basal-1.0-4.5B → basal-1.5, and basal-1.5-max)</summary>

| task | basal-1.0-4.5B | basal-1.5 | change | basal-1.5-max |
|---|---|---|---|---|
| Financial PhraseBank | 0.810 | 0.893 | +8.3 | 0.897 |
| TruthfulQA (yes/no per candidate) | 0.570 | 0.626 | +5.6 | 0.665 |
| TabFact | 0.597 | 0.653 | +5.6 | 0.750 |
| BIG-Bench Hard (21 multiple-choice tasks) | 0.509 | 0.559 | +5.0 | 0.641 |
| RAGTruth | 0.637 | 0.650 | +1.3 | 0.727 |
| Belebele (English) | 0.937 | 0.933 | −0.4 | 0.947 |
| WinoGrande | 0.647 | 0.643 | −0.4 | 0.710 |
| ContractNLI | 0.737 | 0.730 | −0.7 | 0.743 |
| JudgeBench | 0.577 | 0.560 | −1.7 | 0.553 |
| Circa | 0.693 | 0.630 | −6.3 | 0.740 |
| **mean** | **0.671** | **0.688** | **+1.6 [+0.1, +3.0]** | **0.737** (+6.9 [+5.3, +8.5]) |

300 items per task (BIG-Bench Hard 540, TruthfulQA 902 yes/no decisions), both option orders. Decision-model cards
report about 0.857 on their own versions of these tasks; their protocol and items differ, so the numbers are not
directly comparable.

</details>

### Speed

One decision = one question asked in **both option orders** (the default), median latency.

**basal-1.5, release check** (one H100, `basal-serve --mode fast`, bf16, HTTP load test): one question **13.7 ms** p50
/ 23.8 ms p95, **64.4 decisions/s** with 32 concurrent clients. basal-1.0 was not re-measured on that machine, so this
is not a per-question comparison with basal-1.0. The speed-up of basal-1.5 is for requests that ask several questions
about one state (SOAM, below); a single question gains nothing from it.

**basal-1.5-mini, release check** (same setup): one question 10.9 ms p50 / 15.9 ms p95, 155.8 decisions/s with 32 concurrent clients (basal-1.0-1.5B was not re-measured on that machine).

**Per GPU, measured with basal-1.0.** Engine-only numbers (`basal-bench`, offline, batch size 1) of the basal-1.0
weights and engine; basal-1.5 (4.5B) and basal-1.5-mini have the same architectures (each pair is fine-tuned from the
same Bielik base model), so the table shows what each GPU does with these model sizes. It is not a measurement of the
1.5 models, and it is not comparable with the HTTP numbers above:

| GPU | class | 4.5B `fast` (bf16) | 4.5B `fp8` | 1.5B `fast` |
|---|---|---|---|---|
| B300 SXM6 | server (Blackwell) | **8.8 ms**, 109 dec/s | 9.7 ms | **4.7 ms**, 250 dec/s |
| H100 80GB | server | 12.5 ms, 63 dec/s | 11.3 ms | 6.2 ms, 157 dec/s |
| RTX PRO 6000 Blackwell | workstation | 18.9 ms | 14.9 ms | 8.8 ms |
| RTX 5090 | consumer | 27.3 ms | 19.4 ms | 12.7 ms |
| DGX Spark (GB10) | desktop | 92.0 ms | **44.6 ms** | 34.3 ms (fp8: 18.1 ms) |

Offline numbers from `basal-bench` on a private 500-item sample (mean prompt about 360 tokens); throughput with 32
option-order passes per forward. `fast` keeps decisions practically identical to the fp32 reference (agreement
0.99–1.00); `fp8` changes about 2–4% of decisions. Measured with the 1.5 models: basal-1.5-max on one H100 (other
GPUs not measured for max; [details](docs/HARDWARE.md#11b-basal-15-max)); Apple Silicon with MLX (Apple M4 Pro, 24 GB):
basal-1.5 587 ms, basal-1.5-mini 196 ms, basal-1.5-max 1.36 s per decision. Every GPU we measured, and which mode to use where:
[docs/HARDWARE.md](docs/HARDWARE.md).

**Many questions per request (SOAM).** The state is computed once per request, however many questions are asked about
it. 4.5B model, same H100, same requests (Polish decision states, median 336 tokens), latency of one request through
the server path, p50 / p95:

| engine | 5 questions | 12 questions |
|---|---|---|
| basal-1.0 engine (one prompt per question) | 111.7 / 182.6 ms | 222.0 / 331.1 ms |
| **basal-1.5 engine (SOAM)** | **37.8 / 53.6 ms (3.0×)** | **80.3 / 103.9 ms (2.8×)** |

The answers are the same as with separate prompts (exact in fp32, also on the final basal-1.5 weights). For a single
question SOAM brings no gain. With the final basal-1.5 weights on the release machine, one request takes 18.3 / 37.0 /
76.9 ms p50 for 1 / 5 / 12 questions. These are in-process engine numbers, not comparable with the HTTP release check. How it works:
[docs/SOAM.md](docs/SOAM.md).

**Against API models.** On Werdykt, basal-1.5 served with SGLang on one H100 answers in
33.8 ms per decision (p95 270 ms; 31.2 ms in the basal engine's fast
mode) and handles 23.0 decisions/s, which costs $0.048 per 1,000 decisions on a rented H100.
The frontier API models take 2.0–5.5 s and $0.188–$15.82 per 1,000 decisions; Jev 1.13.0, a
hosted decision API, answers in 286 ms at $0.071 per 1,000 decisions. basal runs locally, on
your own hardware and with your data in place. The ranges include the systems marked †: their role in building Werdykt affects
accuracy, not speed or price (see the leaderboard above for every system).

## Quick start

### NVIDIA GPU: the basal engine (recommended)

Install with [uv](https://docs.astral.sh/uv/) into a fresh environment (no git needed; `pip install uv` if it is
missing):

```bash
uv venv --python 3.12 ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B --mode fast --port 8000
```

- **Install torch first, from the CUDA 12.8 index.** A plain `pip install basal` takes the newest torch from PyPI,
  which may be built for a newer CUDA than your GPU driver ("The NVIDIA driver on your system is too old"). The cu128
  build needs a driver that supports CUDA 12.8 or newer (`nvidia-smi` shows it top right). DGX Spark and B300: use the
  cu130 build ([docs/HARDWARE.md](docs/HARDWARE.md)).
- **Use a fresh environment on cloud GPU images** (RunPod, Lambda, …). Their system Python ships a `torchvision` built
  for another torch, which breaks `transformers` ("operator torchvision::nms does not exist"). basal does not need
  torchvision; in a fresh environment it is not installed.
- The first start in mode `fast` compiles the model (a few minutes); `--mode fast-nocompile` starts in seconds.
- Other models: `--model Remek/basal-1.5-max` or `--model Remek/basal-1.5-mini`.
- From a clone instead: `git clone https://github.com/rkinas/basal && cd basal && uv pip install -e ".[fp8]"`
  (after the torch line above).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {"dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
    "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}}}}'
```

Response (real output of basal-1.5-4.5B, one H100, `--mode fast`; probabilities and latency rounded):

```json
{
 "model": "basal-1.5-4.5B",
 "answers": {
  "dept": {
   "type": "choice",
   "choice": "online",
   "probabilities": {"cards": 0.001, "online": 0.998, "loans": 0.001},
   "confidence": 0.998
  }
 },
 "usage": {"input_tokens": 302, "output_tokens": 0, "questions": 1, "branches": 1, "latency_ms": 10.9}
}
```

Python (the package includes a small client):

```python
from basal.client import Basal
b = Basal("http://127.0.0.1:8000")
state = ("Zgłoszenie z oddziału w Gdańsku: od 7:40 nie działa żaden terminal płatniczy, klienci odchodzą od kas, "
         "kolejka na 30 osób. Obejście: tylko gotówka.")
a = b.score(state, "Jak pilne jest to zgłoszenie?",
            ["niska — można zaplanować", "średnia — w ciągu kilku dni", "wysoka — dziś", "krytyczna — natychmiast"])
print(round(a["score"], 2), a["probabilities"])          # expected level 0-3 and the distribution
print(b.yes_no(state, "Czy problem dotyczy płatności kartą?")["noul"])   # P(yes)
```

### vLLM or SGLang

Each in its own environment (they bring their own torch and transformers). The basal server, API and calibration
stay the same; the backend computes the letter probabilities from the option tokens' log-probabilities. A per-engine
calibration file (`CALIBRATION.<mode>.json`) is used when a model ships one; the 1.5 models ship `CALIBRATION.json`,
which these modes use as is.

```bash
# vLLM
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B --mode vllm --port 8000

# SGLang (its kernels need a CUDA 12.9+ toolkit)
uv pip install "sglang[srt]==0.5.21"
uv pip install --no-deps "basal @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
uv pip install starlette uvicorn httpx accelerate safetensors   # if missing
basal-serve --model Remek/basal-1.5-4.5B --mode sglang --port 8000
```

SGLang runs in its own environment: `sglang[srt]==0.5.21` pins its own transformers, so basal goes on top of it without
its dependencies. This is the setup the SGLang numbers were measured with. The basal engine's own modes (`fast`,
`eager`) and vLLM use the default install.

basal-1.5 with vLLM and SGLang (sglang 0.5.21) on one H100, offline (`basal-bench`, 1,000 development decisions, both option orders), vLLM agrees with the engine's fp32 reference on 0.994 of decisions (accuracy 0.928 vs 0.930, 147 decisions/s) and SGLang on 0.995 (accuracy 0.925 vs 0.930, 104.6 decisions/s, 14.1 ms p50 for one question); on Werdykt, SGLang gives the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens). These numbers are not comparable
with the HTTP numbers in [Speed](#speed). Evidence spans need the in-process model and are not available in these two modes; both start
with a 4,096-token context and refuse longer states; raise it with `--max-len` (e.g. `--max-len 32768` for long
documents). The basal engine's `fast` mode needs no setting.

### Apple Silicon (MLX)

Install the engine with the MLX extra (PyTorch comes from PyPI on a Mac) and serve an MLX port:

```bash
uv venv --python 3.12 ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx --port 8000
```

The request and response are the same as above (`curl` or the Python client). `-MLX-8bit` holds affine 8-bit decoder
weights (group 64) and `-MLX-fp4` MLX `nvfp4` decoder weights (4-bit floating point, group 16, FP8 scales); the token
embeddings and the output head stay in bf16, because the decision is read from the output head. Prefer `-MLX-8bit`
unless memory is tight. For basal-1.5-mini use `-MLX-8bit`: its `-MLX-fp4` port loses about 3.6 accuracy points (see
the [quality warning](#engines)). `--mode mlx` also loads a bf16 model directory. The server keeps two option orders, the
calibration and `multi` / `act` / `facts`; the state of a request is computed once into a KV cache shared by all its
questions. Evidence spans are not available in this mode. Agreement with the bf16 model and speed:
[docs/QUANTIZATION.md](docs/QUANTIZATION.md#ports-agreement-and-speed); how the ports work: [docs/PORTS.md](docs/PORTS.md).

With `basal[mlx]` installed, `mlx` is the default mode on a Mac. `--mode mlx-q8` makes 8-bit weights from a bf16
model at load time (about half the memory; the `-MLX-8bit` ports need no flag).

Without MLX, `--mode mps` runs the PyTorch engine on the GPU of the Mac with the packed requests of `fast` (no CUDA
graphs), with evidence spans: `basal-serve --model Remek/basal-1.5-mini --mode mps` (install torch from PyPI instead of
the CUDA index); `--mode eager --device mps` is the plain reference. On basal-1.5-mini both `mps` and `mlx` give the
bf16 reference's decisions on the bundled examples ([docs/HARDWARE.md](docs/HARDWARE.md#apple-silicon), which also
has an engine comparison on an M4 Max).

### Ollama (GGUF)

The basal server runs in front of Ollama: it renders basal's own prompt, sends it raw, reads the log-probabilities
of the option letters of one generated token, and keeps two option orders, the calibration and `multi` / `act` /
`facts`. Ollama 0.12.11 or newer (log-probabilities):

```bash
hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF
cd basal-1.5-4.5B-GGUF && ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0 && cd ..
uv pip install "basal @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0 --port 8000
```

`--model` points at the downloaded folder (or the `-GGUF` repository) for the tokenizer, chat template and calibration
file; Ollama serves the weights. The same files work with the llama.cpp server, which takes the engine's token ids
directly:

```bash
llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

Without a separate server, `--mode gguf` loads the file in process through llama.cpp (extra `basal[gguf]`, i.e.
llama-cpp-python; Metal, CUDA or CPU); the state of a request is computed once and shared by all its questions:

```bash
uv pip install "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
basal-serve --model ./basal-1.5-4.5B-GGUF --mode gguf --gguf basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf --port 8000
```

Quantisations: `Q8_0` and `Q4_K_M` (the output head and token embeddings stay at Q8_0). Requests of one state are sent
one after another, so the prompt cache of Ollama / llama.cpp computes the state once; `--http-parallel N` (with
`OLLAMA_NUM_PARALLEL` / `llama-server -np`) serves concurrent clients. Evidence spans are not available in these modes.
Details: [docs/PORTS.md](docs/PORTS.md), [docs/GGUF.md](docs/GGUF.md).

### Google TPU (JAX)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rkinas/basal/blob/main/notebooks/tpu.ipynb)
— [`notebooks/tpu.ipynb`](notebooks/tpu.ipynb) installs basal on a Colab TPU runtime, answers the bundled examples
and benchmarks latency, throughput and agreement with fp32.

On a TPU VM (Cloud TPU or a Colab TPU runtime), `--mode tpu` runs the model in JAX on plain-JAX kernels compiled by
XLA (RMSNorm, fused QKV, SwiGLU, RoPE and the prefix-trie attention mask). 
It keeps the packing of `fast`: the state computed once for every question and option order of a
request (SOAM), the same shape buckets and token-budget batching. Each (batch, length) bucket compiles to one XLA
executable, and the decoder layers run as one `lax.scan`. Install CPU torch (only the tokenizer and chat template use
transformers), then the `tpu` extra:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
basal-serve --model Remek/basal-1.5-4.5B --port 8000 --mode tpu
```

- **Speed** (TPU v5e-1, both option orders, the 44 bundled examples, `basal-bench`): basal-1.5 **17.7 ms** per
  decision, 49.5 dec/s; basal-1.5-mini **6.7 ms**, 146 dec/s. Over HTTP (`basal-loadtest`, 32 clients): 19.0 ms
  p50 and 36 dec/s for basal-1.5, 7.9 ms p50 and 112 dec/s for mini.
- **Many questions per request**: one state of 464 tokens with 1 / 5 / 12 questions takes 23.5 / 46.9 / 110.3 ms
  (basal-1.5) and 9.0 / 18.0 / 37.9 ms (mini) over HTTP, with the same answers as separate requests (12/12, max
  |Δp| 0.016 in bf16).
- **Agreement** with the fp32 transformers reference: 1.000 for basal-1.5 and basal-1.5-mini in bf16 (and for mini
  with `basal-bench --modes tpu-fp32`).
- **Memory.** basal-1.5 needs 9.6 GB of bf16 weights, which fits the 16 GB of one v5e chip; basal-1.5-max (22 GB)
  does not. `tpu-fp32` fits only mini.
- **Long documents.** As with vLLM / SGLang, `--max-len` (default 4,096 tokens, at most the model's positions) caps
  the prompt; longer states are refused with an error. Above 3,072 tokens the attention runs in blocks (online
  softmax), so memory grows with the length, not its square: basal-1.5 answers a 32k-token state in 12.5 s
  (8k: 1.5 s; mini at 8k: 0.5 s) with the weights and activations in about 9 GB. Start with e.g.
  `--max-len 32768` for long documents (each extra length bucket adds a few seconds of start-up).
- **Start-up.** The server compiles every bucket shape up to `--max-len` before it accepts requests (also the
  readout of requests with up to 32 questions), so no request waits for a compilation: about 3–3.5 minutes on a
  v5e. Executables are cached in `~/.cache/basal/jax` (or `$JAX_COMPILATION_CACHE_DIR`), so a restart takes well
  under a minute. `--mode tpu` fails at start-up if JAX sees no TPU, and accepts `--dtype bfloat16` or `float32`.
- **Not on TPU:** evidence spans (they need the in-process PyTorch model; the request is refused), early exit and
  quantisation.
- **Pin** `jax[tpu]==0.11.1` (needs Python 3.12 or newer). On Colab, the preinstalled jax ships a `libtpu` that
  cannot run code from its own `jaxlib`, so force-install the pinned pair.

### CPU

`basal-serve --model Remek/basal-1.5-mini --mode eager --device cpu` runs everywhere, for tests and low volumes.

## Inference on a JSONL file

`basal-run` sends every line of a JSONL file to a running server (concurrently; the server batches the requests) and
writes one answer per line, in the same order. Start a server first (`basal-serve ...`), then:

```bash
basal-run --input basal/examples/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
```

Each input line is either a **simple item** or a **full request** (the formats can be mixed):

```jsonc
// simple item: one question, options as a list; "type" defaults to "choice", "gold" (index) is optional
{"id": "t1", "state": "Klient: od wczoraj nie mogę zalogować się do bankowości ...", "question": "Do którego działu skierować zgłoszenie?",
 "options": ["Reklamacje kart", "Wsparcie bankowości elektronicznej", "Kredyty"], "gold": 1}
// yes/no item: options are [yes-text, no-text]
{"id": "t2", "type": "noul", "state": "...", "question": "Czy odstąpienie złożono w terminie?", "options": ["Tak", "Nie"]}
// full /v1/systemone request: several typed questions about one state
{"id": "t3", "state": {"ticket": "..."}, "questions": {"category": {"type": "choice", "instructions": "...", "criteria": {"complaint": "...", "other": "..."}},
                                                      "urgent": {"type": "noul", "instructions": "..."}}}
```

Simple items are sent with placeholder keys and `"option_keys": "hide"`, so the model sees exactly the option texts.
Each output line contains `id`, the full `answers` (probabilities, confidence) and the server `latency_ms`; simple items
also get `prediction` (index of the chosen option), `option`, `confidence` and, when `gold` is given, `correct`. At the
end `basal-run` prints a summary (items per second, median latency and accuracy if gold labels are present).
Ready-to-run examples (Polish and English) in `basal/examples/` (also installed with the package):

| file | what it shows |
|---|---|
| `choice.jsonl` | routing, document type, amounts, policy rules, sentiment, next step — one option out of 3–4 (with `gold`) |
| `noul.jsonl` | yes/no decisions: deadlines, approval thresholds, phishing, refunds, missing information, alerts (with `gold`) |
| `score.jsonl` | ordered scales: urgency, satisfaction, fraud risk, answer correctness; the answer includes the expected level |
| `complex.jsonl` | full requests: several typed questions about one JSON state (fan-out), a web-agent step with structured options, a deadline decomposed into simple questions, loan triage, a prompt-injection guard |
| `questions.jsonl` | 20 mixed items; with `choice`, `noul` and `score` the default set of `basal-bench` and `basal-loadtest` |
| `types_demo.py` | one request per decision type, sent to a running server, request and response printed |

```bash
for f in choice noul score complex; do basal-run --input basal/examples/$f.jsonl --output answers_$f.jsonl; done
```

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | hardware |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + SOAM packing + token-budget batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fp8` | `fast` with dynamic FP8 weights + activations (torchao) | Hopper, Blackwell (Ada: FP8 compilation stalled on an RTX 4090) |
| `nvfp4` | `fast` with NVFP4 weights + activations (torchao, experimental; costs accuracy) | Blackwell |
| `vllm` | vLLM backend, letter readout from the option tokens' log-probabilities | CUDA GPUs supported by vLLM |
| `sglang` | SGLang backend, same readout | CUDA GPUs supported by SGLang |
| `eager` | plain PyTorch reference; `--device cuda\|mps\|cpu` (default: `BASAL_DEVICE`, else CUDA, else Apple Silicon, else CPU) | any GPU, Apple Silicon (MPS) or CPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen per request | models that ship `exit_heads/` (basal-1.0 only; the 1.5 models do not) |
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` / `-MLX-fp4` ports or bf16 weights; shared-prefix KV cache | Apple Silicon |
| `mlx-q8` | `mlx` with 8-bit weights made from bf16 at load time (about half the memory, not faster) | Apple Silicon |
| `mps` | the PyTorch engine on the Mac's GPU: SOAM packing + token-budget batching, no CUDA graphs; evidence spans | Apple Silicon |
| `gguf` | a GGUF file in process through llama.cpp (`--gguf <file>`, extra `basal[gguf]`); SOAM as llama.cpp sequences | Apple Silicon (Metal), CUDA, CPU |
| `ollama` | the basal server in front of Ollama (GGUF ports; `--ollama-model`, `--ollama-url`) | anything Ollama runs on |
| `llamacpp` | the basal server in front of `llama-server` (GGUF ports; `--llamacpp-url`) | anything llama.cpp runs on |
| `tpu` *(default on a TPU VM)* | JAX / XLA bf16 + SOAM packing + token-budget batching, one compiled executable per shape ([Google TPU](#google-tpu-jax)) | Google TPU (`[tpu]` extra) |

- Without `--mode`: `fast` on CUDA, `mlx` on Apple Silicon (`mps` without `basal[mlx]`), `eager` otherwise.
  `--quant fp8|nvfp4` applies to the CUDA graph modes and `--quant q8` to `mlx`; other combinations are refused at
  start-up.
- `--calibration file.json` overrides the calibration file (default: `CALIBRATION.<mode>.json` when the model ships
  one for that engine, else `CALIBRATION.json`).
- **Two option orders** (`--orders 2`, default): every question is asked with the options in original and reversed
  order and the probabilities are averaged, which reduces sensitivity to option order. With SOAM both orders share the
  state and the question text, so the second order costs only its options. `--orders 1` is faster but less accurate.
- **SOAM** (`--soam on`, default): every question and option order of a request is one packed row over the state.
  `--soam off` restores the per-question path of basal-1.0 (same answers in fp32).
- **Start-up**: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (about 4–10
  minutes the first time for the 4.5B model, 2.5–4 minutes for the 1.5B, 10–17 minutes in `fp8`; much less with a warm
  compile cache). A packed request longer than the largest compiled shape (3,072 tokens) is split exactly into
  smaller rows; a single prompt longer than that is served with a normal forward pass.
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096; longer states are refused):
  `--max-len 32768` for long documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback). `--max-batch` and `--wait-ms` control batching.

## Benchmark your GPU

Two tools are installed with the package. Both run out of the box on the bundled examples (44 Polish and English items
with gold answers in `basal/examples/`):

```bash
# offline: latency, throughput and agreement of serving modes (no HTTP); the first mode is the reference
# (default: eager-fp32 and the serving default; on Apple Silicon bf16 mps and mlx)
basal-bench --model Remek/basal-1.5-4.5B --modes eager-fp32 fast fp8 --out bench.json

# end to end over HTTP against a running server (basal-serve ...)
basal-loadtest --url http://127.0.0.1:8000/v1/systemone
```

`basal-bench` loads the model once per mode and reports, per mode:

| column | meaning |
|---|---|
| `lat2 ms` | median latency of one decision with **both** option orders at batch size 1 (what the server does by default) |
| `lat1 ms` | the same with one option order |
| `dec/s` | two-order decisions per second when 32 option-order passes are processed together |
| `agree` | share of decisions whose top option equals the first mode's (use `eager-fp32` first; on Apple Silicon the default reference is bf16 `mps`, not fp32: `--modes eager-fp32 mps mlx` if memory allows; each JSON row records `reference_mode`) |
| `acc` | accuracy against `gold` |
| `GB` | CUDA / MLX peak allocation; MPS memory held by the Metal driver; empty for CPU and the GGUF modes |

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with 32
concurrent clients.

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers: latency depends on
prompt length, and accuracy on 44 items is noisy. For numbers that describe *your* workload, write a JSONL file with a
few hundred items in the same simple format (`{"state": ..., "question": ..., "options": [...], "gold": <index>}`,
`gold` optional) with realistic state lengths, and pass it with `--questions my_items.jsonl` (several files are
allowed). The speed tables in this README were measured with the same tools on our private 500-item test sample, which
we do not publish so that it cannot be trained on.

## API

The HTTP API implements the *System One* JSON interface (`POST /v1/systemone`, `GET /v1/models`); responses validate
against the official OpenAPI schema (`tests/test_server.py`). Clients that use this subset work by changing the base
URL (the official SDK was not tested; at most 10 options per question). `POST /v1/basal` is the same endpoint; the
extensions below (`multi`, `act`, `evidence`, `facts`, `option_keys`, `early_exit`) are accepted on both.

### Request

`POST /v1/systemone`

```jsonc
{
  "state": "text or JSON",                         // JSON values are serialised compactly into the prompt
  "questions": {
    "<name>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<description>", ...}},   // or a list of keys
    "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},  // criteria optional
    "<name>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]},      // lowest first
    "<name>": {"type": "multi",  "instructions": "...", "criteria": {...}, "threshold": 0.5, "min": 1, "max": 3},
    "<name>": {"type": "act",    "instructions": "...", "criteria": {...}, "costs": {...}, "max_error": 0.01},
    // optional on every question:
    //   "option_keys": "show" | "hide"   (default "show": options are shown as "key: description")
    //   "evidence": true                 (not for multi; experimental)
  },
  "facts": "off" | "auto",                          // optional, default "off" (Polish calendar / money facts)
  "early_exit": "off" | "0.999" | "0.995" | "0.99" | "0.98"   // optional, --mode fast-exit only
}
```

2–10 options per question. Several questions per request are answered over one shared state (SOAM); asking them in
one request is faster than separate requests and gives the same answers.

### Response

```jsonc
{
  "model": "basal-1.5-4.5B",
  "answers": {
    "<choice>": {"type": "choice", "choice": "<key>", "probabilities": {"<key>": p, ...}, "confidence": p},
    "<noul>":   {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p},
    "<score>":  {"type": "score", "score": expected_level, "legend": {"0": "level 0", ...},
                 "probabilities": {"0": p, ...}, "confidence": p},
    "<multi>":  {"type": "multi", "selected": ["<key>", ...], "probabilities": {"<key>": p_applies, ...},
                 "threshold": 0.5, "set_confidence": p},
    "<act>":    {"type": "act", "action": "<action>", "expected_costs": {"<action>": c, ...}, "answer": "<outcome>",
                 "probabilities": {...}, "confidence": p, "calibration": "validated" | "unvalidated"},
    // with "evidence": true, any of the above (except multi) also has
    //   "evidence": [{"text": "...", "start": 102, "end": 153, "probability": 0.93}, ...]
  },
  "usage": {"input_tokens": n, "output_tokens": 0, "questions": n, "branches": n, "latency_ms": t}
}
```

- `probabilities` are calibrated (per-type temperatures from the calibration file); `confidence` is the probability
  of the chosen answer.
- `score` is the expected level (0 = first level); `legend` maps level keys to their texts.
- `branches` counts the question branches of the request: one per question and one per label of a `multi`
  (each asked in both option orders by default).
- Errors (unknown type, too many options, a field not available in the server's mode) return HTTP 422 with
  `{"error": "..."}`.
- `GET /v1/models` returns `{"models": [{"name", "description", "release_date", "mode", "early_exit"}]}`
  (`release_date` is the engine's release date, not the model's);
  `GET /health` returns `{"status": "ok"}`.

### Question types

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options | route a ticket, pick the applicable rule, the next step of an agent |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |

**`multi`**: each label is asked as a yes/no branch over the shared state ("does label X apply?"). `selected` holds
the labels with probability ≥ `threshold` (default 0.5), at most `max` and at least `min`; `set_confidence` estimates
that the whole set is exactly right.

**`act`**: `criteria` are the possible outcomes (`{"true": ..., "false": ...}` for a yes/no question, or named options),
`costs` gives the cost of every action for every outcome (`{"approve": {"true": 0, "false": 1240}, "reject":
{"true": 300, "false": 0}, "human": {"true": 15, "false": 15}}`), or the short form `{"wrong": 99, "defer": 1}`
(answer with any outcome, wrong costs 99; deferring costs 1). The engine returns the action with the minimum expected
cost under the calibrated probabilities. With `"max_error": 0.01` it also refuses automatic actions below the
confidence threshold certified for 1% error (the response then names the `refused` action) and needs a `defer` /
`human` action (or `"defer_action": "<name>"`). `calibration` is `validated` only for prompt formats whose calibration
was measured (yes/no questions, or `"option_keys": "hide"`). See [docs/FEATURES.md](docs/FEATURES.md#act-cost-aware-actions).

**Option keys.** By default every described option is shown to the model as `key: description`
(`{"B": "Dispatch office: Krakow"}` → `B: Dispatch office: Krakow`), because only you know whether a key is meaningful
(a service code, a decision name); the server never guesses. If your keys are placeholders (`option_1`, `0`, …) and
the descriptions alone define the options, send `"option_keys": "hide"`: the model then sees the descriptions only
(descriptions that are not unique still get their key). A key without a description (`"criteria": {"approve": null}`
or a list of keys) is shown by itself. The same applies to named `score` levels.

**Evidence** (*experimental*): `"evidence": true` on a question returns up to three non-overlapping spans of the state
that support the answer, each with `start` / `end` character offsets into the request's state (the serialised JSON
for a structured state; spans in an appended facts block are dropped) and a probability. It costs one extra pass per
question that asks for it. See [docs/FEATURES.md](docs/FEATURES.md#evidence-spans).

**Facts**: `"facts": "auto"` appends a block of computed facts to a Polish state: the weekday of every date and
whether it is a Saturday or a statutory day off, gaps between dates, date + stated durations, net/gross amounts for
a stated VAT rate, euro × a stated exchange rate, budget remainders, quantity × price and list totals, and
comparisons of amounts (as symbols only, never a verdict). Only arithmetic and calendar facts are added, never a rule
or an answer. The models were trained with and without this block.

### Using the confidence

The server averages both option orders and then applies the per-type temperatures of the calibration file, which were
fitted on exactly that averaged prediction. The file also contains confidence thresholds chosen on calibration data,
before testing, for a target error of 1% and 5% among accepted decisions. For basal-1.5 they accept
51.0% / 76.6% of the test decisions at 0.55% / 3.7% observed error (max: 62.2% / 84.5% at 0.58% / 4.1%; mini: 46.5%
at 0.90% for the 1% tier, while its 5% tier gave 5.3% observed error, above the target: for mini, use the 1% tier when
the error budget matters). Accept
decisions above the threshold automatically and route the rest to a person; with your own data, refit the threshold
on a few hundred labelled requests. The thresholds are validated on descriptions-only prompts (`"option_keys":
"hide"`) and yes/no questions; the MLX and GGUF ports inherit the bf16 calibration, which is not validated for them
(mini's `-MLX-fp4` ships its own temperatures in `CALIBRATION.mlx.json`; its thresholds are copied from bf16).

### How to ask

- **One condition per yes/no question.** "Does the message mention health, finances or home address?" is harder than
  three questions or one `multi` with three labels.
- **Prefer `choice` with described sides** for a binary decision whose two outcomes have names ("approve: the invoice
  matches the order" / "hold: amounts or items differ").
- **Ask in the language of the state.** The prompt language is chosen from the text: Polish when the state or the
  question contains Polish diacritics, English otherwise (Polish without diacritics gets the English template).
- **Put the facts and rules in the state.** The model decides from what it reads; a small model's world knowledge is
  limited, and legal rules change.
- **Let code compute, let basal decide.** Exact arithmetic and thresholds are a weak spot of one-pass models: compute
  numbers in code (or send `"facts": "auto"` for Polish dates and amounts) and ask the typed question on top.
- **Offer "cannot be determined"** as an option when the state may not contain the answer. The models choose it
  when the information is missing or contradictory; when the relevant part of a document is complete and does not
  grant or state something, a yes/no question about it is answered "no".
- **Use `criteria` descriptions** that say what each option means, not only a label; keep keys stable and send
  `"option_keys": "hide"` when keys are placeholders.

## Limitations

- Evaluated on held-out and hidden test sets and on public benchmarks; claims about a specific document collection
  need validation on your own data.
- Polish world knowledge of a small model is limited; supply the relevant facts and rules in the state.
- basal-1.5-mini's 5% confidence tier is not met on the held-out test (5.3% observed error); use its 1% tier when the
  error budget matters.
- Legal rules change; the models do not know rules introduced after their training.
- Exact arithmetic and thresholds remain a weak spot of one-pass decisions; compute them in code or use `"facts":
  "auto"` for Polish dates and amounts.
- Averaging the original and reversed option order reduces, but does not remove, sensitivity to option order for
  three or more options.
- `multi` asks each label as an independent yes/no question; consistency between labels is not enforced.
- Evidence spans are experimental: the returned span is not always the one a person would quote, and removing it does
  not always change the decision ([docs/FEATURES.md](docs/FEATURES.md#evidence-spans)).
- The calibration is validated for the bf16 engine with descriptions-only prompts and for yes/no questions; for other
  prompt formats, ports and quantisations, refit thresholds on your own labelled requests.
- Decisions with serious consequences for people should be reviewed by a person.

## Previous release: basal-1.0

basal-1.0 (September 2026; 4.5B and 1.5B, plus FP8 / NVFP4 ModelOpt checkpoints) remains available on Hugging Face
(`Remek/basal-1.0-4.5B`, `Remek/basal-1.0-1.5B`). On its held-out Polish decisions it scored 0.884 (4.5B) and 0.849
(1.5B), 10.5 points above the best of eleven open decision systems at the time; on English decisions it was within the
95% intervals of the strongest systems. Its full tables, methods and limitations are in the
[basal-1.0 technical report](docs/basal-1.pdf) and the [v1.0.1 README](https://github.com/rkinas/basal/tree/v1.0.1).
The engine of this release serves basal-1.0 models unchanged (`--model Remek/basal-1.0-4.5B`); `fast-exit` and the
`vllm` mode with the ModelOpt checkpoints work as before. Apple Silicon conversions of basal-1.0 by Paweł Kiszczak
(GGUF F16 / Q8_0 / Q4_K_M, MLX 8-bit, oQ6e) are on [Hugging Face](https://huggingface.co/pawelkiszczak); how they
compare: [docs/HARDWARE.md](docs/HARDWARE.md#quantised-formats-mlx-omlx-oq-gguf).

## Citation

```bibtex
@software{kinas2026basal15,
  title   = {basal-1.5: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {1.5.0},
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
```

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models; see [NOTICE](NOTICE).
