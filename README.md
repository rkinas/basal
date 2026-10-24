# basal 2.4.1

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

- [What is new in basal 2.4.1](#what-is-new-in-basal-241) · [What is new in basal 2.4](#what-is-new-in-basal-24) · [What is new in basal 2.3.5](#what-is-new-in-basal-235) · [What is new in basal 2.3.4](#what-is-new-in-basal-234) · [What is new in basal 2.3.3](#what-is-new-in-basal-233) · [What is new in basal 2.3.2](#what-is-new-in-basal-232) · [What is new in basal 2.3.1](#what-is-new-in-basal-231) · [What is new in basal 2.3](#what-is-new-in-basal-23) ·
  [What is new in basal 2.2](#what-is-new-in-basal-22) · [What is new in basal 2.1](#what-is-new-in-basal-21) · [What is new in basal-2.0](#what-is-new-in-basal-20)
- [Models](#models) · [Results](#results)
- [Quick start](#quick-start) · [API](#api)
- [Images and audio: basal-omni](#images-and-audio-basal-omni-experimental) · [MCP for agents](#mcp-basal-as-a-tool-for-agents)
- [Documentation](#documentation): install paths, serving, API reference, results, formats, hardware
- [Limitations](#limitations) · [Previous releases](#previous-releases) · [Citation](#citation) ·
  [Contributors](#contributors) · [License](#license)

## What is new in basal 2.4.1

- **Shorter start-up** (`--mode fast`): every CUDA graph is warmed up once before its capture instead of two or three
  times. On an H100 with a warm compile cache start-up is 15-40% shorter (basal-omni-1.0-max 201 -> 121 s); every
  probability is the same. Contributed by Miłosz Kucharski (@Butterski), #43.
- **Prompt layouts** (`--layout`): models trained on them can read the question before the state (`qfirst`, one pass
  per question) or the state twice (`state_twice`); the canonical prompt is unchanged. Each layout has its own
  calibration file.
- All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.4

- **basal-omni-pico in the fast engine** (`--mode fast`, the default on an H100 with fla): Qwen3.5 models, whose Gated
  DeltaNet layers carry a recurrent state, now run with CUDA graphs and a request's state computed once for all its
  questions; 2.3.2 had to refuse them there. On one H100, against `--mode eager`: one question over a 1.5k-token
  state 13.5 ms instead of 47 ms (135 instead of 32 decisions/s at 16 clients), five questions 26.6 instead of
  111 ms (400 instead of 41 questions/s), five questions over a 3k-10k-token state 91 instead of 926 ms.
- **The same answers as `eager`**: in fp32 the top answer agrees on 99.94-99.99% of 1,700 held-out decisions; in
  bf16 `fast` is as close to the fp32 answers as `eager` is. Needs flash-linear-attention
  ([INSTALL](docs/INSTALL.md#nvidia-gpu-the-basal-engine-recommended)).
- Other models are unchanged (bit-identical to 2.3.5). All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3.5

- **Compress and many-question requests faster on CUDA** (`--mode fast`, basal-2.0, torch 2.13): the questions over a
  long state attend to it without a dense mask. `compress` over 1.5k-6k-token documents up to 1.5x faster, with less
  GPU memory; the same top answers. Contributed by Milosz (@Butterski), #39.
- **basal-2.0-4.5B on 16 GB cards** (`--mode fast`): the graph engine keeps 1.5 GiB of GPU memory free at start-up and
  leaves out the pass-2 graphs that do not fit. On 24 GB and larger cards nothing changes. Contributed by Miłosz
  Kucharski (@Butterski), #41.
- All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3.4

- **Less GPU memory for basal-omni** (`--mode fast`): its CUDA graphs are now captured largest first, as basal-2.0's
  are since 2.3.3. On one H100: basal-omni-1.0-mini 27.8 -> 18.9 GB, 1.0-max 45.3 -> 31.3 GB. This change alone
  leaves every probability and the speed as they were.
- **The same answer in every run** (basal-omni, `--mode fast`): torch.compile timed some kernels at start-up and kept
  the faster one, which changed the rounding, so two runs could differ on a near-tie. The kernels are now chosen
  without timing: a first run on an empty compile cache and later runs give bit-identical probabilities. The
  accuracy on 1,700 test items is the same as 2.3.3's (mini 82.5% -> 82.8%, max 85.1% -> 85.2%).
- **`--deterministic`** still runs one request per forward: batch-invariant kernels made basal-omni bit-identical in
  any batch, but they were slower than one request at a time, so they are not in this release.
- All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3.3

- **Less GPU memory in the graph engine** (`--mode fast`, basal-2.0): the CUDA graphs share their output buffers and are
  captured largest first. On one H100: basal-2.0-mini 10.8 -> 7.1 GB, the 4.5B 20.1 -> 15.2 GB; with compact rows
  16.6 -> 7.3 GB and 27.3 -> 15.4 GB, so the 4.5B with `--batches compact` now starts on a 24 GB card (#33). Every
  probability is the same as in 2.3.2.
- **Long states on the CPU** (`--mode mps --device cpu`): the state is computed once per request and the branch pass
  needs no dense mask: up to 1.5x faster on a 6-core CPU, with about half the peak memory on many-question requests.
- **Windows:** modes that need FlashAttention or Triton are refused before loading where the wheels lack them, and
  every text file is read and written as UTF-8.
- **basal-omni-pico** (0.8B) is in the model table: the smallest basal model, for a CPU or a small GPU.
- **From the community:** Milosz (@Butterski, #34, #35, #37), @lukaszpodgorski-pl (#38), Kamil (@kamilrosa, #36: Apple
  M5 Pro numbers). All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3.2

- **Ready for Qwen3.5 models** (Gated DeltaNet layers, such as the 0.8B pico): the modes that cannot keep their
  recurrent state (`fast`, `fast-nocompile`, `fast-exit`, `fp8`, `nvfp4`, `mps`) refuse them before loading, and
  without `--mode` they run in `eager`; 2.3.1 answered them wrongly in `fast`.
- **`--mode gguf` on hybrid models:** one prompt per decode, so the CPU no longer returns non-finite probabilities
  and requests with several questions work on CPU, CUDA and Metal.
- **`--mode fp8` on Ampere** stops before loading, with a clear message (#33).
- **From the community:** Grzegorz Brzezinka (@agentGreg, #31, #32: two fixes to the facts block),
  @krzysztofkaminski (#33: RTX 3090 numbers). All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3.1

- **Long states on one GPU:** basal-omni-1.0-ultra answers 76k- and 82k-token states on one 80 GB H100, in `--mode fast`
  (prefill in chunks) and in `--mode eager`, also while short requests are being served.
- **Incremental states** (`--state-cache-gb N`): a growing agent trace or chat history computes only its new tokens;
  step 12 of an agent loop 0.39 -> 0.19 s on basal-2.0-4.5B and 0.79 -> 0.27 s on basal-omni-1.1-max.
- **Faster multi-question requests:** no first-use compile pauses; five questions over a long state on basal-omni-1.0-mini
  21.9 -> 33.7 questions/s on an H100.
- **Blackwell workstation GPUs** run basal-omni's fast engine by default; basal-2.0 in `--mode eager` on Hopper answers
  in 52 ms instead of 591 ms.
- **For operators:** HTTP 422 / 503 / 500, `/health` with `running` and `queued`; `basal-calibrate` says when your labels
  are too few for a target.
- **From the community:** Milosz (@Butterski, #16, #20, #26, #29, #30), Grzegorz Brzezinka (@agentGreg, #18, #24, #25,
  #27, #28), @piotrkowski (#21), Kamil (@kamilrosa, #7), @edurdias (#22). All changes: [WHATS_NEW](docs/WHATS_NEW.md).

## What is new in basal 2.3

- **basal-omni on the fast engine** (`--mode fast`, text): the CUDA graph engine now runs basal-omni (Gemma 4). A
  request's state is computed once for every question, mini's last 18 layers run only where answers are read, and the
  layers are compiled with FlashAttention 3 and FlexAttention. Decisions per second at 16 requests in flight, against
  2.2's `--mode eager` on the same GPU:

  | model | GPU | one question | five questions about a long document |
  |---|---|---|---|
  | basal-omni-1.0-mini | H100 | 10.0 -> 61.2 (6.1x) | 1.6 -> 21.4 (13.7x) |
  | basal-omni-1.1-max | H100 | 4.8 -> 16.2 (3.4x) | 0.72 -> 10.6 (14.7x) |
  | basal-omni-1.0-ultra | H100 | 2.0 -> 6.1 (3.0x) | 0.28 -> 4.8 (17x) |
  | basal-omni-1.0-mini | RTX PRO 6000 Blackwell, `--gemma-attn flex` | 5.6 -> 23.2 (4.1x) | 0.85 -> 18.4 (21.6x) |

  The same answers within bf16 rounding: on 10,606 development decisions the top answer matches 2.2 on 99.85-99.99% of
  those where the model is sure, and with the shipped calibration files the share automated at 1% error moves by at
  most 0.1 points. Without `--mode`, basal-omni now runs `fast` on Hopper GPUs with FlashAttention 3 and `eager`
  elsewhere (in 2.2 it stopped at start-up). [docs/OMNI.md](docs/OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **A faster `--mode eager` for basal-omni** (images, audio, `--deterministic`): the state, and the image and audio
  encoders, run once per request instead of once per question: mini 1.6 -> 15.3 questions per second about a long
  document. Opt-in: `--eager-attn auto` (faster attention on Hopper) and `--ple-offload on` (mini in 5.2 GB less GPU
  memory).
- **vLLM for basal-omni:** mini's shared layers are skipped at prefill (1.43x / 1.60x, on by default);
  `--vllm-preset deterministic` gives bit-identical answers run to run; `--mode vllm --media on` for mini.
- **Fixes:** a PyTorch attention bug (pytorch/pytorch#191937) is avoided for every model in `--mode eager`; the prompt
  language is detected once per request (#17, Milosz, @Butterski).

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the
[release notes](https://github.com/rkinas/basal/releases/tag/v2.3.0).

## What is new in basal 2.2

- **Calibrate the model on your own data**, for every basal and basal-omni model: label a few hundred of your own
  requests, run `basal-calibrate`, and serve the file it writes. The probabilities, and the `max_error` thresholds that
  decide what may be automated, are then certified on your data, not ours. A hosted decision API cannot be recalibrated
  on your data; a basal model can. Measured on a domain the model never saw, the shipped file's "1% error" setting would
  have been wrong 8% of the time; after calibration the calibration error fell about 3.2 times for choice questions,
  2 times for scores and 30% for yes / no, and nothing was automated above its target
  ([the example](docs/CALIBRATION.md#example-a-new-domain-measured)).
  ```bash
  basal-calibrate my_labelled.jsonl --model Remek/basal-omni-1.0-max --mode eager    # writes CALIBRATION.custom.json
  basal-serve --model Remek/basal-omni-1.0-max --mode eager --calibration CALIBRATION.custom.json
  ```
- New in 2.2: the task types' `automated` flag (`classify`, `verify`, and now `extract`) is certified from your
  labelled task questions; a few dozen labels already move the temperatures; the report shows how the shipped file
  does on your data, as served; `--report-only`; a calibration file knows the model it was fitted on.
  [docs/CALIBRATION.md](docs/CALIBRATION.md)
- **basal-omni on llama.cpp, Ollama and Apple Silicon** (text only): `--mode gguf` runs a GGUF conversion of
  basal-omni-1.0 through llama.cpp (Metal, CUDA or CPU) and `--mode mlx` / `mlx-q8` the bf16 weights through mlx-lm.
  New model repositories for basal-omni-1.0-mini and -max: **`-GGUF`** (Q8_0 and Q6_K, with a calibration fitted on
  Q8_0) and **`-FP8`** (`--mode vllm`); 8-bit **`-MLX`** ports (`--mode mlx`) of 1.0-mini, 1.0-max and 1.1-max.
  [docs/OMNI.md](docs/OMNI.md#gguf-llamacpp-ollama-and-mlx-text)
- **From the community:** `basal-export-gguf` writes or repairs GGUF files so that llama.cpp and Ollama tokenize like
  the Hugging Face tokenizer (Piotr Durlej, #8); two experimental, community-maintained engines for basal-1.5 /
  basal-2.0: `--mode tpu` on Google TPU (Kuba Sołtys, #15) and `--mode strix` on AMD Strix Halo (Michał Kardyś, #14);
  `basal-viz`, a local page that shows every step of a decision (@lukaspoloki, #2,
  [SERVING.md](docs/SERVING.md#see-how-a-decision-is-made)). [Contributors](#contributors)

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the
[release notes](https://github.com/rkinas/basal/releases/tag/v2.2.0).

## What is new in basal 2.1

- **Task types** (*experimental*), on the same API and the models you already serve: **`classify`** (labels with a
  rubric each), **`verify`** (is each statement supported, contradicted or not stated), **`compress`** (the parts of a
  document a question needs, as shorter text for another model) and **`extract`** (a JSON schema filled with values
  copied exactly from the document, with character offsets). No text is generated: each task is expanded into
  decisions with calibrated probabilities. [docs/TASKS.md](docs/TASKS.md); every question type on one page:
  [docs/QUESTION_TYPES.md](docs/QUESTION_TYPES.md).
- Every other request takes the same code path as in v2.0.0: on basal-omni-1.0-max, 23,276 of 23,278 served-check
  answers were the same (the two others were near-ties).
- **New models on Sat 17 Oct, 09:00 CEST** (the links work from then): **basal-omni-1.0-ultra** (31B, Werdykt 0.879)
  with an FP8 port, and **basal-omni-1.1-max**, an experimental task model (basal-omni-1.0-max with extra training on
  the task formats; 1.0-max stays the default). See [Models](#models).

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the
[release notes](https://github.com/rkinas/basal/releases/tag/v2.1.0).

## What is new in basal-2.0

- **Three models**: **basal-2.0-mini** (1.5B), **basal-2.0** (4.5B) and **basal-2.0-max** (11B), in bf16, FP8, GGUF and
  MLX 8-bit.
- **Up to 255 options per question** (basal-1.5: 10), read in one forward pass with one option order.
- **Werdykt**, the hidden Polish–English decision benchmark: every tier is above its basal-1.5 counterpart, with a
  lower calibration error, and faster. Not everything improved: on a panel of 10 public English decision tasks all
  three are below basal-1.5 ([results](#results)).
- **basal-omni-1.0** (*experimental*): **basal-omni-1.0-mini** (4B) and **basal-omni-1.0-max** (12B), basal models built
  on Google Gemma 4 that answer the same questions and also read images and audio.
- **Engine v2.0.0**: compact batching and FlashAttention 3 on Hopper, native FP8 checkpoints, the `tool` question
  type for agents, an MCP server, `basal-calibrate` and `--deterministic` serving.
- **Thinking mode** (4.5B, *experimental*, off by default): it lowered accuracy on Werdykt; we recommend leaving it
  off ([docs/THINKING.md](docs/THINKING.md)).

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the [release notes](https://github.com/rkinas/basal/releases/tag/v2.0.0).

## Models

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-2.0-mini** | 1.5B | fastest; high-volume routing, laptops and small GPUs; context 8,192 tokens | [`Remek/basal-2.0-mini`](https://huggingface.co/Remek/basal-2.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-mini-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-mini-MLX-8bit) |
| **basal-2.0** | 4.5B | main model: best balance of accuracy and speed; experimental thinking mode | [`Remek/basal-2.0-4.5B`](https://huggingface.co/Remek/basal-2.0-4.5B) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-4.5B-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-4.5B-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-4.5B-MLX-8bit) |
| **basal-2.0-max** | 11B | most accurate; long documents, judging, hard English decisions | [`Remek/basal-2.0-max`](https://huggingface.co/Remek/basal-2.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-max-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-max-MLX-8bit) |

- Each repository ships its prompt format (`basal.json`) and calibration files. `-FP8`: NVIDIA ModelOpt FP8
  checkpoints for the basal engine (sm89+); `-GGUF` for Ollama and llama.cpp; `-MLX-8bit` for Apple Silicon. Every
  port is checked against its bf16 model; its card has the numbers. Files in detail:
  [docs/INSTALL.md](docs/INSTALL.md#what-the-model-repositories-contain).

**basal-omni** (experimental; text, images and audio; built on Google Gemma 4, pico on Qwen3.5):

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-omni-pico** | 0.8B (Qwen3.5-0.8B) | the smallest basal model: text decisions on a CPU or a small GPU (Werdykt 0.660); `--mode fast` on CUDA from basal 2.4 (with flash-linear-attention; 2.3.2-2.3.x: `--mode eager`), `--mode gguf` on the CPU | [`Remek/basal-omni-pico`](https://huggingface.co/Remek/basal-omni-pico) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-pico-GGUF) |
| **basal-omni-1.0-mini** | 4B (Gemma 4 E4B) | decisions about text, images and audio; about 16 GB of GPU memory | [`Remek/basal-omni-1.0-mini`](https://huggingface.co/Remek/basal-omni-1.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX) |
| **basal-omni-1.0-max** | 12B | the default 12B model (Werdykt 0.823); text, images and audio; about 24 GB of GPU memory | [`Remek/basal-omni-1.0-max`](https://huggingface.co/Remek/basal-omni-1.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX) |
| **basal-omni-1.1-max** | 12B | *experimental task model*: better on `classify`, `verify`, `compress` and `extract`, slightly behind 1.0-max on general decisions (Werdykt 0.821) | [`Remek/basal-omni-1.1-max`](https://huggingface.co/Remek/basal-omni-1.1-max) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX) |
| **basal-omni-1.0-ultra** | 31B (Gemma 4 31B) | highest Werdykt score of the basal models (0.879); text and images (no audio); 59 GB of bf16 weights | [`Remek/basal-omni-1.0-ultra`](https://huggingface.co/Remek/basal-omni-1.0-ultra) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-ultra-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-ultra-GGUF) |

- bf16 weights, served by `basal-serve --mode eager` on an NVIDIA GPU (`--media on` for images and audio).
  The `-FP8` checkpoints (mini 12 GB, max 14 GB, ultra 33 GB) are served by `--mode vllm`, text only, each with its own
  calibration (`CALIBRATION.vllm.json`). The `-GGUF` files (Q8_0 and Q6_K for mini and max, Q8_0 for ultra and pico) are served by `--mode gguf`
  (llama.cpp, with `CALIBRATION.gguf.json`) or in front of Ollama, text only. The `-MLX` ports (8-bit; 1.0-mini, 1.0-max
  and 1.1-max) are served by `--mode mlx` on Apple Silicon, text only, with `CALIBRATION.mlx.json`.
- All models are fine-tuned from Apache-2.0 base models (Bielik v3 for basal-2.0, Gemma 4 for basal-omni; see
  [NOTICE](NOTICE)) and released under Apache-2.0.

Which one to choose: start with **basal-2.0 (4.5B)**. Use **max** when accuracy on long or hard inputs matters more
than latency, and **mini** for high volumes, laptops and small GPUs (its context is 8,192 tokens). All three use the
same API, prompt format and calibration procedure. Use **basal-omni** (experimental) when a request carries images or
audio, or when the highest accuracy matters more than speed and an eager-mode server is enough.

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
(mini) and 0.823 (max) on Werdykt. Hosted API models and larger open decision systems score above basal-2.0: the
[board](https://basal.si5.pl/benchmark/) lists every tested system with its accuracy, latency and cost. Intervals,
calibration, other benchmarks and speed: [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

NVIDIA GPU, with [uv](https://docs.astral.sh/uv/) in a fresh environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

- **`--template-prefix off` for the 4.5B model** (bf16 and `-FP8`); mini and max run with the default.
- **Install torch first, from the CUDA 12.8 index**: a plain `pip install basal` may take a torch built for a newer
  CUDA than your driver. DGX Spark and B300: the cu130 build ([docs/HARDWARE.md](docs/HARDWARE.md)).
- The first start in `--mode fast` compiles the model (several minutes); `--mode fast-nocompile` starts in seconds.

Apple Silicon:

```bash
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000
```

Ask questions with curl. Several questions about one state go into one request and share the state:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

Each answer has a probability for every option, the chosen key and the confidence:

```jsonc
{
 "model": "basal-2.0-4.5B",
 "answers": {
  "dept":   {"type": "choice", "choice": "<key>", "probabilities": {"cards": p, "online": p, "loans": p}, "confidence": p},
  "urgent": {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p}
 },
 "usage": {"input_tokens": n, "output_tokens": 0, "questions": 2, "branches": 2, "latency_ms": t}
}
```

| other ways to run | |
|---|---|
| FP8 checkpoints (Hopper, Blackwell) | [docs/INSTALL.md](docs/INSTALL.md#fp8-checkpoints-basal-engine) |
| vLLM or SGLang with bf16 weights | [docs/INSTALL.md](docs/INSTALL.md#vllm-or-sglang-with-bf16-weights) |
| Ollama and llama.cpp (GGUF) | [docs/INSTALL.md](docs/INSTALL.md#ollama-and-llamacpp-gguf) |
| Apple Silicon in detail, CPU | [docs/INSTALL.md](docs/INSTALL.md#apple-silicon-mlx) |
| images and audio (basal-omni) | [below](#images-and-audio-basal-omni-experimental) |
| all seven engines compared | [docs/INSTALL.md](docs/INSTALL.md#engines) |
| a JSONL file, serving flags, benchmarking your GPU | [docs/SERVING.md](docs/SERVING.md) |
| Python client | [docs/INSTALL.md](docs/INSTALL.md#nvidia-gpu-the-basal-engine-recommended) |

## API

`POST /v1/systemone` with a `state` (text or JSON) and named, typed `questions`; every answer has a calibrated
probability for each option and a `confidence`. Up to 255 options per question (basal-1.x models: 10); several
questions over one state are answered in one pass.

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |

To send uncertain decisions to a person, use `act` or `tool` with `"max_error": 0.05`: the answer says
`"deferred": true` below the certified confidence threshold. Request and response fields, every question type,
using the confidence, calibrating on your own data and how to ask: [docs/API.md](docs/API.md).

## Images and audio: basal-omni (experimental)

The basal-omni models answer the same questions, and with `--media on` a request may also carry images and audio
(`"media"`):

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000   # or Remek/basal-omni-1.0-max
```

They were trained on text decisions only: Gemma 4's encoders read the media zero-shot and the answers use the text
calibration, so validate them on your own media. Request format and limits:
[docs/INSTALL.md](docs/INSTALL.md#images-and-audio-basal-omni-experimental); memory and engine details:
[docs/OMNI.md](docs/OMNI.md).

## MCP: basal as a tool for agents

*Experimental.* With the `basal[mcp]` extra, `basal-serve --mcp on` also serves the decisions over the
[Model Context Protocol](https://modelcontextprotocol.io) at `/mcp`, with the read-only tools `decide` and
`decide_many`; `basal-mcp` is a stdio bridge for hosts that start local processes.

```bash
uv pip install "basal[fp8,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

No authentication, like the HTTP API: bind `--host 127.0.0.1` or put a gateway in front. Details:
[docs/MCP.md](docs/MCP.md).

## Documentation

| | |
|---|---|
| [INSTALL](docs/INSTALL.md) | engines, the model files and every install path: NVIDIA (bf16, FP8), vLLM, SGLang, Apple Silicon, Ollama / llama.cpp, CPU, basal-omni |
| [API](docs/API.md) | request, response, question types, using the confidence, how to ask |
| [SERVING](docs/SERVING.md) | serving modes and flags, inference on a JSONL file, benchmarking your GPU |
| [RESULTS](docs/RESULTS.md) | results against basal-1.5: Werdykt, other benchmarks, speed |
| [MCP](docs/MCP.md) · [THINKING](docs/THINKING.md) · [OMNI](docs/OMNI.md) | MCP server and bridge; the experimental thinking mode; basal-omni on the engine |
| [FEATURES](docs/FEATURES.md) · [CALIBRATION](docs/CALIBRATION.md) | `multi`, `act`, `tool` and evidence in detail; calibrating on your own data |
| [QUESTION_TYPES](docs/QUESTION_TYPES.md) | all question types on one page: what each asks and answers, with an example |
| [TASKS](docs/TASKS.md) | *experimental*: `classify`, `verify`, `compress`, `extract` (JSON schema) on the same API |
| [FORMATS](docs/FORMATS.md) · [FORMAT2](docs/FORMAT2.md) · [SOAM](docs/SOAM.md) | prompt formats, option orders and calibration files; the runtime contract; shared-state packing |
| [HARDWARE](docs/HARDWARE.md) · [PORTS](docs/PORTS.md) · [GGUF](docs/GGUF.md) · [QUANTIZATION](docs/QUANTIZATION.md) | which mode on which GPU; MLX and GGUF ports; FP8 and other low precision |
| [WHATS_NEW](docs/WHATS_NEW.md) · [basal-1.0 report](docs/basal-1.pdf) | the full list of changes in 2.0; the basal-1.0 technical report |

## Limitations

- Evaluated on held-out and hidden test sets and on public benchmarks; claims about a specific document collection
  need validation on your own data.
- On a panel of 10 public English decision tasks all three basal-2.0 models are below basal-1.5 (−2.1 / −3.6 / −1.8
  points for mini / 4.5B / max); basal-2.0-mini is also below basal-1.5-mini on Polish decisions (−1.4 points).
- The calibration is cautious on unseen tasks: `max_error: 0.01` defers every decision with the shipped files (except
  `choice` with `basal-2.0-max-GGUF`'s `CALIBRATION.ollama.json`, confidence 0.989); use 0.05 or fit thresholds on your
  data. At the 1% target on Werdykt, basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.
- The thinking mode is experimental: on Werdykt it is less accurate, less calibrated and slower than the default.
- Polish world knowledge of a small model is limited; supply the relevant facts and rules in the state. Legal rules
  change; the models do not know rules introduced after their training.
- Exact arithmetic and thresholds remain a weak spot of one-pass decisions; compute them in code or use `"facts":
  "auto"` for Polish dates and amounts.
- basal-2.0-mini has an 8,192-token context; longer states are cut in the middle (or refused with
  `--context-overflow refuse`).
- `act`, `tool`, `multi` over many labels and evidence spans are experimental; `multi` asks each label as an
  independent yes/no question.
- basal-omni is experimental: trained on text decisions only, with images and audio read zero-shot by Gemma 4's
  encoders and answered with the text calibration (`"calibration": "text"`); validate it on your own media. It runs
  in `eager` mode (ultra's FP8 port: `vllm` mode, text only), without the thinking mode of basal-2.0.
- The task types of 2.1 are experimental: their API may change in a minor release, and `automated` needs thresholds
  certified for the task (only basal-omni-1.1-max ships one, `classify` at 5%).
- Decisions with serious consequences for people should be reviewed by a person.

## Previous releases

**basal-1.5** (October 2026; mini 1.5B, 4.5B, max 11B, with FP8 / NVFP4, MLX 4-bit / 8-bit and GGUF ports) remains
available on Hugging Face ([collection](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20))
and is served unchanged by this engine (`--model Remek/basal-1.5-4.5B`; two option orders by default). Its release
README, with the full Werdykt board of that release: [v1.5.0](https://github.com/rkinas/basal/tree/v1.5.0).

**basal-1.0** (September 2026; 4.5B and 1.5B, plus FP8 / NVFP4 ModelOpt checkpoints) remains available on Hugging Face
(`Remek/basal-1.0-4.5B`, `Remek/basal-1.0-1.5B`). Its full tables, methods and limitations are in the
[basal-1.0 technical report](docs/basal-1.pdf) and the [v1.0.1 README](https://github.com/rkinas/basal/tree/v1.0.1).
The engine serves basal-1.0 models unchanged; `fast-exit` and the `vllm` mode with the ModelOpt checkpoints work as
before. The Apple Silicon engine modes (`mps`, `gguf`, `mlx-q8`) and the Apple Silicon engine study were contributed
by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)).

## Development commands

Run `make` or `make help` to list commands. The Makefile uses `.venv/bin/python` by default; override `PYTHON` to
use another prepared environment. Create it with `uv venv --python 3.12 .venv`, then install the hardware-specific
PyTorch build described in [Quick start](#quick-start) before installing the app. For example:

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
torchao modes require compatible NVIDIA hardware. See [Serving modes](#serving-modes) for backend limitations.

```bash
make serve                                      # automatic backend selection
make serve MODE=mlx MODEL=Remek/basal-1.0-1.5B
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

The [pre-commit configuration](.pre-commit-config.yaml) provides fast, read-only checks for Python syntax and
critical errors (Ruff `E9`, `F63`, `F7`, `F82`), merge conflicts, private keys, TOML, YAML, JSON, and JSONL.
It covers the engine, server, clients, benchmarks, examples, tests, and figure scripts. JSON checks include the
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
```

basal-omni-1.0 (text, images and audio):

```bibtex
@software{kinas2026basalomni10,
  title   = {basal-omni-1.0: typed-decision models for text, images and audio in Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {1.0},
  url     = {https://huggingface.co/collections/Remek/basal-omni-10-6acea28e8b5baed2db475c1f}
}
```

## Contributors

Thank you to the people whose work is in this repository, by pull request:

- Paweł Kiszczak ([@pawelkiszczak](https://github.com/pawelkiszczak)): the Apple Silicon engines `--mode mlx` /
  `mlx-q8` and `--mode mps`, the in-process GGUF engine `--mode gguf` (llama-cpp-python) and `--mode ollama`, and the
  comparison of Apple Silicon engines and quantised formats (MLX, oMLX oQ, GGUF) with its figures in
  [HARDWARE.md](docs/HARDWARE.md#apple-silicon) ([#4](https://github.com/rkinas/basal/pull/4)).
- Piotr Durlej ([@pdurlej](https://github.com/pdurlej)): `basal-export-gguf`, which converts a basal model to GGUF, or
  repairs an existing GGUF file, so that its tokenizer matches the Hugging Face one, as in the official `-GGUF` files
  ([#8](https://github.com/rkinas/basal/pull/8)).
- Kuba Sołtys ([@qooba](https://github.com/qooba)): the experimental Google TPU engine `--mode tpu` in JAX / XLA, with
  its Colab notebook ([#15](https://github.com/rkinas/basal/pull/15)).
- Michał Kardyś ([@kardysm](https://github.com/kardysm)): the experimental `--mode strix`, a basal-1.5 / basal-2.0
  forward with Triton kernels for AMD Strix Halo, and its benchmarks
  ([#14](https://github.com/rkinas/basal/pull/14)).
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
  while GPU memory stays free ([#41](https://github.com/rkinas/basal/pull/41)).
- Grzegorz Brzezinka ([@agentGreg](https://github.com/agentGreg)): the calibration report's per-type ECE as
  `basal-serve` applies the temperatures ([#24](https://github.com/rkinas/basal/pull/24)), the per-type label count
  for the 1% target in CALIBRATION.md ([#25](https://github.com/rkinas/basal/pull/25)), the Apple M5 Max
  benchmark tables in [HARDWARE.md](docs/HARDWARE.md) ([#18](https://github.com/rkinas/basal/pull/18)), and
  `--mlx-packed`, two option orders in one forward on Apple Silicon, with its root cause
  ([#27](https://github.com/rkinas/basal/issues/27), [#28](https://github.com/rkinas/basal/pull/28)), and two fixes
  to the facts block: amounts with a dot thousands separator ([#31](https://github.com/rkinas/basal/pull/31)) and
  "roku" after a date, which is not a one-year duration ([#32](https://github.com/rkinas/basal/pull/32)).
- [@piotrkowski](https://github.com/piotrkowski): `--mode strix` reproduced on a second Strix Halo (ROG Flow Z13),
  bit-identical edge cases and a similar speed-up ([#21](https://github.com/rkinas/basal/pull/21)).
- [@lukaszpodgorski-pl](https://github.com/lukaszpodgorski-pl): basal on Windows: kernel checks for torch builds without
  FlashAttention or Triton, and UTF-8 for every text file ([#38](https://github.com/rkinas/basal/pull/38)).

Measurements in the docs, from issues: Kamil ([@kamilrosa](https://github.com/kamilrosa)), Apple M5 Pro numbers for
basal-1.0, basal-2.0 and basal-omni ([#7](https://github.com/rkinas/basal/issues/7),
[#36](https://github.com/rkinas/basal/issues/36)), and [@krzysztofkaminski](https://github.com/krzysztofkaminski), the
first Ampere numbers on an RTX 3090 ([#33](https://github.com/rkinas/basal/issues/33)).

`--mode tpu` and `--mode strix` are community-maintained: their numbers in [HARDWARE.md](docs/HARDWARE.md) are the
contributors' own measurements.

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models (Bielik v3, SpeakLeash and ACK Cyfronet AGH, for
basal-2.0; Gemma 4, Google, for basal-omni); see [NOTICE](NOTICE).
