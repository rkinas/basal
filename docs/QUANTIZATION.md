# Low precision and ports: FP8, MLX, GGUF

All routes read the same answer: the probabilities of the option letters at the answer position. Lower precision and
other runtimes change a small share of decisions, which we report as agreement of the top answer with the bf16 / fp32
model on the same items.

## What ships with basal-1.5

| format | repositories | runs on | how |
|---|---|---|---|
| bf16 safetensors | `Remek/basal-1.5-4.5B`, `Remek/basal-1.5-max`, `Remek/basal-1.5-mini` | NVIDIA GPUs (basal engine, vLLM, SGLang), Apple Silicon (MPS) and CPU (eager) | `basal-serve --model ...` |
| MLX 8-bit (affine, group 64) | `...-MLX-8bit` | Apple Silicon | `basal-serve --mode mlx` |
| MLX 4-bit (`nvfp4`, group 16, FP8 scales) | `...-MLX-fp4` | Apple Silicon | `basal-serve --mode mlx` |
| GGUF (Q8_0, Q4_K_M) | `...-GGUF` | Ollama, llama.cpp (CPU, Apple Silicon, consumer GPUs) | `basal-serve --mode ollama` / `--mode llamacpp` |
| FP8 (NVIDIA ModelOpt) | `...-FP8` | vLLM on Hopper, Ada and Blackwell | `basal-serve --mode vllm` |
| NVFP4 (NVIDIA ModelOpt) | `...-NVFP4` (4.5B, max) | vLLM on Blackwell | `basal-serve --mode vllm` |

For vLLM, basal-1.5 also ships FP8 / NVFP4 ModelOpt checkpoints ([below](#basal-15-the-modelopt-checkpoints)). In the
basal engine itself, the bf16 weights are quantised on the fly:

| precision | how | GPUs | agreement with fp32 (basal-1.0-4.5B) | basal-1.5 |
|---|---|---|---|---|
| bf16 | `--mode fast` | any CUDA GPU, sm80+ | ≥ 0.99 (identical decisions up to bf16 noise) | 0.997 |
| FP8 (on the fly) | `--mode fp8` (torchao dynamic FP8, per-row scales) | Hopper, Blackwell (Ada: FP8 compilation stalled on an RTX 4090) | ≈ 0.96–0.98 | not measured |
| NVFP4 (on the fly) | `--mode nvfp4` (torchao, experimental) | Blackwell | 0.90–0.91 (−3 accuracy points) | not recommended |

## Engines

The same table as in the README: which engine serves which format, and what is verified on basal-1.5.

<!-- engines -->
| engine | `basal-serve` mode | hardware | what it is for | quick start | status (basal-1.5, 4.5B) |
|---|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`, `fp8`, `nvfp4`); `mps`; `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon with `mps` (packed requests on the Mac's GPU, no CUDA graphs) or `eager`; CPU with `mps --device cpu` (packed requests) or `eager` | the primary server: packed requests (SOAM), evidence spans, `fp8` on Hopper and Blackwell | `basal-serve --model Remek/basal-1.5-4.5B --mode fast` | **verified**: served check on one H100, 13.7 ms p50 per question over HTTP, 64.4 decisions/s with 32 clients |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server | `basal-serve --model Remek/basal-1.5-4.5B --mode vllm` (extra `basal[vllm]`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.994 with the fp32 engine, accuracy 0.928 vs 0.930, 147 decisions/s |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (RadixAttention prefix sharing) | `basal-serve --model Remek/basal-1.5-4.5B --mode sglang` (own environment: `sglang[srt]==0.5.21`, then basal with `--no-deps`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.995 with the fp32 engine, accuracy 0.925 vs 0.930, 104.6 decisions/s; on Werdykt the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens) |
| **MLX** | `mlx` (default on a Mac with `basal[mlx]`); `mlx-q8` | Apple Silicon | Macs: `-MLX-8bit` (recommended) or `-MLX-fp4` (half the memory); `mlx-q8` makes 8-bit weights from bf16 at load time | `basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.992 (`-MLX-8bit`) / 0.970 (`-MLX-fp4`), 587 / 581 ms per question |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with Ollama (`-GGUF`) | 1. `hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF` 2. in that folder: `ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0` 3. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 518 / 542 ms per question |
| **llama.cpp** | `llamacpp`; `gguf` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files; the engine's token ids are sent as is. `--mode gguf --gguf <file>` runs the file in process instead (extra `basal[gguf]`, [GGUF.md](GGUF.md)) | 1. `llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080` 2. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 427 / 446 ms per question |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp run the weights; the typed answers (a calibrated probability for each of your options, both option orders, the state shared by several questions, `multi`, `act` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API in every mode. A plain `ollama run` or `llama-cli` gives free text only. Evidence spans need the PyTorch engine (`fast`, `eager`); vLLM and SGLang start with a 4,096-token context (longer states are refused); raise it with `--max-len` (e.g. 32768 for long documents). Install and first request for each engine: [install and run](INSTALL.md).
<!-- /engines -->

## Ports: agreement and speed

Apple M4 Pro, 24 GB, macOS 15.7; the uploaded files.

| model | port, engine | agreement with the reference | accuracy (change) | ECE | latency, 1 / 5 questions per state | memory |
|---|---|---|---|---|---|---|
| basal-1.5 (4.5B) | PyTorch MPS, bf16 (reference) | – | 0.936 | 0.030 | 817 ms / 3.99 s | 9.5 GB |
| basal-1.5 (4.5B) | MLX `-MLX-8bit` | 0.992 | 0.940 (+0.004) | 0.037 | 587 ms / 2.19 s | 5.2 GB |
| basal-1.5 (4.5B) | MLX `-MLX-fp4` (nvfp4) | 0.970 | 0.922 (−0.014) | 0.039 | 581 ms / 2.10 s | 2.9 GB |
| basal-1.5 (4.5B) | GGUF Q8_0, llama.cpp | 0.994 | 0.938 (+0.002) | 0.031 | 427 ms / 1.67 s | 5.1 GB |
| basal-1.5 (4.5B) | GGUF Q8_0, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 518 ms / 1.76 s | 5.1 GB |
| basal-1.5 (4.5B) | GGUF Q4_K_M, llama.cpp | 0.980 | 0.936 (±0.000) | 0.030 | 446 ms / 1.73 s | 2.9 GB |
| basal-1.5 (4.5B) | GGUF Q4_K_M, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 542 ms / 1.84 s | 2.9 GB |
| basal-1.5-max (11B) | GGUF Q8_0, llama.cpp (reference) | – | 0.948 | 0.033 | 1.16 / 3.67 s | 11.9 GB |
| basal-1.5-max (11B) | GGUF Q8_0, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 1.10 / 3.67 s | 11.9 GB |
| basal-1.5-max (11B) | MLX `-MLX-8bit` | 0.996 | 0.948 (±0.000) | 0.027 | 1.36 / 5.20 s | 12.1 GB |
| basal-1.5-max (11B) | MLX `-MLX-fp4` (nvfp4) | 0.972 | 0.928 (−0.020) | 0.025 | 1.32 / 4.92 s | 6.7 GB |
| basal-1.5-max (11B) | GGUF Q4_K_M, llama.cpp | 0.994 | 0.942 (−0.006) | 0.027 | 1.22 / 3.85 s | 6.8 GB |
| basal-1.5-max (11B) | GGUF Q4_K_M, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 1.16 / 3.84 s | 6.8 GB |
| basal-1.5-mini (1.5B) | PyTorch MPS, bf16 (reference) | – | 0.918 | 0.037 | 279 ms / 1.35 s | 3.2 GB |
| basal-1.5-mini (1.5B) | MLX `-MLX-8bit` | 0.998 | 0.916 (−0.002) | 0.037 | 196 / 713 ms | 1.8 GB |
| basal-1.5-mini (1.5B) | MLX `-MLX-fp4` (nvfp4) | 0.908 | 0.882 (−0.036) | 0.059 (0.046 with its `CALIBRATION.mlx.json`) | 197 / 728 ms | 1.0 GB |
| basal-1.5-mini (1.5B) | GGUF Q8_0, llama.cpp | 0.998 | 0.918 (±0.000) | 0.036 | 160 / 613 ms | 1.7 GB |
| basal-1.5-mini (1.5B) | GGUF Q8_0, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 186 / 647 ms | 1.7 GB |
| basal-1.5-mini (1.5B) | GGUF Q4_K_M, llama.cpp | 0.964 | 0.914 (−0.004) | 0.032 | 157 / 613 ms | 1.0 GB |
| basal-1.5-mini (1.5B) | GGUF Q4_K_M, Ollama | as llama.cpp | as llama.cpp | as llama.cpp | 196 / 641 ms | 1.0 GB |

Measured on 500 held-out Polish development decisions, both option orders, calibrated; reference = the same weights in
PyTorch bf16 on the same machine (for basal-1.5-max, which does not fit this Mac in bf16: GGUF Q8_0 with llama.cpp);
agreement = same top option as the reference; latency = median per decision (both option orders) through the basal
server path, with 1 and 5 questions per state.

Accuracy here is on a 500-decision sample, to check that each port matches its reference; to compare model sizes, see
the basal-1.5 benchmark table ([v1.5.0 README](https://github.com/rkinas/basal/tree/v1.5.0#other-results)).

4-bit ports trade accuracy for memory: on basal-1.5 the 4-bit MLX port costs about 1.4 points (4.5B), 2.0 points (max, against its GGUF Q8_0) and 3.6 points (mini); the 8-bit ports and GGUF Q8_0 stay within noise. Use 8-bit when accuracy matters.

> [!WARNING]
> **Quality warning.** The 4-bit MLX ports of basal-1.5-mini and basal-1.5-max lose accuracy, and their confidence thresholds are not certified: mini's about 3.6 points against bf16 (0.882 vs 0.918), max's about 2.0 points against its GGUF Q8_0 reference (0.928 vs 0.948). For mini, use `-MLX-8bit` or GGUF (`Q8_0` / `Q4_K_M`), whose accuracy stays within noise of bf16; for max, use `-MLX-8bit` (agreement 0.996, accuracy 0.948) or, with Ollama / llama.cpp, GGUF `Q4_K_M` (0.994, 0.942). For vLLM, the NVFP4 checkpoint of basal-1.5 (4.5B) loses about 2.8 accuracy points (0.902 vs 0.930; agreement 0.937); use `-FP8` unless memory or throughput matters more.

## MLX (Apple Silicon)

`basal-serve --mode mlx` runs the `-MLX-8bit` or `-MLX-fp4` ports, or a bf16 model directory, with MLX / mlx-lm on
the GPU of an Apple Silicon Mac (install `basal[mlx]`). The state of a request is computed once into a KV cache shared
by all its questions and option orders.

`--mlx-packed` (opt-in, off by default) runs a group of a Llama model (basal-1.x, 2.x) whose state and option orders fit
in 512 tokens as one packed forward instead: the state once, then every option order, with explicit positions and a
mask that lets each order see the state and itself only; several groups share one forward. Groups that do not fit,
Gemma 4 (basal-omni) and RoPE variants whose frequencies depend on the sequence length (dynamic, yarn, longrope) keep
the KV cache. In fp32 the decisions equal the KV-cache path; in bf16 and 8-bit the probabilities move slightly (a
different summation order), and the shipped `CALIBRATION.mlx.json` was fitted on the default path, so check
`--mlx-packed` against it before relying on the calibrated thresholds.

Our 2.3.1 checks on an Apple M4 Pro (24 GB), `--mlx-packed` against the default path:

| | basal-2.0-mini-MLX-8bit | basal-1.0-1.5B (bf16) | basal-2.0-4.5B-MLX-8bit |
|---|---|---|---|
| 2 orders, 44 bench examples (ms, two runs) | 118 / 124 -> 104 / 107 | 158 / 164 -> 136 / 130 | 379 / 390 -> 331 / 330 |
| decisions/s, two orders | 8.9 / 6.1 -> 9.5 / 8.1 | 6.4 / 6.4 -> 7.7 / 8.1 | 2.7 / 2.7 -> 2.9 / 3.0 |
| top answer = eager-fp32, per readout (frozen dev v2) | 0.9925 -> 0.9925 (4,000) | 0.9975 -> 0.9958 (2,400; McNemar p 0.29) | not run (fp32 does not fit in 24 GB) |
| same decision as the default path at margin >= 0.2 | 1,903 / 1,903 | 1,043 / 1,043 | 2,904 / 2,905 |
| largest probability change | 0.095 | 0.021 | 0.176 |
| automated at 5% with the shipped `CALIBRATION.mlx.json` (no type is certified at 1%) | 41.8% -> 42.1%, +0.25 pt [-0.10, +0.60] | (no MLX file) | 28.7% -> 28.5%, -0.20 pt [-0.60, +0.20] |

One order is unchanged (single prompts never pack), and no length band up to 512 tokens was slower than the default
path. The automated decisions made no error on either path. `--mlx-packed` stays opt-in in 2.3.1: on
basal-2.0-4.5B-MLX-8bit the lower bound of the change in the automated share at 5% (-0.60 pt) is below our -0.5 pt
bar for a change of path under a shipped calibration file.

`--mlx-forward basal` (opt-in, 2.5; the default `mlx-lm` is the path above) runs Llama (basal-1.x, 2.x) and Gemma 4
(basal-omni) through basal's own loop over mlx-lm's layers and weights. From the last layer with its own keys and
values on, only the readout rows run (that layer still computes its keys and values at every position): for Llama
the last layer, for Gemma 4 E2B / E4B the last 19 of 42 layers (the
last 18 reuse the keys and values of earlier layers). A shared state stops after that layer's keys and values. The
projections that read one input become one matrix where a self-check at load finds the same bits (on our models
gate / up; q / k / v stays separate). With every layer at every position this forward equals mlx-lm's bit for bit;
the readout rows only change the rounding. Our checks on an Apple M4 Pro (24 GB): 184 readouts per model (the 44
bundled items in two orders, and four questions in two orders over 1,000- and 3,000-token states), against PyTorch
eager on MPS, and the median of 7 (mini) or 5 (omni) interleaved runs:

| | basal-2.0-mini-MLX-8bit | basal-omni-1.0-mini-MLX |
|---|---|---|
| top answer = eager bf16, `mlx-lm` -> `basal` | 178 -> 178 / 184 | 181 -> 181 / 184 |
| top answer = default path | 184 / 184 | 184 / 184 |
| mean / largest probability change vs the default path | 0.0001 / 0.0008 | 0.0019 / 0.070 |
| one prompt (ms) | 93 -> 90 | 225 -> 131 |
| one question, two orders (ms) | 126 -> 121 | 222 -> 179 |
| four questions x two orders over 1,000 tokens (ms) | 972 -> 972 | 1,717 -> 1,462 |
| one prompt over 3,000 tokens (ms) | 1,877 -> 1,922 | 5,341 -> 3,003 |

In fp32 (basal-2.0-mini) both forwards give eager-fp32's top answer on all 184 readouts. The gain is where the skipped
layers are: basal-omni at its readout rows, little for Llama (one layer of 32). The shipped `CALIBRATION.mlx.json`
was fitted on the default path, so check `--mlx-forward basal` against it before relying on the calibrated
thresholds; on basal-omni the probabilities move more (8-bit matmuls of a few rows round differently).

basal-omni-pico (Qwen3.5, Gated DeltaNet) runs `--mode mlx` from its own repository's bf16 weights, computed in
float32 (2.5). `--mlx-compute bf16` (opt-in, off by default) keeps the weights in bf16 and computes in bf16: faster,
but further from float32. Apple M4 Pro, 1,700 held-out decisions, against CUDA eager float32:

| | `--mlx-compute auto` (float32, the default) | `--mlx-compute bf16` |
|---|---|---|
| single prompts: same top answer / changed where float32's margin >= 0.05 | 100.00% / 0 | 99.41% / 1 |
| five questions in two option orders (17,000 readouts) | 99.98% / 0 | 99.33% / 8 |

`--mlx-compute bf16` read the 1,700 single prompts in 0.136 s each on a 48 GB M4 Pro, 1.25x the GGUF Q8_0 path's
0.171 s on the same Mac. For comparison, CUDA's own bf16 path (eager) agrees with float32 on 99.48% of the 17,000
readouts, with 3 changed at a margin >= 0.05.

- `-MLX-8bit`: affine 8-bit decoder weights (group 64, a bf16 scale and bias per group);
- `-MLX-fp4`: MLX `nvfp4` decoder weights (4-bit floating point E2M1, group 16, FP8 scales);
- in both, the token embeddings and the output head stay in bf16, because the decision is read from the output head's
  logits of the option letters.

**Why these two formats.** On a development 4.5B checkpoint (500 decisions, Mac M4 Pro), the agreement of the top
answer with the bf16 model was 0.998 for affine 8-bit, 0.966 for `nvfp4`, 0.940 for MLX `mxfp8` and 0.938 for `mxfp4`:
MLX's 8-bit floating-point format changed as many decisions as the 4-bit ones, so the 8-bit port uses affine 8-bit, and
the 4-bit port uses `nvfp4`. Prefer `-MLX-8bit`; use `-MLX-fp4` when memory is tight. For basal-1.5-mini use `-MLX-8bit` (1.8 GB, agreement 0.998) or GGUF `Q4_K_M` (1.0 GB, 0.964); its `-MLX-fp4` port costs about 3.6 accuracy points (agreement 0.908, accuracy 0.882 vs 0.918). (`nvfp4` is
still the best MLX 4-bit format on mini.) Mini's `-MLX-fp4` ships its own temperatures in `CALIBRATION.mlx.json`; its
thresholds are copied from bf16 and not certified for that port.

```bash
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx --port 8000
```

## GGUF for Ollama and llama.cpp

The `-GGUF` repositories hold `Q8_0` and `Q4_K_M` files, one Ollama `Modelfile.<quant>` per file, the tokenizer,
chat template and calibration file. Two details of the export matter for decisions:

- **Tokenizer fields.** The stock llama.cpp conversion writes the same score (−1000) for every token of this
  tokenizer, so Ollama and other text clients split prompts differently from the original tokenizer (on a check of
  400 prompts, none was identical). The export sets the scores from the BPE merge ranks and `add_space_prefix =
  false`, and matches every added token (`<|im_start|>`, `<|im_end|>`, …) whole, as the Hugging Face tokenizer does:
  all 400 prompts are then tokenized exactly as by the original tokenizer, and `--mode ollama` and `--mode llamacpp`
  give the same answers.
- **`Q4_K_M`** is quantised with an importance matrix computed on decision prompts and keeps the output head and the
  token embeddings at Q8_0 (on a development 4.5B checkpoint the importance matrix raised agreement with bf16 from
  0.920 to 0.978; `Q8_0`: 0.990). basal is a
typed-decision model, not a chat model, so the basal server runs in front of the runtime: it renders basal's own prompt
with the answer prefix, sends it raw (Ollama 0.12.11 or newer: log-probabilities) or as token ids (llama.cpp), and reads
the option letters' log-probabilities of one generated token.

```bash
hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF
cd basal-1.5-4.5B-GGUF && ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0 && cd ..
basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0 --port 8000

llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

**Requirements.** MLX: `basal[mlx]` (mlx ≥ 0.32, mlx-lm ≥ 0.32), macOS 14 or newer on Apple Silicon (measured on
macOS 15.7). Ollama 0.12.11 or newer (log-probabilities; measured with 0.35.0). llama.cpp: measured with 0.5.0
(b11146). Every port supports both option orders, the calibration file, the shared state of several questions,
`multi`, `act` and `facts`; none supports evidence spans or early exit. On basal-1.5 (4.5B) the temperatures that fit
the ports best are 0.93–1.05 times the bf16 ones, so no per-engine calibration file is shipped; the one exception is
mini's `-MLX-fp4` (`CALIBRATION.mlx.json`).

How the ports are made and read, prefix sharing, tokens and options (`--http-parallel`, `--calibration`):
[PORTS.md](PORTS.md).

**Why through the basal server.** The basal server asks every question in two option orders, averages them and applies
the calibrated temperature of the question type; its thresholds for 1% and 5% error are fitted on exactly that
prediction. Calling a GGUF model directly with another prompt, one option order and raw probabilities gives different
answers and confidences. When we tested the basal-1.0-1.5B GGUF (Q8_0) on a Mac M4 Pro with llama.cpp, the conversion
itself was faithful (97.6% agreement with the H100 bf16 model with basal's own prompt, same accuracy), while a generic
decision prompt instead of basal's own format cost about 3 accuracy points. Even through the basal server, the
calibration of a quantised port is inherited, not validated: refit thresholds on your own labelled requests before
automating decisions with it.

## When does low precision help?

At batch size 1 a 4.5B model on a 400-token prompt is limited by kernel-launch overhead on server GPUs, by compute on
workstation and consumer GPUs, and by memory bandwidth on the DGX Spark. On Apple Silicon (M4 Pro) the prefill of a
decision is compute-bound: the 8-bit and 4-bit ports take about the same time (basal-1.5: MLX 8-bit 587 ms, fp4
581 ms; GGUF Q8_0 427 ms, Q4_K_M 446 ms) and only save memory. On NVIDIA GPUs, FP8 therefore helps most
where memory or compute is the bottleneck (DGX Spark: 92.0 → 44.6 ms, RTX PRO 6000: 18.9 → 14.9 ms, RTX 5090: 27.3 →
19.4 ms per decision, measured with the 4.5B architecture), little on the H100 (12.5 → 11.3 ms) and not at all on the
B300 (8.8 → 9.7 ms). The price is that 2–4% of decisions change. 4-bit formats change more decisions (about 10% for
NVFP4 on the 4.5B model, with −3 accuracy points; about 7 points on the 1.5B). Use bf16 when every decision counts, FP8
on workstation, consumer and desktop GPUs, and 4-bit formats where memory or throughput matters more than the last
points of accuracy.

## Prompt-format-2 models: the FP8 export

The `-FP8` checkpoints of the prompt-format-2 models are made with the same
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) post-training quantisation (`FP8_DEFAULT_CFG`,
calibrated on 2,048 prompts of the model's calibration decisions, nvidia-modelopt 0.47.0), with more of the model
kept in bf16:

- the token embeddings and the output head (the decision is read from its logits);
- **the first two and the last two decoder layers** (layers 0, 1, 30, 31 of the 1.5B model; 0, 1, 58, 59 of the 4.5B;
  0, 1, 48, 49 of the 11B). Of the exports compared on the 4.5B model, this one had the best accuracy and calibration
  at 40 decisions/s or more: a fully quantised decoder agrees with bf16 as often (0.950) and is 4% faster, but loses
  0.2 points more accuracy; keeping every `down_proj` in bf16 as well agrees more often (0.957) but costs 13% of the
  speed.

The basal engine serves them natively in the `fast` mode (`FP8Linear` on `torch._scaled_mm`), with the projections
**unmerged** (`--merge auto` keeps FP8 checkpoints unmerged): re-quantising a merged weight per tensor moves decisions
as much as the quantisation itself.

How an export is accepted: on development decisions with 2–10 options, the items basal-1.5's own `-FP8` checkpoint
answers too, its decisions must agree with the bf16 model's at least as often as basal-1.5's FP8 does with its bf16
(within 0.5 points), both overall and where the bf16 model's top two options differ by at least 0.2; accuracy within
1 point and ECE within 0.01 of the bf16 model. Agreement on questions with 11–255 options, which basal-1.5 cannot
answer, is reported separately. The 4.5B export: agreement 0.950 overall and 0.991 at a margin of at least 0.2
(basal-1.5 FP8: 0.947 / 0.991), accuracy −0.2 points, ECE 0.054 against 0.061; on 11–255 options agreement 0.90.

## basal-omni-1.0-ultra quantised at load time (2.5)

The CUDA graph engine can quantise ultra's bf16 weights (31B, 59 GB) as it loads them. Both modes are opt-in and
serve basal-omni-1.0-ultra only: on basal-omni-1.0-max (12B) both lost too much accuracy against bf16, so basal-serve
refuses them for every other basal-omni model before any weight is loaded.

| | block FP8 | NVFP4 |
|---|---|---|
| command | `--mode fp8` (block scaling is the default for basal-omni) | `--mode nvfp4`, with the calibrated input amax (`--nvfp4-amax <file>` or the model's `NVFP4_AMAX.json`) and its own calibration (`CALIBRATION.nvfp4.json` or `--calibration`) |
| GPUs | Hopper, Blackwell; 80 GB (the bf16 weights are loaded on the GPU before they are quantised) | Blackwell |
| numbers | weights: one scale per 128 x 128 block; activations: one scale per 128 features of each token | e2m1 values, one FP8 scale per 16 inputs and a global scale per tensor; activations with a fixed global scale from calibration, block scales per token |
| bf16 layers | the first and the last 2 decoder layers (`--fp8-bf16-layers`), embeddings and head | the same |
| top answer as bf16 (calibration test) | 99.3% | 97.6% |
| error of the answers accepted at the 1% target | 0.57% (bf16 0.47%) | 0.72% |
| speed | H100: 8.1 decisions/s single (bf16 6.0, 1.35x), 5.9 questions/s with several per request (1.23x); 52.9 GB peak at 16 clients (bf16 74.1 GB) | RTX PRO 6000 Blackwell: scoring 2.3x faster than bf16 on the same GPU |

Both kept the paired check against bf16 that a quantised mode has to pass (accepted answers at the 1% and 5% targets,
with confidence bounds), NVFP4 by a narrow margin. Use block FP8 when ultra has to fit one 80 GB GPU with room for
traffic; use NVFP4 when throughput on Blackwell matters more than the last half point of accuracy. `--nvfp4-scope mlp`
quantises the MLP only (1.6x); it was not better than the full variant.

The state cache is off in block FP8 until 2.5.1: `--state-cache-gb` is not used there (a warning at start-up), since
continuing a cached state gave NaN answers on ultra (H100). Requests are then served as without the cache.

### int8 for GPUs without FP8 (2.5)

Ampere GPUs (A100, A6000, RTX 30xx) have no FP8 tensor cores. `--mode int8` gives ultra the block layout of block FP8
with int8 values: weights in 128 x 128 blocks and activations in 128-feature tiles of each token, each with its own
scale (largest value / 127). The GEMM multiplies int8 by int8 into int32 for each 128-deep slice, which is exact, and
applies both scales in fp32. The first and the last 2 decoder layers stay bf16, as in FP8. It needs a GPU of compute
capability 8.0 or newer and the model's own calibration (`CALIBRATION.int8.json`, or `--calibration`). The weights take
about 31 GB.

| ultra, calibration test (8,560 decisions) | bf16 | int8 |
|---|---|---|
| top answer as bf16 | - | 99.4% |
| answers accepted at the 1% target | 56.8% | 56.7% (-0.16 points, 95% interval -0.40 to +0.08) |
| error of those answers | 0.47% | 0.52% |

It kept the same paired check as FP8 and NVFP4. On basal-omni-1.0-max (12B) the error of the accepted answers rose to
1.13% (target 1%), so int8 serves ultra only. Speeds on Ampere cards will come from users' measurements.

### Ultra on a 48 GB card: `--quant-load cpu` and `--graph-memory-gb` (2.5)

`--quant-load gpu` (the default) puts ultra's bf16 weights on the GPU before they are quantised, which needs an 80 GB
card. `--quant-load cpu` keeps them in host memory and moves one decoder layer at a time to the GPU, where it is
quantised, so the GPU holds about the quantised size: ultra in `--mode fp8` or `--mode int8` fits a 48 GB card. The
weights are the same as with the GPU load.

A GPU with less free memory runs smaller CUDA graph shapes, and its answers then differ from an 80 GB card's in the last
bits. `--graph-memory-gb N` sizes the shapes for N GiB of free memory after the weights instead of what is measured, so
every GPU given the same N runs the same shapes and gives the same answers. Measured on an H100 with 32 GB taken by
another process (about a 48 GB card): with `--mode fp8 --quant-load cpu --graph-memory-gb 10`, 1,700 of 1,700 answers
bit-identical to the same flags on the unloaded card, start-up 276 s, 5.9 decisions/s at 16 clients (single questions)
and 1.1 requests/s with five questions over long states, no errors.

```bash
basal-serve --model Remek/basal-omni-1.0-ultra --mode fp8 --quant-load cpu --graph-memory-gb 10 --port 8000
```


`Remek/basal-1.5-4.5B-FP8` / `-NVFP4`, `Remek/basal-1.5-max-FP8` / `-NVFP4` and `Remek/basal-1.5-mini-FP8` were produced with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) post-training quantisation
(`FP8_DEFAULT_CFG` / `NVFP4_DEFAULT_CFG`), calibrated on basal-1.5 calibration decisions, with the token embeddings and
the output head kept in bf16 (the decision is read from the output head's logits). FP8 runs on Hopper, Ada and Blackwell;
NVFP4 needs Blackwell (B200, B300, RTX 50xx, RTX PRO 6000, DGX Spark). basal-1.5-mini has no NVFP4 checkpoint: NVFP4
costs basal-1.5-mini about 6 accuracy points with almost no speed gain over FP8; use `-FP8`. Run them with vLLM in a separate environment:

```bash
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B-FP8 --mode vllm --port 8000
```

| model | checkpoint | GPU | agreement with the bf16 model | accuracy (change) | speed | size |
|---|---|---|---|---|---|---|
| basal-1.5 (4.5B) | `-FP8` | RTX PRO 6000 Blackwell | 0.983 | 0.931 (+0.001) | 17.2 ms, 143.5 dec/s | 4.9 GB |
| basal-1.5 (4.5B) | `-NVFP4` | RTX PRO 6000 Blackwell | 0.937 | 0.902 (−0.028) | 17.4 ms, 163.4 dec/s | 2.9 GB |
| basal-1.5-max (11B) | `-FP8` | RTX PRO 6000 Blackwell | 0.989 | 0.941 (−0.003) | 24.5 ms, 76.4 dec/s | 11.4 GB |
| basal-1.5-max (11B) | `-NVFP4` | RTX PRO 6000 Blackwell | 0.979 | 0.943 (−0.001) | 22.0 ms, 120.8 dec/s | 6.7 GB |
| basal-1.5-mini (1.5B) | `-FP8` | RTX PRO 6000 Blackwell | 0.979 | 0.903 (−0.013) | 8.8 ms, 337.2 dec/s | 1.7 GB |

Measured on 1,000 development decisions, both option orders, with vLLM 0.30.0 on one RTX PRO 6000 Blackwell (native CUTLASS FP8 / NVFP4 kernels), 2026-10-03; reference: the bf16 weights in fp32 PyTorch on the same GPU (accuracy 0.930 / 0.944 / 0.916 for basal-1.5 / max / mini); latency = one two-order decision at batch 1 (median), decisions/s with 32 option-order passes per call; size = checkpoint size. At batch 1 FP8 and NVFP4 are equally fast on this GPU; NVFP4 gains batched throughput. On max, NVFP4 is as accurate as FP8 (0.943 vs 0.941) at 58% more throughput. `CALIBRATION.json` carries the bf16 model's temperatures and thresholds; the thresholds are not
validated for these checkpoints, so refit them on your own labelled requests before automating decisions with them.

Reproducibility: the checkpoints were quantised with nvidia-modelopt 0.47.0, which requires transformers < 5.15, so the
quantisation environment pins transformers 5.14.1 (the basal engine pins 5.17.0); vLLM 0.30.0 serves the checkpoints
from its own environment.

## basal-1.0: the ModelOpt checkpoints

The `Remek/basal-1.0-4.5B-FP8` / `-NVFP4` and `Remek/basal-1.0-1.5B-FP8` / `-NVFP4` repositories were produced with
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) post-training quantisation (`NVFP4_DEFAULT_CFG` /
`FP8_DEFAULT_CFG`), with the output head kept in bf16 (the decision is read from its logits). They are standard
ModelOpt Hugging Face exports and load in vLLM (tested):

```bash
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"   # separate environment: vLLM brings its own torch
basal-serve --model Remek/basal-1.0-4.5B-NVFP4 --mode vllm --port 8000
```

The `vllm` mode asks vLLM for one token restricted to the option letters and reads the log-probabilities computed
*after* that restriction (`logprobs_mode="processed_logprobs"`), which is exactly the letter softmax of the other modes.
vLLM's prefix caching shares the state between the option orders and the questions of a request. The same mode serves
the bf16 basal-1.5 weights (`--model Remek/basal-1.5-4.5B --mode vllm`), and `--mode sglang` the same way. On one
H100, offline (`basal-bench`, 1,000 development decisions, both option orders), vLLM agrees with the engine's fp32 reference on 0.994 of decisions (accuracy 0.928 vs 0.930, 147 decisions/s) and SGLang on 0.995 (accuracy 0.925 vs 0.930, 104.6 decisions/s, 14.1 ms p50 for one question); on Werdykt, SGLang gives the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens). Neither ships its own
calibration file for 1.5; both use `CALIBRATION.json`.

SGLang runs in its own environment. Install it with `uv pip install --prerelease=allow "sglang[srt]==0.5.21"` (its kernels need a CUDA 12.9+ toolkit), then
`uv pip install --no-deps "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"` and, if missing, `uv pip install starlette uvicorn httpx accelerate
safetensors`. This is the setup the SGLang numbers were measured with; the basal engine's own modes (`fast`, `eager`)
and vLLM use the default install.
