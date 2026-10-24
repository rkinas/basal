# Hardware: which mode on which GPU

**What these numbers are.** The tables below were measured with the **basal-1.0 weights and engine** (offline,
`basal-bench`). basal-1.5 (4.5B) has the architecture of basal-1.0-4.5B and basal-1.5-mini that of basal-1.0-1.5B
(each pair is fine-tuned from the same Bielik base model), so the tables show what each GPU does with these model
sizes and which mode to choose; they are not measurements of the 1.5 models. The *accuracy* and *agreement* columns are
those of the basal-1.0 weights on our speed sample; for the accuracy of the 1.5 models see the
[v1.5.0 README](https://github.com/rkinas/basal/tree/v1.5.0#performance). basal-1.5-max (11B) and Apple Silicon have their own sections below.

**basal-1.5 release check** (one H100, `basal-serve --mode fast`, bf16, HTTP load test): one question 13.7 ms p50 /
23.8 ms p95, 64.4 decisions/s with 32 concurrent clients (basal-1.0 was not re-measured on that machine).
**basal-1.5-mini, release check** (same setup): one question 10.9 ms p50 / 15.9 ms p95, 155.8 decisions/s with 32 concurrent clients (basal-1.0-1.5B was not re-measured on that machine). Do not compare these HTTP numbers with the offline tables below.

One decision = **both option orders** (the default), batch size 1 unless stated. *dec/s*: two-order decisions per
second with 32 option-order passes per forward (offline) or 32 concurrent clients (HTTP). *Agreement*: share of
decisions whose top option equals the fp32 reference on the same machine (RTX 5090 and RTX 4090: the bf16 reference,
marked ²). Measured with `basal-bench` / `basal-loadtest` (H100, RTX PRO 6000 and RTX 5090 with the equivalent research
harness) on a private 500-item sample of the Polish/English test split. Requests with several questions are much
cheaper than separate requests since basal-1.5: see [SOAM.md](SOAM.md).

## Engines

Which engine runs where, and what is verified on basal-1.5 (the per-GPU tables further down are basal-1.0 engine-only
measurements):

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

## Recommendation

| GPU class | recommended mode | why |
|---|---|---|
| Data-centre (B300, B200, H100, H200) | `fast` (bf16) | limited by kernel launches at batch 1: FP8 gains ≤ 10% on H100 and is slower on B300, bf16 keeps decisions unchanged |
| Workstation / consumer (RTX PRO 6000, RTX 5090) | `fp8` | compute-bound; FP8 gives 1.3–1.4× at 97% agreement |
| RTX 4090 | `fast` (bf16) | FP8 compilation stalled on our test machine; bf16 works (1.5B: 12.5 ms) |
| Desktop with unified memory (DGX Spark GB10) | `fp8` | memory-bandwidth-bound; FP8 halves latency |
| Batched high-throughput serving with vLLM | `vllm` + the `-FP8` checkpoint on Hopper / Ada, the `-NVFP4` checkpoint on Blackwell (RTX 50xx, RTX PRO 6000, B200, DGX Spark); see [QUANTIZATION](QUANTIZATION.md#basal-15-the-modelopt-checkpoints) | basal-1.0 NVFP4: 3.2× throughput on the DGX Spark, but −3 accuracy points; the basal-1.5 checkpoints' agreement and speed: FP8 agrees with the bf16 model on 0.979–0.989 of decisions; NVFP4 on 0.979 for max and 0.937 for basal-1.5 (−2.8 points); no NVFP4 for mini |
| A100 and other GPUs without FP8 (not measured) | `fast` | the bf16 path runs on any sm80+ GPU |
| RTX 3090 (Ampere, sm_86; measured by @krzysztofkaminski, [#33](https://github.com/rkinas/basal/issues/33)) | `fast` (bf16); `--batches compact` for basal-2.0-mini | basal-2.0-mini 13.4 ms per decision (one option order), 161.6 decisions/s with compact rows at 32 clients; basal-2.0 (4.5B) `fast` fits 24 GB from v2.3.1 (peak 23.1 GB); no `fp8` (torchao needs sm 8.9 or newer; 2.3.2 says so before loading) |
| Apple Silicon (M-series) | MLX port: `-MLX-8bit` (decisions as with bf16); `-MLX-fp4` for 8 GB Macs (about 1.4 accuracy points less on basal-1.5, 2.0 on max, 3.6 on mini: for mini prefer `-MLX-8bit` or GGUF); without MLX: `gguf` (a `-GGUF` file in process) or `mps` (PyTorch, evidence spans) | see [Apple Silicon](#apple-silicon) |

With basal-1.0 models (which ship exit heads), add `--mode fast-exit` to let requests choose `"early_exit": "0.99"`
(about 1.15× faster, agreement ≥ 0.99). The basal-1.5 models do not ship exit heads; serve them with `fast`.

## Measured (4.5B architecture: basal-1.5, basal-1.0-4.5B)

| GPU | mode | 2 orders (ms) | dec/s | agreement | accuracy |
|---|---|---|---|---|---|
| **B300 SXM6** | `fast` (bf16) | **8.8** | 109 | 1.000 | 0.822 |
| | `fp8` | 9.7 | 102 | 0.974 | 0.824 |
| | `nvfp4` (torchao) | 13.9 | 63 | 0.904 | 0.790 |
| | `fast-exit`, 0.99 | **7.7** | **119** | 0.998 | 0.824 |
| | HTTP `fast` (p50) | **9.7** | 109 | – | – |
| | HTTP `fp8` (p50) | 10.4 | **119** | – | – |
| **H100 80GB** | `fast` (bf16) | 12.5 | 63 | 0.993 | 0.826 |
| | `fp8` | 11.3 | 80 | 0.976 | 0.830 |
| | `fast-exit`, 0.99 | 10.5 | 65 | 0.992 | 0.820 |
| | HTTP `fast` (p50) | 14.1 | 63 | – | – |
| | HTTP `fast-exit`, 0.99 (p50) | 12.1 | 77 | – | – |
| **RTX PRO 6000 Blackwell** | `fast` (bf16) | 18.9 | 39 | 0.994 | 0.822 |
| | `fp8` | 14.9 | 58 | 0.968 | 0.840 |
| | HTTP `fast` (p50) | 22.5 | 39 | – | – |
| **RTX 5090** | `fast` (bf16) | 27.3 (28.0 on a second machine) | 24 | 0.999² | 0.824 |
| | `fp8` | 19.4 | 40 | 0.968² | 0.828 |
| | HTTP `fast` (p50) | 32.3 | 23 | – | – |
| **DGX Spark (GB10)** | `eager` fp32 | 800.4 | 0.4 | reference | 0.822 |
| | `fast` (bf16) | 92.0 | 7.0 | 0.996 | 0.826 |
| | `fp8` | **44.6** | 8.3 | 0.970 | 0.836 |
| | `nvfp4` (torchao) | 41.0 | 10.4 | 0.910 | 0.792 |
| | `vllm` + NVFP4 checkpoint | 46.1 | **26.6** | 0.896¹ | 0.792 |
| | `fast-exit`, 0.99 | 76.9 | 7.6 | 0.994 | 0.828 |

¹ agreement with bf16 (which agrees with fp32 on 99.6%). ² agreement with bf16 eager (no fp32 run on this card).

## 1.5B architecture: basal-1.5-mini, basal-1.0-1.5B

| GPU | `fast` (bf16) | dec/s | `fp8` | dec/s | agreement bf16 / fp8 | vs 4.5B bf16 |
|---|---|---|---|---|---|---|
| B300 SXM6 | **4.7 ms** | **250** | 5.2 ms | 224 | 0.994 / 0.966 | 1.9× faster |
| H100 80GB | 6.2 ms | 157 | 6.3 ms | 177 | 0.992 / 0.964 | 2.0× |
| RTX PRO 6000 Blackwell | 8.8 ms | 101 | 7.6 ms | 128 | 0.998 / 0.968 | 2.2× |
| RTX 5090 | 12.7 ms | 67 | 9.3 ms | 99 | 0.996 / 0.960 | 2.2× |
| RTX 4090 | 12.5 ms | 51 | – | – | reference | – |
| DGX Spark (GB10) | 34.3 ms | 20 | **18.1 ms** | 31 | 0.996 / 0.962 | 2.7× |

Speed-up against the 4.5B model measured with the released engine on the same machine (RTX PRO 6000: 19.1 ms,
RTX 5090: 28.0 ms; the earlier runs above: 18.9 and 27.3 ms). Agreement with fp32 (RTX 4090: bf16 is the reference).

HTTP on H100 (`basal-serve --mode fast`): 7.7 ms p50, 147 decisions/s with 32 clients. NVFP4 on the DGX Spark: 16.5 ms
but agreement 0.87 — not recommended. RTX 4090: the FP8 compilation stalled on our test machine; bf16 works.

## 11B: basal-1.5-max

Measured on one H100; other GPUs were not measured for max.

| engine | measurement | result |
|---|---|---|
| basal engine, `fast` (bf16) | HTTP load test, one question per request, 32 concurrent clients | 27.3 ms p50 / 46.6 ms p95 for one question, 33.4 decisions/s; bf16 with SOAM agrees with the fp32 reference on 0.993 of 1,500 decisions |
| vLLM | – | not measured for max (verified on basal-1.5: agreement 0.994 with the fp32 engine) |
| SGLang (sglang 0.5.21) | Werdykt, 5,000 decisions, against the basal engine's `fast` mode | the same answer on 99.4% of decisions (98.4% for states over 8k tokens); not measured offline for max |

In-process engine, one H100, p50 per request: 30.7 ms for one question; 68 ms for 5 questions and 148 ms for 12 with
SOAM, against 201 and 422 ms as separate prompts. SOAM answers equal separate prompts in fp32 (1.000, largest probability
difference 1.5e-5). The HTTP and engine numbers are not comparable with each other. The 11B model needs about 22 GB
for the bf16 weights alone, plus activations and CUDA graphs: on 24 GB cards use the 4.5B model (`--mode fp8` may fit,
but it was not measured for max). On Apple Silicon, see the ports below.

## Long states under load (one H100)

Requests with one choice question over states of about 1,500 tokens (generated PL / EN case logs, `basal-loadtest
--synthetic 1500`; prompts of 1,640 tokens on average with the basal-1.5 prompt format), HTTP, latency p50 of single
requests and decisions/s with 16 concurrent clients. "earlier engine" is the basal-1.5 release engine on the same
machine; `o1` / `o2` = one option order or both (`--orders`).

| model | engine | weights | orders | p50 (ms) | decisions/s (16 clients) |
|---|---|---|---|---|---|
| basal-1.5-mini | earlier engine `fast` | bf16 | o2 (its default) | 23.0 | 48.9 |
| | this engine `fast` | bf16 | o1 | 16.7 | 76.9 |
| | this engine `fast` | bf16 | o2 | 26.6 | 57.5 |
| | this engine `fast` | `-FP8` checkpoint | o1 | 14.9 | 107.9 |
| basal-1.5 (4.5B) | earlier engine `fast` | bf16 | o2 (its default) | 51.7 | 18.1 |
| | earlier engine `fp8` (torchao) | FP8 | o1 | 43.9 | 21.9 |
| | this engine `fast` | bf16 | o1 | 38.8 | 26.1 |
| | this engine `fast` | bf16 | o2 | 56.3 | 21.3 |
| | this engine `fast` | `-FP8` checkpoint | o1 | 28.7 | 39.6 |
| | this engine `fast` | `-FP8` checkpoint | o2 | 50.1 | 26.5 |
| | vLLM 0.31 (`--mode vllm`) | bf16 | o2 | 41.8 | 32.8 |
| | vLLM 0.31 | `-FP8` checkpoint | o2 | 37.9 | 42.0 |
| | SGLang 0.5.21 (`--mode sglang`) | bf16 | o2 | 62.4 | 18.3 |

- **A long tail** (the same, with 10% of the states 3,000–10,000 tokens long; 2,270 tokens on average): basal-1.5 o2
  52.4 ms / 9.4 decisions/s with the earlier engine, 57.5 ms / 15.9 with this one (o1: 40.0 ms / 18.8; `-FP8` o1:
  29.9 ms / 23.6). The earlier engine computed a long state once per option order.
- **Five questions per request** over one state (SOAM), basal-1.5: 70 questions/s with the earlier engine (o2), 86
  (o2) and 98 (o1) with this one.
- **Two option orders at one client** cost a few ms more than in the earlier engine (mini 26.6 against 23.0 ms; `-FP8`
  basal-1.5 50.1 ms against 28.7 for one order): the branches run in a second, short forward pass (pass 2 in
  [SOAM.md](SOAM.md#which-kernel-runs)). Under load the two passes are faster (57.5 against 48.9 decisions/s).
- **The `-FP8` checkpoints** run natively in the `fast` mode (the ModelOpt FP8 weights with their static scales, the
  same arithmetic as vLLM): 1.5× the bf16 throughput. Their decisions agree with the bf16 model's on 0.98 (basal-1.5)
  and 0.97 (mini) of development decisions. torchao's on-the-fly FP8 (`--mode fp8` on bf16 weights) gains only 10–20%.
  As in bf16, FP8 answers are not bit-identical under load: batching with other requests changes the tensor shapes and
  the rounding, and a near-tie can flip. Use `--deterministic` when the same request must always get the same answer.
- `--deterministic` costs 4% with one option order (mini: 73.1 instead of 76.2 decisions/s) and 21% with two (45.2
  instead of 57.5); its answers were bit-identical alone and under load (100 of 100 requests).
- FP8 GEMMs use cuBLAS's fast accumulation, whose rounding depends on the kernel picked for the batch shape: a 4.5B
  format-2 FP8 checkpoint scored 0.6969 on a 10.6k-decision dev set batched, 0.6978 one request at a time and 0.6915
  with deterministic kernels, and 541 decisions differed between batched and deterministic. `--deterministic` therefore
  runs FP8 GEMMs with full accumulation: 0.6958, the same decisions batched and alone (max |dp| 0.1), 39.6 instead of
  41.2 decisions/s at 16 clients (H100).

## basal-omni-pico (Qwen3.5, Gated DeltaNet) on one H100

basal-omni-pico (0.8B: 18 Gated DeltaNet layers, 6 attention layers) through `basal-serve --no-calibration`, HTTP,
one connection per request (the profiles of the pico's model card): states of about 1,500 tokens ("short") and of
3,000-10,000 tokens ("long"), one question or five per request; p50 latency of single requests and throughput at 16
clients (decisions/s; questions/s with five per request). fla 0.5.2 for the Gated DeltaNet layers.

| path | short, 1 q: p50 / 16 clients | short, 5 q: p50 / 16 clients | long, 1 q: p50 / 16 clients | long, 5 q: p50 / 16 clients |
|---|---|---|---|---|
| `fast` (2.4) | 13.5 ms / 134.6 dec/s | 26.6 ms / 400.4 q/s | 45.9 ms / 34.4 dec/s | 90.6 ms / 94.4 q/s |
| `fast-nocompile` (2.4) | 17.4 ms / 89.3 | 33.5 ms / 328.2 | 64.0 ms / 22.9 | 110.5 ms / 80.1 |
| `eager` | 47.0 ms / 31.7 | 111.4 ms / 41.1 | 187.7 ms / 6.4 | 925.7 ms / 6.4 |
| `vllm` (vLLM 0.31.0; the model card's run) | 54.3 ms / 93.9 | 67 ms / 188.7 | 39 ms / 43.7 | 109 ms / 90.6 |

- `fast` against `eager`: 3.5x lower latency and 4.2x the throughput on one question over a short state, 4.2x /
  9.7x with five questions, 10.2x / 14.8x with five questions over a long state (the state is computed once, not
  once per question). Against vLLM: faster on short states and on five questions over a long state; slower on one
  question over a long state (46 against 39 ms, 34 against 44 decisions/s).
- Start-up: 109 s for `fast` (404 CUDA graphs, compiled), 76 s for `fast-nocompile`; at most 9.2 GB of GPU memory
  in use under these loads (nvidia-smi; `eager` 14.6 GB).
- Where the time goes (in process): one question over a 1.5k-token state takes 10.1 ms, 7.6 ms of it GPU kernels
  (GEMMs ~3 ms, fla's Gated DeltaNet and convolution kernels ~2.3 ms); a five-question request about 27 ms, of which
  8.5 ms is the tokenizer (with `TOKENIZERS_PARALLELISM=false`, as in these runs).

## Kernels of the CUDA graph engine (one H100)

The engine options of the kernel study, measured served over HTTP. The model is a 4.5B development checkpoint of
basal-2.0 (prompt format 2, one option order unless noted), synthetic states of about 1.6k tokens, single runs:

| configuration | bf16, 1 client | bf16, 16 clients | FP8, 1 client | FP8, 16 clients | start-up (bf16 / FP8) |
|---|---|---|---|---|---|
| FlashAttention 2 (torch SDPA), separate projections | 38.0 ms | 27.2 dec/s | 31.6 ms | 34.7 dec/s | 253 s / 536 s |
| `--attn fa3` | 35.0 ms | 28.4 dec/s | 28.7 ms | 36.2 dec/s | – |
| `--attn fa3 --merge on` (the bf16 default on Hopper with FA3 installed) | 34.6 ms | 29.6 dec/s | 27.8 ms | 37.7 dec/s | 253 s / 536 s |
| + `--lean on` | 35.0 ms | 27.4 dec/s | 29.2 ms | 33.7 dec/s | **106 s / 158 s** |

- **FA3 and merged projections** change no bf16 decision with a clear answer: on 10,606 development decisions, all
  decisions whose top two options differ by at least 0.2 are the same as with FA2. Accuracy moves by −0.02 / +0.01
  points. Two option orders agree with each prompt run alone on 0.995 of decisions (0.985 with FA2).
- **FP8 checkpoints** stay unmerged by default. Re-quantising a merged weight per tensor moves FP8 decisions as much as
  the quantisation itself.
- **The FP8 export** of the prompt-format-2 models keeps the first two and the last two decoder layers in bf16
  ([QUANTIZATION.md](QUANTIZATION.md#prompt-format-2-models-the-fp8-export)). On FA3, unmerged: 40.95 decisions/s at
  16 clients and 26.2 ms at one client (median of three servers, one H100 with no other job), 1.4× the bf16 model in
  the same comparison. Keeping every `down_proj` in bf16 too agrees slightly more often (0.957 against 0.950 on 2–10
  options) at 35.75 decisions/s.
- **The lean layer** (`--lean on`, off by default) is one layer function compiled once for every shape instead of the
  whole model per shape. Start-up is 2.4–3.4× faster, and two option orders agree with single prompts on 1.000 of
  decisions. Under load it is 7–8% slower than the compiled transformers forward, so it is an option for fast
  restarts, not the default. `--lean pass2` uses it for the branch pass only.
- **`--tree flex`** (with `--lean pass2`, opt-in): the branches of a request attend through a FlexAttention mask
  instead of a dense mask over repeated KV heads, only inside the graphs captured at start-up (other shapes run the
  dense mask, so no request compiles). Not bit-identical to the dense mask (near-tie decisions change); faster on
  multi-question loads alone (five questions per state: +3% bf16 / +22% FP8 on a development checkpoint), not on the
  basal-2.0 release models' mixed load, so it stays opt-in in v2.0.0.
- Two option orders: FA3 + merged 51.9 ms at one client and 24.3 decisions/s at 16 clients. Five questions per state:
  106.8 questions/s at 16 clients.

## Apple Silicon

Three routes: the **MLX** ports (`-MLX-8bit`, `-MLX-fp4`) with `--mode mlx` (the default on a Mac with
`basal[mlx]`); the **GGUF** ports (Ollama, `llama-server`, or `--mode gguf` in process); and **PyTorch** on the GPU of
the Mac: `--mode mps` (bf16, the packed requests of `fast` without CUDA graphs) or the plain reference
`--mode eager --device mps`. The
reference is the bf16 model in PyTorch; for basal-1.5-max, which does not fit a 24 GB Mac in bf16, it is the GGUF
`Q8_0` port with llama.cpp (on basal-1.5 (4.5B), `Q8_0` agrees with bf16 on 0.994).

| machine | model | route | latency per decision | agreement with the reference | memory |
|---|---|---|---|---|---|
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | PyTorch MPS, bf16 | 817 ms | reference | 9.5 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | MLX `-MLX-8bit` | 587 ms | 0.992 | 5.2 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | MLX `-MLX-fp4` | 581 ms | 0.970 | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | llama.cpp GGUF Q8_0 | 427 ms | 0.994 | 5.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | Ollama GGUF Q8_0 | 518 ms | as llama.cpp | 5.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | llama.cpp GGUF Q4_K_M | 446 ms | 0.980 | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | Ollama GGUF Q4_K_M | 542 ms | as llama.cpp | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | llama.cpp GGUF Q8_0 | 1.16 s | reference | 11.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | Ollama GGUF Q8_0 | 1.10 s | as llama.cpp | 11.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | MLX `-MLX-8bit` | 1.36 s | 0.996 (vs GGUF Q8_0) | 12.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | MLX `-MLX-fp4` | 1.32 s | 0.972 (vs GGUF Q8_0) | 6.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | llama.cpp GGUF Q4_K_M | 1.22 s | 0.994 (vs GGUF Q8_0) | 6.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | Ollama GGUF Q4_K_M | 1.16 s | as llama.cpp | 6.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | PyTorch MPS, bf16 | 279 ms | reference | 3.2 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | MLX `-MLX-8bit` | 196 ms | 0.998 | 1.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | MLX `-MLX-fp4` | 197 ms | 0.908 | 1.0 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | llama.cpp GGUF Q8_0 | 160 ms | 0.998 | 1.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | Ollama GGUF Q8_0 | 186 ms | as llama.cpp | 1.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | llama.cpp GGUF Q4_K_M | 157 ms | 0.964 | 1.0 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | Ollama GGUF Q4_K_M | 196 ms | as llama.cpp | 1.0 GB |

Agreement here is on a 500-decision sample, to check that each port matches its reference; to compare model sizes, see
the basal-1.5 benchmark table ([v1.5.0 README](https://github.com/rkinas/basal/tree/v1.5.0#other-results)). For basal-1.5-mini use `-MLX-8bit` (1.8 GB, agreement 0.998) or GGUF `Q4_K_M` (1.0 GB, 0.964); its `-MLX-fp4` port costs about 3.6 accuracy points (agreement 0.908, accuracy 0.882 vs 0.918).

Serve with `basal-serve --model Remek/<model>-MLX-8bit --mode mlx` ([QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)). On Apple Silicon a decision is compute-bound (prefill): every MLX format takes about the same time, so the formats differ in memory, not speed; llama.cpp is about 25% faster than MLX on the M4 Pro and Ollama about 10% (basal-1.5). Five questions over one state take 2.1–2.2 s with MLX and 1.7–1.8 s with llama.cpp / Ollama, not five times one question, because the state is computed once.

For Ollama (GGUF), see [QUANTIZATION.md](QUANTIZATION.md#gguf-for-ollama-and-llamacpp).

**`mps`, `mlx-q8` and `gguf`** (Apple M4 Pro, 24 GB, basal-1.5-mini, `basal-bench` on the 44 bundled examples, which
are shorter than the 500-decision sample above; reference: bf16 `eager` on MPS; GB: MLX allocator peak, MPS memory held
by the Metal driver):

| mode | weights | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| `eager` (`--device mps`) | bf16 | 204.8 | 116.5 | 4.6 | reference | 0.705 | 5.1 |
| `mps` | bf16 | 141.2 | 116.5 | 7.0 | 1.000 | 0.705 | 4.1 |
| `mlx` | bf16 | 147.1 | 93.7 | 7.0 | 1.000 | 0.705 | 3.2 |
| `mlx-q8` | 8-bit at load time | 158.3 | 115.5 | 7.0 | 1.000 | 0.705 | 2.1 |
| `gguf` | `-GGUF` Q8_0 | 133.1 | 103.9 | 8.0 | 1.000 | 0.705 | – |

Whole requests (the five bundled `complex.jsonl` requests and one request with all their 14 questions over one state)
give the same top answers in `mps`, `mlx` and `gguf` as the reference without SOAM (28 of 28; probabilities within
0.03). `mps` keeps evidence spans (it runs the PyTorch model); `mlx` and `gguf` do not.

### Engine comparison on Apple Silicon (basal-1.0, Apple M4 Max)

Measured by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)) with the **basal-1.0** weights; the `mlx` and `mlx-q8` rows use the MLX backend of that contribution (basal 1.5 ships its own `mlx` backend for the MLX ports and keeps `mlx-q8`; its numbers are above). Apple M4 Max (40-core GPU, 128 GB), macOS 27, torch 2.14 (MPS), MLX 0.32.3, mlx-lm 0.31.3. Measured with `basal-bench`
on the **44 bundled examples** (shorter prompts than the private 500-item sample above, so latencies are not directly
comparable with the CUDA tables; between two runs on this machine latencies differed by up to 30%). Agreement with `eager-fp32`
(PyTorch MPS, fp32) on the same machine; GB: peak memory (MLX: allocator peak, MPS: memory held by the Metal driver).

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| basal-1.0-4.5B | `eager-fp32` | 542.6 | 326.5 | 1.2 | reference | 0.795 | 23.0 |
| | `mps` (bf16) | 284.8 | 242.8 | 3.1 | 0.977 | 0.773 | 10.2 |
| | `mlx` (bf16) | **265.9** | **216.4** | **3.3** | 1.000 | 0.795 | 10.1 |
| | `mlx-q8` | 298.4 | 243.9 | 2.9 | 1.000 | 0.795 | 6.0 |
| | HTTP `mlx` (p50 / p95) | 207.9 / 255.8 | – | 3.6 | – | – | – |
| basal-1.0-1.5B | `eager-fp32` | 175.7 | 101.8 | 4.5 | reference | 0.727 | 10.2 |
| | `mps` (bf16) | 98.2 | 79.5 | 9.7 | 0.977 | 0.750 | 4.2 |
| | `mlx` (bf16) | **87.3** | **70.9** | **12.0** | 0.977 | 0.750 | 4.0 |
| | `mlx-q8` | 94.8 | 76.3 | 8.6 | 0.977 | 0.750 | 2.6 |

HTTP: `basal-loadtest` against `basal-serve --mode mlx` (default prompts of the tool, 32 concurrent clients).

- **Compute-bound.** About 90% of the forward time is in the linear layers, which MLX runs at 10–12 TFLOPS in bf16 on
  the M4 Max; on the M4 Max bf16, fp16 and fp32 differ little, larger batches do not raise throughput, and
  `mx.compile` gave no speed-up (so the `mlx` backend pads only to the longest prompt of a batch instead of to fixed
  shape buckets). The M5 chips below differ: bf16 is 3.4–5.5× faster than fp32 there, and latency does not scale with
  the GPU core count across generations (the 20-core M5 Pro is 2.0–2.5× faster in bf16 than the 40-core M4 Max). Within
  one generation it does: the M5 Max is 1.35–1.63× faster than the M5 Pro with twice the cores.
- **8-bit weights** (`mlx-q8`, MLX affine, group 64, decoder layers only) cut the resident weights of the 4.5B from
  8.9 to 4.8 GB (the checkpoint is loaded lazily, so the bf16 copy is never held); 0–10% slower.
- **4-bit weights** were tried and dropped: agreement fell to 0.86 on the 1.5B without any speed gain.
- **Throttling.** Under sustained load the MacBook's GPU slows down by up to 2× (automatic power mode); the table above
  was measured back to back, the comparison below after a cool-down before every engine.
- **Early exit** (`fast-exit`) and the `fp8` / `nvfp4` modes are CUDA-only.

### Apple M5 Pro (basal-1.0)

Measured by Kamil ([@kamilrosa](https://github.com/kamilrosa), [#7](https://github.com/rkinas/basal/issues/7)) with
`basal-bench` on the **44 bundled examples**, like the M4 Max table above. MacBook Pro, Apple M5 Pro (Mac17,9, 20-core
GPU, 64 GB), macOS 26.5.2, Python 3.12.15, torch 2.14.1 (MPS), MLX 0.32.3, mlx-lm 0.32.0; basal at commit `3fa2eea`,
the merge of #4 (the same `mlx` backend as the M4 Max table); on mains power, automatic power mode, modes measured
back to back. Agreement with `eager-fp32` (PyTorch MPS, fp32) on the same machine; GB: peak memory (MLX: allocator
peak, MPS: memory held by the Metal driver).

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| basal-1.0-4.5B | `eager-fp32` | 589.2 | 309.3 | 1.5 | reference | 0.795 | 22.9 |
| | `mps` (bf16) | 123.0 | 103.3 | 7.6 | 1.000 | 0.795 | 10.1 |
| | `mlx` (bf16) | **106.9** | **93.3** | **8.9** | 1.000 | 0.795 | 10.1 |
| | `mlx-q8` | 127.6 | 102.3 | 7.7 | 0.977 | 0.773 | 6.0 |
| basal-1.0-1.5B | `eager-fp32` | 204.2 | 109.1 | 4.5 | reference | 0.727 | 10.1 |
| | `mps` (bf16) | 49.1 | 39.3 | 21.4 | 0.977 | 0.750 | 4.1 |
| | `mlx` (bf16) | **37.6** | **33.8** | **26.3** | 0.977 | 0.750 | 4.0 |
| | `mlx-q8` | 44.2 | 37.6 | 22.5 | 1.000 | 0.727 | 2.6 |

Two more runs of `--modes mps mlx` for the 4.5B: `mlx` 108.8 / 109.4 ms, `mps` 122.7 / 124.4 ms (2 orders; spread
below 2%).

- **Half the GPU cores of the M4 Max, 2.0–2.5× faster in bf16** (2 orders: `mlx` 4.5B 106.9 vs 265.9 ms, 1.5B 37.6
  vs 87.3 ms; `mps` and `mlx-q8` alike). `eager-fp32` is slightly slower than on the M4 Max (0.86–0.92×), so bf16 is
  4.2–5.5× faster than fp32 here. A likely cause is the matrix hardware of the M5's GPU cores, which speeds up bf16
  but not fp32 (not verified).
- `mlx-q8` changes one decision of 44 on the 4.5B (agreement 0.977; 1.000 on the M4 Max).

### Apple M5 Pro (basal-2.0 and `--mlx-packed`)

Measured by Kamil ([@kamilrosa](https://github.com/kamilrosa), [#36](https://github.com/rkinas/basal/issues/36)) on
the same MacBook Pro as above (Apple M5 Pro, 20-core GPU, 64 GB), macOS 26.5.2, torch 2.14.1 (MPS), MLX 0.32.3, basal
v2.3.1, mains power. `basal-bench` on the 44 bundled examples, both option orders, two repetitions with a cool-down
before each (two values where they differ by more than 1%); agreement with `mps` (bf16) of the same run. `mlx` = bf16
weights, `mlx-q8` = 8-bit weights made at load time; `--mlx-packed` is the opt-in packed forward of 2.3.1 (see the
M5 Max section below).

| model | mode | 2 orders (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|
| basal-2.0-mini | `mps` (bf16) | 38.2 | 30.1 | reference | 0.750 | 4.1 |
| | `mlx` | 46.7 | 21.9 | 1.000 | 0.750 | 3.1 |
| | `mlx` `--mlx-packed` | **31.3** | **43.0** | 1.000 | 0.750 | 3.8 |
| | `mlx-q8` | 73.4 | 14.0 | 0.977 | 0.727 | 2.1 |
| | `mlx-q8` `--mlx-packed` | 41.4 | 35.3 / 33.5 | 0.977 | 0.727 | 2.4 |
| basal-2.0 (4.5B) | `mps` (bf16) | 101.6 | 10.5 / 10.2 | reference | 0.795 | 10.1 |
| | `mlx` | 133.4 / 135.1 | 7.7 / 7.6 | 1.000 | 0.795 | 9.0 |
| | `mlx` `--mlx-packed` | **87.6 / 88.0** | **13.5 / 13.1** | 1.000 | 0.795 | 9.6 |
| | `mlx-q8` | 193.0 / 195.3 | 5.2 / 5.1 | 0.977 | 0.773 | 5.3 |
| | `mlx-q8` `--mlx-packed` | 99.7 / 101.5 | 11.2 / 11.1 | 0.955 | 0.795 | 5.6 |

The `-MLX-8bit` ports (`--modes mlx`, as in #18): basal-2.0-mini 73 -> 41 ms with `--mlx-packed`, basal-2.0-4.5B
186 -> 98 ms (one order: 49 and 95 ms, unchanged by the flag).

- With `--mlx-packed`, `mlx` is the fastest route on this chip for both models (4.5B 88 vs 102 ms for `mps`, mini 31
  vs 38 ms). Without it, `mlx` is 22-31% slower than `mps`.
- Against the M5 Max (40 cores) on the same 4.5B port: 1.39x slower by default and 1.27x slower packed, with half
  the GPU cores.
- The 4.5B `mlx-q8` agreement is 0.955 packed against 0.977 default: one more of the 44 decisions changes. Whether
  packing changes the 8-bit rounding is not checked yet.

### Apple M5 Max (basal-1.0 and basal-2.0)

Measured by Greg Brzezinka with `basal-bench` on the **44 bundled examples**, like the M4 Max and M5 Pro tables above.
Apple M5 Max (Mac17,6, 40-core GPU, 128 GB), macOS 27.0.1, Python 3.12.2, torch 2.14.1 (MPS), MLX 0.32.4, mlx-lm
0.32.2; on mains power, automatic power mode, modes measured back to back. Agreement with `eager-fp32` (PyTorch MPS, fp32) on the same machine; GB: peak memory
(MLX: allocator peak, MPS: memory held by the Metal driver).

**basal-1.0**, at commit `3fa2eea`, the merge of #4: the same `mlx` backend as the M4 Max table above, and the commit of
the M5 Pro measurement in #7, so the three chips compare one to one:

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| basal-1.0-4.5B | `eager-fp32` | 312.8 | 185.9 | 2.8 | reference | 0.795 | 23.0 |
| | `mps` (bf16) | 81.1 | 68.1 | 13.8 | 1.000 | 0.795 | 10.1 |
| | `mlx` (bf16) | **65.7** | **62.3** | **15.6** | 1.000 | 0.795 | 10.1 |
| | `mlx-q8` | 85.4 | 77.9 | 13.5 | 0.977 | 0.773 | 6.0 |
| basal-1.0-1.5B | `eager-fp32` | 119.8 | 72.2 | 7.5 | reference | 0.727 | 10.2 |
| | `mps` (bf16) | 34.8 | 31.5 | 37.3 | 0.977 | 0.750 | 4.1 |
| | `mlx` (bf16) | **26.0** | **25.7** | **43.2** | 0.977 | 0.750 | 4.0 |
| | `mlx-q8` | 32.7 | 30.6 | 38.2 | 1.000 | 0.727 | 2.6 |

Two more runs of `--modes mps mlx` for the 4.5B: `mlx` 71.6 / 73.3 ms, `mps` 86.7 / 88.3 ms (2 orders).

**basal-2.0**, at v2.2.0 (`e49665c`); `mps` in bf16, `mlx` with the `-MLX-8bit` ports. `basal-bench` takes one
`--model`, so these rows come from a single run of `basal.bench.main()` (unchanged) in which only the `mlx` mode loads
its weights from the `-MLX-8bit` folder and every other mode, including the `eager-fp32` reference, from the bf16
folder; agreement is therefore against the same bf16 reference as the other rows. The 20-line wrapper is in #18.

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy | GB |
|---|---|---|---|---|---|---|---|
| basal-2.0-mini | `eager-fp32` | 104.3 | 66.4 | 9.1 | reference | 0.750 | 9.1 |
| | `mps` (bf16) | **30.0** | **28.4** | **45.9** | 1.000 | 0.750 | 4.1 |
| | `mlx` (`-MLX-8bit`) | 49.0 | 30.9 | 21.0 | 0.977 | 0.727 | 2.1 |
| basal-2.0 (4.5B) | `eager-fp32` | 304.9 | 182.0 | 3.0 | reference | 0.795 | 21.1 |
| | `mps` (bf16) | **72.5** | **68.0** | **15.4** | 1.000 | 0.795 | 10.1 |
| | `mlx` (`-MLX-8bit`) | 130.0 | 74.6 | 7.6 | 0.955 | 0.795 | 5.2 |

Two more runs for basal-2.0 (4.5B): `mlx` 133.8 / 134.6 ms, `mps` 72.6 / 72.5 ms (2 orders). With two option orders,
`mlx` at v2.2.0 is slower than at `3fa2eea` on this machine (basal-1.0-4.5B: 115.6 vs 65.7 ms; one order: 68.0 vs
62.3 ms), so on v2.2.0 `mps` is the faster route for two orders on the M5 Max. Reproduced on an Apple M4 Pro
(`basal-bench --modes mlx`, 2 orders, `3fa2eea` against v2.2.0 and 2.3.1): basal-1.0-1.5B 126 vs 149 ms (two runs each;
one order 95-99 ms in both), basal-1.0-4.5B 388 vs 506 ms (one run; one order 295 vs 339 ms). The cause is the
backend: when #4 was merged into basal 1.5 (`4c11a87`), the `mlx` backend of 1.5 replaced the one of #4. It runs the
prefix that the two orders share as one forward into a KV cache and both option blocks as a second forward, where #4
ran the prefix and both blocks as one packed forward, and every forward reads all the weights again
([#27](https://github.com/rkinas/basal/issues/27)). 2.3.1 adds that packed forward back as `--mlx-packed`
(opt-in; [#28](https://github.com/rkinas/basal/pull/28) by Grzegorz Brzezinka). Two orders on the M5 Max (his
measurement): basal-1.0-4.5B 118 -> 68 ms, basal-2.0-4.5B-MLX-8bit 134 -> 77 ms. On the M4 Pro: basal-1.0-1.5B
158 -> 130-136 ms, basal-2.0-4.5B-MLX-8bit 379-390 -> 330 ms. Its checks and why it is not the default yet:
[QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon).

- **Same generation, twice the GPU cores** (M5 Max 40 vs M5 Pro 20, basal-1.0 at `3fa2eea`, 2 orders): bf16 and 8-bit
  modes are 1.35–1.63× faster, `eager-fp32` 1.71–1.88×; the 1.5B gains less than the 4.5B.
- **Same core count, newer generation** (M5 Max vs M4 Max, both 40 cores): bf16 modes are 2.8–4.0× faster,
  `eager-fp32` 1.5–1.7×. On the M5 Max bf16 is 3.4–4.8× faster than fp32.

#### Inference engines on Apple Silicon

basal needs the next-token probabilities of the option letters after a prompt that ends in `{"answer": "`, not generated
text. Every engine that can return them was run on the same M4 Max: 44 bundled examples, both option orders, each
engine on prompts it had not seen (so prefix caches only help between the two orders of a question), 60 s cool-down
before each engine. *ms*: median per decision, one request at a time; *dec/s*: remaining 21 decisions in one call
(in-process) or with 4 concurrent clients (HTTP servers); *TV*: total-variation distance of the averaged two-order
probabilities to the fp32 PyTorch reference.

| engine | 4.5B ms | 4.5B dec/s | 1.5B ms | 1.5B dec/s | agreement 4.5B / 1.5B | TV mean / max (4.5B) |
|---|---|---|---|---|---|---|
| basal `mlx` | 198 | 5.1 | **67** | 15.8 | 1.000 / 0.977 | 0.0047 / 0.024 |
| basal `mlx-q8` | 200 | 4.8 | 70 | 14.3 | 1.000 / 0.977 | 0.0085 / 0.091 |
| basal `mps` | 239 | 4.4 | 89 | 12.4 | 0.977 / 0.977 | 0.0050 / 0.035 |
| basal `gguf` F16 | 200 | 5.5 | 69 | 16.7 | 1.000 / 1.000 | **0.0006** / 0.004 |
| basal `gguf` Q8_0 | 212 | 5.1 | 72 | 15.9 | 1.000 / 0.977 | 0.0039 / 0.038 |
| basal `gguf` Q4_K_M | 222 | 4.9 | 76 | 14.3 | 0.955 / 0.955 | 0.0465 / 0.307 |
| basal `vllm` on vllm-metal 0.30 ¹ | 210 | **6.9** | 80 | **21.4** | 0.977 / 0.977 | 0.0056 / 0.051 |
| mlx-lm 0.31 Python API, KV prompt cache, orders one after the other ¹ | 261 | 3.7 | 94 | 10.7 | 1.000 / 0.977 | 0.0050 / 0.042 |
| llama-cpp-python 0.3.35 F16, orders one after the other | 261 | 3.8 | 101 | 9.8 | 1.000 / 1.000 | 0.0005 / 0.005 |
| mlx_lm.server 0.31 (`top_logprobs` 11) ¹ | 374 | 3.1 | 153 | 7.6 | 1.000 / 0.955 | 0.0053 / 0.031 |
| llama-server b11146 F16 (token ids, `n_probs` 20) | **191** | 5.1 | 79 | 13.2 | 1.000 / 1.000 | 0.0005 / 0.005 |
| MTPLX 2.12, prompt-scoring lane (`echo`, `max_tokens` 0) ¹ | 339 | 2.9 | 129 | 8.0 | 0.977 / 0.977 | 0.0059 / 0.043 |
| basal `ollama` (Ollama 0.34.4, original safetensors import) ³ | 276 | 3.3 | 104 | 9.0 | 1.000 / 0.977 | 0.0055 / 0.059 |
| LM Studio, llama.cpp engine, F16 GGUF (chat with assistant prefill) ² | 211 | 5.8 | 80 | 16.5 | 0.955 / 0.977 | 0.0607 / 0.630 |

¹ These engines build the model with mlx-lm's Llama, which does not read `rope_parameters` (transformers 5) and falls
back to `rope_theta` 10000 instead of 1e6: unpatched, the mean TV is 0.10–0.12 for mlx_lm.server and 0.10–0.13 for
vllm-metal (4.5B–1.5B, max 0.73). They were run on a copy of the checkpoint whose `config.json` also has `"rope_theta": 1000000`
(basal's own `mlx` backend reads the value itself). ² LM Studio (and Ollama with a GGUF import, and llama-server with
text prompts) tokenize with the GGUF vocabulary, which splits basal prompts differently from the training tokenizer;
see [GGUF.md](GGUF.md#other-llamacpp-front-ends-send-token-ids). ³ Measured with the contribution's variant of
`--mode ollama` for a safetensors import of the original checkpoint, imported from the patched copy of ¹; basal
1.5's `--mode ollama` serves the `-GGUF` ports, whose tokenizer fix makes Ollama's text tokenization exact
([PORTS.md](PORTS.md)).

![Latency and throughput of every engine](figures/apple_latency_throughput.png)

![Speed vs faithfulness](figures/apple_speed_vs_fidelity.png)

Per item, the same comparison shows where the deviations come from: the rope bug (bottom rows) and the GGUF text
tokenizer (LM Studio, llama-server with text) shift almost every item, Q4_K_M a few items strongly, the bf16 / 8-bit
paths stay at 1e-3–1e-2, and F16 through llama.cpp mostly below 1e-3. Items where fp32 itself is nearly tied (e.g.
`q-36` on the 1.5B) flip under any perturbation.

![Per-item distance to fp32](figures/apple_fidelity_heatmap.png)

- **Fastest single decision**: llama.cpp (llama-server with token ids, basal `gguf`) and basal `mlx`, within 5%.
  **Highest throughput**: vLLM's scheduler on vllm-metal. basal's `vllm` mode runs on it with the rope patch; it asks
  for the letter ids with `logprob_token_ids`, because vllm-metal ignores `logprobs_mode="processed_logprobs"` and
  returns the top-k of the whole vocabulary: with plain top-k the least likely option was missing (probability 0) on
  14 of the 44 items of the 1.5B.
  **Closest to fp32**: llama.cpp F16.
- Packing both option orders into one forward pass (basal `mlx` / `gguf`) is worth 25–30% against running them one
  after the other with a prompt cache (mlx-lm, llama-cpp-python rows).
- Not usable as shipped: **oMLX** 0.7.0rc1 returns no logprobs (checked with a request), **vllm-mlx** 0.5.0 has no
  logprobs in its API models; **mistral.rs** and **MLC-LLM** implement Llama without the q/k/v/o and MLP biases basal
  needs; **Swama**, **Exo** and Apple's Foundation Models framework expose no logprobs. **MTPLX** accelerates decoding
  with multi-token prediction, which a one-token decision does not use; plain Llama runs on its experimental AR path.

#### Quality

![Accuracy, log-loss and changed decisions per engine](figures/apple_quality.png)

Against the gold labels, the 44 bundled examples cannot rank the engines: every engine gets 33–36 of 44 right with the
4.5B and 31–33 with the 1.5B, and the 95% intervals are ±12 points wide. The engines with the rope bug even score
36/44 and a lower log-loss on the 4.5B (apparently flatter probabilities, which help on the confidently wrong
items), although they change the probabilities of almost every item. For a backend, quality therefore means
faithfulness to the reference model on which accuracy, calibration temperatures and confidence thresholds were
measured: the decisions it changes (right panel) and the TV distance above. The errors themselves belong to the model:
the same items are wrong in every faithful engine.

![Probability of the gold option per item](figures/apple_gold_prob_heatmap.png)

#### Quantised formats: MLX, oMLX oQ, GGUF

Conversions of basal-1.0 by the contributor: [MLX 8-bit](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-MLX-8bit) and
[oQ6e](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-oQ6e) of the 4.5B,
[MLX 8-bit](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-MLX-8bit) and
[oQ6e](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-oQ6e) of the 1.5B (with `CALIBRATION.json`, for
`--mode mlx --model <repo>`), and the GGUF files
([4.5B](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF), [1.5B](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF)).
The other formats below can be converted locally. Pre-converted MLX checkpoints load directly in `--mode mlx` (the
mlx-lm quantisation in their `config.json` is applied; copy `CALIBRATION.json` from the original repository into the
directory so the temperatures apply):

```bash
python -m mlx_lm convert --hf-path <basal dir> --mlx-path basal-1.5B-q8 -q --q-bits 8 --q-group-size 64
python -m mlx_lm convert --hf-path <basal dir> --mlx-path basal-1.5B-mxfp4 -q --q-mode mxfp4   # also nvfp4, mxfp8
python -c "from omlx.oq import quantize_oq_streaming as q; q('<basal dir>', 'basal-1.5B-oQ6e', 6, enhanced=True)"
basal-serve --mode mlx --model basal-1.5B-oQ6e
```

Other MLX runtimes (LM Studio, oMLX, mlx_lm.server) read `rope_theta` from `config.json` (see ¹ above), so convert
from a copy of the checkpoint whose `config.json` has it. Measured like the engines above (M4 Max, cold prompts,
cool-down; *TV*: mean / max total-variation distance to fp32, *changed*: decisions whose top option differs from fp32):

| format | 1.5B GB | 1.5B ms | 1.5B TV | 1.5B changed | 4.5B GB | 4.5B ms | 4.5B TV | 4.5B changed |
|---|---|---|---|---|---|---|---|---|
| bf16 (`mlx`) | 3.19 | 67 | 0.0054 / 0.027 | 1 | 9.51 | 198 | 0.0044 / 0.033 | 0 |
| GGUF F16 (`gguf`) | 3.20 | 69 | **0.0004** / 0.002 | 0 | 9.52 | 200 | **0.0005** / 0.008 | 0 |
| GGUF Q8_0 | 1.70 | 72 | 0.0053 / 0.029 | 1 | 5.06 | 212 | 0.0040 / 0.039 | 0 |
| MLX affine 8-bit (= oMLX oQ8) | 1.70 | 70 | 0.0093 / 0.041 | 0 | 5.06 | 202 | 0.0072 / 0.070 | 0 |
| MLX mxfp8 | 1.65 | 72 | 0.0168 / 0.101 | 1 | – | – | – | – |
| oMLX oQ6e | 1.34 | 73 | 0.0119 / 0.050 | 1 | 3.98 | 205 | 0.0116 / 0.118 | 0 |
| oMLX oQ6 | 1.34 | 71 | 0.0180 / 0.184 | 1 | – | – | – | – |
| MLX affine 6-bit | 1.30 | 72 | 0.0186 / 0.141 | 1 | – | – | – | – |
| GGUF Q4_K_M | 0.97 | 76 | 0.0516 / 0.444 | 2 | 2.88 | 222 | 0.0467 / 0.306 | 2 |
| oMLX oQ4e | 0.94 | 70 | 0.0639 / 0.275 | 3 | 2.80 | 207 | **0.0407** / 0.212 | 2 |
| MLX nvfp4 | 0.90 | 71 | 0.0763 / 0.445 | 2 | – | – | – | – |
| MLX mixed_4_6 | 0.97 | 70 | 0.0814 / 0.427 | 4 | – | – | – | – |
| oMLX oQ4 | 0.94 | 71 | 0.0868 / 0.522 | 4 | – | – | – | – |
| MLX affine 4-bit | 0.90 | 69 | 0.1125 / 0.686 | 5 | 2.68 | 200 | 0.0613 / 0.417 | 1 |
| oMLX oQ3e | 0.74 | 69 | 0.1565 / 0.453 | 7 | – | – | – | – |
| MLX mxfp4 | 0.85 | 71 | 0.2029 / 0.796 | 9 | 2.53 | 204 | 0.1003 / 0.529 | 4 |

![Size vs faithfulness of the quantised formats (ringed: the published conversions)](figures/apple_formats_size_vs_fidelity.png)

- **No format is faster**: the Apple GPU is compute-bound on these prompts, so every format of a model runs within
  15% of bf16. Quantisation only saves memory.
- **8-bit**: GGUF Q8_0 stays at the bf16 level; the MLX formats deviate about twice as much (they also quantise the
  embeddings and the LM head, which basal's runtime `mlx-q8` keeps in bf16). oQ8 gave exactly the probabilities of
  MLX affine 8-bit (presumably its protection rules have nothing to protect in this dense Llama).
- **6-bit**: the imatrix calibration of oQ6e cuts the worst-case deviation of oQ6 / affine 6-bit to about a third
  (1.5B).
- **4-bit** changes 1–9 of 44 decisions in every format and single probabilities by 0.2–0.8: the calibration of the
  original no longer holds. oQ4e and GGUF Q4_K_M are the best 4-bit formats (oQ4e ahead on the 4.5B, Q4_K_M on the
  1.5B); mxfp4 is the worst. 3-bit (oQ3e) is worse still.
- **LM Studio's MLX runner** gives the same faithfulness as the format it runs (1.5B: bf16 0.0051, 8-bit 0.0107,
  oQ4e 0.066 mean TV) at about twice the latency (122–138 ms): its API returns logprobs only on chat completions
  (top 10, and only with `max_tokens` ≥ 2), so both option orders are separate requests. Its top-logprob list can
  contain two tokens that print as the same letter (`B` and `▁B`): match the letter ids on the returned `bytes`.
  **oMLX** runs all of these formats but returns no logprobs.
- Not tried: DWQ, AWQ and GPTQ (mlx-lm), which need a calibration dataset and a training run.

![Per-item distance to fp32 by format](figures/apple_formats_heatmap.png)

Figures: `python docs/figures/make_figures.py` (matplotlib) from the measurements in `docs/figures/apple_engines.json`.

### Apple M4 (state cache on `--mode mlx`)

`--state-cache-gb N` and `--prefix-cache on` work on `--mode mlx` the same way as on the CUDA graph engine. The keys
and values of a request's state (or of the template prefix) are kept across requests. A state that begins with a cached
one computes only its new tokens.

Measured on a MacBook Air (Apple M4, 8-core GPU, 16 GB, fanless), macOS 26.5.2, MLX 0.32.4, mlx-lm 0.32.2, mains power.
Setup: the `-MLX-8bit` ports, two option orders, served over HTTP, one client, requests in dataset order after a
warm-up. Every workload ran in ABBA order (off, cache, cache, off), because the fanless M4 slows down within a long run.
Four public datasets:

- **BAN-PL** (NASK, CC BY 4.0, `BAN-PL_1`): a moderation policy of 2,083 tokens, then one short Wykop.pl comment
  (median 113 characters), and one `choice`: remove or keep. Every state begins with the same policy, so each comment
  continues the cached entry of the previous one. 100 comments, half of them harmful.
- **MultiRC** (SuperGLUE, validation): one request per question about a paragraph, with its candidate answers as
  `noul` questions over the paragraph, so the state repeats across requests. 4.5B: 10 paragraphs, 100 requests, 451
  decisions. mini: 20 paragraphs, 222 requests, 1,153 decisions.
- **MultiWOZ 2.2** (dev): one request per user turn. The state is the dialogue so far, so each state begins with the
  previous one (an agent trace or chat history). One `choice` of the service the last message concerns. 4.5B: 10
  dialogues, 79 requests. mini: 15 dialogues, 120 requests.
- **MASSIVE pl-PL** (test): short Polish voice-assistant commands, one `choice` of the scenario (18 options). Only the
  template prefix repeats. 4.5B: 100 requests. mini: 200.

p50 of the client's wall time per request (the median over both runs of each configuration):

| model | workload | off | cache | speed-up | top answers same | at margin >= 0.2 | \|dp\| median / p99 / max |
|---|---|---|---|---|---|---|---|
| basal-2.0 (4.5B) | BAN-PL `--state-cache-gb 1` | 13,724 ms | **907 ms** | **15.1x** | 100 / 100 | 1.000 | 0.002 / 0.029 / 0.034 |
| | MultiRC `--state-cache-gb 1` | 4,217 ms | **1,601 ms** | **2.6x** | 451 / 451 | 1.000 | 0.002 / 0.019 / 0.115 |
| | MultiWOZ `--state-cache-gb 1` | 2,697 ms | **1,748 ms** | **1.5x** | 79 / 79 | 1.000 | 0.000 / 0.012 / 0.023 |
| | MASSIVE pl `--prefix-cache on` | 980 ms | 873 ms | 1.12x | 100 / 100 | 1.000 | 0.000 / 0.015 / 0.047 |
| basal-2.0-mini | BAN-PL `--state-cache-gb 1` | 3,995 ms | **292 ms** | **13.7x** | 100 / 100 | 1.000 | 0.002 / 0.019 / 0.022 |
| | MultiRC `--state-cache-gb 1` | 1,376 ms | **509 ms** | **2.7x** | 1,147 / 1,153 | 1.000 | 0.002 / 0.018 / 0.037 |
| | MultiWOZ `--state-cache-gb 1` | 831 ms | **571 ms** | **1.5x** | 119 / 120 | 1.000 | 0.000 / 0.013 / 0.022 |
| | MASSIVE pl `--prefix-cache on` | 316 ms | 278 ms | 1.13x | 196 / 200 | 1.000 | 0.000 / 0.016 / 0.058 |

The speed-up follows the share of a request's work that comes from the cache. In BAN-PL the policy is almost the whole
prompt, so only the comment and the question are computed. In MultiRC the 4-7 questions of a request, in two orders,
are about half the work of the request even without the state: five of these requests with the state replaced by "."
took 44-60% of their full time on 4.5B. The cache cannot remove that half.

MultiWOZ by the length of the dialogue so far (state tokens). A state shares a cached prefix only from one full block of
the prefix index (256 tokens) on. Shorter dialogues cannot hit, and they pay the cache's extra work (the state stored, a
second forward over it): 0.96x. Above that, a cached step costs about the same whatever the length of the dialogue,
while without the cache it grows with it:

| model | state tokens | turns | off | cache | speed-up |
|---|---|---|---|---|---|
| basal-2.0 (4.5B) | < 256 | 36 | 1,970 ms | 2,056 ms | 0.96x |
| | 256-511 | 36 | 3,125 ms | 1,711 ms | 1.83x |
| | >= 512 | 7 | 4,368 ms | 1,605 ms | **2.7x** |
| basal-2.0-mini | < 256 | 56 | 619 ms | 641 ms | 0.96x |
| | 256-511 | 53 | 1,002 ms | 530 ms | 1.89x |
| | >= 512 | 11 | 1,382 ms | 554 ms | **2.5x** |

- Each configuration is deterministic: its two runs give bit-identical probabilities. A cached state gives the same
  probabilities as its first, uncached request with the cache on, in any order of the requests.
- Against the server without the cache, the probabilities are not bit-identical. The shared prefix is computed in two
  forwards (the state, then the rest over it) instead of one, and bf16 / 8-bit kernels round differently for other row
  counts. In fp32 the two paths agree:
  - the tiny model of `tests/test_mlx_state_cache.py`: max |dp| 6e-8;
  - the worst decision above (4.5B MultiRC, 0.288 -> 0.403, with the same top answer), recomputed on the server's own
    prompts with the decoder cast to fp32: |dp| 0.0000, against 0.178 in bf16.
- The top answers that changed were near ties (margin below 0.08 without the cache).
- **Short continuations.** A state that continues a cached entry by a few tokens never runs 2-8 tokens:
  `MIN_CONTINUE`, the forward starts earlier. MLX's fused attention (`mx.fast.scaled_dot_product_attention`, MLX
  0.32.4) loses bf16 precision when 2-8 queries times the query heads per KV head exceed 32. Against an fp32
  reference the max error is 4-6e-2 instead of 5-8e-3, for 6-8 queries on basal-2.0-mini (6 query heads per KV head)
  and 5-8 on 4.5B (8 per KV head). Before this guard, one BAN-PL decision on mini went from 0.46 to 0.16 when a comment
  continued the cached policy by 8 tokens, and BAN-PL's max |dp| was 0.30 (mini) and 0.14 (4.5B).
- `--prefix-cache` helps little on MLX, because the template prefix of prompt format 2 is 34-45 tokens. On the CUDA
  graph engine it also keeps short requests on pass-2 graphs. The state cache is the one worth turning on.
- Memory: an entry is a copy of the state's keys and values (`mx.contiguous`, not a slice of mlx-lm's 256-token cache
  buffers), counted against the budget. One 2.1k-token BAN-PL state is about 126 MB on 4.5B (60 layers, 2 KV heads of
  128). The wired limit is raised by the budget.


## AMD Strix Halo (experimental, community-maintained)

`--mode strix` ([basal/strix.py](../basal/strix.py)) was contributed and is maintained by
[@kardysm](https://github.com/kardysm) ([#14](https://github.com/rkinas/basal/pull/14)), not by the core team. It runs
the Llama-architecture models basal-1.5 and basal-2.0 only, and was measured with the 4.5B ones. Shapes its kernels
cannot run are refused at start-up with a message naming what does not fit: the mini models (hidden size 1,536, 6 query
heads per kv head) and basal-omni (Gemma 4) are not supported; the 11B max models pass the checks but have not been
measured. ROCm only, bf16, tile configs tuned on the Radeon 8060S. **Every number in this section was measured by
@kardysm** on the machine below, except the last bullet (a second machine, measured by @piotrkowski); we have not
reproduced them. The benchmark scripts and raw results are in the author's fork,
[kardysm/basal#3](https://github.com/kardysm/basal/pull/3).

One Ryzen AI Max+ PRO 395 (Radeon 8060S, gfx1151, 128 GB unified memory; Fedora 44, PyTorch 2.12 + ROCm 7.2),
basal-1.5 (4.5B). On this iGPU a decision is compute-bound prefill (~9.2 GFLOP per token), and the stock kernels reach
only about half of what the GPU sustains. `--mode strix` is a forward written for it (Triton kernels: flash attention
over the packed prefix trie, gate / up + SwiGLU fused, matmuls with the bias and the residual add in the epilogue).
Request matrix over HTTP, p50 (details, profile and method: [benchmarks/STRIX_HALO.md](benchmarks/STRIX_HALO.md)):

| request | `gguf` (Q8_0, llama.cpp Vulkan) | **`strix`** (bf16) | `fast-nocompile` as shipped |
|---|---|---|---|
| ≈100-token state, 1 question | 178 ms | **159 ms** | 309 ms |
| ≈100-token state, 20 questions | 1.79 s | **1.08 s** | 5.28 s |
| ≈2,000-token state, 1 question | 1.28 s | **736 ms** | 3.09 s |
| ≈6,000-token state, 1 question | 4.16 s | **2.73 s** | 34.0 s |
| single questions, 32 clients | 7.7/s | **10.9/s** | 4.6/s |
| start-up / GPU memory | 3 s / 5.8 GiB | 6 s / 10.0 GiB | 888 s / 32 GiB |

- `strix` agrees with the fp32 reference on 144 of 144 decisions (`gguf` 143). One short question at a time is faster
  with `gguf` (141 vs 159 ms): the bf16 weights are 9 GB per forward, the Q8_0 file 5 GB.
- Long states gain less (1.5× at 6,000 tokens): the matmuls hit the ~35–40 TFLOPS this GPU sustains, and the package
  runs at its power limit (~160 W, then ~140 W) with the GPU at 2.15–2.33 GHz.
- For `gguf` here, build llama-cpp-python with Vulkan (`CMAKE_ARGS="-DGGML_VULKAN=on"`); the host's Mesa matters.
- **basal-2.0** (4.5B, same architecture; both servers up, requests interleaved, median of 5 rounds): `gguf` → `strix`
  136 → 144 ms for one question over a ≈100-token state, 1.27 s → **714 ms** at ≈2,000 tokens, 4.16 → **2.41 s** at
  ≈6,000 tokens (5.25 → **2.90 s** with 20 questions), 11.5 → **19.0/s** with 32 clients. Both agree with fp32 on 142 of
  144 decisions; the other two are one question that fp32 puts at 0.502 / 0.498.
- **A second machine** ([@piotrkowski](https://github.com/piotrkowski)): basal-2.0 (4.5B) on a ROG Flow Z13 tablet
  (same GPU, 32 GB, ~50 W) gives the same answers and a close speed-up (1.05–1.70×), 1.4–1.7× slower in both engines
  ([details](benchmarks/STRIX_HALO.md#a-second-machine-basal-20-on-a-rog-flow-z13)).

## Google TPU (experimental, community-maintained)

`--mode tpu` (extra `basal[tpu]`, [install](INSTALL.md#google-tpu-experimental-community-maintained)) runs the bf16
weights in JAX / XLA on one TPU chip with the same SOAM packing, option codes and calibration as the basal engine.
Contributed and maintained by [@qooba](https://github.com/qooba) ([#15](https://github.com/rkinas/basal/pull/15)), not
by the core team. It runs the Llama-architecture models (basal-1.0, basal-1.5, basal-2.0) and refuses others at
start-up (basal-omni, Gemma 4, is not supported). **Every number in this section was measured by @qooba** on a Colab
TPU v5e-1 (`TPU v5 lite`, 16 GB HBM), JAX 0.11.1, with basal-2.0 and basal-2.0-mini; we have not reproduced them on a
TPU of our own. basal-2.0-max (11B, 22 GB of bf16 weights) needs more HBM than one v5e chip has.

`basal-bench` on the 44 bundled examples (the reference is `eager-fp32`, transformers on the VM's CPU; both option
orders per decision):

| model | mode | 2 orders (ms) | 1 order (ms) | dec/s | agreement | accuracy |
|---|---|---|---|---|---|---|
| basal-2.0 (4.5B) | `eager-fp32` (CPU) | 3884.7 | 2522.1 | 0.3 | reference | 0.795 |
| | `tpu` (bf16) | **16.3** | 15.9 | **64.7** | 1.000 | 0.795 |
| basal-2.0-mini | `eager-fp32` (CPU) | 1212.8 | 754.2 | 0.8 | reference | 0.750 |
| | `tpu-fp32` | 23.1 | 22.8 | 33.5 | 1.000 | 0.750 |
| | `tpu` (bf16) | **6.3** | 6.1 | **188.7** | 1.000 | 0.750 |

Identity gate (the items and paths of `basal-bench --identity`, basal/identity.py: single questions over ~1.8k-token
states, five-question requests over one state and 3.9-5.7k-token states, both option orders; the reference is every
prompt alone in transformers fp32 on the CPU; 20 single questions, 6 requests and 2 long states per model):

| model, `tpu` dtype | plain | plain-batch | orders2 | orders2-batch | soam | soam-batch | long-plain | long-tree |
|---|---|---|---|---|---|---|---|---|
| basal-2.0-mini, fp32 | 1.000 (1.7e-5) | 1.000 (1.7e-5) | 1.000 (1.7e-5) | 1.000 (1.7e-5) | 1.000 (1.4e-5) | 1.000 (1.2e-5) | 1.000 (1.3e-5) | 1.000 (1.3e-5) |
| basal-2.0-mini, bf16 | 0.975 (0.023) | 1.000 (0.018) | 0.975 (0.022) | 0.975 (0.025) | 1.000 (0.037) | 1.000 (0.033) | 1.000 (0.011) | 1.000 (0.011) |
| basal-2.0, bf16 | 1.000 (0.027) | 1.000 (0.049) | 1.000 (0.021) | 0.975 (0.049) | 1.000 (0.054) | 1.000 (0.030) | 1.000 (0.015) | 1.000 (0.017) |

Readout agreement (max |Δp|); 40 readouts per single-question path, 60 per request path, 4 per long path. In fp32
every path is exact (|Δp| ≤ 1.7e-5): the TPU packing (prefix trie, batching, blocked attention for long states) adds no
error. In bf16 each 0.975 is one readout of 40, always at a reference top-2 gap ≤ 0.027 (the near ties of this
sample); the decisions (both orders averaged) agree on 20/20 per path except mini orders2-batch, 19/20 (a decision
with a reference gap ≤ 0.021). The sample is small: one flip moves a path below the 0.999 gate.

Over HTTP (`basal-serve --mode tpu` with the 2.0 defaults: prompt format 2, one option order; `basal-loadtest`, 32
clients): basal-2.0 17.1 ms p50, 59.6 decisions/s; basal-2.0-mini 7.3 ms p50, 165.5 decisions/s. One request with 1 / 5
/ 12 questions over one state: 18.8 / 26.4 / 39.9 ms (basal-2.0) and 7.5 / 10.4 / 15.9 ms (mini). Start-up compiles
every shape: 268 s (basal-2.0) and 180 s (mini) on a v5e, cached for later starts.

Not on TPU: evidence spans, `--media`, early exit and quantisation. The 44 examples are a small sample; agreement and
speed on your own items: `basal-bench --modes eager-fp32 tpu --questions <file.jsonl>`.

## Notes per platform

- **DGX Spark (GB10, aarch64, CUDA 13).** In the README install steps use the CUDA 13 build of PyTorch
  (`uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`) instead of cu128. The GPU shares LPDDR5X memory (about 273 GB/s) with the CPU; every forward pass reads
  all weights, which is why FP8 (half the bytes) halves latency.
- **B300 / B200 (sm_100/sm_103).** PyTorch cu130 wheels work. The `vllm` mode needs a CUDA toolkit (nvcc) of version
  12.9 or newer on the machine: vLLM's FlashInfer kernels are compiled on first use, and nvcc 12.8 (found in some
  container images) cannot target the B300 (`Unsupported gpu architecture 'compute_103a'`). Compilation of `nvfp4` takes long (about 50 minutes on
  the B300) and is not recommended there.
- **RTX 5090 / RTX PRO 6000.** A driver with CUDA 12.8 is enough (`--index-url https://download.pytorch.org/whl/cu128`).
- **Start-up.** `fast` compiles and captures CUDA graphs for all input shapes before serving: 4–10 minutes the first
  time for the 4.5B model (289 s on an H100 with the weights on local disk for the 11B), much less with a warm compile cache. `fast-exit` captures one
  graph per segment and takes longer (14–22 minutes). `fast-nocompile` starts in seconds at about 1.4× the latency.
- **NVFP4 quality.** With or without calibration, 4-bit weights and activations change about 10% of decisions of this
  model and lower accuracy by about 3 points. Prefer bf16 or FP8 unless batch throughput matters more.
