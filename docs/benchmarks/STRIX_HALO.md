# `--mode strix`: basal-1.5 and basal-2.0 on AMD Strix Halo

`--mode strix` (`basal/strix.py`) is a basal forward written for the Radeon 8060S (gfx1151) after a profile of where
`--mode gguf` spends its time on it. It keeps the bf16 weights and replaces the slow kernels with Triton ones. It was
written and tuned on basal-1.5; basal-2.0 has the same architecture and runs through it unchanged. On the request matrix
below it is 1.12–1.75× faster than `--mode gguf` with basal-1.5 and 1.37–1.81× with basal-2.0 (one short question at a
time excepted), and it agrees with an fp32 reference on every decision that is not a tie. Opt-in by name, measured only
on this machine; every other mode is unchanged.

Machine: AMD Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151, 40 CUs), 128 GB unified memory; Fedora 44, kernel 7.2.8,
Mesa 26.2. Model: basal-1.5 (4.5B); `Q8_0` GGUF for llama.cpp, bf16 for PyTorch. Software: PyTorch 2.12.0 + ROCm 7.2,
Triton 3.7 (ships with PyTorch), transformers 5.17, llama-cpp-python 0.3.36 built with Vulkan
(`CMAKE_ARGS="-DGGML_VULKAN=on"`). Every engine was measured after the GPU cooled below 55 °C. The basal-2.0 runs
(basal-2.0-4.5B, engine 2.0.0) were made later on kernel 7.2.10, otherwise with the same software.

## Where `--mode gguf` spends its time

Every request is at least 98.5% GPU decode (server, tokenization and packing take 2–34 ms, at most 0.5%), and the GPU
reaches 16–19 useful TFLOPS at every request size. GPU time by operation, from llama.cpp's Vulkan performance logger
(`GGML_VK_PERF_LOGGER=1`), one decode per request:

| operation | S, 1 q (176 ms) | L, 1 q (1.27 s) | XL, 1 q (4.17 s) | XL, 20 q (6.69 s) | speed |
|---|---|---|---|---|---|
| FFN matmuls (gate / up, down) | 69% | 60% | 51% | 47% | 21–25 TFLOPS |
| flash attention | 2.5% | 10% | 23% | 28.5% | ~11 TFLOPS on long rows |
| element-wise: bias and residual adds, SwiGLU, norms, rotary | 16% | 18% | 15.5% | 14.5% | memory-bound, fp32 |
| attention projections | 12.5% | 12.4% | 10.4% | 9.7% | q / o 18, k / v 8–9 TFLOPS |

- The model has biases on every linear layer (`attention_bias`, `mlp_bias`); llama.cpp adds each one in a separate fp32
  pass, about 535 adds per 512-token batch and ~9% of the time.
- Ceilings measured on this GPU: ~35–40 TFLOPS for a bf16 / fp16 matmul (rocBLAS, hipBLASLt and Triton alike;
  theoretical 54–59); causal flash attention from AOTriton ~10 TFLOPS at 6k tokens, the same as llama.cpp's.
- Not bottlenecks: the CPU and the server, weight bandwidth (even a 314-token row is compute-bound), clock ramp-up (a
  request after 1 s idle takes as long as one after another), llama.cpp's batch size (`n_ubatch` 2048: 7% faster).
- An off-the-shelf PyTorch forward (bf16, hipBLASLt with the bias in the epilogue, AOTriton flash attention) takes
  3.70 s at XL, 1.13× faster than `gguf`, with the same split. `--mode fast-nocompile` as shipped is slower still on
  this GPU (XL 34 s, start-up 888 s: 81 captured shapes, rows over 3,072 tokens run eagerly).

## What `--mode strix` does

| | at 6,277 tokens unless noted | stock kernel |
|---|---|---|
| **attention**: Triton flash attention over the packed prefix trie, GQA-packed (the 8 query heads of a kv head share each key / value tile) | 20 TFLOPS | AOTriton causal 10; FlexAttention with the block mask of a 20-question row 28 ms / layer, this kernel 8 |
| **FFN input**: gate and up projections, their biases and SwiGLU in one kernel (two accumulators, one load of the activations) | 32 TFLOPS | `F.linear` + silu × up: 17 |
| **qkv, o, down**: Triton matmuls with the bias and the residual add in the epilogue, tile configs per row count | o 2.4 ms, down 35–40 TFLOPS; qkv 0.13 ms at 314 tokens | `F.linear` + add: o 3.7 ms; qkv 0.17 ms |
| RMSNorm, rotary (written head-major for the attention kernel) | ~1% of the forward | – |

- GraphBackend's block mask of a packed row (the prefix trie of a request: the state once, every question and option
  order after it) becomes an interval per key: depth-first packing puts every block's subtree in one contiguous range,
  so key k is visible to query q iff k ≤ q < sub_end[k]. Key blocks no query of a block can see are skipped, blocks
  every query sees run without a mask, so a packed row costs what a causal row of the same length costs. Rows of
  several requests are separate trees of one sequence: no padding, no captured shapes, no graphs, up to 16,384 tokens
  per forward; longer requests are bisected like in GraphBackend.
- The last layer computes attention, o and the FFN for the readout rows only.
- Rounding follows the stock bf16 forward step by step (RMSNorm in fp32 then the weight in bf16, rotary as
  round(q cos) + round(rot(q) sin), silu(gate) and its product with up rounded to bf16, bias then residual each
  rounded). q, k and v are one projection; the Hugging Face modules keep views of the fused weights, so the evidence
  head still runs the stock model.
- The server waits for the GPU by polling an event with 0.5 ms sleeps: ROCm's synchronize spins a CPU core at 100% for
  the whole forward.

## Results against `gguf`: basal-2.0

basal-2.0-4.5B has the architecture of basal-1.5 (Llama, 60 layers, hidden 2,048, GQA 16 / 2, biases on every linear
layer), and `strix` runs it without changes: its option-code readout goes through the same readout rows. Prompt format 2
reads one option order by default (basal-1.5 read two), so a request has about half the prompt tokens. Engines:
`--mode strix` with the bf16 weights, `--mode gguf` with `basal-2.0-4.5B-Q8_0.gguf`; both use the model's
`CALIBRATION.json` (the same file in both folders).

Interleaved A/B (`bench_ab.py`: both servers up, each shape sent to both engines in alternating order, median of 5
rounds; 32 clients: median of 2 bursts of 128 single questions each), so thermal drift and the power limit hit both
engines alike:

| request | prompt tokens | `gguf` | **`strix`** | speed-up |
|---|---|---|---|---|
| S (≈100 tokens), 1 question | 201 | **136 ms** | 144 ms | 0.94× |
| S, 5 questions | 1,508 | 277 ms | **203 ms** | 1.37× |
| M (≈570), 1 question | 669 | 397 ms | **263 ms** | 1.51× |
| M, 10 questions | 8,825 | 816 ms | **464 ms** | 1.76× |
| L (≈2,000), 1 question | 2,109 | 1.27 s | **714 ms** | 1.77× |
| L, 5 questions | 16,772 | 1.41 s | **788 ms** | 1.79× |
| XL (≈6,000), 1 question | 6,163 | 4.16 s | **2.41 s** | 1.73× |
| XL, 5 questions | 49,204 | 4.37 s | **2.52 s** | 1.73× |
| XL, 20 questions | 141,818 | 5.25 s | **2.90 s** | 1.81× |
| single questions, 32 clients | – | 11.5/s | **19.0/s** | 1.65× |

- Faster than `gguf` on every shape except one short question at a time (136 vs 144 ms; one client in the matrix:
  8.9 vs 7.1/s). The gain at long states is larger than with basal-1.5 (XL with 1 question, matrix: 1.70× against
  1.52×).
- Each added question costs 12–28 ms with `strix`, 37–55 ms with `gguf` (matrix, 1 → 20 questions).
- Answers: 142 of 144 decisions equal to the fp32 reference for both engines (`strix` mean |Δp| 0.0047, max 0.028;
  `gguf` 0.0064, 0.068). The two others are one question (`cx-03-filing_date`, L with 10 and 20 questions) that fp32
  puts at 0.502 / 0.498 and both engines the other way round, at 0.48–0.50 / 0.50–0.52. All 26 edge-case decisions
  equal fp32. `strix` and `gguf` differ on 4 of 188 matrix decisions, all yes / no labels of a multi-label question
  on XL at 0.47–0.54.
- Start-up 6.2 s and 10.1 GiB GTT (`gguf` 3.3 s, 5.8 GiB), as with basal-1.5.
- The matrix run (one engine at a time, full tables under [Results: basal-2.0](#results-basal-20)) agrees with the A/B
  within 5%, except S with 1 question: `strix` 231 ms in the matrix and 142–148 ms in all 5 A/B rounds, `gguf` 126 and
  131–143 ms.

basal-1.5 → basal-2.0 on the same engines (matrix p50):

| request | prompt tokens | `strix` | `gguf` |
|---|---|---|---|
| S, 20 questions | 11,178 → 4,692 | 1.08 s → 467 ms | 1.79 s → 831 ms |
| M, 20 questions | 32,706 → 15,456 | 1.25 s → 618 ms | 2.12 s → 1.13 s |
| L, 1 question | 4,296 → 2,109 | 736 → 711 ms | 1.28 → 1.26 s |
| L, 20 questions | 98,946 → 48,576 | 2.12 → 1.17 s | 3.22 → 2.07 s |
| XL, 1 question | 12,404 → 6,163 | 2.73 → 2.47 s | 4.16 → 4.20 s |
| XL, 20 questions | 285,430 → 141,818 | 4.57 → 3.00 s | 6.77 → 5.26 s |
| single questions, 32 clients | – | 10.9 → 20.3/s | 7.7 → 12.1/s |

Prompt tokens are `usage.input_tokens`, the sum over all prompts; the state is computed once per request either way, so
one question over a long state costs about the same with both models, and the second option order basal-1.5 read for
every question is what basal-2.0 saves.

## Results against `gguf`: basal-1.5 (request matrix over HTTP)

| state | 1 question | 5 questions | 20 questions |
|---|---|---|---|
| S (≈100 tokens) | 178 → **159 ms** (1.12×) | 573 → **348 ms** (1.65×) | 1.79 → **1.08 s** (1.66×) |
| M (≈570) | 439 → **289 ms** (1.52×) | 817 → **480 ms** (1.70×) | 2.12 → **1.25 s** (1.70×) |
| L (≈2,000) | 1.28 s → **736 ms** (1.74×) | 1.73 → **1.09 s** (1.59×) | 3.22 → **2.12 s** (1.52×) |
| XL (≈6,000) | 4.16 → **2.73 s** (1.52×) | 4.78 → **3.28 s** (1.46×) | 6.77 → **4.57 s** (1.48×) |

- Fastest engine on all 20 request shapes. Each added question costs 48–97 ms (`gguf` 85–137 ms).
- Throughput, single questions over short states: 1 client 6.9 → 6.1/s (0.88×), 8 clients 7.6 → 10.2/s, 32 clients
  7.7 → 10.9/s (1.42×).
- Answers: 144 of 144 decisions equal to the fp32 reference (mean |Δp| 0.0042; `gguf` 143/144, 0.0067), all 26
  edge-case decisions, 187 of 188 equal to `gguf` (the one difference is a near-tie on which `strix` agrees with fp32).
- Start-up 6–7 s (Triton compiles the kernels at the first start and caches them); GTT 10.0 GiB, against 5.8 GiB for
  `gguf` (bf16 weights instead of Q8_0).

### Limits

- **Long states stay below 2×.** The matmuls of the 6,277-token request are 58 TFLOP: at the ~35 TFLOPS this GPU
  sustains that is 1.65 s before any attention. And with `strix` the package runs at its power limit (about 160 W for
  the first ~25 s of load, then about 140 W), so the GPU clock drops from ~2.9 to 2.15–2.33 GHz; llama.cpp's slower
  kernels stay under it. A single XL forward on a cool GPU takes 2.37 s (1.76×); in the matrix, after minutes of load,
  2.73 s (1.52×).
- **One short question at a time stays faster with `gguf`** (141 vs 159 ms): the bf16 weights are 9 GB per forward,
  the Q8_0 file 5 GB.
- Llama models with silu, a power-of-two hidden size and an FFN size divisible by 64 (checked at start-up); bf16 only.
  Tile configs were tuned on the 8060S; the mode has not been run on other GPUs.

## Method

**Request matrix** (`bench_matrix.py`): four states built from the bundled examples, S / M / L / XL ≈ 100 / 570 / 2,000 /
6,060 tokens, each with 1, 2, 5, 10 and 20 questions from a fixed mixed pool (choice, yes/no, score, an `act` with costs
and a 4-label `multi`, so 20 questions are 23 branches and 46 prompts). Pass 0 keeps the answers; two timed passes put a
marker in front of the state so no engine can answer from a prompt cache. Concurrency: single questions over short
states, 48–256 requests per level.

**Reference**: `basal-serve --mode eager --dtype float32 --soam off --max-batch 1`, every prompt computed on its own in
fp32 (S, M and L with all question counts, XL with 1 and 2; XL with more questions takes minutes per request in fp32).

**Edge cases** (`edge_check.py`): 12 requests (mixed PL / EN questions over one state, 10 options, the same question
three times, a JSON state, `facts: auto`, long option texts, a 5-level score, an 8-label `multi` with `min` / `max`, an
`act` with a cost matrix, English only, a 2-character state, `option_keys` show vs hide), sent one after another and
then all at once (the server batches them).

**Engines** (all `basal-serve` on one port):

| engine | server |
|---|---|
| `gguf` | `--mode gguf --model <-GGUF folder> --gguf basal-1.5-4.5B-Q8_0.gguf`, llama-cpp-python built with Vulkan |
| `fast-nocompile` | `--mode fast-nocompile` as shipped (no extra environment); XL with 5 or more questions not measured |
| **`strix`** | `--mode strix --model Remek/basal-1.5-4.5B` |
| fp32 reference | `--mode eager --dtype float32 --soam off --max-batch 1` |

basal-2.0: the same servers with `--model Remek/basal-2.0-4.5B` (`strix`, fp32 reference) and
`--model <basal-2.0-4.5B-GGUF folder> --gguf basal-2.0-4.5B-Q8_0.gguf` (`gguf`); `fast-nocompile` was not run.

Start-up is the time to the first `/health`; GTT the memory the server adds (`mem_info_gtt_used` before the start and
after it; it does not grow under load).

Reproduce: start an engine, then

```bash
python docs/benchmarks/bench_matrix.py --url http://127.0.0.1:8000/v1/systemone --label strix
python docs/benchmarks/edge_check.py --url http://127.0.0.1:8000/v1/systemone --label strix
python docs/benchmarks/report.py results/matrix results/edge > report.md
# interleaved A/B of two running engines (basal-2.0 results in results/2.0/)
python docs/benchmarks/bench_ab.py --a gguf=http://127.0.0.1:8031/v1/systemone \
    --b strix=http://127.0.0.1:8030/v1/systemone --out results/2.0/ab.json
```

`results/matrix` and `results/edge` hold the basal-1.5 runs, `results/2.0/` the basal-2.0 runs.

## Tests

`tests/test_strix.py`: every Triton kernel against its stock counterpart (attention against a dense masked softmax over
packed rows of several requests, readout-only queries included; rotary bit-identical to the Hugging Face rotary;
projections for every row-count config and row chunk, with and without the residual; FFN input and RMSNorm against the
stock layer), the interval mask against the block mask it replaces, and the whole backend on a tiny random Llama with
biases against separate bf16 forwards of every prompt (1–8 questions over one state, several requests in one sequence
and the same request alone, several forwards per call with a request bisected, determinism), plus the mode dispatch of
`basal-serve` and `basal-bench`. Full suite on this machine: 76 passed, 3 skipped (MLX / Apple only); on engine 2.0.0,
511 passed, 19 skipped (MLX and the extras not installed).

## Results: basal-1.5

Generated by `report.py` from `results/matrix` and `results/edge` (raw answers and latencies of every run).

### gguf → strix per request shape

`--mode gguf` (llama.cpp in process, Vulkan, Q8_0) against `--mode strix` (PyTorch + Triton kernels for gfx1151, bf16). p50 of 2 timed passes, unique states.

| state | questions | prompt tokens | gguf | strix | speed-up |
|---|---|---|---|---|---|
| S | 1 | 480 | 178 ms | **159 ms** | 1.12× |
| S | 2 | 928 | 242 ms | **194 ms** | 1.25× |
| S | 5 | 3,640 | 573 ms | **348 ms** | 1.65× |
| S | 10 | 6,496 | 1.13 s | **671 ms** | 1.68× |
| S | 20 | 11,178 | 1.79 s | **1.08 s** | 1.66× |
| M | 1 | 1,416 | 439 ms | **289 ms** | 1.52× |
| M | 2 | 2,800 | 501 ms | **304 ms** | 1.65× |
| M | 5 | 11,128 | 817 ms | **480 ms** | 1.70× |
| M | 10 | 18,664 | 1.39 s | **797 ms** | 1.75× |
| M | 20 | 32,706 | 2.12 s | **1.25 s** | 1.70× |
| L | 1 | 4,296 | 1.28 s | **736 ms** | 1.74× |
| L | 2 | 8,560 | 1.33 s | **766 ms** | 1.74× |
| L | 5 | 34,168 | 1.73 s | **1.09 s** | 1.59× |
| L | 10 | 56,104 | 2.38 s | **1.51 s** | 1.57× |
| L | 20 | 98,946 | 3.22 s | **2.12 s** | 1.52× |
| XL | 1 | 12,404 | 4.16 s | **2.73 s** | 1.52× |
| XL | 2 | 24,776 | 4.25 s | **2.76 s** | 1.54× |
| XL | 5 | 99,032 | 4.78 s | **3.28 s** | 1.46× |
| XL | 10 | 161,508 | 5.65 s | **3.85 s** | 1.47× |
| XL | 20 | 285,430 | 6.77 s | **4.57 s** | 1.48× |

| clients (single questions) | gguf dec/s · p50 | strix dec/s · p50 | throughput |
|---|---|---|---|
| 1 | 6.9/s · 141 ms | **6.1/s** · 159 ms | 0.88× |
| 4 | 7.4/s · 540 ms | **8.4/s** · 470 ms | 1.14× |
| 8 | 7.6/s · 1.06 s | **10.2/s** · 787 ms | 1.35× |
| 16 | 7.7/s · 2.07 s | **10.5/s** · 1.51 s | 1.37× |
| 32 | 7.7/s · 4.12 s | **10.9/s** · 2.92 s | 1.42× |
| 64 | 7.7/s · 8.33 s | **11.0/s** · 5.79 s | 1.44× |

### Start-up and memory

| engine | start-up | GTT after start |
|---|---|---|
| gguf | 3.2 s | 5.8 GiB |
| fast-nocompile | 888.4 s | 32.1 GiB |
| **strix** | 6.1 s | 10.0 GiB |
| eager fp32 (reference) | 5.6 s | 18.4 GiB |

### Latency per request, all engines (p50; bold: fastest)

| state | q | branches | gguf | fast-nocompile | **strix** |
|---|---|---|---|---|---|
| S | 1 | 1 | 178 ms | 309 ms | **159 ms** |
| S | 2 | 2 | 242 ms | 362 ms | **194 ms** |
| S | 5 | 8 | 573 ms | 1.30 s | **348 ms** |
| S | 10 | 13 | 1.13 s | 3.08 s | **671 ms** |
| S | 20 | 23 | 1.79 s | 5.28 s | **1.08 s** |
| M | 1 | 1 | 439 ms | 786 ms | **289 ms** |
| M | 2 | 2 | 501 ms | 792 ms | **304 ms** |
| M | 5 | 8 | 817 ms | 1.66 s | **480 ms** |
| M | 10 | 13 | 1.39 s | 3.62 s | **797 ms** |
| M | 20 | 23 | 2.12 s | 7.33 s | **1.25 s** |
| L | 1 | 1 | 1.28 s | 3.09 s | **736 ms** |
| L | 2 | 2 | 1.33 s | 3.68 s | **766 ms** |
| L | 5 | 8 | 1.73 s | 4.96 s | **1.09 s** |
| L | 10 | 13 | 2.38 s | 19.44 s | **1.51 s** |
| L | 20 | 23 | 3.22 s | 24.44 s | **2.12 s** |
| XL | 1 | 1 | 4.16 s | 34.02 s | **2.73 s** |
| XL | 2 | 2 | 4.25 s | 67.15 s | **2.76 s** |
| XL | 5 | 8 | 4.78 s | – | **3.28 s** |
| XL | 10 | 13 | 5.65 s | – | **3.85 s** |
| XL | 20 | 23 | 6.77 s | – | **4.57 s** |

### Cost of one more question (1 → 20 questions, per added question)

| state | gguf | fast-nocompile | **strix** |
|---|---|---|---|
| S | 85 ms | 262 ms | 48 ms |
| M | 88 ms | 344 ms | 50 ms |
| L | 102 ms | 1.12 s | 73 ms |
| XL | 137 ms | – | 97 ms |

### Concurrency (single questions over short states, unique per request)

| clients | gguf | fast-nocompile | **strix** |
|---|---|---|---|
| 1 | 6.9/s · 141 ms | 3.8/s · 246 ms | 6.1/s · 159 ms |
| 4 | 7.4/s · 540 ms | 3.8/s · 1.10 s | 8.4/s · 470 ms |
| 8 | 7.6/s · 1.06 s | 4.1/s · 1.95 s | 10.2/s · 787 ms |
| 16 | 7.7/s · 2.07 s | 4.3/s · 3.70 s | 10.5/s · 1.51 s |
| 32 | 7.7/s · 4.12 s | 4.6/s · 6.93 s | 10.9/s · 2.92 s |
| 64 | 7.7/s · 8.33 s | 5.3/s · 12.06 s | 11.0/s · 5.79 s |

### Agreement with the fp32 reference

Reference: `--mode eager --dtype float32 --soam off --max-batch 1` (every prompt computed separately in fp32), on 17 requests (S-q1, S-q2, S-q5, S-q10, S-q20, M-q1, M-q2, M-q5, M-q10, M-q20, L-q1, L-q2, L-q5, L-q10, L-q20, XL-q1, XL-q2). A decision is the top option of a question or the yes/no of one label of a multi-label question.

| engine | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| gguf | 143/144 | 0.0067 | 0.078 |
| fast-nocompile | 143/144 | 0.0045 | 0.037 |
| **strix** | 144/144 | 0.0042 | 0.028 |

### strix against gguf (whole matrix)

| comparison | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| gguf (Q8_0) vs strix (bf16) | 187/188 | 0.0080 | 0.100 |

### Edge cases

12 requests: mixed PL / EN questions over one state, 10 options, the same question three times, a JSON state, `facts: auto`, long option texts, a 5-level score, an 8-label multi with min / max, an `act` with a cost matrix, an English-only request, a 2-character state, `option_keys` show vs hide. Sent one after another, then all at once (the server batches them).

| engine | errors | concurrent = sequential decisions | concurrent vs sequential, max \|Δp\| | duplicate questions, max \|Δp\| | multi within min/max | act action | vs fp32: same decision | vs fp32: max \|Δp\| |
|---|---|---|---|---|---|---|---|---|
| gguf | none | 26/26 | 0.0183 | 0.0e+00 | True | human | 26/26 | 0.016 |
| **strix** | none | 26/26 | 0.0112 | 0.0e+00 | True | human | 26/26 | 0.009 |
| eager fp32 (reference) | none | 26/26 | 0.0000 | 0.0e+00 | True | human | – | – |

## Results: basal-2.0

Generated by `report.py` from `results/2.0/matrix` and `results/2.0/edge`.

### gguf → strix per request shape

`--mode gguf` (llama.cpp in process, Vulkan, Q8_0) against `--mode strix` (PyTorch + Triton kernels for gfx1151, bf16). p50 of 2 timed passes, unique states.

| state | questions | prompt tokens | gguf | strix | speed-up |
|---|---|---|---|---|---|
| S | 1 | 201 | 126 ms | **231 ms** | 0.54× |
| S | 2 | 386 | 145 ms | **149 ms** | 0.97× |
| S | 5 | 1,508 | 275 ms | **211 ms** | 1.30× |
| S | 10 | 2,741 | 541 ms | **320 ms** | 1.69× |
| S | 20 | 4,692 | 831 ms | **467 ms** | 1.78× |
| M | 1 | 669 | 397 ms | **264 ms** | 1.50× |
| M | 2 | 1,322 | 409 ms | **374 ms** | 1.09× |
| M | 5 | 5,252 | 539 ms | **326 ms** | 1.66× |
| M | 10 | 8,825 | 816 ms | **463 ms** | 1.76× |
| M | 20 | 15,456 | 1.13 s | **618 ms** | 1.82× |
| L | 1 | 2,109 | 1.26 s | **711 ms** | 1.77× |
| L | 2 | 4,202 | 1.29 s | **719 ms** | 1.79× |
| L | 5 | 16,772 | 1.42 s | **792 ms** | 1.79× |
| L | 10 | 27,545 | 1.72 s | **979 ms** | 1.76× |
| L | 20 | 48,576 | 2.07 s | **1.17 s** | 1.77× |
| XL | 1 | 6,163 | 4.20 s | **2.47 s** | 1.70× |
| XL | 2 | 12,310 | 4.22 s | **2.46 s** | 1.71× |
| XL | 5 | 49,204 | 4.42 s | **2.63 s** | 1.68× |
| XL | 10 | 80,247 | 4.82 s | **2.79 s** | 1.73× |
| XL | 20 | 141,818 | 5.26 s | **3.00 s** | 1.75× |

| clients (single questions) | gguf dec/s · p50 | strix dec/s · p50 | throughput |
|---|---|---|---|
| 1 | 8.9/s · 111 ms | **7.1/s** · 140 ms | 0.80× |
| 4 | 11.1/s · 358 ms | **12.3/s** · 325 ms | 1.11× |
| 8 | 11.6/s · 682 ms | **17.1/s** · 465 ms | 1.47× |
| 16 | 12.1/s · 1.30 s | **20.4/s** · 778 ms | 1.68× |
| 32 | 12.1/s · 2.60 s | **20.3/s** · 1.57 s | 1.68× |
| 64 | 12.0/s · 5.28 s | **20.4/s** · 3.10 s | 1.69× |

### Start-up and memory

| engine | start-up | GTT after start |
|---|---|---|
| gguf | 3.3 s | 5.8 GiB |
| **strix** | 6.2 s | 10.1 GiB |
| eager fp32 (reference) | 4.8 s | 17.9 GiB |

### Latency per request, all engines (p50; bold: fastest)

| state | q | branches | gguf | **strix** |
|---|---|---|---|---|
| S | 1 | 1 | **126 ms** | 231 ms |
| S | 2 | 2 | **145 ms** | 149 ms |
| S | 5 | 8 | 275 ms | **211 ms** |
| S | 10 | 13 | 541 ms | **320 ms** |
| S | 20 | 23 | 831 ms | **467 ms** |
| M | 1 | 1 | 397 ms | **264 ms** |
| M | 2 | 2 | 409 ms | **374 ms** |
| M | 5 | 8 | 539 ms | **326 ms** |
| M | 10 | 13 | 816 ms | **463 ms** |
| M | 20 | 23 | 1.13 s | **618 ms** |
| L | 1 | 1 | 1.26 s | **711 ms** |
| L | 2 | 2 | 1.29 s | **719 ms** |
| L | 5 | 8 | 1.42 s | **792 ms** |
| L | 10 | 13 | 1.72 s | **979 ms** |
| L | 20 | 23 | 2.07 s | **1.17 s** |
| XL | 1 | 1 | 4.20 s | **2.47 s** |
| XL | 2 | 2 | 4.22 s | **2.46 s** |
| XL | 5 | 8 | 4.42 s | **2.63 s** |
| XL | 10 | 13 | 4.82 s | **2.79 s** |
| XL | 20 | 23 | 5.26 s | **3.00 s** |

### Cost of one more question (1 → 20 questions, per added question)

| state | gguf | **strix** |
|---|---|---|
| S | 37 ms | 12 ms |
| M | 38 ms | 19 ms |
| L | 43 ms | 24 ms |
| XL | 55 ms | 28 ms |

### Concurrency (single questions over short states, unique per request)

| clients | gguf | **strix** |
|---|---|---|
| 1 | 8.9/s · 111 ms | 7.1/s · 140 ms |
| 4 | 11.1/s · 358 ms | 12.3/s · 325 ms |
| 8 | 11.6/s · 682 ms | 17.1/s · 465 ms |
| 16 | 12.1/s · 1.30 s | 20.4/s · 778 ms |
| 32 | 12.1/s · 2.60 s | 20.3/s · 1.57 s |
| 64 | 12.0/s · 5.28 s | 20.4/s · 3.10 s |

### Agreement with the fp32 reference

Reference: `--mode eager --dtype float32 --soam off --max-batch 1` (every prompt computed separately in fp32), on 17 requests (S-q1, S-q2, S-q5, S-q10, S-q20, M-q1, M-q2, M-q5, M-q10, M-q20, L-q1, L-q2, L-q5, L-q10, L-q20, XL-q1, XL-q2). A decision is the top option of a question or the yes/no of one label of a multi-label question.

| engine | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| gguf | 142/144 | 0.0064 | 0.068 |
| **strix** | 142/144 | 0.0047 | 0.028 |

### strix against gguf (whole matrix)

| comparison | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| gguf (Q8_0) vs strix (bf16) | 184/188 | 0.0076 | 0.074 |

### Edge cases

12 requests: mixed PL / EN questions over one state, 10 options, the same question three times, a JSON state, `facts: auto`, long option texts, a 5-level score, an 8-label multi with min / max, an `act` with a cost matrix, an English-only request, a 2-character state, `option_keys` show vs hide. Sent one after another, then all at once (the server batches them).

| engine | errors | concurrent = sequential decisions | concurrent vs sequential, max \|Δp\| | duplicate questions, max \|Δp\| | multi within min/max | act action | vs fp32: same decision | vs fp32: max \|Δp\| |
|---|---|---|---|---|---|---|---|---|
| gguf | none | 26/26 | 0.0446 | 0.0e+00 | True | human | 26/26 | 0.038 |
| **strix** | none | 26/26 | 0.0318 | 0.0e+00 | True | human | 26/26 | 0.027 |
| eager fp32 (reference) | none | 26/26 | 0.0000 | 0.0e+00 | True | human | – | – |
