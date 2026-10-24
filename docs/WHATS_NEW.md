# What is new in basal 2.4.2 (not released yet)

- **`--compile-scope layer`** (opt-in, `--mode fast`, the Llama-based basal-2.0 models): torch.compile compiles one
  decoder layer, called for every layer, instead of the whole model, so start-up compiles a layer instead of 32-60.
  The same transformers layers as the whole-model compile; kernels fuse within a layer only, so probabilities move by
  bf16 rounding. RTX 4060 Ti, graph warm-up from cold compile caches: basal-2.0-mini 235 -> 95 s, basal-2.0-4.5B 515
  -> 251 s; five questions at a time as fast, single questions 2-3% slower on mini and as fast on the 4.5B (within
  run-to-run noise); identity readouts: the same top answer on 416 / 420 (mini) and 418 / 420 (4.5B), the changed ones
  near-ties (margins under 0.08), mean |dp| 0.005, as far from fp32 as the whole-model compile (mini: 4.7e-3 against
  4.8e-3). bf16 Llama weights only (FP8 / NVFP4, Gemma 4, Qwen3.5 and --lean are refused). Contributed by Miłosz
  Kucharski (@Butterski), [#49](https://github.com/rkinas/basal/pull/49).

# What is new in basal 2.4.1

- **Shorter start-up of the graph engine** (`--mode fast`): every CUDA graph is warmed up once before its capture
  instead of three times (pass-2 graphs: twice). torch.compile compiles and cuBLAS and Triton set up at the first
  call; the further calls ran the same kernels again. RTX 4060 Ti 16 GB, the graph warm-up: basal-2.0-mini 468 ->
  323 s, basal-2.0-4.5B 1,094 -> 680 s (with the pass-2 graph budget it needs there); the same GPU memory and speed,
  every probability bit-identical (420 readouts per model). Contributed by Miłosz Kucharski (@Butterski),
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
  -0.75 GB, the same top answer on 160 of 160 (max |Δp| 9e-6). On 32 threads of a Xeon 8460Y+ the 40-question case
  runs at about the same speed (0.94-1.04x), 150 questions 1.24x faster. Contributed by Milosz (@Butterski),
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
  per second on single questions (vLLM bf16: 16.7) and 10.6 vs 0.7 questions per second with five questions over a long
  state; mini 61.2 vs 10.0 and 21.4 vs 1.6; ultra 6.1 vs 2.0 and 4.8 vs 0.3. Same answers within bf16 rounding (dev
  v2: 0.998-0.999 of the top answers at margin >= 0.2; with the shipped calibration files the automated share at 1%
  within 0.1 points). On an RTX PRO 6000 Blackwell (`--mode fast --gemma-attn flex`, opt-in until its calibration
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

# What is new in basal 2.2

**Engine v2.2.0: calibration on your own data, for every basal model**
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
