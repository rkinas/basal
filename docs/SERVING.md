# Serving, batch inference and benchmarks

Flags of `basal-serve`, inference on a JSONL file with `basal-run`, a look inside one decision with `basal-viz`, and
benchmarking a GPU.

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | hardware |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + SOAM packing + compact batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fp8` | the `-FP8` ModelOpt checkpoints natively; bf16 weights quantised to FP8 at load time (torchao) | GPUs with FP8 (sm89+): Hopper, Blackwell |
| `nvfp4` | `fast` with bf16 weights quantised to NVFP4 at load time (torchao, experimental; costs accuracy; not the basal-1.5 / 1.0 `-NVFP4` checkpoints) | Blackwell |
| `vllm` | vLLM backend, readout from the option codes' log-probabilities; bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only), and the basal-1.5 / 1.0 `-NVFP4` checkpoints; `--vllm-preset deterministic` for repeatable answers, basal-omni media with `--media on` ([OMNI.md](OMNI.md#vllm---mode-vllm)) | CUDA GPUs supported by vLLM |
| `sglang` | SGLang backend, same readout | CUDA GPUs supported by SGLang |
| `eager` | plain PyTorch reference; `--device cuda\|mps\|cpu` (default: `BASAL_DEVICE`, else CUDA, else Apple Silicon, else CPU) | any GPU, Apple Silicon (MPS) or CPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen per request | models that ship `exit_heads/` (basal-1.0 only) |
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` ports or bf16 weights; shared-prefix KV cache (opt-in `--mlx-packed`: short Llama groups as one packed forward, [QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)); opt-in `--state-cache-gb N` / `--prefix-cache on` across requests, as on the CUDA graph engine ([HARDWARE.md](HARDWARE.md#apple-m4-state-cache-on---mode-mlx)) | Apple Silicon |
| `mlx-q8` | `mlx` with 8-bit weights made from bf16 at load time (about half the memory, not faster) | Apple Silicon |
| `mps` | the PyTorch engine without CUDA graphs: SOAM packing + token-budget batching; on the Mac's GPU, or on the CPU (`--device cpu`, the default without MPS), where a request's state runs once instead of once per prompt as in `eager` | Apple Silicon, CPU |
| `gguf` | a GGUF file in process through llama.cpp (`--gguf <file>`, extra `basal[gguf]`) | Apple Silicon (Metal), CUDA, CPU |
| `strix` *(experimental, community-maintained)* | basal-1.5 / basal-2.0 (Llama architecture): PyTorch + Triton kernels written for gfx1151: flash attention over the packed prefix trie, gate / up + SwiGLU fused, matmuls with the bias and residual add in the epilogue, all rows of a batch in one sequence (bf16; [HARDWARE.md](HARDWARE.md#amd-strix-halo-experimental-community-maintained), [benchmarks](benchmarks/STRIX_HALO.md)) | AMD Strix Halo (ROCm + Triton); measured only there |
| `ollama` | the basal server in front of Ollama (GGUF ports; `--ollama-model`, `--ollama-url`) | anything Ollama runs on |
| `llamacpp` | the basal server in front of `llama-server` (GGUF ports; `--llamacpp-url`) | anything llama.cpp runs on |
| `tpu` *(experimental, community-maintained)* | JAX / XLA on one TPU chip: SOAM packing, one compiled program per shape, `--max-len` (extra `basal[tpu]`, [install](INSTALL.md#google-tpu-experimental-community-maintained)) | Google TPU (v5e measured) |

- Without `--mode`: `fast` on CUDA, `mlx` on Apple Silicon (`mps` without `basal[mlx]`), `tpu` on a TPU VM with
  `basal[tpu]`, `eager` otherwise. On a CPU, `--mode mps` computes a request's state once for all its prompts and is
  faster for requests with several questions or option orders; `eager` stays the default there (the reference path).
- **Qwen3.5 models (Gated DeltaNet layers)**, such as basal-omni-pico: `fast` and `fast-nocompile` serve them since 2.4
  (2.3.2 refused them): the graph engine keeps each request state's recurrent state beside its keys and values and
  continues every question from it (one row per question and option order in the branch pass), with fla
  (flash-linear-attention) for the Gated DeltaNet layers. Without `--mode` they run `fast` on Hopper (H100, the GPU
  class the port was gated on) when fla is installed, `eager` elsewhere. `--state-cache-gb` (2.5) keeps a request's
  state as a full-state checkpoint: the attention layers' keys and values and each Gated DeltaNet layer's recurrent
  state and convolution tail after the state; the same state again runs only its questions. A state that begins with
  a whole cached one (a growing agent trace) continues it from there; one that differs inside the cached tokens runs
  again (a recurrent state cannot be cut there). Since 2.5 they also run with `--prefix-cache on` and `--batches
  compact` (opt-in; compact rows also with `--template-prefix on` and `--readout-layer on`), with answers as close to
  float32 as `--mode eager`'s; `--lean` and `--tree flex` change nothing for them. Not with them yet: `fp8` / `nvfp4`,
  `fast-exit`, `mps` and `--long-lane` (each refused before loading). `--long-lane` runs for them with
  `--long-lane-gdn on` (2.5, opt-in, off by default): its bf16 answers are a little further from float32 than
  `--mode eager`'s. basal-omni-pico on an H100,
  1,700 held-out decisions against eager float32: with five questions per state the same top answer on 99.39% (8,500
  readouts, 52 changed) against eager bf16's 99.51% (42); single questions 99.71%, two option orders 99.65%, both
  99.46%, within 0.1 pt of eager bf16's. No speed was measured for it. `--mode vllm` and `--mode gguf` serve Qwen3.5 models
  too. Other models with recurrent-state layers (Qwen3-Next, Mamba and other hybrids) are refused by every
  shared-prefix engine and run `eager` without `--mode`.
- **`--mode fp8` below compute capability 8.9** (Ampere, e.g. RTX 3090 / A100) stops before loading the model (2.3.2;
  before, torchao failed after the load): serve bf16 with `--mode fast` there.
- **basal-omni (Gemma 4) in `eager`** (2.3): the engine's own Gemma 4 forward: a request's state (and media) once,
  then every question over it, and mini's last layers at the answer rows only; `--eager-forward hf` keeps the
  transformers forward of 2.2. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- **basal-omni (Gemma 4) in `fast` / `fast-nocompile`** (2.3, text): the engine's own Gemma 4 layer code with SOAM
  over its sliding-window layers and the answer-rows-only last layers; `--gemma-attn auto` = FlashAttention 3 + flex,
  compiled, on Hopper, flex on Blackwell workstation GPUs (sm_120), else eager's math kernel uncompiled; `fp8` /
  `nvfp4` quantisation at load time is refused for it (serve an `-FP8` checkpoint). Without `--mode` it runs `fast` on
  Hopper with FlashAttention 3 and on sm_120 Blackwell, `eager` on other GPUs. [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)
- `--calibration file.json` overrides the calibration file (default: `CALIBRATION.<mode>.json` when the model ships
  one for that engine, else `CALIBRATION.json`; `--mode gguf` loads `CALIBRATION.gguf.json` when the model ships one,
  else `CALIBRATION.json`; with `--orders 2`,
  `CALIBRATION.o2.json` when shipped). A file applies its thresholds only to the option orders it was fitted with. `basal-calibrate` fits one on your labelled requests
  ([CALIBRATION.md](CALIBRATION.md)). A missing file is refused at start-up.
- **Option orders** (`--orders 1 | 2 | auto`): `1` asks the order given (the default for prompt-format-2 models such as
  basal-2.0, which are trained with random option orders). `2` also asks the reversed order and averages the
  probabilities (the default for basal-1.5 and basal-1.0); both orders share the state and the question text, so the
  second order costs only its options. `auto` asks order 1 and the reversed order only where order 1 is uncertain
  (`--auto-tau`, `--auto-band`). See [FORMATS.md](FORMATS.md#option-orders).
- **Mean over K option orders** (`--order-average K`, 2.5, off by default): the probabilities are averaged over K
  orders: the given order, the reversed one, then each rotation and its reverse (with 2k orders every option is read at
  every position twice). All K orders read the request's state once, so each order costs only its question and
  options. `--order-average 2` equals `--orders 2`; a question with k options gets at most k! orders. The model's
  calibration was fitted on its own orders: without a `CALIBRATION.o<K>.json` its temperatures are kept, its
  thresholds are dropped and the answers are marked unvalidated (`basal-calibrate --order-average K` fits a file for
  K orders). Measure first whether it helps your model and data with `basal-bench --order-check` (below).
- **SOAM** (`--soam on`, default; off for a two-way model, below): every question and option order of a request is one
  packed row over the state.
  `--soam off` restores the per-question path of basal-1.0 (same answers in fp32). See [SOAM.md](SOAM.md).
- **Start-up**: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (several
  minutes the first time, about half with a warm compile cache). A request with several branches (option orders,
  questions, `multi` labels) runs in two exact passes: the state once, keeping its keys and values, then all branches
  over them ([SOAM.md](SOAM.md#which-kernel-runs)).
- **Prepared start-up** (2.5): `basal-prepare --model <model> [basal-serve's engine flags]` starts the engine once on
  this GPU as `basal-serve` starts it with those flags (its defaults included: on Hopper compact rows with the template
  prefix; give the flags the server will run with, such as `--mode fp8`, `--batches compact` or `--compile-scope
  layer`) and saves what it compiled (Inductor's kernels and autotuning results, and for Qwen3.5 models the configs
  fla's kernels picked) under `~/.cache/basal/prepared/<key>` (or `--out`). `basal-serve --prepared auto` (the
  default) loads such a directory before compiling when its key matches: the GPU, the torch / triton / fla / CUDA and
  basal versions (a basal patch release that changes no compiled code keeps the key: 2.5.1 loads 2.5.0's directories),
  the model's architecture (config.json; fine-tunes of one architecture share it), the mode, dtype and attention. The
  other engine flags are not in the key: preparing again with other flags adds their kernels to the same directory
  (one `inductor*.bin` per flag set, all loaded), and a start with flags it was not prepared with compiles what it
  lacks. A model may ship its own under `prepared/<key>/`. Then the server compiles nothing it has not seen
  and picks the same kernels in every start; CUDA graphs are still captured. `--prepared off` compiles as before;
  `--prepared <dir>` loads that directory or refuses to start. Compiled kernels are code the compiler cache runs, so
  the ones a Hugging Face repository ships load only for trusted publishers: `BASAL_PREPARED_TRUST` (comma-separated
  owners, `*` for every one; default `Remek`). A local model directory, `--prepared <dir>` and what `basal-prepare`
  wrote on this machine are trusted as they are.
- **Batching**: concurrent requests are batched into the same forward passes and every request is answered as soon as
  its passes are read out. Under load the server waits up to `--batch-window-ms` (default 3) for as many requests as
  the previous batch had; an idle server answers at once. `--max-batch` and `--wait-ms` control batching further.
- **`--deterministic`**: answers do not depend on load or on the other requests: every request runs in its own
  forward passes, never batched with others, with deterministic kernels (FP8 checkpoints: GEMMs with full
  accumulation). Costs about 4% of the throughput under load with one option order; in-process engines only
  (`basal-loadtest --determinism N` checks a running server, including decision flips under load). Batch-invariant
  kernels (2.3.4, tested, not shipped) made basal-omni's batched answers bit-identical, but were slower than one
  request per forward (H100, basal-omni-1.0-mini: 154 instead of 163 decisions/s), so `--deterministic` keeps one
  request per forward; a faster batch-invariant GEMM is future work.
- **`--layout canonical | state_twice | qfirst`** (2.4.1; prompt-format-2 models trained on the layouts):
  `canonical` (the default) puts the state first, so a request's questions share it and
  `--state-cache-gb` / `--prefix-cache` can keep it. `state_twice` reads the state twice. `qfirst` also puts the
  question before the state, so the state is read knowing the question: one pass per question, slower for requests
  with several questions, without the state caches. Each layout loads `CALIBRATION.<layout>.json` when the model ships
  one; with only the canonical file its temperatures are kept, its thresholds are not applied, and the answers are
  marked unvalidated. Without `--layout` a model is served in its basal.json `"layout"` (canonical when absent), and
  `/v1/models` reports a non-canonical layout. Images and audio are read in the canonical layout only.
- **Second read of uncertain answers** (2.5, `--reread <threshold>`, prompt-format-2 models, off by default): a choice
  or yes / no answer of a short request (`--reread-max-tokens`, 2,048) whose calibrated top probability is below the
  threshold is read again with the state twice (the `state_twice` layout), and the two readouts are combined: the
  mean of their log-probabilities (`--reread-combine mean`, the default) or the second read alone (`second`), then
  the type's temperature. Such answers carry `"reread": true`, and `usage.reread_branches` counts them. Off, nothing
  changes. With `--media on`, `--reread-image <budget>` does the same for a request with an image: its uncertain
  answers are read again with every image at that many soft tokens (Gemma 4: 70, 140, 280, 560 or 1120; the
  request's `"detail"` sets the first read's), and the two reads are combined the same way.
- **Two-way models** (2.5; basal.json `"two_way"`, a Gemma 4 model trained with two-way reading): the global layers
  read each prompt whole, both ways, so the state is read knowing the question; the sliding-window layers stay
  causal. `"two_way": true` takes the limits 4,096 / 8,192 tokens, or set `{"max_prompt": ..., "max_record": ...}`.
  A prompt over `max_prompt` tokens, or every prompt of a request over `max_record` tokens (its shared prefix once,
  then every prompt's own tokens), is read one way, as the model was trained. Nothing is shared between questions, so
  such a model is served with SOAM off (`--soam` defaults to off for it; `--soam on` is refused) and without
  `--state-cache-gb` / `--prefix-cache`, by `--mode eager` (the engine's own Gemma 4 forward) or the CUDA graph
  engine (`fast`, `fast-nocompile`: every prompt a plain row of its own, the math kernel in the global layers; not
  with `--batches compact`, `--lean`, `--template-prefix` or `--long-lane`). `--reread` is refused too: it reads the
  state twice (the state_twice layout), which a two-way model was not trained on. `/v1/models` reports the limits.
- **Isolated options** (basal.json `"option_isolation": true`, a Gemma 4 model trained so): each option is read in a
  block of its own, after the state and the question but not after the other options, and every block starts at the
  same position; the answer reads all the blocks. The options are then read as a set: apart from its code, an option's
  reading does not depend on where it stands in the list. The prompt is tokenized piece by piece (prefix, one block per
  option, suffix), as such a model is trained; the tokenizer must tokenize those pieces as the whole prompt (Gemma 4
  does; the engine checks it at start). In this version each prompt is read whole by the engine's own eager Gemma 4
  forward: `--mode eager`, SOAM off (its default for such a model; `--soam on` is refused), no `--state-cache-gb` /
  `--prefix-cache`, no `--reread`, text only. `/v1/models` reports `"option_isolation": true`.
- **Long states**: `--context-overflow truncate` (the default) cuts the middle of a state that does not fit the
  context (the first 1/3 and the last 2/3 of the kept tokens, `--truncate-head`; the question and options stay whole)
  and says what was cut in `usage.truncated`; `--context-overflow refuse` returns HTTP 422 instead. The context is
  32,768 tokens for basal-2.0 (4.5B) and max, 8,192 for mini (Ollama mode: 4,096; llama.cpp mode: `llama-server -c`;
  vLLM / SGLang: `--max-len`).
- **Engine v2.0.0 items** (bf16 weights in `--mode fast` on Hopper with FlashAttention 3; each item has a flag,
  `auto` = on):
  - `--batches compact`: single prompts packed back to back into one row (FA3 varlen), graphs per token count, no
    padding to bucket lengths or row counts. On other sm80+ GPUs `--batches compact` runs them through torch's varlen
    attention (Llama models; `auto` keeps them off there), and basal-omni on Blackwell workstation GPUs through flex
    block masks (2.5; faster, but its answers differ from the default rows':
    [OMNI.md](OMNI.md#what-the-engine-does-differently-for-gemma-4)). basal-2.0-mini on an RTX 4060 Ti 16 GB, against
    `--batches default`: single questions 66.8 -> 109.4 decisions/s (template prefix on), the same top answer on 58
    of 58 answers; the compact graphs took 2.4 GB more GPU memory there (13.4 -> 15.8 GB, the card full; with room
    they take about 6 GB for mini and 7 GB for the 4.5B, see below), and with the card full
    five questions over ~1.5k tokens (a path compact rows do not change) ran 5.6% slower. The same kernel on an H100,
    forced with `--attn fa2 --batches compact` (an H100 runs FlashAttention 3 otherwise; 16 clients, 2.3.1 check):
    single questions +16% for basal-2.0-mini (79.0 -> 91.7 decisions/s) and basal-2.0-4.5B (27.9 -> 32.3), five
    questions over 3k-10k tokens unchanged, GPU memory +5.8 / +7.2 GB, start-up +51 / +88 s; on dev v2 every decision
    with margin >= 0.2 kept its top answer (4.5B with `--template-prefix off`, as for every engine; with it on, 4 of
    9,064 changed). On a Blackwell RTX PRO 6000 (Server Edition, sm_120, torch 2.13, where `auto` runs no compact
    rows): single questions +18% for mini (49.0 -> 57.9) and +13% for 4.5B (17.0 -> 19.1), five questions unchanged,
    GPU memory +5.8 / +7.2 GB, start-up +31 / +112 s; dev v2 at margin >= 0.2: 7,632 / 7,632 (mini) and 9,047 /
    9,049 (4.5B, template prefix off) kept their top answer. On an RTX 3090 (Ampere, sm_86; @krzysztofkaminski, #33):
    +68% on short prompts and +20% / +18% (mini / 4.5B) on ~1.6k-token states, GPU memory +5.9 GB for mini, the
    same top answer on 58 of 58; up to 2.3.2 the 4.5B with compact rows ran out of memory at start-up on 24 GB.
    From 2.3.3 (#35, #37, @Butterski) the graphs share their output buffers and are captured largest first, so
    compact rows cost only 0.1-0.2 GB more than default rows (H100 after warm-up: mini 7.1 / 7.3 GB, 4.5B 15.2 /
    15.4 GB, every probability as in 2.3.2), and the 4.5B with compact rows starts on a 24 GB RTX 4090. From 2.3.4
    basal-omni's graphs are captured largest first too (H100 after warm-up: basal-omni-1.0-mini 27.8 -> 18.9 GB,
    1.0-max 45.3 -> 31.3 GB, every probability as in 2.3.3).
  - `--template-prefix on`: the chat template, system prompt and user header of every prompt language are computed
    once at start-up; compact rows run only their own tokens over them (off for basal-2.0 4.5B, see the quick start).
  - `--scheduler on`: requests longer than `--long-tokens` (2,048) wait in a backlog and join engine calls in
    highest-response-ratio order, at most 8,192 of their tokens per call. It trades long-request latency (about +7% p50)
    for short-request latency; `--scheduler off` serves in arrival order.
  - Answers are not bit-identical to engine v1.5.0's (other GEMM row counts); `--deterministic` stays reproducible on a
    given GPU.
- **Kernels of the CUDA graph engine** (`fast`, `fast-nocompile`, `fp8`, `nvfp4`, `fast-exit`):
  - `--attn auto|fa2|fa3|fa4|cudnn` is the attention of causal forwards. `auto` uses FlashAttention 3 on Hopper when
    its wheel is installed, else torch's built-in FlashAttention 2; FP8 checkpoints and `--mode fp8` stay on
    FlashAttention 2 unless `--attn fa3`.
  - `--merge auto|on|off` merges the q/k/v and gate/up projections into one GEMM each (`auto`: on for bf16 weights,
    off for FP8 checkpoints).
  - `--fp8-accum fast|full`: FP8 GEMM accumulation in the default mode (`--deterministic` always uses `full`).
  - Opt-in, off by default: `--compile-scope layer` (each decoder layer compiled instead of the whole model: about
    half the start-up on basal-2.0, single questions up to 2-3% slower on an RTX 4060 Ti, probabilities within bf16
    rounding; bf16 Llama models, plain rows; on an H100 basal-2.0-mini started in 47 s instead of 228 s, with the same
    top answer as a whole-model cold start on 98.69% of 840 readouts, none changed at a margin of 0.05 or more),
    `--lean on|pass2` (one compiled layer function: faster start-up, slower
    under load), `--tree flex` (a FlexAttention mask for the branches of a request), `--readout-layer on`,
    `--state-cache-gb N` (a state cache across requests; 2.3.1: also for states that begin with a cached one, e.g.
    agent traces, which then compute only their new tokens; 2.5: `--state-cache-bits 8` keeps ~1.7x as many
    states per GB, answers close to the uncached ones; `--state-cache-lock on` removes a rare HTTP 500 under
    concurrent requests at ~8% of the throughput at 8 clients, see WHATS_NEW 2.5's known limits; `--state-cache-dir
    DIR`: also every cached state as a `.safetensors` file under DIR, one subdirectory per model signature, read back
    on a memory miss and by later servers over the same directory, here and on `--mode mlx`, not for Qwen3.5;
    `--state-cache-disk-gb G` bounds it, default 20, least recently used files out first; each engine reads only the
    files it wrote), `--prefix-cache`, `--long-lane`. Measurements:
    [HARDWARE.md](HARDWARE.md#kernels-of-the-cuda-graph-engine-one-h100),
    [disk tier](HARDWARE.md#state-cache-disk-tier).
- **vLLM options (2.3):** `--vllm-preset default | deterministic` (deterministic: batch-invariant kernels, no prefix
  cache, no chunked prefill; repeatable answers at a throughput cost) loads `CALIBRATION.vllm.<preset>.json` when the
  model ships one, else `CALIBRATION.vllm.json`, else `CALIBRATION.json`, and is reported in `GET /v1/models`;
  `--vllm-fast-prefill auto | off` (auto, the default: the KV-shared layers of Gemma 4 E2B / E4B run at the answer
  position only during prefill). Measurements: [OMNI.md](OMNI.md#vllm---mode-vllm).
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096): `--max-len 32768` for long
  documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback).
- `--rank-max-candidates N` (default 10,000): the most candidates of `rank` questions in one request (HTTP 422 above);
  every 50 of them are one engine request through the queue all clients share. [NAVIGATE.md](NAVIGATE.md)
- **Status codes and load (2.3.1)**: an invalid request answers HTTP 422, the GPU out of memory 503 (retry later or
  elsewhere) and any other server error 500, all with `{"error": "..."}`. `GET /health` returns the requests in flight
  as `running` and `queued` (waiting for the engine); both 0 means idle, for a process manager that stops or restarts
  the server. Definitions: [API.md](API.md). `--error-format openai` answers errors as `{"error": {"message",
  "type", "code"}}` (same status codes).
- **LoRA adapters:** `--adapter DIR` (basal-serve, basal-calibrate, basal-bench) loads a
  LoRA adapter with the model: an mlx-lm directory (`adapter_config.json` + `adapters.safetensors`) or a PEFT one
  (`adapter_config.json` + `adapter_model.safetensors`; `use_rslora` read, DoRA and per-module ranks refused). Only
  the text decoder's layers are merged: a tensor of another tower (a vision or audio encoder's `layers.N`) refuses the
  adapter rather than land in the decoder's layer N. The
  served name is `<model>+<adapter>` unless `--name`; `GET /v1/models` reports `"adapter": "<name>@<hash>"` (sha256
  of `adapter_config.json` and the weights file, 12 hex digits: another scale is another adapter), and the start-up
  line names the adapter and the calibration file applied.

  | engine | adapter |
  |---|---|
  | `eager`, `mps`, `fast`, `fast-nocompile`, `fp8` (also block FP8 and `--quant-load cpu`), `strix`, `tpu` | merged into the bf16 weights at load, before the projection merge and any quantisation |
  | `mlx`, `mlx-q8`, the `-MLX-8bit` exports | merged at load; a quantised export is dequantised, merged and re-quantised (mlx-lm's fuse) |
  | `vllm`, `sglang` | the engine serves a copy of the checkpoint with the adapter merged, as `basal-fuse-adapter` writes it, so its answers are the fused checkpoint's (a native LoRA adds its product apart and cannot equal the merged weights). The copy is written to `$TMPDIR` at start-up and removed when basal-serve exits: it takes the size of the model's weights on disk (about 16 GB for a Gemma 4 E4B model, 62 GB for 31B), one weights file in memory while it is written, and adds the writing to the start-up (minutes for 31B). Without room it is refused before anything is written, with the space needed and free: set `TMPDIR` to a larger disk, or serve a `basal-fuse-adapter` checkpoint. `vllm` refuses `--adapter` for Qwen3.5 models (basal-omni-pico): vLLM has no deterministic preset for them and its answers move between starts of the same server, so they could not be checked against the merged checkpoint's (serve them with `fast`, `eager` or `mlx`, or a `basal-fuse-adapter` checkpoint with `vllm`) |
  | `gguf`, `ollama`, `llamacpp`, FP8 checkpoints (`-FP8`) | refused: `basal-fuse-adapter --model M --adapter DIR --out FUSED` writes the merged checkpoint, then `basal-export-gguf`; the adapter's calibration files are copied with `fitted_on` naming the merged model (`fused_from` keeps the origin), files fitted with another adapter are left out |
  | `nvfp4`, `int8` | refused: these modes serve only with a calibration (and NVFP4 activation scales) measured on the model itself; fuse the adapter and calibrate the fused checkpoint for the mode |
  | `fast-exit` | refused: its exit heads and their thresholds were trained on the base model's hidden states (serve with `fast`) |

  `basal-fuse-adapter` rewrites the model's safetensors files with the decoder layers' weights merged as `--adapter`
  merges them at load (the same tensor names, dtypes and shards) and copies the other files, except the base model's
  `CALIBRATION` files, NVFP4 activation scales, evidence head and checksums, which do not hold for the merged
  weights. It reads one weights file at a time.

  The merge is PEFT's `merge_and_unload` on every framework: `(B @ A) · scale` in float32, added to the weight in
  float32 and rounded once to the weight's dtype. The update is computed once on the CPU for every engine (the graph
  engines merge before moving the model to the GPU), so `--adapter` and the `basal-fuse-adapter` checkpoint have the
  same weights bit for bit whatever the device, and on the CPU the same weights as PEFT's merge. (mlx-lm's `fuse`
  rounds the update to bf16 before adding it, which leaves about 0.4% of a merged weight's elements one unit in the
  last place apart; on basal-omni models that moved near-tie answers.) Measured with 253 requests (561 questions) sent one at a time:
  `--adapter` answered bit for bit as the `basal-fuse-adapter` checkpoint and as PEFT's merge with rank-128 rsLoRA
  adapters of Gemma 4 models, E4B on an H100 in `fast`, `eager` and `vllm` (`--vllm-preset deterministic`) and on an
  Apple M4 Pro in `mlx`, 31B on an H100 in `fast`, `eager` and `fp8` (also with `--quant-load cpu`), two-way models
  included, and with a `basal-finetune` adapter of basal-omni-pico in `fast`, `eager` and `mlx`. Send requests one at a
  time (`basal-run --concurrency 1`) to compare servers: concurrent requests are batched differently from run to run,
  which moves bf16 probabilities by ~1e-3 on any model. A calibration file in the adapter directory comes before the
  model's; the model's own file is then loaded for its temperatures only
  ([CALIBRATION.md](CALIBRATION.md#which-model-a-file-belongs-to)). The evidence head (`"evidence": true`) was trained
  on the base model's states too: under `--adapter` an evidence request answers HTTP 422.

## Several GPUs, metrics and health (2.5)

- **`basal-router`** puts several basal-serve replicas, one per GPU, behind one endpoint:

  ```bash
  basal-router --gpus 0,1,2,3 --port 8000 -- --model Remek/basal-2.0-mini --mode fast   # one replica per GPU
  basal-router --replicas http://10.0.0.5:8000 http://10.0.0.6:8000 --port 8000          # replicas you run
  ```

  - **Routing:** a request goes to the replica its state is hashed to, so the questions over one document keep
    sharing the state and its cache on one replica. The hash is consistent, so adding or removing a replica moves
    few states.
  - **Bounded load:** a replica holding more than 1.25 times the mean in-flight requests passes new ones on along
    the ring (`--load-factor`).
  - **Requests without a state** go to the less loaded of two random replicas.
  - **Readiness:** a replica gets requests only while its `/health` answers 200.
  - **API keys:** a client's key header goes on to the replicas, which check it with `--api-keys`
    ([API access](#api-access)).
  - **Replicas it starts** (`--gpus`) end with the router: each runs in its own process group, which the router
    ends when it stops (SIGTERM, SIGINT, SIGHUP, an error; SIGKILL after 30 s); on Linux a replica also gets
    SIGTERM when the router is killed. When every replica has exited, the router exits with status 1. They listen on
    127.0.0.1 only (the router reaches them there), unless the arguments after `--` give `--host`.
  - **One live page for all replicas** (`--ui on`, off by default): the router and every replica behind it on one page
    at the router's `/ui` ([Live web page](#live-web-page-25)).
- **`GET /metrics`** (basal-serve and basal-router) is Prometheus text: engine version and model, requests waiting and
  running, requests served and failed, the state cache; per replica on the router ([Prometheus
  metrics](#prometheus-metrics)).
- **Start-up:** while the model loads and warms up, the port answers every request with HTTP 503
  `{"status": "starting"}`. A start-up that fails exits with code 1, so a supervisor or the router never sees a
  half-started server.


**Cascade and routing by script** (off by default; with neither flag the replicas' answers pass through untouched):

```
basal-router --replicas http://127.0.0.1:8001 --escalate http://127.0.0.1:8101 --escalate-below 0.8
basal-router --replicas http://127.0.0.1:8001 --route cyrillic=http://127.0.0.1:8201
```

- `--escalate URL ... --escalate-below P`: a small model answers first. Each question whose top probability (calibrated,
  as that replica returns it) is under P is asked again of the escalation replicas, usually a bigger model, and its
  answer replaces the first one. Every answer says `"answered_by": "first"` or `"escalated"`, the response lists the
  escalated questions under `"cascade"`, and `/metrics` counts both (`basal_router_answers_total`). Questions without
  option probabilities (multi, tasks) are never escalated.
- **Several levels:** repeat the pair, one `--escalate-below` per `--escalate`, in order (without any, 0.8 for each).
  The first replica is level 1 and the k-th `--escalate` is level k + 1. A question goes on from level k to level
  k + 1 only while its top probability at level k is under the k-th `--escalate-below`; the last level's answer stands.
  For example basal-omni-pico first, basal-omni-1.0-mini for what pico is unsure of, basal-omni-1.0-ultra for what
  mini is still unsure of:

  ```
  basal-serve --model Remek/basal-omni-pico --port 8001 &
  basal-serve --model Remek/basal-omni-1.0-mini --port 8002 &
  basal-serve --model Remek/basal-omni-1.0-ultra --port 8003 &
  basal-router --replicas http://127.0.0.1:8001 \
      --escalate http://127.0.0.1:8002 --escalate-below 0.8 \
      --escalate http://127.0.0.1:8003 --escalate-below 0.9 --port 8000
  ```

  With two or more levels every answer also gives its `"level"` (1, 2, 3) and `"model"` (the name the answering
  replica reports), `"cascade"` lists the thresholds (`"below": [0.8, 0.9]`) and the questions answered above level 1,
  and `/metrics` counts answers per level (`basal_router_level_answers_total{level="2"}`). A level that is down or does
  not answer 200 (for example 503 while it loads) leaves the previous answers in place; the request does not fail.
  [docs/RESULTS.md](RESULTS.md#cascade-pico-mini-ultra) shows accuracy and the share each level answers at
  thresholds 0.6 to 0.9.
- `--blend URL ... --blend-weights W0 W1 ...` (opt-in, not with `--escalate`): every question is also asked of each
  `--blend` pool (one model per flag, repeat it), at the same time, and a plain choice, noul or score answer gets the
  weighted mean of the models' probabilities, `W0` for `--replicas` and then one weight per `--blend` in order:
  p = sum(w x p_model) / sum(w). The choice, noul, score and confidence follow from it, and the answer says
  `"blend": {"models": [...], "weights": [...]}` (the names the replicas report) and `"calibrated": false`. A model
  that is down or does not answer 200 is left out and the others' weights are used; the request does not fail.
  Multi, act, tool and task answers, and questions with `coverage`, `max_error` or `evidence`, keep the first
  replica's answer (its certificates hold for that one model). `/metrics` counts blended answers by the number of
  models in them (`basal_router_blend_answers_total`).
  - **A quality option, not a speed option:** every question is asked of every model, so a blend is slower than its
    largest model alone.
  - **Put your strongest model first** (`--replicas`): questions that are not blended get its answer, and its own
    certificates still hold for them.
  - **Not calibrated:** no model's certified thresholds hold for the blend, so a blended answer carries no `abstain`,
    `max_error`, `answer_set` or `calibration` field. The blend's calibration error (ECE) was 0.041 against 0.016 for
    ultra alone on our held-out confirm split, so refit thresholds with `basal-calibrate` on the blend's output before
    you automate blended answers.
  - The weights are yours to choose. In the example, 0.6 / 0.2 / 0.2 were chosen on one half of our held-out tests
    and measured on the other ([docs/RESULTS.md](RESULTS.md#cascade-pico-mini-ultra)).

  ```
  basal-router --replicas http://127.0.0.1:8003 --blend http://127.0.0.1:8002 --blend http://127.0.0.1:8001 \
      --blend-weights 0.6 0.2 0.2 --port 8000          # ultra first, then mini and pico
  ```
- `--fallback URL --fallback-model NAME --fallback-below T` (opt-in): after the last level (or the single model), a
  plain choice question whose top probability is still under T goes to an LLM behind an OpenAI-compatible chat
  endpoint (`URL` is its base, such as `http://127.0.0.1:8100/v1` for vLLM). The router writes the prompt from the
  request's document, question and options (lettered A, B, ...) and asks for JSON that only admits those letters
  (`response_format` with a JSON schema; greedy: temperature 0 and a fixed seed; with logprobs; up to
  `--fallback-max-tokens`, default 8,192, so a model can think first). The answer replaces the basal one and says
  `"answered_by": "fallback"`, the `"model"`, `"calibrated": false`, the option probabilities from the endpoint's
  logprobs renormalised over the options (`null` when it returns none) and the basal answer it replaced
  (`"basal": {choice, confidence, model}`); the response lists `"fallback": {below, asked, answered}`.
  - The basal answer stays when the endpoint is down, answers an error, takes longer than `--fallback-timeout`
    (default 60 s), or returns anything but exactly `{"answer": "<one of the letters>"}`; the request does not fail.
    A connection the endpoint closed just as it was reused (its keep-alive) is tried once more. `/metrics` counts the
    calls by result (`basal_router_fallback_total`).
  - **Thinking takes time.** With a 31B model thinking first, on one H100 with the repeatable settings below, one call
    took 54 s at the median and 308 s at p95 (one request at a time). The default `--fallback-timeout 60` suits live
    serving, but it cuts many such calls: in that measurement 46 of 100 calls took longer and kept the basal answer.
    For offline or review use, where waiting is fine, set `--fallback-timeout 600`.
  - Never sent: multi, act, tool and task questions, noul and score, questions with `coverage`, `max_error` or
    `evidence`, and requests with media.
  - `--fallback-extra JSON` is merged into the chat request, for example a model's thinking switch:
    `'{"chat_template_kwargs": {"enable_thinking": true}}'`. The endpoint's context must hold the prompt plus
    `--fallback-max-tokens`. `BASAL_FALLBACK_API_KEY`, if set, is sent as a bearer token.
  - The LLM's answers are not calibrated: no basal threshold or certificate holds for them.
  - **Repeatable answers need a repeatable endpoint.** Greedy decoding alone is not enough: with vLLM's default
    batching, a long thinking answer can come out differently when other requests share its batch. Serve the model
    with batch-invariant kernels, as basal-serve's `--vllm-preset deterministic` does: the environment variable
    `VLLM_BATCH_INVARIANT=1`, `--no-enable-prefix-caching`, `--no-enable-chunked-prefill` and
    `--attention-backend TRITON_ATTN`.

  ```
  VLLM_BATCH_INVARIANT=1 vllm serve google/gemma-4-31B-it --port 8100 --reasoning-parser gemma4 \
      --max-model-len 16384 --no-enable-prefix-caching --no-enable-chunked-prefill \
      --attention-backend TRITON_ATTN                                          # for example
  basal-router --replicas http://127.0.0.1:8001 --fallback http://127.0.0.1:8100/v1 \
      --fallback-model google/gemma-4-31B-it --fallback-below 0.8 \
      --fallback-extra '{"chat_template_kwargs": {"enable_thinking": true}}' --fallback-timeout 300 --port 8000
  ```
- `--route SCRIPT=URL[,URL]`: requests whose state is mostly in that script (latin, cyrillic, greek, arabic, hebrew,
  cjk, other) go to those replicas, for example a model trained for Ukrainian; the rest go to `--replicas`
  (`basal_router_script_total`).

## Attach a model that was not trained on basal

`basal-attach` (2.5) serves a chat model that was never trained on basal decisions, a Gemma 4 or Qwen model for
example, without retraining:

```bash
basal-attach Qwen/Qwen3-4B --out qwen3-4b-basal          # links the model's files, writes basal.json + CALIBRATION.json
basal-serve --model qwen3-4b-basal --mode eager
basal-calibrate labelled.jsonl --model qwen3-4b-basal --mode eager --out CALIBRATION.custom.json
basal-serve --model qwen3-4b-basal --mode eager --calibration CALIBRATION.custom.json
```

- `basal.json` gets prompt format 1 by default (options A-J with an answer instruction, which a model not trained on
  basal follows best), or 2 with `--format 2` (up to 255 options). basal-attach checks that every option code of the
  format is one token at the answer position of the model's chat template, and refuses the format otherwise.
- A tokenizer without a chat template gets a plain one (`chat_template.jinja`), or yours with `--chat-template`.
- The model starts uncalibrated: temperatures 1.0 and no certified thresholds, so `max_error` defers every automatic
  action and `act` answers say `"calibration": "unvalidated"`. `basal-calibrate` on a few hundred of your labelled
  requests fits its temperatures and certifies its thresholds ([CALIBRATION.md](CALIBRATION.md)).
- Expect lower accuracy than a basal model of the same size: the model reads the options zero-shot.

## Prometheus metrics

`GET /metrics` (basal-serve and basal-router) answers in the Prometheus text format on the serving port, with no
extra dependency. A scrape configuration for Prometheus (and Grafana on top of it), for example:

```yaml
scrape_configs:
  - job_name: basal
    scrape_interval: 30s
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

basal-serve exports:

- `basal_build_info{version,model,mode,layout}` (always 1);
- `basal_requests_waiting` and `basal_requests_running` (the `queued` and `running` of `/health`), also as
  `vllm:num_requests_waiting` / `vllm:num_requests_running`, which routers such as llm-d read;
- `basal_requests_total`, `basal_requests_failed_total` and `basal_request_seconds_total` (requests to
  `/v1/systemone` and `/v1/basal`);
- with `--state-cache-gb` or `--prefix-cache on`: `basal_state_cache_bytes`, `_budget_bytes`, `_entries`,
  `_hits_total`, `_misses_total`, `_prefix_hits_total` and `_evictions_total` (all `basal_state_cache_...`). They
  count cache lookups, not client requests.

The values are read when Prometheus scrapes: nothing is added to the request path, and nothing waits for the GPU. The
state cache is read without waiting for its lock; while the engine holds it, the last values are reported again, and
`basal_state_cache_snapshot_timestamp_seconds` gives their Unix time (0 before the first read; the cache values are
left out until then). Counters start at 0 when the server starts. The endpoint is open to whoever can reach the HTTP API
(with `--api-keys`, it needs a key like every route but `GET /health`). basal-router's own metrics are per replica:
readiness, requests in flight, requests routed by state and by load, failures; with `--escalate`, answers by the first
replica and escalated (with several levels also per level); with `--blend`, blended answers by the number of models;
with `--fallback`, LLM calls by result; with `--route`, requests by script.

## Live web page (2.5)

`basal-serve --ui on` (off by default) also serves a live page of the server at `/ui`, opened in a browser:

- **status**: the served model and its revision, mode, engine version, start and uptime, requests running and queued;
- **traffic since start**: requests and failures, requests per second (last 10 s, last minute, since start), latency
  p50 / p95 / p99, per endpoint, status codes, and charts of throughput and latency that the page draws while it is open;
- **state cache** (with `--state-cache-gb` or `--prefix-cache on`): entries, hits, misses, prefix hits, evictions and
  memory;
- **telemetry**: per GPU utilisation, memory (and this server's part), temperature, power and SM clock (NVML, from
  `nvidia-ml-py`); host CPU and memory, this server's RSS and CPU (`psutil`). Neither package is required: without one,
  its panel says so;
- **configuration**: every option as parsed, the changed ones marked, secrets redacted (`--api-keys`, options named
  key / token / secret / password, user and password in URLs), and what they resolved to at start-up: the calibration
  file, prepared start-up, compile, attention, prompt format and layout, context, and the versions of Python, torch,
  transformers, vLLM, MLX and the other serving packages;
- **recent requests**: the last 200 requests' time, endpoint, status, latency and number of questions and options, never
  their text.

```bash
basal-serve --model Remek/basal-omni-1.0-mini --mode fast --ui on   # then open http://127.0.0.1:8000/ui
```

`--ui full` also shows the requests themselves: a click on a row of the recent requests opens the request as it was
sent and its answer (or the error). The server keeps references to the last 200 requests' bodies (images included) and
answers in memory, with no copy and nothing on disk, and serialises one only when the page asks for it
(`GET /ui/request/<id>`). That text is visible to whoever can open the page: on a shared host use `--api-keys`, with
which a kept request opens only for the key that sent it or an admin key (`"admin": true`; another client's key gets
403). `--ui on` never keeps a request's text.

**Open one page at a time.** What the page costs a request: one append of its metadata to a ring in memory (0.3 µs on an
Apple M4 Pro's CPU, 0.7 µs on the H100 host's). The rest (totals, percentiles, rates, the configuration view) is computed when a page reads
`GET /ui/state`, at most once a second however many pages are open, and the telemetry is read once a second in a side
thread only while a page has polled in the last 10 s. Measured on one H100 (basal-omni-1.0-mini, `--mode fast`, 153
requests, 3 fresh servers per case, a page being one reader of `GET /ui/state` per second): at 1 client, with no page,
1 or 2 pages open and with `--ui full` and 1 page, requests/s and p50 latency stayed within 0.3% and 0.9% of `--ui off`
(29.4 requests/s, p50 28.0 ms), and every answer was byte-identical to `--ui off`'s. At 8 clients in steady state (the 3
passes after a server's first): `--ui off` 40.7 requests/s, p50 172 ms; no page 41.0 / 172 ms; 1 page 40.8 / 176 ms
(+2.2%); 2 pages 41.1 / 172 ms (+0.4%); `--ui full` with 1 page 41.1 / 181 ms (+5.2%), while `--ui off`'s own p50 ranged
from 165 to 181 ms between passes. A fresh server's first pass at 8 clients is slower whatever the setting (31 to 36
requests/s). At 8 clients the answers depend on how concurrent requests are batched, with the page or without it: 95 to
103 of the 153 responses (medians) were byte-identical to another `--ui off` server's, against 93 between two
`--ui off` servers, with at most one top-1 change per pass.

The numbers start at zero with every start and live only in the server's memory (the tables keep the last 16,384 requests); "Reset view" on the page counts from that
moment in that page only. The page and its fonts ship with basal and load nothing from the network.

With `--api-keys`, `/ui` itself is open (it holds no data; it asks for a key, which stays in that browser tab) and
`GET /ui/state` needs a key, like the API. Without `--api-keys` the page is open to whoever reaches the port, as the API
is: bind `--host 127.0.0.1` or put a gateway in front. `GET /ui/state` is one JSON snapshot that a script can read too;
its fields may change between versions.

### One page for basal-router and its replicas

`basal-router --ui on` (off by default) serves one page for the router and every replica behind it at the router's
`/ui`, in the same look:

- **the router**: replicas ready, requests in flight, traffic through the router since start (requests, failures,
  requests per second, latency p50 / p95 / p99, per endpoint, status codes), the configuration as parsed (secrets
  redacted) and its last 200 requests' time, endpoint, status, latency and the replica that answered, never their text;
- **routing**: per replica the requests routed by state and by load, its share, failures and latency through the
  router; with `--escalate`, `--blend`, `--fallback` or `--route`, their counters;
- **each replica**: a card with its model, mode, version, uptime, requests running and queued, throughput, latency,
  state cache and GPUs (utilisation, memory, power, temperature), read from that replica's own `GET /ui/state`; charts of
  throughput and GPU utilisation per replica and of latency through the router, drawn while the page is open.

Start the replicas with `--ui on` too (with `--gpus`, after `--`). A replica without it, one that is not ready, or one
that wants a key the page has not given shows what the router knows of it, and says so.

```bash
basal-router --gpus 0,1 --port 8000 --ui on -- --model Remek/basal-omni-1.0-mini --mode fast --ui on
# then open http://127.0.0.1:8000/ui
```

A request through the router pays one append of its metadata to a ring in memory. The replicas' states are read only
when a page reads the router's `GET /ui/state`, at most once a second however many pages are open (one read per
replica, with the page's API key passed on as for requests), so with no page open nothing is read. Measured on 2 H100
of one node (basal-omni-1.0-mini, `--mode fast`, one replica per GPU, 153 requests, 3 fresh starts per case), router
and replicas with `--ui on` and one page open, against both without `--ui`: at 1 client every answer through the router
was byte-identical (153 of 153), at 23.0 requests/s and p50 29.9 ms either way; at 8 clients (the 3 passes after a
start's first) p50 90.6 against 89.4 ms (+1.3%) and 75.5 against 75.2 requests/s, while `--ui off`'s own p50 ranged
from 84.5 to 95.6 ms between passes. At 8 clients the answers depend on how concurrent requests are batched, with the
page or without it: 98 of the 153 responses (median) were byte-identical to a `--ui off` start's, against 97 between
two `--ui off` starts, with at most 2 top-1 changes per pass in both.

Like the router's `/metrics`, its page and its own numbers are open to whoever reaches its port (it listens on
127.0.0.1 unless `--host` says otherwise). With replicas started with `--api-keys`, the page asks for a key for their
parts: it stays in that browser tab and goes to the router, which passes it on to the replicas.

## API access

`basal-serve` is open by default. To require API keys, pass `--api-keys PATH` with a JSON file containing SHA-256
digests (never plaintext secrets):

```json
{"keys": [{"name": "laptop", "sha256": "<64-character lowercase SHA-256 hex digest>", "admin": false}]}
```

Send the secret as `Authorization: Bearer <secret>` or `X-API-Key: <secret>`. `GET /health` remains open.
`GET /v1/keys` requires an admin key and reports each key's in-memory `last_used` time. File changes take effect
without restarting the server. MCP at `/mcp` uses the same authentication.

`--cors-origins` accepts a comma-separated list of browser origins, for example
`--cors-origins https://app.example,https://admin.example`. CORS is disabled by default; use `*` to allow all origins.
Allowed preflight methods are GET, POST and OPTIONS.
Preflight requests may include `X-Request-Id`; the response echoes it as `x-request-id`.

```bash
basal-serve --model Remek/basal-2.0-mini --api-keys ./api-keys.json --cors-origins https://app.example
```

Without `--api-keys` there is no authentication and `GET /v1/keys` is not served: bind `--host 127.0.0.1` or put a
gateway in front.

basal-router passes a client's `Authorization` / `X-API-Key` header on, as it came, to every replica the request
reaches (`--replicas`, `--escalate`, `--blend`, `--route`; never to the `--fallback` LLM, which gets
`BASAL_FALLBACK_API_KEY` if set). Give every replica the same `--api-keys` file (with `--gpus`, after `--`). The router
checks no key itself: its `GET /health` and `/metrics` stay open (replica URLs and counters, no answers), and the key
goes to the replicas over plain HTTP unless their URLs are HTTPS, so keep them on loopback or a private network.
The router also passes the request's `X-Request-Id` (the client's, or a `req_<hex>` it makes) to every replica the
request reaches and answers with it as `x-request-id`, so one request has one id in the client's and the replicas'
logs.

## Inference on a JSONL file

`basal-run` sends every line of a JSONL file to a running server (concurrently; the server batches the requests) and
writes one answer per line, in the same order. Start a server first (`basal-serve ...`), then:

```bash
EX=$(python -c "import basal, pathlib; print(pathlib.Path(basal.__file__).parent / 'examples')")   # installed examples
basal-run --input $EX/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
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
also get `prediction` (index of the chosen option), `option`, `confidence` and, when `gold` is given, `correct`. A line
that is not valid (not JSON, a simple item without options) or whose request fails gets `error`, and the run goes on.
Answers are written as they arrive, in input order, to `<output>.part`, which becomes `<output>` at the end (an
interrupted run keeps its answers so far there). An existing `--output` file is replaced when the run ends (an
interrupted run leaves it as it was); `--output` cannot be the `--input` file. At the end `basal-run` prints a summary
(items per second, median latency and accuracy if gold labels are present).
Ready-to-run examples (Polish and English) in `basal/examples/` of the repository, also installed with the package
(`$EX` above):

| file | what it shows |
|---|---|
| `choice.jsonl` | routing, document type, amounts, policy rules, sentiment, next step — one option out of 3–4 (with `gold`) |
| `noul.jsonl` | yes/no decisions: deadlines, approval thresholds, phishing, refunds, missing information, alerts (with `gold`) |
| `score.jsonl` | ordered scales: urgency, satisfaction, fraud risk, answer correctness; the answer includes the expected level |
| `complex.jsonl` | full requests: several typed questions about one JSON state (fan-out), a web-agent step with structured options, a deadline decomposed into simple questions, loan triage, a prompt-injection guard |
| `questions.jsonl` | 20 mixed items; with `choice`, `noul` and `score` the default set of `basal-bench` and `basal-loadtest` |
| `types_demo.py` | one request per decision type, sent to a running server, request and response printed |

```bash
for f in choice noul score complex; do basal-run --input $EX/$f.jsonl --output answers_$f.jsonl; done
```

## Agent frameworks and ready question sets (2.5)

`basal.adapters` wraps a running basal server's HTTP API (`/v1/systemone`) as a tool for LangChain, LlamaIndex or
CrewAI. Each function returns the framework's own tool object, named `basal_decide`:

```python
from basal.adapters import langchain_tool, llamaindex_tool, crewai_tool

tool = langchain_tool("http://127.0.0.1:8000")                    # the agent passes state and questions
triage = crewai_tool("http://127.0.0.1:8000", preset="triage")    # the agent passes the state only
review = llamaindex_tool("http://127.0.0.1:8000", preset="compliance", lang="en")
```

- `langchain_tool` returns a LangChain `StructuredTool`, `llamaindex_tool` a LlamaIndex `FunctionTool` and
  `crewai_tool` a CrewAI `BaseTool` (whose result is the answers as JSON text).
- Without a preset the tool takes `state` and `questions`: the HTTP API's questions object ([API.md](API.md)); JSON
  text is accepted too. With `preset=` the questions are fixed and the agent passes only the state; `lang="pl"` or
  `"en"` picks the wording (default: the `lang` of the `client=` given, else the language of the state).
- The frameworks are not dependencies of basal: install the one you use (`uv pip install langchain-core`,
  `llama-index-core` or `crewai`). An adapter whose framework is missing says which package to install.
- MCP hosts need no adapter: `basal-serve --mcp on` serves the same decisions at `/mcp`, and `basal-mcp` is a stdio
  bridge for hosts that start local processes ([MCP.md](MCP.md)).

**Presets** are ready question sets in Polish and English (`basal/presets.py`):

| preset | questions |
|---|---|
| `triage` | `urgency` (score: low / medium / high), `category` (choice: complaint, billing, technical, account, other), `needs_human` (yes / no) |
| `moderation` | `harmful`, `spam`, `personal_data` (yes / no each) |
| `compliance` | `personal_data`, `deadline`, `legal_review` (yes / no each) |

```bash
basal-run --list-presets
basal-run --preset triage --input tickets.jsonl --output out.jsonl     # input lines: {"state": "..."}; --lang pl|en
```

In Python: `Basal(url).preset(state, "triage")` (`from basal.client import Basal`). The wording is in the `lang`
given, else in the client's (`Basal(url, lang="pl")`), else in the language of the state: Polish when it has more
signs of Polish (diacritics, common words such as `nie`, `jest`, `czy`, also typed without diacritics) than common
English words. A preset holds plain HTTP-API questions, so it is a starting point: copy one and edit it for your own
rules.

### Guardrails and routing (2.5)

| preset | questions |
|---|---|
| `guard_input` | about what a user sent: `jailbreak` (rule break, prompt injection), `harmful`, `medical` (asks for medical advice), `self_harm` (yes / no each), `severity` (score 0-3: none, low, moderate, severe) |
| `guard_output` | about what the assistant is about to say: `policy_break`, `harmful`, `medical`, `self_harm` (yes / no each), `severity` (0-3) |
| `route_intent` | `intent` (choice: question, action, complaint, sales, chitchat, other) and `complexity` (score: simple, moderate, complex) in one request |

The questions are plain HTTP-API questions. Two helpers turn the answers into an action:

```python
b = Basal("http://127.0.0.1:8000")
r = b.guard(user_message, kind="input", policy="strict")          # kind="output" for a reply
r["action"]      # "pass" | "review" | "block" | "support"
r["reason"]      # the check that set it, e.g. "self_harm"; r["checks"] has every check; r["answers"] the raw answers
b.route(ticket)  # {"route": "billing" | ... | "human", "complexity": "moderate", "level": 1, ...}
```

**Policies.** `strict` and `permissive` (`presets.POLICIES`) map each hazard to an action: `jailbreak`, `policy_break`
and `harmful` block, `self_harm` is `support` (show help, do not just refuse), `medical` is a review under `strict` and
allowed under `permissive`. `severity` blocks from level 2 (`strict`) or 3 (`permissive`), and sends level 1 or 2 to
review. The precedence is fixed: **support > block > review > pass**, over all checks. Every question of the battery
must be answered: a missing or null answer, and a malformed one (a yes/no probability that is not a finite number in
[0, 1], a severity without its level probabilities, with `max_error` an answer without a true / false `abstain`), is a
review, never a pass. The battery is the built-in one of the answers (`guard_output` when they hold `policy_break`,
else `guard_input`); a battery of your own, or an intentional part of one, is named with
`guard(answers, policy, expect=["harmful", "spam"])` (TypeScript: the fourth argument). A policy is a plain dict
(`act_at`, `review_at`, `severity`, `hazards`): pass your own to `guard(answers, policy)`.

**Where the thresholds come from.** Two modes:

- *Hand-picked* (default): a hazard acts at probability `act_at` (0.5 `strict`, 0.85 `permissive`) and goes to review from
  `review_at` (0.15 and 0.5). These are starting values; they are not certified.
- *Certified*: `b.guard(state, max_error=0.05)` (and `b.route(state, max_error=0.05)`) asks every question with
  `max_error`, so the server marks an answer `abstain` when its confidence is below the threshold the model's
  calibration certifies for that error target and question type ([CALIBRATION.md](CALIBRATION.md)). An abstaining answer
  is a review (route: `human`); a confident answer acts as the policy says (a hazard is hit at probability 0.5 or more).
  Without a calibration for your model, or a target it cannot certify, every answer abstains, and everything goes to
  review: use a target the calibration file certifies (its `thresholds` keys, [CALIBRATION.md](CALIBRATION.md)), or calibrate on your own data first. The certificate
  covers each question's own confidence on data like the calibration's, not the whole battery or your traffic.

`route_intent` without `max_error` sends the request to `human` when the intent's confidence is below
`min_confidence` (default 0.6). The default intents are examples: copy the preset and put your own routes in `intent`.

TypeScript: `await basal.guard(text, {kind: "input", policy: "strict", maxError: 0.05})` and `await basal.route(text)`
(`clients/typescript`; the pure functions `guard(answers, policy, maxError)` and `route(answers)` are exported too).

## See how a decision is made

`basal-viz` (contributed by [@lukaspoloki](https://github.com/lukaspoloki), [#2](https://github.com/rkinas/basal/pull/2))
opens a local web page that runs the model and shows every step of a decision: the prompt of each option order the
server asks (prompt format and option codes from the model's `basal.json`; one order by default for basal-2.0), the
shared-prefix packing when there are two orders, the code probabilities after every layer ("logit lens"), the code
readout, the calibration and the decision against the thresholds the calibration file certifies. The same request is
answered by the server in the same process, and the page shows the difference (0 for a single question; a request with
several questions shares its state in one packed row, which moves bf16 probabilities by a few thousandths). The bundled
examples are in a drop-down, with the expected answer where the file has one.

```bash
basal-viz --model Remek/basal-2.0-mini --port 8080     # then open http://127.0.0.1:8080
basal-viz --model Remek/basal-2.0-mini-MLX-8bit --mode mlx     # Apple Silicon: the MLX engine
```

By default it runs the PyTorch engine, `--mode mps` on Apple Silicon and `--mode eager` elsewhere (`--device`,
`--orders 1|2`, `--calibration` as for `basal-serve`); `--mode mlx` / `mlx-q8` (below) runs the MLX engine instead. The
PyTorch modes read the Llama-architecture models (basal-1.0, basal-1.5, basal-2.0) and basal-omni's Gemma 4 text
decoder (bf16 checkpoints; the final logit softcap is applied as the engines apply it); the MLX modes read the
Llama-architecture models only. Requests with media are refused: basal-viz explains text only. It is meant for
exploring, not serving: every question runs one extra forward per option order to record all layers, and a prompt over
the context is refused (HTTP 422) instead of truncated.
Inspections and their served comparisons run one request at a time because they share the model. Gemma 4
inspections accept at most 2,048 tokens per rendered prompt (including the template, question and options);
longer prompts return a validation error before the lens forward, so shorten the state or question.

`--mode mlx` (the `-MLX-8bit` / `-MLX-fp4` exports or bf16 weights) and `--mode mlx-q8` read the layers of the MLX
engine that `basal-serve` runs on a Mac, so the page shows where a quantised model's decision forms or moves. The
codes are read as that engine reads them (fp32 code rows, or the quantised head itself). On the page's 49 bundled
requests (the 44 items with gold answers and 5 requests with several questions; Apple M5 Max, basal-2.0-mini and 4.5B,
`-MLX-8bit` and `mlx-q8`), the page's answer differs from the served one by at most 2e-6 for a single question; a
request with several questions differs by up to 0.02, as in `--mode mps`.

## Benchmark your GPU

Two tools are installed with the package. Both run out of the box on the bundled examples (44 Polish and English items
with gold answers in `basal/examples/`):

```bash
# offline: latency, throughput and agreement of serving modes (no HTTP); the first mode is the reference
basal-bench --model Remek/basal-2.0-4.5B --modes eager-fp32 fast --out bench.json

# end to end over HTTP against a running server (basal-serve ...)
basal-loadtest --url http://127.0.0.1:8000/v1/systemone
basal-loadtest --url http://127.0.0.1:8000/v1/systemone --synthetic 1500 --concurrency 16   # ~1.5k-token states

# identity gate: every prefix-reuse path of the fast engine against the eager per-prompt reference
basal-bench --identity --model Remek/basal-2.0-4.5B                    # fp32; --dtype bfloat16 as served

# agreement: any backends against the native engine (eager fp32, every prompt alone; its rows are cached)
basal-agree --model Remek/basal-2.0-mini --backends mps mlx mlx=Remek/basal-2.0-mini-MLX-8bit

# option order: how often the top answer changes under other option orders; gain and cost of --order-average K
basal-bench --order-check --model Remek/basal-omni-1.0-mini --questions my_items.jsonl --average 1 2 4
```

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with
concurrent clients (`--concurrency`); `--synthetic N` uses generated items with states of about N tokens. `basal-bench
--identity` runs single prompts, option orders and multi-question requests over one state, alone and batched, and
compares every readout with an eager forward of its own prompt (argmax agreement, max |Δp|). `basal-agree` (2.5) does
the same for any backends (basal-bench modes; `MODE=WEIGHTS` for other weights with the same tokenizer) against the
native engine (`--reference`, default `eager-fp32`): per backend the top-answer agreement with a 95% bootstrap interval,
mean / p99 / max |Δp|, the decisions with both orders averaged, the paired accuracy difference on `--questions` items
with gold, and question independence (each question of a five-question request asked alone through the same backend
gives the answer it gives in the request). The exit status is 0 when every backend passes `--gate` (default 0.99, top
answers outside near ties); `basal-agree --identity` runs the identity gate.

`basal-bench --order-check` (2.5) reads every question in its given, reversed and rotated option orders (one call per
question over one state) and reports the share of questions whose top answer flips (`flip_reversed`, `flip_rotated`,
`flip_any`), how often the top answer sits at the first position against chance (`first_share`), and for each K of
`--average` the accuracy of the mean over K orders (`acc_mean<K>`, with gold), its agreement with the given order and
the latency of one question with K orders at batch size 1. A model that reads the options' content flips rarely and
gains little from `--order-average`; a high flip rate with a gain in `acc_mean<K>` is the case where averaging pays for
its cost.

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers. For numbers that
describe *your* workload, write a JSONL file with a few hundred items in the same simple format (`{"state": ...,
"question": ..., "options": [...], "gold": <index>}`, `gold` optional) with realistic state lengths, and pass it with
`--questions my_items.jsonl`.
