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
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` ports or bf16 weights; shared-prefix KV cache (opt-in `--mlx-packed`: short Llama groups as one packed forward, [QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)) | Apple Silicon |
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
  class the port was gated on) when fla is installed, `eager` elsewhere. Not with them yet: `fp8` / `nvfp4`, `fast-exit`,
  `mps`, `--batches compact`, `--state-cache-gb`, `--prefix-cache`, `--template-prefix on`, `--readout-layer on`,
  `--lean` and `--long-lane` (each refused before loading). `--mode vllm` and `--mode gguf` serve Qwen3.5 models
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
- **SOAM** (`--soam on`, default): every question and option order of a request is one packed row over the state.
  `--soam off` restores the per-question path of basal-1.0 (same answers in fp32). See [SOAM.md](SOAM.md).
- **Start-up**: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (several
  minutes the first time, about half with a warm compile cache). A request with several branches (option orders,
  questions, `multi` labels) runs in two exact passes: the state once, keeping its keys and values, then all branches
  over them ([SOAM.md](SOAM.md#which-kernel-runs)).
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
- **Long states**: `--context-overflow truncate` (the default) cuts the middle of a state that does not fit the
  context (the first 1/3 and the last 2/3 of the kept tokens, `--truncate-head`; the question and options stay whole)
  and says what was cut in `usage.truncated`; `--context-overflow refuse` returns HTTP 422 instead. The context is
  32,768 tokens for basal-2.0 (4.5B) and max, 8,192 for mini (Ollama mode: 4,096; llama.cpp mode: `llama-server -c`;
  vLLM / SGLang: `--max-len`).
- **Engine v2.0.0 items** (bf16 weights in `--mode fast` on Hopper with FlashAttention 3; each item has a flag,
  `auto` = on):
  - `--batches compact`: single prompts packed back to back into one row (FA3 varlen), graphs per token count, no
    padding to bucket lengths or row counts. On other sm80+ GPUs `--batches compact` runs them through torch's varlen
    attention (Llama models; `auto` keeps them off there). basal-2.0-mini on an RTX 4060 Ti 16 GB, against
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
    rounding; bf16 Llama models, plain rows), `--lean on|pass2` (one compiled layer function: faster start-up, slower
    under load), `--tree flex` (a FlexAttention mask for the branches of a request), `--readout-layer on`,
    `--state-cache-gb N` (a state cache across requests; 2.3.1: also for states that begin with a cached one, e.g.
    agent traces, which then compute only their new tokens), `--prefix-cache`, `--long-lane`. Measurements:
    [HARDWARE.md](HARDWARE.md#kernels-of-the-cuda-graph-engine-one-h100).
- **vLLM options (2.3):** `--vllm-preset default | deterministic` (deterministic: batch-invariant kernels, no prefix
  cache, no chunked prefill; repeatable answers at a throughput cost) loads `CALIBRATION.vllm.<preset>.json` when the
  model ships one, else `CALIBRATION.vllm.json`, else `CALIBRATION.json`, and is reported in `GET /v1/models`;
  `--vllm-fast-prefill auto | off` (auto, the default: the KV-shared layers of Gemma 4 E2B / E4B run at the answer
  position only during prefill). Measurements: [OMNI.md](OMNI.md#vllm---mode-vllm).
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096): `--max-len 32768` for long
  documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback).
- **Status codes and load (2.3.1)**: an invalid request answers HTTP 422, the GPU out of memory 503 (retry later or
  elsewhere) and any other server error 500, all with `{"error": "..."}`. `GET /health` returns the requests in flight
  as `running` and `queued` (waiting for the engine); both 0 means idle, for a process manager that stops or restarts
  the server. Definitions: [API.md](API.md).

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
also get `prediction` (index of the chosen option), `option`, `confidence` and, when `gold` is given, `correct`. At the
end `basal-run` prints a summary (items per second, median latency and accuracy if gold labels are present).
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
```

It runs `--mode mps` on Apple Silicon and `--mode eager` elsewhere (`--device`, `--orders 1|2`, `--calibration` as for
`basal-serve`), and only the Llama-architecture models (basal-1.0, basal-1.5, basal-2.0; not basal-omni). It is meant
for exploring, not serving: every question runs one extra forward per option order to record all layers.

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
```

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with
concurrent clients (`--concurrency`); `--synthetic N` uses generated items with states of about N tokens. `basal-bench
--identity` runs single prompts, option orders and multi-question requests over one state, alone and batched, and
compares every readout with an eager forward of its own prompt (argmax agreement, max |Δp|).

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers. For numbers that
describe *your* workload, write a JSONL file with a few hundred items in the same simple format (`{"state": ...,
"question": ..., "options": [...], "gold": <index>}`, `gold` optional) with realistic state lengths, and pass it with
`--questions my_items.jsonl`.
