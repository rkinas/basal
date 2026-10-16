# Serving, batch inference and benchmarks

Flags of `basal-serve`, inference on a JSONL file with `basal-run`, and benchmarking a GPU.

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | hardware |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + SOAM packing + compact batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fp8` | the `-FP8` ModelOpt checkpoints natively; bf16 weights quantised to FP8 at load time (torchao) | GPUs with FP8 (sm89+): Hopper, Blackwell |
| `nvfp4` | `fast` with bf16 weights quantised to NVFP4 at load time (torchao, experimental; costs accuracy; not the basal-1.5 / 1.0 `-NVFP4` checkpoints) | Blackwell |
| `vllm` | vLLM backend, readout from the option codes' log-probabilities; bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only), and the basal-1.5 / 1.0 `-NVFP4` checkpoints | CUDA GPUs supported by vLLM |
| `sglang` | SGLang backend, same readout | CUDA GPUs supported by SGLang |
| `eager` | plain PyTorch reference; `--device cuda\|mps\|cpu` (default: `BASAL_DEVICE`, else CUDA, else Apple Silicon, else CPU) | any GPU, Apple Silicon (MPS) or CPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen per request | models that ship `exit_heads/` (basal-1.0 only) |
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` ports or bf16 weights; shared-prefix KV cache | Apple Silicon |
| `mlx-q8` | `mlx` with 8-bit weights made from bf16 at load time (about half the memory, not faster) | Apple Silicon |
| `mps` | the PyTorch engine on the Mac's GPU: SOAM packing + token-budget batching, no CUDA graphs | Apple Silicon |
| `gguf` | a GGUF file in process through llama.cpp (`--gguf <file>`, extra `basal[gguf]`) | Apple Silicon (Metal), CUDA, CPU |
| `strix` | PyTorch + Triton kernels written for gfx1151: flash attention over the packed prefix trie, gate / up + SwiGLU fused, matmuls with the bias and residual add in the epilogue, all rows of a batch in one sequence (bf16; [benchmarks](benchmarks/STRIX_HALO.md)) | AMD Strix Halo (ROCm + Triton); measured only there |
| `ollama` | the basal server in front of Ollama (GGUF ports; `--ollama-model`, `--ollama-url`) | anything Ollama runs on |
| `llamacpp` | the basal server in front of `llama-server` (GGUF ports; `--llamacpp-url`) | anything llama.cpp runs on |

- Without `--mode`: `fast` on CUDA, `mlx` on Apple Silicon (`mps` without `basal[mlx]`), `eager` otherwise.
- `--calibration file.json` overrides the calibration file (default: `CALIBRATION.<mode>.json` when the model ships
  one for that engine, else `CALIBRATION.json`; `--mode gguf` uses `CALIBRATION.json`; with `--orders 2`,
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
  (`basal-loadtest --determinism N` checks a running server, including decision flips under load).
- **Long states**: `--context-overflow truncate` (the default) cuts the middle of a state that does not fit the
  context (the first 1/3 and the last 2/3 of the kept tokens, `--truncate-head`; the question and options stay whole)
  and says what was cut in `usage.truncated`; `--context-overflow refuse` returns HTTP 422 instead. The context is
  32,768 tokens for basal-2.0 (4.5B) and max, 8,192 for mini (Ollama mode: 4,096; llama.cpp mode: `llama-server -c`;
  vLLM / SGLang: `--max-len`).
- **Engine v2.0.0 items** (bf16 weights in `--mode fast` on Hopper with FlashAttention 3; each item has a flag,
  `auto` = on):
  - `--batches compact`: single prompts packed back to back into one row (FA3 varlen), graphs per token count, no
    padding to bucket lengths or row counts.
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
  - Opt-in, off by default: `--lean on|pass2` (one compiled layer function: faster start-up, slower under load),
    `--tree flex` (a FlexAttention mask for the branches of a request), `--readout-layer on`, `--state-cache-gb N`
    (a state cache across requests), `--prefix-cache`, `--long-lane`. Measurements:
    [HARDWARE.md](HARDWARE.md#kernels-of-the-cuda-graph-engine-one-h100).
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096): `--max-len 32768` for long
  documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback).

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
