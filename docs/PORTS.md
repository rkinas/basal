# Ports: Apple Silicon (MLX) and GGUF (Ollama, llama.cpp)

basal 1.5 runs outside PyTorch in three more serving modes. Every mode uses the same prompt, the same tokens and the
same letter readout as the PyTorch modes (the softmax over the option letters' logits at the prefilled
`{"answer": "` position, both option orders averaged, per-type temperatures from `CALIBRATION.json`), so the HTTP API,
the answer types (`choice`, `noul`, `score`, `act`, `multi`), `facts`, the confidence thresholds and the propensity
log behave as in `--mode fast`. Not available in the ports: the evidence output (it needs the PyTorch hidden states).

| mode | weights | engine | how the letters are read |
|---|---|---|---|
| `mlx` | `-MLX-8bit` (affine 8-bit), `-MLX-fp4` (`nvfp4`), or the bf16 repo | MLX / mlx-lm inside the server | output head at the readout positions; the state is computed once into a KV cache, every question and option order continues it in one batch |
| `ollama` | `-GGUF` (`Q8_0`, `Q4_K_M`) | Ollama 0.12.11 or newer (measured: 0.35.0) | `/api/generate`, `raw: true`, one token, `logprobs` + `top_logprobs` (20) |
| `llamacpp` | `-GGUF` | llama.cpp `llama-server` (measured: 0.5.0, b11146) | `/completion` with the prompt as token ids, `n_probs` (20), log-probabilities of the raw logits |

## Use

```bash
# Apple Silicon, MLX
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.1.0.tar.gz"
basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx --port 8000

# Ollama: the -GGUF folder gives the tokenizer, chat template and calibration; Ollama serves the weights
hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF
cd basal-1.5-4.5B-GGUF && ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0 && cd ..
basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0 --port 8000

# llama.cpp server
llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

Options: `--http-parallel N` sends N prompts at once (start Ollama with `OLLAMA_NUM_PARALLEL=N` or `llama-server -np N`;
the default of 1 lets the engine's prompt cache reuse the state for every question and option order of a request).
`--calibration FILE` overrides the calibration file; in `--mode mlx|ollama|llamacpp` the server prefers
`CALIBRATION.<mode>.json` next to the weights when one is shipped (none is needed, see Calibration).

## Results: basal-1.5-4.5B

500 held-out Polish development decisions (2–7 options; both option orders, calibrated), Apple M4 Pro (10+4 CPU
cores, 20 GPU cores, 24 GB), macOS 15.7. Reference: the same weights in PyTorch 2.14 (bf16, `--mode eager --device mps`)
on the same machine. *Agreement*: share of decisions whose top option equals the reference's; *|Δp|*: mean absolute
difference of the calibrated option probabilities; *ECE*: expected calibration error of the top option (15 bins).
*Latency*: median of 30 requests through the server's decision path (rendering, tokenization, engine, calibration; no
HTTP in front of basal-serve) with 1 and 5 questions over one state; *tokens/s*: prompt tokens of the request (every
question in both option orders) per second. Versions: mlx 0.32.3, mlx-lm 0.32.0, Ollama 0.35.0, llama.cpp 0.5.0.
Rows are the uploaded files, all measured directly. Ollama rows: latency only (same file and tokens as llama.cpp; identical probabilities when both were
scored on an earlier 4.5B candidate).

| engine, weights | size | agreement | accuracy | mean \|Δp\| | ECE | 1 question | 5 questions | tokens/s (1 / 5) |
|---|---|---|---|---|---|---|---|---|
| PyTorch MPS, bf16 (reference) | 9.5 GB | – | 0.936 | – | 0.030 | 817 ms | 3,992 ms | 726 / 730 |
| **`mlx`, `-MLX-8bit`** (affine 8-bit) | 5.2 GB | **0.992** | 0.940 | 0.004 | 0.037 | 587 ms | 2,186 ms | 1,009 / 1,332 |
| **`mlx`, `-MLX-fp4`** (`nvfp4`) | 2.9 GB | **0.970** | 0.922 | 0.028 | 0.039 | 581 ms | 2,102 ms | 1,021 / 1,386 |
| **`llamacpp`, `-GGUF` `Q8_0`** | 5.1 GB | **0.994** | 0.938 | 0.003 | 0.031 | 427 ms | 1,670 ms | 1,387 / 1,744 |
| `ollama`, `-GGUF` `Q8_0` | 5.1 GB | (= llama.cpp) | | | | 518 ms | 1,760 ms | 1,145 / 1,669 |
| **`llamacpp`, `-GGUF` `Q4_K_M`** (importance matrix) | 2.9 GB | **0.980** | 0.936 | 0.014 | 0.030 | 446 ms | 1,732 ms | 1,330 / 1,682 |
| `ollama`, `-GGUF` `Q4_K_M` | 2.9 GB | (= llama.cpp) | | | | 542 ms | 1,841 ms | 1,093 / 1,595 |

## Results: basal-1.5-max (11B)

Same panel and machine. The 11B bf16 model (22 GB) does not fit a 24 GB Mac, so the reference is the uploaded GGUF
`Q8_0` with llama.cpp (on basal-1.5-4.5B `Q8_0` agrees with bf16 on 99.4% of decisions). Ollama rows: latency only (it
runs the same file with the same tokens as llama.cpp; identical probabilities on the 4.5B). Rows are the uploaded files.

| engine, weights | size | agreement with `Q8_0` | accuracy | mean \|Δp\| | ECE | 1 question | 5 questions | tokens/s (1 / 5) |
|---|---|---|---|---|---|---|---|---|
| **`llamacpp`, `-GGUF` `Q8_0`** (reference) | 11.9 GB | – | 0.948 | – | 0.033 | 1,156 ms | 3,667 ms | 513 / 794 |
| `ollama`, `-GGUF` `Q8_0` | 11.9 GB | (= llama.cpp) | | | | 1,100 ms | 3,674 ms | 538 / 800 |
| **`llamacpp`, `-GGUF` `Q4_K_M`** (importance matrix) | 6.8 GB | **0.994** | 0.942 | 0.006 | 0.027 | 1,218 ms | 3,848 ms | 487 / 757 |
| `ollama`, `-GGUF` `Q4_K_M` | 6.8 GB | (= llama.cpp) | | | | 1,158 ms | 3,837 ms | 512 / 766 |
| **`mlx`, `-MLX-8bit`** | 12.1 GB | **0.996** | 0.948 | 0.002 | 0.027 | 1,362 ms | 5,202 ms | 435 / 560 |
| **`mlx`, `-MLX-fp4`** (`nvfp4`) | 6.7 GB | **0.972** | 0.928 | 0.029 | 0.025 | 1,317 ms | 4,918 ms | 450 / 592 |

- The 11B model is about 2.2× slower than the 4.5B on this machine; on Apple Silicon llama.cpp / Ollama are faster than
  MLX for it, too (1.1–1.2 s against 1.3–1.4 s per decision).
- `Q4_K_M` with the importance matrix agrees with `Q8_0` on 99.4% (4.5B, with bf16: 98.0%); MLX `nvfp4` on 97.2% (4.5B:
  97.0%) and costs 2.0 accuracy points here.
- `-MLX-8bit` needs about 14 GB of free memory (weights wired plus the 2 GB cache); on a 16 GB Mac use `-MLX-fp4` or
  `Q4_K_M`.
- Calibration check against `Q8_0`: `-MLX-8bit` t = 1.00 / 1.00, `Q4_K_M` 0.98 / 1.03, `-MLX-fp4` 0.84 / 1.01 (choice /
  yes-no; fitting lowers the KL divergence by 14% for fp4 choice questions, at most 5% otherwise). The fp4 port's choice
  probabilities are slightly flatter than the reference's, but its ECE on this panel is 0.025 against 0.033 for the
  reference, so the bf16 calibration is kept.

## Results: basal-1.5-mini (1.5B)

Same panel and machine; reference: the same weights in PyTorch bf16 on MPS (3.2 GB). All three ports of this model were
exported on the Mac (same tools). Ollama rows: latency only. Rows are the uploaded files.

| engine, weights | size | agreement | accuracy | mean \|Δp\| | ECE | 1 question | 5 questions | tokens/s (1 / 5) |
|---|---|---|---|---|---|---|---|---|
| PyTorch MPS, bf16 (reference) | 3.2 GB | – | 0.918 | – | 0.037 | 279 ms | 1,345 ms | 2,121 / 2,165 |
| **`mlx`, `-MLX-8bit`** | 1.8 GB | **0.998** | 0.916 | 0.003 | 0.037 | 196 ms | 713 ms | 3,022 / 4,088 |
| `mlx`, `-MLX-fp4` (`nvfp4`) | 1.0 GB | 0.908 | 0.882 | 0.054 | 0.059 (0.046 with `CALIBRATION.mlx.json`) | 197 ms | 728 ms | 3,008 / 4,002 |
| **`llamacpp`, `-GGUF` `Q8_0`** | 1.7 GB | **0.998** | 0.918 | 0.003 | 0.036 | 160 ms | 613 ms | 3,715 / 4,753 |
| `ollama`, `-GGUF` `Q8_0` | 1.7 GB | (= llama.cpp) | | | | 186 ms | 647 ms | 3,189 / 4,541 |
| **`llamacpp`, `-GGUF` `Q4_K_M`** (importance matrix) | 1.0 GB | **0.964** | 0.914 | 0.018 | 0.032 | 157 ms | 613 ms | 3,776 / 4,753 |
| `ollama`, `-GGUF` `Q4_K_M` | 1.0 GB | (= llama.cpp) | | | | 196 ms | 641 ms | 3,026 / 4,581 |

- **The 1.5B model is more sensitive to 4-bit MLX weights.** `nvfp4` agrees with bf16 on 90.8% and loses 3.6 accuracy
  points; on an earlier mini checkpoint the other MLX 4-bit formats were worse still (affine 4-bit 0.802, `mxfp4` 0.584),
  so `nvfp4` stays the fp4 format; it only saves another 0.8 GB against `-MLX-8bit`.
- **Per-engine calibration for `-MLX-fp4` (mini only).** Its probabilities are flatter than the bf16 model's (relative
  temperature 0.82 for choice and yes/no questions; ECE 0.059 against 0.037), which counts as clearly off. The
  repository therefore ships `CALIBRATION.mlx.json` (preferred by `--mode mlx`): the bf16 temperatures × 0.94 (choice)
  and × 0.84 (yes/no), fitted gold-free against the bf16 model on the other 500 development
  items, so it is evaluated out of sample here: ECE 0.059 → 0.046. **The 1% and 5% confidence thresholds in
  `CALIBRATION.mlx.json` are copied from the bf16 model and are not certified for this port**: its temperatures were
  refitted on development items, so the bf16 model's coverage and error figures at those thresholds do not carry over
  (the file says so in `thresholds_note`). Do not automate decisions on them without refitting the thresholds on your
  own labelled requests. Every other port of the three models keeps `CALIBRATION.json`.
- **Recommendation for basal-1.5-mini:** use `-MLX-8bit` (1.8 GB, agreement 0.998) or GGUF `Q4_K_M` (1.0 GB, 0.964);
  `-MLX-fp4` costs about 3.6 accuracy points.

## FP8 and NVFP4 checkpoints for vLLM (NVIDIA GPUs)

`Remek/basal-1.5-{4.5B,max,mini}-{FP8,NVFP4}`: NVIDIA ModelOpt post-training quantisation, as for basal-1.0
(`FP8_DEFAULT_CFG` / `NVFP4_DEFAULT_CFG`; the output head and the token embeddings stay in bf16, because the decision is
read from the head's letter logits), calibrated on 512 decision prompts of the calibration split in both option orders
(no development or test items), exported as standard Hugging Face checkpoints. Served with
`basal-serve --model Remek/basal-1.5-4.5B-FP8 --mode vllm` (`basal[vllm]`). `CALIBRATION.json` keeps the bf16 model's
temperatures; its thresholds carry a note that they are not validated for the quantised checkpoint.

Measured with `basal-bench --modes eager-fp32 vllm` on an NVIDIA RTX PRO 6000 Blackwell (sm120, 96 GB; vLLM 0.30.0,
native kernels: CUTLASS FP8 and CUTLASS NVFP4 GEMM) on the 1,000 development decisions of the vLLM / SGLang engine
checks, both option orders. Reference: the bf16 release weights in fp32 PyTorch on the same GPU. *Latency*: one
decision in both option orders, batch 1, median; *dec/s*: two-order decisions per second with 32 option-order passes per
call.

| model | checkpoint | size | agreement with fp32 | accuracy (change) | latency | dec/s |
|---|---|---|---|---|---|---|
| basal-1.5 (4.5B) | fp32 reference (PyTorch eager) | – | – | 0.930 | 127.9 ms | 5.0 |
| | **`-FP8`** | 4.9 GB | **0.983** | 0.931 (+0.001) | 17.2 ms | 143.5 |
| | `-NVFP4` | 2.9 GB | 0.937 | 0.902 (−0.028) | 17.4 ms | 163.4 |
| basal-1.5-max (11B) | fp32 reference (PyTorch eager) | – | – | 0.944 | 302.1 ms | 2.3 |
| | **`-FP8`** | 11.4 GB | **0.989** | 0.941 (−0.003) | 24.5 ms | 76.4 |
| | **`-NVFP4`** | 6.7 GB | **0.979** | 0.943 (−0.001) | 22.0 ms | 120.8 |
| basal-1.5-mini (1.5B) | fp32 reference (PyTorch eager) | – | – | 0.916 | 47.7 ms | 13.7 |
| | **`-FP8`** | 1.7 GB | **0.979** | 0.903 (−0.013) | 8.8 ms | 337.2 |
| | `-NVFP4` (measured, not released) | 1.0 GB | 0.892 | 0.854 (−0.062) | 9.5 ms | 351.8 |

- **FP8 holds for every size** (agreement 0.979–0.989 with fp32, −1.3 to +0.1 accuracy points).
- **NVFP4 depends on the model size**: lossless on the 11B max (0.979, −0.1 points), −2.8 points on the 4.5B and −6.2 on
  the 1.5B mini (0.892 agreement), the same pattern as basal-1.0 (about −3 and −7 points); basal-1.5-mini therefore has
  no NVFP4 release. Use NVFP4 for max; for the 4.5B prefer FP8 unless batch throughput matters more than those points.
- On this GPU a single decision is launch-bound: FP8 and NVFP4 have the same batch-1 latency; NVFP4 gains in batched
  throughput (max: 121 against 76 decisions/s).
- **Reproducing the quantisation.** Two environments, because ModelOpt and vLLM bring different pins. Quantisation:
  `nvidia-modelopt[torch,hf]==0.47.0` with **`transformers==5.14.1`** (ModelOpt 0.47 requires transformers < 5.15; the
  basal engine itself pins 5.17.0; 5.14.1 reads the transformers-5 configs of the models, including `rope_parameters`
  with rotary base 1,000,000, checked before quantising) and torch 2.14.1 (CUDA 13.0). Serving / measuring: vLLM 0.30.0
  (its own torch 2.13.0 and transformers 5.18.0) with the basal package installed `--no-deps`. Shipped: `-FP8` for all
  three models and `-NVFP4` for the 4.5B (with its −2.8-point note) and max; mini's NVFP4 is not released.

## Format choice (measured on a development 4.5B checkpoint)

The formats were chosen on a development 4.5B checkpoint with the same architecture and tokenizer (same panel, machine
and reference; reference accuracy 0.926).

| engine, weights | size | agreement | accuracy | mean \|Δp\| | ECE | 1 question | 5 questions | tokens/s (1 / 5) |
|---|---|---|---|---|---|---|---|---|
| PyTorch MPS, bf16 (reference) | 9.5 GB | – | 0.926 | – | 0.037 | 823 ms | 4,482 ms | 720 / 650 |
| `mlx`, bf16 | 9.5 GB | 0.998 | 0.928 | 0.002 | 0.033 | 601 ms | 2,056 ms | 986 / 1,417 |
| **`mlx`, `-MLX-8bit`** (affine 8-bit) | 5.2 GB | **0.998** | 0.928 | 0.003 | 0.034 | 619 ms | 2,281 ms | 958 / 1,277 |
| **`mlx`, `-MLX-fp4`** (`nvfp4`) | 2.9 GB | **0.966** | 0.918 | 0.025 | 0.041 | 600 ms | 2,238 ms | 988 / 1,301 |
| `mlx`, `mxfp8` (not shipped) | 5.0 GB | 0.940 | 0.908 | 0.034 | 0.034 | 622 ms | 2,331 ms | 953 / 1,250 |
| `mlx`, affine 4-bit (not shipped) | 2.9 GB | 0.932 | 0.900 | 0.032 | 0.031 | 603 ms | 2,235 ms | 983 / 1,304 |
| `mlx`, `mxfp4` (not shipped) | 2.7 GB | 0.938 | 0.898 | 0.041 | 0.053 | 588 ms | 2,197 ms | 1,009 / 1,326 |
| **`llamacpp`, GGUF `Q8_0`** | 5.1 GB | **0.990** | 0.924 | 0.003 | 0.035 | 442 ms | 1,816 ms | 1,342 / 1,604 |
| **`ollama`, GGUF `Q8_0`** | 5.1 GB | **0.990** | 0.924 | 0.003 | 0.035 | 490 ms | 1,843 ms | 1,209 / 1,581 |
| **`llamacpp`, GGUF `Q4_K_M`** (importance matrix, as shipped) | 2.9 GB | **0.978** | 0.920 | 0.015 | 0.036 | 485 ms | 1,909 ms | 1,223 / 1,526 |
| `llamacpp`, GGUF `Q4_K_M`, importance matrix of 200 chunks on bf16 | 2.9 GB | 0.968 | 0.922 | 0.016 | 0.038 | 470 ms | 1,871 ms | 1,261 / 1,557 |
| `llamacpp`, GGUF `Q4_K_M` without importance matrix | 2.9 GB | 0.920 | 0.896 | 0.050 | 0.058 | 476 ms | 1,832 ms | 1,246 / 1,590 |
| `ollama`, GGUF `Q4_K_M` without importance matrix | 2.9 GB | 0.920 | 0.896 | 0.050 | 0.058 | 494 ms | 1,909 ms | 1,200 / 1,526 |

- **8-bit is lossless, 4-bit costs about one point.** MLX bf16, `-MLX-8bit` and GGUF `Q8_0` agree with the reference on
  99–100% of the decisions (the remaining flips are near-ties, |Δp| 0.002–0.003). The best 4-bit formats, MLX `nvfp4`
  and `Q4_K_M` with an importance matrix, agree on 96.6–97.8% and lose 0.6–0.8 accuracy points (basal-1.5-4.5B:
  0.970–0.980, 0 to 1.4 points).
- **MLX `mxfp8` is not an 8-bit option**: its shared power-of-two scale per 32 weights leaves three mantissa bits, and
  it changes as many decisions as the 4-bit formats (0.940). The 8-bit release is affine (bf16 scale and bias per 64
  weights); the 4-bit release is `nvfp4` (FP8 scale per 16 weights).
- **The importance matrix matters for `Q4_K_M`**: 0.920 → 0.978 agreement, 0.896 → 0.920 accuracy. It is computed with
  `llama-imatrix` on the `Q8_0` file from 64 × 1,024 tokens of rendered decision prompts (500 development items
  disjoint from the measured ones, both option orders) and is not shipped; a
  larger matrix (200 chunks on the bf16 file) is not better (0.968).
- **Ollama = llama.cpp.** After the tokenizer fix (below) Ollama's text tokenization equals the engine's tokens, and the
  two give identical probabilities; Ollama adds 15–50 ms per request.
- **Speed.** On the M4 Pro a decision is prefill (compute) bound: every MLX format takes about the same time (0.6 s for
  one question, about 2.2 s for five), so quantisation saves memory, not time; llama.cpp / Ollama are 20–25% faster
  than MLX. Five questions over one state cost 3.5–4× one question, not 5×, because the state is computed once. For
  comparison, a data-centre GPU with `--mode fast` takes 12.5 ms per decision (H100, basal-1.0-4.5B): the ports are for
  local use, development and small volumes.

## Calibration

The ports keep the calibration of the bf16 model. Check (gold-free): for each question type, the extra temperature t
that brings a port's calibrated probabilities closest to the reference's (minimum mean KL divergence; t = 1: the port is
neither over- nor under-confident against its source). Measured t on the uploaded basal-1.5-4.5B files (choice / yes-no):
`-MLX-8bit` 1.005 / 1.000, `Q8_0` 1.005 / 1.004, `-MLX-fp4` 0.939 / 0.931, `Q4_K_M` 1.030 / 1.048 (development
checkpoint: 0.95–1.06); fitting t lowers the KL divergence by at most 4% (6% on the development checkpoint), so the remaining differences are per-decision noise that
a temperature cannot remove. No `CALIBRATION.<mode>.json` is shipped except for basal-1.5-mini's `-MLX-fp4` (see its section);
In such a file the certified confidence thresholds are copied from the bf16 model and should be re-checked.

## How the ports are made

- **MLX.** The decoder layers are quantised; the input embeddings and the output head stay in bf16 (the decision is read
  from the output head's letter logits; the two matrices are about 4% of the 4.5B model). The exported config carries an
  explicit `rope_theta`: transformers 5 writes the rotary base as `rope_parameters`, which mlx-lm does not read, and
  without the fix it silently uses 10000 instead of 1000000 (the backend applies the same fix when it loads a bf16
  Hugging Face folder).
- **GGUF.** llama.cpp `convert_hf_to_gguf.py` to bf16, `llama-quantize` to `Q8_0`, an importance matrix on the `Q8_0`
  file, then `Q4_K_M` from bf16 with the output head and token embeddings at `Q8_0`. Each quant gets an Ollama
  `Modelfile.<quant>` whose template only passes the prompt through (basal renders the prompt and sends it raw). The
  folder also holds the tokenizer, chat template, `CALIBRATION*.json` and `basal.json`, so `--model` can point at it.
- **Tokenizer fix.** The basal tokenizer (Bielik v3 APT4) is a byte-fallback BPE with a Metaspace pre-tokenizer. The
  stock converter stores it as a SentencePiece vocabulary but writes every token score as -1000, so llama.cpp and Ollama
  segment text arbitrarily: 0 of 400 rendered dev prompts were tokenized like the Hugging Face tokenizer, and a prompt
  sent with its BOS text got a second BOS. The export wraps the converter: every token's score is minus the rank
  of the first BPE merge that produces it (SentencePiece's highest-score merging then follows the BPE merge order) and
  `add_space_prefix` is false (no "▁" after special tokens, as in the Hugging Face tokenizer): 400 of 400 identical.
  Third, every added token of `tokenizer.json` (`<s>`, `<|im_start|>`, `<|im_end|>`, …) is typed as a control (or
  user-defined) token, so it is matched whole: llama.cpp only marks the tokens that `tokenizer_config.json` lists as
  special, and the basal-1.5-max config lists none, which split `<|im_start|>` into characters. With the three fixes
  llama.cpp tokenizes the rendered prompts of both the 4.5B and the max tokenizer exactly like the Hugging Face
  tokenizer (300 of 300 for max). `basal-export-gguf` applies the three fixes to a new conversion or, with `--gguf`, to
  an existing file ([GGUF.md](GGUF.md#convert-yourself)); its tokenizer fields equal those of the `-GGUF` files of
  basal-1.5 and basal-2.0. The
  `ollama` mode strips the BOS text (Ollama adds BOS itself) and warns if Ollama's prompt token count differs from the
  engine's. `--mode llamacpp` sends token ids and does not depend on the vocabulary.

## Notes

- **Prefix sharing.** MLX: the longest common token prefix of a request's prompts (system prompt and state; for one
  question also the question) is computed once, and every suffix continues a copy of that KV cache in one right-padded
  batch, which is exact (a suffix sees the prefix and its own earlier tokens only; MLX bf16 agrees with PyTorch bf16 on
  99.8%). Ollama / llama.cpp: the prompts of a request are sent one after another, so the engine's prompt cache reuses
  the state.
- **Memory.** The MLX backend wires its weights and caps MLX's buffer cache at 2 GB. Without the cap the cache grew
  towards the whole recommended working set (17 GB on a 24 GB Mac), one entry per new input shape, and pushed the
  machine into swap. The 4-bit ports fit 8 GB Macs; bf16 needs about 12 GB free.
- **Top-20.** Both HTTP engines return the 20 most likely tokens at the answer position; a letter outside them counts as
  probability 0 (it is below the 20th token). The option letters dominate the answer position, so this does not change
  decisions.
- **Robustness.** Under memory pressure macOS can kill `llama-server`; the HTTP backends retry a dropped connection three
  times.
