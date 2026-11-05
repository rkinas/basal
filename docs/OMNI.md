# basal-omni on the basal engine (Gemma 4)

basal-omni-1.0 are experimental typed-decision models on Google Gemma 4: basal-omni-1.0-mini (4B, Gemma 4 E4B) and
basal-omni-1.0-max (12B, Gemma 4 12B), with prompt format 2 ([FORMAT2.md](FORMAT2.md)): up to 255 options, one forward pass per
question. They are trained on text; with `--media on` the server also takes images and audio in a request (below).

## Serve

```bash
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-serve --model Remek/basal-omni-1.0-max --mode eager --media on --port 8000    # or Remek/basal-omni-1.0-mini
```

- **Modes:** `eager`, the PyTorch forward in bf16 on CUDA (text, images and audio), since v2.1.0 `vllm` (faster, with
  prefix caching and FlashAttention; since 2.3 also images and audio for mini, [below](#vllm---mode-vllm)) and, since
  2.3, `fast` / `fast-nocompile` (text only; the CUDA graph engine, below). Without `--mode`, basal-omni runs `fast`
  where its fast configuration passed its gate: Hopper GPUs with the `flash_attn_3` package and (2.3.1) Blackwell
  workstation GPUs (sm_120, such as the RTX PRO 6000); `eager` on other GPUs until they pass theirs (2.2.0 had no
  working default for basal-omni on CUDA: `fast` stopped while loading the model). The `-FP8` checkpoints (mini, max,
  ultra) run on `--mode vllm`;
  basal-omni-1.0-mini-FP8 also on `--mode fast` (its port gate below).
  With `--mode vllm` a model that ships `CALIBRATION.vllm.json` (basal-omni-1.1-max and the `-FP8` checkpoints) uses
  it; the others use their `CALIBRATION.json`.
- **Repeatable answers with vLLM:** vLLM's numbers depend on what the server computed before (prefix cache,
  batching, where a long prompt was cut), so a decision near a tie can change between runs. `--vllm-preset
  deterministic` (2.3) serves with batch-invariant kernels and without the prefix cache and chunked prefill
  ([below](#vllm---mode-vllm)); `--mode eager --deterministic` is the other way.
- **GGUF and MLX (text only):** `--mode gguf --gguf <file>` runs a GGUF conversion in process through llama.cpp
  (Metal, CUDA or CPU) and `--mode mlx` / `mlx-q8` runs the bf16 weights through mlx-lm on Apple Silicon (below).
- **Option orders:** like every prompt-format-2 model, it asks one option order by default and loads
  `CALIBRATION.json`. `--orders 2` averages both orders and loads `CALIBRATION.o2.json`.
- **Memory:** about 16 GB of GPU memory for mini and 24 GB for max (bf16); long states need more for attention.
- **API:** the requests and responses are the same as for every basal model ([API.md](API.md)).

## What the engine does differently for Gemma 4

- **`eager` (basal 2.3):** Gemma 4 runs through the engine's own forward (`basal/gemma4.py`: transformers' modules and
  weights in the engine's loop; `--eager-forward hf` keeps the transformers forward of 2.2):
  - **The state once (SOAM).** The prompts of a request share their beginning: chat template, media and state (with
    `--soam on`, the default, every question and option order of the request). It runs once into keys and values, and
    every question then runs over them, positions continuing after the state. Sliding-window layers keep only the last
    511 (mini) or 1,023 (max, ultra) keys of the state, all that a later token can see.
  - **Answer rows only.** mini reuses the keys and values of earlier layers in its last 18 of 42 layers, so those
    layers matter only where the answer is read. They run at that one position per prompt, and so do the attention
    and MLP of the last layer with its own keys and values (on max and ultra, the last layer); a state stops after
    that layer's keys and values.
  - **Media once per request.** The image and audio encoders run once per request and the soft tokens are part of the
    shared beginning. The soft tokens of one image attend to each other as transformers builds the masks: on max and
    1.1-max (Gemma 4 unified) in every layer, on ultra in the sliding-window layers within the window.
  - **Attention kernel** (`--eager-attn`): `math` (default) is 2.2's kernel in every layer. `auto` (opt-in) runs on
    Hopper FlashAttention 3 (varlen; the sliding window as its left window) in the sliding-window layers and SDPA's
    memory-efficient kernel in the 512-wide global layers (no flash kernel takes 512 on Hopper), elsewhere the math
    kernel; fp32 and the image rows of max, 1.1-max and ultra keep the masked math kernel. Every attention call of mini
    and max on a 1,857-token prompt and three 12.6k-token ones, against float64: FlashAttention 3 and the
    memory-efficient kernel within bf16 rounding of the math kernel, at 11x and 3x its speed (max). With both option
    orders, the share automated at 5% is 0.6 points lower than the reference path (mini; at 1% within 0.3 points). The
    memory-efficient kernel gets keys and values copied per head: 2.2's transformers forward gave it an expanded
    (zero-stride) view of max's one global key / value head, which torch before 2.14.1 reads wrong at query lengths
    = 1 mod 32 (pytorch/pytorch#191937). Since 2.3 `--mode eager` runs the transformers forward of every model without
    that kernel on such torch versions.
  - **Padding:** the prompts of a batch are padded on the right and read at their last real token, so a padded prompt
    gives the answer it gives alone.
  - **Per-layer embeddings off the GPU** (`--ple-offload on`, mini): mini looks up 256 extra numbers per layer for
    every token in a 5.6 GB table (a third of its weights). With the option the table stays in host memory and only the
    rows of each forward's tokens go to the GPU: 5.2 GB less GPU memory (peak 21.1 -> 15.9 GB) at about 4% fewer
    decisions per second, the same answers (dev v2: 10,606 of 10,606). For mini on smaller GPUs.

  Measured 2026-10-20 on one H100 against `--eager-forward hf` of the same engine (dev v2, 10,606 decisions, one
  option order; automated share at 1% on the calibration test split, each forward with its own calibration fit,
  paired bootstrap; `basal-loadtest` at 16 in flight: one question over a ~1.5k-token state, and five questions over
  3k-10k-token states):

  | model | agreement (all / at margin >= 0.2) | accuracy | ECE | automated at 1% | decisions/s, one question | questions/s, five over a long state |
  |---|---|---|---|---|---|---|
  | mini, `--eager-attn math` | 0.9982 / 1.0000 | -0.06 pt | +0.001 | 41.82% (2.2: 41.99%) | 17.2 (2.2: 10.0) | 15.3 (2.2: 1.6) |
  | max, `--eager-attn math` | 0.9994 / 0.9997 | -0.01 pt | +0.000 | 47.35% (2.2: 47.39%) | 5.0 (2.2: 4.8) | 4.5 (2.2: 0.7) |
  | mini, `--eager-attn auto` (Hopper) | 0.9940 / 0.9995 | +0.05 pt | -0.000 | 41.95% | 24.4 | 19.0 |
  | max, `--eager-attn auto` (Hopper) | 0.9894 / 0.9991 | -0.10 pt | +0.001 | 47.66% | 8.7 | 7.6 |

  With both option orders (`--orders 2`, every decision over the shared question) agreement is 0.9980 / 1.0000 on
  mini and 0.9953 / 0.9999 on max. The disagreements are close calls (eager's margin under 0.05 on mini); the
  transformers forward agrees with itself to 1.0000 when only its batching changes, and both forwards are equally far
  from an fp32 forward (mini, 2,000 items: mean |Δp| 0.0036 and 0.0037).
- **Code logits:** option codes are read in fp32 from the output head's input, with Gemma's final logit softcap
  applied in fp32. The model body runs in bf16.
- **`fast` and `fast-nocompile` (basal 2.3, text):** the CUDA graph engine runs Gemma 4 through its own layer code
  (`basal/gemma4_graph.py`, transformers' modules and weights) under the engine's packing, graphs and caches:
  - **SOAM over sliding-window layers.** The packed row of a request and its two passes work as for every basal model
    ([SOAM.md](SOAM.md)); in the sliding-window layers (window 512 on E4B, 1,024 on 12B and 31B) a branch token sees a
    key only within the window of its own position, as in its prompt alone, so the window is a condition on positions
    inside the trie mask.
  - **Keys and values per layer type** (head size 256 in sliding, 512 in global layers; 12B / 31B: one projection for
    keys and values in global layers). E4B's last 18 of 42 layers reuse the keys and values of earlier layers.
  - **Readout rows only.** Those KV-shared layers, and the attention and MLP of the last layer with its own keys and
    values, run at the answer positions only (they produce nothing a later token reads): same answers, ~40% less
    compute per decision on E4B.
  - **Attention kernel** (`--gemma-attn`): `auto` (default) is `fa3` in `--mode fast` on Hopper with the
    `flash_attn_3` wheel (as `--attn auto`), else `math`. `fa3`: FlashAttention 3 with the window in the sliding layers,
    FlexAttention in the 512-wide global layers (32 x 32 tiles: the default tiles need more shared memory than an
    H100 has) and wherever the trie needs a mask; compiled, with merged q / k / v and gate / up (`--merge auto`) and
    compact rows (`--batches auto`: prompts back to back in one row, no padding). `math`: the kernel of `--mode
    eager`, uncompiled. Also `flex` (FlexAttention everywhere) and `fa3+math`. On Blackwell workstation GPUs
    (sm_120) `auto` is `flex`, compiled, with merged projections and smaller tiles for their 99 KB of shared memory
    per block (no compact rows by default: below). On an RTX PRO 6000 vLLM is faster for single
    basal-omni-1.0-max questions (8.9 against 5.0 decisions per second), `fast` for five questions over a long state
    (5.3 against 4.9 per second). Elsewhere `auto` is `math`.
  - **Compact rows with the flex kernels** (`--batches compact` on Blackwell workstation GPUs; 2.5, opt-in, off by
    default): a flex block mask per prompt in every layer (a query block visits only the key blocks from its prompt's
    start, and its window, on). Faster, but the answers differ from the default rows'. One RTX PRO 6000 Blackwell
    Max-Q, 16 clients, against the default rows of `--mode fast` (each a fresh server):

    | model | workload | questions/s | same top answer | changed where the default's margin >= 0.05 |
    |---|---|---|---|---|
    | basal-omni-1.0-mini | 2,000 development decisions, one per request | 126.6 -> 152.3 (1.20x) | 99.60% | 1 |
    | | 600 single questions, ~1.5k-token states | 22.8 -> 26.6 (1.17x) | 98.67% | 0 |
    | | 120 requests, five questions over 3k-10k tokens | 21.2 -> 21.3 (1.00x) | 100% (580 of 600 bit-identical) | 0 |
    | basal-omni-1.0-max | 2,000 development decisions | 31.0 -> 39.5 (1.27x) | 99.75% | 2 |
    | | 600 single questions | 5.09 -> 6.14 (1.21x) | 98.00% | 4 |
    | | 120 requests, five questions | 5.33 -> 5.32 (1.00x) | 100% (580 of 600 bit-identical) | 0 |

    Requests with several questions do not use compact rows. For comparison, two runs of the default rows agreed on
    99.85-100% (mini) and 99.8-100% (max) of the same readouts, with 0-2 changes at a margin >= 0.05.
  - Text only: `--media on` needs `--mode eager`; the image and audio towers are not loaded. Each captured CUDA graph
    keeps its own cuBLAS workspace, so the engine sets `CUBLAS_WORKSPACE_CONFIG=:4096:2` for Gemma 4 unless you set it
    (1.1-max ran out of GPU memory in warm-up with the default; the smaller workspace did not change its forward
    time). On 31B the graph grid shrinks to fit (key / value buffers within 15% of the free memory). A state beyond the
    largest graph keeps only the last window of the sliding layers' keys and values (a question never reads further
    back), also in the state cache: a fifth of the memory per token on 31B.
  - **A Triton kernel for the global layers** (`--global-attn triton`; 2.5, opt-in, off by default; Hopper): the
    512-wide global layers of plain rows (single prompts and the state of a multi-question request) run basal's own
    causal kernel for 512-wide heads instead of FlexAttention's 32 x 32 tiles; pass 2, compact rows and the readout
    rows keep flex. Faster on long states, but the answers differ from flex's (the two kernels' outputs differ by up to
    one bf16 step per layer, which moves near-ties on long states). One H100, `--mode fast`, against `--global-attn flex` (each a fresh server), 120
    synthetic states of 24k-32k tokens with five questions each (600 readouts) at 4 clients, and 60 states of 15k-17k
    tokens (300 readouts):

    | model | 24k-32k: questions/s | same top answer | changed where flex's margin >= 0.05 | 15k-17k: speed, same top answer, changed |
    |---|---|---|---|---|
    | basal-omni-1.0-mini | 7.85 -> 8.62 (1.10x) | 99.67% | 0 | 1.07x, 99.0%, 0 |
    | basal-omni-1.1-max | 2.22 -> 2.57 (1.16x) | 98.33% | 4 | 1.10x, 99.3%, 1 |
    | basal-omni-1.0-ultra | 0.73 -> 0.75 (1.03x) | 99.0% | 2 | 1.10x, 99.0%, 0 |

    Not measured on shorter states, other GPUs or with `--deterministic`.
  - `--mode fp8` / `nvfp4` (quantisation at load time) are refused for basal-omni: torchao FP8 on 1.1-max missed the
    port gate (0.9892 at margin >= 0.2 on 2-10 options, bar 0.9893). Serve an `-FP8` checkpoint instead.

  Measured 2026-10-20 on one H100 against `--mode eager` of the same engine (the gate of the FP8 ports, as above;
  dev v2, 10,606 decisions, one order; automated share at 1% on the calibration test split with the calibration file
  the mode loads, paired bootstrap, bar: 95% lower bound of the difference >= -0.5 points; `basal-loadtest` at 16 in
  flight: one question over a ~1.5k-token state, and five questions over 3k-10k-token states):

  | model, configuration | agreement (all / at margin >= 0.2) | accuracy | ECE | automated at 1% (eager) | decisions/s, one question (eager) | questions/s, five over a long state (eager) |
  |---|---|---|---|---|---|---|
  | mini, `fast` (auto: fa3, compact rows) | 0.9937 / 0.9994 | +0.00 pt | +0.000 | 41.97% (41.99%) | 61.2 (10.0) | 21.4 (1.6) |
  | 1.1-max, `fast` (auto: fa3, compact rows) | 0.9891 / 0.9990 | +0.04 pt | -0.001 | 47.16% (47.10%) | 16.2 (4.8) | 10.6 (0.7) |
  | mini, `fast --gemma-attn math` | 0.9984 / 0.9997 | -0.02 pt | +0.000 | | 15.8 | 7.9 |
  | 1.1-max, `fast --gemma-attn math` | 0.9992 / 0.9997 | +0.01 pt | -0.000 | | 3.8 | 2.0 |
  | ultra, `fast` (auto: fa3, compact rows) | 0.9918 / 0.9985 | -0.06 pt | +0.002 | 56.73% (56.82%) | 6.1 (2.0 at 4 in flight) | 4.8 (0.3 at 2 in flight) |
  | mini-FP8 checkpoint, `fast` | 0.9635 / 0.9908 | -0.26 pt | +0.002 | 41.82% (41.99%) | 80.6 | 19.6 |

  `--mode fast` has no calibration file of its own (it never loads a `CALIBRATION.fast.json`): it loads the model's
  `CALIBRATION.json`, fitted on eager readouts, and the automated shares above are measured with that file (mini-FP8:
  the `-FP8` repository's `CALIBRATION.json`, fitted on vLLM FP8 readouts; at the 5% target it automates 63.5% against
  eager's 77.4%). A calibration refitted on fast-mode readouts is not shipped: the refit moves the certified thresholds
  by fit noise, not because the readouts differ (top answer 0.996-0.997 of the calibration test against eager's, no
  mean shift of the confidence). Refits of four fast configurations of 1.1-max with equally close readouts automated
  -0.49, -0.39, -0.05 and -0.11 points against eager's own fit at 1% (the default configuration: -0.11), ultra's -0.56
  (its refit's yes / no temperature is 0.958 against 0.944, so fewer yes / no answers clear the threshold), and mini's
  refit certifies choice questions only at confidence 0.9998 at the 5% target (32.7% automated instead of 77.4%). On
  one 80 GB H100 ultra answers five questions over 3k-10k-token states at 4.8 per second (vLLM bf16 with `--max-len
  8192`: 3.8). For comparison, vLLM bf16 on 1.1-max: 0.9866 / 0.9994, 16.7 decisions/s, 10.8 questions/s.
  With the math kernel (fp32 scores, without tensor cores) 1.1-max's single questions are slower than eager's larger
  batches. In fp32 every path of `fast` (fa3's compiled code with flex in place of FA3) gives the answers of separate
  prompts (identity gate: max |dp| 1.6e-5). Start-up: about 5 minutes (compile and graph capture).
- **Not yet for Gemma 4:** the engine's `nvfp4` mode and SGLang.

## vLLM (`--mode vllm`)

The vLLM path of 2.2 is unchanged by default (vLLM 0.31.0). Added in 2.3, each measured on one H100 against
`--mode eager` bf16 on dev v2 (10,606 decisions, 32 in flight) and with the calibration file the mode loads on the
calibration test split:

- **How close to eager (2.5 check, H100, 27 Oct 2026, the default preset):** `--mode vllm` reads in vLLM's own kernels,
  so its answers are close to `--mode eager`'s, not equal. Of 361 readouts per model, the same top answer on 98.3%
  (basal-omni-1.0-mini), 98.3% (basal-omni-pico), 98.6% (basal-omni-1.0-max) and 97.6% (basal-omni-1.1-max); answers can
  differ on near-ties, and the default preset can vary from run to run. With `--vllm-preset deterministic` (below) the
  answers are reproducible and bit-identical to 2.4.1's on these models but basal-omni-pico, for which basal-serve
  refuses that preset before vLLM starts (vLLM has no batch-invariant kernel for its Gated DeltaNet layers).
- **KV-sharing fast prefill** (`--vllm-fast-prefill auto`, the default; `off` restores 2.2): Gemma 4 E2B / E4B (mini)
  run their last 18 of 42 layers, which reuse earlier layers' keys and values, at the answer position only during
  prefill (vLLM's `kv_sharing_fast_prefill`). The generated token, the only one basal reads, is computed the same way.
  mini: the same top answer as eager on 0.9994 of the decisions where eager's margin is at least 0.2 (2.2's vLLM path:
  0.9993), accuracy +0.04 points; with the shipped `CALIBRATION.json` it automates 41.8% of the calibration test at
  1% error against eager's 42.0% (-0.18 points [-0.35, 0.00]; observed error 0.70%, at 5% 3.63%); 1.43x the decisions
  per second of 2.2's vLLM path for one question over a ~1.5k-token state (60.8 vs 41.5 at 16 in flight) and 1.60x the
  questions per second for five questions over a 3k-10k-token state (39.0 vs 24.2). Models without KV-shared layers
  are not affected.
- **`--vllm-preset deterministic`**: batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`, Triton attention) without the
  prefix cache and without chunked prefill, so an answer does not depend on the other requests or on what the server
  computed before. basal-omni-1.1-max, two dev v2 runs at 32 in flight: bit-identical probabilities on every decision
  with 2-10 options (8,545) and on all 2,061 with 11-255 options; the same top answer as eager on 0.9990 of the
  decisions with margin >= 0.2;
  automation at 1% with the shipped `CALIBRATION.vllm.json` -0.12 points [-0.30, +0.07] against eager (mini -0.19,
  ultra +0.01, each with its shipped file). It costs throughput, most where a request asks several questions over one
  state (no prefix cache): 1.1-max 0.64x the decisions per second of the default for single questions and 0.22x the
  questions per second for five questions over a long state; mini 0.66x and 0.26x. A model ships
  `CALIBRATION.vllm.deterministic.json` only if its own file fails there; none does.
- **Images and audio** (`--media on`, as with `--mode eager`, below): the prompts of a media request go to vLLM in one
  call with the chat-template text of the eager media path, and vLLM's Gemma 4 processor expands the media
  placeholders to the same tokens. On the omni media check set (310 images and 285 audio clips with their text twins),
  against `--mode eager --media on`: basal-omni-1.0-mini the same top answer on 96.5% of images and 98.3% of clips
  (all where eager's margin is at least 0.2), accuracy +0.3 / +0.7 points, every prompt of the same length; 6.0x /
  7.9x the requests per second at 8 in flight (75.7 vs 12.7 images, 84.9 vs 10.8 clips) and 17.7 / 42.6 ms per request
  one at a time (eager 102 / 104 ms).
- **basal-omni-1.1-max** (Gemma 4 Unified: no image or audio encoder) since 2.5: vLLM 0.31 reads such a model with the
  rules of the encoder models (an image's soft tokens see each other only in the sliding-window layers, within the
  window; audio counted in mel frames), so the engine patches both in vLLM's worker to the model's own (every layer,
  past the window; one token per 640 samples) and refuses to start if the patch did not take. Same check set: the same
  top answer on 97.4% of images and 95.8% of clips (all where eager's margin is at least 0.2), accuracy +0.3 / +0.4
  points, every prompt of the same length; 2.7x / 2.9x the requests per second at 8 in flight (47.5 vs 17.6 images,
  46.7 vs 16.3 clips) and 25.9 / 23.4 ms per request one at a time (eager 70 / 62 ms). `"detail": "high"` is refused
  for it on `--mode vllm` (the eager answer on 54.8% of images only); use `--mode eager --media on` for it. Ultra
  (images) is not checked yet.
- **`--vllm-overlap on`** (2.5, opt-in, off by default): the backend drives vLLM's async engine and keeps up to four
  engine calls in flight, so vLLM's scheduler batches their prompts together (off: one `LLM.generate` call at a time).
  A little faster, and the answers differ from off's. One RTX PRO 6000 Blackwell Max-Q, `--max-len 12288`, 16
  clients, against off (each a fresh server):

  | model | workload | questions/s | same top answer | changed where off's margin >= 0.05 |
  |---|---|---|---|---|
  | basal-omni-1.0-mini | 2,000 development decisions, one per request | 125.5 -> 151.3 (1.21x) | 99.45% | 3 |
  | | 600 single questions, ~1.5k-token states | 25.2 -> 26.9 (1.07x) | 98.17% | 3 |
  | | 120 requests, five questions over 3k-10k tokens | 12.1 -> 12.9 (1.07x) | 99.0% | 2 |
  | basal-omni-1.0-max | 2,000 development decisions | 44.8 -> 47.6 (1.06x) | 99.65% | 6 |
  | | 600 single questions | 5.71 -> 5.79 (1.01x) | 99.17% | 2 |
  | | 120 requests, five questions | 3.25 -> 3.31 (1.02x) | 98.5% | 5 |

  Two runs with off differed from each other about as much (98.17-99.7% the same top answer, 1-6 changes at a margin
  >= 0.05): vLLM's default preset varies from run to run under load.

  With `--media on`, requests with media are batched the same way (off: each one is its own call, one at a time).
  basal-omni-1.1-max, one H100, the media items of the check set above (310 images, 285 clips, detail "default"), 8
  clients, against off (fresh servers: off three times, on twice):
  - media new to the server: images 7.6-9.2 vs 6.2-6.5 requests per second (1.18-1.49x; p50 170 vs 420 ms), audio
    clips 15.1-20.7 vs 16.8-16.9 (0.89-1.23x);
  - the same requests again (served from vLLM's caches): images 49.2-49.5 vs 33.0-33.7 (1.46-1.50x), audio 44.7-45.1
    vs 33.3-33.7 (1.33-1.35x);
  - answers: the same top answer as off on 99.68% of the images and 99.65-100% of the clips (at most one change, at a
    margin under 0.05), accuracy -0.32 / 0.00 to -0.35 points. With off, the answers are bit-identical between fresh
    starts and with 1 or 8 clients.
- **Not in 2.3:** FP8 weights quantised at load time (automation at 1% -0.5 to -9.5 points on the three tiers) and an
  FP8 KV cache (on mini -0.36 points [-0.57, -0.15] at 1% with the shipped file; on 12B / 31B it needs Triton attention
  and is slower than 2.2's path). The `-FP8` checkpoints are served as they are.
- `basal-loadtest --seed N` draws other synthetic states, so a repeated load test does not hit the prefix cache of an
  earlier one (all numbers above: two runs with fresh states each, the slower one reported against the faster
  baseline run).

## GGUF (llama.cpp, Ollama) and MLX (text)

llama.cpp (b11312, the build in llama-cpp-python 0.3.36) and mlx-lm 0.32.2 implement Gemma 4's text decoder: sliding
and global attention, per-layer inputs (E4B), KV sharing, the final logit softcap. Convert with llama.cpp's converter
(the text decoder only; the image and audio towers are left out), as for basal-1.0 ([GGUF.md](GGUF.md#convert-yourself)):

```bash
python llama.cpp/convert_hf_to_gguf.py basal-omni-1.0-mini --outtype q8_0 --outfile basal-omni-1.0-mini-Q8_0.gguf
basal-serve --model ./basal-omni-1.0-mini --mode gguf --gguf basal-omni-1.0-mini-Q8_0.gguf --port 8000
basal-serve --model ./basal-omni-1.0-mini --mode mlx --port 8000        # Apple Silicon, bf16 weights (mlx-q8: 8-bit)
```

Ready files: [`Remek/basal-omni-1.0-mini-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF) and
[`Remek/basal-omni-1.0-max-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF) (Q8_0 and Q6_K, Modelfiles for
Ollama, `CALIBRATION.gguf.json` fitted on Q8_0); their cards have the commands.

8-bit MLX ports for `--mode mlx` (Apple Silicon, `CALIBRATION.mlx.json`):
[`Remek/basal-omni-1.0-mini-MLX`](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX),
[`Remek/basal-omni-1.0-max-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX),
[`Remek/basal-omni-1.1-max-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX).

llama.cpp tokenizes the rendered basal-omni prompts exactly like the Hugging Face tokenizer (2,000 of 2,000 development
prompts), so text front ends (Ollama, `llama-server` with text prompts) read the same tokens. For Gemma 4 the engine
reads the decoder inside mlx-lm's multimodal wrapper, applies the softcap to the fp32 code logits and keeps the whole
shared prefix in the KV cache (mlx-lm's rotating cache would keep only the last window); `--mode gguf` loads
`CALIBRATION.gguf.json` and `--mode mlx` `CALIBRATION.mlx.json` when the model ships one, else `CALIBRATION.json`.

Measured 2026-10-17 against the eager bf16 engine, one option order, the shipped `CALIBRATION.json`, with the gate of
the FP8 ports (agreement of the top answer on 2-10 options and at bf16 margin >= 0.2 above the tier's bars, accuracy
within 1 point, ECE at most 0.01 above). CUDA: one RTX PRO 6000 (Blackwell), all 10,606 development decisions, 16
requests in flight. Apple: M4 Pro (24 GB), mini on 2,000 of them (1,500 / 200 / 300 with 2-10 / 11-26 / 27-255
options), max on 600 (400 / 100 / 100), 8 in flight. Quantisations below Q8_0 are made with `llama-quantize` from the
F16 file, without an importance matrix.

| model | port | file | where | agreement 2-10 | at margin >= 0.2 | accuracy | ECE | decisions/s | gate |
|---|---|---|---|---|---|---|---|---|---|
| mini | GGUF F16 | 14.9 GB | CUDA | 0.9949 | 0.9999 | -0.1 pt | +0.001 | 17.2 | pass |
| mini | GGUF Q8_0 | 7.9 GB | CUDA | 0.9871 | 0.9977 | -0.1 pt | -0.000 | 16.4 | pass |
| mini | GGUF Q6_K | 6.2 GB | CUDA | 0.9709 | 0.9932 | +0.1 pt | -0.005 | 14.4 | pass |
| mini | GGUF Q5_K_M | 5.7 GB | CUDA | 0.9476 | 0.9789 | -0.3 pt | +0.002 | 15.9 | fail (margin) |
| mini | GGUF Q4_K_M | 5.3 GB | CUDA | 0.9020 | 0.9479 | -0.2 pt | -0.003 | 16.3 | fail |
| mini | GGUF Q8_0 | 7.9 GB | Metal | 0.9887 | 0.9992 | +0.1 pt | -0.003 | 1.2 | pass |
| mini | MLX bf16 | 15 GB | Metal | 0.9933 | 0.9992 | -0.1 pt | +0.001 | 1.0 | pass (1) |
| mini | MLX `mlx-q8` | 8 GB | Metal | 0.9893 | 0.9992 | +0.1 pt | +0.001 | 0.8 | pass |
| mini | Ollama 0.35.0, Q8_0 | 7.9 GB | Metal | 0.9800 | 0.9961 | +0.3 pt | +0.003 | 2.1 | pass (2) |
| max | GGUF F16 | 23.8 GB | CUDA | 0.9930 | 0.9995 | +0.0 pt | +0.001 | 9.2 | pass |
| max | GGUF Q8_0 | 12.7 GB | CUDA | 0.9868 | 0.9983 | -0.1 pt | -0.001 | 8.9 | pass |
| max | GGUF Q6_K | 9.8 GB | CUDA | 0.9761 | 0.9947 | -0.1 pt | -0.005 | 7.5 | pass |
| max | GGUF Q5_K_M | 8.5 GB | CUDA | 0.9579 | 0.9825 | -0.4 pt | +0.001 | 8.5 | fail (margin) |
| max | GGUF Q4_K_M | 7.4 GB | CUDA | 0.9346 | 0.9607 | -1.0 pt | +0.016 | 8.7 | fail |
| max | GGUF Q8_0 | 12.7 GB | Metal | 0.9950 | 1.0000 | +0.0 pt | +0.013 | 0.4 | (3) |
| max | MLX `mlx-q8` | 14 GB | Metal | 0.9875 | 0.9973 | +0.8 pt | +0.008 | 0.3 | pass (4) |
| max | MLX 8-bit port (`-MLX`) | 13.6 GB | Metal | 0.989 | 1.000 | -0.1 pt | +0.002 | 0.25 | pass (5) |
| 1.1-max | MLX 8-bit port (`-MLX`) | 13.6 GB | Metal | 0.984 | 0.999 | +0.6 pt | -0.003 | 0.27 | pass (6) |

(1) On the 24 GB Mac, 191 of the 300 longest prompts (27-255 options) ran out of GPU memory with bf16 weights; the
2-10-option items the gate reads all ran. (2) 300 items with 2-10 options, one request at a time (Ollama returns at
most 20 log-probabilities, so up to 20 options). (3) max Q8_0 with the default 16,384-cell context does not fit in the
24 GB Mac's GPU memory (llama_decode fails); measured with 8,192 cells. Its ECE on 400 items is above the bar while
the CUDA run of the same file on the same 400 items is +0.001 and the relative temperature is 1.00: binned ECE at this
size moves by about 0.01; a Metal run on more items (or a larger Mac) is needed before claiming it. (4) 24 of the 100
longest prompts ran out of GPU memory on the 24 GB Mac; accuracy and ECE are close to the bars on 400 items.
(5) [`Remek/basal-omni-1.0-max-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX) on the same M4 Pro (24 GB),
8 in flight: one question takes 453 ms; long prompts can run out of GPU memory at 24 GB. Its own
`CALIBRATION.mlx.json` automates 36% of decisions at `max_error` 1% and 85% at 5%.
(6) [`Remek/basal-omni-1.1-max-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX), against basal-omni-1.1-max
bf16, on the same M4 Pro (24 GB), 8 in flight, 18-19 Oct: every 8th development decision (1,261 of 1,325 ran; 1,039
with 2-10 options). One question takes 525 ms; long prompts can run out of GPU memory at 24 GB. Its own
`CALIBRATION.mlx.json` (calibration run on every 2nd item, 4,280 test decisions) automates 35% of decisions at
`max_error` 1% and 80% at 5%.
For comparison on the same GPU: eager bf16 985 s (mini) and 2,157 s (max) for the 10,606 decisions at 8 in flight, vLLM
bf16 415 s and 840 s; GGUF Q8_0 646 s and 1,186 s at 16. Use Q8_0 (or Q6_K where memory is short); Q5_K_M and Q4_K_M
move too many close decisions.

**Speed on an Apple M5 Pro (20-core GPU, 64 GB)**, measured by Kamil
([@kamilrosa](https://github.com/kamilrosa), [#36](https://github.com/rkinas/basal/issues/36)), basal v2.3.1, the 44
bundled examples with `basal-bench` (two repetitions), and `basal-serve` + `basal-loadtest` over HTTP (0 errors).
Nothing ran out of memory at 64 GB, including bf16 max on 1.5k-token prompts.

| model | mode | `basal-bench` 2 orders / 1 order (ms) | GB | HTTP short, 32 clients (dec/s, p50 ms) | HTTP 1.5k tokens, 16 clients (dec/s, p50 ms) |
|---|---|---|---|---|---|
| omni-1.0-mini | `mps` (bf16) | 84 / 71 | 18.0 | **19.0**, 59 | **1.69**, 509 |
| | `mlx` (bf16) | 77 / 68 | 14.1 | | |
| | `-MLX` port | 111 / 68 | 8.3 | 17.4, 70 | 1.43, 665 |
| omni-1.0-max | `mps` (bf16) | **190** / 161 | 26.2 | 4.26, 167 | 0.38, 2,239 |
| | `mlx` (bf16) | 270 / 177 | 22.4 | | |
| | `-MLX` port | 317 / 181 | 13.0 | **6.07**, 177 | **0.51**, 1,874 |
| omni-1.1-max | `-MLX` port | 319 / 182 | 13.0 | 6.03, 179 | 0.51, 1,884 |

For omni-mini `mps` bf16 is the faster route; for omni-max the 8-bit `-MLX` port serves more decisions under load
with half the memory, while `mps` is faster for a single decision. Mixed load (30% long prompts, 8 clients) raises
p95 to 5.2 s (mini) and 13.8 s (max), because short requests queue behind long ones.

## Images and audio (`--media on`)

`basal-serve --model <basal-omni model> --mode eager --media on` (or `--mode vllm --media on` for basal-omni-1.0-mini,
[above](#vllm---mode-vllm)) loads the model with Gemma 4's image and audio encoders
and its processor (install `basal[media]`: pillow, soundfile, scipy, torchvision matching torch). A request may then
carry `"media": [{"type": "image" | "audio", "data": "<base64>"}]`, `"detail": "default" | "high"` (280 or 1,120
image tokens) and `"lang": "pl" | "en"`; see `basal/media.py` for the formats and limits. The media go where the state
text goes, before it (`prompt.media_messages`), the prompt the basal-omni evaluation reads media with. Since 2.3 the
image and audio encoders run once per request and the media and state run once for all its questions (above).
Answers carry `"calibration": "text"`, and `usage.media` counts the items and their tokens.

Checked 2026-10-20 on one H100 against `--eager-forward hf` (150 images and 150 audio clips of omni-probe-pl, one
question per request and three questions over the same media): the same top answer on every decision with a margin
of at least 0.2 (mini and max), 1,199 of 1,200 overall on max and all on mini, largest |Δp| 0.036; median latency of a
three-question request 328 -> 194 ms (mini), 204 -> 131 ms (max).

Checked 2026-10-10 on Gemma 4 E4B and 12B: served answers against the fp32 readout of the same model
on 1,800 text items, 120 prompts over 2,048 tokens, 310 images, 285 audio clips and their text twins; every split is
within the bf16 readout's own deviation. On E4B the text answers with
`--media on` equal those of the earlier text-only engine on the same items.
