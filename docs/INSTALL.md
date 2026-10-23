# Install and run

Every way to run basal, all behind the same `basal-serve` HTTP API. The short version is in the
[README](../README.md#quick-start).

## What the model repositories contain

basal-2.0:

- **bf16** weights run in this engine (`fast`, `eager`, `mps`, `mlx`) and in vLLM / SGLang. Each repository ships
  `basal.json` (prompt format and readout), `CALIBRATION.json` (one option order) and `CALIBRATION.o2.json` (two
  orders): per-type temperatures and confidence thresholds. The 4.5B repository also ships `think/` and
  `CALIBRATION.think.json` for the thinking mode. The 2.0 models ship no evidence head.
- **`-FP8`**: NVIDIA ModelOpt FP8 checkpoints, served by the basal engine itself (`--mode fp8`, sm89+).
- **`-GGUF`** (`Q8_0`, `Q5_K_M`) for Ollama and llama.cpp, **`-MLX-8bit`** for Apple Silicon.
- Every port is checked against its bf16 model on the development set (agreement of decisions, accuracy, calibration
  error); the FP8, GGUF and MLX ports of all three tiers pass their bars. Each port's card has its numbers, its
  calibration files and, where a port misses a bar, an *experimental* label.
- The basal-2.0 models are fine-tuned from Apache-2.0 Bielik v3 base models (see [NOTICE](../NOTICE)) and released under
  Apache-2.0.

basal-omni-1.0:

- bf16 weights, served by `basal-serve --mode eager` on an NVIDIA GPU (`--media on` for images and audio). No FP8,
  NVFP4, GGUF or MLX ports and no thinking mode yet. Fine-tuned from Apache-2.0 Gemma 4 models (see [NOTICE](../NOTICE))
  and released under Apache-2.0.

## Engines

basal runs on seven engines, all behind the same `basal-serve` HTTP API (`POST /v1/systemone`). Commands below use
basal-2.0 (4.5B); replace the repository name for mini or max.

<!-- engines -->
| engine | `basal-serve` mode | hardware | what it is for | quick start |
|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`); `mps`; `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon (`mps`); CPU (`mps --device cpu`, `eager`) | the primary server: packed requests (SOAM), compact batching, `--deterministic` | `basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off` |
| **basal engine, FP8** | `fp8` | NVIDIA sm89+ (Hopper, Blackwell) | the `-FP8` checkpoints in the same engine (bf16 weights are quantised on load) | `basal-serve --model Remek/basal-2.0-4.5B-FP8 --mode fp8 --template-prefix off` |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server for bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only) | `basal-serve --model Remek/basal-2.0-4.5B --mode vllm` (extra `basal[vllm]`) |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (measured with basal-1.5) | `basal-serve --model Remek/basal-2.0-4.5B --mode sglang` (own environment) |
| **MLX** | `mlx` (default on a Mac with `basal[mlx]`) | Apple Silicon | Macs: the `-MLX-8bit` ports | `basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with the `-GGUF` files (up to 20 options per question) | `basal-serve --model ./basal-2.0-4.5B-GGUF --mode ollama --ollama-model basal-2.0-4.5b:q8_0` |
| **llama.cpp** | `llamacpp`; `gguf` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files, or the file in process (`--mode gguf`, extra `basal[gguf]`) | `basal-serve --model ./basal-2.0-4.5B-GGUF --mode llamacpp` |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp only run the weights;
the typed, calibrated answers (a probability for each of your options, the state shared by several questions,
`multi`, `act`, `tool` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API
in every mode. A plain `ollama run`, `llama-cli` or `mlx_lm.generate` gives free text only. vLLM and SGLang start with
a 4,096-token context (longer states are cut or refused, see `--context-overflow`); raise it with `--max-len` (e.g.
32768 for long documents). Install and first request for each engine: [quick start](#quick-start).
<!-- /engines -->

The basal-omni models (Gemma 4) run in the basal engine's `eager` mode only, on an NVIDIA GPU; `--media on` adds images
and audio ([images and audio](#images-and-audio-basal-omni-experimental)).

## Quick start

### NVIDIA GPU: the basal engine (recommended)

Install with [uv](https://docs.astral.sh/uv/) into a fresh environment (no git needed; `pip install uv` if it is
missing):

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

- **`--template-prefix off` for the 4.5B model** (bf16 and `-FP8`): the prompt-template cache failed this model's
  check, and its numbers were measured without it. mini and max run with the default (`auto`, on).
- **Install torch first, from the CUDA 12.8 index.** A plain `pip install basal` takes the newest torch from PyPI,
  which may be built for a newer CUDA than your GPU driver ("The NVIDIA driver on your system is too old"). The cu128
  build needs a driver that supports CUDA 12.8 or newer (`nvidia-smi` shows it top right). DGX Spark and B300: use the
  cu130 build ([HARDWARE.md](HARDWARE.md)).
- **Use a fresh environment on cloud GPU images.** Their system Python often ships a `torchvision` built for another
  torch, which breaks `transformers` ("operator torchvision::nms does not exist"). The text models do not need
  torchvision (the `basal[media]` extra for basal-omni installs it).
- The first start in mode `fast` compiles the model and captures CUDA graphs (several minutes); `--mode
  fast-nocompile` starts in seconds. The compiled code is cached in `~/.cache/basal/torchinductor` (set
  `TORCHINDUCTOR_CACHE_DIR` to put it elsewhere), so later starts, after a reboot too, skip most of the compiling;
  delete the directory to free its space. On Hopper, FlashAttention 3 is used when its wheel is installed
  (`uv pip install flash_attn_3 --extra-index-url https://download.pytorch.org/whl/cu128`); compact batching and
  the template cache use it. On other GPUs, `--batches compact` runs them through torch's varlen attention (about
  6 GB more GPU memory for mini and 7 GB for the 4.5B where the card has room, 2.4 GB on a full 16 GB card;
  [SERVING.md](SERVING.md#serving-modes)).
- Other models: `--model Remek/basal-2.0-max` or `--model Remek/basal-2.0-mini`.
- **basal-omni-pico (Qwen3.5) needs flash-linear-attention** for its Gated DeltaNet layers in `--mode fast` (2.4+):
  `uv pip install "flash-linear-attention==0.5.2"`, then `basal-serve --model Remek/basal-omni-pico`. Without it,
  `--mode fast` refuses the model and says so, and without `--mode` the model runs in `eager` (3-10x slower).
- From a clone instead: `git clone https://github.com/rkinas/basal && cd basal && uv pip install -e ".[fp8]"`
  (after the torch line above).

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

To send uncertain decisions to a person instead, ask the same question as an `act` question with a defer action and
`"max_error": 0.05` ([question types](API.md#question-types)), for example with `"option_keys": "hide"` and
`"costs": {"wrong": 19, "defer": 1}`; the answer then says `"deferred": true` when the model is not sure enough.

Python (the package includes a small client):

```python
from basal.client import Basal
b = Basal("http://127.0.0.1:8000")
state = ("Zgłoszenie z oddziału w Gdańsku: od 7:40 nie działa żaden terminal płatniczy, klienci odchodzą od kas, "
         "kolejka na 30 osób. Obejście: tylko gotówka.")
a = b.score(state, "Jak pilne jest to zgłoszenie?",
            ["niska — można zaplanować", "średnia — w ciągu kilku dni", "wysoka — dziś", "krytyczna — natychmiast"])
print(round(a["score"], 2), a["probabilities"])          # expected level 0-3 and the distribution
print(b.yes_no(state, "Czy problem dotyczy płatności kartą?")["noul"])   # P(yes)
```

### FP8 checkpoints (basal engine)

The `-FP8` repositories are NVIDIA ModelOpt FP8 checkpoints that the basal engine loads natively (the same install as
above; GPU with FP8, sm89+):

```bash
basal-serve --model Remek/basal-2.0-mini-FP8 --mode fp8 --port 8000
basal-serve --model Remek/basal-2.0-4.5B-FP8 --mode fp8 --template-prefix off --port 8000
basal-serve --model Remek/basal-2.0-max-FP8 --mode fp8 --port 8000
```

In FP8 the batch can change near-tie decisions; `--deterministic` runs every request alone with full FP8
accumulation. With a bf16 model, `--mode fp8` quantises the weights at load time (torchao) instead. FP8 keeps
FlashAttention 2 (no compact batching or template prefix) also where FlashAttention 3 is installed: the FP8 ports were
checked on that path. `--attn fa3` selects FlashAttention 3 anyway: it is 21–26% faster on an H100 (4.5B, 16 clients:
48.6 vs 39.5 decisions/s) but is not the checked path: 3–8% of decisions then differ from the checked rows, mostly near
ties, and accuracy moved by −0.8 to +1.4 points against them on a development sample.

### vLLM or SGLang with bf16 weights

```bash
# vLLM brings its own torch: a separate environment. vLLM 0.31 brings torch built for CUDA 13.0: NVIDIA driver >= 580
uv venv --python 3.12 --allow-existing ~/basal-vllm && source ~/basal-vllm/bin/activate
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode vllm --port 8000          # --max-len 32768 for long documents

# SGLang (own environment; its kernels need a CUDA 12.9+ toolkit; it also brings torch for CUDA 13.0: driver >= 580)
uv venv --python 3.12 --allow-existing ~/basal-sglang && source ~/basal-sglang/bin/activate
uv pip install --prerelease=allow "sglang[srt]==0.5.21"
uv pip install --no-deps "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
uv pip install starlette uvicorn httpx accelerate safetensors   # if missing
basal-serve --model Remek/basal-2.0-4.5B --mode sglang --port 8000
```

The basal server, API and calibration stay the same; the backend reads the option codes' log-probabilities (vLLM up
to 255 codes, see [FORMATS.md](FORMATS.md#engines-and-the-number-of-options)). A per-engine calibration file
(`CALIBRATION.<mode>.json`) is used when a model ships one, else `CALIBRATION.json`. The basal-2.0 release numbers
were measured with the basal engine, not with these backends; their agreement with the basal engine was measured on
basal-1.5 ([QUANTIZATION.md](QUANTIZATION.md)).

### Apple Silicon (MLX)

```bash
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000
```

The request and response are the same as above. `-MLX-8bit` holds affine 8-bit decoder weights; the token embeddings
and the output head stay in bf16, because the decision is read from the output head. The state of a request is
computed once into a KV cache shared by all its questions. With `basal[mlx]` installed, `mlx` is the default mode on a
Mac; `--mode mlx` also loads a bf16 model directory, and `--mode mps` runs the PyTorch engine on the Mac's GPU
instead ([HARDWARE.md](HARDWARE.md#apple-silicon)). How the ports work: [PORTS.md](PORTS.md).

### Ollama and llama.cpp (GGUF)

The basal server runs in front of Ollama or llama.cpp: it renders basal's own prompt, sends it raw and reads the
log-probabilities of the option codes of one generated token. Ollama 0.12.11 or newer:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"   # also installs the hf CLI
hf download Remek/basal-2.0-4.5B-GGUF --local-dir basal-2.0-4.5B-GGUF
cd basal-2.0-4.5B-GGUF && ollama create basal-2.0-4.5b:q8_0 -f Modelfile.Q8_0 && cd ..
basal-serve --model ./basal-2.0-4.5B-GGUF --mode ollama --ollama-model basal-2.0-4.5b:q8_0 --port 8000
```

`--model` points at the downloaded folder for the tokenizer, chat template and calibration files; Ollama serves the
weights. Ollama returns at most 20 log-probabilities per position, so questions with more than 20 options need
llama.cpp. The llama.cpp server takes the engine's token ids directly:

```bash
llama-server -m basal-2.0-4.5B-GGUF/basal-2.0-4.5B-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-2.0-4.5B-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

Without a separate server, `--mode gguf` loads the file in process through llama-cpp-python (Metal, CUDA or CPU; for
CUDA build it first, see [GGUF.md](GGUF.md)):

```bash
uv pip install "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model ./basal-2.0-4.5B-GGUF --mode gguf --gguf basal-2.0-4.5B-GGUF/basal-2.0-4.5B-Q8_0.gguf --port 8000
```

`--mode gguf` loads `CALIBRATION.gguf.json` when the folder ships one; the basal-2.0 `-GGUF` folders do not, so it uses
`CALIBRATION.json` (the bf16 fit); add `--calibration basal-2.0-4.5B-GGUF/CALIBRATION.llamacpp.json` to use the
llama.cpp fit. Context: Ollama mode reads at most 4,096 tokens per prompt (basal sends `num_ctx` 4096, which
overrides the Modelfile) and llama.cpp mode reads `llama-server -c`; longer states are cut in the middle
(`usage.truncated`). For long documents use `llama-server -c 32768` or `--mode gguf`.

Quantisations: `Q8_0` (recommended) and `Q5_K_M` (smaller). Requests of one state are sent one after another, so the
prompt cache of Ollama / llama.cpp computes the state once; `--http-parallel N` (with `OLLAMA_NUM_PARALLEL` /
`llama-server -np`) serves concurrent clients. GGUF is the slow path for batches: it is for desktops and laptops, not
servers. Details: [PORTS.md](PORTS.md), [GGUF.md](GGUF.md). Other GGUF files with the same tokenizer fix (other
quantisations, basal-1.0): `basal-export-gguf` ([GGUF.md](GGUF.md#convert-yourself)).

### Google TPU (experimental, community-maintained)

`--mode tpu` runs the bf16 weights in JAX / XLA on one TPU chip (measured on a v5e-1, the Colab TPU runtime), with
the same SOAM packing, option codes, calibration and API as the basal engine. basal-2.0-mini and basal-2.0 (4.5B) fit
one 16 GB v5e chip; basal-2.0-max (11B) does not. Install a CPU build of torch (the tokenizer and the chat template
use transformers), then the `tpu` extra (JAX and libtpu from PyPI; Python 3.12 or newer):

```bash
uv venv --python 3.12 --allow-existing ~/basal-tpu && source ~/basal-tpu/bin/activate
uv pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cpu
uv pip install "basal[tpu] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"   # 2.2.0: the first release with --mode tpu
basal-serve --model Remek/basal-2.0-4.5B --port 8000   # --mode tpu is the default on a TPU VM
```

- The server compiles one XLA program per shape before it accepts requests (3–5 minutes; cached in
  `~/.cache/basal/jax`, or `$JAX_COMPILATION_CACHE_DIR`, for the next start).
- `--max-len` (default 4,096 tokens) is the longest prompt; longer states are cut or refused (`--context-overflow`).
  Above 3,072 tokens the attention runs in blocks, so long documents fit (basal-2.0 up to its 32k context).
- On a Colab TPU runtime, install into the system Python as above and remove the torch add-ons Colab ships for its own
  older torch (`pip uninstall torchvision torchaudio torchcodec torch-xla`): transformers fails to import next to them.
  [`notebooks/tpu.ipynb`](../notebooks/tpu.ipynb) does all of this and runs the examples and benchmarks.
- Not on TPU: evidence spans (they need the PyTorch model), `--media`, early exit and quantisation.
  Measurements: [HARDWARE.md](HARDWARE.md#google-tpu-experimental-community-maintained).

### CPU

`basal-serve --model Remek/basal-2.0-mini --mode eager --device cpu` runs everywhere, for tests and low volumes.
Requests with several questions or option orders over one state (tasks, `--orders 2`) run faster with
`--mode mps --device cpu`: the packed engine computes the state once, where `eager` computes it once per prompt
(basal-2.0-mini, `--dtype float32`, a 6-core Zen 2: 838 s -> 88 s on three task requests, 75 of 75 top answers the
same; single questions take as long as in `eager`).

### Images and audio: basal-omni (experimental)

Install the `media` extra (pillow, soundfile, scipy, torchvision) and serve a basal-omni model in `eager` mode with
`--media on`:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000   # or Remek/basal-omni-1.0-max
```

A request may then carry images and audio with its state; they go where the state text goes, before it:

```bash
IMG=$(base64 < paragon.jpg | tr -d '\n')
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "", "lang": "pl",
  "media": [{"type": "image", "data": "'"$IMG"'"}],
  "questions": {
    "card": {"type": "noul", "instructions": "Czy zapłacono kartą?"},
    "vat":  {"type": "choice", "instructions": "Jaka jest najwyższa stawka VAT na paragonie?",
             "criteria": {"23": "23%", "8": "8%", "5": "5%", "0": "0% lub zwolniony"}}}}'
```

- Images: any format Pillow reads; `"detail": "high"` reads small print (1,120 image tokens instead of 280; slower).
  Audio: WAV, FLAC, OGG or MP3, at most 30 s a clip. At most 8 items and 20 MB each per request.
- Answers carry `"calibration": "text"`: the temperatures and thresholds were fitted on text, so validate them on your
  own media before automating with them. Text requests to the same server take the usual text path.
- Details, memory and what the engine does differently for Gemma 4: [OMNI.md](OMNI.md). Media results:
  the model cards.
