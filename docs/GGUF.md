# GGUF in process (`--mode gguf`)

`basal-serve --mode gguf` runs a GGUF file in process through [llama.cpp](https://github.com/ggml-org/llama.cpp)
(via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)): Metal on Apple Silicon, CUDA or CPU elsewhere,
without a separate Ollama or `llama-server`. Only the weights come from the GGUF file. The tokenizer, chat template and
`CALIBRATION.json` come from `--model` (a `-GGUF` folder or repository, or the original model), so the token ids are
exactly those of the other backends. The packed rows of a request (the state once, then every question and option
order: [SOAM](SOAM.md)) become llama.cpp *sequences*: every readout has its own sequence, every token belongs to the
sequences of all readouts below it, and all rows of a batch go through one `llama_decode`.

Hybrid models with recurrent-state layers (Qwen3.5's Gated DeltaNet: GGUF architecture `qwen35`; also Mamba, RWKV and
other architectures llama.cpp flags as recurrent or hybrid) decode one prompt per `llama_decode` instead (2.3.2). On
the CPU, llama.cpp returned non-finite probabilities when one decode held several sequences of such a model (every
request with several questions, and every batch of single questions); one prompt at a time is exact on every device,
at the cost of computing a request's state once per question and option order.

This mode was contributed by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)) for basal-1.0 and extended to
the nested prefixes of SOAM in basal 1.5.

## Files

**basal-1.5**: the `-GGUF` repositories (`Q8_0`, `Q4_K_M`). They are converted with the tokenizer fix described in
[PORTS.md](PORTS.md), so llama.cpp also tokenizes basal prompts like the Hugging Face tokenizer; the folder holds the
tokenizer and calibration, so `--model` can point at it:

```bash
hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF
basal-serve --mode gguf --model ./basal-1.5-4.5B-GGUF --gguf basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf --port 8000
```

**basal-1.0**: conversions by Paweł Kiszczak (F16, Q8_0, Q4_K_M, measured below):
[pawelkiszczak/basal-1.0-4.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF) and
[pawelkiszczak/basal-1.0-1.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF). With these files,
`--model` is the original repository (tokenizer and calibration):

```bash
hf download pawelkiszczak/basal-1.0-4.5B-GGUF basal-1.0-4.5B-F16.gguf --local-dir .
basal-serve --mode gguf --model Remek/basal-1.0-4.5B --gguf basal-1.0-4.5B-F16.gguf --port 8000
```

**basal-omni-1.0** (Gemma 4, text only): convert the model with llama.cpp's converter (below; b11312 or later, the
llama.cpp of llama-cpp-python 0.3.36) and pass its directory as `--model`; the server checks the GGUF file against the
`text_config` of its `config.json`. Gates and speeds of F16 to Q4_K_M: [OMNI.md](OMNI.md#gguf-llamacpp-ollama-and-mlx-text).

With a full model directory as `--model`, the server checks that its `config.json` (layers, hidden size, vocabulary)
matches the GGUF file. In this mode `--model` is downloaded without its weights.

## Serve

```bash
# On a Mac, in an environment of the native arm64 Python (INSTALL.md, Apple Silicon; uname -m prints arm64):
#   uv python install cpython-3.12-macos-aarch64-none
#   uv venv --python cpython-3.12-macos-aarch64-none --allow-existing ~/basal-env && source ~/basal-env/bin/activate
# llama-cpp-python builds with Metal on macOS arm64; on Linux the default build runs on the CPU. For CUDA, build it
# first, without uv's cache (a cached CPU build would be reused silently; checked on an RTX 4090: --mode gguf on the GPU):
#   CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --no-cache llama-cpp-python --no-binary llama-cpp-python
uv pip install "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v2.5.4.tar.gz"
basal-bench --model ./basal-1.5-4.5B-GGUF --modes gguf@basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf \
            gguf@basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q4_K_M.gguf
```

Early exit (`fast-exit`) and evidence spans are not available in this mode. The general `[test]` extra does not install
the native llama.cpp build; to run the GGUF tests, opt in: `uv pip install -e ".[test,gguf]" gguf` and
`pytest -q tests/test_gguf.py`.

## Which file

**basal-1.5-mini** (Apple M4 Pro, `basal-bench`, 44 bundled examples, both option orders; reference: bf16 PyTorch on
MPS): `gguf` with `Q8_0` gives the same top option on 44 of 44 decisions, 133 ms per decision against 204 ms for the
bf16 reference; on whole requests (one state, up to 14 questions) the same top answers as the reference without SOAM
(28 of 28, probabilities within 0.03). For
the agreement of `Q8_0` and `Q4_K_M` with bf16 on 500 development decisions (through llama.cpp and Ollama), see
[QUANTIZATION.md](QUANTIZATION.md).

**basal-1.0** (measured by Paweł Kiszczak): Apple M4 Max, 44 bundled examples, both option orders, every engine
measured on prompts it has not seen before after a 60 s GPU cool-down (the MacBook throttles under sustained load);
*dec/s* with the remaining 21 decisions in one call; *TV*: total-variation distance between the averaged two-order
probabilities and the fp32 PyTorch reference (mean / max over the items); `mlx` for comparison.

| model | mode | file | ms per decision | dec/s | agreement | TV mean / max |
|---|---|---|---|---|---|---|
| 4.5B | `gguf` F16 | 9.5 GB | 200 | **5.5** | 1.000 | **0.0006** / 0.004 |
| | `gguf` Q8_0 | 5.1 GB | 212 | 5.1 | 1.000 | 0.0039 / 0.038 |
| | `gguf` Q4_K_M | 2.9 GB | 222 | 4.9 | 0.955 | 0.047 / 0.307 |
| | `mlx` (bf16) | – | **198** | 5.1 | 1.000 | 0.0047 / 0.024 |
| 1.5B | `gguf` F16 | 3.2 GB | 69 | **16.7** | 1.000 | **0.0004** / 0.002 |
| | `gguf` Q8_0 | 1.7 GB | 72 | 15.9 | 0.977 | 0.0052 / 0.029 |
| | `gguf` Q4_K_M | 1.0 GB | 76 | 14.3 | 0.955 | 0.052 / 0.445 |
| | `mlx` (bf16) | – | **67** | 15.8 | 0.977 | 0.0061 / 0.027 |

- **F16** is the closest to the fp32 reference of all reduced-precision paths measured on Apple Silicon (most likely
  because llama.cpp keeps activations in fp32 between operations), at the speed of `mlx`.
- **Q8_0** halves the memory with bf16-level deviations (like `mlx-q8`).
- **Q4_K_M** changes about 5% of the decisions of basal-1.0 and moves single probabilities by up to 0.3–0.45; prefer
  Q8_0 when memory allows.
- Weight quantisation does not make prefill faster here: Apple GPUs are compute-bound on these prompts.

![basal-1.0 Apple Silicon checkpoints: memory vs faithfulness](figures/apple_memory_vs_fidelity.png)

Per-item deviations of every variant and of other engines:
[HARDWARE.md, inference engines on Apple Silicon](HARDWARE.md#inference-engines-on-apple-silicon).

## Convert yourself

`basal-export-gguf` (contributed by Piotr Durlej, [#8](https://github.com/rkinas/basal/pull/8)) runs llama.cpp's
converter and then applies the tokenizer fix of the `-GGUF` repositories
([PORTS.md](PORTS.md#how-the-ports-are-made)), so llama.cpp and Ollama tokenize basal prompts exactly like the Hugging
Face tokenizer. It is for the Llama-architecture models (basal-1.0, basal-1.5, basal-2.0); basal-omni (Gemma 4) is
converted with llama.cpp's converter alone, as in [Files](#files). The converter lives in llama.cpp and needs transformers 5 to read the basal tokenizer config. Its
current `requirements-convert_hf_to_gguf.txt` pins transformers 4, so install the needed converter packages directly,
and basal from a clone with the `gguf-export` extra in the same environment (the tool runs the converter with its own
Python):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
git clone https://github.com/rkinas/basal ~/basal
# On a Mac, the native arm64 Python in place of the next line (torch has no build for an x86_64 one under Rosetta):
#   uv python install cpython-3.12-macos-aarch64-none
#   uv venv --python cpython-3.12-macos-aarch64-none --allow-existing ~/gguf-env
uv venv --python 3.12 --allow-existing ~/gguf-env
uv pip install --python ~/gguf-env 'numpy~=2.2.6' 'sentencepiece>=0.1.98,<0.3.0' 'protobuf>=4.21,<5' 'transformers==5.17.0' 'torch==2.11.0'
uv pip install --python ~/gguf-env -e "$HOME/basal[gguf-export]"
hf download Remek/basal-2.0-4.5B --local-dir basal-2.0-4.5B
~/gguf-env/bin/basal-export-gguf basal-2.0-4.5B basal-2.0-4.5B-F16.gguf  --outtype f16  --llama-cpp ~/llama.cpp
~/gguf-env/bin/basal-export-gguf basal-2.0-4.5B basal-2.0-4.5B-Q8_0.gguf --outtype q8_0 --llama-cpp ~/llama.cpp
# other llama.cpp quantisations from the F16 file keep the fix (llama-quantize: brew install llama.cpp, or build llama.cpp)
llama-quantize basal-2.0-4.5B-F16.gguf basal-2.0-4.5B-Q5_K_M.gguf Q5_K_M
```

`--gguf` fixes an existing file instead of converting the model: a stock conversion, a `llama-quantize` output, or
the basal-1.0 files of [Files](#files), which carry the stock tokenizer. The model then only supplies
`tokenizer.json`, downloaded without the weights; `--llama-cpp` is not needed and the output must be a new file:

```bash
~/gguf-env/bin/basal-export-gguf Remek/basal-1.0-4.5B fixed/basal-1.0-4.5B-F16.gguf --gguf basal-1.0-4.5B-F16.gguf
```

The attention and MLP biases of basal are converted and used by llama.cpp's Llama architecture. A stock conversion
keeps llama.cpp's own tokenization of text (below): fine for `--mode gguf` and `--mode llamacpp`, which send token
ids, not for Ollama, which takes text. With the fix, `llama-tokenize` splits the bundled examples in both option
orders exactly like the Hugging Face tokenizer (stock conversions: none of them). The fixed tokenizer fields of a
stock conversion equal those of the `-GGUF` files for all three basal-2.0 and basal-1.5 models, and fixing a `-GGUF`
file leaves it unchanged, byte for byte.

## Decision metadata for LM Studio and llama-server (`--systemone`)

llama.cpp's server answers `POST /v1/systemone` itself for a GGUF file that carries decision metadata; LM Studio 0.4.26
and newer serves such a file as a decision model on its own `/v1/systemone` and on `/v1/decisions` (OpenAI's Decisions
API). `basal-export-gguf --systemone` writes that metadata into a copy of a basal `-GGUF` file (the tensors are not
touched; [INSTALL.md](INSTALL.md#lm-studio-gguf-as-a-decision-model) has the LM Studio steps):

```bash
basal-export-gguf ./basal-2.0-4.5B-GGUF lmstudio/basal-2.0-4.5B-Q8_0.gguf \
    --gguf basal-2.0-4.5B-GGUF/basal-2.0-4.5B-Q8_0.gguf --systemone
```

| key | value |
|---|---|
| `<arch>.decision.type` | `openjev`: one prompt per question, the logits of the option codes `A`-`Z` at the last prompt token, a softmax divided by the temperature. This is basal's prompt-format-2 readout for up to 26 options |
| `tokenizer.chat_template.systemone` | basal's prompt as a template: the fixed system prompt, `State:` / `Question:` / `Options:` with the codes, in the model's chat template, plus the answer prefill `{"answer": "`. Polish or English, as `basal-serve` picks; JSON states and descriptions as compact JSON. LM Studio's `/v1/decisions` hands OpenAI choices over as `{"value", "description"}` objects and score levels as `{"label", "description"}`: they are shown as the System One request `{value: description}` would be |
| `<arch>.decision.temperature.<type>.<bucket>` | the temperatures of `CALIBRATION.llamacpp.json` (else `CALIBRATION.gguf.json`, else `CALIBRATION.json`; or `--calibration`) on llama.cpp's option-count buckets 2, 3-5, 6-10 and 11+. basal's buckets 2-10 and 11-26 map onto them exactly; a calibration that would need two temperatures in one bucket is refused |

Before writing, the tool renders a fixed set of requests with the template and with `basal-serve`'s own prompt code
and stops if one differs, or if the codes `A`-`Z` are not the single tokens llama.cpp reads. The tokenizer fields get
the fix above as well (for the basal-omni files: the token types of a few added tokens as `tokenizer.json` has them);
on 620 prompts per model, llama.cpp's tokens of the rendered prompt equal the Hugging Face ones. Prompt-format-1 models
(basal-1.x: two option orders), other layouts and two-way models are refused. Limits of this route:

- **26 options per question.** llama.cpp labels options 27-52 `a`-`z`, basal codes them `AA`, `AB`, ...; the template
  refuses a question with more than 26 options (or fewer than 2) with an HTTP error, never with a wrong answer.
- **choice, noul and score only**, one option order, the keys always shown (no `option_keys: "hide"`), the language
  picked from the text. `act`, `multi`, `tool`, `coverage`, `max_error`, the certified thresholds, `facts` and
  `ask_if` need `basal-serve` (`--mode gguf` or `--mode llamacpp` on the same file).
- **Calibration**: the temperatures above, nothing else. For your own data, fit them with `basal-calibrate` on the same
  file in process (`--model ./basal-2.0-4.5B-GGUF --mode gguf --gguf <file>`) or against
  `basal-serve --mode gguf --no-calibration` with `--url`, then write them with `--systemone --calibration
  CALIBRATION.custom.json`.

llama-server serves the same file: `llama-server -m lmstudio/basal-2.0-4.5B-Q8_0.gguf -c 16384`, then
`POST /v1/systemone` with `state` and `questions` as for `basal-serve`.

## Other llama.cpp front ends: send token ids

In a stock conversion, llama.cpp tokenizes text with the vocabulary stored in the GGUF file, and for basal that
tokenization differs from the Hugging Face tokenizer the models were trained with: on the bundled examples every prompt
is split differently (llama.cpp puts a word-boundary marker before the role names after `<|im_start|>` and chooses other
merges inside Polish words, e.g. `wybierają|c` instead of `wybiera|jąc`). Front ends that send text therefore shift the
probabilities. On the basal-1.0-1.5B F16 file, `llama-server` with text prompts: mean total-variation distance to fp32
0.061 (max 0.35), with the Hugging Face token ids: 0.0005 (max 0.002). Ollama (GGUF import) and LM Studio, which accept
only text, show the same shift (0.049 / 0.31 on the 1.5B, 0.061 / 0.63 for LM Studio on the 4.5B). The basal-1.5
and basal-2.0 `-GGUF` files carry the tokenizer fix, so text and token ids give the same tokens there, as do files
written by `basal-export-gguf` ([Convert yourself](#convert-yourself)).

`--mode gguf` and `--mode llamacpp` always pass token ids. If you call `llama-server` yourself, tokenize with the
Hugging Face tokenizer, send `"prompt": [ids...]` to `/completion` with `n_predict: 1`, `n_probs: 20` and read the
letter ids from `completion_probabilities[0].top_logprobs`.
