# GGUF in process (`--mode gguf`)

`basal-serve --mode gguf` runs a GGUF file in process through [llama.cpp](https://github.com/ggml-org/llama.cpp)
(via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)): Metal on Apple Silicon, CUDA or CPU elsewhere,
without a separate Ollama or `llama-server`. Only the weights come from the GGUF file. The tokenizer, chat template and
`CALIBRATION.json` come from `--model` (a `-GGUF` folder or repository, or the original model), so the token ids are
exactly those of the other backends. The packed rows of a request (the state once, then every question and option
order: [SOAM](SOAM.md)) become llama.cpp *sequences*: every readout has its own sequence, every token belongs to the
sequences of all readouts below it, and all rows of a batch go through one `llama_decode`.

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

With a full model directory as `--model`, the server checks that its `config.json` (layers, hidden size, vocabulary)
matches the GGUF file. In this mode `--model` is downloaded without its weights.

## Serve

```bash
# llama-cpp-python builds with Metal on macOS arm64; on Linux the default build runs on the CPU. For CUDA, build it
# first, without uv's cache (a cached CPU build would be reused silently; checked on an RTX 4090: --mode gguf on the GPU):
#   CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --no-cache llama-cpp-python --no-binary llama-cpp-python
uv pip install "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v2.1.0.tar.gz"
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

`basal-export-gguf` runs llama.cpp's converter and then applies the tokenizer fix of the `-GGUF` repositories
([PORTS.md](PORTS.md#how-the-ports-are-made)), so llama.cpp and Ollama tokenize basal prompts exactly like the Hugging
Face tokenizer. The converter lives in llama.cpp and needs transformers 5 to read the basal tokenizer config. Its
current `requirements-convert_hf_to_gguf.txt` pins transformers 4, so install the needed converter packages directly,
and basal from a clone with the `gguf-export` extra in the same environment (the tool runs the converter with its own
Python):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
git clone https://github.com/rkinas/basal ~/basal
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
