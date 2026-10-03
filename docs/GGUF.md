# GGUF / llama.cpp

`basal-serve --mode gguf` runs a basal checkpoint converted to GGUF through [llama.cpp](https://github.com/ggml-org/llama.cpp)
(via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)): Metal on Apple Silicon, CUDA or CPU elsewhere.
Only the weights come from the GGUF file. The tokenizer, chat template and `CALIBRATION.json` still come from `--model`
(the Hugging Face repository or a local copy), so the token ids are exactly those of the other backends. The two option
orders share their prefix as llama.cpp *sequences*: the prefix tokens belong to the sequences of both orders, each
order's option block to its own sequence, and all questions of a batch go through one `llama_decode`.

## Download

Converted files (F16, Q8_0, Q4_K_M, measured below; these repos are **private until the publisher releases them**):
[pawelkiszczak/basal-1.0-4.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF) and
[pawelkiszczak/basal-1.0-1.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF).

```bash
hf download pawelkiszczak/basal-1.0-4.5B-GGUF basal-1.0-4.5B-F16.gguf --local-dir .
```

## Convert yourself

`basal-export-gguf` runs llama.cpp's converter and then fixes the tokenizer of the file, so that llama.cpp splits basal
prompts exactly like the Hugging Face tokenizer (see [Other llama.cpp front ends](#other-llama-cpp-front-ends-send-token-ids));
with `--ollama-decision` it also writes the decision metadata [Ollama](#ollama-v1systemone) uses:

```bash
uv pip install -e ".[gguf-export]"
basal-export-gguf Remek/basal-1.0-4.5B basal-1.0-4.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp
```

The converter alone (below) gives correct weights, but a tokenizer that differs in text front ends.

The converter lives in llama.cpp and needs transformers 5 to read basal-1.0-1.5B's tokenizer config. Its current
`requirements-convert_hf_to_gguf.txt` pins transformers 4, so install the needed converter packages directly:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
uv venv --python 3.12 ~/gguf-env
uv pip install --python ~/gguf-env 'numpy~=2.2.6' 'sentencepiece>=0.1.98,<0.3.0' 'protobuf>=4.21,<5' gguf 'transformers==5.17.0' 'torch==2.11.0'
hf download Remek/basal-1.0-4.5B --local-dir basal-1.0-4.5B
~/gguf-env/bin/python ~/llama.cpp/convert_hf_to_gguf.py basal-1.0-4.5B --outtype f16  --outfile basal-1.0-4.5B-F16.gguf
~/gguf-env/bin/python ~/llama.cpp/convert_hf_to_gguf.py basal-1.0-4.5B --outtype q8_0 --outfile basal-1.0-4.5B-Q8_0.gguf
# other llama.cpp quantisations from the F16 file (llama-quantize: brew install llama.cpp, or build llama.cpp)
llama-quantize basal-1.0-4.5B-F16.gguf basal-1.0-4.5B-Q4_K_M.gguf Q4_K_M
```

The attention and MLP biases of basal-1.0 are converted and used by llama.cpp's Llama architecture.

## Serve

```bash
uv pip install -e ".[gguf]"          # llama-cpp-python 0.3.35 builds with Metal on macOS arm64
# For CUDA, use this INSTEAD of the line above (not tested with basal):
# CMAKE_ARGS="-DGGML_CUDA=on" uv pip install -e ".[gguf]" --no-binary llama-cpp-python
basal-serve --mode gguf --model Remek/basal-1.0-4.5B --gguf basal-1.0-4.5B-F16.gguf --port 8000
basal-bench --model Remek/basal-1.0-4.5B --modes eager-fp32 gguf@basal-1.0-4.5B-F16.gguf gguf@basal-1.0-4.5B-Q8_0.gguf
```

Early exit (`fast-exit`) is not available in this mode.

The general `[test]` extra does not install the native llama.cpp build. To run the GGUF-specific tests, opt in:

```bash
uv pip install -e ".[test,gguf]" gguf
pytest -q tests/test_gguf.py
```

For MLX-native checkpoints of the same models (8-bit, oQ6e; `--mode mlx`), see the
[README](../README.md#apple-silicon-mlx--mps).

## Which file

Apple M4 Max, 44 bundled examples, both option orders, every engine measured on prompts it has not seen before
after a 60 s GPU cool-down (the MacBook throttles under sustained load); *dec/s* with the remaining 21 decisions in one
call; *TV*: total-variation distance between the averaged two-order probabilities and the fp32 PyTorch reference
(mean / max over the items); `mlx` for comparison.

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
  because llama.cpp keeps activations in fp32 between operations), at the speed of `mlx`: the recommended file.
- **Q8_0** halves the memory with bf16-level deviations (like `mlx-q8`).
- **Q4_K_M** changes about 5% of the decisions and moves single probabilities by up to 0.3–0.45: not recommended.
- Weight quantisation does not make prefill faster here: Apple GPUs are compute-bound on these prompts.

![Selected Apple Silicon checkpoints: memory vs faithfulness](figures/apple_memory_vs_fidelity.png)

Per-item deviations of every variant and of other engines:
[HARDWARE.md, inference engines on Apple Silicon](HARDWARE.md#inference-engines-on-apple-silicon).

## Ollama `/v1/systemone`

Ollama serves typed decisions on `/v1/systemone`. A GGUF written by `basal-export-gguf --ollama-decision` carries
`<arch>.decision.type = "basal"` and the calibrated temperatures of `CALIBRATION.json`
(`<arch>.decision.temperature.{choice,noul,score}`). An Ollama build with the `basal` decision encoding
(proposed in [ollama/ollama#18760](https://github.com/ollama/ollama/issues/18760), branch [pdurlej/ollama@decision-basal-encoding](https://github.com/pdurlej/ollama/tree/decision-basal-encoding)) then renders each question in basal's own prompt format, scores both option orders, averages them and
applies the temperature, like `basal-serve`. No Modelfile template or system prompt is needed.

- Such a file is **decision-only** in Ollama: `/api/generate` and `/api/chat` refuse it. Ollama versions without the
  `basal` encoding reject it on `/v1/systemone` (`unsupported decision encoding "basal"`). Export without
  `--ollama-decision` for any other use.
- Ollama's request schema has no `option_keys`: described choices are always shown as `key: description`
  (`basal-serve`'s default), never in the keys-hidden training format.
- A structured `state` (object or array) is passed as sent, with `, ` / `: ` separators; number spellings and
  duplicate keys are not normalized as Python's `json.dumps` would.

```bash
basal-export-gguf Remek/basal-1.0-1.5B basal-1.0-1.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp --ollama-decision
echo 'FROM ./basal-1.0-1.5B-F16.gguf' > Modelfile
ollama create basal-1.0-1.5b -f Modelfile
curl -s localhost:11434/v1/systemone -d '{"model": "basal-1.0-1.5b", "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej.", "questions": {"dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?", "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej"}}}}'
```

Measured on an Apple M1 Max, 50 private triage items (30 PL / 20 EN; four questions each: a 2-way `choice`, two
`noul`, a 4-level `score`), against `basal-serve --mode eager` (MPS, bf16, both orders, calibrated) on the same items.
*TV*: total-variation distance per question between the two servers' probabilities (mean / max over 200 questions);
latency per item (4 questions).

| model | file | portfolio (`choice`) | escalate (`noul`) | data class (`noul`) | urgency (`score`) | TV mean / max | s per item |
|---|---|---|---|---|---|---|---|
| 1.5B | `basal-serve` reference | 0.92 | 0.86 | 0.70 | 0.38 | – | 1.65 |
| | Ollama F16 | 0.94 | 0.84 | 0.68 | 0.38 | 0.008 / 0.038 | 0.58 |
| | Ollama Q8_0 | 0.94 | 0.86 | 0.68 | 0.38 | 0.010 / 0.041 | 0.63 |
| 4.5B | `basal-serve` reference | 0.96 | 0.88 | 0.60 | 0.46 | – | 4.76 |
| | Ollama F16 | 0.96 | 0.88 | 0.62 | 0.46 | 0.008 / 0.164 | 1.61 |
| | Ollama Q8_0 | 0.96 | 0.88 | 0.60 | 0.46 | 0.010 / 0.168 | 1.71 |

The reference ran in `eager` mode (no compilation), so the speed column compares engines on this machine only.

Without the encoding, Ollama's generic decision prompt (a JSON schema, one option order, no calibration) costs
basal-1.0-1.5B (Q8_0, fixed tokenizer, ChatML template with basal's system prompt) up to 0.24 accuracy on the same items (data-class `noul` 0.70 → 0.46, `score`
0.38 → 0.18): the models are trained on one prompt format.

## Other llama.cpp front ends: send token ids

llama.cpp tokenizes text with the vocabulary stored in the GGUF file, and for basal that tokenization differs from the
Hugging Face tokenizer the models were trained with: on the bundled examples every prompt is split differently
(llama.cpp puts a word-boundary marker before the role names after `<|im_start|>` and chooses other merges inside Polish
words, e.g. `wybierają|c` instead of `wybiera|jąc`). Front ends that send text therefore shift the probabilities. On
the 1.5B F16 file, `llama-server` with text prompts: mean total-variation distance to fp32 0.061 (max 0.35), with
the Hugging Face token ids: 0.0005 (max 0.002). Ollama (GGUF import) and LM Studio, which accept only text, show the
same shift (0.049 / 0.31 on the 1.5B, 0.061 / 0.63 for LM Studio on the 4.5B).
Files written by `basal-export-gguf` remove the difference for basal prompts: on 200 prompts of the item set above
(50 items × 4 questions), the `/tokenize` endpoint of llama-server returns exactly the Hugging Face token ids for every
prompt, for both 1.5B and 4.5B (a file from the converter alone: for none). This holds for text that starts with a
special token, as every basal prompt does with `<s>`; at the very start of plain text Hugging Face adds a word-boundary
marker and llama.cpp does not. The converter writes basal's BPE vocabulary as a SentencePiece vocabulary with one constant score for every
token and with a word-boundary prefix; `basal-export-gguf` sets each token's score from the rank of the BPE merge that
creates it (llama.cpp merges the highest score first) and turns the prefix off.
`--mode gguf` always passes token ids. If you call `llama-server` yourself, tokenize with the Hugging Face tokenizer,
send `"prompt": [ids...]` to `/completion` with `n_predict: 1`, `n_probs: 20` and read the letter ids from
`completion_probabilities[0].top_logprobs`.
