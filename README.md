# basal

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--1.0%20collection-yellow)](https://huggingface.co/collections/Remek/basal-10-6ab8224bf7bd8732d7a6117d)
[![Technical report](https://img.shields.io/badge/technical%20report-PDF-b31b1b.svg)](docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)

![basal-1.0 overview](assets/basal.png)

Inference engine for **basal-1.0** — small, fast, calibrated *typed-decision* models for Polish (and English).

**What it is.** basal-1.0 is inspired by the *System 1* (fast, intuitive) decision models such as Jev: instead of a
chatbot that writes an answer, the model reads a **state** (a message, a document, a case file, a web page as JSON)
and answers a **typed question** about it by returning a **probability for each allowed answer** — in one forward
pass, without generating any text. The answer can therefore never fall outside the options you gave, and the
probability says how sure the model is.

**What it is for: a dynamic classifier.** You describe the classes *in the request* — in plain language, per call —
instead of training a classifier for them. The same model routes tickets today, checks a filing deadline tomorrow and
scores the urgency of an incident report next week, each time with new options and no retraining. This makes it a
drop-in, very fast replacement for:

- ticket, e-mail and document **routing** with categories that change often;
- **rule and policy checks** on a document ("is the claim covered?", "was the appeal filed in time?");
- **scoring** on ordered scales (urgency, risk, satisfaction);
- **agent decisions** (which tool, which next step, which element to click) and **guard checks** in LLM pipelines,
  where a full LLM call is too slow or too expensive;
- **triage with a confidence threshold**: accept confident decisions automatically and send the rest to a person.

**How fast.** One decision (both option orders, calibrated) takes **8.8 ms on a B300, 12.5 ms on an H100, 27 ms on an
RTX 5090 and 45 ms on a desktop DGX Spark (FP8)**; the 1.5B model is about twice as fast. On Polish decisions it is more
accurate than the commercial Jev API and eleven open decision models (see [Quality](#quality)).

A question has one of three types:

| type     | answer                                              | example                                   |
|----------|-----------------------------------------------------|-------------------------------------------|
| `choice` | distribution over named options                     | route a ticket, pick the applicable rule  |
| `noul`   | probability of *yes* (with optional descriptions)   | "was the appeal filed on time?"           |
| `score`  | distribution over ordered levels + expected level   | urgency 0–3, sentiment scale              |

The HTTP API implements the *System One* JSON interface (`POST /v1/systemone`, `GET /v1/models`); responses validate
against the official OpenAPI schema (`tests/test_server.py`). Clients using this subset work by changing the base URL
(the official SDK was not tested; at most 10 options per question).

> The name comes from the *basal ganglia* — the part of the brain that selects one action among competing options.

## Models

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-1.0-4.5B** | 4.5B | main model, highest quality | `Remek/basal-1.0-4.5B` (bf16, includes early-exit heads) |
| basal-1.0-4.5B-FP8 / -NVFP4 | 4.5B | ModelOpt checkpoints for vLLM (FP8: Hopper / Blackwell; NVFP4: Blackwell) | `Remek/basal-1.0-4.5B-FP8`, `Remek/basal-1.0-4.5B-NVFP4` |
| **basal-1.0-1.5B** | 1.5B | *lite*: 2× faster, −2.9 points on the full held-out test (−3.5 on Polish decisions with the released engine) | `Remek/basal-1.0-1.5B` |
| basal-1.0-1.5B-FP8 / -NVFP4 | 1.5B | ModelOpt checkpoints for vLLM | `Remek/basal-1.0-1.5B-FP8`, `Remek/basal-1.0-1.5B-NVFP4` |

Both models are fine-tuned from Apache-2.0 base models (see [NOTICE](NOTICE)) on Polish and English decision data whose
labels are computed by code, grounded in statutes or checked by independent verifiers, and are calibrated per question
type (temperatures stored in `CALIBRATION.json` and applied by the server).

## Quality

Accuracy, both option orders averaged. *PL decisions*: 7,081 held-out Polish decisions (unseen templates, statutes and
domains); *PL general*: Polish knowledge, exams, reading comprehension; *EN decisions*: 1,479 held-out English
decisions; *Public bench.*: the 231-item public English decision benchmark (official harness). basal-1.0 was served with
this engine (`--mode fast`), open systems with their own official servers, all on one H100. Full table with all eleven
open systems and speeds: [jev-pl-benchmark](https://huggingface.co/spaces/Remek/jev-pl-benchmark). The basal public-benchmark scores are measured with engine v1.0.1, which shows option keys next to their
descriptions by default (the benchmark's options have meaningful keys); with v1.0 they were 0.706 (4.5B) and 0.662 (1.5B).

| system | params | PL decisions | PL general | EN decisions | Public bench. |
|---|---|---|---|---|---|
| **basal-1.0-4.5B** | 4.5B | **0.884** | 0.737 | 0.741 | 0.740 |
| basal-1.0-1.5B | 1.5B | 0.849 | 0.656 | 0.734 | 0.675 |
| Jev 1.13.0 (commercial API) | – | 0.780 | – | 0.736 | 0.861 |
| Cygnet | 12B | 0.688 | 0.793 | 0.703 | **0.879** |
| AutoJev-27B | 27B | 0.779 | **0.833** | **0.753** | 0.870 |
| Jev-Omni | 12B | 0.687 | 0.768 | 0.694 | 0.866 |
| JevK5 v0.2 | 4B | 0.630 | 0.744 | 0.670 | 0.857 |
| Winnow-12B | 12B | 0.688 | 0.772 | 0.703 | 0.853 |
| decider-4b v2 | 4B | 0.709 | 0.717 | 0.694 | 0.835 |
| decider-35B-A3B | 35B (3B active) | 0.694 | 0.781 | 0.751 | 0.831 |
| Hopper | 4B | 0.649 | 0.727 | 0.669 | 0.823 |
| reflex-4B | 4B | 0.586 | 0.729 | 0.645 | 0.814 |
| nimble-9B v2 | 9B | 0.685 | 0.758 | 0.669 | 0.805 |
| kev-4B | 4B | 0.694 | 0.690 | 0.666 | 0.758 |

basal-1.0 is a **Polish specialist**: best on Polish decisions, 10.5 points above the best open system, on English decisions not
distinguishable from Jev 1.13.0 or from the best open systems (differences of −1.2 to +0.5 points, all within their 95%
intervals), and weaker on the general-purpose English benchmark, which it was
not trained for. With the confidence threshold shipped in `CALIBRATION.json` (fixed on calibration data before testing,
target 1% error) it decides **58.6%** of the held-out test decisions (8,560 Polish and English items) automatically, at
1.2% observed error; Jev 1.13.0 under the same procedure: 18.1%. Polish-decision scores use the corrected notice-period
labels of the technical report (v1.0.1); the checkpoints still give the old answer on those items (see Limitations).

## Speed

One decision = **both option orders** (the default; reduces sensitivity to option order). Batch size 1, median latency; throughput with
32 option-order passes per forward. Offline numbers from `basal-bench` (H100, RTX PRO 6000 and RTX 5090 with the
equivalent research harness), HTTP numbers from `basal-loadtest` (RTX PRO 6000 and RTX 5090: research server), all on the same private 500-item test
sample.

| GPU | class | `fast` (bf16) | `fp8` | HTTP `fast` |
|---|---|---|---|---|
| B300 SXM6 | server (Blackwell) | **8.8 ms**, 109 dec/s | 9.7 ms, 102 dec/s | **9.7 ms p50, 109 dec/s** |
| H100 80GB | server | **12.5 ms**, 63 dec/s | 11.3 ms, 80 dec/s | 14.1 ms p50, 63 dec/s |
| RTX PRO 6000 Blackwell | workstation | 18.9 ms, 39 dec/s | 14.9 ms, 58 dec/s | 22.5 ms p50, 39 dec/s |
| RTX 5090 | consumer | 27.3 ms, 24 dec/s | 19.4 ms, 40 dec/s | 32.3 ms p50, 23 dec/s |
| DGX Spark (GB10) | desktop | 92.0 ms, 7 dec/s | **44.6 ms**, 8 dec/s | – |
| basal-1.0-1.5B on B300 | server | **4.7 ms**, 250 dec/s | 5.2 ms, 224 dec/s | – |
| basal-1.0-1.5B on H100 | server | 6.2 ms, 157 dec/s | 6.3 ms, 177 dec/s | 7.7 ms p50, 147 dec/s |
| basal-1.0-1.5B on DGX Spark | desktop | 34.3 ms, 20 dec/s | **18.1 ms**, 31 dec/s | – |

`fast` keeps decisions practically identical to the fp32 reference (argmax agreement 0.99–1.00); `fp8` changes about
2–4% of decisions. Which mode is fastest depends on the bottleneck of the card: on the DGX Spark (memory-bandwidth-bound) FP8
halves latency, on workstation and consumer cards it gives 1.3–1.4×, and on the B300 bf16 is already fastest. NVFP4
(4-bit) costs this model about 3 accuracy points and is meant only for batched high-throughput serving (vLLM on the
DGX Spark: 27 instead of 8 decisions/s). See [docs/HARDWARE.md](docs/HARDWARE.md) for every GPU we measured: B300, H100,
RTX PRO 6000, RTX 5090, RTX 4090 (1.5B only) and DGX Spark (GB10).

## Quick start

Install with [uv](https://docs.astral.sh/uv/) into a fresh environment (no git needed; `pip install uv` if it is
missing):

```bash
uv venv --python 3.12 ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz"
basal-serve --model Remek/basal-1.0-4.5B --mode fast --port 8000
```

- **Install torch first, from the CUDA 12.8 index.** A plain `pip install basal` takes the newest torch from PyPI,
  which may be built for a newer CUDA than your GPU driver ("The NVIDIA driver on your system is too old"). The cu128
  build needs a driver that supports CUDA 12.8 or newer (`nvidia-smi` shows it top right).
- **Use a fresh environment on cloud GPU images** (RunPod, Lambda, …). Their system Python ships a `torchvision` built for
  another torch, which breaks `transformers` ("operator torchvision::nms does not exist"). basal does not need
  torchvision; in a fresh environment it is not installed.
- The first start in mode `fast` compiles the model (a few minutes); `--mode fast-nocompile` starts in seconds.
- From a clone instead: `git clone https://github.com/rkinas/basal && cd basal && uv pip install -e ".[fp8]"`
  (after the torch line above).

### Apple Silicon (MLX / MPS)

On a Mac with an M-series chip no CUDA index is needed: the PyPI torch wheel includes MPS, and the `mlx` extra adds
[MLX](https://github.com/ml-explore/mlx). Both backends are on the fork's `main` branch, not yet in
[`rkinas/basal`](https://github.com/rkinas/basal). Install from the fork:

```bash
git clone --branch main https://github.com/pawelkiszczak/basal && cd basal
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mlx,gguf]"
basal-serve --model Remek/basal-1.0-4.5B --port 8000        # default on Apple Silicon: --mode mlx
```

Ready-made Apple Silicon checkpoints (the converted repos linked below are **private until their publisher releases
them**; the original bf16 repos are public). Measured on an M4 Max; all run at about the same speed, but differ in
memory and fidelity ([details](docs/HARDWARE.md#quantised-formats-mlx-omlx-oq-gguf)):

| use | 4.5B | 1.5B | start with |
|---|---|---|---|
| default (bf16) | [Remek/basal-1.0-4.5B](https://huggingface.co/Remek/basal-1.0-4.5B) (9.5 GB) | [Remek/basal-1.0-1.5B](https://huggingface.co/Remek/basal-1.0-1.5B) (3.2 GB) | `--mode mlx --model <repo>` |
| closest to fp32 | [GGUF F16](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF) (9.5 GB) | [GGUF F16](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF) (3.2 GB) | `--mode gguf --model Remek/basal-1.0-<size> --gguf <file>` |
| half the memory | [MLX 8-bit](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-MLX-8bit) (5.1 GB) or [GGUF Q8_0](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF) (5.1 GB) | [MLX 8-bit](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-MLX-8bit) (1.7 GB) or [GGUF Q8_0](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF) (1.7 GB) | `--mode mlx --model <repo>` / `--mode gguf` |
| least memory at the agreement of bf16 | [oQ6e](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-oQ6e) (4.0 GB) | [oQ6e](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-oQ6e) (1.3 GB) | `--mode mlx --model <repo>` |

The MLX repositories include `CALIBRATION.json`, so `--model pawelkiszczak/basal-1.0-4.5B-MLX-8bit` is all the server
needs; for GGUF, `--model` stays the original repository (tokenizer and calibration) and `--gguf` points to the
downloaded file ([docs/GGUF.md](docs/GGUF.md)). GGUF Q4_K_M is staged in a private repo but not recommended: it
changes some decisions. The other measured 4-bit formats are not staged.

![Selected Apple Silicon checkpoints: memory vs faithfulness](docs/figures/apple_memory_vs_fidelity.png)

- `mlx` (default when MLX and mlx-lm are installed) and `mps` (PyTorch) both use the shared prefix and batching of
  `fast`. Neither compiles; startup still includes model download and loading. Agreement with the fp32 reference on the
  44 bundled examples: 1.000 for the 4.5B in `mlx` and `mlx-q8`, 0.977 (one item) for `mps` and for the 1.5B.
- **Memory.** The 4.5B model needs about 9 GB of weights in bf16; on a 16 GB Mac use one of the 8-bit or oQ6e
  checkpoints above, `--mode mlx-q8` (8-bit weights quantised at load time, 4.8 GB) or the 1.5B model.
  On a 16 GB Mac, benchmark with `--modes mps mlx` instead of `eager-fp32` (roughly 18 GB of weights).
- **Speed** (M4 Max, both option orders, bundled examples, cooled GPU): 4.5B 198 ms per decision (`mlx`), 1.5B 67 ms;
  HTTP p50 208 ms for the 4.5B. Apple GPUs are compute-bound on these prompts, so quantised weights save memory but
  not time. See [docs/HARDWARE.md](docs/HARDWARE.md#apple-silicon).
- **GGUF / llama.cpp**: `--mode gguf --gguf <file.gguf>` runs a GGUF checkpoint through llama.cpp (Metal here, CUDA or
  CPU elsewhere); F16 is the closest to fp32 of all reduced-precision paths, Q8_0 halves the memory. Files:
  [pawelkiszczak/basal-1.0-4.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-4.5B-GGUF),
  [pawelkiszczak/basal-1.0-1.5B-GGUF](https://huggingface.co/pawelkiszczak/basal-1.0-1.5B-GGUF); see
  [docs/GGUF.md](docs/GGUF.md).
- **Ollama `/v1/systemone`**: a GGUF from `basal-export-gguf --ollama-decision` is served natively by Ollama's typed-decision API
  (basal's prompt format, both option orders, calibrated temperatures) with an Ollama build that has the `basal`
  decision encoding; see [docs/GGUF.md](docs/GGUF.md#ollama-v1systemone).
- **Ollama**: `--mode ollama --ollama-model <name>` reads letter logprobs from a *safetensors import of the original
  checkpoint*, not an MLX/GGUF conversion. Ollama takes text rather than token IDs; it can omit a letter from its
  top-20 list, in which case basal reports an error instead of returning invented probabilities. See
  [Ollama setup and limitations](docs/HARDWARE.md#ollama-safetensors-import).
- `fast*`, `fp8`, `nvfp4` and `fast-exit` need CUDA; early exit is not available on Apple backends. `vllm` also runs
  on [vllm-metal](https://github.com/vllm-project/vllm-metal) with a patched `config.json`
  ([engine comparison](docs/HARDWARE.md#inference-engines-on-apple-silicon)).
  Quantisation overrides are backend-specific (`--quant q8` for `mlx`; `--quant fp8` / `nvfp4` for CUDA graph modes).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {"dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
    "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}}}}'
```

Response (real output, H100, mode `fast`; probabilities in the examples of this README were produced with the v1.0
engine and calibration; the v1.0.1 temperatures change them slightly but never change the chosen answer, and v1.0.1
shows option keys by default (see *Option keys* below; add `"option_keys": "hide"` for the v1.0 prompt), which can
change the probabilities):

```json
{
 "model": "basal-1.0-4.5B",
 "answers": {
  "dept": {
   "type": "choice",
   "choice": "online",
   "probabilities": {
    "cards": 0.0004,
    "online": 0.9992,
    "loans": 0.0004
   },
   "confidence": 0.9992
  }
 },
 "usage": {
  "input_tokens": 284,
  "output_tokens": 0,
  "questions": 1,
  "latency_ms": 10.49
 }
}
```

Python:

```python
from basal.client import Basal
b = Basal("http://127.0.0.1:8000")
a = b.score("Zgłoszenie z oddziału w Gdańsku: od 7:40 nie działa żaden terminal płatniczy, klienci odchodzą od kas, "
            "kolejka na 30 osób. Obejście: tylko gotówka.",
            "Jak pilne jest to zgłoszenie?",
            ["niska — można zaplanować", "średnia — w ciągu kilku dni", "wysoka — dziś", "krytyczna — natychmiast"])
print(round(a["score"], 2), a["probabilities"])
# real output (rounded): 2.71 {'0': 0.005, '1': 0.003, '2': 0.269, '3': 0.722}
# (expected level 2.71 of 0-3: most likely "krytyczna" 0.72, "wysoka" 0.27)
```

## Inference on a JSONL file

`basal-run` sends every line of a JSONL file to a running server (concurrently; the server batches the requests) and
writes one answer per line, in the same order. Start a server first (`basal-serve ...`), then:

```bash
basal-run --input basal/examples/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
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
Ready-to-run examples (Polish and English) in `basal/examples/` (in a clone of this repository; they are also
installed with the package):

| file | what it shows |
|---|---|
| `choice.jsonl` | routing, document type, amounts, policy rules, sentiment, next step — one option out of 3–4 (with `gold`) |
| `noul.jsonl` | yes/no decisions: deadlines, approval thresholds, phishing, refunds, missing information, alerts (with `gold`) |
| `score.jsonl` | ordered scales: urgency, satisfaction, fraud risk, answer correctness; the answer includes the expected level |
| `complex.jsonl` | full requests: several typed questions about one JSON state (fan-out), a web-agent step with structured options, a deadline decomposed into simple questions, loan triage, a prompt-injection guard |
| `questions.jsonl` | 20 mixed items; with `choice`, `noul` and `score` the default set of `basal-bench` and `basal-loadtest` |

```bash
for f in choice noul score complex; do basal-run --input basal/examples/$f.jsonl --output answers_$f.jsonl; done
```

Add `--early-exit 0.99` when the server runs in `fast-exit` mode.

**What to expect on these files** (basal-1.0-4.5B, `fast`, real run): `choice` 6/8, `noul` 7/8, `score` 6/8 correct.
The mistakes are instructive: the model picks a wrong invoice total (823 zł instead of 738 zł) and a wrong bonus band at 103% of plan, and
misreads "above 5 000 zł net" for an amount of exactly 5 000 zł net — with high confidence. Like other System 1 models,
it is weak at **arithmetic and exact thresholds**. Compute numbers in code and let the model make the typed decision on
top of them; split composite rules into simple questions (`complex.jsonl`, `cx-03`: the model gets the last day and the
filing date right with high confidence, while the direct "was it in time?" question stays uncertain at 0.59). Low
confidence is the signal to route a decision to a person; the thresholds for 1% and 5% error are in `CALIBRATION.json`.

From Python, with the client of a running server:

```python
from basal.client import Basal
import json
b = Basal("http://127.0.0.1:8000")
for line in open("basal/examples/questions.jsonl"):
    q = json.loads(line)
    a = b.choice(q["state"], q["question"], {str(i): o for i, o in enumerate(q["options"])})
    print(a["choice"], round(a["confidence"], 3))
```

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | GPUs |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + shared prefix + token-budget batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen **per request** | any CUDA GPU (4.5B only) |
| `fp8` | `fast` with dynamic FP8 weights + activations (torchao) | Hopper, Blackwell (Ada: FP8 compilation stalled on an RTX 4090) |
| `nvfp4` | `fast` with NVFP4 weights + activations (torchao, experimental) | Blackwell (B200/B300, RTX 50xx/PRO, GB10) |
| `vllm` | vLLM with the ModelOpt **FP8 / NVFP4** checkpoints (native low-precision kernels) | Hopper / Blackwell |
| `mlx` *(default on Apple Silicon)* | MLX bf16 + shared prefix + token-budget batching | Apple Silicon (`[mlx]` extra) |
| `mlx-q8` | `mlx` with 8-bit weights: about half the memory, not faster | Apple Silicon |
| `mps` | PyTorch MPS + shared prefix + token-budget batching (no graphs) | Apple Silicon |
| `gguf` | llama.cpp on a converted GGUF file (`--gguf`), shared prefix as llama.cpp sequences ([docs/GGUF.md](docs/GGUF.md)) | Apple Silicon (Metal), CUDA, CPU (`[gguf]` extra) |
| `ollama` | Ollama safetensors import via raw text and next-token logprobs (two HTTP calls per decision; `--ollama-model`) | Apple Silicon or other Ollama hosts |
| `eager` | plain PyTorch reference | any GPU (CUDA or Apple MPS) or CPU |

- **Two option orders** (`--orders 2`, default): every question is asked with the options in original and reversed
  order and the probabilities are averaged; with the shared prefix this costs only ~8% more than one order.
- **Early exit** (`--mode fast-exit`): *what it is.* The model has 60 layers, and for many questions the answer is
  already clear before the last one. We trained small **exit heads** (a normalisation layer and a low-rank adapter
  that reuse the model's output head) after layers 30, 35, 40, 45 and 50–55. During the forward pass the server checks the exit
  head at each of these points; if the probability of the top option is above a threshold calibrated for that layer,
  the remaining layers are skipped and the answer is taken from the exit head. Thresholds are calibrated so that the
  early answer agrees with the full model on a chosen share of decisions (99.9%, 99.5%, 99% or 98% on calibration
  data). Because the decision becomes readable only in the last ten of 60 layers, the saving is modest: **12.2 → 10.5 ms per
  decision on H100 at `0.99` with 99.2% agreement with fp32** (8.8 → 7.7 ms on B300). Each request chooses its level
  with `"early_exit": "0.99"`; `"off"` (default) always uses the final layer, so one server serves both (servers in other
  modes reject the field). A batch stops
  only when all its requests are confident. Levels: `off`, `0.999`, `0.995`, `0.99`, `0.98`.
- **FP4 with vLLM**: in a separate environment (vLLM brings its own torch), `uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz"`, then
  `basal-serve --model Remek/basal-1.0-4.5B-NVFP4 --mode vllm` (see [docs/QUANTIZATION.md](docs/QUANTIZATION.md)).

Start-up: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (about 4–10 minutes
the first time for the 4.5B model, 2.5–4 minutes for the 1.5B, 10–17 minutes in `fp8`; much less with a warm
compile cache). Prompts longer than 3,072 tokens are served with a normal forward pass.

## Benchmark your GPU

Two tools are installed with the package. Both run out of the box on the bundled examples (`questions.jsonl`,
`choice.jsonl`, `noul.jsonl`, `score.jsonl` in `basal/examples/`: 44 Polish and English items with gold answers):

```bash
# offline: latency, throughput and agreement of serving modes (no HTTP); put eager-fp32 first as the reference
basal-bench --model Remek/basal-1.0-4.5B --modes eager-fp32 fast fp8 fast-exit@0.99 --out bench.json

# end to end over HTTP against a running server (basal-serve ...)
basal-loadtest --url http://127.0.0.1:8000/v1/systemone
```

`basal-bench` loads the model once per mode and reports, per mode:

| column | meaning |
|---|---|
| `lat2 ms` | median latency of one decision with **both** option orders at batch size 1 (what the server does by default) |
| `lat1 ms` | the same with one option order |
| `dec/s` | two-order decisions per second when 32 option-order passes are processed together |
| `agree` | share of decisions whose top option equals the first mode's. On Apple Silicon the default reference is bf16 `mps`, **not fp32**; for fp32 agreement, pass `--modes eager-fp32 mps mlx` if memory allows. Each JSON row records `reference_mode`. |
| `acc` | accuracy against `gold` |
| `GB` | CUDA/MLX peak allocation; MPS driver-held memory after the run; unavailable for CPU or backends without PyTorch/MLX memory tracking (GGUF, Ollama, vLLM) |

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with 32
concurrent clients.

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers: latency depends on
prompt length, and accuracy on 44 items is noisy. For numbers that describe *your* workload, write a JSONL file with a
few hundred items in the same simple format (`{"state": ..., "question": ..., "options": [...], "gold": <index>}`,
`gold` optional) with realistic state lengths, and pass it with `--questions my_items.jsonl` (several files are
allowed). The speed tables in this README were measured with the same tools on our private 500-item test sample (mean
prompt about 360 tokens), which we do not publish so that it cannot be trained on.

## API

`POST /v1/systemone`

```jsonc
{
  "state": "text or JSON",
  "questions": {
    "<name>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<description>", ...},
               "option_keys": "show" | "hide"},                                        // optional, default "show"
    "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},   // criteria optional
    "<name>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}
  },
  "early_exit": "off" | "0.999" | "0.995" | "0.99" | "0.98"      // optional, --mode fast-exit only
}
```

2–10 options per question. Each answer has `probabilities` (calibrated), `confidence` and the type-specific field
(`choice`, `noul` = P(yes), `score` = expected level + `legend`); `usage` has `input_tokens` and `output_tokens`.
**Option keys.** By default every described option is shown to the model as `key: description`
(`{"B": "Dispatch office: Krakow"}` → `B: Dispatch office: Krakow`), because only you know whether a key is meaningful
(a service code, a decision name); the server never guesses. If your keys are mere placeholders (`option_1`, `0`, …) and
the descriptions alone define the options, send `"option_keys": "hide"` in the question: the model then sees the
descriptions only, which is also the format of the training data (descriptions that are not unique still get their key).
A key without a description (`"criteria": {"approve": null}` or a list of keys) is shown by itself. The same applies to
named `score` levels. `GET /v1/models` returns `{"models": [{"name", "description",
"release_date", ...}]}`; `GET /health`.

**Using the confidence.** The server averages both option orders and then applies the per-type temperatures of
`CALIBRATION.json`, which were fitted on exactly that averaged prediction. The file also contains confidence thresholds
chosen on the calibration split, before testing, for a target error of 1% and 5% among accepted decisions; applied
once to the test split they accept 58.6% / 75.0% of decisions at 1.2% / 4.4% observed error (4.5B) and 49.5% / 68.6%
at 1.4% / 5.5% (1.5B). These numbers were measured on descriptions-only prompts (`"option_keys": "hide"`); in the default mode, which
also shows option keys, they are not validated — refit the thresholds on your own labelled requests. Accept decisions above the threshold automatically and route the rest to a person. The FP8 and NVFP4 checkpoints inherit the bf16 temperatures and
thresholds, which are not validated for them.

## Limitations

- Trained and evaluated on generated, grounded or verified decision data; claims about specific real-world document
  collections require validation on your own data.
- Polish world knowledge of a 4.5B model is limited; supply the relevant facts in the state.
- Legal rules change; the model does not know rules introduced after its training.
- A generator error in the training data taught the checkpoints the wrong notice period (art. 36 § 1 KP) when three
  years of employment are completed during a one-month notice: they answer one month instead of three. The evaluation
  labels are corrected; the checkpoints will be retrained in the next release.
- Averaging the original and reversed option order reduces, but does not remove, sensitivity to option order for three
  or more options.
- About 21% of the training items (13,500 English items) come from an aggregated public corpus whose upstream sources
  could not be traced item by item; see the technical report.
- The test split was consulted during development; its results are those of an adaptive process on held-out templates,
  not a single untouched final evaluation.
- Decisions with serious consequences for people should be reviewed by a person.


## Citation

```bibtex
@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}
```

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models; see [NOTICE](NOTICE).
