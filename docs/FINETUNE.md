# Adapt a basal model to your own decisions (LoRA)

`basal-finetune` trains a small LoRA adapter on **your own labelled decisions** and writes a model you serve with
`basal-serve` as any other. You label requests once, in the same JSONL that `basal-calibrate` reads
([CALIBRATION.md](CALIBRATION.md#input)); the same file trains and calibrates.

Do this after calibration, not instead of it. First run `basal-calibrate` on a few hundred of your labelled decisions
and look at the accuracy it reports on them. If it is high enough for your use, calibration alone is what you need. If
the model is wrong about things that are particular to your business (your product names, your rules of thumb, your
labels), adapting it is worth the extra hour.

Everything runs on your machine: your data does not leave it.

**Status (2.5): tested end to end on a Mac (Apple M4 Pro, MPS) with basal-omni-pico.** The T4 route below was not
tested; [What was tested](#what-was-tested) says what is known.

## What you need

- A labelled JSONL as in CALIBRATION.md: one request per line, every question with its correct answer in `"gold"`.
  `choice`, `noul`, `score` and `act` questions are trained; `multi` questions are skipped (and counted).
  `basal/examples/labelled_tickets.jsonl` is a toy file (500 invented support messages, two questions each, 10% of the
  labels wrong on purpose) to try the steps with.
- How many: calibration needs a few hundred decisions. Adaptation moves the model only with more: plan on at least
  500 training decisions per question type, then 300 or more that the model never trains on (below).
- Labels that agree with each other. Two people labelling the same request differently teach the model nothing.
- A GPU, or patience: a Mac with 16 GB or more, a free Colab / Kaggle T4 (16 GB), or any CUDA GPU. A CPU works for the
  toy file only.

```bash
pip install "basal[finetune]"          # basal + peft
```

## Run

```bash
basal-finetune labelled.jsonl --model Remek/basal-omni-pico --out my-pico
basal-calibrate my-pico/heldout.jsonl --model my-pico --mode eager --out my-pico/CALIBRATION.json
basal-serve --model my-pico --mode eager --port 8000
```

1. **Split.** The lines go by `group` (or `id`; as in basal-calibrate, twins and one customer's lines stay together)
   into a training part and a held-out part, 25% (`--holdout`). Lines that share a group on any question (a
   per-question `group`) go to the same part, so no group is on both sides. Training sees the first part only. The
   held-out lines are written to `my-pico/heldout.jsonl`.
2. **Train.** LoRA (rank 16, the decoder's linear layers) on the prompt `basal-serve` builds, in the model's own format, with
   the loss on the option codes at the answer position, which is what the server reads. Each example is shown in the
   option order you wrote and in the reversed one, the two orders `basal-serve` averages. Two epochs, 16 examples per
   step, learning rate 1e-4 with warm-up and cosine decay, gradient clipping 1.0: conservative defaults, and every one
   is a flag (`--epochs`, `--batch`, `--lr`, `--rank`, `--alpha`, `--dropout`, `--max-steps`).
3. **Report.** Top-1 accuracy on the held-out decisions before and after, in the model's given option order. It is
   printed and written to `my-pico/finetune.json`.
4. **Merge and write.** `my-pico/` is a complete model (merged weights, tokenizer, `basal.json`) and the adapter alone
   is in `my-pico/lora/` (a few tens of MB). The model you started from is not changed.
5. **Calibrate on the held-out lines.** The adapter changes how confident the model is, so the shipped
   `CALIBRATION.json` no longer applies: the folder starts with an uncalibrated one (`max_error` defers every automatic
   action until you calibrate). `basal-calibrate` on `heldout.jsonl` fits the temperatures and certifies the
   thresholds on decisions the training never saw. **Calibrating on lines the model trained on makes it
   over-confident**, which is why the split is done for you.

`--model` takes a basal model (a directory with `basal.json`, or a Hugging Face id) or a chat model you attached
with `basal-attach` ([SERVING.md](SERVING.md#attach-a-model-that-was-not-trained-on-basal)).

## Memory and time

| where | notes |
|---|---|
| Mac, Apple silicon (MPS) | `--dtype auto` picks bfloat16. basal-omni-pico needed about 4 GB of unified memory for training. |
| Colab / Kaggle T4, 16 GB | **not tested.** A T4 has no bfloat16, so `--dtype auto` picks float16 with a gradient scaler. basal-omni-pico and basal-2.0-mini should fit; for a 4.5B model add `--grad-checkpointing`. Sessions end after some hours: keep `--max-steps` small first. |
| any CUDA GPU with bfloat16 | the same code path as the Mac's, with CUDA autocast. Not tested either. |
| CPU | float32; hours for the toy file with the smallest model. |

`--max-len 2048` skips the examples whose prompt is longer (they are counted in the output). One example is run at a
time and the gradients are accumulated, so the memory is that of one prompt, not of a batch. Models built on Qwen3.5
(basal-omni-pico) run their Gated DeltaNet layers in plain PyTorch unless `flash-linear-attention` and
`causal-conv1d` are installed, which is slow but exact.

## What to expect

The toy file in `basal/examples/` gave, on basal-omni-pico on the Mac:

RESULTS

Your numbers depend on your data. Read the held-out accuracy before and after, and `basal-calibrate`'s report:

- accuracy up and calibration error small: use it;
- accuracy up but the confidence thresholds do not certify (too few held-out decisions): label more;
- accuracy flat: the labels were already learnable by the base model, or are inconsistent; calibration alone is
  enough, or review the labels;
- accuracy down: the labels are noisy or the learning rate is too high for so little data (try `--lr 5e-5 --epochs 1`).

The adapter learns your labels, including your mistakes. A model adapted on a narrow set can answer worse outside it:
check it on a few requests of other kinds before relying on it for those.

## What was tested

- **Tested:** the whole route on a Mac (Apple M4 Pro, 24 GB, PyTorch 2.13, transformers 5.17, peft 0.21, MPS, bfloat16)
  with basal-omni-pico: split, train, merge, calibrate on `heldout.jsonl`, serve, answer a request. The unit tests
  (`tests/test_finetune.py`) cover the split, the examples and the readout loss without a model.
- **Not tested:** a T4 or any CUDA GPU, float16, `--grad-checkpointing`, Gemma-based models (basal-omni-mini / max /
  ultra) and the basal-2.0 models, `multi` questions (skipped), images and audio (not trained). The memory figures
  for them are estimates.
- Speed on the Mac is noisy: other jobs shared the machine.
