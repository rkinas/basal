# Weight interpolation (`basal-interpolate`)

`basal-interpolate` mixes two checkpoints that share a base model, tensor by tensor
([#42](https://github.com/rkinas/basal/issues/42); WiSE-FT, model soups):

    θ(α) = (1 − α) · θ_old + α · θ_new

α is the weight of the new checkpoint: α = 0 gives the old weights, α = 1 the new ones.
- Each tensor is mixed in fp32 and stored in the new checkpoint's dtype.
- Integer tensors are taken from the new checkpoint.
- The output keeps the new checkpoint's shard layout. It is streamed to disk one tensor at a time, with the bytes
  `safetensors` `save_file` would write. Peak memory is the largest tensor in fp32, whatever the shard size: about
  1.7 GB for the 4.5B models, against 30 GB when a whole shard is held in memory.
- The merge is written to a temporary folder and renamed only when every shard is written, so a failure leaves no
  partial checkpoint. `--out` must not exist or must be empty.
- The result is deterministic: the same inputs give byte-identical safetensors.

    basal-interpolate Remek/basal-1.5-mini Remek/basal-2.0-mini --alpha 0.25 0.5 0.75 --out merged/mini \
        --meta-from old
    basal-calibrate panel.jsonl --model merged/mini-a0.5 --save-rows rows/mini-a0.5.jsonl --out calib/mini-a0.5.json

The tool refuses a pair before writing anything unless the two checkpoints have:
- the same tensor names, shapes and dtypes;
- unquantized weights, with no `quantization_config`. FP8 and NVFP4 exports store encoded weights and scales, which
  do not interpolate, so merge the unquantized checkpoints and quantize the result.
- the same architecture (`config.json`);
- the same vocabulary (`tokenizer.json`: vocab, BPE merges, added tokens).

Configs and tokenizers are compared by content, so a file written by an older transformers release passes.

The tokenizers must also have the same settings that decide how text is split: normalizer, pre-tokenizer,
post-processor, decoder and model options. The `Bielik-*-v3.0-Instruct` base models have basal-2.0's vocabulary but an
older tokenizer layout, which adds a word boundary after special tokens and at a leading space. They are accepted only
with `--allow-tokenizer-difference`. When a refused pair splits one of the probe texts differently, the error message
shows that text.

The merged model is served with the `--meta-from` tokenizer only. With basal-2.0 on the new side, that is the tokenizer
basal-2.0 was fine-tuned with.

**Prompt format.** The non-weight files (`basal.json`, tokenizer, chat template, `CALIBRATION*.json`) come from
`--meta-from`: `new` (default), `old`, or a path / repository id. Its model card, checksums, evidence head, thinking
adapter and NVFP4 activation scales are not copied: they describe or were fitted to its own weights. The merged model
is served in that checkpoint's prompt format. For a basal-1.5 / basal-2.0 pair:
- `--meta-from old` gives a [format 1](FORMATS.md) model: letters, two option orders averaged.
- The default gives a [format 2](FORMAT2.md) model: codes, one order.

The format can move accuracy as much as the weights do ([#42](https://github.com/rkinas/basal/issues/42)), so evaluate a
merge in the format it will ship with.

**Calibration.** The copied `CALIBRATION.json` belongs to the source checkpoint. Fit a new one for every merged model
with `basal-calibrate`.

## Results

The measurements that led to this tool (merges of several basal pairs on public English and Polish panels) are the
contributor's, in [#42](https://github.com/rkinas/basal/issues/42); they are not results of the basal project. Fit a
calibration for every merged model and evaluate it on your own data before you use it.

## Evaluating a merge

`tools/interpolation/` holds the contributor's evaluation scripts from #42 (not part of the installed package):
- **Panels:** `panel_en.py` and `panel_pl.py` build public panels from Hugging Face datasets, with a fixed seed.
- **Analysis:** `analyze.py` turns `basal-calibrate --save-rows` files into accuracy per task, macro accuracy, raw and
  calibrated ECE, and a paired bootstrap against a reference.
- **Repeatability:** `compare_runs.py` compares a run with its repeat.

    uv run --with datasets python tools/interpolation/panel_en.py --out panel_en.jsonl
    uv run --with datasets python tools/interpolation/panel_pl.py --out panel_pl.jsonl
    basal-interpolate Remek/basal-1.5-4.5B Remek/basal-2.0-4.5B --alpha 0.5 --out merged/4.5B
    for m in Remek/basal-2.0-4.5B merged/4.5B-a0.5; do n=$(basename $m)
      basal-calibrate panel_en.jsonl --model $m --mode fast-nocompile --save-rows rows/$n.jsonl --out calib/$n.json
    done
    uv run --with numpy python tools/interpolation/analyze.py ref=rows/basal-2.0-4.5B.jsonl a0.5=rows/4.5B-a0.5.jsonl \
        --panel panel_en.jsonl --ref ref --out report.md

Two-order runs add `--orders 2` to `basal-calibrate`.
