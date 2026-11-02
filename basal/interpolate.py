"""Linear weight interpolation of two checkpoints that share a base model (WiSE-FT / model soups, issue #42).

    theta(alpha) = (1 - alpha) * theta_old + alpha * theta_new

alpha is the weight of the new checkpoint: 0 gives the old weights, 1 the new ones. Each tensor is mixed in fp32 and
stored in the dtype of the new checkpoint; tensors that are not floating point (position ids) are taken from the new
checkpoint. The output keeps the new checkpoint's shard layout and is streamed to disk tensor by tensor, byte for byte
as safetensors.torch.save_file would write it, so memory holds one tensor at a time (peak: the largest tensor in fp32,
three times), whatever the shard size. The result is written to a temporary folder next to --out and renamed only once
every shard is written, so a failed merge leaves no partial checkpoint.

The checkpoints must have the same tensor names, shapes and dtypes, unquantized weights (no quantization_config: FP8 /
NVFP4 exports store encoded weights and scales, which do not interpolate), the same architecture (config.json,
compared without serialization-only keys) and the same vocabulary (tokenizer.json: vocab, BPE merges, added tokens).
The tokenizers must also have the same settings that decide how text is split (normalizer, pre-tokenizer,
post-processor, decoder, model options; merges compared in either serialization). A pair whose vocabulary matches but
whose settings differ, such as the Bielik-*-v3.0-Instruct base models (an older tokenizers layout that adds a word
boundary after special tokens) against basal-2.0, is refused unless --allow-tokenizer-difference is given: the merged
model is served with the --meta-from tokenizer only. The refusal shows a probe text the two split differently, when
one does.

The non-weight files (basal.json, tokenizer, chat template, config, CALIBRATION*.json) come from --meta-from (default:
the new checkpoint), so the merged model is served in that checkpoint's prompt format: merging basal-1.5 and basal-2.0
weights with --meta-from old gives a format 1 model, otherwise format 2. The prompt format moves accuracy as much as
the weights do (docs/INTERPOLATION.md), so evaluate a merge together with the format it ships with. The shipped
calibration is only a starting point: fit one per merged model with basal-calibrate.

    basal-interpolate Remek/basal-1.5-mini Remek/basal-2.0-mini --alpha 0.25 0.5 0.75 --out merged/mini --meta-from old
"""
import argparse
import functools
import json
import shutil
from pathlib import Path

# not copied from --meta-from: the model card and checksums describe the source checkpoint, the evidence head and
# the thinking adapter were trained on its weights
SKIP = {".gitattributes", "README.md", "SHA256SUMS", "evidence_head.pt"}
# weight files of any format, and their indexes: the merged weights are the only ones in the output
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack", ".onnx"}
WEIGHT_PREFIXES = ("model.safetensors.index", "pytorch_model", "tf_model", "flax_model", "model.onnx")
# config.json keys that do not describe the architecture (serialization, runtime defaults)
CONFIG_IGNORE = {"transformers_version", "torch_dtype", "dtype", "use_cache"}
@functools.cache
def torch_dtypes():
    """safetensors dtype name -> torch dtype (torch is imported lazily, as everywhere in this module)."""
    import torch

    names = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64", "I32": "int32",
             "I16": "int16", "I8": "int8", "U8": "uint8", "U16": "uint16", "U32": "uint32", "U64": "uint64",
             "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
    return {k: getattr(torch, v) for k, v in names.items() if hasattr(torch, v)}


# tokenizer.json sections that do not change the token ids of a text
TOKENIZER_IGNORE = {"version", "truncation", "padding"}
# safetensors dtypes of encoded (quantized) weights
QUANTIZED_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "U8", "I8", "U4", "I4"}
PROBES = ["Zażółć gęślą jaźń. Ala ma kota, 123 zł!", " leading space\n\nnew lines\ttab", "x  y   ", "",
          "<s>[INST] Pytanie? [/INST]", "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n",
          "emoji 🙂 and ünïcödé", "A) tak\nB) nie\nOdpowiedź:"]


def shards(model_dir):
    """tensor name -> safetensors file holding it."""
    from safetensors import safe_open

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        out = {}
        for k, v in json.loads(index.read_text())["weight_map"].items():
            p = Path(v)
            # the shard paths are reused under the output folder: nothing absolute, nothing that leaves the model
            if p.is_absolute() or ".." in p.parts or not (d / p).resolve().is_relative_to(d.resolve()):
                raise SystemExit(f"{index}: shard path {v!r} is outside the model folder")
            out[k] = d / p
        return out
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            out.update(dict.fromkeys(s.keys(), f))
    return out


def config_key(path):
    """config.json as compared: rope_theta folded into rope_parameters (the transformers 5 layout), keys that do not
    describe the architecture dropped."""
    c = json.loads(Path(path).read_text())
    if "rope_theta" in c:
        c.setdefault("rope_parameters", {"rope_theta": c["rope_theta"], "rope_type": "default"})
        if c.get("rope_scaling") is None:
            c.pop("rope_scaling", None)
        c.pop("rope_theta")
    return {k: v for k, v in c.items() if k not in CONFIG_IGNORE}


def tokenizer_key(path):
    """What fixes the meaning of a token id: vocabulary, BPE merges (string or pair form) and added tokens with their
    flags."""
    t = json.loads(Path(path).read_text())
    m = t.get("model", {})
    merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m.get("merges") or []]
    return m.get("type"), m.get("vocab"), merges, t.get("added_tokens", [])


def tokenizer_settings(path):
    """tokenizer.json without the sections that do not change token ids, merges in pair form."""
    t = json.loads(Path(path).read_text())
    t = {k: v for k, v in t.items() if k not in TOKENIZER_IGNORE}
    m = t.get("model", {})
    if m.get("merges"):
        t["model"] = {**m, "merges": [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m["merges"]]}
    return t


def tokenization_difference(po, pn):
    """The first probe text the two tokenizers split differently, as a message, or None."""
    from tokenizers import Tokenizer

    a, b = Tokenizer.from_file(str(po)), Tokenizer.from_file(str(pn))
    for text in PROBES:
        for special in (True, False):
            ea, eb = a.encode(text, add_special_tokens=special), b.encode(text, add_special_tokens=special)
            if ea.ids != eb.ids:
                return f"{text!r}: {ea.tokens[:12]} vs {eb.tokens[:12]}"
    return None


def check_pair(old, new, allow_tokenizer_difference=False):
    """Refuses (SystemExit) a pair that cannot be interpolated; returns the tensor -> shard maps of both and
    tensor -> (safetensors dtype, shape) of the new checkpoint, read from the headers."""
    from safetensors import safe_open

    for name in ("config.json", "tokenizer.json"):
        missing = [d for d in (old, new) if not (Path(d) / name).is_file()]
        if missing:
            raise SystemExit(f"{name} missing in {missing[0]}: cannot check that the checkpoints share an "
                             "architecture / tokenizer")
    for d in (old, new):
        if "quantization_config" in json.loads((Path(d) / "config.json").read_text()):
            raise SystemExit(f"{d} is quantized (quantization_config): interpolate the unquantized checkpoints and "
                             "quantize the result")
    po, pn = Path(old) / "config.json", Path(new) / "config.json"
    if po.read_bytes() != pn.read_bytes() and config_key(po) != config_key(pn):
        raise SystemExit("config.json differs: the checkpoints do not share an architecture")
    po, pn = Path(old) / "tokenizer.json", Path(new) / "tokenizer.json"
    if po.read_bytes() != pn.read_bytes():
        if tokenizer_key(po) != tokenizer_key(pn):
            raise SystemExit("tokenizer.json differs: the checkpoints do not share a vocabulary")
        so, sn = tokenizer_settings(po), tokenizer_settings(pn)
        if so != sn and not allow_tokenizer_difference:
            keys = sorted(k for k in so.keys() | sn.keys() if so.get(k) != sn.get(k))
            example = tokenization_difference(po, pn)
            raise SystemExit(f"the tokenizers share the vocabulary but their settings differ ({', '.join(keys)}"
                             + (f"; for example {example}" if example else "") + "); the merged model is served with "
                             "the --meta-from tokenizer only. Pass --allow-tokenizer-difference if that is intended "
                             "(a base model written by an older tokenizers release)")

    a, b = shards(old), shards(new)
    if not b:
        raise SystemExit(f"{new}: no safetensors weights")
    if a.keys() != b.keys():
        raise SystemExit(f"different tensors: only old {sorted(a.keys() - b.keys())[:5]}, "
                         f"only new {sorted(b.keys() - a.keys())[:5]}")
    specs = {}
    for k in b:  # shapes and dtypes from the headers, before anything is written
        with safe_open(a[k], "pt") as fo, safe_open(b[k], "pt") as fn:
            so, sn = fo.get_slice(k), fn.get_slice(k)
            if so.get_shape() != sn.get_shape():
                raise SystemExit(f"{k}: shape {tuple(so.get_shape())} vs {tuple(sn.get_shape())}")
            dt = {so.get_dtype(), sn.get_dtype()}
            if dt & QUANTIZED_DTYPES and k.endswith("weight"):
                raise SystemExit(f"{k}: quantized weights ({'/'.join(sorted(dt))}) do not interpolate")
            if sn.get_dtype() not in torch_dtypes():
                raise SystemExit(f"{k}: unsupported dtype {sn.get_dtype()}")
            specs[k] = (sn.get_dtype(), tuple(sn.get_shape()))
    return a, b, specs


def save_streamed(path, specs, produce, metadata):
    """Writes a safetensors file tensor by tensor: specs is name -> (safetensors dtype, shape), produce(name) returns
    the tensor. The layout is that of safetensors.torch.save_file (tensors ordered by dtype, then name; header padded to
    8 bytes): the order is read from a file of 1-element tensors, so only one tensor is in memory at a time."""
    import struct

    import torch
    from safetensors.torch import save

    DTYPES = torch_dtypes()
    proxy = save({k: torch.zeros(1, dtype=DTYPES[d]) for k, (d, _) in specs.items()})
    order = [k for k in json.loads(proxy[8:8 + struct.unpack("<Q", proxy[:8])[0]]) if k != "__metadata__"]
    header, offset = {"__metadata__": metadata}, 0
    for k in order:
        d, shape = specs[k]
        size = torch.Size(shape).numel() * torch.empty(0, dtype=DTYPES[d]).element_size()
        header[k] = {"dtype": d, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    h = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for k in order:
            t = produce(k)
            if t.dtype != DTYPES[specs[k][0]] or tuple(t.shape) != specs[k][1]:
                raise SystemExit(f"{k}: produced {t.dtype} {tuple(t.shape)}, header says {specs[k]}")
            f.write(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def interpolate(old, new, alpha, out, meta_from=None, allow_tokenizer_difference=False):
    """Writes the merged checkpoint to out (a new or empty directory) and returns its path."""
    import torch
    from safetensors import safe_open

    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"alpha {alpha} outside [0, 1]")
    old, new, out = Path(old), Path(new), Path(out)
    meta_from = Path(meta_from or new)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"{out} exists and is not empty; remove it or choose another --out")
    a, b, specs = check_pair(old, new, allow_tokenizer_difference)
    tmp = out.parent / f".{out.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        by_file = {}  # tensors grouped by the new checkpoint's shard (path relative to the model, as in the index)
        for k, f in b.items():
            by_file.setdefault(f.relative_to(new), []).append(k)

        def mixed(k):
            with safe_open(a[k], "pt") as fo, safe_open(b[k], "pt") as fn:  # no handle outlives its tensor
                t_old, t_new = fo.get_tensor(k), fn.get_tensor(k)
            if not t_new.is_floating_point():
                return t_new
            return torch.lerp(t_old.float(), t_new.float(), alpha).to(t_new.dtype)

        for rel, keys in by_file.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            save_streamed(tmp / rel, {k: specs[k] for k in keys}, mixed, {"format": "pt"})
        if (new / "model.safetensors.index.json").exists():
            shutil.copy(new / "model.safetensors.index.json", tmp)

        for f in meta_from.rglob("*"):
            rel = f.relative_to(meta_from)
            if not f.is_file() or rel.name in SKIP or rel.parts[0] == "think" or rel.parts[0].startswith("."):
                continue
            if f.suffix in WEIGHT_SUFFIXES or rel.name.startswith(WEIGHT_PREFIXES):
                continue
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, tmp / rel)
        info = json.loads((tmp / "basal.json").read_text()) if (tmp / "basal.json").exists() else {}
        info.update({"name": f"{info.get('name', new.name)}-interp-a{alpha:g}",
                     "interpolation": {"old": str(old), "new": str(new), "alpha": alpha,
                                       "rule": "(1 - alpha) * old + alpha * new, fp32 accumulate"}})
        (tmp / "basal.json").write_text(json.dumps(info, indent=1) + "\n")
        if out.exists():
            out.rmdir()  # empty, checked above
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # no docstring under python -OO
    ap.add_argument("old", help="local directory or Hugging Face repo id")
    ap.add_argument("new", help="local directory or Hugging Face repo id")
    ap.add_argument("--alpha", type=float, nargs="+", required=True, help="weight of the new checkpoint, in [0, 1]")
    ap.add_argument("--out", required=True, help="output prefix: one new directory <out>-a<alpha> per alpha")
    ap.add_argument("--meta-from", default=None, help="checkpoint whose non-weight files (prompt format, tokenizer, "
                                                      "calibration) are copied: old, new (default) or a path / repo id")
    ap.add_argument("--allow-tokenizer-difference", action="store_true",
                    help="accept tokenizers with the same vocabulary that split text differently (see above)")
    a = ap.parse_args(argv)
    if any(not 0.0 <= x <= 1.0 for x in a.alpha):
        ap.error("--alpha values must be in [0, 1]")
    names = [f"{a.out}-a{x:g}" for x in a.alpha]
    if len(set(names)) != len(names):
        ap.error(f"--alpha values give the same output directory: {sorted({n for n in names if names.count(n) > 1})}")
    from huggingface_hub import snapshot_download

    def local(m):
        return Path(m) if Path(m).exists() else Path(snapshot_download(m))

    old, new = local(a.old), local(a.new)
    meta = {"old": old, "new": new, None: new}.get(a.meta_from) or local(a.meta_from)
    for alpha, out in zip(a.alpha, names):
        print(interpolate(old, new, alpha, out, meta, a.allow_tokenizer_difference))


if __name__ == "__main__":
    main()
