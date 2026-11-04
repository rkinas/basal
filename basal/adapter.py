"""LoRA adapters: read an mlx-lm or PEFT adapter directory, merge it into a model's weights at load (torch, MLX, the
TPU backend's numpy weights), and fuse it into a checkpoint (basal-fuse-adapter; also the copy vLLM and SGLang serve
under --adapter, fused_copy) for the engines that load weights basal cannot touch.

Formats, by the files in the directory:
  mlx-lm  adapter_config.json (fine_tune_type "lora" or absent, lora_parameters.scale) + adapters.safetensors,
          keys <path>.lora_a [in, r] / <path>.lora_b [r, out]
  PEFT    adapter_config.json (peft_type "LORA", r, lora_alpha, use_rslora) + adapter_model.safetensors,
          keys base_model.model.<path>.lora_A.weight [r, in] / .lora_B.weight [out, r]

The merge is PEFT's merge_and_unload on the CPU, on every framework: delta = (b @ a) * scale in float32 ([out, in],
torch's CPU GEMM, the factors float32 whatever their stored dtype), added to the weight in float32 and rounded once to
the weight's dtype. (mlx-lm's LoRALinear.fuse rounds the delta to the weight's dtype before adding it: about 0.4% of a
merged bf16 weight's elements then end one unit in the last place apart, which moved near-tie answers.)"""
import argparse
import atexit
import functools
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MLX_WEIGHTS, PEFT_WEIGHTS, CONFIG = "adapters.safetensors", "adapter_model.safetensors", "adapter_config.json"
# a text decoder's layer under the prefixes of mlx-lm, PEFT and transformers (model., base_model.model.model.,
# language_model.model., base_model.model.model.language_model. ...); anything else (vision_tower.encoder.layers.N,
# audio_tower...) is another tower whose layer N is not the decoder's
_LAYER = re.compile(r"^(?:(?:base_model|model|language_model)\.)*layers\.(\d+)\.(.+)$")
_MLX_KEY = re.compile(r"^(.+)\.lora_([ab])$")
_PEFT_KEY = re.compile(r"^(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")


@dataclass
class Adapter:
    path: Path
    name: str  # the directory's basename
    identity: str  # name@sha256(adapter_config.json + weights file)[:12]: what a calibration file is fitted with
    scale: float
    rank: int
    fmt: str  # "mlx" | "peft"
    deltas: dict = field(default_factory=dict)  # canonical key -> (a [r, in], b [out, r]), float32


def _name(path):
    return Path(path).resolve().name


def _fail(path, why):
    raise SystemExit(f"--adapter {_name(path)}: {why}")


def validate(path):
    """Refuse (SystemExit) what is not a plain LoRA adapter directory -> (fmt, weights file, scale)."""
    p = Path(path)
    if not (p / CONFIG).is_file():
        _fail(path, f"no {CONFIG} in {p} (an mlx-lm or PEFT LoRA adapter directory)")
    cfg = json.loads((p / CONFIG).read_text(encoding="utf-8"))
    if "peft_type" in cfg:
        if cfg["peft_type"] != "LORA":
            _fail(path, f"peft_type {cfg['peft_type']}: only LORA adapters can be merged")
        if cfg.get("use_dora"):
            _fail(path, "a DoRA adapter (use_dora): only LORA adapters can be merged")
        if cfg.get("fan_in_fan_out"):
            _fail(path, "fan_in_fan_out (Conv1D layers): not a layout the engines use")
        if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            _fail(path, "rank_pattern / alpha_pattern: per-module scales are not supported")
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        if not r or alpha is None:
            _fail(path, f"{CONFIG} lacks r / lora_alpha")
        fmt, weights, scale = "peft", p / PEFT_WEIGHTS, alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
    else:
        kind = cfg.get("fine_tune_type", "lora")
        if kind != "lora":
            _fail(path, f"fine_tune_type {kind}: only LoRA adapters can be merged")
        lp = cfg.get("lora_parameters") or {}
        if "scale" not in lp:
            _fail(path, f"{CONFIG} lacks lora_parameters.scale")
        fmt, weights, scale = "mlx", p / MLX_WEIGHTS, float(lp["scale"])
    if not weights.is_file():
        _fail(path, f"no {weights.name} next to its {fmt} {CONFIG}")
    return fmt, weights, float(scale)


def identity(path):
    """The adapter's identity without reading the tensors: name@sha256 of adapter_config.json (scale, rank) followed by
    the weights file, 12 hex digits."""
    _, weights, _ = validate(path)
    h = hashlib.sha256((Path(path) / CONFIG).read_bytes())
    with open(weights, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{_name(path)}@{h.hexdigest()[:12]}"


def canonical(key):
    """'layers.N.<rest>' from a module path of the text decoder under any framework prefix; ValueError for a module
    outside the decoder layers (an embedding, the head, a vision or audio tower)."""
    m = _LAYER.match(key)
    if not m:
        raise ValueError(f"{key}: not a module of the text decoder's layers (layers.N.<module>)")
    return f"layers.{m[1]}.{m[2]}"


def read(path):
    from safetensors.torch import load_file  # (the numpy reader has no bfloat16: PEFT / Unsloth save bf16 adapters)
    fmt, weights, scale = validate(path)
    pat = _MLX_KEY if fmt == "mlx" else _PEFT_KEY
    parts = {}
    for k, v in load_file(str(weights)).items():
        m = pat.match(k)
        if not m:
            _fail(path, f"{k}: not a LoRA tensor (lora_a / lora_b, lora_A / lora_B)")
        try:
            key = canonical(m[1])
        except ValueError as e:
            _fail(path, str(e))
        side = m[2].lower()
        if side in parts.setdefault(key, {}):
            _fail(path, f"{k}: a second {side} tensor for {key}")
        v = np.asarray(v.float().numpy(), dtype=np.float32)
        parts[key][side] = v.T if fmt == "mlx" else v  # mlx-lm stores a [in, r], b [r, out]
    deltas = {}
    for key, ab in sorted(parts.items()):
        if len(ab) != 2:
            _fail(path, f"{key}: lora_{'b' if 'a' in ab else 'a'} is missing")
        a, b = np.ascontiguousarray(ab["a"]), np.ascontiguousarray(ab["b"])
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            _fail(path, f"{key}: lora shapes a {a.shape} / b {b.shape} do not chain")
        deltas[key] = (a, b)
    if not deltas:
        _fail(path, f"{weights.name} holds no LoRA tensors")
    p = Path(path).resolve()
    return Adapter(path=p, name=p.name, identity=identity(p), scale=scale,
                   rank=max(a.shape[0] for a, _ in deltas.values()), fmt=fmt, deltas=deltas)


def delta(ad, key):
    """The update of one module as PEFT's merge computes it on the CPU: (b @ a) * scale in float32 ([out, in]), a torch
    tensor. Every merge (torch on any device, MLX, the TPU's numpy weights, basal-fuse-adapter) takes it from here and
    adds it with merged(), so --adapter and the fused checkpoint get bit-identical weights whatever GEMM the device
    would have used."""
    import torch
    a, b = ad.deltas[key]
    with torch.no_grad():
        return (torch.from_numpy(b) @ torch.from_numpy(a)) * ad.scale


def merged(w, d):
    """Weight w (torch) plus the float32 delta d, added in float32 and rounded once to w's dtype (PEFT's
    `weight.data += delta` with a float32 delta)."""
    return (w.float() + d.to(w.device)).to(w.dtype)


def _layers(ad, get):
    try:
        return get()
    except AttributeError:
        raise SystemExit(f"--adapter {ad.name}: the model has no decoder layers (model.layers) to merge into") from None


def _target(ad, layers, key):
    _, n, rest = key.split(".", 2)
    try:
        layer = layers[int(n)]
        parent = functools.reduce(getattr, rest.split(".")[:-1], layer)
        return parent, rest.split(".")[-1], getattr(parent, rest.split(".")[-1])
    except (IndexError, AttributeError):
        raise SystemExit(f"--adapter {ad.name}: {key}: no such module in the model") from None


def _check_shape(ad, key, shape):
    a, b = ad.deltas[key]
    if tuple(shape) != (b.shape[0], a.shape[1]):
        raise SystemExit(f"--adapter {ad.name}: {key}: the model's weight is {tuple(shape)}, the adapter's delta "
                         f"{(b.shape[0], a.shape[1])}")


def merge_torch(model, ad):
    """Merge into a transformers model's decoder linears in place -> the number of merged modules."""
    import torch
    root = getattr(model, "model", model)
    root = getattr(root, "language_model", root)
    layers = _layers(ad, lambda: root.layers)
    with torch.no_grad():
        for key in ad.deltas:
            _, _, mod = _target(ad, layers, key)
            if not isinstance(mod, torch.nn.Linear):
                raise SystemExit(f"--adapter {ad.name}: {key}: {type(mod).__name__} is not a linear layer the "
                                 "adapter can be merged into")
            w = mod.weight
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise SystemExit(f"--adapter {ad.name}: {key}: FP8 weights cannot take a merge")
            if not w.dtype.is_floating_point or w.element_size() < 2:  # (int8 / 4-bit quantised linears)
                raise SystemExit(f"--adapter {ad.name}: {key}: {str(w.dtype).replace('torch.', '')} weights cannot "
                                 "take a merge (merge into the bf16 model)")
            _check_shape(ad, key, w.shape)
            w.copy_(merged(w, delta(ad, key)))
    return len(ad.deltas)


def merge_mlx(model, ad):
    """Merge into an mlx-lm model's decoder linears in place (QuantizedLinear: dequantised, merged and re-quantised
    with its group size and bits, as LoRALinear.fuse) -> the number of merged modules. Each merged weight is evaluated
    here, so on a lazily loaded model (MLXBackend's q8) only the adapter's layers are materialised in bf16."""
    import mlx.core as mx
    import mlx.nn as nn
    lm = getattr(model, "language_model", model)
    layers = _layers(ad, lambda: lm.model.layers)
    for key in ad.deltas:
        parent, attr, mod = _target(ad, layers, key)
        quantized = isinstance(mod, nn.QuantizedLinear)
        if not (quantized or isinstance(mod, nn.Linear)):
            raise SystemExit(f"--adapter {ad.name}: {key}: {type(mod).__name__} is not a linear layer the adapter "
                             "can be merged into")
        w = mod.weight
        if quantized:
            w = mx.dequantize(w, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode)
        _check_shape(ad, key, w.shape)
        w = (w.astype(mx.float32) + mx.array(delta(ad, key).numpy())).astype(w.dtype)  # one rounding, as merged()
        if quantized:
            out, inp = w.shape
            fused = nn.Linear(inp, out, bias="bias" in mod)
            fused.weight = w
            if "bias" in mod:
                fused.bias = mod.bias
            fused = nn.QuantizedLinear.from_linear(fused, mod.group_size, mod.bits, mode=mod.mode)
            mx.eval(fused.parameters())
            setattr(parent, attr, fused)
        else:
            mx.eval(w)
            mod.weight = w
    return len(ad.deltas)


def merge_numpy(weights, ad):
    """Merge into Hugging Face-named numpy weights ("model.layers.N.<rest>.weight"; bf16 through ml_dtypes); the
    entries are replaced (the arrays may be read-only memory maps) -> the number of merged modules."""
    for key in ad.deltas:
        name = f"model.{key}.weight"
        if name not in weights:
            raise SystemExit(f"--adapter {ad.name}: {key}: no such module in the model")
        w = np.asarray(weights[name])
        _check_shape(ad, key, w.shape)
        weights[name] = (w.astype(np.float32) + delta(ad, key).numpy()).astype(w.dtype)  # one rounding, as merged()
    return len(ad.deltas)


def _temp_dir():
    out = Path(tempfile.mkdtemp(prefix="basal-adapter-"))
    atexit.register(shutil.rmtree, out, True)  # read by the engine's own process until basal-serve exits
    return out


# files of a model directory a fused checkpoint does not take over: the base model's calibration, NVFP4 activation
# scales, evidence head (trained on its hidden states; --adapter refuses evidence for the same reason) and checksums no
# longer hold for the merged weights (the adapter's calibration is copied, fused_calibration)
NOT_FUSED = ("CALIBRATION*", "NVFP4_AMAX.json", "SHA256SUMS", "evidence_head.pt")


def _weight_names(ad, md):
    """{canonical key: (safetensors file, tensor name)} of the adapter's modules in checkpoint md; refused when a module
    is missing or found twice."""
    from safetensors import safe_open
    found = {}
    for f in sorted(Path(md).glob("*.safetensors")):
        with safe_open(f, "pt") as st:
            for name in st.keys():
                if not name.endswith(".weight"):
                    continue
                try:
                    key = canonical(name.removesuffix(".weight"))
                except ValueError:
                    continue  # not a decoder layer (embeddings, the head, a vision or audio tower)
                if key in ad.deltas:
                    if key in found:
                        raise SystemExit(f"--adapter {ad.name}: {key}: two weights in {md} ({found[key][1]}, {name})")
                    found[key] = (f, name)
    missing = sorted(set(ad.deltas) - set(found))
    if missing:
        raise SystemExit(f"--adapter {ad.name}: {missing[0]}: no such module in the model"
                         + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""))
    return found


def fuse_checkpoint(md, ad, out):
    """Write checkpoint md with the adapter merged into `out` (a new or empty directory): every safetensors file again,
    with the same tensor names, dtypes and metadata, its decoder layers' weights merged as merge_torch merges them
    (merged()), so a server of `out` loads the weights --adapter merges at load bit for bit; every other file is copied
    as it is (config.json, the index, tokenizer, template, basal.json), but not NOT_FUSED nor subdirectories -> the
    number of merged modules. FP8 weights cannot take a merge (refused)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    md, out = Path(md), Path(out)
    names = _weight_names(ad, md)
    by_file = {}
    for key, (f, name) in names.items():
        by_file.setdefault(f, {})[name] = key
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(md.glob("*.safetensors")):
        if f not in by_file:  # nothing of the adapter in this file: the same bytes
            shutil.copyfile(f, out / f.name)
            continue
        tensors = {}
        with safe_open(f, "pt") as st:
            meta = st.metadata()
            for name in st.keys():
                t = st.get_tensor(name)
                key = by_file[f].get(name)
                if key is not None:
                    if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                        raise SystemExit(f"--adapter {ad.name}: {key}: {t.dtype} weights cannot take a merge (an FP8 "
                                         "checkpoint: fuse the bf16 repository)")
                    _check_shape(ad, key, t.shape)
                    t = merged(t, delta(ad, key))
                tensors[name] = t
        save_file(tensors, str(out / f.name), metadata=meta)
        del tensors
    for f in sorted(md.iterdir()):
        if f.is_file() and f.suffix != ".safetensors" and not any(f.match(s) for s in NOT_FUSED):
            shutil.copy2(f, out / f.name)
    return len(names)


def check_space(md, where, why):
    """Refuse (SystemExit, before anything is written) when the file system of `where` (or its nearest existing
    parent) has less free space than checkpoint md's weights plus 1 GB."""
    where = Path(where).resolve()
    while not where.exists():
        where = where.parent
    need = sum(f.stat().st_size for f in Path(md).glob("*.safetensors")) + (1 << 30)
    free = shutil.disk_usage(where).free
    if free < need:
        raise SystemExit(f"{why}: {need / 2 ** 30:.1f} GB needed in {where}, {free / 2 ** 30:.1f} GB free")


def fused_copy(md, ad):
    """--adapter for the engines that load a checkpoint themselves (vLLM, SGLang): the checkpoint with the adapter
    merged (fuse_checkpoint) in a temporary directory ($TMPDIR; the model's size on disk), removed when basal exits;
    the engine serves it as it serves basal-fuse-adapter's output -> its path. Written by a child process
    (basal-fuse-adapter): the server process never runs torch's CPU thread pool before vLLM forks its engine (a forked
    child deadlocks on that pool), and the shard it holds in memory is freed when the child exits."""
    import os
    import subprocess
    import sys
    check_space(md, tempfile.gettempdir(), f"--adapter {ad.name}: vLLM / SGLang serve a merged copy of the checkpoint "
                "(the size of its weights) in $TMPDIR; set TMPDIR to a directory with room for it, or serve a "
                "checkpoint written by basal-fuse-adapter")
    out = _temp_dir()
    here = str(Path(__file__).resolve().parent.parent)  # this basal package, whatever the child's working directory
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (here, os.environ.get("PYTHONPATH")) if x))
    r = subprocess.run([sys.executable, "-m", "basal.adapter", "--model", str(md), "--adapter", str(ad.path),
                        "--out", str(out)], capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit((r.stderr.strip().splitlines() or [f"basal-fuse-adapter exited with {r.returncode}"])[-1])
    print(f"basal: --adapter {ad.name}: {len(ad.deltas)} layers merged into a copy of the checkpoint at {out} "
          "(removed at exit)", flush=True)
    return out


def fused_calibration(cal, ad, out):
    """An adapter's calibration file for the merged checkpoint `out`: fitted_on names the merged model (no revision, no
    adapter) and keeps the origin as fused_from; None for a file fitted with another adapter (or none)."""
    fitted = cal.get("fitted_on")
    if not isinstance(fitted, dict) or not fitted.get("model"):
        return cal  # not written by basal-calibrate: copied as it is
    if fitted.get("adapter") != ad.identity:
        return None
    origin = fitted["model"] + (f"@{fitted['revision'][:12]}" if fitted.get("revision") else "")
    fitted = {k: v for k, v in fitted.items() if k not in ("revision", "adapter")}
    return dict(cal, fitted_on=dict(fitted, model=Path(out).resolve().name, fused_from=f"{origin}+{ad.identity}"))


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into a model and write the merged checkpoint (for "
                                 "the engines that cannot take --adapter: GGUF via basal-export-gguf, Ollama, "
                                 "llama.cpp, FP8 ports). The safetensors files are rewritten with the decoder layers' "
                                 "weights merged as --adapter merges them at load (same names and dtypes); the other "
                                 "files are copied, except the base model's CALIBRATION files.")
    ap.add_argument("--model", required=True, help="model directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (mlx-lm or PEFT)")
    ap.add_argument("--out", required=True, help="output directory (new or empty)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not out.is_dir():
        raise SystemExit(f"--out {out}: not a directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out}: exists and is not empty")
    from .engine import resolve
    ad = read(a.adapter)
    md = resolve(a.model, a.revision)
    check_space(md, out, f"--out {out}: the merged checkpoint has the size of the model's weights")
    n = fuse_checkpoint(md, ad, out)
    for f in sorted(ad.path.glob("CALIBRATION*.json")):  # fitted with this adapter
        cal = fused_calibration(json.loads(f.read_text(encoding="utf-8")), ad, out)
        if cal is None:
            print(f"basal-fuse-adapter: {f.name} was fitted with another adapter than {ad.identity}: not copied")
        else:
            (out / f.name).write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fused {n} layers of {ad.name} into {out}")


if __name__ == "__main__":
    main()
