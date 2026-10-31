"""LoRA adapters: read an mlx-lm or PEFT adapter directory, merge it into a model's weights at load (torch, MLX, the
TPU backend's numpy weights), write it as a PEFT directory (vLLM, SGLang), and fuse it into a checkpoint
(basal-fuse-adapter) for the engines that serve weights basal cannot touch (GGUF, Ollama, llama.cpp).

Formats, by the files in the directory:
  mlx-lm  adapter_config.json (fine_tune_type "lora" or absent, lora_parameters.scale) + adapters.safetensors,
          keys <path>.lora_a [in, r] / <path>.lora_b [r, out]
  PEFT    adapter_config.json (peft_type "LORA", r, lora_alpha, use_rslora) + adapter_model.safetensors,
          keys base_model.model.<path>.lora_A.weight [r, in] / .lora_B.weight [out, r]

The merge is mlx-lm's LoRALinear.fuse on every framework: delta = (scale * b) @ a in float32 ([out, in]), cast to the
weight's dtype, added in the weight's dtype. The factors are float32 whatever their stored dtype (PEFT's own merge
multiplies them in the weight's dtype): one rounding, at the cast, on every framework."""
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
    from safetensors.numpy import load_file
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
        v = np.asarray(v, dtype=np.float32)
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


def delta(ad, key, np_dtype=np.float32):
    """The update of one module, (scale * b) @ a in float32 on the CPU ([out, in]), cast to np_dtype. Every merge (torch
    on any device, MLX, the TPU's numpy weights, basal-fuse-adapter) takes it from here, so --adapter and the fused
    checkpoint get bit-identical weights whatever GEMM the device would have used."""
    a, b = ad.deltas[key]
    return ((np.float32(ad.scale) * b) @ a).astype(np_dtype)


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
            _check_shape(ad, key, w.shape)
            d = torch.from_numpy(delta(ad, key)).to(w.dtype)  # rounded on the CPU, then moved (bf16)
            w.add_(d.to(w.device))
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
        w = w + mx.array(delta(ad, key)).astype(w.dtype)
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
        d = delta(ad, key, w.dtype)
        weights[name] = (w.astype(np.float32) + d.astype(np.float32)).astype(w.dtype)
    return len(ad.deltas)


def _temp_dir():
    out = Path(tempfile.mkdtemp(prefix="basal-adapter-"))
    atexit.register(shutil.rmtree, out, True)  # read by the engine's own process until basal-serve exits
    return out


def peft_dir(ad, model_dir=None):
    """A PEFT directory of the adapter (vLLM, SGLang): its own for a PEFT adapter, else a converted copy in a temporary
    directory ($TMPDIR, removed at exit). An rsLoRA PEFT adapter gets a copy whose adapter_config.json states the same
    scale as lora_alpha / r (SGLang ignores use_rslora; vLLM reads both the same way), its tensors linked unchanged. The
    converted copy names the modules base_model.model.model.layers.N.<module>, the Hugging Face path of a text decoder;
    a multimodal model (config.json text_config) nests its decoder elsewhere, so an mlx-lm adapter for one is refused
    (fuse it with basal-fuse-adapter, or give a PEFT adapter of that model)."""
    if ad.fmt == "peft":
        cfg = json.loads((ad.path / CONFIG).read_text(encoding="utf-8"))
        if not cfg.get("use_rslora"):
            return ad.path
        out = _temp_dir()
        for f in ad.path.iterdir():
            if f.is_file() and f.name != CONFIG:
                (out / f.name).symlink_to(f.resolve())
        cfg.update(use_rslora=False, lora_alpha=ad.scale * cfg["r"])
        (out / CONFIG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return out
    cfg = Path(model_dir) / "config.json" if model_dir else None
    if cfg and cfg.is_file() and "text_config" in json.loads(cfg.read_text(encoding="utf-8")):
        raise SystemExit(f"--adapter {ad.name}: an mlx-lm adapter of a multimodal model cannot be given to vLLM / SGLang "
                         "(its module names differ from the Hugging Face model's); merge it with basal-fuse-adapter or "
                         "use a PEFT adapter of the model")
    from safetensors.numpy import save_file
    tensors = {}
    for key, (a, b) in ad.deltas.items():
        if a.shape[0] != ad.rank:
            raise SystemExit(f"--adapter {ad.name}: {key}: rank {a.shape[0]} differs from the adapter's {ad.rank}; "
                             "a PEFT directory has one rank")
        tensors[f"base_model.model.model.{key}.lora_A.weight"] = a
        tensors[f"base_model.model.model.{key}.lora_B.weight"] = b
    out = _temp_dir()
    cfg = {"peft_type": "LORA", "r": ad.rank, "lora_alpha": ad.scale * ad.rank,
           "target_modules": sorted({k.rsplit(".", 1)[-1] for k in ad.deltas}), "bias": "none",
           "task_type": "CAUSAL_LM", "fan_in_fan_out": False, "use_rslora": False}
    (out / CONFIG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file(tensors, str(out / PEFT_WEIGHTS))
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
                                 "llama.cpp, FP8 ports).")
    ap.add_argument("--model", required=True, help="model directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (mlx-lm or PEFT)")
    ap.add_argument("--out", required=True, help="output directory (new or empty)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not out.is_dir():
        raise SystemExit(f"--out {out}: not a directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out}: exists and is not empty")
    import torch
    from transformers import AutoModelForCausalLM

    from .engine import resolve
    ad = read(a.adapter)
    md = resolve(a.model, a.revision)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=getattr(torch, a.dtype))
    n = merge_torch(model, ad)
    model.save_pretrained(out)
    skip = ("*.safetensors*", "CALIBRATION*", "SHA256SUMS")
    for f in sorted(Path(md).iterdir()):  # tokenizer, chat template, basal.json; save_pretrained wrote config.json
        if f.is_file() and not any(f.match(s) for s in skip) and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)
    for f in sorted(ad.path.glob("CALIBRATION*.json")):  # fitted with this adapter
        cal = fused_calibration(json.loads(f.read_text(encoding="utf-8")), ad, out)
        if cal is None:
            print(f"basal-fuse-adapter: {f.name} was fitted with another adapter than {ad.identity}: not copied")
        else:
            (out / f.name).write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fused {n} layers of {ad.name} into {out}")


if __name__ == "__main__":
    main()
