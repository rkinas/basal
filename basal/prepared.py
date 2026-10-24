"""Prepared start-up (2.5): what the CUDA graph engine computes at its first start on a GPU class, saved once and loaded
at every later start, so a server starts warm and with the same kernels every time.

A start compiles the engine's layer functions (torch.compile / Inductor: code generation and the autotuning of its
kernels by timing) and, for Qwen3.5 models, times fla's Triton kernels at their first launch. That is most of the
150-600 s a server needs before it answers, and the timing can pick other kernels in another start (other last bits).
`basal-prepare` starts the engine once (its warm-up runs every graph shape) and writes into one directory:

- inductor.bin: Inductor's cache artifacts (torch.compiler.save_cache_artifacts: compiled graphs, Triton kernels and
  autotuning results);
- fla_configs.json (Qwen3.5 models): the configs fla's kernels picked (gdn_graph.fla_record);
- manifest.json: the key below and what was saved.

`basal-serve --prepared auto` (the default) loads a directory whose key matches before the engine is built: from the
model directory (prepared/<key>/, shipped with a model) or the user cache (~/.cache/basal/prepared/<key>/, where
basal-prepare writes). The key covers what the compiled code depends on: the GPU (name and compute capability), the
torch / triton / fla / CUDA versions, the basal version, the model's architecture (config.json, not its weights: they
are inputs of the compiled functions), the mode, dtype and attention. Any difference: nothing is loaded and the engine
compiles as before. CUDA graphs are still captured at every start (they hold device addresses).
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

FILES = ("inductor.bin", "fla_configs.json", "manifest.json")


def cache_root():
    return Path(os.environ.get("BASAL_PREPARED_DIR") or Path.home() / ".cache" / "basal" / "prepared")


def environment():
    """What compiled code depends on besides the model: GPU, library versions, basal."""
    from . import __version__
    env = dict(basal=__version__, torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version())
    try:
        import triton
        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    try:
        import fla
        env["fla"] = fla.__version__
    except ImportError:
        env["fla"] = None
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name()
        env["capability"] = list(torch.cuda.get_device_capability())
    else:
        env["gpu"], env["capability"] = None, None
    return env


def architecture(model_dir):
    """sha256 of the model's config.json with its keys sorted (the architecture; the weights do not enter the key)."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    for k in ("_name_or_path", "transformers_version"):
        cfg.pop(k, None)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def key_fields(model_dir, mode, dtype="bfloat16", attn="auto", env=None):
    return dict(env or environment(), architecture=architecture(model_dir), mode=mode, dtype=dtype, attn=attn)


def key(fields):
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:20]


def locate(model_dir, fields, spec="auto"):
    """The prepared directory to load: spec "off" -> None; a path -> that directory if its manifest has these fields;
    "auto" -> <model_dir>/prepared/<key>, then <cache_root>/<key>, the first whose manifest matches."""
    if spec in (None, "off"):
        return None
    k = key(fields)
    cands = [Path(spec)] if spec != "auto" else [Path(model_dir) / "prepared" / k, cache_root() / k]
    for d in cands:
        m = d / "manifest.json"
        if m.exists():
            man = json.loads(m.read_text(encoding="utf-8"))
            if man.get("fields") == fields:
                return d
    return None


def load(d):
    """Loads a prepared directory before the engine compiles -> what was loaded (for the start-up line)."""
    out = {"dir": str(d)}
    f = Path(d) / "inductor.bin"
    if f.exists():
        info = torch.compiler.load_cache_artifacts(f.read_bytes())
        out["inductor"] = _count(info)
    f = Path(d) / "fla_configs.json"
    if f.exists():
        from . import gdn_graph
        gdn_graph.FLA_CONFIGS = f  # fla_pin reads it when the Qwen3.5 engine is built
        out["fla"] = str(f)
    return out


def _count(info):
    """Artifacts per kind of a torch CacheInfo (a dict of lists in recent torch)."""
    if info is None:
        return 0
    d = getattr(info, "artifacts", None)
    if isinstance(d, dict):
        return sum(len(v) for v in d.values())
    return 1


def save(d, fields, fla=False):
    """Writes the artifacts of this process's compiles (and fla's picked configs) into d with the manifest."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    saved = {}
    got = torch.compiler.save_cache_artifacts()
    if got is not None:
        data, info = got
        (d / "inductor.bin").write_bytes(data)
        saved["inductor"] = _count(info)
        saved["inductor_bytes"] = len(data)
    if fla:
        from .gdn_graph import fla_record
        saved["fla_kernels"] = fla_record(d / "fla_configs.json")
    man = dict(fields=fields, key=key(fields), saved=saved, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def main(argv=None):
    """basal-prepare: start the engine once (its warm-up compiles and captures every graph shape) and save what a later
    start of the same model architecture, mode and environment can load (basal-serve --prepared auto)."""
    ap = argparse.ArgumentParser(description="prepare the start-up of the CUDA graph engine for this GPU")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", default="fast", choices=["fast", "fp8", "nvfp4"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--out", default=None, help="directory to write (default: the user cache, ~/.cache/basal/prepared/"
                                                 "<key>, where basal-serve --prepared auto looks)")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("basal-prepare needs a CUDA GPU (the prepared kernels are for the GPU it runs on)")
    from .engine import GraphBackend, resolve
    from .gdn_graph import is_gdn
    md = resolve(a.model, a.revision)
    fields = key_fields(md, a.mode, a.dtype, a.attn)
    t0 = time.time()
    quant = {"fast": None, "fp8": "fp8", "nvfp4": "nvfp4"}[a.mode]
    GraphBackend(md, a.dtype, quant, compile=True, attn=a.attn)  # compiles, autotunes and captures every shape
    out = Path(a.out) if a.out else cache_root() / key(fields)
    man = save(out, fields, fla=is_gdn(md))
    print(json.dumps(dict(out=str(out), start_s=round(time.time() - t0, 1), **man), indent=1))


if __name__ == "__main__":
    main()
