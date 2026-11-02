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
torch / triton / CUDA versions (fla's for Qwen3.5 models), the Python and basal versions, the model's architecture
(config.json, not its weights: they are inputs of the compiled functions), the mode, dtype and attention. Any
difference: nothing is loaded and the engine compiles as before. CUDA graphs are still captured at every start (they
hold device addresses). Inductor checks every loaded entry against its own key as well, so a directory that matches
but holds kernels for another graph only leaves the start cold: answers never depend on what was loaded.

Shipped with a model: a Hugging Face model repository holds prepared/<key>/ per GPU class and environment it was
built for (basal-prepare --into-model). Downloading the model leaves prepared/ out (resolve); a start with --prepared
auto fetches the one directory of its own key, if the repository has it (fetch).
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
    python = ".".join(platform.python_version_tuple()[:2])  # (a patch release does not change compiled code)
    env = dict(basal=__version__, torch=torch.__version__, cuda=torch.version.cuda, python=python)
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
    from .gdn_graph import is_gdn
    env = dict(env or environment())
    if not is_gdn(model_dir):  # fla's kernels run in Qwen3.5 models only: whether it is installed does not matter
        env.pop("fla", None)
    return dict(env, architecture=architecture(model_dir), mode=mode, dtype=dtype, attn=attn)


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


def fetch(repo, revision, fields):
    """prepared/<key>/ of a Hugging Face model repository, downloaded next to the model's files (its snapshot) ->
    that directory, or None when the repository has none for this key or the Hub cannot be reached (the engine then
    compiles as before; with HF_HUB_OFFLINE a directory fetched before is used)."""
    k = key(fields)
    try:
        from huggingface_hub import hf_hub_download
        m = Path(hf_hub_download(repo, f"prepared/{k}/manifest.json", revision=revision))
        man = json.loads(m.read_text(encoding="utf-8"))
    except Exception:  # (no such file, offline, no network)
        return None
    if man.get("fields") != fields:
        return None
    for f in man.get("files", FILES):
        if f != "manifest.json":
            try:
                hf_hub_download(repo, f"prepared/{k}/{f}", revision=revision)
            except Exception:  # (a file this directory does not have; load reads what arrived)
                pass
    return m.parent


def load(d):
    """Loads a prepared directory before the engine compiles -> what was loaded (for the start-up line). Inductor's
    cache directory is set first, as the compiling engine sets it (engine.persist_compile_cache): the artifacts are
    written into that directory, so they must land where the engine then looks. A file that cannot be read (a broken
    download) is left out: those kernels compile as without it."""
    from .engine import persist_compile_cache
    persist_compile_cache()
    out = {"dir": str(d)}
    f = Path(d) / "inductor.bin"
    if f.exists():
        try:
            info = torch.compiler.load_cache_artifacts(f.read_bytes())
        except Exception as e:  # noqa: BLE001
            out["inductor"] = f"not loaded ({type(e).__name__})"
            return out
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
    files = sorted(p.name for p in d.iterdir() if p.name in FILES) + ["manifest.json"]
    man = dict(fields=fields, key=key(fields), saved=saved, files=sorted(set(files)),
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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
    ap.add_argument("--into-model", dest="into_model", action="store_true",
                    help="write into the local model directory, as prepared/<key> (to ship with the model: basal-serve "
                         "--prepared auto finds it there, also in the model's Hugging Face repository)")
    a = ap.parse_args(argv)
    if a.into_model and (a.out or not Path(a.model).is_dir()):
        raise SystemExit("--into-model: give --model as a local model directory, without --out")
    if not torch.cuda.is_available():
        raise SystemExit("basal-prepare needs a CUDA GPU (the prepared kernels are for the GPU it runs on)")
    from .engine import GraphBackend, resolve
    from .gdn_graph import is_gdn
    md = resolve(a.model, a.revision)
    fields = key_fields(md, a.mode, a.dtype, a.attn)
    t0 = time.time()
    quant = {"fast": None, "fp8": "fp8", "nvfp4": "nvfp4"}[a.mode]
    GraphBackend(md, a.dtype, quant, compile=True, attn=a.attn)  # compiles, autotunes and captures every shape
    out = Path(a.out) if a.out else (Path(md) / "prepared" if a.into_model else cache_root()) / key(fields)
    man = save(out, fields, fla=is_gdn(md))
    print(json.dumps(dict(out=str(out), start_s=round(time.time() - t0, 1), **man), indent=1))


if __name__ == "__main__":
    main()
