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
torch / triton / CUDA versions (fla's for Qwen3.5 models), the Python version and the basal release whose compiled
code this is (PREPARED_ENGINE), the model's architecture
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
# The compiled-code version, the basal version in the key: bump it when a release changes compiled / graph code (the
# engine's layer functions); a patch that only changes Python around it keeps it, so the directories shipped in the
# model repositories still load (2.5.1 keeps 2.5.0's). A new minor release sets its own (tests/test_prepared.py).
PREPARED_ENGINE = "2.5.0"


def cache_root():
    return Path(os.environ.get("BASAL_PREPARED_DIR") or Path.home() / ".cache" / "basal" / "prepared")


def environment():
    """What compiled code depends on besides the model: GPU, library versions, basal (PREPARED_ENGINE)."""
    python = ".".join(platform.python_version_tuple()[:2])  # (a patch release does not change compiled code)
    env = dict(basal=PREPARED_ENGINE, torch=torch.__version__, cuda=torch.version.cuda, python=python)
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
    "auto" -> <model_dir>/prepared/<key>, then <cache_root>/<key>, the first whose manifest matches. A directory
    counts only when every file its manifest lists is there: one that holds its manifest alone (as a metadata download
    of basal 2.5.0 left it in a model's snapshot) is not loaded, so --prepared auto fetches the rest (fetch)."""
    if spec in (None, "off"):
        return None
    k = key(fields)
    cands = [Path(spec)] if spec != "auto" else [Path(model_dir) / "prepared" / k, cache_root() / k]
    for d in cands:
        m = d / "manifest.json"
        if m.exists():
            man = json.loads(m.read_text(encoding="utf-8"))
            if man.get("fields") == fields and all((d / f).exists() for f in man.get("files", ())):
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


TRUSTED = ("Remek",)  # publishers whose repositories' prepared kernels --prepared auto loads (BASAL_PREPARED_TRUST)


def trusted(repo):
    """Whether --prepared auto loads the prepared kernels a Hugging Face repository ships: compiled kernels are code the
    compiler cache runs, a trust beyond reading tensor weights, so only for the publishers in BASAL_PREPARED_TRUST
    (comma-separated owners, "*" = every one; default TRUSTED). A local model directory or --prepared <dir> is
    trusted by whoever points to it; kernels basal-prepare wrote into the user cache are local too."""
    env = os.environ.get("BASAL_PREPARED_TRUST")
    owners = TRUSTED if env is None else tuple(o.strip() for o in env.split(",") if o.strip())
    return "*" in owners or str(repo).split("/")[0] in owners


def startup(model_dir, fields, spec="auto", model=None, revision=None):
    """basal-serve's prepared start-up -> what was loaded (load), or None: the directory locate finds, else with spec
    "auto" and a Hugging Face model (`model` not a local path) the one fetched from its repository. A Hugging Face
    model of an untrusted publisher (trusted) loads neither: only the user cache. A given path that holds none for these
    fields stops the start."""
    hub = spec == "auto" and model is not None and not Path(model).exists()
    if hub and not trusted(model):
        d = locate(None, fields, str(cache_root() / key(fields)))  # (basal-prepare's own, on this machine)
        if d is None:
            print(f"basal: prepared kernels of {model} not loaded: its publisher is not trusted "
                  f"(BASAL_PREPARED_TRUST={str(model).split('/')[0]} trusts it)", flush=True)
        return load(d) if d is not None else None
    d = locate(model_dir, fields, spec)
    if d is None and hub:
        d = fetch(model, revision, fields)  # shipped in the model's repository
    if d is None and spec not in ("auto", "off", None):
        raise SystemExit(f"--prepared {spec}: no prepared start-up for this model, mode and GPU there "
                         f"(key {key(fields)}; basal-prepare writes one)")
    return load(d) if d is not None else None


def load(d):
    """Loads a prepared directory before the engine compiles -> what was loaded (for the start-up line). Inductor's
    cache directory is set first, as the compiling engine sets it (engine.persist_compile_cache): the artifacts are
    written into that directory, so they must land where the engine then looks. A file that cannot be read (a broken
    download) is left out: those kernels compile as without it. Every Inductor file is loaded (inductor.bin: the
    kernels of basal-serve's default flags; inductor-<flags>.bin: of other flag sets, save): Inductor checks each
    entry against its own key, so kernels of flags this start does not use only cost the reading."""
    from .engine import persist_compile_cache
    persist_compile_cache()
    out = {"dir": str(d)}
    files, res = sorted(Path(d).glob("inductor*.bin")), []
    for f in files:
        try:
            res.append(_count(torch.compiler.load_cache_artifacts(f.read_bytes())))
        except Exception as e:  # noqa: BLE001  (the fla configs below are loaded all the same)
            res.append(f"not loaded ({type(e).__name__})")
    if res:  # (several flag sets: per file)
        out["inductor"] = res[0] if len(res) == 1 else {f.name: r for f, r in zip(files, res)}
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


def inductor_file(flags=None):
    """The Inductor file of one engine flag set in a prepared directory: inductor.bin for basal-serve's defaults (as
    2.5.0 named it), inductor-<hash of the flags>.bin for others, so preparing one model with several flag sets
    (the same key: the flags are not in it) adds their kernels instead of replacing them."""
    if not flags:
        return "inductor.bin"
    h = hashlib.sha256(json.dumps(flags, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"inductor-{h}.bin"


def _replace(path, write):
    """write(tmp) then rename it over path: an interrupted or concurrent save never leaves a truncated file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        out = write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def save(d, fields, fla=False, flags=None):
    """Writes the artifacts of this process's compiles (and fla's picked configs) into d with the manifest; flags:
    the engine flags that differ from basal-serve's defaults (inductor_file). A directory prepared before for the
    same key keeps its other flag sets' files."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    saved = {}
    got = torch.compiler.save_cache_artifacts()
    if got is not None:
        data, info = got
        _replace(d / inductor_file(flags), lambda p: p.write_bytes(data))
        saved["inductor"] = _count(info)
        saved["inductor_bytes"] = len(data)
    if fla:
        from .gdn_graph import fla_record
        saved["fla_kernels"] = _replace(d / "fla_configs.json", fla_record)
    try:
        prev = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    sets = prev.get("flag_sets", {}) if prev.get("fields") == fields else {}
    sets[inductor_file(flags)] = flags or {}
    files = sorted(p.name for p in d.iterdir() if p.name in FILES or p.name in sets) + ["manifest.json"]
    man = dict(fields=fields, key=key(fields), saved=saved, files=sorted(set(files)), flag_sets=sets,
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _replace(d / "manifest.json", lambda p: p.write_text(json.dumps(man, indent=1, sort_keys=True)))
    return man


def main(argv=None):
    """basal-prepare: start the engine once as basal-serve starts it with the same flags (its warm-up compiles and
    captures every graph shape) and save what a later start of the same model architecture, mode and environment can
    load (basal-serve --prepared auto). Every option but --out / --into-model is basal-serve's, and the engine is built
    by basal-serve's Server: the compiled code depends on the engine flags (--batches, --template-prefix,
    --compile-scope, --merge, ...), so the prepared kernels are those of the configuration the server runs, its
    defaults included (on Hopper: compact rows with the template prefix)."""
    ap = argparse.ArgumentParser(description="prepare the start-up of the CUDA graph engine for this GPU",
                                 epilog="Every other option is basal-serve's (basal-serve --help): give the model and "
                                        "the engine flags the server will run with.")
    ap.add_argument("--out", default=None, help="directory to write (default: the user cache, ~/.cache/basal/prepared/"
                                                 "<key>, where basal-serve --prepared auto looks)")
    ap.add_argument("--into-model", dest="into_model", action="store_true",
                    help="write into the local model directory, as prepared/<key> (to ship with the model: basal-serve "
                         "--prepared auto finds it there, also in the model's Hugging Face repository)")
    a, rest = ap.parse_known_args(argv)
    from .server import MODES, Server, add_engine_args, default_mode, parser, refuse_lean_flex
    s = parser().parse_args(rest + ["--prepared", "off"])  # (this start compiles what a later one loads)
    if a.into_model and (a.out or not Path(s.model).is_dir()):
        raise SystemExit("--into-model: give --model as a local model directory, without --out")
    if not torch.cuda.is_available():
        raise SystemExit("basal-prepare needs a CUDA GPU (the prepared kernels are for the GPU it runs on)")
    refuse_lean_flex(s.lean, s.tree)  # as basal-serve's main, before default_mode
    s.mode = s.mode or default_mode(s.model, s.revision)
    kind, _, comp = MODES[s.mode]
    if kind not in ("graph", "exit") or not comp:
        raise SystemExit(f"basal-prepare: --mode {s.mode} compiles nothing to prepare (the CUDA graph engine's "
                         "compiled modes: fast, fp8, nvfp4, ...)")
    defaults = vars(add_engine_args(argparse.ArgumentParser()).parse_args([]))
    keyed = ("model", "revision", "mode", "dtype", "attn", "prepared")  # (mode, dtype and attention are in the key)
    flags = {k: v for k, v in sorted(vars(s).items()) if k in defaults and k not in keyed and v != defaults[k]}
    from .gdn_graph import is_gdn
    t0 = time.time()
    if s.think == "off":  # checks the flags as basal-serve does, then compiles, autotunes and captures every shape
        srv = Server(s)
    else:  # its sidecar takes GPU memory before the graphs are sized, as in basal-serve
        from .think import ThinkServer
        srv = ThinkServer(s)
    fields = key_fields(srv.md, s.mode, s.dtype, s.attn)  # the key basal-serve computes for these flags
    out = Path(a.out) if a.out else (Path(srv.md) / "prepared" if a.into_model else cache_root()) / key(fields)
    man = save(out, fields, fla=is_gdn(srv.md), flags=flags)
    print(json.dumps(dict(out=str(out), start_s=round(time.time() - t0, 1), **man), indent=1))


if __name__ == "__main__":
    main()
