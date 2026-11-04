"""Prepared start-up (basal/prepared.py): the key of a prepared directory, where --prepared auto finds one, and what
saving and loading write and read (torch's cache-artifact calls stubbed: the compiled kernels themselves are checked on
a GPU)."""
import argparse
import json

import pytest
import torch

from basal import gdn_graph, prepared
from basal.server import add_engine_args

ENV = dict(basal="2.5.0", torch="2.11.0", cuda="12.8", python="3.13.0", triton="3.5.0", fla="0.5.2", gpu="NVIDIA H100",
           capability=[9, 0])


def model(d, **cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "hidden_size": 1024, **cfg}))
    return d


def test_key_covers_what_compiled_code_depends_on(tmp_path):
    m = model(tmp_path / "m")
    f = prepared.key_fields(m, "fast", env=ENV)
    assert prepared.key(f) == prepared.key(prepared.key_fields(m, "fast", env=dict(ENV)))
    # the weights' location and the transformers version that wrote config.json do not change the compiled code
    same = model(tmp_path / "m2", _name_or_path="elsewhere", transformers_version="5.17.0")
    assert prepared.key(prepared.key_fields(same, "fast", env=ENV)) == prepared.key(f)
    for other in (prepared.key_fields(m, "fp8", env=ENV), prepared.key_fields(m, "fast", "float16", env=ENV),
                  prepared.key_fields(m, "fast", attn="fa3", env=ENV),
                  prepared.key_fields(m, "fast", env=dict(ENV, gpu="NVIDIA RTX PRO 6000")),
                  prepared.key_fields(m, "fast", env=dict(ENV, torch="2.12.0")),
                  prepared.key_fields(model(tmp_path / "m3", hidden_size=2048), "fast", env=ENV)):
        assert prepared.key(other) != prepared.key(f)


def write(d, fields):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(fields=fields, key=prepared.key(fields))))
    return d


def test_locate_model_dir_then_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    m = model(tmp_path / "m")
    f = prepared.key_fields(m, "fast", env=ENV)
    assert prepared.locate(m, f) is None
    c = write(tmp_path / "cache" / prepared.key(f), f)
    assert prepared.locate(m, f) == c
    shipped = write(m / "prepared" / prepared.key(f), f)
    assert prepared.locate(m, f) == shipped  # shipped with the model first
    assert prepared.locate(m, f, "off") is None
    assert prepared.locate(m, f, str(c)) == c
    other = prepared.key_fields(m, "fast", env=dict(ENV, gpu="NVIDIA A100"))
    assert prepared.locate(m, other) is None  # another GPU: nothing
    write(tmp_path / "x", other)
    assert prepared.locate(m, f, str(tmp_path / "x")) is None  # a directory for other fields is not loaded


def test_save_and_load(tmp_path, monkeypatch):
    """Loading writes the artifacts into the compile cache the engine then reads: ~/.cache/basal/torchinductor when
    TORCHINDUCTOR_CACHE_DIR holds Inductor's /tmp default (engine.persist_compile_cache, which the engine calls only
    when it is built, after the load)."""
    import os

    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    import basal
    calls = {}
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(tmp_path / "home"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", default_cache_dir())  # as Inductor sets it on import
    # basal-serve's case: nothing given before basal was imported (a test that loads Triton first sets the default)
    monkeypatch.setattr(basal, "INDUCTOR_CACHE_DIR_GIVEN", False)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"kernels", None))
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts",
                        lambda b: calls.setdefault("loaded", (b, os.environ["TORCHINDUCTOR_CACHE_DIR"])))
    monkeypatch.setattr(gdn_graph, "fla_record", lambda p: (p.write_text("{}"), 3)[1])
    f = dict(ENV, architecture="a", mode="fast", dtype="bfloat16", attn="auto")
    man = prepared.save(tmp_path / "p", f, fla=True)
    assert man["key"] == prepared.key(f) and man["saved"] == {"inductor": 0, "inductor_bytes": 7, "fla_kernels": 3}
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == sorted(prepared.FILES)
    monkeypatch.setattr(gdn_graph, "FLA_CONFIGS", gdn_graph.FLA_CONFIGS)  # restored after the test
    out = prepared.load(tmp_path / "p")
    assert calls["loaded"] == (b"kernels", str(tmp_path / "home" / ".cache" / "basal" / "torchinductor"))
    assert out["fla"] == str(tmp_path / "p" / "fla_configs.json")
    assert gdn_graph.FLA_CONFIGS == tmp_path / "p" / "fla_configs.json"  # fla_pin() reads the prepared table
    from torch._inductor.utils import clear_caches
    monkeypatch.undo()
    clear_caches()  # Inductor's memoised paths follow the restored environment again (later tests compile)


def test_server_flag():
    ap = add_engine_args(argparse.ArgumentParser())
    assert ap.parse_args([]).prepared == "auto"
    assert ap.parse_args(["--prepared", "off"]).prepared == "off"


def test_prepare_needs_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA GPU"):
        prepared.main(["--model", "x"])


def test_prepare_builds_the_engine_basal_serve_runs(tmp_path, monkeypatch):
    """basal-prepare starts the engine through basal-serve's Server with basal-serve's parse of the same flags (its
    defaults too: 2.5.2 built GraphBackend with the constructor's, padded rows without the template prefix, which
    basal-serve does not run on Hopper), loads no prepared directory itself, and writes where basal-serve's lookup for
    those flags finds it."""
    import basal.server as S
    m = model(tmp_path / "m")
    seen = []

    class Built:
        def __init__(self, a):
            seen.append(a)
            self.md = str(m)
    monkeypatch.setattr(S, "Server", Built)
    monkeypatch.setattr(S, "default_mode", lambda model, revision=None: "fast")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prepared, "environment", lambda: dict(ENV))
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    for flags in ([], ["--batches", "compact", "--compile-scope", "layer", "--attn", "fa2", "--mode", "fast"]):
        prepared.main(["--model", str(m)] + flags)
        want = S.parser().parse_args(["--model", str(m)] + flags + ["--prepared", "off"])
        want.mode = "fast"
        assert vars(seen[-1]) == vars(want)
        attn = "fa2" if flags else "auto"
        assert prepared.locate(m, prepared.key_fields(m, "fast", "bfloat16", attn)) is not None  # Server's lookup


def test_other_flag_sets_add_their_kernels_to_the_directory(tmp_path, monkeypatch):
    """The engine flags are not in the key: preparing the same model with basal-serve's defaults and with other flags
    writes one Inductor file each (inductor.bin, inductor-<flags>.bin) into the one directory, the manifest lists
    both, and a start loads both; preparing a flag set again replaces only its own file."""
    blobs, loaded = iter([b"defaults", b"compact", b"compact again"]), []
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (next(blobs), None))
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: loaded.append(b))
    f = dict(ENV, architecture="a", mode="fast", dtype="bfloat16", attn="auto")
    d = tmp_path / "p"
    prepared.save(d, f)
    compact = prepared.inductor_file({"batches": "compact"})
    prepared.save(d, f, flags={"batches": "compact"})
    man = prepared.save(d, f, flags={"batches": "compact"})
    assert man["files"] == sorted(["inductor.bin", compact, "manifest.json"]) and compact != "inductor.bin"
    assert man["flag_sets"] == {"inductor.bin": {}, compact: {"batches": "compact"}}
    assert (d / "inductor.bin").read_bytes() == b"defaults" and (d / compact).read_bytes() == b"compact again"
    assert prepared.locate(tmp_path, f, str(d)) == d
    out = prepared.load(d)
    assert sorted(loaded) == [b"compact again", b"defaults"] and set(out["inductor"]) == {"inductor.bin", compact}


def test_an_interrupted_save_leaves_the_previous_file(tmp_path):
    """Files of a prepared directory are written under a temporary name and renamed over the old one: a write that
    fails (an interrupted or overlapping save) leaves the previous file whole and no temporary file behind."""
    p = tmp_path / "fla_configs.json"
    p.write_text('{"kernels": {}}')

    def broken(tmp):
        tmp.write_text('{"ker')
        raise OSError("disk full")
    with pytest.raises(OSError):
        prepared._replace(p, broken)
    assert p.read_text() == '{"kernels": {}}' and sorted(x.name for x in tmp_path.iterdir()) == ["fla_configs.json"]


def test_prepare_with_think_builds_the_think_server(tmp_path, monkeypatch):
    """--think auto: basal-serve builds ThinkServer (its sidecar takes GPU memory before the graphs are sized), and so
    does basal-prepare."""
    import basal.server as S
    import basal.think as T
    m, built = model(tmp_path / "m"), []

    class Built:
        def __init__(self, a):
            built.append(type(self).__name__)
            self.md = str(m)
    monkeypatch.setattr(S, "Server", type("Server", (Built,), {}))
    monkeypatch.setattr(T, "ThinkServer", type("ThinkServer", (Built,), {}))
    monkeypatch.setattr(S, "default_mode", lambda model, revision=None: "fast")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prepared, "environment", lambda: dict(ENV))
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    prepared.main(["--model", str(m), "--think", "auto"])
    prepared.main(["--model", str(m)])
    assert built == ["ThinkServer", "Server"]


def gdn_model(d):
    return model(d, layer_types=["linear_attention", "full_attention"])


def test_key_of_a_shipped_directory(tmp_path, monkeypatch):
    """What a directory built on the release machine must share with a user's start: the Python minor version only
    (patch releases do not change compiled code); fla's version only for Qwen3.5 models, whose kernels it runs."""
    monkeypatch.setattr("platform.python_version_tuple", lambda: ("3", "12", "13"))
    assert prepared.environment()["python"] == "3.12"
    m, g = model(tmp_path / "m"), gdn_model(tmp_path / "g")
    assert "fla" not in prepared.key_fields(m, "fast", env=ENV)
    assert prepared.key(prepared.key_fields(m, "fast", env=ENV)) == prepared.key(
        prepared.key_fields(m, "fast", env=dict(ENV, fla=None)))  # a Gemma 4 / Llama start with or without fla
    assert prepared.key_fields(g, "fast", env=ENV)["fla"] == "0.5.2"
    assert prepared.key(prepared.key_fields(g, "fast", env=ENV)) != prepared.key(
        prepared.key_fields(g, "fast", env=dict(ENV, fla="0.5.3")))


def test_patch_release_keeps_the_key():
    """The basal version in the key is the compiled-code version: 2.5.1 changes no compiled code, so it keys as 2.5.0
    and loads the directories shipped for 2.5.0."""
    assert prepared.environment()["basal"] == prepared.PREPARED_ENGINE == "2.5.0"


def test_compiled_code_version_follows_the_release():
    """Guard: PREPARED_ENGINE must be bumped when basal's major.minor moves past it (a new minor release changes
    compiled code), and it is never ahead of the release."""
    import basal

    def ver(v):
        return tuple(int(x) for x in v.split(".")[:3])
    assert ver(prepared.PREPARED_ENGINE)[:2] == ver(basal.__version__)[:2], (
        f"basal {basal.__version__}: set prepared.PREPARED_ENGINE (now {prepared.PREPARED_ENGINE}) to this release")
    assert ver(prepared.PREPARED_ENGINE) <= ver(basal.__version__)


def test_fetch_from_the_model_repository(tmp_path, monkeypatch):
    """--prepared auto with a Hugging Face model: prepared/<key>/ of its repository only (the files its manifest
    lists), next to the snapshot; none when the repository has no such key, its fields differ or the Hub fails."""
    import huggingface_hub
    snap = gdn_model(tmp_path / "snap")
    f = prepared.key_fields(snap, "fast", env=ENV)
    k = prepared.key(f)
    repo = {f"prepared/{k}/manifest.json": json.dumps(dict(fields=f, key=k, files=["inductor.bin", "manifest.json"])),
            f"prepared/{k}/inductor.bin": "kernels"}
    asked = []

    def download(name, filename, revision=None):
        asked.append((name, filename, revision))
        if filename not in repo:
            raise FileNotFoundError(filename)
        p = snap / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(repo[filename])
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    d = prepared.fetch("Remek/x", "main", f)
    assert d == snap / "prepared" / k and (d / "inductor.bin").read_text() == "kernels"
    assert [a[1] for a in asked] == [f"prepared/{k}/manifest.json", f"prepared/{k}/inductor.bin"]
    assert prepared.locate(snap, f) == d  # the next start finds it in the snapshot
    other = prepared.key_fields(snap, "fast", env=dict(ENV, gpu="NVIDIA A100"))
    assert prepared.fetch("Remek/x", "main", other) is None  # not in the repository
    repo[f"prepared/{prepared.key(other)}/manifest.json"] = json.dumps(dict(fields=f, key=prepared.key(other)))
    assert prepared.fetch("Remek/x", "main", other) is None  # a manifest for other fields is not used


def test_download_leaves_prepared_out(monkeypatch, tmp_path):
    """Both downloads leave prepared/ out: the metadata one too, whose *.json would bring a manifest without its
    kernels into the snapshot (2.5.0: --prepared auto then found that directory and never fetched inductor.bin)."""
    import huggingface_hub
    from basal.engine import METADATA_FILES, resolve
    seen = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda name, **kw: (seen.update(kw), str(tmp_path))[1])
    resolve("Remek/x-not-local")
    assert seen["ignore_patterns"] == ["prepared/*"] and seen["allow_patterns"] is None
    resolve("Remek/x-not-local", metadata_only=True)
    assert seen["ignore_patterns"] == ["prepared/*"] and seen["allow_patterns"] == METADATA_FILES


def manifest_only(d, fields, files=("inductor.bin", "manifest.json")):
    """A prepared directory as a 2.5.0 metadata download left it in a snapshot: the manifest, none of its files."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(fields=fields, key=prepared.key(fields), files=list(files))))
    return d


def test_incomplete_directory_is_not_located(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    m = gdn_model(tmp_path / "m")
    f = prepared.key_fields(m, "fast", env=ENV)
    shipped = manifest_only(m / "prepared" / prepared.key(f), f, ("fla_configs.json", "inductor.bin", "manifest.json"))
    assert prepared.locate(m, f) is None
    assert prepared.locate(m, f, str(shipped)) is None  # a given path as well
    c = manifest_only(tmp_path / "cache" / prepared.key(f), f)
    (c / "inductor.bin").write_bytes(b"kernels")
    assert prepared.locate(m, f) == c  # the next candidate, complete
    (shipped / "inductor.bin").write_bytes(b"kernels")
    assert prepared.locate(m, f) == c  # fla_configs.json still missing
    (shipped / "fla_configs.json").write_text("{}")
    assert prepared.locate(m, f) == shipped


def test_start_fetches_what_the_snapshot_misses(tmp_path, monkeypatch):
    """basal-serve's path (startup) over a snapshot that holds the manifest only: the directory is fetched from the
    model's repository and its kernels loaded; a later start finds it complete and asks the Hub for nothing."""
    import huggingface_hub

    import basal.engine
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    snap = model(tmp_path / "snap")
    f = prepared.key_fields(snap, "fast", env=ENV)
    k = prepared.key(f)
    manifest_only(snap / "prepared" / k, f)
    repo = {f"prepared/{k}/manifest.json": (snap / "prepared" / k / "manifest.json").read_text(),
            f"prepared/{k}/inductor.bin": "kernels"}
    asked, loaded = [], []

    def download(name, filename, revision=None):
        asked.append(filename)
        p = snap / filename
        p.write_text(repo[filename])
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(basal.engine, "persist_compile_cache", lambda: None)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: loaded.append(b))
    out = prepared.startup(snap, f, "auto", "Remek/x", "main")
    assert asked == [f"prepared/{k}/manifest.json", f"prepared/{k}/inductor.bin"]
    assert loaded == [b"kernels"] and out == {"dir": str(snap / "prepared" / k), "inductor": 0}
    asked.clear()
    assert prepared.startup(snap, f, "auto", "Remek/x", "main")["dir"] == str(snap / "prepared" / k) and asked == []
    assert prepared.startup(snap, f, "off", "Remek/x", "main") is None and asked == []
    assert prepared.startup(snap, f, "auto", str(snap), None)["dir"] == str(snap / "prepared" / k)  # a local model
    with pytest.raises(SystemExit, match="no prepared start-up"):
        prepared.startup(snap, f, str(manifest_only(tmp_path / "given", f)), "Remek/x", "main")


def test_broken_inductor_file_is_left_out(tmp_path, monkeypatch):
    def broken(b):
        raise EOFError("truncated")
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", broken)
    (tmp_path / "inductor.bin").write_bytes(b"x")
    assert prepared.load(tmp_path)["inductor"] == "not loaded (EOFError)"


def test_broken_inductor_file_still_loads_the_fla_configs(tmp_path, monkeypatch):
    def broken(b):
        raise EOFError("truncated")
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", broken)
    monkeypatch.setattr(gdn_graph, "FLA_CONFIGS", None)
    (tmp_path / "inductor.bin").write_bytes(b"x")
    (tmp_path / "fla_configs.json").write_text("{}")
    out = prepared.load(tmp_path)
    assert out["inductor"] == "not loaded (EOFError)" and out["fla"] == str(tmp_path / "fla_configs.json")
    assert gdn_graph.FLA_CONFIGS == tmp_path / "fla_configs.json"


def test_save_lists_its_files(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"kernels", None))
    man = prepared.save(tmp_path / "p", dict(ENV, architecture="a", mode="fast", dtype="bfloat16", attn="auto"))
    assert man["files"] == ["inductor.bin", "manifest.json"]


def test_prepare_into_model_needs_a_local_directory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="local model directory"):
        prepared.main(["--model", "Remek/not-a-dir", "--into-model"])


def test_only_trusted_publishers_kernels_load_from_the_hub(tmp_path, monkeypatch, capsys):
    """Compiled kernels are code the compiler cache runs: --prepared auto loads those a Hugging Face repository ships
    (fetched, or in its snapshot) only for a trusted publisher (BASAL_PREPARED_TRUST, default Remek); for others only
    what basal-prepare wrote into the user cache on this machine. A local model directory is trusted."""
    import huggingface_hub

    import basal.engine
    monkeypatch.setenv("BASAL_PREPARED_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("BASAL_PREPARED_TRUST", raising=False)
    snap = model(tmp_path / "snap")
    f = prepared.key_fields(snap, "fast", env=ENV)
    k = prepared.key(f)
    d = snap / "prepared" / k
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(fields=f, key=k, files=["inductor.bin", "manifest.json"])))
    (d / "inductor.bin").write_text("kernels")
    asked, loaded = [], []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: asked.append(a) or (_ for _ in ()).throw(
        FileNotFoundError("none")))
    monkeypatch.setattr(basal.engine, "persist_compile_cache", lambda: None)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: loaded.append(b))
    assert prepared.startup(snap, f, "auto", "someone/x", "main") is None  # its snapshot's kernels: not loaded
    assert loaded == [] and asked == [] and "not trusted" in capsys.readouterr().out
    assert prepared.startup(snap, f, "auto", "Remek/x", "main")["dir"] == str(d) and loaded == [b"kernels"]
    monkeypatch.setenv("BASAL_PREPARED_TRUST", "someone, other")
    assert prepared.startup(snap, f, "auto", "someone/x", "main")["dir"] == str(d)
    monkeypatch.setenv("BASAL_PREPARED_TRUST", "")
    assert prepared.startup(snap, f, "auto", "Remek/x", "main") is None  # nobody trusted
    assert prepared.startup(snap, f, "auto", str(snap), None)["dir"] == str(d)  # a local model directory
    own = tmp_path / "cache" / k  # what basal-prepare wrote on this machine loads for any publisher
    own.mkdir(parents=True)
    (own / "manifest.json").write_text(json.dumps(dict(fields=f, key=k, files=["manifest.json"])))
    assert prepared.startup(snap, f, "auto", "someone/x", "main")["dir"] == str(own)
