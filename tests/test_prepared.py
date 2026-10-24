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
    calls = {}
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"kernels", None))
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: calls.setdefault("loaded", b))
    monkeypatch.setattr(gdn_graph, "fla_record", lambda p: (p.write_text("{}"), 3)[1])
    f = dict(ENV, architecture="a", mode="fast", dtype="bfloat16", attn="auto")
    man = prepared.save(tmp_path / "p", f, fla=True)
    assert man["key"] == prepared.key(f) and man["saved"] == {"inductor": 0, "inductor_bytes": 7, "fla_kernels": 3}
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == sorted(prepared.FILES)
    monkeypatch.setattr(gdn_graph, "FLA_CONFIGS", gdn_graph.FLA_CONFIGS)  # restored after the test
    out = prepared.load(tmp_path / "p")
    assert calls["loaded"] == b"kernels" and out["fla"] == str(tmp_path / "p" / "fla_configs.json")
    assert gdn_graph.FLA_CONFIGS == tmp_path / "p" / "fla_configs.json"  # fla_pin() reads the prepared table


def test_server_flag():
    ap = add_engine_args(argparse.ArgumentParser())
    assert ap.parse_args([]).prepared == "auto"
    assert ap.parse_args(["--prepared", "off"]).prepared == "off"


def test_prepare_needs_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA GPU"):
        prepared.main(["--model", "x"])
