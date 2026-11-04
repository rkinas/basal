"""One rule set for models derived from a basal model (basal/derive.py): basal-finetune's merged model, a fused adapter
and an interpolated checkpoint leave out every file that was fitted to the source weights or describes them, each
tool keeps or drops the source's calibrations by its own documented rule, and the tools that read with the plain
causal forward refuse the same models (on tiny checkpoints)."""
import json

import pytest

from basal import adapter as adapter_module
from basal import derive, finetune, gguf_export
from basal.interpolate import interpolate
from tests.test_adapter import bf16_llama, deltas, write_adapter
from tests.test_interpolate import make_ckpt

# a source folder's files under every rule (folders: a file inside them)
FITTED = ["NVFP4_AMAX.json", "evidence_head.pt", "exit_heads/heads.pt", "exit_heads/thresholds.json",
          "think/adapter/adapter_config.json", "think/config.json"]
DESCRIBES = ["README.md", "MANIFEST.md", "SHA256SUMS"]
CALIBRATIONS = ["CALIBRATION.json", "CALIBRATION.nvfp4.json", "CALIBRATION.think.json"]
KEPT = ["tokenizer.json", "chat_template.jinja", "processor_config.json"]


def fill(d):
    for rel in FITTED + DESCRIBES + CALIBRATIONS + KEPT:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (d / rel).exists():
            (d / rel).write_text("{}", encoding="utf-8")
    return d


def files(d):
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


def test_the_rules():
    for rel in FITTED + DESCRIBES + CALIBRATIONS:
        assert derive.stale(rel), rel
    assert not any(derive.stale(rel, calibrations=False) for rel in CALIBRATIONS)
    assert not any(derive.stale(rel) for rel in KEPT + ["config.json", "basal.json", "model.safetensors"])


def test_finetune_merged_model(tmp_path):
    src, out = fill(tmp_path / "src"), tmp_path / "out"
    src.mkdir(exist_ok=True), out.mkdir()
    finetune.copy_side_files(src, out)
    assert files(out) == set(KEPT)


def test_fused_adapter(tmp_path):
    md = tmp_path / "model"
    bf16_llama(md)
    fill(md)
    ad = adapter_module.read(write_adapter(tmp_path / "edu", "peft", deltas()))
    adapter_module.fuse_checkpoint(md, ad, tmp_path / "fused")
    got = files(tmp_path / "fused")
    assert set(KEPT) <= got and "config.json" in got and "basal.json" not in got  # (bf16_llama writes none)
    assert not got & set(FITTED + DESCRIBES + CALIBRATIONS)


def test_interpolated_checkpoint_keeps_the_calibrations_only(tmp_path):
    old, _ = make_ckpt(tmp_path / "old", 1)
    new, _ = make_ckpt(tmp_path / "new", 2)
    fill(new)
    got = files(interpolate(old, new, 0.5, tmp_path / "o"))
    assert set(KEPT + CALIBRATIONS) <= got
    assert not got & set(FITTED + DESCRIBES)  # (exit_heads/thresholds.json was copied without its heads.pt)


@pytest.mark.parametrize("key", ["two_way", "option_isolation"])
def test_plain_forward_tools_refuse_the_same_models(tmp_path, key):
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "layout": "canonical", key: True}))
    assert key in derive.reading_refused(tmp_path) and key in finetune.refused(tmp_path)
    with pytest.raises(ValueError, match="read one way"):
        gguf_export.systemone_metadata(tmp_path)
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "layout": "canonical"}))
    assert derive.reading_refused(tmp_path) is None and finetune.refused(tmp_path) is None
