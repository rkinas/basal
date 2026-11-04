"""GraphBackend's option checks as one pure function (engine.graph_options): every refusal that needs no weights,
tested on a model folder holding only its config (nothing loaded)."""
import json

import pytest

from basal import engine
from basal.engine import graph_options


def model(tmp_path, cfg, basal=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    if basal is not None:
        (tmp_path / "basal.json").write_text(json.dumps(basal))
    return str(tmp_path)


@pytest.fixture(autouse=True)
def cpu_machine(monkeypatch):
    monkeypatch.setattr(engine, "resolve_attention", lambda choice="auto", fp8=False: "fa2")
    monkeypatch.setattr(engine, "fp8_refused", lambda: None)


LLAMA = {"model_type": "llama", "hidden_size": 64}
GEMMA = {"model_type": "gemma4"}


@pytest.mark.parametrize("cfg, basal, kw, why", [
    (GEMMA, {"two_way": True}, dict(state_cache_gb=2), "--state-cache-gb"),
    (LLAMA, None, dict(quant="fp8", fp8_scaling="block"), "--fp8-scaling block: --mode fp8 with basal-omni"),
    (LLAMA, None, dict(quant="int8"), "--mode int8"),
    (GEMMA, None, dict(compile="layer"), "--compile-scope layer: the bf16 Llama forward only, not Gemma 4"),
    (LLAMA, None, dict(compile="layer", lean=True), "not --lean"),
    (LLAMA, None, dict(fp8_bf16_layers=-1), "--fp8-bf16-layers -1"),
])
def test_refusals(tmp_path, cfg, basal, kw, why, monkeypatch):
    monkeypatch.setattr("basal.int8block.int8_refused", lambda: None)
    with pytest.raises(ValueError, match=why):
        graph_options(model(tmp_path, cfg, basal), **kw)


def test_resolved_values(tmp_path):
    opt = graph_options(model(tmp_path, GEMMA, {"two_way": True}), batches="auto")
    assert opt.two_way == {"max_prompt": 4096, "max_record": 8192} and opt.batches == "default" and opt.attn == "fa2"
    opt = graph_options(model(tmp_path, LLAMA, {}), batches="auto")
    assert opt.two_way is None and opt.batches == "auto" and not opt.fp8_checkpoint
