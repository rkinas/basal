"""Request handling of the HTTP server without a model: option names reach the prompt, and responses and the model
list conform to the official System One OpenAPI schema (tests/data/systemone_openapi.json)."""
import asyncio
import json
import sys
from pathlib import Path

import jsonschema
import pytest

import basal.bench as bench_module
import basal.server as server_module
from basal.server import Server, default_mode, models_payload, named_options, to_items

SPEC = json.loads((Path(__file__).parent / "data/systemone_openapi.json").read_text())


def schema(name):
    s = json.loads(json.dumps(SPEC["components"]["schemas"][name]).replace("#/components/schemas/", "#/$defs/"))
    s["$defs"] = json.loads(json.dumps(SPEC["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return s


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) for c in text]})()


class FakeServer(Server):
    """Server.decide with a stub backend that returns fixed position probabilities for every job."""
    def __init__(self, pos_probs):
        self.name, self.temps, self.tok, self.letters, self.pos_probs = "basal-test", {}, Tok(), {}, pos_probs
        self.orders = 2
        self.backend = type("B", (), {"policies": {}})()

    async def _run(self, body):
        self.queue = asyncio.Queue()

        async def worker():
            while True:
                prompts, ids, fut, _ = await self.queue.get()
                fut.set_result([self.pos_probs[: len(i)] for i in ids])
        w = asyncio.get_running_loop().create_task(worker())
        try:
            return await self.decide(body)
        finally:
            w.cancel()


def test_default_mode_prefers_cuda_over_mlx(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    assert default_mode() == "fast"


def test_benchmark_mps_default_avoids_eager_fp32(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda: "mlx")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench_module.torch.backends.mps, "is_available", lambda: True)
    assert bench_module.default_modes() == ["mps", "mlx"]


def test_benchmark_cuda_default_preserves_fp32_reference(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda: "fast")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: True)
    assert bench_module.default_modes() == ["eager-fp32", "fast"]


def test_benchmark_sync_and_memory_follow_backend_device(monkeypatch):
    from types import SimpleNamespace
    import math

    calls = []
    monkeypatch.setattr(bench_module.torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(bench_module.torch.mps, "driver_allocated_memory", lambda: 2**30)
    cpu = SimpleNamespace(dev="cpu")
    mps = SimpleNamespace(dev=bench_module.torch.device("mps"))
    bench_module.timed(lambda: calls.append("work"), 1, cpu)
    assert calls == ["work"]
    assert math.isnan(bench_module.memory_gb(cpu))

    calls.clear()
    bench_module.timed(lambda: calls.append("work"), 1, mps)
    assert calls == ["mps", "work", "mps"]
    assert bench_module.memory_gb(mps) == 1.0


def test_named_options_show_keys_by_default():
    assert named_options({"returns": "Returns", "it": "IT"}) == (["returns", "it"], ["returns: Returns", "it: IT"])
    assert named_options({"x": None, "y": "why"})[1] == ["x", "y: why"]
    keys, opts = named_options({"approve": {"requires_manager": False}, "reject": {"requires_manager": False}})
    assert opts == ['approve: {"requires_manager": false}', 'reject: {"requires_manager": false}']


def test_named_options_hide_only_on_request():
    assert named_options({"option_1": "Yes", "opcja_2": "Nie"}, "hide")[1] == ["Yes", "Nie"]
    assert named_options({"a": "same", "b": "same"}, "hide")[1] == ["a: same", "b: same"]   # still distinguishable
    with pytest.raises(ValueError):
        named_options({"a": "x", "b": "y"}, "maybe")


def test_structured_choices_give_distinct_prompts():
    q = to_items("Action requested: reject.", {"action": {"type": "choice", "instructions": "Return the action.",
                 "criteria": {"approve": {"requires_manager": False}, "reject": {"requires_manager": False}}}})[0]
    assert len(set(q["options"])) == 2 and q["options"][1].startswith("reject")


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "Dept?", "criteria": {"returns": "Returns", "it": "IT"}},
    {"type": "noul", "instructions": "Damaged?"},
    {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
])
def test_response_conforms_to_official_schema(q):
    srv = FakeServer([0.7, 0.2, 0.1])
    r = asyncio.run(srv._run({"model": "basal", "state": "Parcel arrived damaged.", "questions": {"q": q}}))
    jsonschema.validate(r, schema("SystemOneResponse"))
    assert isinstance(r["usage"]["input_tokens"], int)


def test_model_list_conforms_to_official_schema():
    jsonschema.validate(models_payload("basal-1.0-4.5B", "fast", {"0.99": None}), schema("ModelMetadataList"))

@pytest.mark.parametrize(("mode", "quant"), [
    ("eager", "fp8"), ("fast", "q8"), ("mps", "fp8"), ("mlx", "nvfp4"),
    ("vllm", "fp8"), ("gguf", "q8"), ("mlx", "fp8"), ("mps", "q8"), ("eager", "nvfp4"),
])
def test_unsupported_quant_fails_before_model_loading(monkeypatch, mode, quant):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--quant", quant])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="not supported"):
        server_module.main()


def test_default_mode_requires_both_mlx_packages_on_mps(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    assert default_mode() == "mps"

def swapped_prompts_differ(crit_a, crit_b, t="choice"):
    q = lambda c: to_items("Customer selected shipping service B.", {"q": {"type": t, "instructions": "Which code?",
                                                                            "criteria": c}})[0]["options"]
    return q(crit_a) != q(crit_b)


def test_swapping_keys_changes_what_the_model_sees():
    """Regression: whoever owns which description must reach the prompt (short codes, names mentioned in text)."""
    assert swapped_prompts_differ({"approve": "Handled by Alice", "reject": "Handled by Bob"},
                                  {"reject": "Handled by Alice", "approve": "Handled by Bob"})
    assert swapped_prompts_differ({"A": "Dispatch office: Warsaw", "B": "Dispatch office: Krakow"},
                                  {"B": "Dispatch office: Warsaw", "A": "Dispatch office: Krakow"})
    d1, d2 = "Processes approve/reject requests; handled by Alice", "Processes approve/reject requests; handled by Bob"
    assert swapped_prompts_differ({"approve": d1, "reject": d2}, {"reject": d1, "approve": d2})
    assert swapped_prompts_differ({"low": "minor", "high": "severe"}, {"high": "minor", "low": "severe"}, "score")
