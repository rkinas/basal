"""Thinking and early exit: refused where the model or machine cannot run them; exercised where they can (basal-2.0 4.5B
with vLLM for thinking, basal-1.0 bf16 on CUDA for --mode fast-exit)."""
import pytest

from tests.e2e.support import CHOICE, catalog

THINK_REASON = "thinking is available for basal-2.0 4.5B"


def options_of(ui, model_id):
    return ui.get("/api/models/options", params={"id": model_id})


def test_thinking_and_early_exit_are_refused_where_they_cannot_run(ui, text_model):
    model_id = text_model["id"]
    think = options_of(ui, model_id)["think"]
    if not think["model"]:
        assert think["reason"].startswith(THINK_REASON)
        assert ui.detail("POST", "/api/load", {"id": model_id, "options": {"think": True}}, 400).startswith(THINK_REASON)
    if text_model["mode"] != "fast-exit":
        assert "needs --mode fast-exit" in ui.detail("POST", "/api/load", {"id": model_id,
                                                                           "options": {"early_exit": "0.99"}}, 400)
        assert ui.load(model_id)["entry"]["early_exit"] == []  # the Playground hides its Early exit control then


def test_a_thinking_model_reasons_before_hard_decisions(ui):
    model = next((m for m in catalog(ui) if m["downloaded"] and m["family"] == "basal-2.0" and m["tier"] == "4.5B"
                  and m["runnable"] and options_of(ui, m["id"])["think"]["vllm"]), None)
    if model is None:
        pytest.skip("thinking needs a downloaded basal-2.0 4.5B and vLLM in basal-ui's environment")
    ui.load(model["id"], {"think": True})
    r = ui.decide(model["id"], {**CHOICE, "think": "force", "think_text": True})
    answer = r["answers"]["dept"]
    assert answer["meta"]["think"]["used"] is True and r["usage"]["think_questions"] == 1
    assert answer["think"]["tokens"] > 0 and answer["think"]["text"] and "confidence" in answer["think"]["no_think"]
    off = ui.decide(model["id"], {**CHOICE, "think": "off"})["answers"]["dept"]
    assert off["meta"]["think"]["used"] is False
    ui.unload(model["id"])


def test_early_exit_levels_on_a_fast_exit_model(ui):
    model = next((m for m in catalog(ui) if m["downloaded"] and m["family"] == "basal-1.0" and m["port"] == "bf16"
                  and any(x["mode"] == "fast-exit" and x["available"] for x in options_of(ui, m["id"])["modes"])),
                 None)
    if model is None:
        pytest.skip("early exit needs a downloaded basal-1.0 bf16 model on a CUDA GPU (--mode fast-exit)")
    m = ui.load(model["id"], {"mode": "fast-exit", "early_exit": "0.99"})
    assert m["mode"] == "fast-exit" and "0.99" in m["entry"]["early_exit"]
    for level in ["off", *m["entry"]["early_exit"]]:
        assert ui.decide(model["id"], {**CHOICE, "early_exit": level})["answers"]["dept"]["choice"]
    ui.unload(model["id"])
