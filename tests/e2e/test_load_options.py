"""Load options: the engine modes this machine offers, options reaching basal-serve, the context policy, basal-serve's
own refusals, and options remembered for a model."""
from tests.e2e.support import CHOICE, long_state


def options_of(ui, model_id):
    return ui.get("/api/models/options", params={"id": model_id})


def test_the_mode_list_covers_every_mode_and_offers_the_default(ui, text_model, hardware):
    d = options_of(ui, text_model["id"])
    modes = {m["mode"]: m for m in d["modes"]}
    assert len(modes) == 16 and {"ollama", "llamacpp"} <= {m for m, v in modes.items() if v["available"]}
    default = [m for m, v in modes.items() if v["default"]]
    assert default == [text_model["mode"]] and modes[text_model["mode"]]["available"]
    assert all(v["reason"] for v in modes.values() if not v["available"])
    if hardware["platform"] != "cuda":
        assert not any(modes[m]["available"] for m in ("fast", "fast-nocompile", "fast-exit", "fp8", "nvfp4"))
    if "tpu" not in hardware["engines"]:
        assert not modes["tpu"]["available"]


def test_options_reach_basal_serve_and_the_context_policy_applies(ui, text_model):
    model_id = text_model["id"]
    chosen = {"deterministic": True, "orders": "2", "context_overflow": "refuse"}
    m = ui.load(model_id, chosen)
    assert m["command"][-5:] == ["--deterministic", "--orders", "2", "--context-overflow", "refuse"]
    assert m["entry"]["context"]["overflow"] == "refuse" and m["entry"].get("orders") in (2, "2")
    limit = m["entry"]["context"]["limit"]
    body = {"state": long_state(limit), "questions": {"q": {"type": "noul", "instructions": "Czy to umowa najmu?"}}}
    refused = ui.decide(model_id, body, expect=(422,))
    assert "the prompt is" in refused["error"] and refused["code"] == "INVALID_REQUEST"
    assert ui.decide(model_id, CHOICE)["answers"]["dept"]["choice"]
    ui.load(model_id)  # back to the defaults
    assert "--deterministic" not in ui.model(model_id)["command"]


def test_invalid_unavailable_and_refused_combinations(ui, text_model):
    model_id = text_model["id"]
    ui.load(model_id)
    assert "orders" in ui.detail("POST", "/api/restart", {"id": model_id, "options": {"orders": "7"}}, 400)
    assert "unknown load option" in ui.detail("POST", "/api/restart", {"id": model_id, "options": {"x": 1}}, 400)
    unavailable = next(m["mode"] for m in options_of(ui, model_id)["modes"] if not m["available"])
    assert ui.detail("POST", "/api/restart", {"id": model_id, "options": {"mode": unavailable}}, 400).startswith(
        f"--mode {unavailable}:")
    # basal-serve's own refusal reaches the model's row (on every engine: prefix cache with --deterministic)
    ui.post("/api/restart", {"id": model_id, "options": {"deterministic": True, "prefix_cache": True}})
    m = ui.wait_model(model_id)
    assert m["state"] == "error" and "--prefix-cache" in m["error"]
    assert ui.load(model_id)["state"] == "ready"


def test_remembered_options_are_used_by_a_plain_load(ui, text_model):
    model_id = text_model["id"]
    try:
        saved = ui.put("/api/models/options", {"id": model_id, "options": {"max_batch": 16}})
        assert saved["saved"] is True and saved["options"]["max_batch"] == 16
        ui.unload(model_id)
        ui.post("/api/load", {"id": model_id})  # no options: the remembered ones
        m = ui.wait_model(model_id)
        assert m["state"] == "ready" and m["options"] == {"max_batch": 16} and m["command"][-2:] == ["--max-batch", "16"]
    finally:
        ui.put("/api/models/options", {"id": model_id, "options": {}})
    assert options_of(ui, model_id)["saved"] is False
    ui.load(model_id)
