"""Start-up, auto-load and shutdown of a basal-ui process of its own (not the shared session one)."""
import json

from tests.e2e.support import UI


def test_auto_load_restores_the_models_with_their_options_and_ctrl_c_leaves_no_child(tmp_path, text_model):
    model_id, options = text_model["id"], {"orders": "2"}
    app = UI(tmp_path / "life").start()
    try:
        assert app.put("/api/serve", {"auto_load": True})["auto_load"] is True
        assert app.load(model_id, options)["command"][-2:] == ["--orders", "2"]
    finally:
        assert app.stop() == []  # every basal-serve child is gone after Ctrl+C
    remembered = json.loads(app.settings.read_text())["last_models"]
    assert remembered == [{"id": model_id, "media": False, "options": options}]

    app.start()  # the same settings: the model comes back by itself, with its load options
    try:
        m = app.wait_model(model_id, appear=True)
        assert m is not None and m["state"] == "ready", m
        assert m["options"] == options and m["command"][-2:] == ["--orders", "2"]
        events = [line["text"] for line in app.get("/api/logs", params={"source": "basal-ui"})["lines"]]
        assert f"info auto-load: {model_id}" in events
    finally:
        assert app.stop() == []
