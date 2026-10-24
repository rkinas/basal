"""Custom calibration: basal-calibrate against a temporary uncalibrated child, task questions with the engine in
basal-ui's process, files fitted elsewhere imported, the files and their report, selecting one for the model (and back
to the shipped one), cancelling, and the inputs refused."""
import json
import time

from tests.e2e.support import EXAMPLES

LABELLED = str(EXAMPLES / "questions.jsonl")  # simple basal-run items with gold: basal-calibrate takes them


def task_input(path, n=40):
    """Labelled classify (a task) and choice questions: Polish customer messages, complaint or praise."""
    lines = []
    for i in range(n):
        praise = i % 2 == 0
        state = (["Wszystko dotarło szybko, dziękuję!", "Świetna obsługa, polecam."] if praise else
                 ["Paczka przyszła zniszczona.", "Zamówienie nie dotarło od dwóch tygodni."])[i // 2 % 2]
        lines.append({"id": f"t{i}", "state": f"{state} (zgłoszenie {i})", "questions": {
            "intent": {"type": "classify", "labels": {"complaint": "skarga klienta", "praise": "pochwała"},
                       "gold": "praise" if praise else "complaint"},
            "tone": {"type": "choice", "criteria": {"pos": "opinia pozytywna", "neg": "opinia negatywna"},
                     "gold": "pos" if praise else "neg"}}})
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return str(path)


def test_task_questions_calibrate_with_the_engine_in_process(ui, text_model, tmp_path):
    model_id = text_model["id"]
    committed = ui.status()["memory"]["committed_bytes"]
    job = ui.run_job("/api/calibrate", {"id": model_id, "input": task_input(tmp_path / "tasks.jsonl")}, "calibrate")
    result = job["result"]
    assert result["engine_in_process"] is True and "in process" in job["label"]
    assert "classify" in result["report"] and result["fitted_on"]["model"] == text_model["repo"]
    assert ui.status()["memory"]["committed_bytes"] == committed  # the reservation is released


def test_a_file_fitted_elsewhere_is_imported_with_basal_serves_warning(ui, text_model, second_model, tmp_path):
    model_id, other = text_model["id"], second_model["id"]
    fitted = ui.run_job("/api/calibrate", {"id": other, "input": LABELLED, "targets": ["0.05"]}, "calibrate")["result"]
    done = ui.post("/api/calibrations/import", {"id": model_id, "path": fitted["file"]})
    entry = next(f for f in done["files"] if f["file"] == done["file"])
    assert entry["imported"] and entry["has_report"] and f"fitted on {second_model['repo']}," in entry["warning"]
    assert ui.get("/api/calibrations/report", params={"id": model_id, "file": done["file"]})["report"].strip()
    junk = tmp_path / "junk.json"
    junk.write_text('{"hello": 1}')
    assert "not a basal calibration file" in ui.detail("POST", "/api/calibrations/import",
                                                       {"id": model_id, "path": str(junk)}, 400)


def test_fit_inspect_select_and_apply_a_calibration(ui, text_model):
    model_id = text_model["id"]
    ui.load(model_id)
    before = ui.get("/api/calibrations", params={"id": model_id})
    assert before["shipped"] and before["shipped"]["file"].endswith(".json")
    committed = ui.status()["memory"]["committed_bytes"]
    job = ui.run_job("/api/calibrate", {"id": model_id, "input": LABELLED, "targets": ["0.05"]}, "calibrate")
    result = job["result"]
    assert result["fitted_on"]["decisions"] == 20 and result["fitted_on"]["model"] == text_model["repo"]
    assert "temperature" in result["report"] and "max_error" in result["report"]
    assert ui.status()["memory"]["committed_bytes"] == committed  # the temporary child is gone
    sources = {s["id"]: s["state"] for s in ui.get("/api/logs")["sources"]}
    assert sources["job:calibrate"] == "done"

    listing = ui.get("/api/calibrations", params={"id": model_id})
    assert listing["files"][0]["file"] == result["file"] and len(listing["files"]) == len(before["files"]) + 1
    report = ui.get("/api/calibrations/report", params={"id": model_id, "file": result["file"]})["report"]
    assert report.strip() == result["report"].strip()

    try:
        assert ui.post("/api/calibrations/select", {"id": model_id, "file": result["file"]})["selected"] == result["file"]
        assert ui.model(model_id)["restart_required"] is True
        ui.post("/api/restart", {"id": model_id})
        m = ui.wait_model(model_id)
        assert m["state"] == "ready" and m["restart_required"] is False
        assert m["command"][m["command"].index("--calibration") + 1] == result["file"]
        assert not m["entry"].get("calibration_warning"), m["entry"].get("calibration_warning")
        assert m["entry"]["temperature_per_prim"] != {}
    finally:
        ui.post("/api/calibrations/select", {"id": model_id, "file": None})
        ui.post("/api/restart", {"id": model_id})
        assert "--calibration" not in ui.wait_model(model_id)["command"]


def test_a_calibration_can_be_cancelled_and_bad_inputs_are_refused(ui, text_model, tmp_path):
    model_id = text_model["id"]
    ui.post("/api/calibrate", {"id": model_id, "input": LABELLED})
    deadline = time.monotonic() + 30
    while True:
        phase = ui.status()["jobs"]["calibrate"]["phase"]
        if phase in ("starting model", "calibrating"):
            break
        assert time.monotonic() < deadline, f"calibration never reached a cancellable phase: {phase}"
        time.sleep(0.1)
    ui.delete("/api/calibrate")
    assert ui.wait_job("calibrate")["phase"] == "cancelled"
    assert "cannot read" in ui.detail("POST", "/api/calibrate", {"id": model_id, "input": str(tmp_path / "no.jsonl")},
                                      400)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\nnot json\n')
    assert "line 2" in ui.detail("POST", "/api/calibrate", {"id": model_id, "input": str(bad)}, 400)
    assert "targets" in ui.detail("POST", "/api/calibrate", {"id": model_id, "input": LABELLED, "targets": [3]}, 400)
    assert ui.detail("POST", "/api/calibrations/select", {"id": model_id, "file": str(tmp_path / "x.json")}, 400)
