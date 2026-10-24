import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from basal import calibrate
from basal.ui import calibration, catalog, settings
from basal.ui.app import Fleet, Runner, create_app
from basal.ui.catalog import Variant
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

APPLE = Hardware("apple", "mps", 1024**4, None, None, frozenset({"mps"}))
HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())
MINI = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
FITTED = {"prompt": 2, "temperature_per_prim": {"choice": 1.2}, "engine": "eager",
          "fitted_on": {"decisions": 20, "per_type": {"choice": 20}},
          "gates": {"automated_at": {"0.05": 0.4}},
          "thresholds": {"0.05": {"per_type": {"choice": {"confidence": 0.93}}}}}


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "CALIBRATION.json").write_text(json.dumps({"temperature_per_prim": {"choice": 1.1}}))
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: model)

    def start(self, assessment, gen, media=False, cache_dir=None, serve=None):
        with self.lock:
            if gen == self.gen:
                self.state, self.base_url, self.ready_at = "ready", "http://127.0.0.1:1", time.time()
                self.serve_snapshot = {"log_decisions": "", "max_len": 4096, "calibration": self._calibration(MINI)}
                self.command = self._build_command(MINI, model, assessment, False, 1, settings.serve_settings())
    monkeypatch.setattr(Runner, "_start", start)
    return model


def wait_job(client, kind="calibrate"):
    deadline = time.monotonic() + 5
    while client.get("/api/status").json()["jobs"][kind]["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    return client.get("/api/status").json()["jobs"][kind]


def test_calibrate_fits_against_a_temporary_child_lists_and_selects_the_file(tmp_path, monkeypatch, config):
    release, calls = threading.Event(), []

    def fake_main(argv):
        calls.append(argv)
        release.wait(5)
        print("report line")
        out = argv[argv.index("--out") + 1]
        with open(out, "w") as f:
            json.dump(FITTED, f)

    monkeypatch.setattr(calibrate, "main", fake_main)
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                  hardware_fn=lambda: HW, stats=Stats(tmp_path / "s.json"), history=History(tmp_path / "h.jsonl"))
    app = create_app(catalog_fn=lambda **_: ([MINI], "test"), hardware_fn=lambda: HW, fleet=fleet, stats=fleet.stats,
                     history=fleet.history)
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\n')
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/load", json={"id": "mini"}).status_code == 202
        missing = client.post("/api/calibrate", json={"id": "mini", "input": str(tmp_path / "none.jsonl")})
        assert missing.status_code == 400 and "cannot read" in missing.json()["detail"]
        assert client.post("/api/calibrate", json={"id": "mini", "input": str(labelled)}).status_code == 202
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "calibrate" in fleet.temporary  # the uncalibrated child, without --api-keys, counted in the budget
        child = fleet.temporary["calibrate"].command
        assert "--no-calibration" in child and "--api-keys" not in child
        busy = client.post("/api/calibrate", json={"id": "mini", "input": str(labelled)})
        assert busy.status_code == 409 and busy.json()["detail"] == "a calibrate job is already running"
        release.set()
        job = wait_job(client)
        assert job["phase"] == "done", job["error"]
        assert calls[0][calls[0].index("--url") + 1] == "http://127.0.0.1:1"
        assert calls[0][calls[0].index("--model") + 1] == str(config)
        result = job["result"]
        assert result["report"] == "report line" and result["thresholds"] == {"0.05": {"choice": 0.93}}
        assert fleet.temporary == {}

        listing = client.get("/api/calibrations", params={"id": "mini"}).json()
        assert listing["selected"] is None and [f["file"] for f in listing["files"]] == [result["file"]]
        assert listing["shipped"]["file"] == str(config / "CALIBRATION.json")
        report = client.get("/api/calibrations/report", params={"id": "mini", "file": result["file"]})
        assert report.json() == {"report": "report line\n"}
        stranger = client.post("/api/calibrations/select", json={"id": "mini", "file": str(tmp_path / "x.json")})
        assert stranger.status_code == 400

        runner = fleet.runners["mini"]
        assert runner.status()["restart_required"] is False
        assert client.post("/api/calibrations/select", json={"id": "mini", "file": result["file"]}).json()[
            "selected"] == result["file"]
        assert runner.status()["restart_required"] is True  # the running child has the shipped calibration
        command = runner._build_command(MINI, config, catalog.assess(MINI, HW), False, 1, settings.serve_settings())
        assert command[command.index("--calibration") + 1] == result["file"]
        client.post("/api/calibrations/select", json={"id": "mini", "file": None})
        assert runner.status()["restart_required"] is False


def test_calibrate_refuses_an_unknown_model_and_a_model_over_the_budget(tmp_path):
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                  hardware_fn=lambda: HW, stats=Stats(tmp_path / "s.json"), history=History(tmp_path / "h.jsonl"))
    app = create_app(catalog_fn=lambda **_: ([MINI], "test"), hardware_fn=lambda: HW, fleet=fleet,
                     stats=fleet.stats, history=fleet.history)
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\n')
    settings.update_memory({"reserve_bytes": 1024**4})  # nothing fits
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/calibrate", json={"id": "nope", "input": str(labelled)}).status_code == 400
        over = client.post("/api/calibrate", json={"id": "mini", "input": str(labelled)})
        assert over.status_code == 409 and over.json()["detail"].startswith("over memory budget")
        assert client.post("/api/calibrate", json={"id": "mini", "input": str(labelled), "targets": [2]}).status_code == 400


def make_app(tmp_path, hw=HW):
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: hw.platform, settings_mod=settings,
                  hardware_fn=lambda: hw, stats=Stats(tmp_path / "s.json"), history=History(tmp_path / "h.jsonl"))
    return fleet, create_app(catalog_fn=lambda **_: ([MINI], "test"), hardware_fn=lambda: hw, fleet=fleet,
                             stats=fleet.stats, history=fleet.history)


TASK = {"state": "The parcel arrived broken.", "questions": {"intent": {
    "type": "classify", "labels": {"complaint": "a complaint", "praise": "praise"}, "gold": "complaint"}}}


def test_a_bad_input_is_refused_with_its_line_before_any_model_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "main", lambda argv: pytest.fail("basal-calibrate ran on a bad input"))
    fleet, app = make_app(tmp_path)
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\n{"state": "s"}\n')
    with TestClient(app, base_url="http://127.0.0.1") as client:
        bad = client.post("/api/calibrate", json={"id": "mini", "input": str(labelled)})
        assert bad.status_code == 400 and "line 2" in bad.json()["detail"]
        assert "calibrate" not in fleet.temporary


def test_task_questions_calibrate_with_the_engine_in_process_and_reserve_its_memory(tmp_path, monkeypatch, config):
    calls, reserved = [], []

    def fake_main(argv):
        calls.append(argv)
        reserved.append(dict(fleet.reserved))
        out = argv[argv.index("--out") + 1]
        with open(out, "w") as f:
            json.dump({**FITTED, "task_thresholds": {"classify": {"0.05": 0.81}}}, f)
        if "--save-rows" in argv:
            open(argv[argv.index("--save-rows") + 1], "w").write("{}\n")
    monkeypatch.setattr(calibrate, "main", fake_main)
    fleet, app = make_app(tmp_path, APPLE)
    settings.set_load_options("mini", {"mode": "eager", "max_batch": 8, "deterministic": True,
                                       "strict_calibration": True})
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text(json.dumps(TASK) + "\n")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/calibrate", json={"id": "mini", "input": str(labelled), "advanced": {
            "strict": True, "reference": "basal-1.5", "min_per_type": 12, "min_blend": 3, "min_family": 4,
            "min_per_set": 5, "concurrency": 7, "compare": "none", "save_rows": True}}).status_code == 202
        job = wait_job(client)
        assert job["phase"] == "done", job["error"]
        argv = calls[0]
        for flag, value in (("--reference", "basal-1.5"), ("--min-per-type", "12"), ("--min-blend", "3"),
                            ("--min-family", "4"), ("--min-per-set", "5"), ("--concurrency", "7")):
            assert argv[argv.index(flag) + 1] == value
        assert "--strict" in argv and "--no-calibration" in argv
        assert "--url" not in argv and argv[argv.index("--mode") + 1] == "eager"
        assert argv[argv.index("--max-batch") + 1] == "8" and "--deterministic" in argv
        assert "--strict-calibration" not in argv  # basal-serve's own, not the engine's
        assert "calibrate" in reserved[0] and "calibrate" not in fleet.reserved  # released after the run
        assert "calibrate" not in fleet.temporary  # no basal-serve child
        assert job["result"]["engine_in_process"] is True
        listed = client.get("/api/calibrations", params={"id": "mini"}).json()["files"][0]
        assert listed["task_thresholds"] == {"classify": {"0.05": 0.81}}
        assert listed["rows"] == str(Path(listed["file"]).with_suffix(".rows.jsonl"))


def test_a_calibration_fitted_elsewhere_is_imported_selectable_and_warned_about(tmp_path, config):
    _, app = make_app(tmp_path)
    elsewhere = tmp_path / "other" / "CALIBRATION.custom.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text(json.dumps({**FITTED, "fitted_on": {**FITTED["fitted_on"], "model": "Remek/basal-2.0-max"}}))
    elsewhere.with_suffix(".report.txt").write_text("their report\n")
    junk = tmp_path / "junk.json"
    junk.write_text('{"hello": 1}')
    with TestClient(app, base_url="http://127.0.0.1") as client:
        refused = client.post("/api/calibrations/import", json={"id": "mini", "path": str(junk)})
        assert refused.status_code == 400 and "not a basal calibration file" in refused.json()["detail"]
        missing = client.post("/api/calibrations/import", json={"id": "mini", "path": str(tmp_path / "none.json")})
        assert missing.status_code == 400
        done = client.post("/api/calibrations/import", json={"id": "mini", "path": str(elsewhere)}).json()
        entry = next(f for f in done["files"] if f["file"] == done["file"])
        assert entry["imported"] and "fitted on Remek/basal-2.0-max" in entry["warning"]
        assert client.get("/api/calibrations/report", params={"id": "mini", "file": done["file"]}).json()[
            "report"] == "their report\n"
        selected = client.post("/api/calibrations/select", json={"id": "mini", "file": done["file"]}).json()
        assert selected["selected"] == done["file"]


def test_importing_same_calibration_twice_never_overwrites_file_or_report(tmp_path):
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(FITTED))
    source.with_suffix(".report.txt").write_text("original report")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    class Settings:
        @staticmethod
        def calibrations_dir(_model):
            return model_dir

    first = calibration.import_file("mini", Settings, source)
    first.write_text(json.dumps({**FITTED, "engine": "first"}))
    first.with_suffix(".report.txt").write_text("first report")
    second = calibration.import_file("mini", Settings, source)
    assert second != first
    assert json.loads(first.read_text())["engine"] == "first"
    assert first.with_suffix(".report.txt").read_text() == "first report"
    assert json.loads(second.read_text()) == FITTED
    assert second.with_suffix(".report.txt").read_text() == "original report"


def test_from_rows_report_only_skips_child_and_memory_reservation(tmp_path, monkeypatch, config):
    calls = []
    monkeypatch.setattr(calibrate, "main", lambda argv: (calls.append(argv), print("rows report")))
    fleet, app = make_app(tmp_path)
    rows = tmp_path / "fit.rows.jsonl"
    rows.write_text('{"basal_calibrate_rows":1}\n')
    settings.update_memory({"reserve_bytes": 1024**4})  # nothing fits: a refit runs no engine, so no check
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/calibrate", json={"id": "mini", "input": str(rows), "from_rows": True,
                                                       "advanced": {"report_only": True}})
        assert response.status_code == 202
        job = wait_job(client)
    assert job["phase"] == "done", job["error"]
    assert job["result"] == {"file": None, "report": "rows report", "report_only": True}
    assert "--from-rows" in calls[0] and "--report-only" in calls[0]
    assert not fleet.temporary and not fleet.reserved


def test_report_only_on_decision_input_uses_the_temporary_child_and_writes_no_file(tmp_path, monkeypatch, config):
    calls, children = [], []

    def fake_main(argv):
        calls.append(argv)
        children.append(set(fleet.temporary))
        print("report only")
    monkeypatch.setattr(calibrate, "main", fake_main)
    fleet, app = make_app(tmp_path)
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\n')
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/calibrate", json={"id": "mini", "input": str(labelled),
                                                       "advanced": {"report_only": True}})
        assert response.status_code == 202
        job = wait_job(client)
    assert job["phase"] == "done", job["error"]
    assert job["result"] == {"file": None, "report": "report only", "report_only": True}
    assert "--url" in calls[0] and "--report-only" in calls[0]
    assert children == [{"calibrate"}] and not fleet.temporary and fleet.reserved == {}
    assert not list(settings.calibrations_dir("mini").glob("*.report.txt"))


def test_a_missing_compare_or_reference_file_is_refused_before_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "main", lambda argv: pytest.fail("basal-calibrate ran"))
    fleet, app = make_app(tmp_path)
    labelled = tmp_path / "labelled.jsonl"
    labelled.write_text('{"state": "s", "question": "q?", "options": ["a", "b"], "gold": 0}\n')
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for key in ("reference", "compare"):
            missing = str(tmp_path / f"{key}.json")
            response = client.post("/api/calibrate", json={"id": "mini", "input": str(labelled),
                                                           "advanced": {key: missing}})
            assert response.status_code == 400
            assert response.json()["detail"] == f"{key}: {missing} is not a readable file"
        assert not fleet.temporary
