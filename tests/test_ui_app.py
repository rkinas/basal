import os
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from basal.ui import calibration, catalog, settings
from basal.ui.app import Fleet, Runner, create_app, rss_by_pgid
from basal.ui.catalog import Variant
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())
MINI = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
OMNI = Variant("omni", "Remek/basal-omni-1.0-mini", "basal-omni-1.0", "mini", "bf16", "bf16", None, 100, True,
               ("image", "audio"))
SMALL = Variant("small", "Remek/basal-2.0-small", "basal-2.0", "small", "bf16", "bf16", None, 100, False)


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def instant_start(monkeypatch):
    """Runner._start reaches ready at once, without a download or a child process."""
    def start(self, assessment, gen, media=False, cache_dir=None, serve=None):
        with self.lock:
            if gen == self.gen:
                self.state, self.media, self.ready_at = "ready", (["image", "audio"] if media else []), time.time()
                self.serve_snapshot = {"log_decisions": "", "max_len": 4096, "calibration": None}
        if not self.temporary:
            settings.remember_loaded(assessment.variant.id, media)
    monkeypatch.setattr(Runner, "_start", start)


def wait_for(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def make_fleet(tmp_path, hw=HW):
    return Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                 hardware_fn=lambda: hw, stats=Stats(tmp_path / "stats.json"),
                 history=History(tmp_path / "history.jsonl"))


def app_for(tmp_path, items, hw=HW, fleet=None):
    fleet = fleet or make_fleet(tmp_path, hw)
    app = create_app(catalog_fn=lambda **_: (items, "hub"), hardware_fn=lambda: hw, fleet=fleet,
                     stats=fleet.stats, history=fleet.history)
    return app, fleet


def states(client):
    return {m["id"]: m["state"] for m in client.get("/api/status").json()["models"]}


def test_examples_include_a_full_request_of_every_advanced_type(tmp_path):
    app, _ = app_for(tmp_path, [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        examples = client.get("/api/examples").json()
    types = {q["type"] for e in examples for q in e["request"]["questions"].values()}
    assert {"multi", "act", "tool", "classify", "verify", "compress", "extract"} <= types


def test_model_api_sorting_load_validation_and_decide_requires_a_loaded_model(tmp_path):
    large = Variant("large", "Remek/basal-2.0-4.5B-FP8", "basal-2.0", "4.5B", "FP8", "FP8", None, 100, False)
    app, _ = app_for(tmp_path, [MINI, large])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        models = client.get("/api/models").json()
        assert models["source"] == "hub"
        assert [m["id"] for m in models["models"]] == ["mini", "large"]
        assert "suggestions" in models and all("fit_now" in model for model in models["models"])
        assert client.get("/api/hardware").json()["available_bytes"] is None
        assert client.post("/api/load", json={"id": "large"}).status_code == 400
        missing = client.post("/api/decide", json={"state": "x", "questions": {}})
        assert missing.status_code == 400 and missing.json()["detail"] == "model is required; loaded: []"
        assert client.post("/api/decide", json={"model": "mini", "state": "x", "questions": {}}).status_code == 404


def test_load_with_media_requires_engine_and_echoes_status(tmp_path, monkeypatch, instant_start):
    app, _ = app_for(tmp_path, [OMNI])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/load", json={"id": "omni", "media": True})
        assert response.status_code == 400
        assert "install basal[media]" in response.json()["detail"]

    media_hw = Hardware("cpu", "cpu", 1024**4, None, None, frozenset({"media"}))
    monkeypatch.setattr(catalog, "downloaded_bytes", lambda variant, cache_dir: None)
    app, _ = app_for(tmp_path, [OMNI], hw=media_hw)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        model = client.get("/api/models").json()["models"][0]
        assert model["media_mode"] == "eager" and model["media_reason"] is None
        response = client.post("/api/load", json={"id": "omni", "media": True})
        assert response.status_code == 202
        assert response.json()["models"][0]["media_requested"] is True
        wait_for(lambda: states(client) == {"omni": "ready"})
        assert client.get("/api/status").json()["models"][0]["media"] == ["image", "audio"]


def test_several_models_load_side_by_side_and_unload_one_or_all(tmp_path, instant_start):
    app, _ = app_for(tmp_path, [MINI, SMALL])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/load", json={"id": "mini"}).status_code == 202
        assert client.post("/api/load", json={"id": "small"}).status_code == 202
        wait_for(lambda: states(client) == {"mini": "ready", "small": "ready"})
        assert [m["id"] for m in client.get("/api/status").json()["models"]] == ["mini", "small"]
        again = client.post("/api/load", json={"id": "mini"})
        assert again.status_code == 409 and again.json()["detail"] == "model mini is already loaded"
        assert [e["id"] for e in settings.last_models()] == ["mini", "small"]

        assert client.post("/api/unload", json={"id": "nope"}).status_code == 404
        assert [m["id"] for m in client.post("/api/unload", json={"id": "mini"}).json()["models"]] == ["small"]
        assert [e["id"] for e in settings.last_models()] == ["small"]
        assert client.post("/api/unload", json={}).json()["models"] == []
        assert settings.last_models() == []


def test_directory_change_rejected_while_a_model_is_loaded(tmp_path, instant_start):
    app, _ = app_for(tmp_path, [MINI])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post("/api/load", json={"id": "mini"})
        response = client.put("/api/settings", json={"models_dir": str(tmp_path / "models")})
    assert response.status_code == 409
    assert response.json()["detail"] == "unload every model before changing the models directory"


def test_delete_rejects_loaded_and_unknown_models(tmp_path, instant_start):
    app, _ = app_for(tmp_path, [MINI])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post("/api/load", json={"id": "mini"})
        loaded = client.post("/api/models/delete", json={"id": MINI.id})
        unknown = client.post("/api/models/delete", json={"id": "missing"})
    assert loaded.status_code == 409
    assert loaded.json()["detail"] == "unload the model first"
    assert unknown.status_code == 400


def test_hardware_endpoint_reads_available_memory_each_request(tmp_path):
    item1 = Hardware("cpu", "cpu", 1000, 2000, None, frozenset(), 100)
    item2 = Hardware("cpu", "cpu", 1000, 2000, None, frozenset(), 200)
    values = iter((item1, item2))
    app = create_app(catalog_fn=lambda **_: ([], "hub"), hardware_fn=lambda: next(values),
                     fleet=make_fleet(tmp_path))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/hardware").json()["available_bytes"] == 100
        assert client.get("/api/hardware").json()["available_bytes"] == 200


def test_security_and_invalid_models_directory_type(tmp_path):
    app, _ = app_for(tmp_path, [MINI])
    with TestClient(app, base_url="http://127.0.0.1:7860") as client:
        assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 400
        rejected = client.post("/api/unload", headers={"origin": "http://evil.example"})
        assert rejected.status_code == 403
        assert rejected.json() == {"detail": "cross-origin request refused"}
        assert client.post("/api/unload", headers={"origin": "http://127.0.0.1:7860"}).status_code == 200
        invalid = client.put("/api/settings", json={"models_dir": 123})
        assert invalid.status_code == 400
        assert invalid.json()["detail"] == "models_dir must be a string or null"


def test_delete_is_refused_while_repository_download_is_alive(tmp_path):
    app, fleet = app_for(tmp_path, [MINI])
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    fleet.downloads.records[MINI.repo] = {"thread": worker, "bar": SimpleNamespace(done=0), "total": 0, "error": None}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/models/delete", json={"id": MINI.id})
    release.set()
    worker.join()
    assert response.status_code == 409
    assert response.json()["detail"] == "unload the model first"


def test_download_runs_without_a_model_and_reports_progress_then_errors(tmp_path, monkeypatch):
    on_disk = {}
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: on_disk.get(variant.repo))
    release, outcome = threading.Event(), {}

    def run(self, record, variant, cache_dir):
        record["bar"].done = 40
        release.wait()
        if outcome.get("error"):
            record["error"] = RuntimeError(outcome["error"])
        else:
            record["path"] = str(tmp_path)
        record["ended"] = time.monotonic()

    monkeypatch.setattr("basal.ui.app.Downloads._run", run)
    app, _ = app_for(tmp_path, [MINI])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/download", json={"id": "missing"}).status_code == 400
        status = client.post("/api/download", json={"id": "mini"})
        assert status.status_code == 202
        assert status.json()["models"] == []
        wait_for(lambda: client.get("/api/status").json()["downloads"][MINI.repo]["done_bytes"] == 40)
        assert client.get("/api/status").json()["downloads"][MINI.repo] == {"done_bytes": 40, "total_bytes": 100, "error": None}
        release.set()
        wait_for(lambda: client.get("/api/status").json()["downloads"] == {})
        on_disk[MINI.repo] = tmp_path
        already = client.post("/api/download", json={"id": "mini"})
        assert already.status_code == 409 and already.json()["detail"] == "already on disk"

        on_disk.clear()
        outcome["error"] = "disk full"
        client.post("/api/download", json={"id": "mini"})
        wait_for(lambda: (client.get("/api/status").json()["downloads"].get(MINI.repo) or {}).get("error"))
        assert client.get("/api/status").json()["downloads"][MINI.repo]["error"] == "disk full"


def test_startup_timeout_kills_child_process_group(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("process-group lifecycle is POSIX-specific")
    assessment = catalog.assess(MINI, Hardware("cpu", "cpu", 1000, 2000, None, frozenset()))
    pid_file = tmp_path / "grandchild.pid"
    code = ("import subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"open({str(pid_file)!r},'w').write(str(p.pid)); time.sleep(60)")
    runner = Runner(models_dir_fn=lambda: tmp_path)
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: tmp_path)
    monkeypatch.setattr(runner, "_build_command", lambda *args: [sys.executable, "-c", code])
    monkeypatch.setattr("basal.ui.app.START_TIMEOUT_S", 0.5)
    runner.load(assessment)
    wait_for(lambda: runner.status()["state"] == "error", timeout=5)
    assert pid_file.exists()
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild process survived startup timeout")


def test_api_serve_settings_and_key_routes(tmp_path):
    app, _ = app_for(tmp_path, [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/serve").json()["settings"]["enabled"] is False
        assert client.put("/api/serve", json={"enabled": True, "port": 8123}).json()["settings"]["port"] == 8123
        key = client.post("/api/serve/keys", json={"name": "laptop"})
        assert key.status_code == 201 and key.json()["key"].startswith("bsl_")
        assert client.delete("/api/serve/keys/laptop").json() == {"deleted": "laptop"}
        assert client.put("/api/serve", json={"port": 80}).status_code == 400


def test_runner_command_is_private_with_internal_key_and_model_name(tmp_path):
    model = Variant("model-id", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
    runner = Runner(settings_mod=settings, platform_fn=lambda: "cpu", internal_secret="internal")
    assessment = SimpleNamespace(variant=model, mode="vllm")
    settings.update_serve({"enabled": True, "host": "0.0.0.0", "port": 8123, "require_key": False,
                           "cors_origins": ["https://example.com"], "max_len": 8192,
                           "log_decisions": str(tmp_path / "decisions.jsonl")})
    command = runner._build_command(model, tmp_path, assessment, False, 43217, settings.serve_settings())
    assert command[command.index("--name") + 1] == model.id
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "43217"
    key_file = command[command.index("--api-keys") + 1]
    records = __import__("json").loads(open(key_file).read())["keys"]
    assert [(r["name"], r["admin"]) for r in records] == [("basal-ui", True)]  # user keys never reach a child
    assert key_file != str(settings.keys_file("a" * 64))  # the API server's file holds the user keys too
    assert command[command.index("--log-decisions") + 1] == str(tmp_path / "decisions.jsonl")
    assert command[command.index("--max-len") + 1] == "8192"
    assert not {"--cors-origins", "--mcp"} & set(command)
    assessment.mode = "eager"
    assert "--max-len" not in runner._build_command(model, tmp_path, assessment, False, 1, settings.serve_settings())


def test_real_subprocess_requires_internal_key_and_records_the_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: tmp_path)
    app, fleet = app_for(tmp_path, [MINI])
    secret = fleet.internal_secret
    code = f"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body=b''):
        self.send_response(status); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path != '/health' and self.headers.get('Authorization') != 'Bearer {secret}':
            return self.reply(401)
        self.reply(200, b'{{"status":"ok"}}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))  # read first: replying early resets
        if self.headers.get('Authorization') != 'Bearer {secret}':
            return self.reply(401)
        answer = {{'type': 'choice', 'choice': 'a', 'confidence': 0.75}}
        self.reply(200, json.dumps({{'received': body, 'answers': {{'q': answer}},
                                     'usage': {{'input_tokens': 12}}}}).encode())
    def log_message(self, *args): pass
print('child up', flush=True)
HTTPServer(('127.0.0.1', int(__import__('sys').argv[1])), Handler).serve_forever()
"""
    monkeypatch.setattr(Runner, "_build_command", lambda self, v, path, assessment, media, port, serve:
                        [sys.executable, "-c", code, str(port)])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/load", json={"id": MINI.id}).status_code == 202
        wait_for(lambda: states(client) == {"mini": "ready"}, timeout=5)
        runner = fleet.runners["mini"]
        assert httpx.get(runner.base_url + "/health").status_code == 200
        assert httpx.get(runner.base_url + "/v1/models").status_code == 401
        body = {"state": "hello", "images": [{"data": "aGVsbG8="}]}
        assert httpx.post(runner.base_url + "/v1/systemone", json=body).status_code == 401
        assert client.post("/api/decide", json={"model": "mini", **body}).json()["received"] == body

        items = client.get("/api/history").json()["items"]
        assert [(i["model"], i["source"], i["state_preview"]) for i in items] == [("mini", "playground", "hello")]
        assert items[0]["answers"] == {"q": {"type": "choice", "best": "a", "confidence": 0.75}}
        record = client.get(f"/api/history/{items[0]['id']}").json()
        assert record["request"] == {"model": "mini", "state": "hello", "images": [{"data": "<8 bytes>"}]}
        assert client.get("/api/history/h_missing").status_code == 404
        stats = client.get("/api/stats").json()["models"]["mini"]
        assert (stats["requests"], stats["questions"], stats["input_tokens"], stats["avg_best_prob"]) == (1, 1, 12, 0.75)

        logs = client.get("/api/logs", params={"source": "mini"}).json()
        assert "child up" in [line["text"] for line in logs["lines"]]
        client.post("/api/unload", json={"id": "mini"})
        assert {"id": "mini", "state": "gone"} in client.get("/api/logs").json()["sources"]
    assert [r["model"] for r in History(tmp_path / "history.jsonl").records] == ["mini"]


def test_runner_restart_required_tracks_only_child_settings():
    runner = Runner(settings_mod=settings)
    runner.state, runner.model = "ready", "Remek/basal-2.0-mini"
    runner.serve_snapshot = {"log_decisions": "", "max_len": 4096, "calibration": None}
    settings.update_serve({"enabled": True, "port": 8123, "require_key": True})  # the API server's business
    assert runner.status()["restart_required"] is False
    settings.update_serve({"max_len": 8192})
    assert runner.status()["restart_required"] is True


def test_api_serve_routes_validate_settings_and_hide_secrets(tmp_path, monkeypatch):
    app, _ = app_for(tmp_path, [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/serve")
        assert response.status_code == 200
        assert response.json()["settings"]["port"] == 8000
        assert {"auto_load", "keys", "mcp_available", "mcp_hint", "lan_ip"} <= response.json().keys()
        for patch in ({"host": "10.0.0.1"}, {"port": 80}, {"port": 70000}, {"max_len": 100},
                      {"cors_origins": ["bad-origin"]}, {"log_decisions": str(tmp_path / "missing" / "x.log")}):
            assert client.put("/api/serve", json=patch).status_code == 400
        monkeypatch.setattr("basal.mcp_server.available", lambda: "install basal[mcp]")
        assert client.put("/api/serve", json={"mcp": True}).status_code == 400
        assert client.put("/api/serve", json={"enabled": True, "port": 8123}).json()["settings"]["port"] == 8123
        invalid_auto_load = client.put("/api/serve", json={"port": 9000, "auto_load": "yes"})
        assert invalid_auto_load.status_code == 400
        assert invalid_auto_load.json()["detail"] == "auto_load must be a boolean"
        assert client.get("/api/serve").json()["settings"]["port"] == 8123
        created = client.post("/api/serve/keys", json={"name": "desk"})
        assert created.status_code == 201
        secret = created.json()["key"]
        assert client.post("/api/serve/keys", json={"name": "desk"}).status_code == 400
        assert client.post("/api/serve/keys", json={"name": "!"}).status_code == 400
        listing = client.get("/api/serve").json()
        assert listing["keys"][0]["name"] == "desk"
        assert "sha256" not in str(listing) and secret not in str(listing)
        assert client.delete("/api/serve/keys/missing").status_code == 404
        key_file = settings.keys_file("a" * 64)
        api_keys = __import__("json").loads(key_file.read_text())["keys"]
        assert {entry["name"] for entry in api_keys} == {"desk", "basal-ui"}
        assert key_file.stat().st_mode & 0o777 == 0o600
        assert client.delete("/api/serve/keys/desk").json() == {"deleted": "desk"}
        revoked = __import__("json").loads(key_file.read_text())["keys"]
        assert [entry["name"] for entry in revoked] == ["basal-ui"]
    assert not key_file.exists()  # basal-ui removes its own key file when it exits


def test_restart_requires_a_loaded_model_and_keeps_media(tmp_path, monkeypatch):
    starts = []
    monkeypatch.setattr(Runner, "_start", lambda self, assessment, gen, media=False, *_: starts.append(media))
    app, fleet = app_for(tmp_path, [OMNI], hw=Hardware("cpu", "cpu", 1024**4, None, None, frozenset({"media"})))
    settings.remember_loaded("omni", True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/restart", json={"id": "omni"}).status_code == 404
        client.post("/api/load", json={"id": "omni", "media": True})
        assert client.post("/api/restart", json={"id": "omni"}).status_code == 409  # still downloading
        fleet.runners["omni"].state = "ready"
        response = client.post("/api/restart", json={"id": "omni"})
        assert response.status_code == 202
        assert response.json()["models"][0]["media_requested"] is True
    wait_for(lambda: starts == [True, True])
    assert settings.last_models() == [{"id": "omni", "media": True}]  # a restart is no unload


def test_auto_load_loads_every_remembered_model_and_skips_unknown_ones(tmp_path, capsys, instant_start):
    settings.set_auto_load(True)
    for model_id in ("mini", "missing", "small"):
        settings.remember_loaded(model_id, False)
    app, fleet = app_for(tmp_path, [MINI, SMALL])
    with TestClient(app, base_url="http://127.0.0.1"):
        wait_for(lambda: set(fleet.loaded_ids()) == {"mini", "small"})
        events = [line["text"] for line in fleet.logs["basal-ui"].since()]
    assert "auto-load of missing skipped: model is unknown" in capsys.readouterr().err
    assert "warn auto-load of missing skipped: model is unknown" in events


def test_auto_load_waiting_on_catalog_is_cancelled_during_shutdown(tmp_path, monkeypatch):
    settings.set_auto_load(True)
    settings.remember_loaded("mini", False)
    fleet = make_fleet(tmp_path)
    loads = []
    monkeypatch.setattr(fleet, "load", lambda assessment, media=False: loads.append(assessment.variant.id))
    real_event = threading.Event
    catalog_entered, release_catalog, shutdown_started = real_event(), real_event(), real_event()
    auto_load_event = real_event()
    set_auto_load_event = auto_load_event.set
    auto_load_event.set = lambda: (shutdown_started.set(), set_auto_load_event())
    event_count = 0

    def event_factory():
        nonlocal event_count
        event_count += 1
        return auto_load_event if event_count == 1 else real_event()

    def blocked_catalog(**_):
        catalog_entered.set()
        release_catalog.wait()
        return [MINI], "test"

    monkeypatch.setattr("basal.ui.app.threading.Event", event_factory)
    app = create_app(catalog_fn=blocked_catalog, hardware_fn=lambda: HW, fleet=fleet, stats=fleet.stats,
                     history=fleet.history)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.__enter__()
    assert catalog_entered.wait(timeout=2)
    shutdown_thread = threading.Thread(target=lambda: client.__exit__(None, None, None))
    shutdown_thread.start()
    assert shutdown_started.wait(timeout=2)
    release_catalog.set()
    shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()
    assert loads == []


def test_create_app_passes_settings_module_to_default_fleet(tmp_path, monkeypatch):
    import basal.ui.app as app_module

    fake_settings = SimpleNamespace(models_dir=lambda: None, stats_path=lambda: tmp_path / "s.json",
                                    history_path=lambda: tmp_path / "h.jsonl")
    captured = {}

    def fleet_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(app_module, "Fleet", fleet_factory)
    create_app(settings_mod=fake_settings)
    assert captured["settings_mod"] is fake_settings


def test_stale_overlapping_start_does_not_replace_last_ready_model(tmp_path, monkeypatch):
    import io

    runner = Runner(settings_mod=settings)
    old = SimpleNamespace(variant=Variant("old", "repo/old", "basal-2.0", "mini", "bf16", "bf16",
                                          None, 100, False), mode="eager")
    newer = SimpleNamespace(variant=Variant("newer", "repo/newer", "basal-2.0", "mini", "bf16", "bf16",
                                            None, 100, False), mode="eager")
    runner.gen = 1
    runner.state, runner.model = "starting", "old"
    monkeypatch.setattr(catalog, "is_downloaded", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(runner, "_build_command", lambda *args: ["fake"])

    class Process:
        stdout = io.StringIO("")
        returncode = 0
        pid = 0

        def poll(self):
            return None

    monkeypatch.setattr("basal.ui.app.subprocess.Popen", lambda *args, **kwargs: Process())
    old_health_entered, release_old_health = threading.Event(), threading.Event()

    def healthy(*args, **kwargs):
        if threading.current_thread().name == "old-start":
            old_health_entered.set()
            release_old_health.wait()
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr("basal.ui.app.httpx.get", healthy)
    old_thread = threading.Thread(target=runner._start, args=(old, 1, False, tmp_path, None), name="old-start")
    old_thread.start()
    assert old_health_entered.wait(timeout=2)
    runner.gen = 2
    runner.state, runner.model = "starting", "newer"
    runner._start(newer, 2, False, tmp_path, None)
    assert [e["id"] for e in settings.last_models()] == ["newer"]
    release_old_health.set()
    old_thread.join(timeout=2)
    assert not old_thread.is_alive()
    assert [e["id"] for e in settings.last_models()] == ["newer"]


def test_load_over_the_memory_budget_is_refused_until_the_budget_allows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_start", lambda *args: None)
    fleet = make_fleet(tmp_path, Hardware("cpu", "cpu", 16 * 1024**3, None, None, frozenset()))

    def assessment(model_id):
        variant = Variant(model_id, f"repo/{model_id}", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
        return SimpleNamespace(variant=variant, mode="eager", est_memory_bytes=10 * 1024**3)

    fleet.load(assessment("a"))
    with pytest.raises(RuntimeError) as refused:
        fleet.load(assessment("b"))
    assert str(refused.value) == ("over memory budget: 10.0 GiB loaded + 10.0 GiB needed > 14.0 GiB (16.0 GiB budget − "
                                  "2.0 GiB kept free); unload a model or raise the budget")
    assert list(fleet.runners) == ["a"]
    settings.update_memory({"budget_bytes": 21 * 1024**3})
    with pytest.raises(RuntimeError, match="over memory budget"):
        fleet.load(assessment("b"))
    settings.update_memory({"reserve_bytes": 0})
    fleet.load(assessment("b"))
    memory = fleet.status()["memory"]
    assert (memory["committed_bytes"], memory["budget_bytes"], memory["over"]) == (20 * 1024**3, 21 * 1024**3, False)


def test_memory_of_an_unloading_model_stays_committed_until_its_child_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_start", lambda *args: None)
    fleet = make_fleet(tmp_path, Hardware("cpu", "cpu", 16 * 1024**3, None, None, frozenset()))
    variant = Variant("a", "repo/a", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
    fleet.load(SimpleNamespace(variant=variant, mode="eager", est_memory_bytes=10 * 1024**3))
    fleet.runners["a"].process = SimpleNamespace(pid=0, poll=lambda: None)
    terminating, release = threading.Event(), threading.Event()

    def slow_terminate(proc):
        terminating.set()
        release.wait()

    monkeypatch.setattr(Runner, "_terminate", staticmethod(slow_terminate))
    threading.Thread(target=fleet.unload, args=("a",), daemon=True).start()
    assert terminating.wait(2)
    assert fleet.status()["models"] == [] and fleet.status()["memory"]["committed_bytes"] == 10 * 1024**3
    with pytest.raises(RuntimeError, match="over memory budget: 10.0 GiB loaded"):
        fleet.load(SimpleNamespace(variant=variant, mode="eager", est_memory_bytes=10 * 1024**3))
    release.set()
    wait_for(lambda: fleet.status()["memory"]["committed_bytes"] == 0)


def test_temporary_runner_memory_stays_committed_until_unload_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_start", lambda *args: None)
    fleet = make_fleet(tmp_path, Hardware("cpu", "cpu", 16 * 1024**3, None, None, frozenset()))
    assessment = SimpleNamespace(
        variant=Variant("temp", "repo/temp", "basal-2.0", "mini", "bf16", "bf16", None, 100, False),
        mode="eager", est_memory_bytes=10 * 1024**3)
    fleet.start_temporary("job", assessment, fleet.logs["basal-ui"])
    started, release = threading.Event(), threading.Event()

    def slow_unload(self):
        started.set()
        release.wait()

    monkeypatch.setattr(Runner, "unload", slow_unload)
    stopping = threading.Thread(target=fleet.stop_temporary, args=("job",), daemon=True)
    stopping.start()
    assert started.wait(2)
    assert fleet.status()["memory"]["committed_bytes"] == 10 * 1024**3
    other = SimpleNamespace(
        variant=Variant("other", "repo/other", "basal-2.0", "mini", "bf16", "bf16", None, 100, False),
        mode="eager", est_memory_bytes=10 * 1024**3)
    with pytest.raises(RuntimeError, match="over memory budget"):
        fleet.load(other)
    release.set()
    stopping.join(timeout=2)
    assert not stopping.is_alive() and fleet.status()["memory"]["committed_bytes"] == 0


def test_restart_reserves_memory_until_the_replacement_load_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_start", lambda *args: None)
    fleet = make_fleet(tmp_path, Hardware("cpu", "cpu", 16 * 1024**3, None, None, frozenset()))

    def assessment(model_id):
        variant = Variant(model_id, f"repo/{model_id}", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
        return SimpleNamespace(variant=variant, mode="eager", est_memory_bytes=10 * 1024**3)

    fleet.load(assessment("a"))
    fleet.runners["a"].state = "ready"
    entered, release = threading.Event(), threading.Event()
    original_load = fleet.load

    def delayed_load(value, media=False, options=None, _reservation=None):
        if _reservation is not None:
            entered.set()
            release.wait()
        return original_load(value, media=media, options=options, _reservation=_reservation)

    monkeypatch.setattr(fleet, "load", delayed_load)
    errors = []

    def restart():
        try:
            fleet.restart(assessment("a"))
        except (RuntimeError, KeyError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=restart, daemon=True)
    thread.start()
    assert entered.wait(2)
    with pytest.raises(RuntimeError, match="over memory budget"):
        fleet.load(assessment("b"))
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and not errors and "restart:a" not in fleet.reserved


def test_memory_routes_validate_and_report_the_hardware_budget(tmp_path):
    app, _ = app_for(tmp_path, [], hw=Hardware("cpu", "cpu", 16 * 1024**3, None, None, frozenset(), 5))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/memory").json() == {
            "settings": {"budget_bytes": None, "reserve_bytes": 2 * 1024**3},
            "hardware_budget_bytes": 16 * 1024**3, "available_bytes": 5}
        assert client.put("/api/memory", json={"budget_bytes": 1024}).status_code == 400
        assert client.put("/api/memory", json={"reserve_bytes": -1}).status_code == 400
        saved = client.put("/api/memory", json={"budget_bytes": 8 * 1024**3, "reserve_bytes": 0}).json()
        assert saved["settings"] == {"budget_bytes": 8 * 1024**3, "reserve_bytes": 0}
        assert client.get("/api/status").json()["memory"]["budget_bytes"] == 8 * 1024**3


def test_rss_is_summed_per_process_group(monkeypatch):
    if os.name != "posix":
        pytest.skip("ps is POSIX-only")
    output = "  100  2048\n  100   1024\n  200 10\nbad line\n"
    monkeypatch.setattr("basal.ui.app.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=output))
    assert rss_by_pgid() == {100: 3072 * 1024, 200: 10 * 1024}


def test_benchmark_decisions_leave_statistics_and_history_untouched(tmp_path, monkeypatch, instant_start):
    response = httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.9, "confidence": 0.9}}})
    monkeypatch.setattr(Runner, "decide", lambda self, body, path="/v1/systemone": response)
    fleet = make_fleet(tmp_path)
    fleet.load(SimpleNamespace(variant=MINI, mode="eager", est_memory_bytes=None))
    wait_for(lambda: fleet.loaded_ids() == ["mini"])
    fleet.decide("mini", {"state": "x"}, source="benchmark")
    assert fleet.stats.snapshot()["models"] == {} and fleet.history.list() == []
    fleet.decide("mini", {"state": "x"}, source="playground")
    assert fleet.stats.snapshot()["models"]["mini"]["requests"] == 1
    assert [(r["source"], r["answers"]["q"]["best"]) for r in fleet.history.list()] == [("playground", "true")]


def test_logs_route_lists_sources_and_reads_incrementally(tmp_path, instant_start):
    app, fleet = app_for(tmp_path, [MINI])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post("/api/load", json={"id": "mini"})
        logs = client.get("/api/logs").json()
        assert logs["sources"] == [{"id": "basal-ui", "state": "ui"}, {"id": "mini", "state": "ready"}]
        assert "info load requested: mini" in [line["text"] for line in logs["lines"]]
        last = logs["lines"][-1]["seq"]
        assert client.get("/api/logs", params={"after": last}).json()["lines"] == []
        fleet.runners["mini"].log.append("a child line")
        assert [line["text"] for line in client.get("/api/logs", params={"source": "mini"}).json()["lines"]] == [
            "a child line"]
        assert client.get("/api/logs", params={"source": "nope"}).status_code == 404


def test_mutation_origin_requires_matching_scheme_host_and_port(tmp_path):
    app, _ = app_for(tmp_path, [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        accepted = client.post("/api/download", json={"id": "missing"}, headers={"Origin": "http://localhost"})
        assert accepted.status_code == 400
        assert client.post("/api/download", json={"id": "missing"},
                           headers={"Origin": "http://localhost:12345"}).status_code == 403
        assert client.post("/api/download", json={"id": "missing"},
                           headers={"Origin": "https://localhost"}).status_code == 403


def test_shutdown_waits_for_gateway_drain_before_shared_state_teardown(tmp_path, monkeypatch):
    order = []

    class GatewayStub:
        attempts = 0

        def start(self):
            return None

        def stop(self):
            self.attempts += 1
            order.append("gateway")
            return self.attempts > 1

    fleet = make_fleet(tmp_path)
    gateway = GatewayStub()
    fleet.shutdown = lambda: order.append("fleet")
    fleet.stats.close = lambda: order.append("stats")
    monkeypatch.setattr(settings, "remove_keys_file", lambda: order.append("keys"))
    app = create_app(catalog_fn=lambda **_: ([], "test"), hardware_fn=lambda: HW, fleet=fleet,
                     stats=fleet.stats, history=fleet.history, gateway=gateway)
    with TestClient(app, base_url="http://127.0.0.1"):
        pass
    assert order == ["gateway", "gateway", "fleet", "stats", "keys"]


def test_presets_validate_and_strip_request_model(tmp_path):
    app, _ = app_for(tmp_path, [])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/presets").json() == {"load": {}, "request": {}}
        result = client.put("/api/presets", json={"kind": "load", "name": "cpu", "value": {"mode": "eager"}})
        assert result.status_code == 200 and result.json()["load"]["cpu"] == {"mode": "eager"}  # the non-default ones
        result = client.put("/api/presets", json={"kind": "request", "name": "hello",
                                                  "value": {"model": "ignored", "questions": {"q": {"type": "choice"}}}})
        assert result.json()["request"]["hello"] == {"questions": {"q": {"type": "choice"}}}
        assert client.put("/api/presets", json={"kind": "load", "name": "bad", "value": {"unknown": 1}}).status_code == 400
        assert client.put("/api/presets", json={"kind": "request", "name": "bad", "value": {"questions": {}}}).status_code == 400
        assert client.put("/api/presets", json={"kind": "load", "name": ".hidden", "value": {}}).status_code == 400


def test_model_card_reports_downloaded_local_data_and_unknown_directory(tmp_path, monkeypatch):
    community = Variant("community", "pawelkiszczak/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16",
                       None, 123, False, (), "pawelkiszczak")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text("Local model card")
    (snapshot / "basal.json").write_text('{"prompt": 2}')
    checks, modes, urls = [], [], []

    def is_downloaded(variant, cache, revision=None):
        checks.append((variant.id, revision))
        return snapshot if variant.id == community.id else None
    monkeypatch.setattr(catalog, "is_downloaded", is_downloaded)
    monkeypatch.setattr(calibration, "shipped_file", lambda local_dir, mode: modes.append(mode) or snapshot / "none")
    fp8 = {"quantization_config": {"quant_algo": "FP8", "quant_method": "modelopt", "producer": {"version": "0.47.0"}}}
    monkeypatch.setattr(httpx, "get", lambda url, **k: urls.append(url) or (
        httpx.Response(200, json=fp8) if url.endswith("config.json") else httpx.Response(404)))
    app, _ = app_for(tmp_path, [community, MINI])
    settings.set_load_options("community", {"mode": "mlx-q8", "revision": "abc"})
    settings.set_load_options("mini", {"revision": "refs/pr/2"})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        card = client.get("/api/models/card", params={"id": "community"}).json()
        missing = client.get("/api/models/card", params={"id": "mini"}).json()
    assert card["downloaded"] and card["local_dir"] == str(snapshot)
    assert card["author"] == "pawelkiszczak" and card["community"] is True
    assert card["basal_json"] == {"prompt": 2} and card["readme"] == "Local model card"
    assert missing["downloaded"] is False and missing["local_dir"] is None and missing["basal_json"] is None
    assert checks == [("community", "abc"), ("mini", "refs/pr/2")]  # the snapshot of the served revision
    assert modes == ["mlx-q8"]  # the calibration file of the served mode, not the default one
    assert card["quantisation"] == {"scheme": "bf16", "description": None, "engine": None}
    assert missing["quantisation"]["scheme"] == "FP8 (modelopt 0.47.0)"
    hub = "https://huggingface.co/Remek/basal-2.0-mini/raw/refs/pr/2"
    assert urls == [f"{hub}/README.md", f"{hub}/basal.json", f"{hub}/config.json"]


def test_catalog_rows_carry_the_saved_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "downloaded_bytes", lambda variant, cache_dir: None)
    app, _ = app_for(tmp_path, [MINI, SMALL])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.put("/api/models/options", json={"id": "mini", "options": {"revision": " abc "}}).status_code == 200
        rows = {m["id"]: m for m in client.get("/api/models").json()["models"]}
    assert rows["mini"]["revision"] == "abc" and rows["small"]["revision"] is None


def test_revision_reaches_cached_check_and_download_but_not_child_flags(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(catalog, "is_downloaded",
                        lambda variant, cache_dir, revision=None: calls.append(("check", revision)) or None)

    class Finished:
        @staticmethod
        def is_alive():
            return False

    class Downloads:
        @staticmethod
        def start(variant, cache_dir, revision=None):
            calls.append(("download", revision))
            return {"thread": Finished(), "error": None, "path": tmp_path}

    runner = Runner(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", downloads=Downloads())
    assessment = catalog.assess(MINI, HW)
    runner.options = {"revision": "refs/pr/5"}
    runner._build_command = lambda *args: ["fake"]
    runner._models_entry = lambda media: (None, [])
    monkeypatch.setattr("basal.ui.app.subprocess.Popen",
                        lambda *args, **kwargs: SimpleNamespace(stdout=__import__("io").StringIO(), poll=lambda: None,
                                                                returncode=None, pid=0))
    monkeypatch.setattr("basal.ui.app.httpx.get", lambda *args, **kwargs: SimpleNamespace(status_code=200))
    runner._start(assessment, runner.gen, cache_dir=tmp_path)
    assert calls == [("check", "refs/pr/5"), ("download", "refs/pr/5")]
    del runner._build_command
    command = runner._build_command(MINI, tmp_path, assessment, False, 1, settings.serve_settings())
    assert "--revision" not in command
