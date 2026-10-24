import hashlib
import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from basal.auth import ApiKeys
from basal.ui import settings
from basal.ui.app import Fleet, Runner
from basal.ui.gateway import Gateway, build_app
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A real fleet with two ready models whose basal-serve echoes the body it receives."""
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                  hardware_fn=lambda: HW, stats=Stats(tmp_path / "stats.json"),
                  history=History(tmp_path / "history.jsonl"))
    for model_id in ("a/one", "b/two"):
        runner = Runner(internal_secret="x")
        runner.state, runner.model, runner.ready_at = "ready", model_id, 1700000000.0
        runner.models_entry = {"name": model_id, "media": []}
        fleet.runners[model_id] = runner
    calls = []

    async def adecide(self, client, body, path="/v1/systemone"):
        calls.append((self.model, path, body))
        if body.get("state") == "fail":
            return httpx.Response(422, json={"error": "bad question", "code": "INVALID_REQUEST"})
        return httpx.Response(200, json={"model": self.model, "answers": {
            "answer": {"type": "choice", "choice": "x", "confidence": 0.6}}, "usage": {"input_tokens": 3}})

    monkeypatch.setattr(Runner, "adecide", adecide)
    fleet.calls = calls
    return fleet


def test_model_is_required_and_must_be_loaded_in_both_error_envelopes(fleet):
    with TestClient(build_app(fleet, fleet.stats), base_url="http://127.0.0.1") as client:
        missing = client.post("/v1/systemone", json={"state": "x", "questions": {}})
        assert missing.status_code == 400
        assert missing.json() == {"error": "model is required; loaded models: a/one, b/two", "code": "MODEL_REQUIRED"}
        unknown = client.post("/v1/decisions", json={"model": "nope", "input": "x", "questions": []})
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "MODEL_NOT_FOUND"
        assert unknown.json()["error"]["message"] == "model 'nope' is not loaded; loaded models: a/one, b/two"
        assert client.post("/decisions", content=b"{").json()["error"]["code"] == "INVALID_JSON"
        assert client.post("/v1/basal", json=[1]).json()["code"] == "INVALID_JSON"
    assert fleet.calls == []


def test_gateway_rejects_untrusted_host_on_forwarding_routes(fleet):
    with TestClient(build_app(fleet, fleet.stats), base_url="http://127.0.0.1") as client:
        response = client.post("/v1/systemone", json={"model": "a/one"},
                               headers={"Host": "evil.example"})
    assert response.status_code in (400, 403)


def test_forwarding_counts_and_records_api_decisions(fleet):
    with TestClient(build_app(fleet, fleet.stats), base_url="http://127.0.0.1") as client:
        ok = client.post("/v1/systemone", json={"model": "b/two", "state": "s", "questions": {}},
                         headers={"x-request-id": "client-1"})
        assert ok.status_code == 200 and ok.json()["model"] == "b/two"
        assert ok.headers["x-request-id"] == "client-1"
        failed = client.post("/v1/basal", json={"model": "a/one", "state": "fail", "questions": {}})
        assert failed.status_code == 422 and failed.json()["error"] == "bad question"
        client.post("/v1/decisions", json={"model": "a/one", "input": "s", "questions": []})
    assert fleet.calls == [("b/two", "/v1/systemone", {"state": "s", "questions": {}}),
                           ("a/one", "/v1/systemone", {"state": "fail", "questions": {}}),
                           ("a/one", "/v1/decisions", {"model": "a/one", "input": "s", "questions": []})]
    models = fleet.stats.snapshot()["models"]
    assert (models["b/two"]["requests"], models["a/one"]["requests"], models["a/one"]["errors"]) == (1, 2, 1)
    history = fleet.history.list()
    assert [(r["model"], r["source"], r["status"]) for r in history] == [
        ("a/one", "api", 200), ("a/one", "api", 422), ("b/two", "api", 200)]
    assert fleet.history.get(history[-1]["id"])["request"] == {"model": "b/two", "state": "s", "questions": {}}


def test_models_list_in_basal_and_openai_shapes(fleet):
    fleet.runners["b/two"].state = "starting"
    with TestClient(build_app(fleet, fleet.stats), base_url="http://127.0.0.1") as client:
        listing = client.get("/v1/models").json()
        assert listing == {"object": "list", "models": [{"name": "a/one", "media": []}],
                           "data": [{"id": "a/one", "object": "model", "created": 1700000000, "owned_by": "basal"}]}
        assert client.get("/health").json() == {"status": "ok", "models": ["a/one"]}


def test_keys_guard_every_route_and_stats_need_the_admin_key(fleet, tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"keys": [
        {"name": "user", "sha256": hashlib.sha256(b"user-secret").hexdigest()},
        {"name": "basal-ui", "sha256": hashlib.sha256(b"admin-secret").hexdigest(), "admin": True}]}))
    fleet.stats.record("a/one", {"answers": {}}, 1.0)
    app = build_app(fleet, fleet.stats, ApiKeys(keys_file), cors_origins=["https://app.example"])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/v1/systemone", json={"model": "a/one"}).status_code == 401
        assert client.get("/health").status_code == 200
        user = {"Authorization": "Bearer user-secret"}
        assert client.post("/v1/systemone", json={"model": "a/one", "state": "s"}, headers=user).status_code == 200
        assert client.get("/v1/stats", headers=user).status_code == 403
        stats = client.get("/v1/stats", headers={"X-API-Key": "admin-secret"})
        assert stats.status_code == 200 and stats.json()["models"]["a/one"]["requests"] == 2
        preflight = client.options("/v1/systemone", headers={"Origin": "https://app.example",
                                                             "Access-Control-Request-Method": "POST"})
        assert preflight.headers["access-control-allow-origin"] == "https://app.example"


def test_mcp_tools_take_a_model_and_list_the_loaded_ones(fleet):
    pytest.importorskip("mcp")
    headers = {"Accept": "application/json, text/event-stream", "Host": "127.0.0.1:8793"}

    def rpc(client, method, params):
        response = client.post("/mcp", headers=headers,
                               json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        assert response.status_code == 200, response.text
        return response.json()["result"]

    with TestClient(build_app(fleet, fleet.stats, mcp=True), base_url="http://127.0.0.1") as client:
        assert rpc(client, "tools/call", {"name": "list_models", "arguments": {}})["structuredContent"] == {
            "models": ["a/one", "b/two"]}
        answer = rpc(client, "tools/call", {"name": "decide", "arguments": {"model": "b/two", "state": "s",
                                                                           "question": "?"}})
        assert answer["structuredContent"]["model"] == "b/two"
        refused = rpc(client, "tools/call", {"name": "decide", "arguments": {"model": "zz", "state": "s",
                                                                            "question": "?"}})
        assert refused["isError"] and "model 'zz' is not loaded" in refused["content"][0]["text"]
    assert [(r["model"], r["source"]) for r in fleet.history.list()] == [("b/two", "mcp")]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_gateway_serves_on_the_configured_port_and_reports_restarts(fleet):
    gateway = Gateway(fleet, settings, fleet.stats)
    assert gateway.start() is None  # serving disabled
    port = free_port()
    settings.update_serve({"enabled": True, "port": port})
    try:
        status = gateway.start()
        assert (status["running"], status["url"], status["models"]) == (True, f"http://127.0.0.1:{port}",
                                                                          ["a/one", "b/two"])
        assert status["restart_required"] is False
        assert httpx.get(f"http://127.0.0.1:{port}/health").json()["models"] == ["a/one", "b/two"]
        settings.update_serve({"require_key": True})
        assert gateway.status()["restart_required"] is True
        status = gateway.restart()
        assert status["require_key"] is True and status["restart_required"] is False
        assert httpx.post(f"http://127.0.0.1:{port}/v1/systemone", json={}).status_code == 401
        internal = {"Authorization": f"Bearer {fleet.internal_secret}"}
        assert httpx.get(f"http://127.0.0.1:{port}/v1/stats", headers=internal).status_code == 200
        assert "basal-ui" in gateway.key_usage()
    finally:
        gateway.stop()
    assert gateway.status()["running"] is False and gateway.status()["restart_required"] is True


def test_port_in_use_is_reported_and_lan_url_is_used_for_all_interfaces(fleet, monkeypatch):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    settings.update_serve({"enabled": True, "port": port})
    try:
        status = Gateway(fleet, settings, fleet.stats).start()
        assert (status["running"], status["error"]) == (False, f"port {port} on 127.0.0.1 is in use")
        assert f"error API server: port {port} on 127.0.0.1 is in use" in [
            line["text"] for line in fleet.logs["basal-ui"].since()]
    finally:
        listener.close()
    calls = []
    monkeypatch.setattr("basal.ui.app._lan_ip", lambda: calls.append(1) or "192.0.2.10")
    gateway = Gateway(fleet, settings, fleet.stats)
    gateway.snapshot = {"host": "0.0.0.0", "port": 8123, "require_key": False, "cors_origins": [], "mcp": False}
    assert gateway.status()["url"] == "http://192.0.2.10:8123"
    gateway.status()
    assert calls == [1]
