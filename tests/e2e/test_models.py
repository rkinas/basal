"""Several loaded models: the API server on one port (keys, model routing, both request shapes, MCP), the memory
budget, statistics, history, logs, restart and unload."""
import httpx
import pytest

from tests.e2e.support import CHOICE, free_port


@pytest.fixture
def api_server(ui):
    """basal-ui's API server on a free port with a required key; -> (client with the key, the key, the serve state)."""
    port = free_port()
    serve = ui.get("/api/serve")
    ui.put("/api/serve", {"enabled": True, "port": port, "require_key": True, "mcp": bool(serve["mcp_available"])})
    ui.post("/api/serve/restart")
    key = ui.post("/api/serve/keys", {"name": "e2e"})["key"]
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {key}"}, timeout=300)
    try:
        yield client, key, ui.get("/api/serve")
    finally:
        client.close()
        ui.delete("/api/serve/keys/e2e")
        ui.put("/api/serve", {"enabled": False, "require_key": False, "mcp": False})
        ui.post("/api/serve/restart")


def test_two_models_answer_behind_one_api_port_with_a_key(ui, text_model, second_model, api_server):
    a, b = text_model["id"], second_model["id"]
    ui.load(a)
    ui.load(b)
    api, key, serve = api_server
    listing = api.get("/v1/models").json()
    assert {m["id"] for m in listing["data"]} >= {a, b} and len(listing["models"]) == len(listing["data"])
    assert httpx.post(f"{api.base_url}/v1/systemone", json={"model": a, **CHOICE}).status_code == 401  # no key
    missing = api.post("/v1/systemone", json=CHOICE)
    assert missing.status_code == 400 and missing.json()["code"] == "MODEL_REQUIRED"
    unknown = api.post("/v1/systemone", json={"model": "Remek/not-loaded", **CHOICE})
    assert unknown.status_code == 404
    for i, model_id in enumerate((a, b)):
        r = api.post("/v1/systemone", json={"model": model_id, **CHOICE}, headers={"x-request-id": f"e2e-{i}"})
        assert r.status_code == 200 and r.headers["x-request-id"] == f"e2e-{i}"
        answer = r.json()["answers"]["dept"]
        assert answer["choice"] in ("cards", "loans", "online") and abs(sum(answer["probabilities"].values()) - 1) < 1e-3
    decision = api.post("/v1/decisions", json={
        "model": a, "input": CHOICE["state"],
        "questions": [{"type": "choice", "name": "dept", "instructions": "Do którego działu?",
                       "choices": [{"value": "cards"}, {"value": "loans"}]}]})
    assert decision.status_code == 200, decision.text
    assert api.get("/v1/stats").status_code == 403  # a user key does not read the statistics

    stats = ui.get("/api/stats")["models"]
    assert stats[a]["requests"] >= 2 and stats[b]["requests"] >= 1 and "choice" in stats[a]["by_type"]
    history = ui.get("/api/history", params={"source": "api", "model": a})["items"]
    assert history and history[0]["status"] == 200 and history[0]["answers"]
    record = ui.get(f"/api/history/{history[-1]['id']}")
    assert record["request"]["model"] == a and "answers" in record["response"]
    assert any(k["name"] == "e2e" and k["last_used"] for k in ui.get("/api/serve")["keys"])

    if serve["settings"]["mcp"]:
        mcp = {"Accept": "application/json, text/event-stream", "Authorization": f"Bearer {key}"}

        def rpc(method, params):
            r = api.post("/mcp", headers=mcp, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            assert r.status_code == 200, r.text
            return r.json()["result"]

        assert set(rpc("tools/call", {"name": "list_models", "arguments": {}})["structuredContent"]["models"]) >= {a, b}
        answer = rpc("tools/call", {"name": "decide", "arguments": {"model": b, "state": CHOICE["state"], "type": "noul",
                                                                    "question": "Czy klient prosi o odblokowanie?"}})
        assert not answer.get("isError") and answer["structuredContent"]["model"] == b, answer
        assert 0 <= answer["structuredContent"]["noul"] <= 1
        assert any(r["source"] == "mcp" for r in ui.get("/api/history", params={"model": b})["items"])


def test_memory_budget_refuses_a_load_that_does_not_fit(ui, text_model, second_model, hardware):
    ui.load(text_model["id"])
    ui.unload(second_model["id"])
    memory = ui.get("/api/memory")["settings"]
    try:
        ui.put("/api/memory", {"budget_bytes": None, "reserve_bytes": hardware["memory_bytes"]})  # nothing fits
        detail = ui.detail("POST", "/api/load", {"id": second_model["id"]}, 409)
        assert detail.startswith("over memory budget")
        assert ui.status()["memory"]["over"] is True
    finally:
        ui.put("/api/memory", memory)
    assert ui.status()["memory"]["over"] is False


def test_restart_unload_logs_and_the_download_button(ui, second_model):
    model_id = second_model["id"]
    first = ui.load(model_id)
    ui.post("/api/restart", {"id": model_id})
    again = ui.wait_model(model_id)
    assert again["state"] == "ready" and again["base_url"] != first["base_url"]  # a new child on a new port
    sources = {s["id"]: s["state"] for s in ui.get("/api/logs")["sources"]}
    assert sources[model_id] == "ready"
    assert ui.get("/api/logs", params={"source": model_id})["lines"]
    assert ui.detail("POST", "/api/download", {"id": model_id}, 409) == "already on disk"
    ui.unload(model_id)
    assert ui.model(model_id) is None
    assert ui.detail("POST", "/api/decide", {"model": model_id, **CHOICE}, 404).endswith("is not loaded")
