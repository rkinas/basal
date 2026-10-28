"""MCP: the two tools build the HTTP API's request and return its answer, the server's errors reach the agent,
--mcp on serves the tools at /mcp next to /v1/systemone (Host checked on a loopback bind), the basal-mcp bridge
forwards over HTTP, and --mcp on without the package is refused before a model loads."""
import asyncio
import json
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager

import httpx
import pytest

pytest.importorskip("mcp")

import basal.server as server_module  # noqa: E402
from basal import mcp_server  # noqa: E402
from mcp.client.client import Client  # noqa: E402
from tests.test_server import FakeServer  # noqa: E402


class Engine:
    """A basal server stand-in for the tools: Server.decide over the stub backend of the server tests."""
    def __init__(self, pos_probs=(0.8, 0.2)):
        self.fake, self.name, self.bodies = FakeServer(list(pos_probs)), "basal-test", []

    async def decide(self, body):
        self.bodies.append(body)
        return await self.fake._run(body)


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)
    return asyncio.run(go())


def test_two_read_only_tools_without_models_keep_their_schemas():
    async def go():
        async with Client(mcp_server.build(Engine().decide, "basal-test")) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"decide", "decide_many"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools.values())
    assert tools["decide"].input_schema["required"] == ["state", "question"]
    assert "model" not in tools["decide"].input_schema["properties"]
    assert "model" not in tools["decide_many"].input_schema["properties"]


def test_decide_asks_one_question():
    eng = Engine()
    r = call(mcp_server.build(eng.decide), "decide", {"state": "Invoice 1240 PLN, order 1240 PLN.",
                                                      "question": "Approve the invoice?",
                                                      "options": {"approve": "amounts match", "hold": "they differ"},
                                                      "lang": "en"})
    assert not r.is_error
    assert eng.bodies[-1] == {"state": "Invoice 1240 PLN, order 1240 PLN.", "lang": "en", "questions": {"answer": {
        "type": "choice", "instructions": "Approve the invoice?",
        "criteria": {"approve": "amounts match", "hold": "they differ"}}}}
    out = r.structured_content
    assert out["model"] == "basal-test" and out["type"] == "choice" and out["choice"] == "approve"
    assert set(out["probabilities"]) == {"approve", "hold"} and out["confidence"] == max(out["probabilities"].values())


def test_decide_noul_without_options():
    eng = Engine()
    r = call(mcp_server.build(eng.decide), "decide", {"state": {"filed": "2026-10-01"}, "question": "Filed on time?",
                                                      "type": "noul"})
    assert not r.is_error and r.structured_content["type"] == "noul"
    assert eng.bodies[-1]["questions"]["answer"] == {"type": "noul", "instructions": "Filed on time?"}


def test_decide_many_passes_the_api_request_through():
    eng = Engine()
    qs = {"route": {"type": "choice", "instructions": "Department?", "criteria": ["billing", "tech"]},
          "urgent": {"type": "noul", "instructions": "Urgent?"}}
    r = call(mcp_server.build(eng.decide), "decide_many", {"state": "Printer on fire.", "questions": qs,
                                                           "facts": "auto"})
    assert not r.is_error
    assert eng.bodies[-1] == {"state": "Printer on fire.", "questions": qs, "facts": "auto"}
    assert set(r.structured_content["answers"]) == {"route", "urgent"} and "usage" in r.structured_content


def test_server_errors_reach_the_agent():
    r = call(mcp_server.build(Engine().decide), "decide", {"state": "x", "question": "?", "options": ["only"]})
    assert r.is_error and "1 options (supported: 2..10)" in r.content[0].text   # the server's message, not a generic one


def test_multiple_models_are_listed_and_selected():
    eng = Engine()

    async def go():
        async with Client(mcp_server.build(eng.decide, models=lambda: ["a", "b"])) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            listed = await c.call_tool("list_models", {})
            result = await c.call_tool(
                "decide", {"model": "b", "state": "s", "question": "q?", "options": ["yes", "no"]})
            many = await c.call_tool("decide_many", {"model": "a", "state": "s",
                                                     "questions": {"answer": {"type": "noul", "instructions": "q?"}}})
            return tools, listed, result, many

    tools, listed, result, many = asyncio.run(go())
    assert set(tools) == {"decide", "decide_many", "list_models"}
    assert "model" in tools["decide"].input_schema["properties"]
    assert "model" not in tools["decide"].input_schema["required"]
    assert "model" in tools["decide_many"].input_schema["properties"]
    assert "model" not in tools["decide_many"].input_schema["required"]
    assert listed.structured_content == {"models": ["a", "b"]}
    assert not result.is_error and not many.is_error
    assert eng.bodies[-2] == {"state": "s", "questions": {"answer": {
        "type": "choice", "instructions": "q?", "criteria": ["yes", "no"]}}, "model": "b"}
    assert eng.bodies[-1] == {"state": "s", "questions": {"answer": {"type": "noul", "instructions": "q?"}},
                              "model": "a"}


def test_unknown_model_error_reaches_the_agent():
    async def decide(body):
        raise ValueError(f"unknown model: {body['model']}")

    result = call(mcp_server.build(decide, models=lambda: ["a"]), "decide",
                  {"model": "unknown", "state": "s", "question": "q?"})
    assert result.is_error and "unknown model: unknown" in result.content[0].text


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port):
    import uvicorn
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            return srv
        time.sleep(0.02)
    raise RuntimeError("server did not start")


def test_mcp_next_to_the_api():
    """As basal-serve --mcp on --host 127.0.0.1 mounts it: /mcp and the HTTP API in one app, one queue."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    eng = Engine()
    routes, run = mcp_server.http(eng, "127.0.0.1")

    async def systemone(request):
        return JSONResponse(await eng.decide(await request.json()))

    @asynccontextmanager
    async def lifespan(_app):
        async with run():
            yield

    port = free_port()
    srv = serve(Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]), *routes], lifespan=lifespan),
                port)
    try:
        r = call(f"http://127.0.0.1:{port}/mcp", "decide", {"state": "s", "question": "q?", "options": ["a", "b"]})
        assert not r.is_error and r.structured_content["choice"] == "a"
        api = httpx.post(f"http://127.0.0.1:{port}/v1/systemone", json=eng.bodies[-1]).json()["answers"]["answer"]
        assert {k: v for k, v in r.structured_content.items() if k != "model"} == api   # the same answer
        bad = httpx.post(f"http://127.0.0.1:{port}/mcp", headers={"Host": "attacker.example"}, json={})
        assert bad.status_code in (400, 403, 421)       # DNS rebinding: a foreign Host is refused on a loopback bind
    finally:
        srv.should_exit = True


def test_bridge_forwards_over_http_and_reports_errors():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.read()))
        if b"guess" in request.content:
            return httpx.Response(422, json={"error": "unknown question type: guess"})
        return httpx.Response(200, json={"model": "m", "answers": {"answer": {"type": "noul", "noul": 0.7}}})

    decide = mcp_server.remote("http://basal:8000/", transport=httpx.MockTransport(handler))
    r = call(mcp_server.build(decide), "decide", {"state": "s", "question": "q?", "type": "noul"})
    assert r.structured_content == {"model": "m", "type": "noul", "noul": 0.7} and seen[0][0] == "/v1/systemone"
    r = call(mcp_server.build(decide), "decide_many", {"state": "s", "questions": {"q": {"type": "guess"}}})
    assert r.is_error and "422" in r.content[0].text and "unknown question type: guess" in r.content[0].text


def test_bridge_model_option_names_the_model_in_every_request():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {}})

    decide = mcp_server.remote("http://basal:8000", transport=httpx.MockTransport(handler), model="Remek/m")
    asyncio.run(decide({"state": "x"}))
    assert seen == [{"model": "Remek/m", "state": "x"}]


def test_bridge_model_flag_configures_remote(monkeypatch):
    captured = {}

    def remote(url, timeout, model):
        captured["remote"] = (url, timeout, model)

    class Bridge:
        def run(self, transport):
            captured["transport"] = transport

    monkeypatch.setattr(sys, "argv", ["basal-mcp", "--model", "Remek/m"])
    monkeypatch.setattr(mcp_server, "available", lambda: None)
    monkeypatch.setattr(mcp_server, "remote", remote)
    monkeypatch.setattr(mcp_server, "build", lambda decide, model: Bridge())
    mcp_server.main()
    assert captured == {"remote": ("http://127.0.0.1:8000", 60.0, "Remek/m"), "transport": "stdio"}


def test_mcp_without_the_package_fails_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "eager", "--mcp", "on"])
    monkeypatch.setattr(mcp_server, "available", lambda: mcp_server.INSTALL)
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match=r"basal\[mcp\]"):
        server_module.main()
