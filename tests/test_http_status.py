"""basal-serve's status codes and load (#22): an invalid request answers 422, the GPU out of memory 503 and any other
server error 500, all with {"error": ...}; GET /health counts the requests in flight as running and queued. Through
the app of basal-serve (http_app) with its batching worker and a stub engine."""
import asyncio
import hashlib
import json
import threading

import httpx
import pytest
import torch

from basal.auth import ApiKeys, protect, request_ids
from basal.prompt import RequestError
from basal.server import http_app, http_status, out_of_memory
from tests.test_server import FakeServer

Q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta"}}
BODY = {"state": "Invoice 1240 PLN.", "questions": {"q": Q}}


class Engine:
    """run_shared of a backend: equal probabilities, an exception to raise, or a wait for `go` (a request in flight)."""
    policies = {}

    def __init__(self, error=None):
        self.error, self.entered, self.go = error, threading.Event(), threading.Event()
        self.go.set()

    def run_shared(self, groups, policy=None):
        self.entered.set()
        self.go.wait(10)
        if self.error is not None:
            raise self.error
        return [[[1 / len(x)] * len(x) for x in g[1]] for g in groups]


class AppServer(FakeServer):
    def __init__(self, engine):
        super().__init__([0.5, 0.5])
        self.backend, self.max_batch, self.wait, self.long_tokens = engine, 8, 0.0, None
        self.context, self.overflow, self.trunc_head = (8192, "model context"), "refuse", 1 / 3
        self.cal, self.cal_mismatch, self.temps, self.fmt, self.orders = {}, None, {}, 1, 2
        self.queue = asyncio.Queue()


async def on_app(srv, fn, decorate=None):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        app = http_app(srv, "fast")
        transport = httpx.ASGITransport(app=decorate(app) if decorate else app)
        async with httpx.AsyncClient(transport=transport, base_url="http://basal") as c:
            return await fn(c)
    finally:
        worker.cancel()


def post(srv, body=None, content=None, decorate=None):
    async def go(c):
        return await (c.post("/v1/systemone", content=content) if content is not None
                      else c.post("/v1/systemone", json=body))
    return asyncio.run(on_app(srv, go, decorate))


def test_models_endpoint_reports_context_and_hides_mismatched_certification():
    srv = AppServer(Engine())
    srv.cal = {"thresholds": {"0.05": {"per_type": {"choice": {"confidence": 0.9}}}}}

    async def go(c):
        return await c.get("/v1/models")

    response = asyncio.run(on_app(srv, go))
    entry = response.json()["models"][0]
    assert entry["certified"] == {"0.05": {"choice": 0.9}}
    assert entry["context"] == {"limit": 8192, "overflow": "refuse", "truncate_head": 1 / 3}
    srv.cal_mismatch = "wrong model"
    response = asyncio.run(on_app(srv, go))
    assert "certified" not in response.json()["models"][0]


def test_models_endpoint_reports_the_bucket_temperatures():
    srv = AppServer(Engine())
    srv.cal = {"temperature_per_prim": {"choice": 1.0}, "temperature_per_bucket": {"choice": {"2-10": 2.0}}}

    async def go(c):
        return await c.get("/v1/models")

    entry = asyncio.run(on_app(srv, go)).json()["models"][0]
    assert entry["temperature_per_bucket"] == {"choice": {"2-10": 2.0}}


@pytest.mark.parametrize("body,content,message", [
    (None, b"{not json", "not valid JSON"),
    (["a list"], None, "must be a JSON object"),
    ({"state": "s"}, None, "missing field 'questions'"),
    ({"state": "s", "questions": {"q": "Which?"}}, None, "question object"),
    ({"state": "s", "questions": {}}, None, "at least one question"),
    ({"state": "s", "questions": {"q": dict(Q, criteria={"a": "Alpha"})}}, None, "1 options"),
    ({"state": "s", "questions": {"q": dict(Q, option_keys="maybe")}}, None, "option_keys"),
    ({"state": "s", "questions": {"q": dict(Q, type="act", costs={"defer": 1})}}, None, "missing field 'wrong'"),
    ({"state": "s", "questions": {"q": dict(Q, type="act", costs={"wrong": 1, "defer": .1},
                                             max_error="low")}}, None, "max_error must be a number"),
    ({"state": "s", "questions": {"q": dict(Q, type="multi", min="one")}}, None, "min must convert to an integer"),
    ({"state": "s", "questions": {"q": dict(Q, type="multi", max="one")}}, None, "max must convert to an integer"),
    ({"state": "s", "questions": {"q": dict(Q, type="multi", threshold="high")}}, None, "could not convert"),
    ({"state": "s", "questions": {"q": {"type": "tool", "tools": [{"type": "web"}]}}}, None, "function tools only"),
    ({"state": "s", "questions": {"q": dict(Q, type="chioce")}}, None, "unknown type 'chioce'"),
    ({"state": "s", "questions": {"q": Q}, "lang": "de"}, None, "lang must be"),
    (None, b'{"state": "s", "questions": {"q": {"type": "choice", "criteria": [NaN, "b"]}}}', "option key"),
    (None, b'{"state": "s", "questions": {"q": {"type": "multi", "criteria": ["a", 1e999]}}}', "option key"),
    (None, b'{"state": "s", "questions": {"q": {"type": "act", "criteria": [-Infinity, "b"], '
           b'"costs": {"wrong": 1}}}}', "option key"),
])
def test_invalid_requests_answer_422(body, content, message):
    srv = AppServer(Engine())
    r = post(srv, body, content)
    error = r.json()
    assert r.status_code == 422 and message in error["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert error["detail"] == [{"loc": ["body"], "msg": error["error"], "type": "value_error"}]
    assert not srv.backend.entered.is_set()
    assert srv.load() == {"running": 0, "queued": 0}


@pytest.mark.parametrize("error,status,code", [
    (RuntimeError("engine failed"), 500, "INTERNAL_ERROR"),
    (KeyError("layer"), 500, "INTERNAL_ERROR"),  # a bug in the engine is the server's error, not the request's
    (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), 503, "OUT_OF_MEMORY"),
    (RuntimeError("MPS backend out of memory (MPS allocated: 20 GB)"), 503, "OUT_OF_MEMORY"),
    (RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory "
                  "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"), 503, "OUT_OF_MEMORY"),
])
def test_server_errors_answer_500_and_out_of_memory_503(error, status, code):
    srv = AppServer(Engine(error))
    r = post(srv, BODY)
    assert r.status_code == status and r.json() == {"error": str(error), "code": code}
    assert srv.load() == {"running": 0, "queued": 0}


def test_a_valid_request_answers_200_without_error_fields():
    r = post(AppServer(Engine()), BODY)
    assert r.status_code == 200 and r.json()["answers"]["q"]["choice"] in ("a", "b")
    assert "error" not in r.json() and "code" not in r.json()


def test_request_ids_preserve_unprotected_response_bytes():
    async def endpoint(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    async def go():
        app = request_ids(protect(endpoint, None))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
            return await c.get("/")

    decorated = asyncio.run(go())
    assert decorated.content == b'{"ok":true}'
    assert decorated.headers["x-request-id"].startswith("req_")
    supplied = asyncio.run(on_app(AppServer(Engine()), lambda c: c.get("/health", headers={"X-Request-Id": "trace-1"}),
                                  decorate=lambda app: request_ids(protect(app, None))))
    assert supplied.headers["x-request-id"] == "trace-1"


def test_api_keys_protect_api_but_not_health(tmp_path):
    secret = "key-123"
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": "test", "sha256": hashlib.sha256(secret.encode()).hexdigest()}]}))
    decorate = lambda app: request_ids(protect(app, ApiKeys(path)))

    async def go(c):
        return (await c.post("/v1/systemone", json=BODY),
                await c.post("/v1/systemone", json=BODY, headers={"Authorization": f"Bearer {secret}"}),
                await c.post("/v1/systemone", json=BODY, headers={"X-API-Key": secret}),
                await c.get("/health"))

    unauth, bearer, api_key, health = asyncio.run(on_app(AppServer(Engine()), go, decorate))
    assert unauth.status_code == 401
    assert unauth.json() == {"error": "missing or invalid API key", "code": "UNAUTHORIZED"}
    assert bearer.status_code == api_key.status_code == 200
    assert health.status_code == 200
    assert all("x-request-id" in response.headers for response in (unauth, bearer, api_key, health))


def test_cors_preflight_is_limited_to_configured_origins():
    decorate = lambda app: request_ids(protect(app, None, ("https://allowed.example",)))

    async def go(c):
        allowed = await c.options("/v1/systemone", headers={"Origin": "https://allowed.example",
                                                            "Access-Control-Request-Method": "POST"})
        denied = await c.options("/v1/systemone", headers={"Origin": "https://other.example",
                                                           "Access-Control-Request-Method": "POST"})
        return allowed, denied

    allowed, denied = asyncio.run(on_app(AppServer(Engine()), go, decorate))
    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in denied.headers


def test_status_of_errors():
    assert http_status(RequestError("x")) == 422 and http_status(ValueError("x")) == 500
    assert out_of_memory(torch.OutOfMemoryError("x")) and not out_of_memory(RuntimeError("shape mismatch"))


def test_health_counts_running_and_queued_requests():
    """The engine holds the first request (one engine call at a time): it is running and a second one is queued behind
    it; both answered, the server is idle again."""
    engine = Engine()
    engine.go.clear()
    srv = AppServer(engine)

    async def until(cond):
        for _ in range(500):
            if cond():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out")

    async def go(c):
        health = []

        async def load_from_metrics():
            response = await c.get("/metrics")
            assert response.status_code == 200
            values = {line.split(" ", 1)[0]: int(line.split(" ", 1)[1])
                      for line in response.text.splitlines() if line.startswith("basal_requests_")}
            return values["basal_requests_running"], values["basal_requests_waiting"]

        assert (await c.get("/health")).json() == {"status": "ok", "running": 0, "queued": 0}
        assert await load_from_metrics() == (0, 0)
        first = asyncio.get_running_loop().create_task(c.post("/v1/systemone", json=BODY))
        await until(engine.entered.is_set)
        health.append((await c.get("/health")).json())
        assert await load_from_metrics() == (1, 0)
        second = asyncio.get_running_loop().create_task(c.post("/v1/systemone", json=BODY))
        await until(lambda: srv.in_flight == 2)
        health.append((await c.get("/health")).json())
        assert await load_from_metrics() == (1, 1)
        engine.go.set()
        codes = [(await t).status_code for t in (first, second)]
        health.append((await c.get("/health")).json())
        assert await load_from_metrics() == (0, 0)
        return health, codes

    health, codes = asyncio.run(on_app(srv, go))
    assert codes == [200, 200]
    assert health == [{"status": "ok", "running": 1, "queued": 0}, {"status": "ok", "running": 1, "queued": 1},
                      {"status": "ok", "running": 0, "queued": 0}]
