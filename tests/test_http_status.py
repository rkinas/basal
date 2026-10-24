"""basal-serve's status codes and load (#22): an invalid request answers 422, the GPU out of memory 503 and any other
server error 500, all with {"error": ...}; GET /health counts the requests in flight as running and queued. Through
the app of basal-serve (build_app) with its batching worker and a stub engine."""
import asyncio
import threading

import httpx
import pytest
import torch

from basal.prompt import RequestError
from basal.server import build_app, parser, http_status, out_of_memory
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
        self.queue = asyncio.Queue()


async def on_app(srv, fn):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        transport = httpx.ASGITransport(app=build_app(srv, parser().parse_args(["--mode", "fast"])))
        async with httpx.AsyncClient(transport=transport, base_url="http://basal") as c:
            return await fn(c)
    finally:
        worker.cancel()


def post(srv, body=None, content=None):
    async def go(c):
        return await (c.post("/v1/systemone", content=content) if content is not None
                      else c.post("/v1/systemone", json=body))
    return asyncio.run(on_app(srv, go))


@pytest.mark.parametrize("body,content,message", [
    (None, b"{not json", "malformed JSON"),
    (["a list"], None, "must be a JSON object"),
    ({"state": "s"}, None, "questions must be a non-empty object"),
    ({"state": "s", "questions": {"q": "Which?"}}, None, "must be an object"),
    ({"state": "s", "questions": {"q": dict(Q, criteria={"a": "Alpha"})}}, None, "1 options"),
    ({"state": "s", "questions": {"q": dict(Q, option_keys="maybe")}}, None, "option_keys"),
    ({"state": "s", "questions": {"q": dict(Q, type="act", costs={"defer": 1})}}, None, "missing field 'wrong'"),
    ({"state": "s", "questions": {"q": dict(Q, type="multi", threshold="high")}}, None, "could not convert"),
    ({"state": "s", "questions": {"q": {"type": "tool", "tools": [{"type": "web"}]}}}, None, "function tools only"),
    ({"state": "s", "questions": {"q": Q}, "lang": "de"}, None, "lang must be"),
])
def test_invalid_requests_answer_422(body, content, message):
    srv = AppServer(Engine())
    r = post(srv, body, content)
    assert r.status_code == (400 if content is not None or isinstance(body, list) else 422) and message in r.json()["error"]
    assert srv.load() == {"running": 0, "queued": 0}


@pytest.mark.parametrize("error,status", [
    (RuntimeError("engine failed"), 500),
    (KeyError("layer"), 500),  # a bug in the engine is the server's error, not the request's
    (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), 503),
    (RuntimeError("MPS backend out of memory (MPS allocated: 20 GB)"), 503),
    (RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory "
                  "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"), 503),
])
def test_server_errors_answer_500_and_out_of_memory_503(error, status):
    srv = AppServer(Engine(error))
    r = post(srv, BODY)
    assert r.status_code == status and r.json()["error"] == str(error)
    assert srv.load() == {"running": 0, "queued": 0}


def test_a_valid_request_answers_200():
    r = post(AppServer(Engine()), BODY)
    assert r.status_code == 200 and r.json()["answers"]["q"]["choice"] in ("a", "b")


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
        assert (await c.get("/health")).json() == {"status": "ok", "running": 0, "queued": 0}
        first = asyncio.get_running_loop().create_task(c.post("/v1/systemone", json=BODY))
        await until(engine.entered.is_set)
        health.append((await c.get("/health")).json())
        second = asyncio.get_running_loop().create_task(c.post("/v1/systemone", json=BODY))
        await until(lambda: srv.in_flight == 2)
        health.append((await c.get("/health")).json())
        engine.go.set()
        codes = [(await t).status_code for t in (first, second)]
        health.append((await c.get("/health")).json())
        return health, codes

    health, codes = asyncio.run(on_app(srv, go))
    assert codes == [200, 200]
    assert health == [{"status": "ok", "running": 1, "queued": 0}, {"status": "ok", "running": 1, "queued": 1},
                      {"status": "ok", "running": 0, "queued": 0}]


@pytest.mark.parametrize("question", [
    dict(Q, type="act", costs={"wrong": 1, "defer": .1}, max_error="bad"),
    *[{"type": "multi", "criteria": ["a", "b"], key: "bad"} for key in ("min", "max")],
    *[{"type": "tool", "multi": True, "tools": [{"type": "function", "function": {"name": "lookup"}}],
       key: "bad"} for key in ("min", "max")],
])
def test_invalid_assembly_options_are_rejected_before_inference(question):
    engine = Engine()
    srv = AppServer(engine)
    response = post(srv, {"state": "s", "questions": {"q": question}})
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "INVALID_REQUEST"
    assert not engine.entered.is_set()
    assert srv.load() == {"running": 0, "queued": 0}


@pytest.mark.parametrize("question", [
    dict(Q, type="act", costs={"wrong": 1, "defer": .1}, max_error="0.05"),
    {"type": "multi", "criteria": ["a", "b"], "min": "1", "max": "2"},
    {"type": "tool", "multi": True, "tools": [{"type": "function", "function": {"name": "lookup"}}],
     "min": "1", "max": "2"},
])
def test_numeric_assembly_options_keep_legacy_coercion(question):
    response = post(AppServer(Engine()), {"state": "s", "questions": {"q": question}})
    assert response.status_code == 200, response.text
