"""basal-serve under load: --max-requests in flight (one more answers 503 OVERLOADED with Retry-After before any work),
--request-timeout (504 TIMEOUT, its queued work dropped), --max-prompt-mb (422 before a prompt is rendered), and the
prompts of a long request rendered and tokenized off the event loop, which answers /health meanwhile. Through
basal-serve's app with a stub engine."""
import asyncio
import threading
import time

import httpx
from test_http_status import BODY, AppServer, Engine

from basal import client
from basal import server as S
from basal.server import http_app, parser


class Counting(Engine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run_shared(self, groups, policy=None):
        self.calls += 1
        return super().run_shared(groups, policy)


async def _with_app(srv, fn):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        transport = httpx.ASGITransport(app=http_app(srv, "fast"))
        async with httpx.AsyncClient(transport=transport, base_url="http://basal", timeout=30) as c:
            return await fn(c)
    finally:
        worker.cancel()


def test_flags_and_defaults():
    a = parser().parse_args([])
    assert a.max_requests == S.MAX_REQUESTS == 1024 and a.max_prompt_mb == S.MAX_PROMPT_MB == 512
    assert a.request_timeout is None


def test_a_request_beyond_max_requests_answers_503_before_any_work():
    eng = Counting()
    srv = AppServer(eng)
    srv.max_requests = 1

    async def go(c):
        eng.go.clear()  # the first request holds the engine
        first = asyncio.create_task(c.post("/v1/systemone", json=BODY))
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        r = await c.post("/v1/systemone", json=BODY)
        d = await c.post("/v1/decisions", json={"input": "x", "questions": [{"type": "predicate", "instructions": "Q?"}]})
        eng.go.set()
        r1 = await first
        r3 = await c.post("/v1/systemone", json=BODY)
        return r, d, r1, r3
    r, d, r1, r3 = asyncio.run(_with_app(srv, go))
    assert r.status_code == 503 and r.json()["code"] == "OVERLOADED" and r.headers["retry-after"] == "1"
    assert "--max-requests 1" in r.json()["error"]
    assert d.status_code == 503 and d.headers["retry-after"] == "1"
    assert r1.status_code == 200 and r3.status_code == 200 and eng.calls == 2 and srv.in_flight == 0
    err = client._error(r)
    assert isinstance(err, client.NotReady) and isinstance(err, client.OutOfMemory)  # (retried by the client)


def test_a_request_past_its_timeout_answers_504_and_its_queued_work_is_dropped():
    eng = Counting()
    srv = AppServer(eng)
    srv.request_timeout, srv.max_batch = 0.3, 1

    async def go(c):
        eng.go.clear()
        held = asyncio.create_task(c.post("/v1/systemone", json=BODY))  # in the engine
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        r = await c.post("/v1/systemone", json=dict(BODY, state="queued behind it"))  # waits in the queue
        eng.go.set()
        await held
        for _ in range(30):
            await asyncio.sleep(0.01)
        return r
    t = time.perf_counter()
    r = asyncio.run(_with_app(srv, go))
    assert r.status_code == 504 and r.json()["code"] == "TIMEOUT" and time.perf_counter() - t < 5
    assert eng.calls == 1 and srv.queued == 0 and srv.in_flight == 0  # (the timed-out request never ran)


def test_prompts_over_max_prompt_mb_are_refused_before_rendering(monkeypatch):
    rendered = []
    real = S.render
    monkeypatch.setattr(S, "render", lambda *a, **k: rendered.append(1) or real(*a, **k))
    eng = Counting()
    srv = AppServer(eng)  # 2 option orders
    srv.max_prompt = 1000
    body = {"state": "x" * 600, "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}}
    r = asyncio.run(_with_app(srv, lambda c: c.post("/v1/systemone", json=body)))
    assert r.status_code == 422 and "--max-prompt-mb" in r.json()["error"] and rendered == [] and eng.calls == 0
    body["state"] = "x" * 400
    r = asyncio.run(_with_app(srv, lambda c: c.post("/v1/systemone", json=body)))
    assert r.status_code == 200 and rendered and eng.calls == 1


def test_health_answers_while_a_long_request_is_prepared(monkeypatch):
    monkeypatch.setattr(S, "PREP_INLINE", 0)  # (every request is long; by default prompts over 1 MiB of text)
    release = threading.Event()
    real = AppServer.encode

    def slow_encode(self, branches):  # tokenizing a long request
        release.wait(10)
        return real(self, branches)
    monkeypatch.setattr(AppServer, "encode", slow_encode)
    srv = AppServer(Engine())

    async def go(c):
        timer = threading.Timer(3.0, release.set)  # (on a blocked loop nothing else could release it)
        timer.start()
        t = time.perf_counter()
        req = asyncio.create_task(c.post("/v1/systemone", json=BODY))
        await asyncio.sleep(0.2)
        h = await c.get("/health")
        waited = time.perf_counter() - t  # (from the request's start: a loop blocked in encode answers after 3 s)
        release.set()
        timer.cancel()
        return h, waited, await req
    h, waited, r = asyncio.run(_with_app(srv, go))
    assert h.status_code == 200 and h.json()["running"] == 1 and waited < 1.0 and r.status_code == 200


def test_a_server_decides_in_process_under_several_event_loops():
    srv = AppServer(Engine())

    async def once():
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await asyncio.wait_for(srv.serve(dict(BODY, facts="auto")), 10)
        finally:
            worker.cancel()
    for _ in range(3):  # (each asyncio.run a new loop; the queue moves to it)
        assert asyncio.run(once())["answers"]["q"]["choice"] in ("a", "b")


def test_a_short_request_is_prepared_on_the_loop_as_before():
    srv = AppServer(Engine())
    r = asyncio.run(_with_app(srv, lambda c: c.post("/v1/systemone", json=BODY)))
    assert r.status_code == 200 and getattr(srv, "_prep_pool", None) is None


def test_answers_are_the_same_prepared_on_the_loop_or_off_it(monkeypatch):
    bodies = [BODY, {"state": "Faktura 1240, 1230,00 zł.", "questions": {
        "m": {"type": "multi", "criteria": ["a", "b", "c"]}, "n": {"type": "noul", "instructions": "Paid?"},
        "s": {"type": "score", "criteria": ["lo", "mid", "hi"]}}}]

    def answers():
        srv = AppServer(Engine())
        return [asyncio.run(_with_app(srv, lambda c, b=b: c.post("/v1/systemone", json=b))).json()["answers"]
                for b in bodies]
    inline = answers()
    monkeypatch.setattr(S, "PREP_INLINE", 0)
    assert answers() == inline


def test_a_request_beyond_max_requests_is_refused_before_its_body_is_read():
    eng = Counting()
    srv = AppServer(eng)
    srv.max_requests = 1
    sent = []

    async def chunks():
        for part in (b'{"state": "x", ', b'"questions": {"q": {"type": "noul"}}}'):
            sent.append(part)
            yield part

    async def go(c):
        eng.go.clear()
        first = asyncio.create_task(c.post("/v1/systemone", json=BODY))
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        r = await c.post("/v1/systemone", content=chunks(), headers={"content-type": "application/json"})
        eng.go.set()
        await first
        return r
    r = asyncio.run(_with_app(srv, go))
    assert r.status_code == 503 and r.json()["code"] == "OVERLOADED" and sent == [] and srv.reading == 0
