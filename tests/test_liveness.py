"""Engine liveness: with --stall-s, GET /health answers 503 "stalled" while an engine call has run longer than S
seconds (a hung GPU call), so basal-router stops sending the replica requests; /metrics splits a request's time into
waiting for the engine and engine calls."""
import asyncio

import httpx
from starlette.testclient import TestClient
from test_http_status import BODY, AppServer, Engine

from basal import router as R
from basal.server import http_app, parser


async def _with_app(srv, fn):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        transport = httpx.ASGITransport(app=http_app(srv, "fast"))
        async with httpx.AsyncClient(transport=transport, base_url="http://basal", timeout=30) as c:
            return await fn(c)
    finally:
        worker.cancel()


def held_health(stall_s):
    eng = Engine()
    srv = AppServer(eng)
    srv.stall_s = stall_s

    async def go(c):
        eng.go.clear()
        req = asyncio.create_task(c.post("/v1/systemone", json=BODY))
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
        during = await c.get("/health")
        eng.go.set()
        await req
        return during, await c.get("/health")
    return asyncio.run(_with_app(srv, go))


def test_health_says_stalled_while_an_engine_call_runs_past_stall_s():
    assert parser().parse_args([]).stall_s is None and parser().parse_args(["--stall-s", "60"]).stall_s == 60
    during, after = held_health(0.2)
    assert during.status_code == 503 and during.json()["status"] == "stalled" and during.json()["engine_call_s"] > 0.2
    assert after.status_code == 200 and after.json() == {"status": "ok", "running": 0, "queued": 0}


def test_without_stall_s_health_stays_as_it_was():
    during, after = held_health(None)
    assert during.status_code == 200 and during.json()["running"] == 1 and after.status_code == 200


def test_metrics_split_queue_wait_from_engine_time():
    srv = AppServer(Engine())

    async def go(c):
        await c.post("/v1/systemone", json=BODY)
        return (await c.get("/metrics")).text
    m = asyncio.run(_with_app(srv, go))
    values = {line.split(" ")[0]: float(line.split(" ")[1]) for line in m.splitlines() if line and line[0] != "#"}
    assert values["basal_engine_calls_total"] == 1 and values["basal_engine_seconds_total"] > 0
    assert values["basal_queue_wait_seconds_total"] >= 0 and values["basal_engine_call_running_seconds"] == 0
    assert values["basal_queue_oldest_seconds"] == 0


def test_a_router_sends_nothing_to_a_stalled_replica():
    seen = []

    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(503, json={"status": "stalled"}) if base == "http://stuck" else \
                httpx.Response(200, json={"status": "ok"})
        seen.append(base)
        return httpx.Response(200, json={"answers": {}})
    rt = R.Router(["http://stuck", "http://ok"], transport=httpx.MockTransport(handler))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        for i in range(20):
            c.post("/v1/systemone", json=dict(BODY, state=f"document {i}"))
    assert seen == ["http://ok"] * 20


def test_metrics_count_the_errors_by_code():
    eng = Engine()
    srv = AppServer(eng)
    srv.max_requests = 1

    async def go(c):
        await c.post("/v1/systemone", json={"state": "x"})                                   # INVALID_REQUEST
        await c.post("/v1/decisions", content=b"{not json")                                 # INVALID_REQUEST
        eng.go.clear()
        held = asyncio.create_task(c.post("/v1/systemone", json=BODY))
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        await c.post("/v1/systemone", json=BODY)                                             # OVERLOADED
        eng.go.set()
        await held
        return (await c.get("/metrics")).text
    m = asyncio.run(_with_app(srv, go))
    assert 'basal_request_errors_total{code="INVALID_REQUEST"} 2' in m
    assert 'basal_request_errors_total{code="OVERLOADED"} 1' in m and 'basal_request_errors_total{code="TIMEOUT"} 0' in m
    assert "basal_requests_failed_total 3" in m
