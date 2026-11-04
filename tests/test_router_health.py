"""basal-router readiness: /health is 200 only while the pool of ordinary requests (--replicas / --gpus) has a ready
replica, and /v1/models describes that pool's model; the readiness of every pool is in the /health body."""
import asyncio

import httpx
from starlette.testclient import TestClient

from basal import router as R

MAIN, ESC, BLEND = "http://main", "http://esc", "http://blend"


def transport(ready):
    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200 if base in ready else 503, json={"status": "ok"})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": base}]})
        return httpx.Response(200, json={"answers": {}, "model": base})
    return httpx.MockTransport(handler)


def _get(rt):
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        return c.get("/health"), c.get("/v1/models"), c.post("/v1/systemone", json={"state": "s", "questions": {}})


def test_only_an_escalation_replica_ready_is_not_ready():
    rt = R.Router([MAIN], transport=transport({ESC}), escalate=[ESC], below=0.8)
    h, m, r = _get(rt)
    assert r.status_code == 503                                          # an ordinary request cannot be served
    assert h.status_code == 503 and h.json()["status"] == "starting"
    assert h.json()["pools"] == {"main": 0, "level:2": 1}
    assert m.status_code == 503                                          # not the escalation model's metadata


def test_only_a_blend_replica_ready_is_not_ready():
    rt = R.Router([MAIN], transport=transport({BLEND}), blend=[([BLEND], 1.0)])
    h, m, _ = _get(rt)
    assert h.status_code == 503 and m.status_code == 503


def test_main_pool_ready_is_ready_and_models_comes_from_it():
    rt = R.Router([MAIN], transport=transport({MAIN, ESC}), escalate=[ESC], below=0.8)
    h, m, _ = _get(rt)
    assert h.status_code == 200 and h.json()["pools"] == {"main": 1, "level:2": 1} and h.json()["ready"] == 2
    assert m.json() == {"models": [{"name": MAIN}]}                     # ("http://esc" sorts first)


def test_escalation_down_keeps_the_router_ready():
    rt = R.Router([MAIN], transport=transport({MAIN}), escalate=[ESC], below=0.8)
    h, _, _ = _get(rt)
    assert h.status_code == 200 and h.json()["pools"] == {"main": 1, "level:2": 0}
