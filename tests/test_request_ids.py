"""Counting and correlation: a body that is not JSON counts as a failed request on /v1/systemone as on /v1/decisions;
basal-router passes the request's X-Request-Id (the client's, or one it makes) to every replica the request reaches
and answers with it."""
import asyncio
import json
import re

import httpx
from starlette.testclient import TestClient
from test_server import FakeServer

from basal import router as R
from basal.server import http_app


class AppServer(FakeServer):
    async def serve(self, body):
        return await self._run(body)


def _counts(c):
    m = c.get("/metrics").text
    return [int(re.search(rf"^{k} (\d+)", m, re.M).group(1)) for k in ("basal_requests_total",
                                                                         "basal_requests_failed_total")]


def test_a_body_that_is_not_json_counts_on_every_route():
    c = TestClient(http_app(AppServer([0.5, 0.5]), "fast"), raise_server_exceptions=False)
    for path, status in (("/v1/systemone", 422), ("/v1/basal", 422), ("/v1/decisions", 400)):
        before = _counts(c)
        r = c.post(path, content=b"{not json", headers={"content-type": "application/json"})
        assert r.status_code == status
        assert _counts(c) == [before[0] + 1, before[1] + 1]


def _router(seen, **kw):
    def handler(req):
        base = req.url.host
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.setdefault(base, []).append(req.headers.get("x-request-id"))
        qs = json.loads(req.content)["questions"]
        p = 0.6 if base == "small" else 0.9
        return httpx.Response(200, json={"answers": {q: {"type": "choice", "choice": "a",
                                                         "probabilities": {"a": p, "b": 1 - p}} for q in qs}})
    return R.Router(["http://small"], transport=httpx.MockTransport(handler), **kw)


def test_router_passes_the_clients_request_id_to_every_replica():
    seen = {}
    rt = _router(seen, escalate=["http://big"], below=0.8)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}}, headers={"X-Request-Id": "trace-17"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "trace-17"
    assert seen == {"small": ["trace-17"], "big": ["trace-17"]}


def test_router_makes_one_request_id_when_the_client_gives_none():
    seen = {}
    rt = _router(seen)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}})
        bad = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}}, headers={"X-Request-Id": "a b"})
    rid = r.headers["x-request-id"]
    assert rid.startswith("req_") and seen["small"][0] == rid
    assert bad.headers["x-request-id"] != "a b" and seen["small"][1] == bad.headers["x-request-id"]
