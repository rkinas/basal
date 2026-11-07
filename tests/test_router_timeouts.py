"""basal-router --replica-timeout: a replica that accepts a request and never answers costs the client at most the
timeout (504), its in-flight count recovers, the other replicas keep serving, and a stalled escalation level leaves
the previous answers. Off by default (no deadline, as before); connecting to a replica is bounded either way."""
import asyncio
import time

import httpx
import pytest
from starlette.testclient import TestClient

from basal import router as R

GOOD, STUCK = "http://good", "http://stuck"


def transport(conf):
    async def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if base == STUCK:
            await asyncio.sleep(60)                                       # connected, never answers
        qs = __import__("json").loads(req.content)["questions"]
        p = conf.get(base, 0.9)
        ans = {q: {"type": "choice", "choice": "a", "probabilities": {"a": p, "b": 1 - p}} for q in qs}
        return httpx.Response(200, json={"answers": ans, "model": base})
    return httpx.MockTransport(handler)


def test_a_stalled_replica_answers_504_within_the_timeout():
    rt = R.Router([STUCK], transport=transport({}), timeout=0.3)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        t = time.monotonic()
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}})
        assert r.status_code == 504 and "timed out" in r.json()["error"] and time.monotonic() - t < 5
        assert rt.load[STUCK] == 0 and rt.failed[STUCK] == 1


def test_a_stalled_escalation_level_keeps_the_first_answers():
    rt = R.Router([GOOD], transport=transport({GOOD: 0.6}), escalate=[STUCK], below=0.8, timeout=0.3)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}})
        assert r.status_code == 200 and r.json()["answers"]["q"]["answered_by"] == "first"
        assert r.json()["cascade"]["escalated"] == []
        assert rt.load == {GOOD: 0, STUCK: 0}


def test_one_slow_request_leaves_its_replica_in_rotation():
    """A call past the deadline was slow, not a replica that is down: the other clients' requests still go to it
    (the router's own /health stays 200). A replica that cannot be reached at all leaves the rotation as before."""
    async def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = __import__("json").loads(req.content)
        if body["state"] == "slow":
            await asyncio.sleep(60)
        ans = {q: {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}} for q in body["questions"]}
        return httpx.Response(200, json={"answers": ans, "model": GOOD})
    rt = R.Router([GOOD], transport=httpx.MockTransport(handler), timeout=0.3)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        assert c.post("/v1/systemone", json={"state": "slow", "questions": {"q": {}}}).status_code == 504
        assert c.post("/v1/systemone", json={"state": "fast", "questions": {"q": {}}}).status_code == 200
        assert GOOD in rt.ready and c.get("/health").status_code == 200

    async def unreachable(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ConnectTimeout("connect timed out", request=req)
    rt = R.Router([GOOD], transport=httpx.MockTransport(unreachable), timeout=0.3)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}})
        assert r.status_code == 504 and "connecting" in r.json()["error"] and GOOD not in rt.ready


def test_without_a_timeout_requests_have_no_deadline_but_connecting_has_one():
    rt = R.Router([GOOD], transport=transport({}))
    t = rt.client.timeout
    assert t.read is None and t.write is None and t.pool is None and t.connect == R.CONNECT_S
    rt = R.Router([GOOD], transport=transport({}), timeout=5)
    assert rt.client.timeout.read == 5 and rt.client.timeout.connect == 5


def test_replica_timeout_from_the_command_line(monkeypatch):
    got = []
    monkeypatch.setattr(R, "Router", lambda urls, **kw: got.append(kw["timeout"]) or type("F", (), {"dead": False})())
    monkeypatch.setattr(R, "app", lambda *a, **k: None)
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    R.main(["--replicas", GOOD])
    R.main(["--replicas", GOOD, "--replica-timeout", "30"])
    assert got == [None, 30.0]
    for bad in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit, match="--replica-timeout"):
            R.main(["--replicas", GOOD, "--replica-timeout", bad])
