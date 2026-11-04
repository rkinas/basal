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
