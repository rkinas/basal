"""basal-router (2.5): state-hash routing with bounded load over ready replicas; the server's /metrics and its 503
answers during start-up."""
import asyncio
import json
import socket
import time
import urllib.error
import urllib.request

import httpx
from starlette.testclient import TestClient

from basal import router as R
from basal import server as S


def test_state_key_normalises_whitespace():
    assert R.state_key(json.dumps({"state": "  A  b\n c "})) == R.state_key(json.dumps({"state": "A b c"})) == "A b c"
    assert R.state_key(json.dumps({"questions": {}})) is None and R.state_key(b"not json") is None


def test_ring_is_consistent_balanced_and_bounded():
    names = [f"http://r{i}" for i in range(4)]
    ring = R.Ring(names)
    ready = set(names)
    picks = [ring.pick(f"doc {i}", {}, ready) for i in range(4000)]
    assert picks == [ring.pick(f"doc {i}", {}, ready) for i in range(4000)]          # the same state, the same replica
    share = [picks.count(n) / len(picks) for n in names]
    assert min(share) > 0.15 and max(share) < 0.35                                      # spread over the replicas
    home = ring.pick("doc 1", {}, ready)
    busy = {n: 0 for n in names}
    busy[home] = 10                                                                     # far above the mean
    assert ring.pick("doc 1", busy, ready) != home                                      # bounded load: passed on
    assert ring.pick("doc 1", {}, ready - {home}) != home                               # not ready: skipped
    assert ring.pick("doc 1", {}, set()) is None


def replicas_transport(names, ready):
    """Fake replicas: /health 200 when ready, POSTs answer with the replica's name."""
    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200 if base in ready else 503, json={"status": "ok"})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": base}]})
        return httpx.Response(200, json={"replica": base, "body": json.loads(req.content)})
    return httpx.MockTransport(handler)


def test_router_routes_by_state_and_reports():
    names = [f"http://r{i}" for i in range(3)]
    rt = R.Router(names, transport=replicas_transport(names, set(names[:2])))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        assert rt.ready == set(names[:2])                                               # r2 answers 503: not ready
        got = {c.post("/v1/systemone", json={"state": "Doc A", "questions": {}}).json()["replica"] for _ in range(5)}
        assert len(got) == 1 and got <= set(names[:2])                                  # one state, one replica
        assert c.post("/v1/basal", json={"questions": {}}).status_code == 200           # no state: by load
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["ready"] == 2
        m = c.get("/metrics").text
        assert 'basal_router_routed_total{replica="' in m and 'by="state"} 5' in m and "basal_router_replica_ready" in m
        assert c.get("/v1/models").status_code == 200


def test_router_without_ready_replicas_answers_503():
    rt = R.Router(["http://r0"], transport=replicas_transport(["http://r0"], set()))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        assert c.get("/health").status_code == 503
        assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 503


def test_router_stops_when_every_started_replica_exited():
    class Gone:
        def poll(self):
            return 1
    called = []
    rt = R.Router(["http://r0"], transport=replicas_transport(["http://r0"], set()))
    with TestClient(R.app(rt, poll_s=0.05, procs=[Gone()], on_dead=lambda: called.append(1))):
        for _ in range(100):
            if called:
                break
            time.sleep(0.02)
    assert called == [1] and rt.dead


def test_prometheus_text():
    t = S.prometheus([("a_total", "help a", "counter", [({}, 3)]),
                      ("b", "help b", "gauge", [({"replica": 'x"y'}, 1.5)])])
    assert "# TYPE a_total counter\na_total 3" in t and 'b{replica="x\'y"} 1.5' in t


def test_server_metrics_route():
    class Srv:
        name, layout, long_tokens = "basal-test", "canonical", None
        backend = type("B", (), {"state_cache": None, "policies": {}})()

        def load(self):
            return {"running": 1, "queued": 2}

        async def worker(self):
            await asyncio.sleep(3600)
    with TestClient(S.http_app(Srv(), "fast")) as c:
        m = c.get("/metrics").text
    assert "basal_requests_waiting 2" in m and "vllm:num_requests_running 1" in m and 'mode="fast"' in m


def test_starting_answers_503_then_frees_the_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    st = S.Starting("127.0.0.1", port)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
        raise AssertionError("expected 503")
    except urllib.error.HTTPError as e:
        assert e.code == 503 and json.loads(e.read()) == {"status": "starting"}
    st.stop()
    with socket.socket() as s:                                                          # the port is free again
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
