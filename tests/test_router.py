"""basal-router (2.5): state-hash routing with bounded load over ready replicas; the server's /metrics and its 503
answers during start-up."""
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import httpx
import pytest
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
                      ("b", "help b", "gauge", [({"replica": 'x"y\\z\nw'}, 1.5)])])
    assert "# TYPE a_total counter\na_total 3" in t and 'b{replica="x\\"y\\\\z\\nw"} 1.5' in t


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
    from basal import __version__
    assert f'basal_build_info{{version="{__version__}",model="basal-test",mode="fast",layout="canonical"}} 1' in m
    assert "basal_state_cache" not in m


def test_server_metrics_state_cache_never_waits_for_its_lock():
    """With a state cache: its counters (#48's evictions and prefix hits too); while the engine holds the cache's lock,
    /metrics answers at once with the last snapshot and its time."""
    import torch

    from basal.engine import StateCache
    cache = StateCache(2 ** 20)
    cache.put(cache.key([1, 2, 3]), [(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))], tokens=[1, 2, 3])
    cache.get(cache.key([1, 2, 3])), cache.get(cache.key([4]))

    class Srv:
        name, layout, long_tokens = "basal-test", "canonical", None
        backend = type("B", (), {"state_cache": cache, "policies": {}})()

        def load(self):
            return {"running": 0, "queued": 0}

        async def worker(self):
            await asyncio.sleep(3600)

    def values(text):
        return {ln.split(" ")[0]: float(ln.split(" ")[1]) for ln in text.splitlines() if ln.startswith("basal_state")}
    with TestClient(S.http_app(Srv(), "fast")) as c:
        with cache.lock:  # the engine is storing a state: no snapshot yet, so no cache values, timestamp 0
            first = values(c.get("/metrics").text)
        second = values(c.get("/metrics").text)
        cache.put(cache.key([5]), [(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))])
        with cache.lock:  # busy again: the last snapshot, with its time
            third = values(c.get("/metrics").text)
    assert first == {"basal_state_cache_snapshot_timestamp_seconds": 0.0}
    assert second["basal_state_cache_hits_total"] == 1 and second["basal_state_cache_misses_total"] == 1
    assert second["basal_state_cache_entries"] == 1 and second["basal_state_cache_bytes"] == 192
    assert second["basal_state_cache_evictions_total"] == 0 and second["basal_state_cache_prefix_hits_total"] == 0
    assert second["basal_state_cache_snapshot_timestamp_seconds"] > 0 and third == second


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


def cascade_transport(conf):
    """Fake replicas that answer every question with top probability conf[replica] (and say who answered)."""
    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(req.content)
        c = conf[base](body) if callable(conf[base]) else conf[base]
        ans = {q: {"type": "choice", "choice": "a", "probabilities": {"a": c[q] if isinstance(c, dict) else c,
                                                                       "b": 1 - (c[q] if isinstance(c, dict) else c)},
                   "replica": base} for q in body["questions"]}
        return httpx.Response(200, json={"answers": ans, "replica": base})
    return httpx.MockTransport(handler)


def test_cascade_escalates_only_low_confidence_questions():
    small, big = "http://small", "http://big"
    rt = R.Router([small], transport=cascade_transport({small: {"q1": 0.55, "q2": 0.95}, big: 0.9}),
                  escalate=[big], below=0.8)
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        body = {"state": "Doc", "questions": {"q1": {"type": "choice"}, "q2": {"type": "choice"}}}
        r = c.post("/v1/systemone", json=body).json()
        assert r["answers"]["q1"]["replica"] == big and r["answers"]["q1"]["answered_by"] == "escalated"
        assert r["answers"]["q2"]["replica"] == small and r["answers"]["q2"]["answered_by"] == "first"
        assert r["cascade"] == {"below": 0.8, "escalated": ["q1"]}
        m = c.get("/metrics").text
        assert 'basal_router_answers_total{answered_by="escalated"} 1' in m
        assert 'basal_router_answers_total{answered_by="first"} 1' in m


def test_router_without_cascade_passes_answers_through_untouched():
    """Off by default: the replica's response bytes come back as they are."""
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, content=b'{"answers":{"q":{"probabilities":{"a":0.5,"b":0.5}}},  "x": 1}')
    rt = R.Router(["http://r0"], transport=httpx.MockTransport(handler))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {}}})
        assert r.content == b'{"answers":{"q":{"probabilities":{"a":0.5,"b":0.5}}},  "x": 1}'


def test_route_by_script():
    assert R.script_of("Звіт про інцидент") == "cyrillic" and R.script_of("Raport z incydentu") == "latin"
    assert R.script_of("事件报告") == "cjk" and R.script_of("1234") is None
    main, cyr = "http://main", "http://cyr"
    rt = R.Router([main], transport=cascade_transport({main: 0.9, cyr: 0.9}), routes={"cyrillic": [cyr]})
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        q = {"q": {"type": "choice"}}
        assert c.post("/v1/systemone", json={"state": "Звіт про інцидент", "questions": q}).json()["replica"] == cyr
        assert c.post("/v1/systemone", json={"state": "Raport z incydentu", "questions": q}).json()["replica"] == main
        m = c.get("/metrics").text
        assert 'basal_router_script_total{script="cyrillic"} 1' in m and 'basal_router_script_total{script="latin"} 1' in m
# --gpus replicas: the processes basal-router starts must end with it, whatever ends it (2.5 gate: a replica outlived
# a router stopped by SIGTERM, since uvicorn re-raises the signal after its shutdown, past main's finally)
SLEEPER = ("import os, sys, time\n"
           "port = sys.argv[sys.argv.index('--port') + 1]\n"
           "open(os.path.join(os.environ['PIDDIR'], port), 'w').write(str(os.getpid()))\n"
           "time.sleep(600)\n")
EXITER = "import sys; sys.exit(3)\n"
ROUTER = ("import sys\n"
          "from basal import router as R\n"
          "R.SERVE = [sys.executable, '-c', sys.argv[1]]\n"
          "R.POLL_S = 0.2\n"
          "R.main(sys.argv[2:])\n")
posix = pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups (POSIX)")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie (exited, not reaped by its parent) is not running
        return open(f"/proc/{pid}/stat").read().split(") ")[1][0] != "Z"
    except OSError:
        return True


def _router(tmp_path, replica=SLEEPER, gpus="0,1"):
    """basal-router --gpus in a subprocess whose replicas are `replica` (pid files in tmp_path) -> (process, pids)."""
    port, rport = _free_port(), _free_port()
    env = dict(os.environ, PIDDIR=str(tmp_path), PYTHONPATH=os.pathsep.join(sys.path))
    p = subprocess.Popen([sys.executable, "-c", ROUTER, replica, "--gpus", gpus, "--port", str(port),
                          "--replica-port", str(rport)], env=env)
    n = len(gpus.split(","))
    for _ in range(300):
        pids = [int(f.read_text()) for f in tmp_path.iterdir() if f.read_text()]
        up = False
        try:
            up = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) is not None
        except urllib.error.HTTPError:
            up = True  # 503: no replica answers /health (they only sleep)
        except OSError:
            pass
        if len(pids) == n and up:
            return p, pids
        time.sleep(0.05)
    p.kill()
    raise AssertionError("the router or its replicas did not start")


def _gone(pids, seconds=20):
    for _ in range(int(seconds * 20)):
        if not any(_alive(x) for x in pids):
            return True
        time.sleep(0.05)
    return False


@posix
def test_spawned_replicas_have_their_own_process_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("PIDDIR", str(tmp_path))
    monkeypatch.setattr(R, "SERVE", [sys.executable, "-c", SLEEPER])
    procs, urls = R.spawn(["0", "1"], _free_port(), [])
    reps = R.Replicas(procs)
    try:
        assert len(urls) == 2 and all(os.getpgid(p.pid) == p.pid != os.getpgrp() for p in procs)
    finally:
        reps.stop(timeout=5)
    assert all(p.poll() is not None for p in procs)
    reps.stop()  # once only


@posix
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGINT"])
def test_router_stopped_by_a_signal_stops_its_replicas(tmp_path, sig):
    p, pids = _router(tmp_path)
    p.send_signal(getattr(signal, sig))
    p.wait(timeout=30)
    assert _gone(pids), "a replica outlived the router"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG (Linux)")
def test_replicas_end_when_the_router_is_killed(tmp_path):
    p, pids = _router(tmp_path)
    p.kill()
    p.wait(timeout=30)
    assert _gone(pids), "a replica outlived a killed router"


@posix
def test_router_exits_1_when_every_replica_exited(tmp_path):
    port, rport = _free_port(), _free_port()
    p = subprocess.Popen([sys.executable, "-c", ROUTER, EXITER, "--gpus", "0,1", "--port", str(port),
                          "--replica-port", str(rport)], env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert p.wait(timeout=60) == 1
