"""basal-router (2.5): state-hash routing with bounded load over ready replicas; the server's /metrics and its 503
answers during start-up."""
import asyncio
import hashlib
import json
import math
import multiprocessing
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


FIRST_METRICS = """
import asyncio, json, sys, time
import httpx
from starlette.testclient import TestClient
from basal import router as R
names = ["http://r0", "http://r1"]
rt = R.Router(names, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"})))
with TestClient(R.app(rt, poll_s=3600)) as c:
    asyncio.run(rt.poll())
    before = "torch" in sys.modules
    t = time.perf_counter()
    status = c.get("/metrics").status_code
    print(json.dumps(dict(status=status, seconds=time.perf_counter() - t, torch_before=before,
                          torch_after="torch" in sys.modules)))
"""


def test_first_router_metrics_imports_no_torch():
    """RMET: basal-router's first /metrics after a start formats the text without importing basal.server (torch,
    transformers: 6-9 s with the router's event loop blocked); a fresh interpreter, since this test process has torch."""
    out = subprocess.run([sys.executable, "-c", FIRST_METRICS], capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["status"] == 200 and not r["torch_before"] and not r["torch_after"]
    assert r["seconds"] < 1.0


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


def _raw_request(port, request):
    """The bytes the server sends back for a raw request over a plain socket (no client library hides the body)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
        c.sendall(request)
        data = b""
        while chunk := c.recv(4096):
            data += chunk
    return data.partition(b"\r\n\r\n")


@pytest.mark.parametrize("method", ["GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH", "TRACE", "CONNECT",
                                    "PROPFIND"])
def test_starting_answers_every_http_method_with_503_and_retry_after(method):
    """A router or a load balancer may probe with any method: PUT, DELETE, OPTIONS and PATCH used to get the standard
    library's 501 HTML page without Retry-After, the others the 503; TRACE, CONNECT and made-up methods still did.
    HEAD sends the status and the headers only, never the body (checked on the raw socket)."""
    st = S.Starting("127.0.0.1", 0)  # port 0: the system picks a free one, no race with another process
    try:
        port = st.httpd.server_address[1]
        head, _, body = _raw_request(port, f"{method} /health HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode())
        lines = head.decode().split("\r\n")
        headers = {k.lower(): v for k, v in (ln.split(": ", 1) for ln in lines[1:])}
        assert lines[0].split()[1] == "503" and headers["retry-after"] == "5"
        assert headers["content-type"] == "application/json"
        if method == "HEAD":
            assert body == b""
        else:
            assert json.loads(body) == {"status": "starting"}
    finally:
        st.stop()


def test_starting_needs_no_fork_hook_where_the_platform_has_none(monkeypatch):
    """Native Windows has no os.register_at_fork: basal-serve died with AttributeError before loading the model."""
    monkeypatch.delattr(os, "register_at_fork", raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    S.Starting("127.0.0.1", port).stop()


def _sleep(seconds):
    time.sleep(seconds)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no fork on this platform")
def test_starting_frees_the_port_of_a_process_forked_during_start_up():
    """vLLM forks its engine core while the 503 server holds the port: the child must not keep the port taken after
    stop(), or the real server cannot bind it (basal-serve --mode vllm exited at the end of its start-up)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    st = S.Starting("127.0.0.1", port)
    child = multiprocessing.get_context("fork").Process(target=_sleep, args=(5,))
    child.start()
    try:
        st.stop()
        with socket.socket() as s:                                                      # free while the child lives
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen()
    finally:
        child.terminate()
        child.join()



def test_starting_port_is_not_inherited_by_a_spawned_process():
    """Processes started with exec while the 503 server holds the port (basal-router's replicas, Triton / Inductor
    compile workers, any subprocess) never get its socket: Python creates it non-inheritable (close-on-exec)."""
    import subprocess
    import sys
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    st = S.Starting("127.0.0.1", port)
    try:
        fd = st.httpd.socket.fileno()
        assert not st.httpd.socket.get_inheritable()
        code = ("import os, stat, sys\ntry:\n"
                f"    sys.exit(0 if stat.S_ISSOCK(os.fstat({fd}).st_mode) else 3)\nexcept OSError:\n    sys.exit(3)")
        assert subprocess.run([sys.executable, "-c", code]).returncode == 3  # no socket at that fd in the child
    finally:
        st.stop()

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


@pytest.mark.parametrize("args,host", [
    (["--model", "m"], "127.0.0.1"),  # basal-serve's default 0.0.0.0 would expose the replicas past the router
    (["--model", "m", "--host", "0.0.0.0"], "0.0.0.0"),  # the user's --host after --
    (["--host=10.0.0.5", "--model", "m"], "10.0.0.5"),
])
def test_spawned_replicas_listen_on_localhost_unless_given_a_host(monkeypatch, args, host):
    argvs = []

    class Popen:
        def __init__(self, argv, **_):
            argvs.append(argv)

    monkeypatch.setattr(R.subprocess, "Popen", Popen)
    _, urls = R.spawn(["0", "1"], 9100, args)
    assert urls == ["http://127.0.0.1:9100", "http://127.0.0.1:9101"]
    for argv in argvs:
        given = [a.split("=", 1)[1] if a.startswith("--host=") else argv[i + 1]
                 for i, a in enumerate(argv) if a == "--host" or a.startswith("--host=")]
        assert given == [host] and argv[-len(args):] == args


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


def test_router_exits_1_when_every_replica_exited(tmp_path):
    port, rport = _free_port(), _free_port()
    p = subprocess.Popen([sys.executable, "-c", ROUTER, EXITER, "--gpus", "0,1", "--port", str(port),
                          "--replica-port", str(rport)], env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert p.wait(timeout=60) == 1


# Multi-level cascade (2.5): --escalate URL ... --escalate-below P repeated as ordered levels; the first replica is
# level 1, the k-th --escalate level k + 1
SMALL, MID, BIG = "http://small", "http://mid", "http://big"


def level_transport(conf, asked, health=None, post=None):
    """Fake replicas: conf[replica][question] = the top probability of a choice answer (or a whole answer dict); each
    response names its model ("model-" + host); asked[replica] records the questions each was asked. health / post:
    status codes per replica (default 200), or an exception to raise for a POST."""
    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response((health or {}).get(base, 200), json={"status": "ok"})
        st = (post or {}).get(base, 200)
        if isinstance(st, Exception):
            raise st
        if st != 200:
            return httpx.Response(st, json={"status": "starting"})
        body = json.loads(req.content)
        asked.setdefault(base, []).append(sorted(body["questions"]))
        ans = {}
        for q in body["questions"]:
            c = conf[base][q]
            ans[q] = dict(c) if isinstance(c, dict) else {"type": "choice", "choice": "a",
                                                          "probabilities": {"a": c, "b": round(1 - c, 6)}}
        return httpx.Response(200, json={"model": "model-" + req.url.host, "answers": ans, "usage": {"questions": len(ans)}})
    return httpx.MockTransport(handler)


def three_levels(conf, asked, below=(0.8, 0.9), **kw):
    return R.Router([SMALL], transport=level_transport(conf, asked, **kw), levels=[([MID], below[0]), ([BIG], below[1])])


QS = {q: {"type": "choice"} for q in ("q1", "q2", "q3", "q4", "q5")}
CONF = {SMALL: {"q1": 0.95, "q2": 0.5, "q3": 0.5, "q4": 0.8, "q5": 0.79},  # q4: at level 1's threshold, not under it
        MID: {"q2": 0.95, "q3": 0.85, "q5": 0.9},                          # q5: at level 2's threshold
        BIG: {"q3": 0.7}}


def test_cascade_levels_follow_each_levels_threshold():
    asked = {}
    rt = three_levels(CONF, asked)
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "Doc", "questions": QS}).json()
    a = r["answers"]
    assert asked == {SMALL: [sorted(QS)], MID: [["q2", "q3", "q5"]], BIG: [["q3"]]}  # only questions under the bar move
    assert {q: a[q]["level"] for q in QS} == {"q1": 1, "q2": 2, "q3": 3, "q4": 1, "q5": 2}
    assert {q: a[q]["model"] for q in QS} == {"q1": "model-small", "q2": "model-mid", "q3": "model-big",
                                              "q4": "model-small", "q5": "model-mid"}
    assert {q: a[q]["answered_by"] for q in QS} == {"q1": "first", "q2": "escalated", "q3": "escalated",
                                                    "q4": "first", "q5": "escalated"}
    assert a["q3"]["probabilities"] == {"a": 0.7, "b": 0.3}  # the last level's answer stands, whatever its probability
    assert r["model"] == "model-small" and r["cascade"] == {"below": [0.8, 0.9], "escalated": ["q2", "q3", "q5"]}


def test_cascade_metrics_count_answers_per_level():
    rt = three_levels(CONF, {})
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        for _ in range(2):
            assert c.post("/v1/basal", json={"state": "Doc", "questions": QS}).status_code == 200
        m = c.get("/metrics").text
    for lev, n in ((1, 4), (2, 4), (3, 2)):
        assert f'basal_router_level_answers_total{{level="{lev}"}} {n}' in m
    assert 'basal_router_answers_total{answered_by="first"} 4' in m
    assert 'basal_router_answers_total{answered_by="escalated"} 6' in m


@pytest.mark.parametrize("down", ["not ready", "503", "unreachable"])
@pytest.mark.parametrize("level", [2, 3])
def test_cascade_level_down_keeps_the_previous_answers(down, level):
    url = MID if level == 2 else BIG
    kw = {"not ready": {"health": {url: 503}}, "503": {"post": {url: 503}},
          "unreachable": {"post": {url: httpx.ConnectError("refused")}}}[down]
    asked = {}
    rt = three_levels(CONF, asked, **kw)
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        resp = c.post("/v1/systemone", json={"state": "Doc", "questions": QS})
    assert resp.status_code == 200
    a = resp.json()["answers"]
    want = {"q1": 1, "q2": 2, "q3": 2, "q4": 1, "q5": 2} if level == 3 else {q: 1 for q in QS}
    assert {q: a[q]["level"] for q in QS} == want
    assert all(a[q]["probabilities"]["a"] == CONF[SMALL if want[q] == 1 else MID][q] for q in QS)
    assert BIG not in asked


def test_cascade_never_escalates_multi_or_tasks():
    asked = {}
    conf = {SMALL: {"m": {"type": "multi", "selected": ["x"], "probabilities": {"x": 0.51, "y": 0.2}, "threshold": 0.5},
                    "c": {"type": "classify", "label": "x", "top_label": "x", "scores": {"x": 0.4, "y": 0.35},
                          "confidence": 0.4},
                    "v": {"type": "verify", "claims": {"0": {"verdict": "true", "probabilities": {"true": 0.5}}},
                          "all_supported": True, "not_supported": []},
                    "q": 0.4},
            MID: {"q": 0.5}, BIG: {"q": 0.6}}
    rt = three_levels(conf, asked, below=(0.99, 0.99))
    qs = {"m": {"type": "multi"}, "c": {"type": "classify"}, "v": {"type": "verify"}, "q": {"type": "choice"}}
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        a = c.post("/v1/systemone", json={"state": "Doc", "questions": qs}).json()["answers"]
    assert asked == {SMALL: [sorted(qs)], MID: [["q"]], BIG: [["q"]]}
    assert {q: a[q]["level"] for q in qs} == {"m": 1, "c": 1, "v": 1, "q": 3}
    assert a["m"]["probabilities"] == conf[SMALL]["m"]["probabilities"] and a["c"]["scores"] == conf[SMALL]["c"]["scores"]


def test_one_level_given_as_levels_answers_as_before():
    """The one-level form gives the same bytes through levels= as through escalate= / below= (no level or model)."""
    def run(**kw):
        rt = R.Router([SMALL], transport=level_transport(CONF, {}), **kw)
        with TestClient(R.app(rt, poll_s=0.05)) as c:
            asyncio.run(rt.poll())
            r = c.post("/v1/systemone", json={"state": "Doc", "questions": QS}).content
            return r, c.get("/metrics").text
    old, new = run(escalate=[MID], below=0.8), run(levels=[([MID], 0.8)])
    assert old == new and b'"level"' not in old[0] and "level_answers" not in old[1]
    assert json.loads(old[0])["cascade"] == {"below": 0.8, "escalated": ["q2", "q3", "q5"]}


def test_levels_from_the_command_line(monkeypatch):
    got = []
    monkeypatch.setattr(R, "Router", lambda urls, **kw: got.append(kw["levels"]) or type("F", (), {"dead": False})())
    monkeypatch.setattr(R, "app", lambda *a, **k: None)
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    base = ["--replicas", SMALL]
    R.main(base)
    R.main(base + ["--escalate-below", "0.7"])
    R.main(base + ["--escalate", MID])
    R.main(base + ["--escalate", MID, BIG, "--escalate-below", "0.7"])
    R.main(base + ["--escalate", MID, "--escalate-below", "0.7", "--escalate", BIG, "--escalate-below", "0.9"])
    R.main(base + ["--escalate", MID, "--escalate", BIG])
    assert got == [[], [], [([MID], 0.8)], [([MID, BIG], 0.7)], [([MID], 0.7), ([BIG], 0.9)],
                   [([MID], 0.8), ([BIG], 0.8)]]
    for bad in (["--escalate", MID, "--escalate", BIG, "--escalate-below", "0.7"],
                ["--escalate", MID, "--escalate"]):
        with pytest.raises(SystemExit, match="basal-router: "):
            R.main(base + bad)


# Blend (2.5): --blend URL ... --blend-weights W0 W1 ...: the weighted mean of the models' probabilities
def _choice(p):
    return {"type": "choice", "choice": max(p, key=p.get), "probabilities": dict(p), "confidence": max(p.values())}


def _noul(t):
    return {"type": "noul", "noul": t, "probabilities": {"true": t, "false": 1 - t}, "confidence": max(t, 1 - t)}


def _score(p):
    return {"type": "score", "score": float(sum(i * v for i, v in enumerate(p.values()))), "legend": {"0": "low"},
            "probabilities": dict(p), "confidence": max(p.values())}


BLEND_CONF = {SMALL: {"c": _choice({"a": 0.5, "b": 0.3, "c": 0.2}), "n": _noul(0.3), "s": _score({"0": 0.6, "1": 0.4})},
              MID: {"c": _choice({"a": 0.2, "b": 0.7, "c": 0.1}), "n": _noul(0.9), "s": _score({"0": 0.1, "1": 0.9})},
              BIG: {"c": _choice({"a": 0.45, "b": 0.45, "c": 0.1}), "n": _noul(0.6), "s": _score({"0": 0.5, "1": 0.5})}}
W = (0.2, 0.2, 0.6)


def blend_router(conf, asked, **kw):
    return R.Router([SMALL], transport=level_transport(conf, asked, **kw), blend=[([MID], W[1]), ([BIG], W[2])],
                    blend_weight=W[0])


def _mean(q, urls, ws, conf=BLEND_CONF):
    keys = list(conf[SMALL][q]["probabilities"])
    return {k: sum(w * conf[u][q]["probabilities"][k] for w, u in zip(ws, urls)) / sum(ws) for k in keys}


def test_blend_mixes_the_models_probabilities():
    asked = {}
    rt = blend_router(BLEND_CONF, asked)
    qs = {"c": {"type": "choice"}, "n": {"type": "noul"}, "s": {"type": "score"}}
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "Doc", "questions": qs}).json()
        m = c.get("/metrics").text
    a, urls = r["answers"], (SMALL, MID, BIG)
    for q in qs:
        p = _mean(q, urls, W)
        assert a[q]["probabilities"] == p and a[q]["confidence"] == max(p.values())  # bit for bit: the same formula
        assert a[q]["blend"] == {"models": ["model-small", "model-mid", "model-big"], "weights": list(W)}
        assert a[q]["calibrated"] is False
    assert a["c"]["choice"] == "b"                                # a 0.41, b 0.47, c 0.12 (the first replica said a)
    assert a["n"]["noul"] == _mean("n", urls, W)["true"]
    assert a["s"]["score"] == sum(i * v for i, v in enumerate(_mean("s", urls, W).values()))
    assert a["s"]["legend"] == {"0": "low"} and r["model"] == "model-small"
    assert asked == {SMALL: [sorted(qs)], MID: [sorted(qs)], BIG: [sorted(qs)]}
    assert 'basal_router_blend_answers_total{models="3"} 3' in m


def test_blended_answers_carry_no_single_model_calibration():
    """Fields that rest on one model's calibration are dropped from a blended answer (a replica that adds them to
    plain answers: an older or another server)."""
    extra = {"abstain": False, "max_error": "x", "answer_set": ["a"], "coverage": 0.9, "coverage_note": "x",
             "calibration": "validated", "reread": True}
    conf = {u: dict(BLEND_CONF[u], c=dict(BLEND_CONF[u]["c"], **extra)) for u in (SMALL, MID, BIG)}
    rt = blend_router(conf, {})
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        a = c.post("/v1/systemone", json={"state": "Doc", "questions": {"c": {"type": "choice"}}}).json()["answers"]["c"]
    assert not set(a) & set(R.CERTIFIED) and a["calibrated"] is False and a["reread"] is True


def test_blend_leaves_other_answers_to_the_first_replica():
    asked = {}
    conf = {u: dict(BLEND_CONF[u], m={"type": "multi", "selected": ["x"], "probabilities": {"x": 0.6, "y": 0.1}},
                    t={"type": "classify", "label": "x", "scores": {"x": 0.6}, "confidence": 0.6},
                    k=_choice({"a": 0.9, "b": 0.1}), e=_choice({"a": 0.9, "b": 0.1})) for u in (SMALL, MID, BIG)}
    rt = blend_router(conf, asked)
    qs = {"c": {"type": "choice"}, "m": {"type": "multi"}, "t": {"type": "classify"},
          "k": {"type": "choice", "max_error": 0.05}, "e": {"type": "choice", "evidence": True}}
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        a = c.post("/v1/systemone", json={"state": "Doc", "questions": qs}).json()["answers"]
    assert asked[MID] == [["c"]] and asked[BIG] == [["c"]]       # only the blendable question is asked again
    for q in ("m", "t", "k", "e"):
        assert a[q] == conf[SMALL][q]
    assert "blend" in a["c"]


@pytest.mark.parametrize("down", ["not ready", "503", "unreachable"])
def test_blend_without_a_model_uses_the_others(down):
    kw = {"not ready": {"health": {BIG: 503}}, "503": {"post": {BIG: 503}},
          "unreachable": {"post": {BIG: httpx.ConnectError("refused")}}}[down]
    rt = blend_router(BLEND_CONF, {}, **kw)
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        resp = c.post("/v1/systemone", json={"state": "Doc", "questions": {"c": {"type": "choice"}}})
    assert resp.status_code == 200
    a = resp.json()["answers"]["c"]
    assert a["probabilities"] == _mean("c", (SMALL, MID), W[:2]) and a["blend"]["models"] == ["model-small", "model-mid"]


def test_blend_with_only_the_first_model_keeps_its_answer():
    rt = blend_router(BLEND_CONF, {}, post={MID: 503, BIG: 503})
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        a = c.post("/v1/systemone", json={"state": "Doc", "questions": {"c": {"type": "choice"}}}).json()["answers"]["c"]
    assert a == dict(BLEND_CONF[SMALL]["c"], blend={"models": ["model-small"], "weights": [W[0]]})


def test_blend_from_the_command_line(monkeypatch):
    got = []
    monkeypatch.setattr(R, "Router", lambda urls, **kw: got.append((kw["blend_weight"], kw["blend"]))
                        or type("F", (), {"dead": False})())
    monkeypatch.setattr(R, "app", lambda *a, **k: None)
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    base = ["--replicas", SMALL]
    R.main(base)
    R.main(base + ["--blend", MID, "--blend", BIG, "http://big2", "--blend-weights", "0.2", "0.2", "0.6"])
    assert got == [(1.0, []), (0.2, [([MID], 0.2), ([BIG, "http://big2"], 0.6)])]
    for bad in (["--blend", MID, "--blend-weights", "0.5"], ["--blend-weights", "1", "1"],
                ["--blend", MID, "--blend-weights", "0.5", "-0.1"], ["--blend", MID, "--blend-weights", "0", "0"],
                ["--blend", MID, "--blend-weights", "1", "1", "--escalate", BIG]):
        with pytest.raises(SystemExit, match="basal-router: "):
            R.main(base + bad)


# LLM fallback (2.5): --fallback URL --fallback-model NAME --fallback-below T after the last level
LLM = "http://llm/v1"


def llm_reply(letter="B", alts=None, split=False, content=None, model="llm-x"):
    """A chat completion choosing letter; alts {letter: logprob} for its top logprobs (None: no logprobs); split: the
    letter inside a '"B' token."""
    body = {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant",
                                                     "content": content or json.dumps({"answer": letter})}}]}
    if alts is not None:
        lead = ['{"', 'answer', '":', ' "'] if not split else ['{"', 'answer', '":', ' ']
        mid = letter if not split else '"' + letter
        toks = [{"token": t, "logprob": -0.01, "top_logprobs": []} for t in lead]
        toks.append({"token": mid, "logprob": alts[letter],
                     "top_logprobs": [{"token": (mid[:-1] if split else "") + c, "logprob": v} for c, v in alts.items()]
                     + [{"token": "zz", "logprob": -1.0}]})
        toks.append({"token": '"}' if not split else '"}', "logprob": -0.01, "top_logprobs": []})
        body["choices"][0]["logprobs"] = {"content": toks}
    return body


def fallback_transport(conf, asked, llm=None, **kw):
    """level_transport's replicas plus an LLM endpoint at LLM: llm(request body) -> a chat completion, an int status
    or an exception; every chat request is kept in asked["llm"]."""
    base = level_transport(conf, asked, **kw)

    def handler(req):
        if req.url.host == "llm":
            body = json.loads(req.content)
            asked.setdefault("llm", []).append(body)
            out = (llm or (lambda b: llm_reply()))(body)
            if isinstance(out, Exception):
                raise out
            return httpx.Response(out, json={"error": "x"}) if isinstance(out, int) else httpx.Response(200, json=out)
        return base.handle_request(req)
    return httpx.MockTransport(handler)


FB = {"url": LLM, "model": "llm-x", "below": 0.6, "timeout": 5.0, "max_tokens": 8192, "extra": {}}
OPTS = {"a": "Approve", "b": "Reject", "c": "Escalate"}


def _ch(p):  # a choice answer over a, b, c with top probability p on a
    rest = (1 - p) / 2
    return {"type": "choice", "choice": "a", "probabilities": {"a": p, "b": rest, "c": rest}, "confidence": p}


def _ask(rt, body):
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json=body)
        return r, c.get("/metrics").text


def test_fallback_takes_only_questions_under_its_threshold():
    asked = {}
    conf = {SMALL: {"q1": _ch(0.4), "q2": _ch(0.9)}}
    rt = R.Router([SMALL], transport=fallback_transport(conf, asked, lambda b: llm_reply("B", {"A": -2.0, "B": -0.2,
                                                                                              "C": -3.0})),
                  fallback=FB)
    q = {"type": "choice", "instructions": "What to do?", "criteria": OPTS}
    r, m = _ask(rt, {"state": "Invoice 17 is overdue.", "questions": {"q1": q, "q2": q}})
    out = r.json()
    a = out["answers"]["q1"]
    p = [math.exp(-2.0), math.exp(-0.2), math.exp(-3.0)]
    assert a["choice"] == "b" and a["answered_by"] == "fallback" and a["model"] == "llm-x" and a["calibrated"] is False
    assert a["probabilities"] == pytest.approx({k: v / sum(p) for k, v in zip("abc", p)}, abs=1e-12)
    assert a["confidence"] == max(a["probabilities"].values())
    assert a["basal"] == {"choice": "a", "confidence": 0.4, "model": "model-small"}
    assert out["answers"]["q2"] == conf[SMALL]["q2"]                                # above the threshold: untouched
    assert out["fallback"] == {"below": 0.6, "asked": ["q1"], "answered": ["q1"]}
    (req,) = asked["llm"]
    user = req["messages"][-1]["content"]
    assert "Invoice 17 is overdue." in user and "What to do?" in user and "A. a: Approve\nB. b: Reject\nC. c: Escalate" in user
    assert req["model"] == "llm-x" and req["temperature"] == 0 and req["seed"] == 0 and req["logprobs"] is True
    assert req["max_tokens"] == 8192
    assert req["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["A", "B", "C"]
    assert 'basal_router_fallback_total{result="answered"} 1' in m


def test_fallback_options_follow_option_keys_and_the_letter_inside_a_token():
    asked = {}
    conf = {SMALL: {"q": _ch(0.4)}}
    rt = R.Router([SMALL], transport=fallback_transport(
        conf, asked, lambda b: llm_reply("C", {"A": -4.0, "C": -0.1}, split=True)), fallback=FB)
    r, _ = _ask(rt, {"state": {"ticket": "down"}, "questions": {"q": {"criteria": OPTS, "option_keys": "hide"}}})
    a = r.json()["answers"]["q"]
    assert a["choice"] == "c" and a["probabilities"]["b"] == 0.0                     # B not among the top logprobs
    assert a["probabilities"]["c"] == pytest.approx(1 / (1 + math.exp(-3.9)))
    user = asked["llm"][0]["messages"][-1]["content"]
    assert '{"ticket": "down"}' in user and "A. Approve\nB. Reject\nC. Escalate" in user


def test_fallback_without_logprobs_gives_no_probabilities():
    conf = {SMALL: {"q": _ch(0.4)}}
    rt = R.Router([SMALL], transport=fallback_transport(conf, {}, lambda b: llm_reply("A")), fallback=FB)
    a = _ask(rt, {"state": "s", "questions": {"q": {"criteria": OPTS}}})[0].json()["answers"]["q"]
    assert a["choice"] == "a" and a["probabilities"] is None and a["confidence"] is None
    assert a["answered_by"] == "fallback"


@pytest.mark.parametrize("llm,result", [
    (lambda b: httpx.ConnectError("refused"), "error"),
    (lambda b: httpx.ReadTimeout("slow"), "timeout"),
    (lambda b: 500, "error"),
    (lambda b: llm_reply(content="B"), "unparseable"),
    (lambda b: llm_reply(content='{"answer": "B", "why": "x"}'), "unparseable"),
    (lambda b: llm_reply(content='{"answer": "D"}'), "not_an_option"),
    (lambda b: llm_reply(content='{"answer": "AB"}'), "not_an_option"),
    (lambda b: {"choices": []}, "unparseable"),
])
def test_fallback_failure_keeps_the_basal_answer(llm, result):
    conf = {SMALL: {"q": _ch(0.4)}}
    rt = R.Router([SMALL], transport=fallback_transport(conf, {}, llm), fallback=FB)
    r, m = _ask(rt, {"state": "s", "questions": {"q": {"criteria": OPTS}}})
    assert r.status_code == 200
    out = r.json()
    assert out["answers"]["q"] == conf[SMALL]["q"] and out["fallback"]["answered"] == []
    assert f'basal_router_fallback_total{{result="{result}"}} 1' in m


@pytest.mark.parametrize("errors,result,calls", [
    ([httpx.ReadError("closed")], "answered", 2),                                    # a stale pooled connection: retried
    ([httpx.RemoteProtocolError("disconnected")], "answered", 2),
    ([httpx.ReadError("closed"), httpx.ReadError("closed")], "error", 2),              # once only
    ([httpx.ConnectError("refused")], "error", 1),                                   # down: no retry
])
def test_fallback_retries_once_on_a_connection_the_endpoint_closed(errors, result, calls):
    conf, asked, left = {SMALL: {"q": _ch(0.4)}}, {}, list(errors)
    rt = R.Router([SMALL], transport=fallback_transport(conf, asked, lambda b: left.pop(0) if left else llm_reply("B")),
                  fallback=FB)
    r, m = _ask(rt, {"state": "s", "questions": {"q": {"criteria": OPTS}}})
    assert r.status_code == 200 and len(asked["llm"]) == calls
    assert (r.json()["answers"]["q"].get("answered_by") == "fallback") == (result == "answered")
    assert f'basal_router_fallback_total{{result="{result}"}} 1' in m


@pytest.mark.parametrize("first_s,second_s,result,calls", [
    (0.5, 3.0, "timeout", 2),    # the retry gets only the 1.5 s left of 2 s, then times out
    (1.3, 0.0, "error", 1),      # 0.7 s left < RETRY_MIN_S: no retry
])
def test_fallback_retry_stays_inside_the_timeout(first_s, second_s, result, calls):
    """The two attempts together stay inside --fallback-timeout."""
    calls_made = []

    async def handler(req):
        if req.url.host == "llm":
            calls_made.append(time.monotonic())
            if len(calls_made) == 1:
                await asyncio.sleep(first_s)
                raise httpx.ReadError("closed")
            await asyncio.sleep(second_s)
            return httpx.Response(200, json=llm_reply("B"))
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"model": "m", "answers": {"q": _ch(0.4)}})
    rt = R.Router([SMALL], transport=httpx.MockTransport(handler), fallback=dict(FB, timeout=2.0))
    t = time.monotonic()
    r, m = _ask(rt, {"state": "s", "questions": {"q": {"criteria": OPTS}}})
    assert r.status_code == 200 and len(calls_made) == calls
    assert calls_made[0] - t < 1.0 and time.monotonic() - calls_made[0] < 2.0 + 0.5   # (TestClient start-up excluded)
    assert f'basal_router_fallback_total{{result="{result}"}} 1' in m


def test_fallback_never_takes_multi_tasks_limits_evidence_media_or_other_types():
    asked = {}
    low = _ch(0.3)
    conf = {SMALL: {"m": {"type": "multi", "selected": ["a"], "probabilities": {"a": 0.4, "b": 0.3}},
                    "t": {"type": "classify", "label": "a", "scores": {"a": 0.4}, "confidence": 0.4},
                    "k": low, "v": low, "e": low,
                    "n": {"type": "noul", "noul": 0.45, "probabilities": {"true": 0.45, "false": 0.55}, "confidence": 0.55},
                    "s": {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5}}}
    rt = R.Router([SMALL], transport=fallback_transport(conf, asked), fallback=FB)
    qs = {"m": {"type": "multi", "criteria": OPTS}, "t": {"type": "classify", "labels": ["a"]},
          "k": {"criteria": OPTS, "max_error": 0.05}, "v": {"criteria": OPTS, "coverage": 0.9},
          "e": {"criteria": OPTS, "evidence": True}, "n": {"type": "noul"}, "s": {"type": "score", "criteria": ["0", "1"]}}
    r, _ = _ask(rt, {"state": "s", "questions": qs})
    assert "llm" not in asked and r.json()["answers"] == conf[SMALL] and "fallback" not in r.json()
    conf[SMALL]["q"] = low
    media = [{"type": "image", "data": "x"}]
    r, _ = _ask(R.Router([SMALL], transport=fallback_transport(conf, asked), fallback=FB),
                {"state": "s", "questions": {"q": {"criteria": OPTS}}, "media": media})
    assert r.status_code == 200 and r.json()["answers"]["q"] == low and "llm" not in asked
    _ask(R.Router([SMALL], transport=fallback_transport(conf, asked), fallback=FB),
         {"state": "s", "questions": {"q": {"criteria": OPTS}}})                          # without media it goes
    assert len(asked["llm"]) == 1


def test_fallback_after_a_cascade_takes_only_the_last_level():
    asked = {}
    conf = {SMALL: {"q1": _ch(0.85), "q2": _ch(0.5)}, MID: {"q2": _ch(0.6)}, BIG: {"q2": _ch(0.7)}}
    rt = R.Router([SMALL], transport=fallback_transport(conf, asked, lambda b: llm_reply("C")),
                  levels=[([MID], 0.8), ([BIG], 0.9)], fallback=dict(FB, below=0.9))
    out = _ask(rt, {"state": "s", "questions": {"q1": {"criteria": OPTS}, "q2": {"criteria": OPTS}}})[0].json()
    assert out["answers"]["q1"]["level"] == 1 and out["answers"]["q1"]["choice"] == "a"   # 0.85 < 0.9 but level 1
    assert out["answers"]["q2"]["answered_by"] == "fallback" and out["answers"]["q2"]["basal"]["model"] == "model-big"
    assert len(asked["llm"]) == 1 and out["fallback"]["asked"] == ["q2"]


def test_fallback_with_nothing_to_ask_answers_as_without_it():
    conf = {SMALL: {"q": _ch(0.9)}}
    body = {"state": "s", "questions": {"q": {"criteria": OPTS}}}
    a = _ask(R.Router([SMALL], transport=fallback_transport(conf, {})), body)[0].content
    b = _ask(R.Router([SMALL], transport=fallback_transport(conf, {}), fallback=FB), body)[0].content
    assert a == b


def test_fallback_from_the_command_line(monkeypatch):
    got = []
    monkeypatch.setattr(R, "Router", lambda urls, **kw: got.append(kw["fallback"]) or type("F", (), {"dead": False})())
    monkeypatch.setattr(R, "app", lambda *a, **k: None)
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    base = ["--replicas", SMALL]
    R.main(base)
    R.main(base + ["--fallback", LLM, "--fallback-model", "m"])
    R.main(base + ["--fallback", LLM, "--fallback-model", "m", "--fallback-below", "0.7", "--fallback-timeout", "20",
                   "--fallback-max-tokens", "16384",
                   "--fallback-extra", '{"chat_template_kwargs": {"enable_thinking": true}}'])
    assert got == [None, {"url": LLM, "model": "m", "below": 0.5, "timeout": 60.0, "max_tokens": 8192, "extra": {}},
                   {"url": LLM, "model": "m", "below": 0.7, "timeout": 20.0, "max_tokens": 16384,
                    "extra": {"chat_template_kwargs": {"enable_thinking": True}}}]
    for bad in (["--fallback", LLM], ["--fallback-model", "m"], ["--fallback-max-tokens", "10"],
                ["--fallback", LLM, "--fallback-model", "m", "--fallback-max-tokens", "0"], ["--fallback", LLM, "--fallback-model", "m",
                "--fallback-extra", "[1]"], ["--fallback", LLM, "--fallback-model", "m", "--fallback-below", "0"],
                ["--fallback", LLM, "--fallback-model", "m", "--blend", MID, "--blend-weights", "1", "1"]):
        with pytest.raises(SystemExit, match="basal-router: "):
            R.main(base + bad)


def test_fallback_request_merges_extra():
    msgs = R.fallback_messages("doc", {"instructions": "Q?"}, [("a", "x"), ("b", "y")])
    body = R.fallback_request("m", msgs, 2, {"chat_template_kwargs": {"enable_thinking": True}}, max_tokens=16384)
    assert body["max_tokens"] == 16384 and body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 0 and body["seed"] == 0
    assert R.fallback_request("m", msgs, 2, {"max_tokens": 100})["max_tokens"] == 100  # extra is merged last
    assert body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["A", "B"]


# API keys (2.5): a client's key passed on to replicas started with --api-keys (basal.auth.protect, as basal-serve)
KEY = "bsl_router-test"


class Hosts(httpx.AsyncBaseTransport):
    """Each host its own ASGI app (a stub replica, the LLM endpoint)."""

    def __init__(self, apps):
        self.apps = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.apps[request.url.host].handle_async_request(request)


def key_replica(name, conf, seen, keys=None):
    """A stub replica behind basal.auth.protect (keys: its --api-keys, an ApiKeys; None: open): every question gets
    top probability conf; the headers of every request it serves go to seen[name]."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from basal.auth import protect

    async def health(_):
        return JSONResponse({"status": "ok"})

    async def decide(request):
        seen.setdefault(name, []).append(dict(request.headers))
        qs = (await request.json())["questions"]
        return JSONResponse({"model": name, "answers": {q: {"type": "choice", "choice": "a",
                                                            "probabilities": {"a": conf, "b": 1 - conf}} for q in qs}})

    async def models(request):
        seen.setdefault(name, []).append(dict(request.headers))
        return JSONResponse({"object": "list", "data": [{"id": name}]})

    async def chat(request):
        seen.setdefault(name, []).append(dict(request.headers))
        return JSONResponse(llm_reply("A"))

    return protect(Starlette(routes=[Route("/health", health), Route("/v1/systemone", decide, methods=["POST"]),
                                     Route("/v1/basal", decide, methods=["POST"]), Route("/v1/models", models),
                                     Route("/v1/chat/completions", chat, methods=["POST"])]), keys)


def test_key_headers_are_the_last_authorization_and_x_api_key():
    raw = [(b"Authorization", b"Bearer a"), (b"x-api-key", b"k1"), (b"x-api-key", b"k2"), (b"x-request-id", b"r"),
           (b"content-type", b"text/plain")]
    assert R.key_headers(raw) == {"authorization": b"Bearer a", "x-api-key": b"k2"}
    assert R.key_headers([(b"content-type", b"application/json")]) == {}
    # a value with surrounding whitespace ("Bearer " with no token) is no valid HTTP header to send on: the router
    # answered 502 instead of the replica's 401
    assert R.key_headers([(b"authorization", b"Bearer "), (b"x-api-key", b" k\t")]) == {"authorization": b"Bearer",
                                                                                         "x-api-key": b"k"}


def test_router_passes_client_errors_on_to_the_replica():
    """A state with a lone surrogate and a body nested too deeply for the parser are the replica's client errors to
    answer; the router failed on them with 500 (hashing the routing key, reading the state)."""
    seen = []

    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.append(len(req.content))
        return httpx.Response(422, json={"error": "client error", "code": "INVALID_REQUEST"})
    rt = R.Router(["http://r0"], transport=httpx.MockTransport(handler))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        deep = b'{"state": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
        for body in (b'{"state": "a \\ud800 b", "questions": {}}', deep):
            r = c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
            assert r.status_code == 422 and r.json()["code"] == "INVALID_REQUEST"
    assert len(seen) == 2


def test_router_passes_api_keys_on_to_replicas_with_api_keys(tmp_path):
    """Replicas with --api-keys (one key file): the client's key reaches the first replica, the escalation replica and
    /v1/models, and never the --fallback LLM; without a key or with a wrong one the replica's 401 comes back, and the
    replica stays ready."""
    from basal.auth import ApiKeys
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": "client", "sha256": hashlib.sha256(KEY.encode()).hexdigest()}]}))
    keys, seen = ApiKeys(path), {}
    small, big = "http://small", "http://big"
    rt = R.Router([small], escalate=[big], below=0.8, fallback=dict(FB, below=0.95),
                  transport=Hosts({"small": key_replica("small", 0.6, seen, keys),
                                   "big": key_replica("big", 0.9, seen, keys), "llm": key_replica("llm", 0, seen)}))
    body = {"state": "Doc", "questions": {"q": {"type": "choice", "criteria": {"a": "Yes", "b": "No"}}}}
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        for headers in ({}, {"Authorization": "Bearer bsl_wrong"}, {"X-API-Key": "zażółć".encode()}):
            r = c.post("/v1/systemone", json=body, headers=headers)
            assert r.status_code == 401 and r.json() == {"error": "missing or invalid API key", "code": "UNAUTHORIZED"}
        assert rt.ready == {small, big} and "big" not in seen and "llm" not in seen
        for headers in ({"Authorization": f"Bearer {KEY}"}, {"X-API-Key": KEY}):
            r = c.post("/v1/systemone", json=body, headers=headers)
            assert r.status_code == 200 and r.json()["answers"]["q"]["answered_by"] == "fallback"
            assert r.json()["cascade"] == {"below": 0.8, "escalated": ["q"]}
            name, value = next(iter(headers.items()))
            assert [h[name.lower()] for h in (seen["small"][-1], seen["big"][-1])] == [value, value]
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": KEY}).json() == {"object": "list", "data": [{"id": "small"}]}
    assert len(seen["llm"]) == 2 and not any({"authorization", "x-api-key"} & set(h) for h in seen["llm"])
    assert keys.usage()[0]["last_used"] is not None


def test_router_without_api_keys_sends_no_key():
    """Open replicas (no --api-keys): a request without a key reaches them with no key header, as before; one with a
    key is answered too (an open replica ignores it)."""
    seen = {}
    rt = R.Router(["http://r0"], transport=Hosts({"r0": key_replica("r0", 0.9, seen)}))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "Doc", "questions": {"q": {}}})
        assert r.status_code == 200 and r.json()["model"] == "r0"
        assert c.get("/v1/models").status_code == 200
        assert c.post("/v1/basal", json={"questions": {"q": {}}}, headers={"X-API-Key": KEY}).status_code == 200
    first, models, keyed = seen["r0"]
    assert first["content-type"] == "application/json"
    assert not {"authorization", "x-api-key"} & (set(first) | set(models)) and keyed["x-api-key"] == KEY
