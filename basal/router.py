"""basal-router (2.5): several basal-serve replicas (one per GPU) behind one HTTP endpoint.

A request goes to the replica its state is routed to, so the questions over one document meet the same replica's
state sharing (SOAM) and state cache: consistent hashing of the state's text over a ring of the ready replicas, with
bounded load (a replica already holding more than LOAD_FACTOR x the mean in-flight requests passes the request on
along the ring). Requests without a state go to the less loaded of two random replicas. Replicas are polled at
/health; one answering 503 (starting) or not answering gets no requests until it is ready again. With --gpus, the
router exits (status 1) when every replica it started has exited, and the replicas it started end with it: each runs
in its own process group, which the router ends when it stops (SIGTERM, SIGINT, SIGHUP, an error) and, on Linux, which
gets SIGTERM when the router is killed.

  basal-router --replicas http://127.0.0.1:8001 http://127.0.0.1:8002 --port 8000
  basal-router --gpus 0,1,2,3 --port 8000 -- --model Remek/basal-2.0-mini --mode fast   (starts one basal-serve per
                                                                                          GPU on ports 8001..)
GET /health: 200 when at least one replica is ready, else 503; /metrics: per replica in flight, requests routed by
state / by load, failures, readiness (Prometheus); /v1/models: the first ready replica's.
"""
import argparse
import asyncio
import atexit
import bisect
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time

LOAD_FACTOR = 1.25
VNODES = 64
POLL_S = 2.0
SERVE = [sys.executable, "-m", "basal.server"]  # what --gpus starts per GPU (spawn)


def state_key(body):
    """The routing key of a request body: its state with whitespace normalised (None without a state)."""
    try:
        state = json.loads(body).get("state")
    except (ValueError, AttributeError):
        return None
    if not isinstance(state, str) or not state.strip():
        return None
    return re.sub(r"\s+", " ", state.strip())


def _h(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class Ring:
    """Consistent hashing with bounded load over replica names (VNODES points per replica)."""

    def __init__(self, names, vnodes=VNODES):
        self.points = sorted((_h(f"{n}#{v}"), n) for n in names for v in range(vnodes))
        self.keys = [p[0] for p in self.points]

    def pick(self, key, load, ready, factor=LOAD_FACTOR):
        """The replica for key: the first ready one clockwise from hash(key) whose load stays within factor x the
        mean load (counting this request); None when no replica is ready."""
        names = [n for n in dict.fromkeys(n for _, n in self.points) if n in ready]
        if not names:
            return None
        cap = math.ceil(factor * (sum(load.get(n, 0) for n in names) + 1) / len(names))  # (at least 1)
        i = bisect.bisect(self.keys, _h(key)) % len(self.points)
        seen = set()
        for j in range(len(self.points)):
            n = self.points[(i + j) % len(self.points)][1]
            if n in ready and n not in seen:
                seen.add(n)
                if load.get(n, 0) + 1 <= cap:
                    return n
        return min(names, key=lambda n: load.get(n, 0))


class Router:
    def __init__(self, replicas, transport=None, factor=LOAD_FACTOR, seed=0):
        import httpx
        self.replicas = list(replicas)
        self.ring = Ring(self.replicas)
        self.ready = set()
        self.load = {r: 0 for r in self.replicas}
        self.routed = {(r, how): 0 for r in self.replicas for how in ("state", "load")}
        self.failed = {r: 0 for r in self.replicas}
        self.factor = factor
        self.rng = random.Random(seed)
        self.dead = False
        self.client = httpx.AsyncClient(timeout=None, transport=transport,
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))

    def choose(self, key):
        """(replica, how) for a request with routing key key (None: by load)."""
        if key is not None:
            r = self.ring.pick(key, self.load, self.ready, self.factor)
            return r, "state"
        ready = sorted(self.ready)
        if not ready:
            return None, "load"
        a, b = self.rng.choice(ready), self.rng.choice(ready)
        return (a if self.load[a] <= self.load[b] else b), "load"

    async def poll(self):
        """Readiness of every replica (GET /health: 200 = ready)."""
        async def one(r):
            try:
                ok = (await self.client.get(r + "/health", timeout=5)).status_code == 200
            except Exception:  # noqa: BLE001  (down or still binding)
                ok = False
            (self.ready.add if ok else self.ready.discard)(r)
        await asyncio.gather(*(one(r) for r in self.replicas))

    async def forward(self, path, body, headers):
        """-> (status, content, content type)."""
        r, how = self.choose(state_key(body))
        if r is None:
            return 503, b'{"error": "no basal replica is ready"}', "application/json"
        self.load[r] += 1
        self.routed[(r, how)] += 1
        try:
            resp = await self.client.post(r + path, content=body, headers=headers)
            if resp.status_code >= 500:
                self.failed[r] += 1
            return resp.status_code, resp.content, resp.headers.get("content-type", "application/json")
        except Exception as e:  # noqa: BLE001
            self.failed[r] += 1
            self.ready.discard(r)
            return 502, json.dumps({"error": f"replica {r}: {type(e).__name__}"}).encode(), "application/json"
        finally:
            self.load[r] -= 1

    def metrics(self):
        from .server import prometheus
        return prometheus([
            ("basal_router_replica_ready", "1 when the replica answers /health with 200", "gauge",
             [({"replica": r}, int(r in self.ready)) for r in self.replicas]),
            ("basal_router_in_flight", "requests the router has in flight at the replica", "gauge",
             [({"replica": r}, self.load[r]) for r in self.replicas]),
            ("basal_router_routed_total", "requests routed, by the state's hash or by load", "counter",
             [({"replica": r, "by": how}, n) for (r, how), n in self.routed.items()]),
            ("basal_router_failed_total", "requests the replica answered with 5xx or not at all", "counter",
             [({"replica": r}, n) for r, n in self.failed.items()])])


def app(router, poll_s=None, procs=(), on_dead=lambda: signal.raise_signal(signal.SIGTERM), on_stop=None):
    """The router's app. procs: the replicas it started (--gpus); on_stop: called at shutdown (main: Replicas.stop)."""
    from contextlib import asynccontextmanager
    poll_s = POLL_S if poll_s is None else poll_s

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route

    async def post(request):
        status, content, ctype = await router.forward(request.url.path, await request.body(),
                                                      {"content-type": "application/json"})
        return Response(content, status_code=status, media_type=ctype)

    async def health(_):
        body = {"status": "ok" if router.ready else "starting", "ready": len(router.ready),
                "replicas": len(router.replicas), "in_flight": sum(router.load.values())}
        return JSONResponse(body, status_code=200 if router.ready else 503)

    async def models(_):
        for r in sorted(router.ready):
            try:
                resp = await router.client.get(r + "/v1/models", timeout=10)
                return Response(resp.content, status_code=resp.status_code, media_type="application/json")
            except Exception:  # noqa: BLE001
                continue
        return JSONResponse({"error": "no basal replica is ready"}, status_code=503)

    async def metrics(_):
        return PlainTextResponse(router.metrics(), media_type="text/plain; version=0.0.4")

    @asynccontextmanager
    async def lifespan(_app):
        async def loop():
            while True:
                await router.poll()
                if procs and all(p.poll() is not None for p in procs):
                    router.dead = True
                    print("basal-router: every replica exited", file=sys.stderr, flush=True)
                    on_dead()  # SIGTERM: uvicorn shuts down, main exits with status 1
                    return
                await asyncio.sleep(poll_s)
        task = asyncio.get_running_loop().create_task(loop())
        yield
        task.cancel()
        await router.client.aclose()
        if on_stop is not None:  # (uvicorn's shutdown: before it re-raises the signal that stopped it)
            await asyncio.get_running_loop().run_in_executor(None, on_stop)

    return Starlette(routes=[Route("/v1/systemone", post, methods=["POST"]), Route("/v1/basal", post, methods=["POST"]),
                             Route("/health", health), Route("/v1/models", models), Route("/metrics", metrics)],
                     lifespan=lifespan)


class Replicas:
    """The basal-serve processes of --gpus. stop() ends each one's process group (SIGTERM, then SIGKILL after
    `timeout` seconds), once, from whichever exit path of the router comes first: the app's shutdown, main's signal
    handler (uvicorn re-raises the signal it caught after its own shutdown, which would end the process before main's
    finally), main's finally, atexit."""

    def __init__(self, procs=()):
        self.procs, self.stopped = list(procs), False

    def stop(self, timeout=30):
        if self.stopped:
            return
        self.stopped = True
        for p in self.procs:
            _end_group(p)
        end = time.monotonic() + timeout
        for p in self.procs:
            try:
                p.wait(timeout=max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for p in self.procs:  # a replica past the timeout, or a process a replica left in its group
            _end_group(p, hard=True)
            p.wait()


def _end_group(p, hard=False):
    """SIGTERM (hard: SIGKILL) to a replica's process group; without process groups (Windows) to the replica."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(p.pid, signal.SIGKILL if hard else signal.SIGTERM)
        elif p.poll() is None:
            p.kill() if hard else p.terminate()
    except (ProcessLookupError, PermissionError):
        pass


def _death_signal():
    """preexec_fn of a replica on Linux: SIGTERM when the router dies, however it dies (prctl PR_SET_PDEATHSIG);
    None elsewhere. libc is loaded here, before the fork."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()

    def child():
        prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:  # the router already died
            os._exit(1)
    return child


def spawn(gpus, port0, serve_args):
    """One basal-serve per GPU (CUDA_VISIBLE_DEVICES), on ports port0, port0 + 1, ... -> (processes, URLs). Each in
    its own process group (Replicas.stop ends the replica and whatever it started) that, on Linux, gets SIGTERM when
    the router dies."""
    procs, urls, pre = [], [], _death_signal()
    for i, g in enumerate(gpus):
        port = port0 + i
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([*SERVE, "--port", str(port), *serve_args], env=env,
                                      start_new_session=hasattr(os, "killpg"), preexec_fn=pre))
        urls.append(f"http://127.0.0.1:{port}")
    return procs, urls


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    serve_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", nargs="*", default=[], help="URLs of running basal-serve replicas")
    ap.add_argument("--gpus", default=None, help="start one basal-serve per GPU, e.g. 0,1,2,3 (arguments after --)")
    ap.add_argument("--replica-port", dest="replica_port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--load-factor", dest="load_factor", type=float, default=LOAD_FACTOR)
    a = ap.parse_args(argv)
    procs, urls = ([], list(a.replicas))
    if a.gpus:
        procs, more = spawn(a.gpus.split(","), a.replica_port, serve_args)
        urls += more
    if not urls:
        raise SystemExit("basal-router: no replicas (--replicas URL ... or --gpus 0,1 -- <basal-serve arguments>)")
    import uvicorn
    print(f"basal-router: {len(urls)} replicas, port {a.port}", flush=True)
    router = Router(urls, factor=a.load_factor)
    replicas = Replicas(procs)
    atexit.register(replicas.stop)

    def ended(sig, _frame):
        """SIGTERM / SIGINT / SIGHUP outside uvicorn, or re-raised by it after its shutdown: the replicas end first."""
        replicas.stop()
        if router.dead:
            raise SystemExit(1)
        signal.signal(sig, signal.SIG_DFL)
        signal.raise_signal(sig)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), ended)
    try:
        uvicorn.run(app(router, procs=procs, on_stop=replicas.stop), host=a.host, port=a.port, log_level="warning")
    finally:
        replicas.stop()
    if router.dead:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
