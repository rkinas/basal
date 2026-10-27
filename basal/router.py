"""basal-router (2.5): several basal-serve replicas (one per GPU) behind one HTTP endpoint.

A request goes to the replica its state is routed to, so the questions over one document meet the same replica's
state sharing (SOAM) and state cache: consistent hashing of the state's text over a ring of the ready replicas, with
bounded load (a replica already holding more than LOAD_FACTOR x the mean in-flight requests passes the request on
along the ring). Requests without a state go to the less loaded of two random replicas. Replicas are polled at
/health; one answering 503 (starting) or not answering gets no requests until it is ready again. With --gpus, the
router exits (status 1) when every replica it started has exited, and the replicas it started end with it: each runs
in its own process group, which the router ends when it stops (SIGTERM, SIGINT, SIGHUP, an error) and, on Linux, which
gets SIGTERM when the router is killed.

Off by default (the replicas' answers then pass through untouched):
  --escalate URL ... --escalate-below P   a cascade: questions whose top probability (calibrated, as the first replica
                                          answers) is under P are asked again of the escalation replicas (a bigger
                                          model); every answer says "answered_by": "first" or "escalated". Repeated,
                                          the pairs are ordered levels: the first replica is level 1, the k-th
                                          --escalate level k + 1, and a question goes on from level k to level k + 1
                                          only while its top probability at level k is under the k-th
                                          --escalate-below; every answer then also gives its "level" and "model" (as
                                          that replica reports it). A level that is down or does not answer 200 leaves
                                          the previous answers. Multi and task answers never escalate.
  --blend URL ... --blend-weights W0 W1 ...
                                          a blend: every question is also asked of each --blend pool (one model
                                          each; repeat the flag), and a plain choice / noul / score answer gets the
                                          weighted mean of the models' probabilities (W0 for --replicas, then one per
                                          --blend, in order): choice, noul, score and confidence follow from it, and
                                          the answer says "blend": {"models": [...], "weights": [...]} and
                                          "calibrated": false (no model's thresholds hold for it: refit them on the
                                          blend's output). A model that is down or does not answer 200 is left out
                                          (the weights of the rest are used). Every question costs a call per model.
                                          Multi, act, tool, task answers, questions with coverage / max_error /
                                          evidence keep the first replica's answer. Not with --escalate.
  --route SCRIPT=URL[,URL]                requests whose state is mostly in that script (latin, cyrillic, greek,
                                          arabic, hebrew, cjk) go to those replicas instead

  basal-router --replicas http://127.0.0.1:8001 http://127.0.0.1:8002 --port 8000
  basal-router --gpus 0,1,2,3 --port 8000 -- --model Remek/basal-2.0-mini --mode fast   (starts one basal-serve per
                                                                                          GPU on ports 8001..)
  basal-router --replicas http://127.0.0.1:8001 --escalate http://127.0.0.1:8002 --escalate-below 0.8 \
               --escalate http://127.0.0.1:8003 --escalate-below 0.9   (pico on 8001, mini on 8002, ultra on 8003)
  basal-router --replicas http://127.0.0.1:8003 --blend http://127.0.0.1:8002 --blend http://127.0.0.1:8001 \
               --blend-weights 0.6 0.2 0.2   (ultra first: answers that are not blended are its own)
GET /health: 200 when at least one replica is ready, else 503; /metrics: per replica in flight, requests routed by
state / by load, failures, readiness, answers per cascade level, blended answers (Prometheus); /v1/models: the first
ready replica's.
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


SCRIPTS = ("latin", "cyrillic", "greek", "arabic", "hebrew", "cjk")


def script_of(text):
    """The script most of the letters of text are in (SCRIPTS, or "other"; None without letters)."""
    import unicodedata
    count = {}
    for ch in text or "":
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            sc = next((x for x in SCRIPTS[:5] if name.startswith(x.upper())), None)
            if sc is None:
                sc = "cjk" if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) else "other"
            count[sc] = count.get(sc, 0) + 1
    return max(count, key=count.get) if count else None


def top_probability(answer):
    """The top probability of an answer (its option probabilities), None for answers without one (multi, tasks)."""
    p = answer.get("probabilities") if isinstance(answer, dict) else None
    if answer.get("type") == "multi" or not isinstance(p, dict) or not p:
        return None
    return max(p.values())


BLENDS = ("choice", "noul", "score")
PLAIN = ("coverage", "max_error", "evidence", "multi")  # question fields whose answers are not blended
CERTIFIED = ("abstain", "max_error", "answer_set", "coverage", "coverage_note", "calibration")  # one model's calibration


def blendable(question, answer):
    """A plain choice / noul / score answer (its option probabilities only) to a question without limits or evidence."""
    return (isinstance(answer, dict) and answer.get("type") in BLENDS and isinstance(answer.get("probabilities"), dict)
            and bool(answer["probabilities"]) and isinstance(question, dict) and not any(k in question for k in PLAIN))


def blend_answer(answer, members):
    """answer (the first replica's) with the weighted mean of members' probabilities, [(weight, model, answer)] in
    order: p[k] = sum(w * p_m[k]) / sum(w); choice / noul / score / confidence recomputed from p as basal-serve does.
    The blend is not calibrated: "calibrated": false, and no field that rests on one model's calibration (CERTIFIED)."""
    total = sum(w for w, _, _ in members)
    p = {k: sum(w * a["probabilities"][k] for w, _, a in members) / total for k in answer["probabilities"]}
    out = {k: v for k, v in answer.items() if k not in CERTIFIED}
    out.update(probabilities=p, confidence=max(p.values()))
    if answer["type"] == "choice":
        out["choice"] = max(p, key=p.get)
    elif answer["type"] == "noul":
        out["noul"] = p["true"]
    else:
        out["score"] = float(sum(i * v for i, v in enumerate(p.values())))
    out["blend"] = {"models": [m for _, m, _ in members], "weights": [w for w, _, _ in members]}
    out["calibrated"] = False
    return out


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
    def __init__(self, replicas, transport=None, factor=LOAD_FACTOR, seed=0, escalate=(), below=0.0, routes=None,
                 levels=None, blend=None, blend_weight=1.0):
        """escalate, below: one cascade level; levels: [(URLs, threshold), ...] for levels 2, 3, ... instead;
        blend: [(URLs, weight), ...] asked beside the replicas (weight blend_weight)."""
        import httpx
        levels = [(list(u), b) for u, b in (levels if levels is not None else [(escalate, below)]) if u]
        self.levels = [(f"level:{k}", b) for k, (_, b) in enumerate(levels, start=2)]  # (level k's pool, k - 1's threshold)
        blend = [(list(u), w) for u, w in (blend or []) if u]
        self.blends = [(f"blend:{k}", w) for k, (_, w) in enumerate(blend, start=2)]
        self.blend_weight = blend_weight
        self.blended = {}
        self.pools = {"main": list(replicas), **{f"level:{k}": u for k, (u, _) in enumerate(levels, start=2)},
                      **{f"blend:{k}": u for k, (u, _) in enumerate(blend, start=2)},
                      **{f"script:{k}": list(v) for k, v in (routes or {}).items()}}
        self.replicas = list(dict.fromkeys(u for urls in self.pools.values() for u in urls))
        self.rings = {name: Ring(urls) for name, urls in self.pools.items() if urls}
        self.ring = self.rings.get("main")
        self.answered = {"first": 0, "escalated": 0}
        self.by_level = {k: 0 for k in range(1, len(self.levels) + 2)}
        self.scripts = {}
        self.ready = set()
        self.load = {r: 0 for r in self.replicas}
        self.routed = {(r, how): 0 for r in self.replicas for how in ("state", "load")}
        self.failed = {r: 0 for r in self.replicas}
        self.factor = factor
        self.rng = random.Random(seed)
        self.dead = False
        self.client = httpx.AsyncClient(timeout=None, transport=transport,
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))

    def choose(self, key, pool="main"):
        """(replica, how) for a request with routing key key (None: by load), among the replicas of pool."""
        if key is not None:
            r = self.rings[pool].pick(key, self.load, self.ready, self.factor)
            return r, "state"
        ready = sorted(self.ready & set(self.pools[pool]))
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
        """-> (status, content, content type): the request to the replica its pool and state give, then (--escalate)
        its low-confidence questions to the escalation replicas."""
        pool = "main"
        if any(name.startswith("script:") for name in self.rings):  # --route
            try:
                sc = script_of(json.loads(body).get("state"))
            except (ValueError, AttributeError):
                sc = None
            if sc is not None:
                self.scripts[sc] = self.scripts.get(sc, 0) + 1
                pool = f"script:{sc}" if f"script:{sc}" in self.rings else pool
        status, content, ctype = await self._send(pool, path, body, headers)
        if self.levels and status == 200:
            content = await self._cascade(path, body, content, headers)
        if self.blends and status == 200:
            content = await self._blend(path, body, content, headers)
        return status, content, ctype

    async def _blend(self, path, body, content, headers):
        """The first replica's answers, the blendable ones mixed with the --blend pools' (asked at once, concurrently)."""
        try:
            req, resp = json.loads(body), json.loads(content)
            answers = resp["answers"]
            qs = req.get("questions") or {}
            want = [q for q, a in answers.items() if q in qs and blendable(qs[q], a)]
        except (ValueError, KeyError, TypeError, AttributeError):
            return content
        if not want:
            return content
        sub = json.dumps(dict(req, questions={q: qs[q] for q in want})).encode()
        outs = await asyncio.gather(*(self._send(pool, path, sub, headers) for pool, _ in self.blends))
        others = []
        for (_, w), (st, c, _) in zip(self.blends, outs):
            try:
                r = json.loads(c) if st == 200 else None
                if isinstance(r, dict) and isinstance(r.get("answers"), dict):
                    others.append((w, r.get("model"), r["answers"]))
            except ValueError:
                continue
        for q in want:
            a = answers[q]
            keys = list(a["probabilities"])
            members = [(self.blend_weight, resp.get("model"), a)] + [
                (w, m, x[q]) for w, m, x in others if isinstance(x.get(q), dict) and x[q].get("type") == a["type"]
                and isinstance(x[q].get("probabilities"), dict) and list(x[q]["probabilities"]) == keys]
            members = [m for m in members if m[0] > 0]
            if len(members) > 1:
                answers[q] = blend_answer(a, members)
            elif members:  # only one model answered: its answer as it is
                a["blend"] = {"models": [members[0][1]], "weights": [members[0][0]]}
            self.blended[len(members)] = self.blended.get(len(members), 0) + 1
        return json.dumps(resp, ensure_ascii=False).encode()

    async def _cascade(self, path, body, content, headers):
        """The answers of the first replica (level 1), with those under level k's threshold asked again of level k + 1,
        level by level. A level that is down or does not answer 200 leaves the previous answers (and ends the chain).
        With one level the response is as before; with more, every answer also gives its level and model."""
        try:
            req, resp = json.loads(body), json.loads(content)
            answers = resp["answers"]
        except (ValueError, KeyError, TypeError):
            return content
        named = len(self.levels) > 1
        level = {}
        for q, a in answers.items():
            if isinstance(a, dict):
                a["answered_by"] = "first"
                if named:
                    a["level"], a["model"] = 1, resp.get("model")
                level[q] = 1
        done = []
        for k, (pool, below) in enumerate(self.levels, start=1):
            low = [q for q, a in answers.items() if q in (req.get("questions") or {}) and level.get(q) == k
                   and (top_probability(a) if isinstance(a, dict) else None) is not None and top_probability(a) < below]
            if not low:
                break
            sub = dict(req, questions={q: req["questions"][q] for q in low})
            st, c2, _ = await self._send(pool, path, json.dumps(sub).encode(), headers)
            if st != 200:
                break
            r2 = json.loads(c2)
            for q, a in (r2.get("answers") or {}).items():
                if q in low and isinstance(a, dict):
                    a["answered_by"] = "escalated"
                    if named:
                        a["level"], a["model"] = k + 1, r2.get("model")
                    answers[q] = a
                    level[q] = k + 1
                    if k == 1:
                        done.append(q)
        self.answered["escalated"] += len(done)
        self.answered["first"] += sum(1 for a in answers.values() if isinstance(a, dict)) - len(done)
        for k in level.values():
            self.by_level[k] += 1
        below = [b for _, b in self.levels] if named else self.levels[0][1]
        resp["cascade"] = {"below": below, "escalated": done}
        return json.dumps(resp, ensure_ascii=False).encode()

    async def _send(self, pool, path, body, headers):
        r, how = self.choose(state_key(body), pool)
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
             [({"replica": r}, n) for r, n in self.failed.items()])]
            + ([("basal_router_answers_total", "answers by the first replica / escalated (--escalate)", "counter",
                 [({"answered_by": k}, n) for k, n in self.answered.items()])] if self.levels else [])
            + ([("basal_router_level_answers_total", "answers by cascade level (1: the first replica)", "counter",
                 [({"level": str(k)}, n) for k, n in self.by_level.items()])] if len(self.levels) > 1 else [])
            + ([("basal_router_blend_answers_total", "answers blended (--blend), by the number of models in the blend",
                 "counter", [({"models": str(k)}, n) for k, n in sorted(self.blended.items())])] if self.blends else [])
            + ([("basal_router_script_total", "requests by the script of their state (--route)", "counter",
                 [({"script": k}, n) for k, n in sorted(self.scripts.items())])] if self.scripts else []))


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


def levels(escalate, below):
    """The cascade levels of --escalate URL ... (repeated: levels 2, 3, ...) and --escalate-below P (one per
    --escalate, in order; none: 0.8 for each) -> [(URLs, threshold), ...]; [] without escalation replicas."""
    if not any(escalate):
        return []
    if not all(escalate):
        raise SystemExit("basal-router: --escalate without URLs among several --escalate levels")
    if below and len(below) != len(escalate):
        raise SystemExit(f"basal-router: {len(escalate)} --escalate levels and {len(below)} --escalate-below "
                         "thresholds (give one per level, in the same order, or none for 0.8)")
    return [(list(u), b) for u, b in zip(escalate, below or [0.8] * len(escalate))]


def blends(blend, weights, cascade):
    """The blend of --blend URL ... (repeated: one model each) and --blend-weights W0 W1 ... (W0 for --replicas, then
    one per --blend) -> (first replica's weight, [(URLs, weight), ...]); (1.0, []) without --blend."""
    if not blend:
        if weights is not None:
            raise SystemExit("basal-router: --blend-weights without --blend")
        return 1.0, []
    if cascade:
        raise SystemExit("basal-router: --blend and --escalate do not go together")
    if not all(blend):
        raise SystemExit("basal-router: --blend without URLs")
    if weights is None or len(weights) != 1 + len(blend):
        raise SystemExit(f"basal-router: --blend-weights needs {1 + len(blend)} weights (--replicas first, then one "
                         "per --blend, in order)")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise SystemExit("basal-router: --blend-weights must be finite, >= 0 and not all 0")
    return weights[0], [(list(u), w) for u, w in zip(blend, weights[1:])]


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
    ap.add_argument("--escalate", nargs="*", action="append", default=[],
                    help="escalation replicas (a bigger model) for the cascade; repeat for further levels")
    ap.add_argument("--escalate-below", dest="escalate_below", type=float, action="append", default=[],
                    help="--escalate: questions whose top probability is under this are asked again there (default "
                         "0.8); one per --escalate, in the same order")
    ap.add_argument("--route", action="append", default=[], help="SCRIPT=URL[,URL]: states in that script go there")
    ap.add_argument("--blend", nargs="+", action="append", default=[],
                    help="replicas of one more model whose probabilities are blended in; repeat for each model")
    ap.add_argument("--blend-weights", dest="blend_weights", nargs="+", type=float, default=None,
                    help="--blend: the weight of --replicas, then of each --blend, in order")
    a = ap.parse_args(argv)
    cascade = levels(a.escalate, a.escalate_below)
    first, blend = blends(a.blend, a.blend_weights, cascade)
    procs, urls = ([], list(a.replicas))
    if a.gpus:
        procs, more = spawn(a.gpus.split(","), a.replica_port, serve_args)
        urls += more
    replicas = Replicas(procs)
    atexit.register(replicas.stop)  # (from here: a SystemExit below ends the replicas too)
    if not urls:
        raise SystemExit("basal-router: no replicas (--replicas URL ... or --gpus 0,1 -- <basal-serve arguments>)")
    import uvicorn
    print(f"basal-router: {len(urls)} replicas, port {a.port}", flush=True)
    routes = {}
    for x in a.route:
        k, _, v = x.partition("=")
        if k not in SCRIPTS + ("other",) or not v:
            raise SystemExit(f"basal-router: --route {x!r}: SCRIPT=URL[,URL], SCRIPT one of {', '.join(SCRIPTS)}, other")
        routes[k] = v.split(",")
    router = Router(urls, factor=a.load_factor, levels=cascade, routes=routes, blend=blend, blend_weight=first)

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
