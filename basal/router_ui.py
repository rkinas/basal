"""basal-router --ui on (2.5): one live page for the router and every replica behind it, at GET /ui, read from
GET /ui/state.

The router's part (replicas, readiness, requests in flight, routing by state / by load, failures, the cascade, blend,
fallback and script counters, traffic through the router since start, the recent requests) comes from its own
counters and from a bounded ring: a decision request pays one append of its metadata (end time, endpoint, status,
seconds, the replica it went to and how it was chosen; never its text). Each replica's part is that replica's own
GET /ui/state (basal-serve --ui on), read when GET /ui/state of the router is read, at most once a second however
many pages poll it, with the client's API key headers passed on as for requests. A replica without --ui (404), one
that wants a key (401), or one that is not ready shows what the router knows of it. Nothing is read from the replicas
while no page is open, and everything lives in memory only: every start begins at zero. With --ui off (the default)
this module is not imported and the router handles requests as without it."""
import asyncio
import bisect
import hashlib
import json
import sys
import time
from collections import deque
from contextvars import ContextVar
from importlib import metadata

from .router import POLL_S, key_headers
from .ui import END, FOLD, FONTS, HEADERS, HERE, MEMO_S, RECENT, RING, Hist, _ms, exact, redact

DECISIONS = ("/v1/systemone", "/v1/basal")  # the router's routes whose requests are recorded
REPLICA_TIMEOUT = 2.0  # seconds a replica's GET /ui/state may take; then the page shows what the router knows
MEMO_KEYS = 8  # snapshots kept, one per set of key headers (a viewer's key never reads another key's snapshot)
KEEP = ("now", "started", "ready_at", "uptime", "status", "traffic", "cache", "telemetry")  # of a replica's state
PACKAGES = ("starlette", "uvicorn", "httpx", "uvloop", "httptools")
UI_BY_STATUS = {404: "off", 401: "key", 403: "key", 503: "not ready"}
PICK = ContextVar("basal_router_ui_pick", default=None)  # a recorded request's first (replica, how), set by choose


def slim(state):
    """The parts of a replica's GET /ui/state the router's page shows: status, traffic, state cache, telemetry and the
    resolved facts of its configuration (not its option list nor its recent requests). None when it is not a state."""
    if not isinstance(state, dict) or not isinstance(state.get("status"), dict):
        return None
    out = {k: state.get(k) for k in KEEP}
    cfg = state.get("config")
    out["facts"] = cfg.get("facts") if isinstance(cfg, dict) else None
    return out


def shown(name, value):
    """An option's value as the page shows it: redact()'s, and for --route SCRIPT=URL[,URL] each URL redacted on its
    own (the whole value parses as no URL: its user, password and query would be shown)."""
    if name == "fallback_extra":
        return "(set)" if value else value
    if name == "route" and isinstance(value, list):
        return [x.partition("=")[0] + "=" + ",".join(redact("url", u) for u in x.partition("=")[2].split(","))
                if isinstance(x, str) and "=" in x else redact(name, x) for x in value]
    return redact(name, value)


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


class RouterUI:
    """The state behind the router's page. Built in main before the router serves; app takes its routes and page."""

    def __init__(self, router, a=None, defaults=None):
        from . import __version__
        self.router, self.version = router, __version__
        self.ring = deque(maxlen=RING)  # (end perf_counter, endpoint, status, seconds, replica URL or None, how or None)
        self.counts = [0, 0]  # requests answered and failed (status >= 400) since start, whatever the ring kept
        self.wall0, self.perf0 = time.time(), time.perf_counter()
        self.folded, self.seen = self.perf0, 0  # end time of the last request in the totals, and their count
        self.total, self.endpoints, self.codes, self.per = Hist(), {}, {}, {}
        self.memo, self.lock = {}, None  # {hash of the key headers: (perf_counter, bytes)}; asyncio.Lock at a read
        self.pools = {}
        for name, urls in router.pools.items():
            for u in urls:
                self.pools.setdefault(u, []).append(name)
        defaults = defaults or {}
        opts = vars(a) if a is not None else {}
        self.config = {"options": [[k, shown(k, v), k in defaults and defaults[k] != v] for k, v in opts.items()],
                       "facts": self.facts()}
        self.attach()

    def attach(self):
        """The router's choose, wrapped (on this router only) to tell a recorded request which replica it went to:
        the first choice made for it (the replica its pool and state give; cascade / blend calls come after)."""
        choose = self.router.choose

        def chosen(key, pool="main"):
            out = choose(key, pool)
            box = PICK.get()
            if box is not None and not box:
                box.extend(out)
            return out
        self.router.choose = chosen

    def facts(self):
        rt = self.router
        pools = ", ".join(f"{name} ({len(u)})" for name, u in rt.pools.items() if u)
        out = [["replicas", len(rt.replicas)], ["pools", pools],
               ["routing", "consistent hash of the state, bounded load; no state: the less loaded of two"],
               ["load factor", rt.factor], ["readiness", f"GET /health every {POLL_S:g} s"],
               ["cascade levels", len(rt.levels) + 1 if rt.levels else None],
               ["blend models", len(rt.blends) + 1 if rt.blends else None],
               ["fallback model", rt.fallback["model"] if rt.fallback else None],
               ["python", sys.version.split()[0]]]
        out += [[name, v] for name, v in versions().items()]
        return [[k, v] for k, v in out if v not in (None, "")]

    def record(self, handler, endpoint):
        """handler with its request's metadata appended to the ring once it is answered: all the work --ui on adds to
        a request (the replica comes from choose, through a context variable). The body is not read here."""
        ring, clock, counts = self.ring, time.perf_counter, self.counts

        async def recorded(request):
            box = []
            token = PICK.set(box)
            t = clock()
            try:
                response = await handler(request)
            finally:
                PICK.reset(token)
            t1 = clock()
            status = response.status_code
            counts[0] += 1
            counts[1] += status >= 400
            ring.append((t1, endpoint, status, t1 - t, *(box or (None, None))))
            return response
        recorded.__name__ = f"recorded_{handler.__name__}"
        return recorded

    def routes(self, routes):
        """The router's routes with the decision routes recorded, and GET /ui/state."""
        from starlette.responses import Response
        from starlette.routing import Route

        async def state(request):
            body = await self.state(key_headers(request.headers.raw))
            return Response(body, media_type="application/json", headers={"Cache-Control": "no-store"})
        out = [Route(r.path, self.record(r.endpoint, r.path), methods=sorted(r.methods))
               if isinstance(r, Route) and r.path in DECISIONS else r for r in routes]
        return out + [Route("/ui/state", state)]

    def page(self, app):
        """app with GET /ui (the page) and its fonts (basal-serve's) in front of it. The files are read once, here."""
        from starlette.responses import Response
        files = {"/ui": ((HERE / "router_ui.html").read_bytes(), "text/html; charset=utf-8", HEADERS)}
        for name in FONTS:
            kind = "font/woff2" if name.endswith(".woff2") else "text/plain; charset=utf-8"
            files[f"/ui/fonts/{name}"] = ((HERE / "fonts" / name).read_bytes(), kind,
                                          {"Cache-Control": "max-age=86400", "X-Content-Type-Options": "nosniff"})

        async def with_page(scope, receive, send):
            f = files.get(scope["path"]) if scope["type"] == "http" and scope["method"] == "GET" else None
            if f is None:
                await app(scope, receive, send)
                return
            await Response(f[0], media_type=f[1], headers=f[2])(scope, receive, send)
        return with_page

    async def state(self, keys):
        """GET /ui/state: the JSON snapshot, built at most once per MEMO_S for each set of key headers (the replicas'
        parts depend on the key the page sent)."""
        mk = hashlib.sha256(repr(sorted(keys.items())).encode()).digest()
        hit = self.memo.get(mk)
        if hit is not None and time.perf_counter() - hit[0] < MEMO_S:
            return hit[1]
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:  # (pages polling at once: one build, the others take it)
            hit = self.memo.get(mk)
            if hit is not None and time.perf_counter() - hit[0] < MEMO_S:
                return hit[1]
            reps = await asyncio.gather(*(self.replica(r, keys) for r in self.router.replicas))
            data = self.build(reps)
            if len(self.memo) >= MEMO_KEYS:
                self.memo.clear()
            self.memo[mk] = (time.perf_counter(), data)
            return data

    async def replica(self, url, keys):
        """(a replica's slimmed GET /ui/state or None, what its UI said: on, off, key, not ready, unreachable, error)."""
        if url not in self.router.ready:
            return None, "not ready", None
        try:
            r = await self.router.client.get(url + "/ui/state", headers=keys, timeout=REPLICA_TIMEOUT)
        except Exception as e:  # noqa: BLE001  (down, refused, slower than REPLICA_TIMEOUT)
            return None, "unreachable", type(e).__name__
        if r.status_code != 200:
            return None, UI_BY_STATUS.get(r.status_code, "error"), r.status_code
        try:
            s = slim(json.loads(r.content))
        except ValueError:
            s = None
        return (s, "on", None) if s is not None else (None, "error", "not a basal-serve state")

    def fold(self, rows):
        """Requests answered since the last read (rows: the ring's, oldest first) into the totals since start, at most
        FOLD of them per read (the rest at the next); "seen" counts the ones in the totals."""
        new = rows[bisect.bisect_right(rows, self.folded, key=END):][:FOLD]
        if not new:
            return
        self.folded = new[-1][0]
        self.seen += len(new)
        bin_of, total, endpoints, codes, per = Hist.bin, self.total, self.endpoints, self.codes, self.per
        for _, ep, status, s, rep, _ in new:
            ms = s * 1000
            i = bin_of(ms)
            total.add(ms, i)
            codes[status] = codes.get(status, 0) + 1
            for name, table in ((ep, endpoints), (rep, per)):
                if name is None:
                    continue
                e = table.get(name)
                if e is None:
                    e = table[name] = {"requests": 0, "failed": 0, "hist": Hist()}
                e["requests"] += 1
                e["failed"] += status >= 400
                e["hist"].add(ms, i)

    def build(self, reps):
        rt = self.router
        now = time.perf_counter()
        rows = list(self.ring)  # one copy; the rows are in the order of their end times
        self.fold(rows)
        i60 = bisect.bisect_left(rows, now - 60, key=END)
        w60 = [r[3] for r in rows[i60:]]  # seconds of the requests answered in the last 60 and 10 s
        w10 = w60[bisect.bisect_left(rows, now - 10, lo=i60, key=END) - i60:]
        up = now - self.perf0
        q50, q95, q99 = self.total.quantiles()
        index = {u: k for k, u in enumerate(rt.replicas)}
        replicas = []
        for k, (url, (state, ui, detail)) in enumerate(zip(rt.replicas, reps)):
            e = self.per.get(url)
            p50, p95 = e["hist"].quantiles((0.5, 0.95)) if e else (None, None)
            replicas.append({
                "index": k, "url": redact("url", url), "pools": self.pools.get(url, []), "ready": url in rt.ready,
                "in_flight": rt.load.get(url, 0), "routed": {h: rt.routed.get((url, h), 0) for h in ("state", "load")},
                "failed": rt.failed.get(url, 0), "requests": e["requests"] if e else 0, "p50": p50, "p95": p95,
                "ui": ui, "detail": detail, "state": state})
        features = {}
        if rt.levels:
            features["cascade"] = {"below": [b for _, b in rt.levels], "answered": dict(rt.answered),
                                   "levels": {str(k): n for k, n in rt.by_level.items()}}
        if rt.blends:
            features["blend"] = {"weights": [rt.blend_weight] + [w for _, w in rt.blends],
                                 "blended": {str(k): n for k, n in sorted(rt.blended.items())}}
        if rt.fallback:
            features["fallback"] = {"model": rt.fallback["model"], "below": rt.fallback["below"],
                                    "results": dict(rt.fallback_counts)}
        scripts = [name[7:] for name in rt.pools if name.startswith("script:") and rt.pools[name]]
        if scripts:
            features["route"] = {"scripts": scripts, "requests": dict(sorted(rt.scripts.items()))}
        d = {
            "now": round(time.time(), 3), "started": round(self.wall0, 3), "uptime": round(up, 3),
            "version": self.version,
            "router": {"replicas": len(rt.replicas), "ready": len(rt.ready), "in_flight": sum(rt.load.values()),
                       "load_factor": rt.factor},
            "traffic": {
                "requests": self.counts[0], "failed": self.counts[1], "seen": self.seen, "ring": RING,
                "rate10": round(len(w10) / min(10.0, up), 3) if up > 0 else 0,
                "rate60": round(len(w60) / min(60.0, up), 3) if up > 0 else 0,
                "latency": {"p50": q50, "p95": q95, "p99": q99, "max": self.total.max or None,
                            "mean": self.total.sum / self.total.n if self.total.n else None},
                "w10": dict(zip(("p50", "p95", "p99"), map(_ms, exact(w10))), n=len(w10)),
                "w60": dict(zip(("p50", "p95", "p99"), map(_ms, exact(w60))), n=len(w60)),
                "endpoints": [{"endpoint": ep, "requests": e["requests"], "failed": e["failed"],
                               **dict(zip(("p50", "p95"), e["hist"].quantiles((0.5, 0.95)))),
                               "mean": e["hist"].sum / e["hist"].n} for ep, e in sorted(self.endpoints.items())],
                "codes": {str(k): v for k, v in sorted(self.codes.items())}},
            "replicas": replicas,
            "features": features,
            "config": self.config,
            "recent": [[round(self.wall0 + r[0] - self.perf0, 3), r[1], r[2], _ms(r[3]), index.get(r[4]), r[5]]
                       for r in reversed(rows[-RECENT:])],
        }
        return json.dumps(d, separators=(",", ":")).encode()
