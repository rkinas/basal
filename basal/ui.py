"""basal-serve --ui on (2.5): a live page of the server at GET /ui (status, traffic since start, state cache, telemetry,
configuration, recent requests), read from GET /ui/state.

The page (ui.html and its fonts, shipped in the package; no build step, no CDN) is open, so it can ask for the API key;
GET /ui/state needs a key when --api-keys is set. A decision request pays one append to a bounded ring: its metadata
(end time, endpoint, status, seconds, questions, options; never the request's text). Everything else (totals since
start, percentiles, rates, telemetry, configuration) is computed when GET /ui/state is read, at most once a second
however many pages poll it, and lives in memory only: every start begins at zero. GPU telemetry (NVML, from
nvidia-ml-py) and host telemetry (psutil) are read when installed, once a second in a side thread that runs only while a
page polled in the last 10 seconds; never a torch.cuda call. With --ui off (the default) none of this module is
imported and basal-serve handles requests as without it.

--ui full also keeps references to the last RECENT requests' parsed bodies and their answers (the bytes already sent),
no copy and no serialisation on the request path; GET /ui/request/<id> serialises one of them when the page asks
(behind the same key check as /ui/state; with --api-keys, only for the key that made the request or an admin key: the
keys are clients, who do not read each other's documents). They live in memory only, like everything here."""
import bisect
import itertools
import json
import math
import os
import sys
import threading
import time
from collections import deque
from importlib import metadata
from operator import itemgetter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).parent
RING = 16384  # requests kept (recent rows, the last minute's rates and percentiles; folded into the totals at a read)
FOLD = 2048  # requests folded into the totals at one read at most (a read's work stays bounded; the rest at the next)
RECENT = 200  # rows of the recent-requests table
MEMO_S = 1.0  # GET /ui/state is built at most once per MEMO_S, however many pages poll it
IDLE_S = 10.0  # telemetry stops sampling this long after the last GET /ui/state
DECISIONS = ("/v1/systemone", "/v1/basal", "/v1/decisions")  # the routes whose requests are recorded
FONTS = ("dm-sans-latin.woff2", "dm-sans-latin-ext.woff2", "space-grotesk-latin.woff2",
         "space-grotesk-latin-ext.woff2", "OFL-DMSans.txt", "OFL-SpaceGrotesk.txt")
SECRET = {"key", "keys", "token", "secret", "password", "passwd", "auth", "credentials"}  # words of a secret's name
PACKAGES = ("torch", "transformers", "vllm", "sglang", "mlx", "mlx-lm", "llama-cpp-python", "torchao", "triton",
            "flash-attn", "starlette", "uvicorn", "nvidia-ml-py", "psutil")
END = itemgetter(0)  # a ring row's end time
HEADERS = {  # the page: nothing from outside this server, never framed, no referrer
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                               "font-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
                               "base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache"}


def size(body):
    """(questions, options) of a decision request body for the recent-requests table: the options of a question are its
    criteria, choices, levels or tools (2 for a yes / no question without them). Never raises: (None, None) for a body
    it cannot read (not parsed, not JSON)."""
    try:
        qs = body["questions"]
        n = k = 0
        for q in qs.values() if isinstance(qs, dict) else qs:  # /v1/systemone, /v1/basal: a dict; /v1/decisions: a list
            c = q.get("criteria") or q.get("choices") or q.get("levels") or q.get("tools")
            n, k = n + 1, k + (len(c) if c else 2)
        return n, k
    except Exception:  # noqa: BLE001
        return None, None


class Hist:
    """Latencies (ms) in log-spaced bins 1% wide: percentiles since start in bounded memory, within 1%."""
    STEP = math.log(1.01)

    def __init__(self):
        self.bins, self.n, self.sum, self.max = {}, 0, 0.0, 0.0

    @classmethod
    def bin(cls, ms):
        return math.floor(math.log(max(ms, 1e-3)) / cls.STEP)

    def add(self, ms, i=None):
        i = self.bin(ms) if i is None else i
        self.bins[i] = self.bins.get(i, 0) + 1
        self.n, self.sum, self.max = self.n + 1, self.sum + ms, max(self.max, ms)

    def quantiles(self, qs=(0.5, 0.95, 0.99)):
        """Nearest-rank quantiles: the geometric middle of the bin that holds each rank (at most the maximum)."""
        if not self.n:
            return [None] * len(qs)
        ranks, out, acc = [max(1, math.ceil(q * self.n)) for q in qs], [None] * len(qs), 0
        for i, c in sorted(self.bins.items()):
            acc += c
            for j, r in enumerate(ranks):
                if out[j] is None and acc >= r:
                    out[j] = min(math.exp((i + 0.5) * self.STEP), self.max)
            if out[-1] is not None:
                break
        return out


def exact(values, qs=(0.5, 0.95, 0.99)):
    """Nearest-rank quantiles of a list of values."""
    if not values:
        return [None] * len(qs)
    v = sorted(values)
    return [v[max(1, math.ceil(q * len(v))) - 1] for q in qs]


def redact(name, value):
    """An option's value for the configuration view: secrets (an option named key, token, secret, password, auth or
    credentials, and the --api-keys file) as "(set)", URLs without user, password, query or fragment."""
    if value is None or isinstance(value, bool):
        return value
    if name == "api_keys" or SECRET & set(name.lower().split("_")):
        return "(set)"
    if isinstance(value, (list, tuple)):
        return [redact(name, v) for v in value]
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else str(value)
    value = str(value)
    if "://" in value:
        try:
            u = urlsplit(value)
            return urlunsplit((u.scheme, u.hostname + (f":{u.port}" if u.port else "") if u.hostname else "", u.path,
                               "", ""))
        except ValueError:
            return "(set)"
    return value


def versions():
    """Installed versions of the packages basal can serve with (metadata only: nothing is imported)."""
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    torch = sys.modules.get("torch")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)  # the build's CUDA (a constant, not a device call)
    if cuda:
        out["CUDA (torch build)"] = cuda
    return out


def revision(srv, a):
    """--revision, else the commit of a Hugging Face cache snapshot the model was resolved to, else None."""
    if a.revision:
        return a.revision
    md = getattr(srv, "md", None)
    return md.name if md is not None and md.parent.name == "snapshots" else None


def facts(srv, a, mode):
    """What the options resolved to at start-up: calibration, prepared start-up, compile, attention, prompt, packages."""
    from .server import MODES
    kind, quant, comp = MODES.get(mode, (None, None, False))
    b = srv.backend
    cal = getattr(srv, "cal_file", None)
    fitted = (getattr(srv, "cal", None) or {}).get("fitted_on") or {}
    prep = getattr(srv, "prepared", None)
    ctx = getattr(srv, "context", None)
    out = [
        ["calibration", "off" if a.no_calibration else None if cal is None else cal.name if cal.exists() else
         "none (no file found)"],
        ["calibration fitted on", " @ ".join(str(fitted[k])[:12] if k == "revision" else str(fitted[k])
                                           for k in ("model", "revision") if fitted.get(k)) or None],
        ["calibration warning", getattr(srv, "cal_mismatch", None) or getattr(srv, "cal_unvalidated", None)],
        ["prepared start-up", None if prep is None else
         f"loaded ({prep['inductor']} compiled kernels)" if isinstance(prep.get("inductor"), int) else "loaded"],
        ["engine", f"{type(b).__name__} ({kind})" if kind else type(b).__name__],
        ["quantisation", quant or getattr(a, "quant", None)],
        ["compile", ("torch.compile" + (" per layer" if getattr(a, "compile_scope", "model") == "layer" else "")
                     + " + CUDA graphs") if comp else "off"],
        ["attention", getattr(b, "attn", None)],
        ["compact rows", getattr(b, "compact_kernel", None)],
        ["convolution", getattr(b, "conv_kernel", None)],
        ["vLLM preset", getattr(b, "preset", None)],
        ["device", str(getattr(b, "dev", "")) or None],
        ["prompt format", getattr(srv, "fmt", None)],
        ["layout", getattr(srv, "layout", None)],
        ["option orders", getattr(srv, "orders", None)],
        ["two-way", getattr(srv, "two_way", None)],
        ["state once, ask many", getattr(srv, "soam", None)],
        ["context limit", f"{ctx[0]} tokens ({ctx[1]})" if ctx else None],
        ["context overflow", getattr(srv, "overflow", None)],
        ["state cache", (f"{a.state_cache_gb:g} GB, {a.state_cache_bits}-bit" if getattr(a, "state_cache_gb", 0)
                         else "off") + (f", disk {a.state_cache_disk_gb:g} GB at {a.state_cache_dir}"
                                        if getattr(a, "state_cache_dir", None) else "")],
        ["max batch", getattr(srv, "max_batch", None)],
        ["python", sys.version.split()[0]],
        ["platform", f"{sys.platform} {os.uname().machine}" if hasattr(os, "uname") else sys.platform],
    ]
    out += [[name, v] for name, v in versions().items()]
    return [[k, v] for k, v in out if v not in (None, "")]


class Telemetry:
    """GPU (NVML) and host (psutil) readings for the page, sampled every `every` seconds in a side thread that runs only
    while a page polled GET /ui/state in the last `idle` seconds. Never a torch.cuda call. A missing library or a
    machine without NVIDIA's driver leaves its part empty with a note."""

    def __init__(self, idle=IDLE_S, every=1.0):
        self.idle, self.every = idle, every
        self.viewed_at, self.thread, self.opened = 0.0, None, False
        self.nvml = self.ps = self.proc = None
        self.handles = []
        self.gpu_note = self.host_note = None
        self.sample = {"gpus": [], "gpu_note": None, "host": {}, "host_note": None, "at": None}

    def view(self):
        """A page read the state: sample (again) for the next `idle` seconds."""
        self.viewed_at = time.monotonic()
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name="basal-ui-telemetry", daemon=True)
            self.thread.start()

    def _run(self):
        if not self.opened:
            self.open()
        while time.monotonic() - self.viewed_at < self.idle:
            self.sample = self.read()
            time.sleep(self.every)

    def open(self):
        self.opened = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml  # (CUDA index, handle, NVML index, PCI bus id): the last two name the physical GPU
            self.handles = [(j, h, k, pci_id(pynvml, h)) for j, h, k in visible(pynvml)]
        except ImportError:
            self.gpu_note = "GPU telemetry needs nvidia-ml-py (pip install nvidia-ml-py)"
        except Exception as e:  # noqa: BLE001  (no NVIDIA driver: Apple, AMD, CPU)
            self.gpu_note = f"no NVIDIA GPU telemetry on this machine ({type(e).__name__})"
        try:
            import psutil
            self.ps, self.proc = psutil, psutil.Process()
            psutil.cpu_percent(None)  # the first readings start the intervals
            self.proc.cpu_percent(None)
        except ImportError:
            self.host_note = "CPU and memory need psutil (pip install psutil)"

    def read(self):
        gpus = [dict(gpu(self.nvml, j, h), nvml_index=k, pci=pci) for j, h, k, pci in self.handles] \
            if self.nvml is not None else []
        host = {"cpus": os.cpu_count(), "load": _try(lambda: [round(x, 2) for x in os.getloadavg()])}
        if self.ps is not None:
            vm = self.ps.virtual_memory()
            mem = _try(self.proc.memory_info)
            host.update(cpu=self.ps.cpu_percent(None), ram_used=vm.total - vm.available, ram_total=vm.total,
                        rss=mem.rss if mem else None, proc_cpu=_try(lambda: self.proc.cpu_percent(None)),
                        threads=_try(self.proc.num_threads))
        return {"gpus": gpus, "gpu_note": self.gpu_note, "host": host, "host_note": self.host_note,
                "at": round(time.time(), 3)}


def _ms(seconds):
    return None if seconds is None else round(seconds * 1000, 3)


def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def visible(nvml):
    """[(CUDA index, NVML handle, NVML index)] of the GPUs this process sees (CUDA_VISIBLE_DEVICES as indices or UUIDs;
    every GPU without it, or when none of its entries is found). The NVML index names the physical GPU: a process pinned
    to one GPU sees it as cuda:0 whichever it is."""
    n = nvml.nvmlDeviceGetCount()
    ids = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    out = []
    for j, s in enumerate(ids):
        if s.startswith(("GPU-", "MIG-")):
            h = _try(lambda: nvml.nvmlDeviceGetHandleByUUID(s))
            k = None if h is None else _try(lambda: nvml.nvmlDeviceGetIndex(h))
        else:
            k = _try(lambda: int(s))
            h = None if k is None else _try(lambda: nvml.nvmlDeviceGetHandleByIndex(k))
        if h is not None:
            out.append((j, h, k))
    return out or [(i, nvml.nvmlDeviceGetHandleByIndex(i), i) for i in range(n)]


def pci_id(nvml, h):
    """A GPU's PCI bus id, short: domain (last 4 digits):bus:device, e.g. 0000:61:00 (None when NVML has none)."""
    bus = _try(lambda: nvml.nvmlDeviceGetPciInfo(h).busId)
    bus = bus.decode() if isinstance(bus, bytes) else bus
    parts = bus.split(":") if isinstance(bus, str) else []
    return f"{parts[0][-4:]}:{parts[1]}:{parts[2].split('.')[0]}" if len(parts) == 3 else bus


def gpu(nvml, i, h):
    """One GPU's reading: utilisation (%), memory (bytes; this process's part when NVML reports it), temperature (C),
    power and its limit (W), SM clock (MHz)."""
    name = _try(lambda: nvml.nvmlDeviceGetName(h))
    util = _try(lambda: nvml.nvmlDeviceGetUtilizationRates(h))
    mem = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(h))
    power = _try(lambda: nvml.nvmlDeviceGetPowerUsage(h))
    limit = _try(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(h))
    pid = os.getpid()
    own = _try(lambda: sum(p.usedGpuMemory or 0 for p in nvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid == pid))
    return {"index": i, "name": name.decode() if isinstance(name, bytes) else name,
            "util": getattr(util, "gpu", None), "mem_util": getattr(util, "memory", None),
            "mem_used": getattr(mem, "used", None), "mem_total": getattr(mem, "total", None), "mem_own": own or None,
            "temp": _try(lambda: nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)),
            "power": None if power is None else round(power / 1000, 1),
            "power_limit": None if limit is None else round(limit / 1000, 1),
            "sm_clock": _try(lambda: nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM))}


class UI:
    """The state behind the page of one basal-serve process. Built when the server is ready (start-up work such as
    reading package versions is done here, not at a request); http_app takes its routes, main its page."""

    def __init__(self, srv, a, mode, defaults=None, started=None, full=False):
        from . import __version__
        self.srv, self.a, self.mode, self.version = srv, a, mode, __version__
        self.ring = deque(maxlen=RING)  # (end perf_counter, endpoint, status, seconds, questions, options[, id])
        # --ui full: (id, parsed body (or its bytes when it was not JSON), answer bytes, ring row, the name of the API
        # key that made it or None) of the last RECENT
        self.bodies, self.ids = (deque(maxlen=RECENT), itertools.count(1)) if full else (None, None)
        self.wall0, self.perf0 = time.time(), time.perf_counter()
        self.started = started or self.wall0
        self.memo, self.memo_at = None, -math.inf
        self.folded, self.seen = self.perf0, 0  # end time of the last request in the totals, and their count
        self.total, self.endpoints, self.codes = Hist(), {}, {}
        self.cache = None  # the last state-cache snapshot read
        self.telemetry = Telemetry()
        defaults = defaults or {}
        self.config = {"options": [[k, redact(k, v), k in defaults and defaults[k] != v] for k, v in vars(a).items()],
                       "facts": facts(srv, a, mode)}

    def record(self, handler, endpoint):
        """handler with its request's metadata appended to the ring once it is answered: all the work --ui on adds to
        a request. The body was parsed by the handler (Starlette keeps it on the request); its text is not kept. --ui
        full: also a reference to that body and to the answer's bytes (no copy), under a request id."""
        ring, clock, bodies, ids = self.ring, time.perf_counter, self.bodies, self.ids

        async def recorded(request):
            t = clock()
            response = await handler(request)
            t1 = clock()
            ring.append((t1, endpoint, response.status_code, t1 - t, *size(getattr(request, "_json", None))))
            return response

        async def recorded_full(request):
            t = clock()
            response = await handler(request)
            t1 = clock()
            body, rid = getattr(request, "_json", None), next(ids)
            row = (t1, endpoint, response.status_code, t1 - t, *size(body), rid)
            ring.append(row)
            key = (getattr(request, "scope", None) or {}).get("basal.api_key")  # (set by protect with --api-keys)
            bodies.append((rid, body if body is not None else getattr(request, "_body", None),
                           getattr(response, "body", None), row, key.get("name") if key else None))
            return response
        out = recorded if bodies is None else recorded_full
        out.__name__ = f"recorded_{handler.__name__}"
        return out

    def routes(self, routes, counts):
        """http_app's routes with the decision routes recorded, and GET /ui/state. counts: http_app's totals (the
        ones GET /metrics reports)."""
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        async def state(_):
            return Response(self.state(counts), media_type="application/json", headers={"Cache-Control": "no-store"})

        async def one(request):
            data = self.request_json(request.path_params["rid"], request.scope.get("basal.api_key"))
            if data is False:
                return JSONResponse({"error": "this request was made with another API key (an admin key reads every "
                                     "request)", "code": "FORBIDDEN"}, status_code=403)
            if data is None:
                return JSONResponse({"error": f"not kept: --ui full keeps the last {RECENT} requests",
                                     "code": "NOT_FOUND"}, status_code=404)
            return Response(data, media_type="application/json", headers={"Cache-Control": "no-store"})
        out = [Route(r.path, self.record(r.endpoint, r.path), methods=sorted(r.methods))
               if isinstance(r, Route) and r.path in DECISIONS else r for r in routes]
        return out + [Route("/ui/state", state)] + ([Route("/ui/request/{rid:int}", one)] if self.bodies is not None else [])

    def request_json(self, rid, key=None):
        """GET /ui/request/<id> (--ui full): the request's metadata, its body as parsed (a body that was not JSON as
        text) and the answer it got (the bytes sent, as JSON), or None when it is no longer kept. key: the caller's API
        key record (--api-keys; None without): False when it is neither the key that made the request nor an admin
        key."""
        kept = next((b for b in self.bodies if b[0] == rid), None)
        if kept is None:
            return None
        _, body, answer, row, owner = kept
        if key is not None and not key.get("admin", False) and key.get("name") != owner:
            return False
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "replace")
        meta = {"id": rid, "time": round(self.wall0 + row[0] - self.perf0, 3), "endpoint": row[1], "status": row[2],
                "ms": _ms(row[3]), "questions": row[4], "options": row[5]}
        try:
            answer = answer if answer is not None and json.loads(answer) is not None else b"null"
        except ValueError:
            answer = json.dumps(answer.decode("utf-8", "replace")).encode()
        return (json.dumps(meta)[:-1] + ', "request": ' + json.dumps(body, ensure_ascii=False) + ', "answer": ').encode() \
            + answer + b"}"

    def page(self, app):
        """app with GET /ui (the page) and its fonts in front of it, open (the page asks for the key when GET
        /ui/state needs one). The files are read once, here."""
        from starlette.responses import Response
        files = {"/ui": ((HERE / "ui.html").read_bytes(), "text/html; charset=utf-8", HEADERS)}
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

    def fold(self, rows):
        """Requests answered since the last read (rows: the ring's rows, oldest first) into the totals since start,
        the oldest first and at most FOLD of them (the rest at the next reads). A request that left the ring before it
        was folded (more than RING requests between two reads) is not in the totals: "seen" counts the ones that are."""
        new = rows[bisect.bisect_right(rows, self.folded, key=END):][:FOLD]
        if not new:
            return
        self.folded = new[-1][0]
        self.seen += len(new)
        bin_of, total, endpoints, codes = Hist.bin, self.total, self.endpoints, self.codes
        for _, ep, status, s, *_ in new:
            ms = s * 1000
            i = bin_of(ms)
            total.add(ms, i)
            e = endpoints.get(ep)
            if e is None:
                e = endpoints[ep] = {"requests": 0, "failed": 0, "hist": Hist()}
            e["requests"] += 1
            e["failed"] += status >= 400
            e["hist"].add(ms, i)
            codes[status] = codes.get(status, 0) + 1

    def state(self, counts):
        """GET /ui/state: the JSON snapshot, built at most once per MEMO_S."""
        self.telemetry.view()
        now = time.perf_counter()
        if now - self.memo_at < MEMO_S:
            return self.memo
        srv, a = self.srv, self.a
        rows = list(self.ring)  # one copy; the rows are in the order of their end times
        self.fold(rows)
        i60 = bisect.bisect_left(rows, now - 60, key=END)
        w60 = [r[3] for r in rows[i60:]]  # seconds of the requests answered in the last 60 and 10 s
        w10 = w60[bisect.bisect_left(rows, now - 10, lo=i60, key=END) - i60:]
        up = now - self.perf0
        q50, q95, q99 = self.total.quantiles()
        load = srv.load()
        cache = getattr(srv.backend, "state_cache", None)
        if cache is not None:
            self.cache = cache.stats(blocking=False) or self.cache  # never waits for the engine's lock
        d = {
            "now": round(time.time(), 3), "started": round(self.started, 3), "ready_at": round(self.wall0, 3),
            "uptime": round(up, 3),
            "status": {"state": "ready", "model": srv.name, "source": a.model, "revision": revision(srv, a),
                       "mode": self.mode, "version": self.version, "think": getattr(a, "think", "off"),
                       "running": load["running"], "queued": load["queued"],
                       "ui": "full" if self.bodies is not None else "on",
                       "auth": bool(getattr(a, "api_keys", None))},
            "traffic": {
                "requests": counts["requests"], "failed": counts["failed"], "seconds": round(counts["seconds"], 6),
                "seen": self.seen, "ring": RING,
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
            "cache": self.cache,
            "telemetry": self.telemetry.sample,
            "config": self.config,
            "recent": [[round(self.wall0 + r[0] - self.perf0, 3), r[1], r[2], _ms(r[3]), *r[4:]]
                       for r in reversed(rows[-RECENT:])],  # (--ui full: and the request id)
        }
        self.memo, self.memo_at = json.dumps(d, separators=(",", ":")).encode(), now
        return self.memo
