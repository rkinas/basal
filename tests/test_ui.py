"""basal-serve --ui (2.5): off (the default) adds no route and no work to a request; on serves the page (open) and its
state (behind --api-keys), counts from zero at every start, keeps no request text and no secret, samples telemetry only
while a page polls, and ships its page and fonts for offline use. Through the app of basal-serve (http_app) with its
batching worker and a stub engine."""
import ast
import asyncio
import hashlib
import json
import math
import random
import re
import sys
import time
import types
from pathlib import Path

import httpx

import basal.server as server_module
import basal.ui as ui_module
from basal.auth import ApiKeys, protect, request_ids
from basal.server import http_app, parser
from basal.ui import UI, Hist, Telemetry, exact, redact, size
from tests.test_http_status import BODY, AppServer, Engine

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ["/v1/systemone", "/v1/basal", "/v1/decisions", "/v1/models", "/health", "/metrics"]
SECRET = "bsl_ui-test-secret"


def app_of(args=(), keys=None, ui=True, full=False):
    """(stub server, UI or None, the ASGI app as basal-serve's main builds it)."""
    a = parser().parse_args([*(["--ui", "full" if full else "on"] if ui else []), *args])
    srv = AppServer(Engine())
    u = UI(srv, a, "fast", vars(parser().parse_args([])), full=full) if ui else None
    app = protect(http_app(srv, "fast", ui=u), ApiKeys(keys) if keys else None, ())
    return srv, u, request_ids(u.page(app) if u is not None else app)


async def on(srv, app, fn):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
            return await fn(c)
    finally:
        worker.cancel()


def keyfile(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": "viewer", "sha256": hashlib.sha256(SECRET.encode()).hexdigest()}]}))
    return path


def fresh(u):
    """The next read builds the state again (it is memoised for a second)."""
    u.memo_at = -math.inf


def test_off_by_default_no_route_and_handlers_as_before():
    assert parser().parse_args([]).ui == "off"
    srv, _, app = app_of(ui=False)
    routes = http_app(srv, "fast").routes
    assert [r.path for r in routes] == ROUTES
    assert all(r.endpoint.__qualname__.startswith("http_app.<locals>.") for r in routes)  # not wrapped

    async def go(c):
        return [(await c.get(p)).status_code for p in ("/ui", "/ui/state", "/ui/fonts/dm-sans-latin.woff2")]
    assert asyncio.run(on(srv, app, go)) == [404, 404, 404]


def test_server_imports_the_ui_only_when_it_is_on():
    """basal.ui is imported only in main, after --ui on: off runs none of its code."""
    tree = ast.parse(Path(server_module.__file__).read_text(encoding="utf-8"))
    owners = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                owners.setdefault(id(node), fn.name)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "ui"]
    assert imports and all(owners.get(id(n)) == "main" for n in imports)


def test_on_wraps_only_the_decision_routes_and_adds_the_state():
    _, u, _ = app_of()
    routes = http_app(AppServer(Engine()), "fast", ui=u).routes
    assert [r.path for r in routes] == ROUTES + ["/ui/state"]
    wrapped = [r.path for r in routes if "recorded" in r.endpoint.__qualname__]
    assert wrapped == ["/v1/systemone", "/v1/basal", "/v1/decisions"]
    assert all(r.methods == {"POST"} for r in routes[:3])


def test_page_is_open_and_the_state_needs_a_key(tmp_path):
    srv, u, app = app_of(["--api-keys", str(keyfile(tmp_path))], keys=keyfile(tmp_path))

    async def go(c):
        page = await c.get("/ui")
        font = await c.get("/ui/fonts/dm-sans-latin.woff2")
        lic = await c.get("/ui/fonts/OFL-SpaceGrotesk.txt")
        no_key = await c.get("/ui/state")
        bad_key = await c.get("/ui/state", headers={"Authorization": "Bearer wrong"})
        key = await c.get("/ui/state", headers={"Authorization": f"Bearer {SECRET}"})
        decide = await c.post("/v1/systemone", json=BODY)
        other = [(await c.get(p)).status_code for p in ("/ui/fonts/../ui.py", "/ui/ui.html", "/ui/")]
        return page, font, lic, no_key, bad_key, key, decide, other
    page, font, lic, no_key, bad_key, key, decide, other = asyncio.run(on(srv, app, go))
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in page.headers["content-security-policy"] and b"API key required" in page.content
    assert font.status_code == 200 and font.headers["content-type"] == "font/woff2" and font.content[:4] == b"wOF2"
    assert lic.status_code == 200 and b"SIL Open Font License" in lic.content
    assert no_key.status_code == 401 and bad_key.status_code == 401
    assert key.status_code == 200 and key.json()["status"]["auth"] is True
    assert key.headers["cache-control"] == "no-store"
    assert decide.status_code == 401  # the API itself unchanged
    assert other == [401, 401, 401]  # nothing else under /ui is open
    assert len(u.ring) == 0  # a request refused by the key check is not recorded


def test_counters_start_at_zero_and_record_metadata_only():
    srv, u, app = app_of()
    decision = {"input": "Invoice 1240 PLN.", "questions": [
        {"type": "choice", "instructions": "Which?",
         "choices": [{"value": "a", "description": "Alpha"}, {"value": "b"}, {"value": "c", "description": "Gamma"}]},
        {"type": "predicate", "instructions": "Paid?"}]}

    async def go(c):
        first = (await c.get("/ui/state")).json()
        for _ in range(3):
            assert (await c.post("/v1/systemone", json=BODY)).status_code == 200
        assert (await c.post("/v1/basal", json=BODY)).status_code == 200
        assert (await c.post("/v1/decisions", json=decision)).status_code == 200
        assert (await c.post("/v1/systemone", json={"state": "Invoice 1240 PLN."})).status_code == 422
        assert (await c.post("/v1/systemone", content=b"{not json")).status_code == 422
        fresh(u)
        return first, (await c.get("/ui/state"))
    first, r = asyncio.run(on(srv, app, go))
    t = first["traffic"]
    assert (t["requests"], t["failed"], t["seen"], t["codes"], t["endpoints"]) == (0, 0, 0, {}, [])
    assert first["recent"] == [] and first["traffic"]["latency"]["p50"] is None
    assert first["status"]["state"] == "ready" and first["status"]["auth"] is False
    s = r.json()
    t = s["traffic"]
    assert t["requests"] == 7 and t["failed"] == 2  # as GET /metrics counts (a body that is not JSON too)
    assert t["seen"] == 7 and t["codes"] == {"200": 5, "422": 2}
    assert {e["endpoint"]: (e["requests"], e["failed"]) for e in t["endpoints"]} == {
        "/v1/basal": (1, 0), "/v1/decisions": (1, 0), "/v1/systemone": (5, 2)}
    rows = s["recent"]
    assert [row[1:3] for row in rows] == [["/v1/systemone", 422]] * 2 + [["/v1/decisions", 200], ["/v1/basal", 200]] + \
        [["/v1/systemone", 200]] * 3
    assert [row[4:] for row in rows] == [[None, None], [None, None], [2, 5], [1, 2], [1, 2], [1, 2], [1, 2]]
    assert all(row[3] > 0 and s["ready_at"] <= row[0] <= s["now"] for row in rows)
    assert t["latency"]["p50"] > 0 and t["w10"]["n"] == 7 and t["rate10"] > 0
    text = r.text
    # request text never reaches the state; words only: a number of the request (1240) can be a digit run of a timing
    for word in ("Invoice", "PLN", "Alpha", "Gamma", "Which?", "Paid?"):
        assert word not in text

    def strings(x):
        if isinstance(x, dict):
            x = list(x.values())
        return [y for v in x for y in strings(v)] if isinstance(x, list) else [x] if isinstance(x, str) else []
    assert not any("1240" in x for x in strings(s))  # nor in a string value (the state's numbers are timings)
    _, u2, _ = app_of()  # a new server starts at zero
    assert json.loads(u2.state({"requests": 0, "failed": 0, "seconds": 0.0}))["recent"] == []


def test_state_is_built_at_most_once_a_second(monkeypatch):
    _, u, _ = app_of()
    counts = {"requests": 0, "failed": 0, "seconds": 0.0}
    first = u.state(counts)
    assert u.state(counts) is first  # a second poller within the second gets the same bytes
    counts["requests"] = 5
    assert json.loads(u.state(counts))["traffic"]["requests"] == 0
    fresh(u)
    assert json.loads(u.state(counts))["traffic"]["requests"] == 5


def test_ring_is_bounded_and_folded_totals_say_what_they_saw(monkeypatch):
    monkeypatch.setattr(ui_module, "RING", 8)
    _, u, _ = app_of()

    async def handler(_):
        return types.SimpleNamespace(status_code=200)
    rec = u.record(handler, "/v1/systemone")
    request = types.SimpleNamespace(_json=BODY)
    counts = {"requests": 0, "failed": 0, "seconds": 0.0}

    async def burst(n):
        for _ in range(n):
            await rec(request)
        counts["requests"] += n
    asyncio.run(burst(5))
    assert json.loads(u.state(counts))["traffic"]["seen"] == 5
    asyncio.run(burst(20))  # more than the ring holds between two reads
    fresh(u)
    s = json.loads(u.state(counts))
    assert len(u.ring) == 8 and s["traffic"]["seen"] == 13 and s["traffic"]["requests"] == 25
    assert len(s["recent"]) == 8


def test_a_read_folds_a_bounded_number_of_requests_and_the_next_reads_catch_up(monkeypatch):
    monkeypatch.setattr(ui_module, "FOLD", 3)
    _, u, _ = app_of()

    async def handler(_):
        return types.SimpleNamespace(status_code=200)
    rec = u.record(handler, "/v1/basal")

    async def burst(n):
        for _ in range(n):
            await rec(types.SimpleNamespace(_json=BODY))
    asyncio.run(burst(10))
    counts = {"requests": 10, "failed": 0, "seconds": 0.0}
    seen = []
    for _ in range(5):
        fresh(u)
        seen.append(json.loads(u.state(counts))["traffic"]["seen"])
    assert seen == [3, 6, 9, 10, 10]
    assert u.endpoints["/v1/basal"]["requests"] == 10 and u.total.n == 10


def test_config_view_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_envsecret123")
    keys = keyfile(tmp_path)
    srv, u, app = app_of(["--api-keys", str(keys), "--ollama-url", "http://user:pw123@10.1.2.3:11434/api?token=tk999",
                          "--max-batch", "32"], keys=keys)

    async def go(c):
        return await c.get("/ui/state", headers={"X-API-Key": SECRET})
    text = asyncio.run(on(srv, app, go)).text
    options = {k: (v, changed) for k, v, changed in json.loads(text)["config"]["options"]}
    assert options["api_keys"] == ("(set)", True)
    assert options["ollama_url"] == ("http://10.1.2.3:11434/api", True)
    assert options["max_batch"] == (32, True) and options["port"] == (8000, False) and options["ui"] == ("on", True)
    for secret in (SECRET, hashlib.sha256(SECRET.encode()).hexdigest(), str(keys), "pw123", "tk999", "hf_envsecret123"):
        assert secret not in text
    facts = dict(json.loads(text)["config"]["facts"])
    assert facts["python"] == sys.version.split()[0] and "torch" in facts


def test_redact():
    assert redact("think_api_key", "abc") == "(set)" and redact("hf_token", "x") == "(set)"
    assert redact("long_tokens", 2048) == 2048 and redact("reread_max_tokens", 2048) == 2048
    assert redact("auto_band", [0.2, 0.8]) == [0.2, 0.8] and redact("x", float("nan")) == "nan"
    assert redact("think_url", "https://u:p@h/v1#frag") == "https://h/v1"
    m = redact("model", Path("/models/m"))
    assert isinstance(m, str) and Path(m).as_posix() == "/models/m" and redact("calibration", None) is None


def test_size_counts_questions_and_options():
    assert size(BODY) == (1, 2)
    assert size({"questions": {"a": {"type": "noul"}, "b": {"criteria": ["x", "y", "z"]},
                               "c": {"type": "tool", "tools": [{}, {}, {}, {}]}}}) == (3, 9)
    assert size({"questions": [{"type": "score", "levels": ["lo", "mid", "hi"]}, {"type": "predicate"}]}) == (2, 5)
    for body in (None, [], "text", {"questions": 3}, {"questions": [1]}):
        assert size(body) == (None, None)


def test_latency_percentiles_since_start_within_one_percent():
    rng = random.Random(0)
    values = [rng.lognormvariate(3, 0.8) for _ in range(20000)]
    h = Hist()
    for v in values:
        h.add(v)
    for got, want in zip(h.quantiles((0.5, 0.95, 0.99)), exact(values, (0.5, 0.95, 0.99))):
        assert abs(got - want) / want < 0.01
    assert h.quantiles((1.0,))[0] == max(values) and Hist().quantiles() == [None] * 3
    assert exact([3, 1, 2], (0.5,)) == [2]


def test_telemetry_without_libraries_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    t = Telemetry()
    t.open()
    s = t.read()
    assert s["gpus"] == [] and "nvidia-ml-py" in s["gpu_note"] and "psutil" in s["host_note"]
    assert s["host"]["cpus"] and "ram_total" not in s["host"]


def fake_nvml(seen):
    """A stand-in for nvidia-ml-py's pynvml: two GPUs; records every call."""
    def call(name, value):
        def f(*args):
            seen.append(name)
            return value
        return f
    m = types.ModuleType("pynvml")
    m.NVML_TEMPERATURE_GPU, m.NVML_CLOCK_SM = 0, 1
    m.nvmlInit = call("init", None)
    m.nvmlDeviceGetCount = call("count", 2)
    m.nvmlDeviceGetHandleByIndex = lambda i: ("h", i)
    m.nvmlDeviceGetHandleByUUID = lambda u: ("h", {"GPU-aa": 0, "GPU-bb": 1}[u])
    m.nvmlDeviceGetIndex = lambda h: h[1]
    m.nvmlDeviceGetPciInfo = lambda h: types.SimpleNamespace(busId=[b"00000000:1A:00.0", b"00000000:61:00.0"][h[1]])
    m.nvmlDeviceGetName = call("name", b"NVIDIA H100 80GB HBM3")
    m.nvmlDeviceGetUtilizationRates = call("util", types.SimpleNamespace(gpu=87, memory=40))
    m.nvmlDeviceGetMemoryInfo = call("mem", types.SimpleNamespace(used=60 * 2 ** 30, total=80 * 2 ** 30))
    m.nvmlDeviceGetTemperature = call("temp", 61)
    m.nvmlDeviceGetPowerUsage = call("power", 512_300)
    m.nvmlDeviceGetEnforcedPowerLimit = call("limit", 700_000)
    m.nvmlDeviceGetClockInfo = call("clock", 1980)
    m.nvmlDeviceGetComputeRunningProcesses = call("procs", [types.SimpleNamespace(pid=1, usedGpuMemory=2 ** 30)])
    return m


def test_telemetry_reads_visible_gpus_and_stops_when_no_page_polls(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(seen))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    t = Telemetry(idle=0.4, every=0.05)
    assert t.thread is None and not seen  # nothing runs before a page polls
    t.view()
    deadline = time.monotonic() + 5
    while not t.sample["gpus"] and time.monotonic() < deadline:
        time.sleep(0.02)
    g = t.sample["gpus"]
    assert len(g) == 1 and g[0]["index"] == 0 and g[0]["name"] == "NVIDIA H100 80GB HBM3"
    assert (g[0]["util"], g[0]["temp"], g[0]["power"], g[0]["power_limit"], g[0]["mem_own"]) == (87, 61, 512.3, 700.0,
                                                                                                    None)
    t.thread.join(5)
    assert not t.thread.is_alive()  # no poll for `idle` seconds: the side thread ended
    n = len(seen)
    time.sleep(0.2)
    assert len(seen) == n
    t.view()  # a page again: sampling again
    assert t.thread.is_alive() and seen.count("init") == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert [(i, k) for i, _, k in ui_module.visible(fake_nvml([]))] == [(0, 0), (1, 1)]


def test_gpu_card_names_the_physical_gpu(monkeypatch):
    """A server pinned to one GPU (CUDA_VISIBLE_DEVICES) sees it as cuda:0; the card also names the physical GPU, its
    NVML index and PCI bus id, so pages of two replicas on one box tell their GPUs apart."""
    for env, want in (("1", (1, "0000:61:00")), ("0", (0, "0000:1A:00")), ("GPU-bb", (1, "0000:61:00"))):
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml([]))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
        t = Telemetry()
        t.open()
        g = t.read()["gpus"]
        assert len(g) == 1 and g[0]["index"] == 0 and (g[0]["nvml_index"], g[0]["pci"]) == want
    page = (ROOT / "basal/ui.html").read_text(encoding="utf-8")
    assert "GPU ${g.nvml_index}" in page and "g.pci" in page and "cuda:${esc(g.index)}" in page


def test_page_ships_offline_and_fits_the_budget():
    page = (ROOT / "basal/ui.html").read_text(encoding="utf-8")
    assert not re.search(r"https?:|//[a-z0-9.-]+\.[a-z]{2,}|@import|cdn\.|googleapis", page, re.I)
    fonts = sorted((ROOT / "basal/fonts").glob("*.woff2"))
    assert {f.name for f in fonts} | {"OFL-DMSans.txt", "OFL-SpaceGrotesk.txt"} == set(ui_module.FONTS)
    assert all(f"ui/fonts/{f.name}" in page for f in fonts)
    assert len(page.encode()) + sum(f.stat().st_size for f in fonts) <= 400 * 1024
    for lic in ("OFL-DMSans.txt", "OFL-SpaceGrotesk.txt"):
        assert "SIL Open Font License" in (ROOT / "basal/fonts" / lic).read_text(encoding="utf-8")
    data = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"ui.html"' in data and '"fonts/*.woff2"' in data and '"fonts/*.txt"' in data


def test_help_mentions_the_ui():
    text = " ".join(parser().format_help().split())
    assert "--ui {off,on,full}" in text and "never their text" in text and "last 200 requests" in text


def test_on_never_keeps_or_serves_request_bodies():
    srv, u, app = app_of()

    async def go(c):
        assert (await c.post("/v1/systemone", json=BODY)).status_code == 200
        fresh(u)
        return (await c.get("/ui/request/1")).status_code, (await c.get("/ui/state")).json()
    status, s = asyncio.run(on(srv, app, go))
    assert status == 404 and u.bodies is None and s["status"]["ui"] == "on"
    assert all(len(row) == 6 for row in s["recent"]) and all(len(r) == 6 for r in u.ring)
    assert all("request/" not in getattr(r, "path", "") for r in http_app(srv, "fast", ui=u).routes)


def test_full_shows_the_last_requests_and_their_answers(tmp_path):
    keys = keyfile(tmp_path)
    srv, u, app = app_of(["--api-keys", str(keys)], keys=keys, full=True)
    auth = {"Authorization": f"Bearer {SECRET}"}
    decision = {"input": "Invoice 1240 PLN.", "questions": [{"type": "predicate", "instructions": "Paid?"}]}

    async def go(c):
        ok = await c.post("/v1/systemone", json=BODY, headers=auth)
        dec = await c.post("/v1/decisions", json=decision, headers=auth)
        bad = await c.post("/v1/systemone", content=b"{not json", headers=auth)
        fresh(u)
        state = (await c.get("/ui/state", headers=auth)).json()
        views = [await c.get(f"/ui/request/{i}", headers=auth) for i in (1, 2, 3, 4)]
        no_key = await c.get("/ui/request/1")
        return ok, dec, bad, state, views, no_key
    ok, dec, bad, state, views, no_key = asyncio.run(on(srv, app, go))
    assert state["status"]["ui"] == "full" and [row[6] for row in state["recent"]] == [3, 2, 1]
    one, two, three, four = views
    assert one.status_code == 200 and one.json()["request"] == BODY and one.json()["answer"] == ok.json()
    assert one.json()["endpoint"] == "/v1/systemone" and one.json()["status"] == 200 and one.json()["questions"] == 1
    assert two.json()["request"] == decision and two.json()["answer"] == dec.json() and two.json()["endpoint"] == \
        "/v1/decisions"
    assert three.json()["request"] == "{not json" and three.json()["status"] == 422 and three.json()["answer"] == \
        bad.json()
    assert four.status_code == 404 and no_key.status_code == 401


def test_full_keeps_200_requests(monkeypatch):
    _, u, _ = app_of(full=True)
    assert u.bodies.maxlen == 200

    async def handler(_):
        return types.SimpleNamespace(status_code=200, body=b'{"answers": {}}')
    rec = u.record(handler, "/v1/systemone")

    async def burst(n):
        for i in range(n):
            await rec(types.SimpleNamespace(_json={"state": f"text {i}", "questions": {}}))
    asyncio.run(burst(205))
    assert len(u.bodies) == 200 and u.request_json(5) is None
    kept = json.loads(u.request_json(6))
    assert kept["request"] == {"state": "text 5", "questions": {}} and kept["answer"] == {"answers": {}}
    assert json.loads(u.request_json(205))["request"]["state"] == "text 204"
