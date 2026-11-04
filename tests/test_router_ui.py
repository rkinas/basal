"""basal-router --ui on (2.5): one page for the router and every replica (GET /ui, GET /ui/state); off adds nothing."""
import asyncio
import json
import os
import re
import subprocess
import sys
import time

import httpx
from starlette.testclient import TestClient

from basal import router as R
from basal import router_ui as RU

KEY = "k-123"


def replica_state(name):
    """A replica's GET /ui/state as basal-serve --ui on builds it (the fields the router's page reads)."""
    return {"now": 1.0, "started": 0.5, "ready_at": 0.9, "uptime": 0.1,
            "status": {"state": "ready", "model": "basal-test", "source": "m", "revision": None, "mode": "fast",
                       "version": "2.5.0", "think": "off", "running": 1, "queued": 0, "ui": "on", "auth": False},
            "traffic": {"requests": 3, "failed": 0, "seconds": 0.1, "seen": 3, "ring": 16384, "rate10": 1.0, "rate60": 0.5,
                        "latency": {"p50": 10.0, "p95": 12.0, "p99": 12.0, "max": 12.0, "mean": 10.5},
                        "w10": {"p50": 10.0, "p95": 12.0, "p99": 12.0, "n": 3},
                        "w60": {"p50": 10.0, "p95": 12.0, "p99": 12.0, "n": 3}, "endpoints": [], "codes": {"200": 3}},
            "cache": None,
            "telemetry": {"gpus": [{"index": 0, "name": "GPU-X", "util": 40, "mem_used": 1, "mem_total": 2}],
                          "gpu_note": None, "host": {}, "host_note": None, "at": 1.0},
            "config": {"options": [["api_keys", "(set)", True]] * 50, "facts": [["engine", name]]},
            "recent": [[1.0, "/v1/systemone", 200, 10.0, 1, 2]] * 200}


def fleet(ready, ui, asked=None):
    """Fake replicas: /health 200 when ready; POSTs answer with the replica's name; GET /ui/state by ui[name]: "on"
    (a state), "off" (404), "key" (401 unless the client's key came along). asked: GET /ui/state calls, by replica."""
    asked = {} if asked is None else asked

    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200 if base in ready else 503, json={"status": "ok"})
        if req.url.path == "/ui/state":
            asked[base] = asked.get(base, 0) + 1
            mode = ui.get(base, "off")
            if mode == "key" and req.headers.get("authorization") != f"Bearer {KEY}":
                return httpx.Response(401, json={"error": "key"})
            if mode in ("on", "key"):
                return httpx.Response(200, json=replica_state(base))
            return httpx.Response(404, json={"error": "not found"})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": base}]})
        return httpx.Response(200, json={"replica": base, "body": json.loads(req.content)})
    return httpx.MockTransport(handler)


NAMES = [f"http://r{i}" for i in range(3)]


def ui_router(ready=None, ui=None, asked=None, **kw):
    ready = set(NAMES) if ready is None else ready
    rt = R.Router(NAMES, transport=fleet(ready, ui or {}, asked), **kw)
    return rt, RU.RouterUI(rt)


def test_off_adds_no_route_and_leaves_choose_alone():
    rt = R.Router(NAMES, transport=fleet(set(NAMES), {}))
    a = R.app(rt, poll_s=3600)
    assert [r.path for r in a.routes] == ["/v1/systemone", "/v1/basal", "/health", "/v1/models", "/metrics"]
    assert rt.choose.__func__ is R.Router.choose                                    # not wrapped
    with TestClient(a) as c:
        asyncio.run(rt.poll())
        assert c.get("/ui").status_code == 404 and c.get("/ui/state").status_code == 404
        assert c.get("/ui/fonts/OFL-DMSans.txt").status_code == 404


MAIN_OFF = """
import json, sys
import uvicorn
from basal import router as R
seen = {}
uvicorn.run = lambda app, **k: seen.update(app=app)
R.main(["--replicas", "http://127.0.0.1:9"])
print(json.dumps(dict(routes=[r.path for r in seen["app"].routes], ui="basal.router_ui" in sys.modules,
                      ui_mod="basal.ui" in sys.modules)))
"""


def test_main_without_ui_imports_nothing_of_it():
    out = subprocess.run([sys.executable, "-c", MAIN_OFF], capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r == {"routes": ["/v1/systemone", "/v1/basal", "/health", "/v1/models", "/metrics"], "ui": False,
                 "ui_mod": False}


def test_page_and_state_show_every_replica():
    rt, ui = ui_router(ui={NAMES[0]: "on", NAMES[1]: "off", NAMES[2]: "key"})
    with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
        asyncio.run(rt.poll())
        page = c.get("/ui")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "<title>basal-router</title>" in page.text and "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert c.get("/ui/fonts/dm-sans-latin.woff2").status_code == 200
        got = [c.post("/v1/systemone", json={"state": f"Doc {i}", "questions": {}}).json()["replica"] for i in range(12)]
        c.post("/v1/basal", json={"questions": {}})
        s = c.get("/ui/state")
        assert s.status_code == 200 and s.headers["cache-control"] == "no-store"
        d = s.json()
        assert d["router"] == {"replicas": 3, "ready": 3, "in_flight": 0, "load_factor": R.LOAD_FACTOR}
        assert d["traffic"]["requests"] == 13 and d["traffic"]["failed"] == 0 and d["traffic"]["seen"] == 13
        assert [e["endpoint"] for e in d["traffic"]["endpoints"]] == ["/v1/basal", "/v1/systemone"]
        reps = d["replicas"]
        assert [x["ui"] for x in reps] == ["on", "off", "key"] and [x["url"] for x in reps] == NAMES
        assert reps[0]["state"]["status"]["model"] == "basal-test" and reps[0]["state"]["telemetry"]["gpus"]
        assert "recent" not in reps[0]["state"] and reps[0]["state"]["facts"] == [["engine", NAMES[0]]]
        assert reps[1]["state"] is None and reps[2]["state"] is None and reps[2]["detail"] == 401
        assert sum(x["requests"] for x in reps) == 13 == sum(x["routed"]["state"] + x["routed"]["load"] for x in reps)
        for x in reps:                                                              # the router's view of each one
            assert x["routed"]["state"] == got.count(x["url"]) and x["ready"] and x["in_flight"] == 0
        # the recent rows name the replica that answered, newest first, and how it was chosen
        rows = d["recent"]
        assert len(rows) == 13 and rows[0][1] == "/v1/basal" and rows[0][5] == "load"
        assert [NAMES[r[4]] for r in reversed(rows[1:])] == got and {r[5] for r in rows[1:]} == {"state"}
        # the key goes on to the replicas, as for requests: the third one opens up
        d2 = c.get("/ui/state", headers={"Authorization": f"Bearer {KEY}"}).json()
        assert [x["ui"] for x in d2["replicas"]] == ["on", "off", "on"]


def test_state_is_read_at_most_once_a_second_per_key_and_never_shared_across_keys():
    asked = {}
    rt, ui = ui_router(ui={n: "key" for n in NAMES}, asked=asked)
    with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
        asyncio.run(rt.poll())
        keyed = {"X-API-Key": KEY}
        assert [x["ui"] for x in c.get("/ui/state", headers={"Authorization": f"Bearer {KEY}"}).json()["replicas"]] \
            == ["on"] * 3
        for _ in range(5):
            assert [x["ui"] for x in c.get("/ui/state").json()["replicas"]] == ["key"] * 3   # no key: not the keyed one
        assert asked == {n: 2 for n in NAMES}                                        # one read per key, then memoised
        c.get("/ui/state", headers=keyed)
        assert asked == {n: 3 for n in NAMES}
        time.sleep(RU.MEMO_S + 0.05)
        c.get("/ui/state")
        assert asked == {n: 4 for n in NAMES}


def test_replicas_not_ready_are_not_read_and_failures_are_counted():
    asked = {}
    rt, ui = ui_router(ready=set(NAMES[:1]), ui={n: "on" for n in NAMES}, asked=asked)
    with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
        asyncio.run(rt.poll())
        assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 200
        d = c.get("/ui/state").json()
        assert [x["ui"] for x in d["replicas"]] == ["on", "not ready", "not ready"] and asked == {NAMES[0]: 1}
        rt.ready.clear()
        assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 503
        time.sleep(RU.MEMO_S + 0.05)
        d = c.get("/ui/state").json()
        assert d["traffic"]["requests"] == 2 and d["traffic"]["failed"] == 1 and d["traffic"]["codes"] == {"200": 1, "503": 1}
        assert d["recent"][0][2] == 503 and d["recent"][0][4] is None


def test_answers_through_the_router_are_the_same_with_and_without_the_page():
    def run(with_ui):
        rt = R.Router(NAMES, transport=fleet(set(NAMES), {n: "on" for n in NAMES}))
        ui = RU.RouterUI(rt) if with_ui else None
        out = []
        with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
            asyncio.run(rt.poll())
            for i in range(30):
                body = {"state": f"Document {i % 7}", "questions": {"q": {"criteria": ["a", "b"]}}} if i % 5 else {"questions": {}}
                r = c.post("/v1/basal" if i % 2 else "/v1/systemone", json=body)
                out.append((r.status_code, r.headers["content-type"], r.content))
                if with_ui and i % 10 == 0:
                    c.get("/ui/state")
            out.append(c.get("/metrics").text)
        return out
    assert run(True) == run(False)


def test_state_keeps_no_request_text():
    rt, ui = ui_router(ui={n: "on" for n in NAMES})
    with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
        asyncio.run(rt.poll())
        c.post("/v1/systemone", json={"state": "SECRET-DOCUMENT", "questions": {"q": {"instructions": "SECRET-Q"}}})
        text = c.get("/ui/state").text
        assert "SECRET" not in text


def test_cascade_blend_and_route_counters_on_the_page():
    rt, ui = ui_router(ui={}, levels=[([NAMES[2]], 0.8)], routes={"cyrillic": [NAMES[1]]})
    d = json.loads(asyncio.run(ui.state({})))
    assert d["features"]["cascade"]["below"] == [0.8] and d["features"]["route"]["scripts"] == ["cyrillic"]
    assert [x["pools"] for x in d["replicas"]] == [["main"], ["main", "script:cyrillic"], ["main", "level:2"]]
    rt2 = R.Router(NAMES[:1], transport=fleet(set(), {}), blend=[([NAMES[1]], 0.5)], blend_weight=0.5,
                   fallback={"url": "http://llm/v1", "model": "llm-x", "below": 0.5})
    d2 = json.loads(asyncio.run(RU.RouterUI(rt2).state({})))
    assert d2["features"]["blend"]["weights"] == [0.5, 0.5] and d2["features"]["fallback"]["model"] == "llm-x"
    assert d2["features"]["fallback"]["results"] == {k: 0 for k in R.FALLBACK_RESULTS}


def test_main_with_ui_serves_the_page_and_redacts_its_options(monkeypatch):
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **k: seen.update(app=app))
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    R.main(["--replicas", "http://127.0.0.1:9", "--ui", "on", "--fallback", "http://user:pw@llm:8100/v1?token=t",
            "--fallback-model", "m", "--fallback-extra", '{"api_key": "sk-secret"}'])
    with TestClient(seen["app"]) as c:
        assert c.get("/ui").status_code == 200
        d = c.get("/ui/state").json()
    opts = {k: (v, changed) for k, v, changed in d["config"]["options"]}
    assert opts["ui"] == ("on", True) and opts["fallback_extra"] == ("(set)", True)
    assert opts["fallback"] == ("http://llm:8100/v1", True) and opts["load_factor"] == (R.LOAD_FACTOR, False)
    text = json.dumps(d)
    assert "sk-secret" not in text and "user:pw" not in text and "token=t" not in text
    assert dict(d["config"]["facts"])["replicas"] == 1


def test_route_urls_are_redacted_like_the_others(monkeypatch):
    """--route SCRIPT=URL[,URL]: each URL of the value is shown without user, password and query, as --replicas and
    --fallback are; the script name stays."""
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **k: seen.update(app=app))
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)
    R.main(["--replicas", "http://127.0.0.1:9", "--ui", "on",
            "--route", "cyrillic=http://user:s3cret@127.0.0.1:10/?token=t,http://u2:p2@127.0.0.1:11"])
    with TestClient(seen["app"]) as c:
        d = c.get("/ui/state").json()
    opts = {k: v for k, v, _ in d["config"]["options"]}
    assert opts["route"] == ["cyrillic=http://127.0.0.1:10/,http://127.0.0.1:11"]
    text = json.dumps(d)
    assert "s3cret" not in text and "p2" not in text and "token=t" not in text


def test_page_loads_nothing_from_outside_and_stays_small():
    page = (RU.HERE / "router_ui.html").read_text()
    assert not re.search(r"(src|href)=[\"']?(https?:)?//", page) and "@import" not in page
    assert not re.search(r"fetch\(['\"]https?:", page)
    size = len(page.encode()) + sum((RU.HERE / "fonts" / f).stat().st_size for f in RU.FONTS)
    assert size <= 400_000


def test_record_adds_one_append_per_request():
    rt, ui = ui_router(ui={})
    with TestClient(R.app(rt, poll_s=3600, ui=ui)) as c:
        asyncio.run(rt.poll())
        for i in range(5):
            c.post("/v1/systemone", json={"state": f"d{i}", "questions": {}})
        assert len(ui.ring) == 5 and ui.counts == [5, 0]
        assert all(len(r) == 6 and r[4] in NAMES and r[5] == "state" for r in ui.ring)
