"""Regression tests for fix/auth-encoding (Z-1082)."""
import asyncio
import hashlib
import json
import threading
import time

import pytest

from basal import auth as A
from basal.auth import ApiKeys
from basal.server import load_calibration


CAL = {"prompt": 1, "temperature_per_prim": {"choice": 1.25}}
KEYS = {"keys": [{"name": "ci", "sha256": hashlib.sha256(b"s3cret").hexdigest()}]}
ENC = {"utf-8": lambda s: s.encode(), "utf-8-bom": lambda s: ("\ufeff" + s).encode("utf-8"),
       "utf-16": lambda s: s.encode("utf-16")}  # PowerShell 5 `>` and Out-File


@pytest.mark.parametrize("enc", ENC)
def test_i15_calibration_file_in_a_windows_encoding(tmp_path, enc):
    path = tmp_path / "cal.json"
    path.write_bytes(ENC[enc](json.dumps(CAL)))
    assert load_calibration(None, "eager", path, None, 2)[0].get("temperature_per_prim") == {"choice": 1.25}


@pytest.mark.parametrize("enc", ENC)
def test_i15_api_key_file_in_a_windows_encoding(tmp_path, enc):
    path = tmp_path / "keys.json"
    path.write_bytes(ENC[enc](json.dumps(KEYS)))
    assert ApiKeys(path).check("s3cret") is not None


def test_s25_a_request_during_a_rewrite_of_the_key_file_is_not_rejected(tmp_path):
    path = tmp_path / "keys.json"
    full = json.dumps({"keys": [{"name": "k", "sha256": hashlib.sha256(b"sk-test").hexdigest(), "admin": False}]}).encode()
    path.write_bytes(full)
    store = ApiKeys(path)
    assert store.check("sk-test") is not None
    half = threading.Event()

    def writer():  # what `cat new > keys.json` or an editor does: truncate, then write
        with open(path, "wb") as f:
            f.write(full[:20])
            f.flush()
            half.set()
            time.sleep(0.02)
            f.write(full[20:])
    store._good = None
    t = threading.Thread(target=writer)
    t.start()
    half.wait()
    result = store.check("sk-test")
    t.join()
    assert result is not None


def test_s25_a_file_that_stays_broken_still_fails_closed(tmp_path):
    path = tmp_path / "keys.json"
    path.write_bytes(b'{"keys": [{"na')
    assert ApiKeys(path).check("sk-test") is None


async def ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def status(tmp_path, method, path, headers):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"keys": [{"name": "k1", "sha256": hashlib.sha256(b"good").hexdigest()}]}),
                         encoding="utf-8")
    out = {}

    async def send(m):
        if m["type"] == "http.response.start":
            out["s"] = m["status"]

    async def receive():
        return {"type": "http.request", "body": b""}
    scope = {"type": "http", "method": method, "path": path,
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(A.protect(ok_app, A.ApiKeys(keys_file))(scope, receive, send))
    return out["s"]


def test_p4a_a_valid_bearer_is_accepted_when_a_stale_x_api_key_header_is_also_sent(tmp_path):
    assert status(tmp_path, "POST", "/v1/systemone", {"x-api-key": "stale", "authorization": "Bearer good"}) == 200
    assert status(tmp_path, "POST", "/v1/systemone", {"x-api-key": "stale"}) == 401
    assert status(tmp_path, "POST", "/v1/systemone", {"authorization": "Bearer good"}) == 200


def test_p4b_a_load_balancer_probe_of_health_is_open(tmp_path):
    for method, path in (("GET", "/health"), ("HEAD", "/health"), ("HEAD", "/health/")):
        assert status(tmp_path, method, path, {}) == 200
    assert status(tmp_path, "POST", "/v1/systemone", {}) == 401


# --- second round (Z-1094) ---
def _rewrite_cut(path, full, cut):
    """What an editor does: truncate, write part of the bytes, wait, write the rest."""
    half = threading.Event()

    def writer():
        with open(path, "wb") as f:
            f.write(full[:cut])
            f.flush()
            half.set()
            time.sleep(0.02)
            f.write(full[cut:])
    t = threading.Thread(target=writer)
    t.start()
    half.wait()
    return t


@pytest.mark.parametrize("enc,shift", [("utf-8", 1), ("utf-8", 2), ("utf-8", 3), ("utf-16", 1), ("utf-16", 2)])
def test_a1_a_rewrite_cut_inside_a_character_is_read_once_more(tmp_path, enc, shift):
    path = tmp_path / "keys.json"
    text = json.dumps({"keys": [{"name": "k\U0001F511", "sha256": hashlib.sha256(b"sk").hexdigest()}]}, ensure_ascii=False)
    full = text.encode(enc)
    cut = full.index("\U0001F511".encode(enc)[-4:] if enc == "utf-16" else "\U0001F511".encode()) + shift
    with pytest.raises(UnicodeDecodeError):  # the cut is what it claims to be
        full[:cut].decode(enc)
    path.write_bytes(full[:cut])
    t = _rewrite_cut(path, full, cut)
    result = ApiKeys(path).check("sk")
    t.join()
    assert result is not None


@pytest.mark.parametrize("enc", ["utf-8", "utf-16"])
def test_a1_a_file_that_stays_cut_inside_a_character_still_fails_closed(tmp_path, enc):
    path = tmp_path / "keys.json"
    full = json.dumps({"keys": [{"name": "k\U0001F511", "sha256": hashlib.sha256(b"sk").hexdigest()}]},
                      ensure_ascii=False).encode(enc)
    path.write_bytes(full[:full.index("\U0001F511".encode(enc)[-4:]) + 2])
    assert ApiKeys(path).check("sk") is None


def test_a2_a_file_that_stays_broken_costs_one_wait_not_one_per_request(tmp_path):
    path = tmp_path / "keys.json"
    path.write_bytes(b'{"keys": [{"na')
    store = ApiKeys(path)
    t = time.perf_counter()
    for _ in range(10):
        assert store.check("sk") is None
    assert time.perf_counter() - t < 0.2  # one retry of 50 ms, then the failure is remembered
    path.write_text(json.dumps({"keys": [{"name": "k", "sha256": hashlib.sha256(b"sk").hexdigest()}]}), encoding="utf-8")
    assert store.check("sk") is not None  # a repaired file is read at once


def test_a2_the_event_loop_is_not_held_up_while_a_key_file_is_re_read(tmp_path):
    path = tmp_path / "keys.json"
    path.write_bytes(b'{"keys": [{"na')
    gaps = []

    async def ticker(stop):
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ticker(stop))
        await asyncio.sleep(0.03)
        out = await asyncio.gather(*(ApiKeys(path).acheck("sk") for _ in range(3)))  # three stores: three waits
        stop.set()
        await task
        return out
    assert asyncio.run(go()) == [None, None, None]
    assert max(gaps) < 0.035, max(gaps)


def test_a2_a_request_with_a_key_on_a_broken_file_does_not_block_the_loop(tmp_path):
    path = tmp_path / "keys.json"
    path.write_bytes(b'{"keys": [{"na')
    out, gaps = {}, []

    async def send(m):
        if m["type"] == "http.response.start":
            out["s"] = m["status"]

    async def receive():
        return {"type": "http.request", "body": b""}
    scope = {"type": "http", "method": "POST", "path": "/v1/systemone", "headers": [(b"x-api-key", b"sk")]}

    async def go():
        stop = asyncio.Event()

        async def ticker():
            last = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                gaps.append(time.perf_counter() - last)
                last = time.perf_counter()
        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.03)
        await A.protect(ok_app, ApiKeys(path))(scope, receive, send)
        stop.set()
        await task
    asyncio.run(go())
    assert out["s"] == 401 and max(gaps) < 0.035, max(gaps)


def test_a3_the_probe_of_health_is_open_on_the_real_app_with_and_without_the_slash(tmp_path):
    from basal.server import http_app
    from tests.test_http_status import AppServer, Engine
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"keys": [{"name": "k", "sha256": hashlib.sha256(b"good").hexdigest()}]}), encoding="utf-8")
    app = A.protect(http_app(AppServer(Engine()), "fast"), ApiKeys(keys))  # the app of basal-serve, not ok_app

    def call(method, path):
        out = {}

        async def send(m):
            if m["type"] == "http.response.start":
                out["s"] = m["status"]

        async def receive():
            return {"type": "http.request", "body": b""}
        asyncio.run(app({"type": "http", "method": method, "path": path, "headers": [], "query_string": b"",
                         "root_path": "", "scheme": "http", "server": ("basal", 80), "client": ("1.2.3.4", 1)},
                        receive, send))
        return out["s"]
    assert [call(m, p) for m, p in (("GET", "/health"), ("HEAD", "/health"), ("GET", "/health/"), ("HEAD", "/health/"),
                                    ("POST", "/health"), ("GET", "/v1/models"), ("GET", "/metrics"))] \
        == [200, 200, 200, 200, 401, 401, 401]


def test_a4_the_docs_say_what_the_code_opens():
    from pathlib import Path
    docs = Path(__file__).resolve().parent.parent / "docs"
    assert "HEAD /health" in (docs / "SERVING.md").read_text(encoding="utf-8")
    assert "HEAD /health" in (docs / "MCP.md").read_text(encoding="utf-8")


def test_z4_a_half_published_cache_state_cannot_be_read(tmp_path):
    """The cache is one tuple (signature, records, loaded_at): no state with a signature and no time exists to be read
    without the lock (it raised TypeError float - None, a 500 on the first request)."""
    import os
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": "a", "sha256": hashlib.sha256(b"sk").hexdigest()}]}), encoding="utf-8")
    store = ApiKeys(path)
    assert store._cached() is None and store._good is None
    assert store.check("sk") is not None
    sig, records, at = store._good
    assert store._cached() == records and isinstance(at, float)
    stat = os.stat(path)
    assert sig == (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    store._good = (sig, records, at - 3)  # older than 2 s: has to be read again
    assert store._cached() is None


def test_z4_cached_never_raises_while_other_threads_reload(tmp_path):
    import os
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": "a", "sha256": hashlib.sha256(b"sk").hexdigest()}]}), encoding="utf-8")
    store = ApiKeys(path)
    errors, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                store._cached()
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    threads = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in threads]
    for i in range(300):
        os.utime(path, ns=(10**18 + i, 10**18 + i))  # a new signature each round: _load publishes again
        store._load()
    stop.set()
    [t.join() for t in threads]
    assert errors == []
