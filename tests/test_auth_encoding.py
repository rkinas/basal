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
    store._signature = None
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
