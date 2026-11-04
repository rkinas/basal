"""Regression tests for fix/auth-encoding (Z-1082)."""
import hashlib
import json
import threading
import time

import pytest

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
