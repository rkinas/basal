"""The JSON files a person edits (calibration, API keys, references) are read with a BOM or in UTF-16."""
import hashlib
import json

import pytest

from basal.auth import ApiKeys
from basal.server import load_calibration

CAL = {"prompt": 1, "temperature_per_prim": {"choice": 1.25}}
KEYS = {"keys": [{"name": "ci", "sha256": hashlib.sha256(b"s3cret").hexdigest()}]}
ENC = {"utf-8": lambda s: s.encode(), "utf-8-bom": lambda s: ("﻿" + s).encode("utf-8"),
       "utf-16": lambda s: s.encode("utf-16")}  # PowerShell 5 `>` and Out-File


@pytest.mark.parametrize("enc", ENC)
def test_a_calibration_file_in_a_windows_encoding(tmp_path, enc):
    path = tmp_path / "cal.json"
    path.write_bytes(ENC[enc](json.dumps(CAL)))
    assert load_calibration(None, "eager", path, None, 2)[0].get("temperature_per_prim") == {"choice": 1.25}


@pytest.mark.parametrize("enc", ENC)
def test_an_api_key_file_in_a_windows_encoding(tmp_path, enc):
    path = tmp_path / "keys.json"
    path.write_bytes(ENC[enc](json.dumps(KEYS)))
    assert ApiKeys(path).check("s3cret") is not None


@pytest.mark.parametrize("enc", ENC)
def test_a_calibrate_reference_file_in_a_windows_encoding(tmp_path, enc):
    from basal import calibrate
    path = tmp_path / "ref.json"
    path.write_bytes(ENC[enc](json.dumps({"gates": {"ece_lofo": {"choice": {"mean": 0.05}}}})))
    assert calibrate.reference_values(str(path))[0] == {"choice": 0.05}


@pytest.mark.parametrize("enc", ENC)
def test_a_selfcheck_reference_file_in_a_windows_encoding(tmp_path, enc, monkeypatch):
    from basal import selfcheck
    path = tmp_path / "ref.json"
    path.write_bytes(ENC[enc](json.dumps({"checks": []})))

    def past_the_read(url):
        raise RuntimeError("read ok")
    monkeypatch.setattr(selfcheck, "server_info", past_the_read)
    with pytest.raises(RuntimeError, match="read ok"):  # a JSON or decode error would stop before the server
        selfcheck.check("http://x", str(path), None, "")
