"""A calibration file in the model's own folder that was fitted on another model is served with a start-up warning
(--strict-calibration refuses it): one basal-interpolate copied from a source checkpoint (basal.json "interpolation")
and not refitted, or one basal-calibrate wrote for another model. A model's own shipped file starts quietly."""
import json

import pytest
from test_calibrate import SHIPPED, StubBackend

from basal import server as server_module
from basal.server import Server, parser

INTERP = {"interpolation": {"old": "Remek/basal-1.5-mini", "new": "Remek/basal-2.0-mini", "alpha": 0.5}}


def model(tmp_path, monkeypatch, cal, info=None):
    md = tmp_path / "merged"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(cal), encoding="utf-8")
    if info is not None:
        (md / "basal.json").write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md


@pytest.mark.parametrize("cal,info,said", [
    (SHIPPED, INTERP, "basal-interpolate copied it"),
    (dict(SHIPPED, fitted_on={"model": "Remek/basal-2.0-mini"}), None, "fitted on Remek/basal-2.0-mini"),
])
def test_a_folder_calibration_of_another_model_warns(tmp_path, monkeypatch, capsys, cal, info, said):
    model(tmp_path, monkeypatch, cal, info)
    srv = Server(parser().parse_args(["--mode", "eager", "--model", "merged"]))
    out = capsys.readouterr().out
    assert "warning: CALIBRATION.json" in out and said in out and srv.cal_foreign
    assert srv.cal["thresholds"] == SHIPPED["thresholds"]  # (served as it is: a warning, not a refusal)


def test_strict_calibration_refuses_it(tmp_path, monkeypatch):
    model(tmp_path, monkeypatch, SHIPPED, INTERP)
    with pytest.raises(SystemExit, match="basal-interpolate copied it"):
        Server(parser().parse_args(["--mode", "eager", "--model", "merged", "--strict-calibration"]))


@pytest.mark.parametrize("cal,info", [
    (SHIPPED, None),                                                   # a model's own shipped file
    (dict(SHIPPED, fitted_on={"model": "merged"}), INTERP),            # refitted on the merge with basal-calibrate
])
def test_its_own_file_starts_quietly(tmp_path, monkeypatch, capsys, cal, info):
    model(tmp_path, monkeypatch, cal, info)
    srv = Server(parser().parse_args(["--mode", "eager", "--model", "merged"]))
    assert "warning: CALIBRATION.json" not in capsys.readouterr().out and srv.cal_foreign is None
