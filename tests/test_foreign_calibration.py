"""A calibration file in the model's own folder that was fitted on another model is served with a start-up warning
(--strict-calibration refuses it): one basal-interpolate copied from a source checkpoint (basal.json "interpolation")
and not refitted, or one basal-calibrate wrote for another model. A model's own shipped file starts quietly."""
import json
from pathlib import Path

import pytest
from test_calibrate import SHIPPED, StubBackend

from basal import server as server_module
from basal.server import Server, foreign_calibration, parser

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


def test_a_shipped_file_follows_the_model_files_across_a_revision(tmp_path, monkeypatch, capsys):
    """A file shipped in the model's folder, fitted on revision R1: a later revision that changes only the card holds
    the same model files and starts quietly (the shipped file itself is committed after its fit); one that changes the
    weights or the tokenizer warns. Read as a --calibration file is (calibration_mismatch)."""
    from test_calibration_mismatch import CHANGES, FILES, R1, R2, R3, R4, snapshot
    shipped = json.dumps(dict(SHIPPED, fitted_on={"model": "Org/m", "revision": R1})).encode()
    monkeypatch.setattr(server_module, "resolve", lambda name, *_, **__: Path(name))
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    hub = {rev: snapshot(tmp_path / "hub", "Org/m", rev, dict(FILES, **change, **{"CALIBRATION.json": shipped}))
           for rev, change in CHANGES.items()}
    for rev, quiet in ((R1, True), (R2, True), (R3, False), (R4, False)):
        srv = Server(parser().parse_args(["--mode", "eager", "--model", str(hub[rev])]))
        out = capsys.readouterr().out
        if quiet:
            assert srv.cal_foreign is None and "warning: CALIBRATION.json" not in out
        else:
            assert "other model files" in srv.cal_foreign and "warning: CALIBRATION.json" in out
    # an engine that serves weights the folder does not hold (--gguf, Ollama, llama.cpp) compares the revision
    cal = json.loads(shipped)
    assert foreign_calibration(cal, str(hub[R2]), hub[R2]) is None
    assert f"the served revision is {R2[:12]}" in foreign_calibration(cal, str(hub[R2]), hub[R2], "gguf")
