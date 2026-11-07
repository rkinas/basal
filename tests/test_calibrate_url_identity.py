"""basal-calibrate --url records the revision and model files the server serves (GET /v1/models reports them), not
the revision the local Hugging Face cache resolves --model to; an older server that does not report them keeps the
cache's, with a warning. basal-router passes the fields through."""
import asyncio
import json
from types import SimpleNamespace

import httpx
from starlette.testclient import TestClient

import basal.calibrate as calibrate_module
from basal import router as R
from basal.calibrate import main
from basal.server import Server, calibration_mismatch, http_app, model_files_id, parser
from tests.test_calibrate import SHIPPED
from tests.test_calibration_mismatch import R1, R2, R3, hub  # noqa: F401 (fixture)


def models_of(md):
    """GET /v1/models of a basal-serve (stub engine, no calibration) on the model directory md."""
    srv = Server(parser().parse_args(["--mode", "eager", "--model", str(md), "--no-calibration"]))
    return TestClient(http_app(srv, "eager")).get("/v1/models").json()["models"][0]


def test_the_server_reports_the_revision_and_model_files_it_serves(hub, tmp_path):  # noqa: F811
    entry = models_of(hub[R3])
    assert entry["revision"] == R3 and entry["model_files"] == model_files_id(hub[R3]) is not None
    plain = tmp_path / "plain"
    plain.mkdir()
    entry = models_of(plain)
    assert "revision" in entry and entry["revision"] is None and entry["model_files"] is None  # no snapshot


def calibrate_through(entry, monkeypatch, local, tmp_path, *extra):
    """basal-calibrate --url --model Org/m against a server whose GET /v1/models is `entry`, with the local cache
    resolving Org/m to the snapshot `local` -> the fitted file's path."""
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"state": f"s{i}", "group": f"g{i}", "questions": {"q": {
        "type": "noul", "instructions": "?", "gold": "yes" if i % 3 else "no"}}}) + "\n" for i in range(40)),
        encoding="utf-8")
    reply = SimpleNamespace(status_code=200, json=lambda: {"models": [entry]})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: reply)
    monkeypatch.setattr(calibrate_module, "resolve", lambda *a, **k: local)

    async def readouts(endpoint, bodies, *rest):
        return [{"answers": {"q": {"probabilities": {"true": 0.8, "false": 0.2}}}} for _ in bodies]
    monkeypatch.setattr(calibrate_module, "over_http", readouts)
    out = tmp_path / "cal.json"
    main([str(labels), "--url", "http://h", "--model", "Org/m", "--min-per-type", "1", "--out", str(out), *extra])
    return out


def test_a_file_fitted_through_a_server_records_the_servers_revision(hub, monkeypatch, tmp_path, capsys):  # noqa: F811
    # the server runs R3 (other weights than R1), the local cache resolves Org/m to R1
    entry = models_of(hub[R3])
    out = calibrate_through(entry, monkeypatch, hub[R1], tmp_path)
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    assert (fitted["model"], fitted["revision"], fitted["model_files"]) == ("Org/m", R3, model_files_id(hub[R3]))
    err = capsys.readouterr().err
    assert f"the server serves revision {R3[:12]} of Org/m, the local cache resolves Org/m to {R1[:12]}" in err
    # the loop closes: the file applies where it was fitted, and not to R1's other weights
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert calibration_mismatch(cal, str(hub[R3]), hub[R3]) is None
    assert "has other model files" in calibration_mismatch(cal, str(hub[R1]), hub[R1])


def test_the_compared_file_is_judged_by_the_servers_model_files(hub, monkeypatch, tmp_path, capsys):  # noqa: F811
    """A --calibration file fitted on R1 compared with readouts of a server on R3 (other weights) is not kept, although
    the local cache's snapshot is R1's; on R2 (a commit to the card) it is."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "Org/m", "revision": R1,
                                                        "model_files": model_files_id(hub[R1])})), encoding="utf-8")
    for rev, kept in ((R3, False), (R2, True)):
        calibrate_through(models_of(hub[rev]), monkeypatch, hub[R1], tmp_path, "--calibration", str(base))
        err = capsys.readouterr().err
        assert ("has other model files" in err) is not kept and ("not compared or kept" in err) is not kept


def test_an_older_server_keeps_the_caches_revision_with_a_warning(hub, monkeypatch, tmp_path, capsys):  # noqa: F811
    entry = {k: v for k, v in models_of(hub[R3]).items() if k not in ("revision", "model_files")}  # before 2.8
    fitted = json.loads(calibrate_through(entry, monkeypatch, hub[R1], tmp_path).read_text(encoding="utf-8"))
    assert fitted["fitted_on"]["revision"] == R1 and fitted["fitted_on"]["model_files"] == model_files_id(hub[R1])
    assert "the server does not report the revision it serves (basal-serve before 2.8)" in capsys.readouterr().err


def test_a_server_without_a_snapshot_keeps_the_caches_revision(hub, monkeypatch, tmp_path, capsys):  # noqa: F811
    entry = dict(models_of(hub[R3]), revision=None, model_files=None)  # e.g. served from a plain directory
    fitted = json.loads(calibrate_through(entry, monkeypatch, hub[R1], tmp_path).read_text(encoding="utf-8"))
    assert fitted["fitted_on"]["revision"] == R1
    assert "revision" not in capsys.readouterr().err


def test_the_same_revision_records_it_once_and_warns_of_nothing(hub, monkeypatch, tmp_path, capsys):  # noqa: F811
    out = calibrate_through(models_of(hub[R1]), monkeypatch, hub[R1], tmp_path)
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    assert (fitted["revision"], fitted["model_files"]) == (R1, model_files_id(hub[R1]))
    assert "warning: the server" not in capsys.readouterr().err


def test_the_router_passes_the_fields_through():
    main_url = "http://main"
    entry = {"name": "m", "revision": R3, "model_files": "0123456789abcdef"}

    def handler(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [entry]})
        return httpx.Response(200, json={"status": "ok"})
    rt = R.Router([main_url], transport=httpx.MockTransport(handler))
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        assert c.get("/v1/models").json() == {"models": [entry]}
