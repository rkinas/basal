"""basal-run, basal-loadtest and basal-calibrate --url against a server started with --api-keys (basal.auth.protect
over a stub app, through httpx's ASGI transport): a valid key passes, a missing or a wrong key ends the run with a
message naming --api-key; counts of 0 are refused."""
import argparse
import asyncio
import hashlib
import json
import sys

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from basal import calibrate, loadtest, think
from basal import run as run_module
from basal.auth import ApiKeys, protect

SECRET = "s3cret-key"
ITEM = {"state": "Order 1240 arrived damaged.", "question": "Refund?", "options": ["yes", "no"], "gold": 0}


async def decide(request):
    body = await request.json()
    answers = {k: {"type": q.get("type", "choice"), "confidence": 0.8, "probabilities":
                   {"true": 0.8, "false": 0.2} if q.get("type") == "noul" else
                   {c: (0.8 if i == 0 else 0.2) for i, c in enumerate(q["criteria"])}}
               for k, q in body["questions"].items()}
    return JSONResponse({"model": "stub", "answers": answers, "usage": {"latency_ms": 1.0}})


async def models(request):
    return JSONResponse({"models": [{"name": "stub", "mode": "eager", "temperature_per_prim": {}}]})


@pytest.fixture
def server(tmp_path):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"keys": [{"name": "ci", "sha256": hashlib.sha256(SECRET.encode()).hexdigest()}]}),
                    encoding="utf-8")
    app = Starlette(routes=[Route("/v1/systemone", decide, methods=["POST"]), Route("/v1/models", models)])
    return protect(app, ApiKeys(keys))


def asgi(monkeypatch, app):
    """httpx.AsyncClient talks to the app in process."""
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=app), **kw))


def run_args(tmp_path, key):
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(dict(ITEM, id=i)) + "\n" for i in range(5)), encoding="utf-8")
    return argparse.Namespace(input=str(tmp_path / "in.jsonl"), output=str(tmp_path / "out.jsonl"),
                              url="http://basal/v1/systemone", concurrency=2, early_exit=None, preset=None, lang=None,
                              api_key=key)


def test_run_with_a_valid_key(server, tmp_path, capsys):
    asyncio.run(run_module.run(run_args(tmp_path, SECRET), transport=httpx.ASGITransport(app=server)))
    out = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(out) == 5 and not any("error" in r for r in out) and all(r["correct"] for r in out)


@pytest.mark.parametrize("key, why", [(None, "the server needs an API key: give --api-key"),
                                      ("wrong", "the API key was refused")])
def test_run_without_a_valid_key_stops_with_a_message(server, tmp_path, monkeypatch, capsys, key, why):
    a = run_args(tmp_path, key)
    real = httpx.AsyncClient
    monkeypatch.setattr(run_module.httpx, "AsyncClient",
                        lambda **kw: real(**dict(kw, transport=httpx.ASGITransport(app=server))))
    monkeypatch.setattr(sys, "argv", ["basal-run", "--input", a.input, "--output", a.output, "--url", a.url,
                                      *(["--api-key", key] if key else [])])
    monkeypatch.delenv("BASAL_API_KEY", raising=False)
    with pytest.raises(SystemExit, match=f"HTTP 401: {why}"):
        run_module.main()
    assert not (tmp_path / "out.jsonl").exists()


def test_run_reads_the_key_from_the_environment(server, tmp_path, monkeypatch):
    a = run_args(tmp_path, None)
    real = httpx.AsyncClient
    monkeypatch.setattr(run_module.httpx, "AsyncClient",
                        lambda **kw: real(**dict(kw, transport=httpx.ASGITransport(app=server))))
    monkeypatch.setenv("BASAL_API_KEY", SECRET)
    monkeypatch.setattr(sys, "argv", ["basal-run", "--input", a.input, "--output", a.output, "--url", a.url])
    run_module.main()
    out = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(out) == 5 and not any("error" in r for r in out)


@pytest.mark.parametrize("key, ok", [(SECRET, True), (None, False), ("wrong", False)])
def test_loadtest(server, monkeypatch, capsys, key, ok):
    asgi(monkeypatch, server)
    monkeypatch.delenv("BASAL_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["basal-loadtest", "--url", "http://basal/v1/systemone", "--n-seq", "3",
                                      "--n-conc", "4", "--concurrency", "2", *(["--api-key", key] if key else [])])
    if ok:
        loadtest.main()
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["errors"] == 0
    else:
        with pytest.raises(SystemExit, match="basal-loadtest: http://basal/v1/systemone: HTTP 401"):
            loadtest.main()


@pytest.mark.parametrize("key, ok", [(SECRET, True), (None, False), ("wrong", False)])
def test_calibrate_over_http(server, tmp_path, monkeypatch, key, ok):
    """--url: GET /v1/models and the requests carry the key; without it the tool said no basal server answers."""
    asgi(monkeypatch, server)

    def get(url, headers=None, timeout=None):
        with TestClient(server) as c:
            return c.get(url, headers=headers)
    monkeypatch.setattr(httpx, "get", get)
    labelled = tmp_path / "l.jsonl"
    labelled.write_text("".join(json.dumps({"state": f"s{i}", "questions": {
        "q": {"type": "noul", "instructions": "?", "gold": "yes" if i % 2 else "no"}}}) + "\n" for i in range(4)),
        encoding="utf-8")
    argv = [str(labelled), "--url", "http://basal", "--no-calibration", "--out", str(tmp_path / "cal.json"),
            "--min-per-type", "1", *(["--api-key", key] if key else [])]
    monkeypatch.delenv("BASAL_API_KEY", raising=False)
    if ok:
        calibrate.main(argv)
        assert json.loads((tmp_path / "cal.json").read_text())["fitted_on"]["requests"] == 4
    else:
        with pytest.raises(SystemExit, match="--url http://basal: http://basal: HTTP 401"):
            calibrate.main(argv)


@pytest.mark.parametrize("tool, argv", [
    (run_module, ["basal-run", "--input", "x", "--output", "y", "--concurrency", "0"]),
    (calibrate, ["x.jsonl", "--concurrency", "0"]),
    (think, ["x.jsonl", "--out", "y", "--concurrency", "0"])])  # python -m basal.think rows
def test_zero_concurrency_is_refused(monkeypatch, capsys, tool, argv):
    """--concurrency 0 made a semaphore that admits nothing: the run waited for ever."""
    with pytest.raises(SystemExit) as e:
        if tool is calibrate:
            calibrate.main(argv)
        elif tool is think:
            think.rows_main(argv)
        else:
            monkeypatch.setattr(sys, "argv", argv)
            tool.main()
    assert e.value.code == 2 and "must be at least 1" in capsys.readouterr().err
