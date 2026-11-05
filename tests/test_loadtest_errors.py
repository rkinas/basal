"""basal-loadtest on bad input and on a server that does not answer: --n-seq 0 stops before any request, and a
transport error in the concurrent phase is counted in errors instead of ending the run without numbers."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import loadtest


class Resp:
    status_code = 200

    def raise_for_status(self):
        pass


def args(**kw):
    d = dict(url="http://test/v1/systemone", questions=[], synthetic=None, long_frac=0.0, seed=0,
             questions_per_request=1, n_seq=5, n_conc=3, concurrency=2, early_exit=None, out=None)
    d.update(kw)
    return argparse.Namespace(**d)


def client(calls, fail_on=None):
    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            calls.append(json)
            if fail_on is not None and len(calls) == fail_on:
                raise httpx.ReadTimeout("read timeout")
            return Resp()
    return Client


@pytest.fixture
def questions(monkeypatch):
    one = dict(state="s", question="q", options=["a", "b"])
    monkeypatch.setattr(loadtest, "load_questions", lambda path, n: [one] * n)


def test_n_seq_zero_stops_before_any_request(monkeypatch, questions):
    calls = []
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", client(calls))
    with pytest.raises(SystemExit, match="--n-seq"):
        asyncio.run(loadtest.run(args(n_seq=0)))
    assert calls == []


def test_transport_error_in_the_concurrent_phase_is_counted(monkeypatch, questions, capsys):
    calls = []
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", client(calls, fail_on=10 + 5 + 2))
    asyncio.run(loadtest.run(args()))
    out = json.loads(capsys.readouterr().out)
    assert out["errors"] == 1 and len(calls) == 10 + 5 + 3


def test_clean_run_has_no_errors(monkeypatch, questions, capsys):
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", client([]))
    asyncio.run(loadtest.run(args()))
    assert json.loads(capsys.readouterr().out)["errors"] == 0


@pytest.mark.parametrize("argv", [["--concurrency", "0"], ["--concurrency", "-3"],
                                  ["--concurrency", "0", "--determinism", "2"]])
def test_concurrency_below_one_stops_at_start(monkeypatch, argv):
    import sys
    monkeypatch.setattr(sys, "argv", ["basal-loadtest", "--synthetic", "200"] + argv)
    monkeypatch.setattr(loadtest, "run", lambda a: pytest.fail("a run started"))
    with pytest.raises(SystemExit, match="--concurrency must be at least 1"):
        loadtest.main()
