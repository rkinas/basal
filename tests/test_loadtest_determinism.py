"""basal-loadtest --determinism: the same requests alone and under load -> identical answers, max |dp| and decision
flips (answers whose top option changed); the p95 of the sequential latencies."""
import argparse
import asyncio
import itertools
import json
import sys
import types

import httpx
import pytest

from basal import loadtest


def test_flips_and_differences(monkeypatch, tmp_path):
    calls = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        calls["n"] += 1
        p = 0.6 if calls["n"] <= 2 else (0.4 if body["state"] == "a" else 0.6)  # request "a" flips under load
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": p, "y": 1 - p}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(tmp_path / "d.json"))
    asyncio.run(loadtest.determinism(a, bodies))
    d = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["determinism"]
    assert d["requests"] == 2 and d["identical"] == 1 and d["decision_flips"] == 1
    assert abs(d["max_abs_diff"] - 0.2) < 1e-9


def test_seed_gives_other_synthetic_states(monkeypatch):
    """--seed: a second load test with another seed asks states a prefix-caching server has not seen yet."""
    seen = []
    monkeypatch.setattr(loadtest, "run", lambda a: seen.append(a.seed))
    monkeypatch.setattr(loadtest.asyncio, "run", lambda x: None)
    for argv in (["--synthetic", "200"], ["--synthetic", "200", "--seed", "2"]):
        monkeypatch.setattr(sys, "argv", ["basal-loadtest"] + argv)
        loadtest.main()
    assert seen == [0, 2]
    from basal import synth
    assert synth.items(3, 200, 1)[0]["state"] != synth.items(3, 200, 2)[0]["state"]


@pytest.mark.parametrize("n_seq, p95", [(19, 19.0), (20, 19.0), (21, 20.0), (200, 190.0)])
def test_p95_is_the_nearest_rank(monkeypatch, tmp_path, n_seq, p95):
    """The sequential latencies are 1, 2, ... n_seq ms; p95 is the ceil(0.95 n)-th smallest: with 20 the 19th (19 ms),
    not the largest; with 200 (the default --n-seq) the 190th, not the 191st."""
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"answers": {}})), **kw))
    ticks = itertools.chain(*[(10.0 * k, 10.0 * k + (k + 1) / 1000) for k in range(n_seq)], itertools.count(1e6))
    monkeypatch.setattr(loadtest, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    a = argparse.Namespace(url="http://test/v1/systemone", synthetic=None, questions=loadtest.DEFAULT_QUESTIONS,
                           early_exit=None, questions_per_request=1, n_seq=n_seq, n_conc=4, concurrency=2,
                           out=str(tmp_path / "l.json"), long_frac=0.0, seed=0)
    asyncio.run(loadtest.run(a))
    out = json.loads((tmp_path / "l.json").read_text(encoding="utf-8"))
    assert round(out["p95_ms"], 6) == p95 and out["errors"] == 0, out


@pytest.mark.parametrize("extra", [[], ["--determinism", "3"]])
def test_no_simple_items_exits_instead_of_spinning(monkeypatch, tmp_path, extra, capsys):
    """A question file of full requests only (or an empty one) gave no bodies, and doubling an empty list until it
    reached n never ended (one CPU at 100%, no request sent)."""
    full = tmp_path / "full.jsonl"
    full.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}) + "\n",
                    encoding="utf-8")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    sent = []
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(
        lambda r: sent.append(r) or httpx.Response(200, json={"answers": {}})), **kw))
    for f in (full, tmp_path / "empty.jsonl"):
        monkeypatch.setattr(sys, "argv", ["basal-loadtest", "--questions", str(f), "--n-seq", "2", "--n-conc", "2",
                                          *extra])
        with pytest.raises(SystemExit, match="no simple items in"):
            loadtest.main()
    assert sent == []


@pytest.mark.parametrize("flag", ["--n-conc", "--determinism"])
def test_negative_counts_are_refused(monkeypatch, flag, capsys):
    monkeypatch.setattr(sys, "argv", ["basal-loadtest", flag, "-1"])
    with pytest.raises(SystemExit) as e:
        loadtest.main()
    assert e.value.code == 2 and "must be 0 or more" in capsys.readouterr().err
