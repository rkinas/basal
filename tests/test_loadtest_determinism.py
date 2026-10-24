"""basal-loadtest --determinism: the same requests alone and under load -> identical answers, max |dp| and decision
flips (answers whose top option changed)."""
import argparse
import asyncio
import json
import sys

import httpx

from basal import loadtest


def test_flips_and_differences(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        calls["n"] += 1
        p = 0.6 if calls["n"] <= 2 else (0.4 if body["state"] == "a" else 0.6)  # request "a" flips under load
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": p, "y": 1 - p}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2)
    d = asyncio.run(loadtest.determinism(a, bodies))["determinism"]
    assert d["requests"] == 2 and d["identical"] == 1 and d["decision_flips"] == 1
    assert abs(d["max_abs_diff"] - 0.2) < 1e-9


def test_seed_gives_other_synthetic_states(monkeypatch):
    """--seed: a second load test with another seed asks states a prefix-caching server has not seen yet."""
    seen = []
    monkeypatch.setattr(loadtest, "run", lambda a: seen.append(a.seed))
    monkeypatch.setattr(loadtest.asyncio, "run", lambda x: {})
    for argv in (["--synthetic", "200"], ["--synthetic", "200", "--seed", "2"]):
        monkeypatch.setattr(sys, "argv", ["basal-loadtest"] + argv)
        loadtest.main()
    assert seen == [0, 2]
    from basal import synth
    assert synth.items(3, 200, 1)[0]["state"] != synth.items(3, 200, 2)[0]["state"]
