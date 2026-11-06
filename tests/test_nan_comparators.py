"""basal-selfcheck, basal-agree and basal-loadtest --determinism count a NaN or infinite readout as the largest change,
not as a comparison that max() and < skip."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import agree
from basal import loadtest
from basal import selfcheck


def test_a_nan_answer_is_not_an_agreement():
    agrees, outside_ties, max_dp, _, _ = selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": float("nan"), "b": 0.4})
    assert agrees is False and outside_ties is False and max_dp == 1.0


def test_a_clean_answer_still_agrees():
    assert selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": 0.61, "b": 0.39})[0] is True


def test_one_nan_readout_of_four_disagrees():
    ref = [[0.6, 0.4]] * 4
    got = [[float("nan"), 0.4]] + [[0.6, 0.4]] * 3
    c = agree.compare(ref, got)
    assert c["agreement"] == 0.75 and c["max_dp"] == 1.0


def test_clean_readouts_agree():
    c = agree.compare([[0.6, 0.4]] * 4, [[0.6, 0.4]] * 4)
    assert c["agreement"] == 1.0 and c["max_dp"] == 0.0


def nan_under_load(monkeypatch, tmp_path, body):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] > 2 and (calls["n"] - 2) % 2 == 1:  # the requests of the loaded pass: probe, other, probe, other
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": 0.5, "y": 0.5}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=1, out=str(tmp_path / "d.json"))
    asyncio.run(loadtest.determinism(a, bodies))
    return json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["determinism"]


@pytest.mark.parametrize("body", [b'{"answers": {"q": {"probabilities": {"x": NaN, "y": 0.5}}}}',
                                  b'{"answers": {"q": {"probabilities": {"x": 0.5, "y": NaN}}}}'])
def test_a_nan_under_load_is_a_change_of_1_wherever_it_stands(monkeypatch, tmp_path, body):
    d = nan_under_load(monkeypatch, tmp_path, body)
    assert d["max_abs_diff"] == 1.0 and d["identical"] == 0
