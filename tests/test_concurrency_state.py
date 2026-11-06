"""State a run leaves behind: the requests that failed under load, the reference cache of basal-agree."""
import argparse
import asyncio
import json
import pathlib

import httpx
import pytest

from basal import agree
from basal import loadtest


def test_requests_that_failed_under_load_are_counted(monkeypatch, tmp_path):
    def handler(request):
        if json.loads(request.content)["state"] in ("c", "d"):  # the load requests
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": 0.6, "y": 0.4}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(tmp_path / "d.json"))
    asyncio.run(loadtest.determinism(a, bodies))
    d = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["determinism"]
    assert d["load_errors"] == 2


def test_a_kill_in_the_middle_of_the_write_leaves_no_truncated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(agree, "alone", lambda v: [])
    monkeypatch.setattr(agree, "run_groups", lambda *a: [])
    monkeypatch.setattr(agree, "flat", lambda x: x)
    monkeypatch.setattr(agree, "regroup", lambda flat, v: [0.5] * 100)
    monkeypatch.setattr(agree, "free", lambda be: None)
    real = pathlib.Path.write_text

    def killed(self, data, *a, **k):
        real(self, data[: len(data) // 2], *a, **k)
        raise KeyboardInterrupt
    path = tmp_path / "cache" / "ref.json"
    monkeypatch.setattr(pathlib.Path, "write_text", killed)
    with pytest.raises(KeyboardInterrupt):
        agree.cached_reference(path, lambda: object(), {"s": []})
    monkeypatch.setattr(pathlib.Path, "write_text", real)
    assert not path.exists() or json.loads(path.read_text(encoding="utf-8"))


def test_a_complete_write_is_read_back(monkeypatch, tmp_path):
    monkeypatch.setattr(agree, "alone", lambda v: [])
    monkeypatch.setattr(agree, "run_groups", lambda *a: [])
    monkeypatch.setattr(agree, "flat", lambda x: x)
    monkeypatch.setattr(agree, "regroup", lambda flat, v: [0.5])
    monkeypatch.setattr(agree, "free", lambda be: None)
    path = tmp_path / "ref.json"
    ref, hit = agree.cached_reference(path, lambda: object(), {"s": []})
    assert hit is False and agree.cached_reference(path, lambda: object(), {"s": []}) == (ref, True)
    assert [p.name for p in tmp_path.iterdir()] == ["ref.json"]
