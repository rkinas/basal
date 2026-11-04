"""Regression tests for fix/concurrency-state (Z-1089)."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import loadtest


def test_q1_requests_that_failed_under_load_are_counted(monkeypatch, tmp_path):
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
