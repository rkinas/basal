"""Regression tests for fix/files-and-encoding (Z-1089)."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import bench
from basal import calibrate as cal
from basal import loadtest


GOOD = json.dumps({"state": "s", "options": ["a", "b"]})


def test_v1_a_file_saved_with_a_bom_is_read(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + GOOD.encode() + b"\n")
    assert len(bench.load_questions(p, 10)) == 1


@pytest.mark.parametrize("line", ["5", "null", '"a text with options"', "{broken"])
def test_v1_a_row_that_is_not_an_object_names_file_and_line(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2"):
        bench.load_questions(p, 10)


def test_v2_a_labelled_file_with_a_bom_is_read(tmp_path):
    q = {"type": "noul", "instructions": "Late?", "gold": "yes"}
    p = tmp_path / "x.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"id": "r1", "state": "s", "questions": {"q": q}}).encode() + b"\n")
    _, decisions, _, _ = cal.read(str(p))
    assert len(decisions) == 1


def test_v5_determinism_creates_the_directory_of_out(monkeypatch, tmp_path):
    def handler(request):
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": 0.6, "y": 0.4}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    out = tmp_path / "new" / "dir" / "d.json"
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(out))
    asyncio.run(loadtest.determinism(a, bodies))
    assert json.loads(out.read_text(encoding="utf-8"))["determinism"]["requests"] == 2
