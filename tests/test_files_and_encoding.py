"""The files the tools read and write: a bad question line names its place, --out folders, cache homes, document ids
and the encoding of every text file a module opens."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import bench
from basal import loadtest

GOOD = json.dumps({"state": "s", "options": ["a", "b"]})


@pytest.mark.parametrize("line", ["5", "null", "{broken", "true"])
def test_a_row_that_is_not_an_object_names_file_and_line_in_a_ValueError(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"q\.jsonl:2"):
        bench.load_questions(p, 10)


@pytest.mark.parametrize("line", ['"a text"', "[1, 2]"])
def test_a_text_or_a_list_row_is_skipped(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(line + "\n" + GOOD + "\n", encoding="utf-8")
    assert len(bench.load_questions(p, 10)) == 1


@pytest.mark.parametrize("line", ['"a text with options"', '["options", "a", "b"]'])
def test_a_text_or_a_list_that_holds_options_is_no_question(tmp_path, line):
    p = tmp_path / "q.jsonl"  # "options" in a str or a list was a substring or element test, and let the line in
    p.write_text(line + "\n" + GOOD + "\n", encoding="utf-8")
    assert bench.load_questions(p, 10) == [json.loads(GOOD)]


def test_the_command_ends_with_the_message_not_a_traceback(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n{broken\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2"):
        bench.questions_or_exit(p, 10)


def test_determinism_creates_the_directory_of_out(monkeypatch, tmp_path):
    def handler(request):
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": 0.6, "y": 0.4}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    out = tmp_path / "new" / "dir" / "d.json"
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(out))
    asyncio.run(loadtest.determinism(a, bodies))
    assert json.loads(out.read_text(encoding="utf-8"))["determinism"]["requests"] == 2
