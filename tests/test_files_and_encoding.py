"""Regression tests for fix/files-and-encoding (Z-1089)."""
import json

import pytest

from basal import bench
from basal import calibrate as cal


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
