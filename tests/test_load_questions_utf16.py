"""Z-1113: a question file in UTF-16 (PowerShell 5) is read like any other."""
import pytest

from basal import bench

GOOD = '{"state": "s", "options": ["a", "b"]}'


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16"])
def test_z3_load_questions_reads_utf_16_too(tmp_path, enc):
    p = tmp_path / "q.jsonl"
    p.write_bytes((GOOD + "\n").encode(enc))
    assert len(bench.load_questions(p, 10)) == 1
