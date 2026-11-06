"""The files the tools read and write: a bad question line names its place, --out folders, cache homes, document ids
and the encoding of every text file a module opens."""
import json

import pytest

from basal import bench

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


def test_the_command_ends_with_the_message_not_a_traceback(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n{broken\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2"):
        bench.questions_or_exit(p, 10)
