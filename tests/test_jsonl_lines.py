"""JSONL readers split at "\n" only: json.dumps(..., ensure_ascii=False) leaves U+2028, U+2029 and U+0085 unescaped
in strings (text pasted from a word processor or a PDF), and str.splitlines cut such a valid line in two."""
import json

from basal import calibrate
from basal.bench import load_questions
from basal.jsonl import read_lines

STATE = "Pierwszy akapit. Drugi akapit. Trzeci\u0085koniec."


def write(path, records, newline="\n"):
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + newline for r in records).encode("utf-8"))
    return path


def test_read_lines_keeps_unicode_separators_inside_a_line(tmp_path):
    recs = [{"id": 1, "state": STATE}, {"id": 2, "state": "two"}]
    for newline in ("\n", "\r\n"):
        lines = read_lines(write(tmp_path / "a.jsonl", recs, newline))
        assert [json.loads(x) for x in lines] == recs
    (tmp_path / "b.jsonl").write_text('{"a": 1}\n\n{"a": 2}', encoding="utf-8")
    assert read_lines(tmp_path / "b.jsonl") == ['{"a": 1}', "", '{"a": 2}']  # line numbers as before
    (tmp_path / "c.jsonl").write_text("", encoding="utf-8")
    assert read_lines(tmp_path / "c.jsonl") == []


def test_bench_questions_with_a_line_separator_in_the_state(tmp_path):
    f = write(tmp_path / "q.jsonl", [{"state": STATE, "question": "Ile akapitów?", "options": ["1", "3"], "gold": 1},
                                     {"state": "x", "question": "y", "options": ["a", "b"]}])
    qs = load_questions([f], 10)
    assert len(qs) == 2 and STATE in [q["state"] for q in qs]


def test_calibrate_reads_the_line_whole_and_keeps_line_numbers(tmp_path):
    f = write(tmp_path / "l.jsonl", [
        {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "Trzy akapity?", "gold": "yes"}}},
        {"state": "dwa", "questions": {"q": {"type": "noul", "instructions": "Trzy akapity?", "gold": "no"}}}])
    bodies, decisions = calibrate.load(f)
    assert [b["state"] for b in bodies] == [STATE, "dwa"]
    assert [d["line"] for d in decisions] == [1, 2]
