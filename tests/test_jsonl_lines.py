"""JSONL readers split at "\n" only: json.dumps(..., ensure_ascii=False) leaves U+2028, U+2029 and U+0085 unescaped
in strings (text pasted from a word processor or a PDF), and str.splitlines cut such a valid line in two."""
import json
import os
import threading

import pytest

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


# Notepad saves UTF-8 with a BOM, PowerShell 5 (`>`, Out-File) UTF-16 with a BOM, little-endian; big-endian for good
# measure. Every JSONL reader takes them, and still splits at "\n" only.
CODECS = {"utf-8-bom": lambda s: ("﻿" + s).encode("utf-8"), "utf-16": lambda s: s.encode("utf-16"),
          "utf-16-be": lambda s: ("﻿" + s).encode("utf-16-be")}


def write_as(path, records, codec, newline="\n"):
    path.write_bytes(CODECS[codec]("".join(json.dumps(r, ensure_ascii=False) + newline for r in records)))
    return path


@pytest.mark.parametrize("codec", CODECS)
def test_read_lines_takes_a_bom_and_utf_16(tmp_path, codec):
    recs = [{"id": 1, "state": STATE}, {"id": 2, "state": "two"}]
    for newline in ("\n", "\r\n"):
        lines = read_lines(write_as(tmp_path / "a.jsonl", recs, codec, newline))
        assert [json.loads(x) for x in lines] == recs


def test_read_lines_of_a_plain_file_is_unchanged(tmp_path):
    (tmp_path / "a.jsonl").write_bytes(b'{"a": 1}\r{"a": 2}\n\xef\xbb\xbf')  # a lone "\r" ends a line, as before
    assert read_lines(tmp_path / "a.jsonl") == ['{"a": 1}', '{"a": 2}', "﻿"]  # only a leading BOM is dropped
    (tmp_path / "b.jsonl").write_bytes(b"\xff\xfe")  # an empty UTF-16 file
    assert read_lines(tmp_path / "b.jsonl") == []


@pytest.mark.parametrize("codec", CODECS)
def test_bench_and_calibrate_read_a_bom_and_utf_16(tmp_path, codec):
    f = write_as(tmp_path / "q.jsonl", [{"state": STATE, "question": "Ile?", "options": ["1", "3"], "gold": 1}], codec)
    assert [q["state"] for q in load_questions([f], 10)] == [STATE]
    f = write_as(tmp_path / "l.jsonl", [
        {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "Trzy akapity?", "gold": "yes"}}}], codec)
    bodies, decisions = calibrate.load(f)
    assert [b["state"] for b in bodies] == [STATE] and [d["line"] for d in decisions] == [1]


@pytest.mark.parametrize("codec", CODECS)
def test_run_reads_a_bom_and_utf_16(tmp_path, codec):
    from basal.run import lines
    recs = [{"id": "a", "state": STATE, "question": "q", "options": ["x", "y"]}, {"id": "b", "state": "two"}]
    assert list(lines(write_as(tmp_path / "in.jsonl", recs, codec, "\r\n"))) == recs


@pytest.mark.parametrize("codec", CODECS)
def test_navigate_and_navtree_read_a_bom_and_utf_16(tmp_path, codec):
    from basal.navigate import Tree
    from basal.navtree import documents
    nodes = [{"id": "r", "text": "root"}, {"id": "x", "parent": "r", "text": STATE}]
    assert Tree.load(write_as(tmp_path / "t.jsonl", nodes, codec)).nodes["x"]["text"] == STATE
    docs = list(documents([str(write_as(tmp_path / "d.jsonl", [{"id": "d1", "text": STATE}], codec))]))
    assert [(d[0], d[4]) for d in docs] == [("d1", STATE)]


@pytest.mark.parametrize("codec", CODECS)
def test_the_think_calibration_reads_a_bom_and_utf_16(tmp_path, codec):
    from basal.think import think_calibration
    cal = {"temperature_per_prim": {"choice": 1.25}, "note": STATE}
    (tmp_path / "c.json").write_bytes(CODECS[codec](json.dumps(cal, ensure_ascii=False)))
    assert think_calibration(None, str(tmp_path / "c.json")) == (cal, "c.json")


# A pipe can be read once: `basal-run --input <(jq ...)`, `--input /dev/stdin`, `navigate --queries <(...)`. The
# codec is picked from bytes the reader keeps, not from a first open that eats them.
RECS = [{"id": "a", "state": STATE, "question": "q", "options": ["x", "y"]}, {"id": "b", "state": "two"}]
PIPED = {"utf-8": lambda s: s.encode(), **CODECS}


def piped(data):
    """data behind a pipe, as the shell's <(...) hands it over: /dev/fd/N, no seeking, read once."""
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    return r, f"/dev/fd/{r}"


@pytest.mark.skipif(not os.path.isdir("/dev/fd") or os.name == "nt", reason="no /dev/fd")
@pytest.mark.parametrize("codec", PIPED)
def test_a_pipe_is_read_once_with_its_first_record_whole(codec):
    from basal.navigate import Tree
    from basal.run import lines
    from basal.textio import read_text
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in RECS)
    nodes = [{"id": "r", "text": "root"}, {"id": "x", "parent": "r", "text": STATE}]
    cases = [(text, lambda p: [json.loads(x) for x in read_lines(p)], RECS), (text, lambda p: list(lines(p)), RECS),
             (json.dumps(RECS[0], ensure_ascii=False), lambda p: json.loads(read_text(p)), RECS[0]),
             ("".join(json.dumps(n) + "\n" for n in nodes), lambda p: Tree.load(p).nodes["x"]["text"], STATE)]
    for data, read, want in cases:
        r, path = piped(PIPED[codec](data))
        try:
            assert read(path) == want
        finally:
            os.close(r)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
def test_a_fifo_is_read_once(tmp_path):
    path = tmp_path / "in.jsonl"
    os.mkfifo(path)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in RECS).encode("utf-8")

    def write():
        with open(path, "wb") as f:  # opens once a reader does
            f.write(data)
    threading.Thread(target=write, daemon=True).start()
    got = []
    reader = threading.Thread(target=lambda: got.append([json.loads(x) for x in read_lines(path)]), daemon=True)
    reader.start()
    reader.join(5)
    if reader.is_alive():  # a second open waits for a writer that never comes: give it an end of file
        open(path, "wb").close()
        reader.join(5)
    assert got == [RECS]
