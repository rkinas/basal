"""Regression tests for fix/files-and-encoding (Z-1089)."""
import argparse
import asyncio
import json
import pathlib
from pathlib import Path

import httpx
import pytest

from basal import agree
from basal import bench
from basal import calibrate as cal
from basal import interpolate
from basal import loadtest
from basal import navtree


GOOD = json.dumps({"state": "s", "options": ["a", "b"]})


def test_v1_a_file_saved_with_a_bom_is_read(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + GOOD.encode() + b"\n")
    assert len(bench.load_questions(p, 10)) == 1


@pytest.mark.parametrize("line", ["5", "null", "{broken", "true"])
def test_v1_a_row_that_is_not_an_object_names_file_and_line_in_a_ValueError(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"q\.jsonl:2"):
        bench.load_questions(p, 10)


@pytest.mark.parametrize("line", ['"a text"', "[1, 2]"])
def test_v1_a_text_or_a_list_row_is_skipped_as_on_main(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(line + "\n" + GOOD + "\n", encoding="utf-8")
    assert len(bench.load_questions(p, 10)) == 1


def test_v1_the_command_ends_with_the_message_not_a_traceback(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(GOOD + "\n{broken\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2"):
        bench.questions_or_exit(p, 10)


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


def ids(*dirs):
    return [d[0] for d in navtree.documents([str(x) for x in dirs])]


def test_e3_notes_md_and_notes_txt_get_distinct_ids(tmp_path):
    (tmp_path / "notes.md").write_text("# A\n\ntext one", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text two", encoding="utf-8")
    assert ids(tmp_path) == ["notes", "notes.txt"]


def test_e3_one_file_name_under_two_inputs_gets_distinct_ids(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "readme.md").write_text("# R\n\ntext " + d, encoding="utf-8")
    got = ids(tmp_path / "a", tmp_path / "b")
    assert len(got) == 2 and len(set(got)) == 2


def test_e3_different_names_keep_the_plain_id(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\ntext", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\ntext", encoding="utf-8")
    assert ids(tmp_path) == ["a", "b"]


def agree_cache_default(monkeypatch):
    """The default of agree's --cache as the command builds it."""
    seen = {}

    def grab(self, args=None, namespace=None):
        seen["cache"] = self.get_default("cache")
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        agree.main([])
    return Path(seen["cache"])


@pytest.mark.parametrize("value", ["", "cache/here"])
def test_e4_an_empty_or_relative_xdg_cache_home_is_ignored_by_the_cache_option(monkeypatch, value):
    monkeypatch.setenv("XDG_CACHE_HOME", value)
    assert agree_cache_default(monkeypatch) == Path.home() / ".cache" / "basal" / "agree"


def test_e4_an_absolute_one_is_used_by_the_cache_option(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert agree_cache_default(monkeypatch) == tmp_path / "basal" / "agree"


def test_e4_none_gives_the_home_cache_in_the_cache_option(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert agree_cache_default(monkeypatch) == Path.home() / ".cache" / "basal" / "agree"


@pytest.fixture
def ansi(monkeypatch):
    """The Windows default (cp1252) for a read_text / write_text without an encoding."""
    read, write = pathlib.Path.read_text, pathlib.Path.write_text
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, encoding=None, errors=None: read(self, encoding or "cp1252", errors))
    monkeypatch.setattr(pathlib.Path, "write_text",
                        lambda self, data, encoding=None, errors=None, newline=None: write(self, data, encoding or "cp1252", errors, newline))


def test_e6_the_tokenizer_and_config_of_a_utf8_model_are_read_under_ansi(tmp_path, ansi):
    tok = {"model": {"type": "BPE", "vocab": {"\u0101": 0, "b": 1}, "merges": []}, "added_tokens": []}  # U+0101 is C4 81 in UTF-8
    (tmp_path / "tokenizer.json").write_bytes(json.dumps(tok, ensure_ascii=False).encode("utf-8"))
    (tmp_path / "config.json").write_bytes(json.dumps({"architectures": ["\u0101"], "hidden_size": 8}, ensure_ascii=False).encode("utf-8"))
    assert interpolate.tokenizer_key(tmp_path / "tokenizer.json")[1] == {"\u0101": 0, "b": 1}
    assert interpolate.tokenizer_settings(tmp_path / "tokenizer.json")["model"]["vocab"]["\u0101"] == 0
    assert interpolate.config_key(tmp_path / "config.json")["architectures"] == ["\u0101"]


@pytest.mark.parametrize("name", ["gdn_graph", "server", "gemma4_graph", "interpolate", "calibrate", "bench", "loadtest", "agree"])
def test_e6_every_text_file_these_modules_open_names_its_encoding(name):
    """open() and read_text() / write_text() without an encoding use the locale of the machine (cp1252 on Windows)."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parent.parent / "basal" / f"{name}.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        kw = {k.arg for k in node.keywords}
        if isinstance(f, ast.Name) and f.id == "open":
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else next(
                (k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), "r")
            if "b" not in str(mode) and "encoding" not in kw:
                bad.append(node.lineno)
        elif isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text") and "encoding" not in kw:
            if not (f.attr == "write_text" and len(node.args) > 1) and not (f.attr == "read_text" and node.args):
                bad.append(node.lineno)
    assert bad == [], f"basal/{name}.py: text open without encoding at lines {bad}"
