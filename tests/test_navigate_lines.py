"""A tree file and a --queries file are split on "\\n" only: U+2028, U+2029 and U+0085 may stand raw in a JSON string."""
import json

import basal.client
from basal import navigate
from basal.navigate import Tree

ODD = "a b c\u0085d"
NODES = [{"id": "r", "text": "root"}, {"id": "x", "parent": "r", "text": ODD}]


def test_a_tree_with_u2028_u2029_u0085_in_a_text_is_saved_and_loaded(tmp_path):
    path = tmp_path / "t.jsonl"
    Tree(NODES).save(path)  # ensure_ascii=False: the characters are written raw
    assert ODD in path.read_text(encoding="utf-8")
    assert Tree.load(path).nodes["x"]["text"] == ODD


def test_a_tree_file_with_crlf_loads(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes("\r\n".join(json.dumps(n, ensure_ascii=False) for n in NODES).encode("utf-8") + b"\r\n")
    assert Tree.load(path).nodes["x"]["text"] == ODD


def test_a_queries_file_with_those_characters_is_read_line_by_line(tmp_path, monkeypatch, capsys):
    seen = []

    class Stub:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Nav:
        def __init__(self, *a, **k):
            pass

        def search(self, query, **k):
            seen.append(query)
            return {"passages": [], "stages": [], "latency_ms": 0}

    monkeypatch.setattr(basal.client, "Basal", Stub)
    monkeypatch.setattr(navigate, "Navigator", Nav)
    tree, qs = tmp_path / "t.jsonl", tmp_path / "q.jsonl"
    Tree(NODES).save(tree)
    qs.write_bytes(("\r\n".join(json.dumps({"id": i, "query": ODD + str(i)}, ensure_ascii=False)
                                for i in range(2)) + "\r\n").encode("utf-8"))
    navigate.main(["search", str(tree), "--queries", str(qs)])
    assert seen == [ODD + "0", ODD + "1"]
