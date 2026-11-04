"""Z-1113: every JSON/JSONL file a person edits is read with a BOM or in UTF-16."""
import json

import pytest

ENC = {"utf-8": lambda s: s.encode(), "utf-8-bom": lambda s: ("\ufeff" + s).encode("utf-8"),
       "utf-16": lambda s: s.encode("utf-16")}  # PowerShell 5 `>` and Out-File


# Z3: every file a person edits is read through textio.read_text (BOM and UTF-16), not only the two B-023 named.
def _jsonl(rows, enc):
    return ENC[enc]("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("enc", ENC)
def test_z3_calibrate_reference_file(tmp_path, enc):
    from basal import calibrate
    path = tmp_path / "ref.json"
    path.write_bytes(ENC[enc](json.dumps({"gates": {"ece_lofo": {"choice": {"mean": 0.05}}}})))
    assert calibrate.reference_values(str(path))[0] == {"choice": 0.05}


@pytest.mark.parametrize("enc", ENC)
def test_z3_selfcheck_reference_file(tmp_path, enc, monkeypatch):
    from basal import selfcheck
    path = tmp_path / "ref.json"
    path.write_bytes(ENC[enc](json.dumps({"checks": []})))

    def past_the_read(url):
        raise RuntimeError("read ok")
    monkeypatch.setattr(selfcheck, "server_info", past_the_read)
    with pytest.raises(RuntimeError, match="read ok"):  # a JSON or decode error would stop before the server
        selfcheck.check("http://x", str(path), None, "")


@pytest.mark.parametrize("enc", ENC)
def test_z3_navigate_tree_file(tmp_path, enc):
    from basal import navigate
    cls = next(c for c in vars(navigate).values() if isinstance(c, type) and hasattr(c, "load") and c.__module__ == navigate.__name__)
    path = tmp_path / "tree.jsonl"
    path.write_bytes(_jsonl([{"id": "a", "parent": None, "text": "t"}], enc))
    try:
        cls.load(str(path))
    except (KeyError, TypeError, ValueError) as e:  # a row of another shape is not the point; the decoding is
        assert "BOM" not in str(e) and "decode" not in str(e) and "Expecting value" not in str(e)


@pytest.mark.parametrize("module", ["agree", "identity", "gguf_export", "run", "finetune", "navigate", "selfcheck", "calibrate"])
def test_z3_user_file_readers_go_through_textio(module):
    """The readers deep inside a command (they need a model): checked in the source, no plain utf-8 read of a user file."""
    import ast
    import inspect
    import importlib
    src = inspect.getsource(importlib.import_module(f"basal.{module}"))
    assert "read_text" in src and "textio import read_text" in src
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
            enc = [k.value.value for k in node.keywords if k.arg == "encoding" and isinstance(k.value, ast.Constant)]
            # model-side files (config.json, tokenizer.json) keep their encoding; a user file never does
            assert enc != ["utf-8"] or not any(w in ast.unparse(node.func.value) for w in ("a.questions", "a.input", "reference", "cal_path", "path)"))
