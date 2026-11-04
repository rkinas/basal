"""Text files are opened as UTF-8 in the tools and CLIs that read or write Polish text: without an encoding, open()
and Path.read_text / write_text use the locale's (cp1252 on Windows), which cannot write "ą" and misreads tokenizer
vocabularies. A static check over the source (no tool is run)."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted([*ROOT.glob("tools/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("docs/**/*.py"),
                ROOT / "basal" / "interpolate.py"])


def text_calls_without_encoding(path):
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name == "open" and isinstance(f, ast.Name):
            mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            if isinstance(mode, ast.Constant) and "b" in mode.value or len(node.args) > 2:
                continue
            out.append(f"{path.relative_to(ROOT)}:{node.lineno}: open")
        elif name in ("read_text", "write_text") and isinstance(f, ast.Attribute) and len(node.args) < 2:
            out.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_text_is_utf8(path):
    assert text_calls_without_encoding(path) == []
