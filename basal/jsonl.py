"""JSONL input as JSON writes it: one record per "\n"-terminated line."""
from pathlib import Path


def read_lines(path):
    """The lines of a UTF-8 JSONL file, split at "\n" only (a "\r" before it is dropped; no empty last line).

    str.splitlines also splits at U+2028, U+2029 and U+0085, which json.dumps(..., ensure_ascii=False) leaves
    unescaped inside strings (text pasted from a word processor or a PDF), and so cut a valid line in two."""
    out = [x[:-1] if x.endswith("\r") else x for x in Path(path).read_text(encoding="utf-8").split("\n")]
    if out and not out[-1]:
        out.pop()
    return out
