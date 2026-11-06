"""Reading the text files a person edits (calibration, API keys, JSONL input), on Windows too."""
from pathlib import Path


def encoding(path):
    """The codec of a text file a person wrote: "utf-16" when it starts with a UTF-16 BOM (PowerShell 5 `>` and
    Out-File), else "utf-8-sig" (UTF-8 with or without a BOM, as Notepad saves it)."""
    with open(path, "rb") as f:
        return "utf-16" if f.read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"


def read_text(path):
    """UTF-8 with or without a BOM (Notepad), UTF-16 with a BOM (PowerShell 5 `>` and Out-File)."""
    return Path(path).read_text(encoding=encoding(path))
