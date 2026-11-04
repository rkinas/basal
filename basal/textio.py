"""Reading the JSON files a person edits (calibration, API keys), on Windows too."""
from pathlib import Path


def read_text(path):
    """UTF-8 with or without a BOM (Notepad), UTF-16 with a BOM (PowerShell 5 `>` and Out-File)."""
    raw = Path(path).read_bytes()
    return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
