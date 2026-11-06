"""Reading the text files a person edits (calibration, API keys, JSONL input), on Windows too."""
import io


def open_text(path, newline=None):
    """A text file a person wrote, opened once: UTF-16 when it starts with a UTF-16 BOM (PowerShell 5 `>` and
    Out-File), else UTF-8 with or without a BOM (Notepad). The BOM is peeked at, not read, so a pipe or a FIFO
    (`--input <(jq ...)`, /dev/stdin) keeps every byte. newline: as for open() (None: universal newlines)."""
    f = open(path, "rb")
    try:
        codec = "utf-16" if f.peek(2)[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        return io.TextIOWrapper(f, encoding=codec, newline=newline)
    except BaseException:
        f.close()
        raise


def read_text(path):
    """The whole text of open_text(path), with universal newlines as Path.read_text."""
    with open_text(path) as f:
        return f.read()
