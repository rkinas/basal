"""Counts below 1 are refused by the HTTP tools' argument parsers."""
import sys

import pytest

from basal import calibrate
from basal import run as run_module


@pytest.mark.parametrize("tool, argv", [
    (run_module, ["basal-run", "--input", "x", "--output", "y", "--concurrency", "0"]),
    (calibrate, ["x.jsonl", "--concurrency", "0"])])
def test_zero_concurrency_is_refused(monkeypatch, capsys, tool, argv):
    """--concurrency 0 made a semaphore that admits nothing: the run waited for ever."""
    with pytest.raises(SystemExit) as e:
        if tool is calibrate:
            calibrate.main(argv)
        else:
            monkeypatch.setattr(sys, "argv", argv)
            tool.main()
    assert e.value.code == 2 and "must be at least 1" in capsys.readouterr().err
