"""Z-1113: a labelled file in UTF-16 (PowerShell 5) is read like any other."""
import json

import pytest

from basal import calibrate as cal


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16"])
def test_z3_a_labelled_file_in_utf_16_is_read(tmp_path, enc):
    q = {"type": "noul", "instructions": "Late?", "gold": "yes"}
    p = tmp_path / "x.jsonl"
    p.write_bytes((json.dumps({"id": "r1", "state": "s", "questions": {"q": q}}) + "\n").encode(enc))
    assert len(cal.read(str(p))[1]) == 1


def test_z3_think_and_from_rows_read_through_textio():
    """think (--data) and calibrate --from-rows read inside a command that needs a model: checked in the source."""
    import inspect

    from basal import think
    assert "read_text(a.data)" in inspect.getsource(think) and "read_text(a.input)" in inspect.getsource(cal)
