"""basal-calibrate with --min-per-type 0 and --min-per-set 0, and --from-rows on a file that is not a rows file:
a clear result or message, no traceback."""
import argparse

import pytest

from basal import calibrate


def row(prim="choice", k=4, group="g0"):
    return dict(prim=prim, p=[1 / k] * k, gold=0, group=group, family="f", keys=list("abcd")[:k])


def test_min_per_type_zero_does_not_fit_a_type_without_decisions():
    rows = [row() for _ in range(60)]  # no noul / score / ... decisions at all
    temps, source, _, _ = calibrate.fit_temps(rows, {}, {}, 0, 1)
    assert "choice" in temps and "noul" not in temps
    assert source["choice"].startswith("fitted on 60")


def test_min_per_type_zero_in_format_2_buckets():
    rows = [row() for _ in range(60)]
    temps, _, buckets, _ = calibrate.fit_temps(rows, {}, {}, 0, 2)
    assert "choice" in temps and isinstance(buckets, dict)


def test_min_per_set_zero_with_a_label_set_that_has_no_fit_rows():
    cert = next(g for g in ("c%d" % i for i in range(999)) if calibrate.is_cert(g))
    r = dict(prim="multi", p=[.5], keys=["x"], gold=[0], group=cert,
             set=dict(id="s", objective="f1", threshold=.5, min=0, max=None, exclusive=False, fallback="none"))
    out = calibrate.fit_fallback([r], 1.0, min_per_set=0)
    assert out["s"]["fallback_threshold"] is None and out["s"]["n_fit"] == 0
    assert out["s"]["reason"] == "0 FIT questions < 1"  # the minimum that applies, not the 0 asked for


@pytest.mark.parametrize("content", ["", "\n" + '{"basal_calibrate_rows": 1}\n', "oops\n", "[1]\n"])
def test_refit_on_a_file_that_is_not_a_rows_file_exits_with_a_message(tmp_path, content):
    path = tmp_path / "rows.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit, match="not a file written by basal-calibrate --save-rows"):
        calibrate.refit(argparse.Namespace(input=str(path)), [0.05], None, None)


def test_refit_on_a_binary_file_exits_with_a_message(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"\xff\xfe\x00\x01\n")
    with pytest.raises(SystemExit, match="not a file written by basal-calibrate --save-rows"):
        calibrate.refit(argparse.Namespace(input=str(path)), [0.05], None, None)
    with pytest.raises(SystemExit, match="not a file written by basal-calibrate --save-rows"):
        calibrate.refit(argparse.Namespace(input=str(tmp_path)), [0.05], None, None)  # a directory


@pytest.mark.parametrize("bad", ["{not json", "[1]", "7", "null"])
def test_refit_on_a_good_header_and_a_bad_row_names_the_line(tmp_path, bad):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"basal_calibrate_rows": 1}\n' + '{"prim": "choice"}\n' + bad + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"line 3 is not a row"):
        calibrate.refit(argparse.Namespace(input=str(path)), [0.05], None, None)
