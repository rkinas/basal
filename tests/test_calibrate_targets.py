"""basal-calibrate's --targets bound and its warnings about bucket temperatures on the edge of the grid; the counts
of a --from-rows refit."""
import json
import math
import random
import time

import pytest

from basal import calibrate as cal


@pytest.mark.parametrize("targets", ["1e-9", "1e-17", "0"])
def test_targets_below_one_in_a_million_are_refused_at_the_parser(targets):
    with pytest.raises(SystemExit, match="targets"):
        cal.main(["/nonexistent.jsonl", "--targets", targets])


def test_the_smallest_accepted_target_costs_little_and_normal_ones_pass_the_parser():
    t = time.time()
    assert cal.accepted_needed(1e-4, 1) > 0 and time.time() - t < 60
    assert cal.accepted_needed(0.05) == 59
    with pytest.raises(BaseException) as e:
        cal.main(["/nonexistent.jsonl", "--targets", "0.05"])
    assert "targets" not in str(e.value)


def test_a_bucket_whose_temperature_sits_on_the_edge_of_the_grid_is_warned_about():
    rows = [dict(prim="choice", p=[0.999, 0.001], gold=0, group=str(i), family="f") for i in range(300)]
    rnd = random.Random(1)
    for i in range(300, 600):
        g = 0 if rnd.random() < 0.5 else rnd.randrange(1, 12)
        rows.append(dict(prim="choice", p=[0.5] + [0.5 / 11] * 11, gold=g, group=str(i), family="f"))
    _, report = cal.fit(rows, {}, [0.05], min_per_type=50, fmt=2)
    edge = [b for b, r in report["buckets"].items()
            if r.get("T") and not r.get("T_type") and any(math.isclose(r["T"], x) for x in cal.T_GRID)]
    assert edge, "the fixture no longer puts a bucket on the edge"
    assert any(f"choice {b} options" in w for b in edge for w in cal.warnings(report, [0.05]))


def test_a_refit_counts_only_the_rows_it_fits(tmp_path, capsys):
    """Rows that fit() leaves out (NaN probabilities, a gold out of range) are reported, and no longer counted in
    fitted_on or in the sample-size check of --from-rows."""
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=1, baseline=None, orders=2)
    rnd = random.Random(0)
    good = [dict(prim="noul", group=f"g{i}", family="f", gold=int(rnd.random() < 0.8), p=[0.2, 0.8])
            for i in range(300)]
    bad = [dict(prim="noul", group=f"b{i}", family="f", gold=0, p=[float("nan"), 0.5]) for i in range(40)] + \
          [dict(prim="noul", group=f"o{i}", family="f", gold=5, p=[0.4, 0.6]) for i in range(20)]
    rows, out = tmp_path / "rows.jsonl", tmp_path / "cal.json"
    rows.write_text("".join(json.dumps(x) + "\n" for x in [head] + good + bad), encoding="utf-8")
    cal.main([str(rows), "--from-rows", "--targets", "0.01", "--out", str(out)])
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    assert fitted["decisions"] == fitted["fit"] + fitted["cert"] == 300 and fitted["multi"] == 0
    cap = capsys.readouterr()
    assert "60 row(s) left out of the fit" in cap.out
    fit_n = sum(not cal.is_cert(r["group"]) for r in good)
    assert f"this file gives {fit_n} FIT decisions of 300" in cap.err
