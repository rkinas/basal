"""Regression tests for fix/calibrate-nan (Z-1082)."""
import json
import math
import random
import time

import numpy as np
import pytest

from basal import calibrate as cal


def rows(n, K=4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = np.exp(rng.normal(size=K) * 2)
        p /= p.sum()
        out.append({"p": p.tolist(), "gold": int(rng.choice(K, p=p))})
    return out


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_i10_a_row_with_a_non_finite_probability_is_skipped(bad):
    clean = rows(400)
    row = dict(clean[0], p=[bad, 0.2, 0.3, 0.5])
    assert cal.fit_temperature([row] + clean[1:]) == pytest.approx(cal.fit_temperature(clean[1:]))


def test_i10_rows_that_are_all_non_finite_are_refused():
    with pytest.raises(ValueError):
        cal.fit_temperature([dict(r, p=[float("nan")] * 4) for r in rows(20)])


def test_i10_clean_calibrated_rows_still_fit_near_one():
    assert 0.8 <= cal.fit_temperature(rows(400)) <= 1.25


@pytest.mark.parametrize("targets", ["1e-9", "1e-17", "0"])
def test_h1_targets_below_one_in_a_million_are_refused_at_the_parser(targets):
    with pytest.raises(SystemExit, match="targets"):
        cal.main(["/nonexistent.jsonl", "--targets", targets])


def test_h1_the_smallest_accepted_target_costs_little_and_normal_ones_pass_the_parser():
    t = time.time()
    assert cal.accepted_needed(1e-4, 1) > 0 and time.time() - t < 60
    assert cal.accepted_needed(0.05) == 59
    with pytest.raises(BaseException) as e:
        cal.main(["/nonexistent.jsonl", "--targets", "0.05"])
    assert "targets" not in str(e.value)


def test_h2_a_bucket_whose_temperature_sits_on_the_edge_of_the_grid_is_warned_about():
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


def test_h4_lines_with_a_null_id_are_not_one_group(tmp_path):
    q = {"type": "noul", "instructions": "Late?", "gold": "yes"}
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps({"id": None, "state": "s", "questions": {"q": q}}) for _ in range(5)),
                    encoding="utf-8")
    _, decisions, _, _ = cal.read(str(path))
    assert len({d["group"] for d in decisions}) == 5


# --- second round (Z-1094): the NaN is left out where the rows are read, counted, reported ---
def labelled(n, p=(0.7, 0.3), gold=0, off=0):
    return [dict(prim="choice", p=list(p), gold=gold if i % 3 else 1, group=str(off + i), family="f") for i in range(n)]


def test_c1_nan_rows_do_not_enter_the_ece_the_certification_or_the_counts_and_are_reported():
    good = labelled(400)
    nan = labelled(200, p=(float("nan"), 0.3), off=1000)
    _, clean = cal.fit(good, {}, [0.05], min_per_type=50)
    _, dirty = cal.fit(good + nan, {}, [0.05], min_per_type=50)
    for k in ("n_fit", "n_cert"):
        assert dirty[k] == clean[k]
    assert dirty["types"]["choice"]["ece"] == clean["types"]["choice"]["ece"]
    assert dirty["n_nonfinite"] == 200 and clean["n_nonfinite"] == 0
    assert any("200 row(s) with a NaN" in w for w in cal.warnings(dirty, [0.05]))
    assert not any("NaN" in w for w in cal.warnings(clean, [0.05]))


def test_c1_finite_rows_counts_what_it_leaves_out_and_keeps_inf_out_too():
    rs = labelled(5) + labelled(2, p=(float("inf"), 0.0)) + labelled(3, p=(0.5, float("nan")))
    keep, n = cal.finite_rows(rs)
    assert len(keep) == 5 and n == 5


def test_c1_a_type_whose_rows_are_nearly_all_nan_is_not_fitted_on_the_one_that_is_left():
    rs = labelled(1, off=0) + labelled(29, p=(float("nan"), 0.3), off=100)
    cal_, report = cal.fit(rs, {"temperature_per_prim": {"choice": 1.7}}, [0.05], min_per_type=30)
    assert cal_["temperature_per_prim"]["choice"] == 1.7 and report["n_nonfinite"] == 29


def test_c2_rows_that_are_all_nan_exit_cleanly_from_rows(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps({"basal_calibrate_rows": 1}) + "\n"
                 + "\n".join(json.dumps(dict(r, p=[float("nan"), 0.1])) for r in labelled(80)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no row with finite probabilities"):
        cal.main([str(f), "--from-rows"])


def test_c2_fit_on_all_nan_rows_is_a_refusal_the_callers_can_name():
    with pytest.raises(cal.InputError):
        cal.fit(labelled(80, p=(float("nan"), 0.1)), {}, [0.05])


def _bucket_report(source, T):
    return {"n_fit": 500, "n_cert": 500, "types": {}, "targets": {},
            "buckets": {"2-4": {"T": T, "T_type": False, "source": source}}}


@pytest.mark.parametrize("T", [8.0, 0.25])
def test_c3_a_bucket_temperature_carried_over_from_the_shipped_file_is_not_warned_about(T):
    assert not any("edge" in w for w in cal.warnings(_bucket_report("shipped (3 FIT decisions < 30)", T), []))
    assert not any("edge" in w for w in cal.warnings(_bucket_report("type temperature (3 FIT decisions < 30)", T), []))
    assert any("edge" in w for w in cal.warnings(_bucket_report("fitted on 80 FIT decisions", T), []))
