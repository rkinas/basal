"""Regression tests for fix/calibrate-nan (Z-1082)."""
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
