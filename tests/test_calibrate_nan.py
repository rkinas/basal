"""Regression tests for fix/calibrate-nan (Z-1082)."""
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
