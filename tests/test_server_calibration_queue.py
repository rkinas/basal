"""Regression tests for fix/server-calibration-queue (Z-1082)."""
import pytest

from basal.server import check_calibration


def refused(cal):
    try:
        check_calibration(cal, 1, "CAL.json")
        return False
    except SystemExit:
        return True


@pytest.mark.parametrize("T", [0, 0.0, -1.0, float("nan"), float("inf"), True, "2"])
def test_i09_a_temperature_that_is_not_a_finite_number_above_zero_is_refused_at_start_up(T):
    assert refused({"prompt": 1, "temperature_per_prim": {"choice": T}})
    assert refused({"prompt": 1, "temperature_per_bucket": {"choice": {"2-10": T}}})


@pytest.mark.parametrize("cal", [{"temperature_per_prim": {"choice": 1.5}}, {"temperature_per_prim": {"choice": 2}},
                                 {"temperature_per_bucket": {"choice": {"2-10": 2.0}}}, {}])
def test_i09_a_valid_calibration_is_accepted(cal):
    assert not refused({"prompt": 1, **cal})
