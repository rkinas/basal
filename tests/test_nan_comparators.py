"""Regression tests for fix/nan-comparators (Z-1089)."""
import pytest

from basal import selfcheck


def test_e2_a_nan_answer_is_not_an_agreement():
    agrees, outside_ties, max_dp, _, _ = selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": float("nan"), "b": 0.4})
    assert agrees is False and outside_ties is False and max_dp == 1.0


def test_e2_a_clean_answer_still_agrees():
    assert selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": 0.61, "b": 0.39})[0] is True
