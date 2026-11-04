"""Regression tests for fix/nan-comparators (Z-1089)."""
import json

import pytest

from basal import agree
from basal import bench
from basal import selfcheck


def test_e2_a_nan_answer_is_not_an_agreement():
    agrees, outside_ties, max_dp, _, _ = selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": float("nan"), "b": 0.4})
    assert agrees is False and outside_ties is False and max_dp == 1.0


def test_e2_a_clean_answer_still_agrees():
    assert selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": 0.61, "b": 0.39})[0] is True


def test_e3_one_nan_readout_of_four_disagrees():
    ref = [[0.6, 0.4]] * 4
    got = [[float("nan"), 0.4]] + [[0.6, 0.4]] * 3
    c = agree.compare(ref, got)
    assert c["agreement"] == 0.75 and c["max_dp"] == 1.0


def test_e3_clean_readouts_agree():
    c = agree.compare([[0.6, 0.4]] * 4, [[0.6, 0.4]] * 4)
    assert c["agreement"] == 1.0 and c["max_dp"] == 0.0


def test_v3_nan_and_inf_become_null():
    row = bench.finite({"mode": "gguf", "mem_gb": float("nan"), "x": float("inf"), "lat_ms": 1.5, "n": 3})
    assert row == {"mode": "gguf", "mem_gb": None, "x": None, "lat_ms": 1.5, "n": 3}
    json.dumps(row, allow_nan=False)
