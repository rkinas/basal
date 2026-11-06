"""basal-selfcheck, basal-agree and basal-loadtest --determinism count a NaN or infinite readout as the largest change,
not as a comparison that max() and < skip."""
from basal import agree
from basal import selfcheck


def test_a_nan_answer_is_not_an_agreement():
    agrees, outside_ties, max_dp, _, _ = selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": float("nan"), "b": 0.4})
    assert agrees is False and outside_ties is False and max_dp == 1.0


def test_a_clean_answer_still_agrees():
    assert selfcheck.compare({"a": 0.6, "b": 0.4}, {"a": 0.61, "b": 0.39})[0] is True


def test_one_nan_readout_of_four_disagrees():
    ref = [[0.6, 0.4]] * 4
    got = [[float("nan"), 0.4]] + [[0.6, 0.4]] * 3
    c = agree.compare(ref, got)
    assert c["agreement"] == 0.75 and c["max_dp"] == 1.0


def test_clean_readouts_agree():
    c = agree.compare([[0.6, 0.4]] * 4, [[0.6, 0.4]] * 4)
    assert c["agreement"] == 1.0 and c["max_dp"] == 0.0
