"""Degenerate counts of basal-agree: no single questions, no multi readouts, --per-call below 1."""
from basal import agree


def test_no_single_questions_gives_a_row_without_decision_agreement():
    readouts = [[0.6, 0.4], [0.3, 0.7]]
    ref = {"single": [], "multi": [readouts]}
    row = agree.score(("single", "multi"), ref, ref, None, None)
    assert "decision_agreement" not in row and row["agreement"] == 1.0


def test_no_multi_readouts_gives_a_row_without_independence():
    ref = {"single": [[[0.6, 0.4], [0.4, 0.6]]], "multi": []}
    row = agree.score(("single", "multi"), ref, ref, [], None)
    assert "independence" not in row and row["decision_agreement"] == 1.0
