"""Degenerate counts of basal-agree: no single questions, no multi readouts, --per-call below 1."""
import pytest

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


@pytest.mark.parametrize("n", [0, -2])
def test_per_call_below_1_is_refused(n):
    with pytest.raises(SystemExit, match="per-call"):
        agree.run_groups(object(), [1, 2], n)


def test_per_call_0_is_refused_before_any_model_is_loaded(monkeypatch, capsys):
    import basal.engine

    def no_model(*a, **k):
        raise AssertionError("a model was resolved")
    monkeypatch.setattr(basal.engine, "resolve", no_model)
    with pytest.raises(SystemExit) as e:
        agree.main(["--model", "m", "--backends", "eager", "--per-call", "0"])
    assert e.value.code == 2 and "--per-call: 0: must be at least 1" in capsys.readouterr().err
