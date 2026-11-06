"""Edges of basal-agree, basal-bench and basal-loadtest: golds that are no option index, determinism under load."""
from basal import agree


def test_a_gold_that_is_not_an_option_index_is_left_out_and_said(capsys):
    qs = [dict(gold=True, options=["a", "b"]), dict(gold=1, options=["a", "b"]), dict(gold=-1, options=["a", "b"]),
          dict(gold=7, options=["a", "b"]), dict(gold="1", options=["a", "b"]), dict(options=["a", "b"])]
    assert agree.gold_items(qs, 10) == [(11, 1)]
    assert "4 item(s)" in capsys.readouterr().err
