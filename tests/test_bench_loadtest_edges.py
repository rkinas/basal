"""Regression tests for fix/bench-loadtest-edges (Z-1082)."""
import json

import pytest

from basal import agree
from basal import bench


def test_h5_a_gold_that_is_not_an_option_index_is_left_out_and_said(capsys):
    qs = [dict(gold=True, options=["a", "b"]), dict(gold=1, options=["a", "b"]), dict(gold=-1, options=["a", "b"]),
          dict(gold=7, options=["a", "b"]), dict(gold="1", options=["a", "b"]), dict(options=["a", "b"])]
    assert agree.gold_items(qs, 10) == [(11, 1)]
    assert "4 item(s)" in capsys.readouterr().err


@pytest.mark.parametrize("gold", [7, -1, True, False, "B"])
def test_bg_a_gold_that_is_not_an_option_index_stops_the_bench(tmp_path, gold):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=g))
                              for g in [0] * 19 + [gold]), encoding="utf-8")
    qs = bench.load_questions(str(path), 50)  # the shared reader does not judge the gold
    assert len(qs) == 20
    with pytest.raises(SystemExit, match="option index"):
        bench.check_gold(qs)


def test_bg_valid_golds_and_questions_without_a_gold_are_loaded(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=1)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=["a", "b"])), encoding="utf-8")
    qs = bench.load_questions(str(path), 50)
    bench.check_gold(qs)
    assert len(qs) == 2


def test_bg_options_that_are_not_a_list_stop_the_bench_with_file_and_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=0)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=5, gold=1)), encoding="utf-8")
    qs = bench.load_questions(str(path), 50)
    with pytest.raises(SystemExit, match=r"q\.jsonl:2: options must be a list, not int"):
        bench.check_gold(qs)


def test_bg_the_gold_message_names_file_and_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=0)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=["a", "b"], gold="B")), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2: gold 'B'"):
        bench.check_gold(bench.load_questions(str(path), 50))
