"""Edges of basal-agree, basal-bench and basal-loadtest: golds that are no option index, determinism under load."""
import json
import random

import pytest

from basal import agree
from basal import bench
from basal.jsonl import read_lines


def item(question):
    return json.dumps(dict(state="s", question=question, options=["x", "y"]))


def test_numbered_questions_are_the_items_of_load_questions_with_their_place(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(item(f"a{i}") for i in range(5)) + "\n\n" + json.dumps({"state": "s", "questions": {}})
                 + "\n" + item("a-last") + "\n", encoding="utf-8")
    b.write_text("\n".join(item(f"b{i}") for i in range(7)), encoding="utf-8")
    located = bench.numbered_questions([a, b], 50, seed=3)
    assert [q for _, q in located] == bench.load_questions([a, b], 50, seed=3)
    before = [json.loads(x) for p in (a, b) for x in read_lines(p) if x.strip()]  # the reader as it was
    before = [q for q in before if "options" in q]
    random.Random(3).shuffle(before)
    assert [q for _, q in located] == before
    where = {q["question"]: w for w, q in located}
    assert where["a-last"] == f"{a}:8" and where["a0"] == f"{a}:1" and where["b6"] == f"{b}:7"
    assert len(bench.numbered_questions(str(a), 2)) == 2


def test_a_gold_that_is_not_an_option_index_is_left_out_and_said(capsys):
    qs = [dict(gold=True, options=["a", "b"]), dict(gold=1, options=["a", "b"]), dict(gold=-1, options=["a", "b"]),
          dict(gold=7, options=["a", "b"]), dict(gold="1", options=["a", "b"]), dict(options=["a", "b"])]
    assert agree.gold_items(qs, 10) == [(11, 1)]
    assert "4 item(s)" in capsys.readouterr().err


@pytest.mark.parametrize("gold", [7, -1, True, False, "B"])
def test_a_gold_that_is_not_an_option_index_stops_the_bench(tmp_path, gold):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=g))
                              for g in [0] * 19 + [gold]), encoding="utf-8")
    located = bench.numbered_questions(str(path), 50)  # the shared reader does not judge the gold
    assert len(located) == 20
    with pytest.raises(SystemExit, match="option index"):
        bench.check_gold(located)


def test_valid_golds_and_questions_without_a_gold_are_loaded(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=1)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=["a", "b"])), encoding="utf-8")
    located = bench.numbered_questions(str(path), 50)
    bench.check_gold(located)
    assert len(located) == 2


def test_options_that_are_not_a_list_stop_the_bench_with_file_and_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=0)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=5, gold=1)), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2: options must be a list, not int"):
        bench.check_gold(bench.numbered_questions(str(path), 50))


def test_the_gold_message_names_file_and_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(dict(state="s", question="q", options=["a", "b"], gold=0)) + "\n"
                    + json.dumps(dict(state="s", question="q", options=["a", "b"], gold="B")), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"q\.jsonl:2: gold 'B'"):
        bench.check_gold(bench.numbered_questions(str(path), 50))
