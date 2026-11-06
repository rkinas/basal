"""Edges of basal-agree, basal-bench and basal-loadtest: golds that are no option index, determinism under load."""
import argparse
import asyncio
import contextlib
import io
import json
import random
import sys

import pytest

from basal import agree
from basal import bench
from basal import loadtest as lt
from basal import metrics
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


class Reply:
    status_code = 200

    def __init__(self, n_answers=2):
        self.n = n_answers

    def raise_for_status(self):
        pass

    def json(self):
        return {"answers": {f"q{i}": {"probabilities": {"a": .5, "b": .5}} for i in range(self.n)}}


class Client:
    hook = staticmethod(lambda count: Reply())
    count = 0

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        Client.count += 1
        return Client.hook(Client.count)


@pytest.fixture
def fake_http(monkeypatch):
    Client.count, Client.hook = 0, staticmethod(lambda count: Reply())
    monkeypatch.setattr(lt.httpx, "AsyncClient", Client)
    monkeypatch.setattr(lt, "questions_or_exit", lambda p, n: [dict(state="s", question="q", options=["a", "b"])] * n)
    return Client


def ns(**kw):
    d = dict(url="http://x", questions=[], synthetic=None, long_frac=0.0, seed=0, questions_per_request=1,
             n_seq=3, n_conc=2, concurrency=4, early_exit=None, out=None, determinism=0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_determinism_warns_when_the_mixed_pass_repeats_the_probe(fake_http, monkeypatch):
    seen = {}

    async def fake_determinism(a, bodies):
        seen["same"] = bodies[:a.determinism] == bodies[a.determinism:2 * a.determinism]
    monkeypatch.setattr(lt, "determinism", fake_determinism)
    monkeypatch.setattr(sys, "argv", ["lt", "--determinism", "4"])
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        lt.main()
    assert not seen["same"] or "distinct" in err.getvalue()


def determinism_report(capsys, answers_for):
    Client.count, Client.hook = 0, staticmethod(answers_for)
    body = lt.body(dict(state="s", question="q", options=["a", "b"]))
    asyncio.run(lt.determinism(ns(determinism=4, concurrency=2), [body] * 8))
    return json.loads(capsys.readouterr().out)["determinism"]


def test_an_answer_that_vanishes_under_load_is_not_identical(fake_http, capsys):
    # the first 4 calls are the sequential "alone" pass, the rest run under load and lose an answer
    assert determinism_report(capsys, lambda c: Reply(2 if c <= 4 else 1))["identical"] < 4


def test_equal_answers_are_identical(fake_http, capsys):
    assert determinism_report(capsys, lambda c: Reply(2))["identical"] == 4


def test_help_text_is_escaped_like_a_label_value():
    out = metrics.prometheus([("basal_x", "a\\b\nc", "gauge", [({}, 1)])])
    assert out.count("\n") == 3 and "a\\\\b\\nc" in out
