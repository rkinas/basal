"""basal-selfcheck's verdicts: a 200 that is not JSON or not an answer, a near tie, unanswered questions, and a run
where most requests were cut by the context."""
import contextlib
import io
import json
import urllib.request

import pytest

from basal import selfcheck as sc


def test_a_200_that_is_not_json_is_a_clean_exit(monkeypatch):
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502</html>"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Reply())
    with pytest.raises(SystemExit, match="not JSON"):
        sc.call("http://x", "/v1/models")


def test_a_near_tie_is_forgiven_only_when_the_server_picks_one_of_the_tied_pair():
    assert sc.compare({"a": .40, "b": .39, "c": .21}, {"a": .10, "b": .20, "c": .70})[1] is False
    assert sc.compare({"a": .40, "b": .39, "c": .21}, {"a": .38, "b": .40, "c": .22})[1] is True
    assert sc.compare({"a": .6, "b": .3, "c": .1}, {"a": .1, "b": .3, "c": .6})[1] is False


GOOD = {"probabilities": {"true": 0.9, "false": 0.1}}
NAMES = ("u", "t", "c")


def run_check(monkeypatch, tmp_path, answers_for, n_items=40, gate=0.98):
    items = [{"id": f"i{i}", "state": "s", "questions": {k: {} for k in NAMES},
              "answers": {k: {"type": "noul", "probabilities": {"true": 0.9, "false": 0.1}} for k in NAMES}}
             for i in range(n_items)]
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"model": "m", "engine": "e", "basal": "x", "note": "", "items": items}), encoding="utf-8")

    def fake_call(url, path, body=None, timeout=300):
        return {"models": [{"name": "m", "mode": "x"}]} if path == "/v1/models" else answers_for(body)
    monkeypatch.setattr(sc, "call", fake_call)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = sc.check("http://x", str(path), gate, "")
    return rc


def test_questions_the_server_left_unanswered_are_not_a_pass(monkeypatch, tmp_path):
    assert run_check(monkeypatch, tmp_path, lambda b: {"answers": {k: GOOD for k in NAMES}, "usage": {}}) == 0
    assert run_check(monkeypatch, tmp_path, lambda b: {"answers": {"u": GOOD}, "usage": {}}) != 0


def test_a_run_where_almost_every_request_was_cut_by_the_context_is_not_a_pass(monkeypatch, tmp_path):
    n = {"n": 0}

    def cut_most(body):
        n["n"] += 1
        return {"answers": {k: GOOD for k in NAMES}, "usage": {"truncated": n["n"] > 1}}
    assert run_check(monkeypatch, tmp_path, cut_most) != 0


def test_a_tie_inside_the_tolerance_is_agreement():
    # docs/OLD_GPUS.md: a flip between options the reference had within 0.02 of each other counts as agreement
    assert sc.compare({"a": .34, "b": .33, "c": .33}, {"a": .30, "b": .30, "c": .40})[1] is True
    assert sc.compare({"a": .34, "b": .33, "c": .33}, {"a": .30, "b": .40, "c": .30})[1] is True
    # the boundary: exactly TIE apart is not forgiven, and a clearly lower option never is
    assert sc.compare({"a": .40, "b": .20, "c": .40}, {"a": .1, "b": .8, "c": .1})[1] is False


@pytest.mark.parametrize("reply", [{"error": "overloaded"}, None, [], "x", {"answers": None}, {"answers": []}])
def test_a_200_json_that_is_not_an_answer_object_is_a_failed_request_not_a_traceback(monkeypatch, tmp_path, reply):
    with pytest.raises(SystemExit, match="no readouts compared"):
        run_check(monkeypatch, tmp_path, lambda b: reply, n_items=4)
    n = {"n": 0}

    def one_bad(body):
        n["n"] += 1
        return reply if n["n"] == 2 else {"answers": {k: GOOD for k in NAMES}, "usage": {}}
    assert run_check(monkeypatch, tmp_path, one_bad, n_items=4) == 1


@pytest.mark.parametrize("answer", [{"type": "noul"}, None, [], "x", {"probabilities": None}, {"probabilities": []}])
def test_an_answer_without_probabilities_is_a_missing_answer(monkeypatch, tmp_path, answer):
    answers = lambda b: {"answers": {"u": GOOD, "t": GOOD, "c": answer}, "usage": {}}  # noqa: E731
    assert run_check(monkeypatch, tmp_path, lambda b: {"answers": {k: GOOD for k in NAMES}, "usage": {}}) == 0
    assert run_check(monkeypatch, tmp_path, answers) == 1
