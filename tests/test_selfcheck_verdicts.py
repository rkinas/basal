"""Regression tests for fix/selfcheck-verdicts (Z-1082)."""
import contextlib
import io
import json
import urllib.request

import pytest

from basal import selfcheck as sc


def test_sc4_a_200_that_is_not_json_is_a_clean_exit(monkeypatch):
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


def test_sc1_a_near_tie_is_forgiven_only_when_the_server_picks_one_of_the_tied_pair():
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


def test_sc2_questions_the_server_left_unanswered_are_not_a_pass(monkeypatch, tmp_path):
    assert run_check(monkeypatch, tmp_path, lambda b: {"answers": {k: GOOD for k in NAMES}, "usage": {}}) == 0
    assert run_check(monkeypatch, tmp_path, lambda b: {"answers": {"u": GOOD}, "usage": {}}) != 0
