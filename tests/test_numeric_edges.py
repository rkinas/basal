"""Regression tests for fix/numeric-edges (Z-1089)."""
import asyncio
import contextlib
import json
import math
import signal
import sys

import pytest

from basal import agree
from basal import bench
from basal import calibrate as cal
from basal import synth
from basal import tasks
from basal.client import Basal


@contextlib.contextmanager
def within(seconds):
    """A hang becomes a failure (SIGALRM; not on Windows)."""
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("needs SIGALRM")

    def stop(*_):
        raise TimeoutError(f"still running after {seconds} s")
    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("n", [0, -1])
def test_z1_concurrency_below_1_is_refused_not_a_hang(n):
    async def decide(b):
        return {}
    with within(2), pytest.raises(ValueError, match="concurrency"):
        asyncio.run(cal.collect(decide, [{}], concurrency=n))


def test_z1_concurrency_1_still_collects():
    async def decide(b):
        return {"ok": 1}
    assert asyncio.run(cal.collect(decide, [{}, {}], concurrency=1)) == [{"ok": 1}, {"ok": 1}]


@pytest.mark.parametrize("level,n", [(0.9, 9), (0.95, 19), (0.99, 99)])
def test_z2_the_usual_levels_keep_their_counts(level, n):
    assert cal.conformal_needed(level) == n


def test_z2_a_level_next_to_one_does_not_walk_from_n_1():
    level = 0.999999999999
    with within(2):
        n = cal.conformal_needed(level)
    assert math.ceil(round((n + 1) * level, 9)) <= n


def questions_file(tmp_path, n):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps({"state": "s", "options": ["a", "b"], "gold": 0}) for _ in range(n)), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("n,lat_n", [(3, 1), (10, 0)])
def test_z3_too_few_questions_or_lat_n_0_stop_before_the_model(tmp_path, monkeypatch, n, lat_n):
    argv = ["basal-bench", "--questions", questions_file(tmp_path, n), "--n", str(n), "--lat-n", str(lat_n), "--modes", "eager-fp32"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(bench, "resolve", lambda *a, **k: object())
    with pytest.raises(SystemExit, match="at least 6 questions"):
        bench.main()


def test_z4_no_single_questions_gives_a_row_without_decision_agreement():
    readouts = [[0.6, 0.4], [0.3, 0.7]]
    ref = {"single": [], "multi": [readouts]}
    row = agree.score(("single", "multi"), ref, ref, None, None)
    assert "decision_agreement" not in row and row["agreement"] == 1.0


def test_z4_no_multi_readouts_gives_a_row_without_independence():
    ref = {"single": [[[0.6, 0.4], [0.4, 0.6]]], "multi": []}
    row = agree.score(("single", "multi"), ref, ref, [], None)
    assert "independence" not in row and row["decision_agreement"] == 1.0


@pytest.mark.parametrize("n", [0, -2])
def test_z5_per_call_below_1_is_refused(n):
    with pytest.raises(SystemExit, match="per-call"):
        agree.run_groups(object(), [1, 2], n)


@pytest.mark.parametrize("integer", [False, True])
def test_z6_a_number_of_400_digits_is_none(integer):
    assert tasks.parse_number("9" * 400, integer=integer) is None


def test_z6_an_ordinary_number_still_parses():
    assert tasks.parse_number("12 345") == 12345


@pytest.mark.parametrize("options", [0, 1, -3])
def test_z7_options_below_2_are_refused(options):
    with pytest.raises(ValueError, match="options"):
        synth.items(3, options=options)


def test_z7_two_options_still_work():
    assert len(synth.items(3, options=2)) == 3


@pytest.mark.parametrize("backoff", [-1, float("nan"), True, "1"])
def test_z8_a_bad_backoff_is_refused_in_the_constructor(backoff):
    with pytest.raises(ValueError, match="backoff"):
        Basal("http://basal/", backoff=backoff)


@pytest.mark.parametrize("backoff", [0, 0.5, 2])
def test_z8_a_good_backoff_is_accepted(backoff):
    assert Basal("http://basal/", backoff=backoff).backoff == backoff
