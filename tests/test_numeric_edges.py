"""Regression tests for fix/numeric-edges (Z-1089)."""
import asyncio
import contextlib
import json
import math
import signal
import sys

import pytest

from basal import bench
from basal import calibrate as cal


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
