"""Regression tests for fix/numeric-edges (Z-1089)."""
import asyncio
import contextlib
import math
import signal

import pytest

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
