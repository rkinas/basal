"""basal-bench times HTTP engines (llama-server, Ollama) on prompts they have to compute: their prompt cache answers a
prompt seen a moment ago almost for free, and no call of the benchmark can clear it (bench.cold only reaches vLLM /
SGLang). Pure logic: a fake engine with a llama-server-like prompt cache, no torch model, no network."""
import itertools

import pytest

from basal import bench as bench_module


def common(a, b):
    return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(a, b)))


class SlotBackend:
    """A request computes the tokens after the longest common prefix with a cached prompt (llama.cpp: at least the
    last token). keep=1: one slot that holds the last prompt; keep=None: every prompt seen stays cached (several
    slots, or a prompt cache that outlives the slot). Prompts are strings, one character is one token."""

    http = object()  # as _HTTPLetters: marks an HTTP engine

    def __init__(self, keep):
        self.keep, self.cache, self.timing, self.timed_calls = keep, [], False, []

    def run(self, prompts, ids_list):
        for k, p in enumerate(prompts):
            lcp = max((common(c, p) for c in self.cache), default=0)
            if self.timing and k == 0:  # the second order of a call may reuse the first one's state, as in serving
                self.timed_calls.append(max(len(p) - lcp, 1) / len(p))
            self.cache = (self.cache + [p])[-self.keep:] if self.keep else self.cache + [p]
        return [[1.0] * len(i) for i in ids_list]

    def run_shared(self, groups, policy=None):
        return [self.run(p, i) for p, i in groups]


def groups(n=14):
    out = []
    for g in range(n):
        state = f"state {g} " * 40
        prompts = [f"{state}question {g}: options {o}" for o in ("ab", "ba")]
        out.append(({}, [0, 1], prompts, [[1, 2], [1, 2]]))
    return out


@pytest.mark.parametrize("keep", [1, None], ids=["one-slot", "every-prompt"])
def test_timed_calls_of_an_http_engine_compute_their_prompt(monkeypatch, keep):
    be = SlotBackend(keep)
    timed = bench_module.timed

    def watch(fn, reps, be_=None):
        be.timing = True
        try:
            return timed(fn, reps, be_)
        finally:
            be.timing = False

    monkeypatch.setattr(bench_module, "timed", watch)
    monkeypatch.setattr(bench_module, "decisions", lambda g, res: [])
    monkeypatch.setattr(bench_module, "sync", lambda be_: None)
    bench_module.measure(be, groups(), 3)
    assert len(be.timed_calls) == 8 + 8 + 14  # single-order, two-order, then every group of the throughput call
    assert min(be.timed_calls) > 0.5, be.timed_calls  # a prompt answered from the cache computes ~1 token of 300+


class Quiet:
    """Not an HTTP engine and nothing to time: only the calls of measure() are counted."""

    def run(self, prompts, ids_list):
        return []

    def run_shared(self, groups, policy=None):
        return []


@pytest.mark.parametrize("n, p95", [(19, 48.0), (20, 48.0), (21, 50.0), (40, 86.0)])
def test_lat2_p95_is_the_nearest_rank(monkeypatch, n, p95):
    """The n two-order samples are 12, 14, 16 ... ms; the 95th percentile is the ceil(0.95 n)-th smallest: with 20
    samples the 19th (48 ms), not the largest (50 ms); with 40 the 38th (86 ms), not the 39th (88 ms)."""
    calls = []

    def fake_timed(fn, reps, be_=None):
        calls.append(1)
        return [len(calls) / 1000]  # the single-order calls are 1, 3, 5 ms; the two-order calls 2, 4, 6 ... ms

    monkeypatch.setattr(bench_module, "timed", fake_timed)
    monkeypatch.setattr(bench_module, "decisions", lambda g, res: [])
    monkeypatch.setattr(bench_module, "sync", lambda be_: None)
    _, out = bench_module.measure(Quiet(), groups(n + 5), n)  # the first 5 samples of each kind are dropped
    assert out["lat2_p95_ms"] == p95, out
