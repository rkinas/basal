"""basal-bench reports the 95th percentile of the two-order latency as the nearest rank. Pure logic: a fake timer, no
torch model, no network."""
from basal import bench as bench_module


def groups(n=14):
    return [({}, [0, 1], ["a", "b"], [[1, 2], [1, 2]]) for _ in range(n)]


class Quiet:
    """Not an HTTP engine and nothing to time: only the calls of measure() are counted."""

    def run(self, prompts, ids_list):
        return []

    def run_shared(self, groups, policy=None):
        return []


def test_lat2_p95_is_the_nearest_rank(monkeypatch):
    """With 20 samples the 95th percentile is the 19th smallest (index 18), not the largest (index 19)."""
    calls = []

    def fake_timed(fn, reps, be_=None):
        calls.append(1)
        return [len(calls) / 1000]  # the single-order calls are 1, 3, 5 ms; the two-order calls 2, 4, 6 ... 50 ms

    monkeypatch.setattr(bench_module, "timed", fake_timed)
    monkeypatch.setattr(bench_module, "decisions", lambda g, res: [])
    monkeypatch.setattr(bench_module, "sync", lambda be_: None)
    _, out = bench_module.measure(Quiet(), groups(40), 20)  # 25 groups, the first 5 samples of each kind are dropped
    assert out["lat2_p95_ms"] == 48.0, out
