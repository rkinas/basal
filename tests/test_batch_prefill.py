"""Count model inputs, excluding graph capture, and check shared cold roots against separate forwards."""
import random

import pytest
import torch

from basal.engine import StateCache
from tests.test_engine import tiny
from tests.test_engine16 import Stub, check
from tests.test_state_cache import requests


def observe_prefills(be, monkeypatch):
    seen, capturing = [], False
    capture, forward = be._capture, be.fwd

    def observed(ids, *args, **kw):
        if not capturing:
            seen.append(ids.clone())
        return forward(ids, *args, **kw)

    def captured(*args):
        nonlocal capturing
        capturing = True
        try:
            return capture(*args)
        finally:
            capturing = False

    monkeypatch.setattr(be, "fwd", observed)
    monkeypatch.setattr(be, "_forward", observed)
    monkeypatch.setattr(be, "_capture", captured)
    return seen


def test_repeated_roots_without_state_cache_keep_all_prefill_rows(monkeypatch):
    # --deterministic refuses StateCache, so shared roots must retain their original batch shape.
    be = Stub(tiny())
    assert be.state_cache is None
    rnd = random.Random(42)
    state = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    gs = [requests(rnd, state)[2] for _ in range(3)]
    assert be._cache_keys(gs, be._units(gs)) == (None, {})
    seen = observe_prefills(be, monkeypatch)
    first = check(be, gs)
    assert [tuple(ids.shape) for ids in seen] == [(3, 32)]
    assert all(row[:21].tolist() == state for row in seen[0])
    assert be.run_shared(gs) == first


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("bits,budget", [(16, 1 << 24), (8, 1 << 24), (16, 0), (8, 1)])
def test_identical_cold_roots_run_once(monkeypatch, warm, bits, budget):
    be = Stub(tiny())
    seen = observe_prefills(be, monkeypatch)
    if warm:
        be.SUFFIX_LENS = [8, 16]
        be.warm()
        seen.clear()
    be.state_cache = StateCache(budget, bits=bits)
    rnd = random.Random(42)
    state = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    gs = requests(rnd, state)
    be.replays.clear()
    first = check(be, gs)
    assert [tuple(ids.shape) for ids in seen] == [(1, 32)]
    assert seen[0][0, :21].tolist() == state
    assert be.state_cache.stats()["hits"] == 0
    assert all(rows == 1 for kind, rows, _ in be.replays if kind == "tree")
    seen.clear()
    again = be.run_shared(gs)
    if budget > 1:
        assert not seen and be.state_cache.stats()["hits"] == 3
        for a, b in zip(first, again):
            for x, y in zip(a, b):
                assert torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-5 if bits == 16 else 1e-2)
    else:
        assert [tuple(ids.shape) for ids in seen] == [(1, 32)]


@pytest.mark.parametrize("bits", [16, 8])
@pytest.mark.parametrize("budget", [0, 1 << 24])
def test_reuse_across_batches_with_mixed_cold_roots(monkeypatch, bits, budget):
    be = Stub(tiny())
    be.state_cache = StateCache(budget, bits=bits)
    rnd = random.Random(5)
    states = [[1] + [rnd.randrange(3, 50) for _ in range(20)] for _ in range(3)]
    gs = [requests(rnd, states[i], 1)[0] for i in (0, 0, 0, 1, 2, 0)]
    monkeypatch.setattr(be, "_forwards", lambda *a: [([0, 1], "tree"), ([2, 3, 4], "tree"), ([5], "tree")])
    tree, locals_seen = be._tree_hidden, []

    def observed_tree(*args):
        locals_seen.append(args[-1])
        return tree(*args)

    monkeypatch.setattr(be, "_tree_hidden", observed_tree)
    seen = observe_prefills(be, monkeypatch)
    check(be, gs)
    assert [tuple(ids.shape) for ids in seen] == [(1, 32)] * 3
    assert [r[:21].tolist() for ids in seen for r in ids] == states
    assert all(not memo for memo in locals_seen)  # the last consumer releases each checkpoint


def test_split_groups_keep_original_batching_and_complete_once(monkeypatch):
    be = Stub(tiny())
    be.state_cache = StateCache(0)
    rnd = random.Random(12)
    state = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    other = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    gs = [requests(rnd, state)[2], requests(rnd, other)[1], requests(rnd, other)[1]]
    monkeypatch.setattr(be, "_oversize", lambda pack, n: n > 2)
    assert len(be._units(gs)) > len(gs)
    seen = observe_prefills(be, monkeypatch)
    done = []
    got = be.run_shared(gs, done=lambda k, result: done.append((k, result)))
    assert sorted(done) == list(enumerate(got))
    assert len(done) == len(gs)
    assert [tuple(ids.shape) for ids in seen] == [(3, 32), (1, 32)]
    assert [rows for kind, rows, _ in be.replays if kind == "tree"] == [3, 1]
    check(be, gs)


def test_uncached_root_readout_keeps_its_forward_row(monkeypatch):
    be = Stub(tiny())
    be.state_cache = StateCache(0)
    rnd = random.Random(6)
    state = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    gs = requests(rnd, state, 2)
    other = [1, 8, 7, 6]
    gs.append(([other + [20], other + [21], other], [[10, 11]] * 3))
    monkeypatch.setattr(be, "_forwards", lambda *a: [([0], "tree"), ([1, 2], "tree")])
    check(be, gs)


def test_root_readout_still_computes_hidden_states():
    be = Stub(tiny())
    be.state_cache = StateCache(1 << 24)
    state = [1, 8, 7, 6]
    gs = [([state, state + [20], state + [21]], [[10, 11]] * 3, len(state))] * 2
    assert be._cache_keys(gs, be._units(gs)) == ([be.state_cache.key(state)] * 2, {})
    check(be, gs)
    check(be, gs)


def test_unique_states_and_mixed_persistent_hits(monkeypatch):
    be = Stub(tiny())
    be.state_cache = StateCache(1 << 24)
    rnd = random.Random(9)
    states = [[1] + [rnd.randrange(3, 50) for _ in range(20)] for _ in range(3)]
    seen = observe_prefills(be, monkeypatch)
    check(be, requests(rnd, states[0], 1))
    seen.clear()
    gs = [requests(rnd, states[i], 1)[0] for i in (0, 1, 2)]
    done = []
    got = be.run_shared(gs, done=lambda k, result: done.append((k, result)))
    assert [r[:21].tolist() for ids in seen for r in ids] == states[1:]
    assert sorted(done) == list(enumerate(got))
    seen.clear()
    check(be, gs)
    assert not seen


def test_long_roots_reuse_one_eager_prefill(monkeypatch):
    be = Stub(tiny())
    be.state_cache = StateCache(0)
    rnd = random.Random(7)
    state = [1] + [rnd.randrange(3, 50) for _ in range(60)]
    seen = observe_prefills(be, monkeypatch)
    check(be, requests(rnd, state))
    assert [tuple(ids.shape) for ids in seen] == [(1, 61)]


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("captured", [False, True])
def test_gemma_sliding_layers_share_raw_checkpoints(monkeypatch, kind, captured):
    from tests.test_gemma4_graph import Stub as GemmaStub, separate, tiny as gemma_tiny

    be = GemmaStub(gemma_tiny(kind, torch.float32))
    be.fwd = be._forward
    if captured:
        be.PLAIN_LENS = be.LENS = [16, 32]
        be.BATCHES, be.TREE_ROWS = [1, 2, 3, 4], [1, 2, 4]
        be._alloc_kv()
        monkeypatch.setattr(be, "_capture", Stub._capture.__get__(be))
    be.state_cache = StateCache(1 << 24, bits=8)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    rnd = random.Random(10)
    gs = requests(rnd, state)
    seen = observe_prefills(be, monkeypatch)
    got = be.run_shared(gs)
    assert [tuple(ids.shape) for ids in seen] == [(1, 16 if captured else len(state))]
    for (tokens, letters, _), group in zip(gs, got):
        for p, ref in zip(group, separate(be.model, tokens, letters)):
            assert torch.allclose(torch.tensor(p), ref, atol=1e-6)


@pytest.mark.parametrize("captured", [False, True])
def test_gdn_reuses_kv_and_recurrent_states(monkeypatch, tmp_path, captured):
    from tests.test_engine16 import Replay
    from tests.test_recurrent import gdn_backend, max_dp, tiny_qwen35

    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    rnd = random.Random(8)
    states = [[1] + [rnd.randrange(3, 50) for _ in range(20)] for _ in range(3)]
    gs = [requests(rnd, states[i])[2] for i in (0, 0, 0, 1, 2, 0)]
    gs[4] = ([states[2], states[2] + [20], states[2] + [21]], [[10, 11]] * 3, len(states[2]))
    reference = be.run_shared([g[:2] for g in gs])
    be.state_cache = StateCache(0)
    if captured:
        be.pool = object()  # CPU replay of the actual static-buffer graphs

        def capture(key, fn, *static):
            r = Replay(fn, static)
            be.graphs[key] = (r, *static, r.out)
            return be.graphs[key]

        monkeypatch.setattr(be, "_capture", capture)
    monkeypatch.setattr(be, "_forwards", lambda *a: [([0, 1], "tree"), ([2, 3, 4], "tree"), ([5], "tree")])
    seen = observe_prefills(be, monkeypatch)
    assert max_dp(be.run_shared(gs), reference) < 1e-5
    assert [r[:21].tolist() for ids in seen for r in ids if r[0] != 0] == states
    assert sum(ids.shape[0] for ids in seen) == 3
