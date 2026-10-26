"""MLX backend with a state cache (--state-cache-gb / --prefix-cache on --mode mlx): a group cut at its state takes the
state's keys / values from the cache across requests. Cold and cached requests give the same probabilities, equal to
the PyTorch readout; single prompts are cut too; a state that begins with a cached one computes only its new tokens;
entries are copies within the byte budget."""
import gc
import random

import pytest

from basal.engine import StateCache
from tests.test_mlx import LIDS, STATE, TOKS, mlx_stub, reference, tiny_llama


def cached_stub(path, gb=1.0, packed=False):
    be = mlx_stub(path, packed=packed)
    be.state_cache = StateCache(gb * 2 ** 30)
    return be


def close(ref, got, tol=2e-4):
    return all(max(abs(x - y) for x, y in zip(r, g)) < tol for r, g in zip(ref, got))


def count_tokens(be):
    """Records the token count of every forward of the decoder."""
    seen, inner = [], be.lm.model

    class Counting:
        def __call__(self, ids, cache=None):
            seen.append(ids.shape[0] * ids.shape[1])
            return inner(ids, cache=cache)

        def __getattr__(self, name):
            return getattr(inner, name)

    be.lm.model = Counting()
    return seen


def test_cached_state_gives_the_same_answers(tmp_path):
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path)
    cold = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    warm = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    assert warm == cold  # the same forwards over the same keys / values
    assert close(reference(m, TOKS, LIDS), warm)
    st = be.state_cache.stats()
    assert (st["entries"], st["misses"], st["hits"]) == (1, 1, 1)


def test_a_cut_is_ignored_without_a_cache(tmp_path):
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    be = mlx_stub(tmp_path)
    assert be.run_shared([(TOKS, LIDS, len(STATE))])[0] == be.run_shared([(TOKS, LIDS)])[0]


def test_single_prompts_and_the_packed_path(tmp_path):
    """A single prompt with a cut reuses the cached state (it would otherwise go to the batches without a prefix); with
    --mlx-packed a cut group runs on the KV-cache path, an uncut one stays packed."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, packed=True)
    one = [TOKS[4]]
    first = be.run_shared([(one, [LIDS[4]], len(STATE))])[0]
    both = be.run_shared([(TOKS[:2], LIDS[:2], len(STATE)), (TOKS[2:4], LIDS[2:4])])
    assert be.state_cache.stats()["hits"] == 1  # the second request's cut group found the state of the first
    assert close(reference(m, one, [LIDS[4]]), first)
    assert close(reference(m, TOKS[:4], LIDS[:4]), both[0] + both[1])


def test_growing_state_computes_only_its_new_tokens(tmp_path):
    """An agent trace: the second state is the first plus a new message. It continues the cached entry (one prefix hit,
    the entry replaced, not duplicated) and reads out as a fresh forward of its prompts."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path)
    rnd = random.Random(0)
    s1 = [rnd.randrange(1, 64) for _ in range(300)]  # longer than one 256-token block of the prefix index
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(40)]
    q = [[5, 6, 10, 11], [5, 6, 11, 10]]
    lids = [[10, 11], [10, 11]]
    be.run_shared([([s1 + x for x in q], lids, len(s1))])
    seen = count_tokens(be)
    got = be.run_shared([([s2 + x for x in q], lids, len(s2))])[0]
    st = be.state_cache.stats()
    assert st["prefix_hits"] == 1 and st["entries"] == 1
    assert seen[0] == len(s2) - len(s1)  # the state's forward: its 40 new tokens only
    assert close(reference(m, [s2 + x for x in q], lids), got)


def test_entries_are_copies_within_the_budget(tmp_path):
    """An entry holds exactly its state's keys / values, not the 256-token buffers of mlx-lm's KVCache they were
    computed in (a slice would keep its whole buffer alive), and the least recently used state is evicted when the next
    one would exceed the budget."""
    mx = pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    be = cached_stub(tmp_path)
    be.run_shared([(TOKS, LIDS)])  # warm-up without the cache
    gc.collect(); mx.clear_cache()
    before = mx.get_active_memory()
    be.run_shared([(TOKS, LIDS, len(STATE))])
    gc.collect(); mx.clear_cache()
    (kv, size, _), = be.state_cache.entries.values()
    assert mx.get_active_memory() - before < 1.5 * size  # the buffers hold 256 of the state's 120 tokens
    layers, kv_heads, head = 2, 2, 16  # tiny_llama
    assert size == 2 * layers * kv_heads * len(STATE) * head * kv[0][0].itemsize
    assert all(k.shape[2] == len(STATE) for k, _ in kv)
    be.state_cache = StateCache(size)  # room for one state
    other = [t + 1 for t in STATE]
    be.run_shared([(TOKS, LIDS, len(STATE))])
    be.run_shared([([other + t[len(STATE):] for t in TOKS], LIDS, len(other))])
    st = be.state_cache.stats()
    assert (st["entries"], st["evictions"]) == (1, 1)


def test_same_answers_as_without_the_cache_in_fp32(tmp_path):
    """The cached run splits the shared prefix at the cut (the state, then the rest of the prefix over it), so in bf16 /
    quantised weights it may differ from the uncached run in the last bits; in fp32 (the tiny model) only by rounding,
    for a repeated state and for a state that grows from a cached one."""
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    plain, cached = mlx_stub(tmp_path), cached_stub(tmp_path)
    rnd = random.Random(1)
    s1 = [rnd.randrange(1, 64) for _ in range(270)]
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(30)]
    worst = 0.0
    for state in (STATE, STATE, s1, s2):
        toks = [state + t[len(STATE):] for t in TOKS]
        ref = plain.run_shared([(toks, LIDS)])[0]
        got = cached.run_shared([(toks, LIDS, len(state))])[0]
        worst = max(worst, max(abs(x - y) for r, g in zip(ref, got) for x, y in zip(r, g)))
    assert worst < 1e-5
    assert cached.state_cache.stats()["prefix_hits"] == 1


@pytest.mark.parametrize("new", [2, 5, 8])
def test_a_continuation_never_runs_2_to_8_tokens(tmp_path, new):
    """MLX's fused attention loses bf16 precision for a few queries over a cache (MLXBackend.MIN_CONTINUE): a state that
    grows by 2-8 tokens starts its forward earlier, over a shorter part of the cached entry."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path)
    rnd = random.Random(2)
    s1 = [rnd.randrange(1, 64) for _ in range(300)]
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(new)]
    q, lids = [[5, 6, 10, 11], [5, 6, 11, 10]], [[10, 11], [10, 11]]
    be.run_shared([([s1 + x for x in q], lids, len(s1))])
    seen = count_tokens(be)
    got = be.run_shared([([s2 + x for x in q], lids, len(s2))])[0]
    assert seen[0] == be.MIN_CONTINUE
    assert close(reference(m, [s2 + x for x in q], lids), got)
