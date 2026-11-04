"""MLX backend with a state cache (--state-cache-gb / --prefix-cache on --mode mlx): a group cut at its state takes the
state's keys / values from the cache across requests. Cold and cached requests give the same probabilities, equal to
the PyTorch readout; single prompts are cut too; a state that begins with a cached one computes only its new tokens;
entries are copies within the byte budget. Each test runs with mlx-lm's forward and with basal's (--mlx-forward
basal), which continues the cached keys / values in its own loop."""
import gc
import random

import pytest

from basal.engine import MLXBackend, StateCache
from tests.test_mlx import FORWARDS, LIDS, STATE, TOKS, mlx_stub, reference, tiny_llama


def cached_stub(path, gb=1.0, packed=False, forward="mlx-lm", continuing=False):
    """continuing: a growing state continues its cached entry whatever the row counts (MLXBackend._kernels: on the tiny
    fp32 model the attention sums by the key count, so it would be computed afresh), for the tests of how it does."""
    be = mlx_stub(path, packed=packed, forward=forward)
    be.state_cache = StateCache(gb * 2 ** 30)
    if continuing:
        be._kernels = lambda ref: ()
    return be


def close(ref, got, tol=2e-4):
    return all(max(abs(x - y) for x, y in zip(r, g)) < tol for r, g in zip(ref, got))


def count_tokens(be):
    """Records the token count of every forward of the decoder (basal's forward: of every call of its loop)."""
    if be.fwd is not None:
        seen, run = [], be.fwd._run

        def counting(ids, *a, **kw):
            seen.append(ids.shape[0] * ids.shape[1])
            return run(ids, *a, **kw)
        be.fwd._run = counting
        return seen
    seen, inner = [], be.lm.model

    class Counting:
        def __call__(self, ids, cache=None):
            seen.append(ids.shape[0] * ids.shape[1])
            return inner(ids, cache=cache)

        def __getattr__(self, name):
            return getattr(inner, name)

    be.lm.model = Counting()
    return seen


@FORWARDS
def test_cached_state_gives_the_same_answers(tmp_path, forward):
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, forward=forward)
    cold = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    warm = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    assert warm == cold  # the same forwards over the same keys / values
    assert close(reference(m, TOKS, LIDS), warm)
    st = be.state_cache.stats()
    assert (st["entries"], st["misses"], st["hits"]) == (1, 1, 1)


@FORWARDS
def test_a_cut_is_ignored_without_a_cache(tmp_path, forward):
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    be = mlx_stub(tmp_path, forward=forward)
    assert be.run_shared([(TOKS, LIDS, len(STATE))])[0] == be.run_shared([(TOKS, LIDS)])[0]


@FORWARDS
def test_single_prompts_and_the_packed_path(tmp_path, forward):
    """A single prompt with a cut reuses the cached state (it would otherwise go to the batches without a prefix); with
    --mlx-packed a cut group runs on the KV-cache path, an uncut one stays packed. (continuing: the two requests' row
    counts give the same bits, so the second request may take the first one's state.)"""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, packed=True, forward=forward, continuing=True)
    one = [TOKS[4]]
    first = be.run_shared([(one, [LIDS[4]], len(STATE))])[0]
    both = be.run_shared([(TOKS[:2], LIDS[:2], len(STATE)), (TOKS[2:4], LIDS[2:4])])
    assert be.state_cache.stats()["hits"] == 1  # the second request's cut group found the state of the first
    assert close(reference(m, one, [LIDS[4]]), first)
    assert close(reference(m, TOKS[:4], LIDS[:4]), both[0] + both[1])


@FORWARDS
def test_growing_state_computes_only_its_new_tokens(tmp_path, forward):
    """An agent trace: the second state is the first plus a new message. It continues the cached entry (one prefix hit,
    the entry replaced, not duplicated) and reads out as a fresh forward of its prompts."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, forward=forward, continuing=True)
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


@FORWARDS
def test_entries_are_copies_within_the_budget(tmp_path, forward):
    """An entry holds exactly its state's keys / values, not the 256-token buffers of mlx-lm's KVCache they were
    computed in (a slice would keep its whole buffer alive), and the least recently used state is evicted when the next
    one would exceed the budget."""
    mx = pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    be = cached_stub(tmp_path, forward=forward)
    be.run_shared([(TOKS, LIDS)])  # warm-up without the cache
    gc.collect(); mx.clear_cache()
    before = mx.get_active_memory()
    be.run_shared([(TOKS, LIDS, len(STATE))])
    gc.collect(); mx.clear_cache()
    (kv, size, *_), = be.state_cache.entries.values()  # (kv, size, tokens[, blocks])
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


@FORWARDS
def test_same_answers_as_without_the_cache_in_fp32(tmp_path, forward):
    """The cached run splits the shared prefix at the cut (the state, then the rest of the prefix over it), so in bf16 /
    quantised weights it may differ from the uncached run in the last bits; in fp32 (the tiny model) only by rounding,
    for a repeated state and for a state that grows from a cached one."""
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    plain, cached = mlx_stub(tmp_path, forward=forward), cached_stub(tmp_path, forward=forward, continuing=True)
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


@FORWARDS
@pytest.mark.parametrize("new", [1, 2, 8, 15])
def test_a_continuation_never_runs_fewer_than_min_continue_tokens(tmp_path, new, forward):
    """MLX's fused attention takes other kernels for a few queries over a cache (MLXBackend.MIN_CONTINUE): a state that
    grows by fewer tokens starts its forward earlier, over a shorter part of the cached entry."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, forward=forward, continuing=True)
    rnd = random.Random(2)
    s1 = [rnd.randrange(1, 64) for _ in range(300)]
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(new)]
    q, lids = [[5, 6, 10, 11], [5, 6, 11, 10]], [[10, 11], [10, 11]]
    be.run_shared([([s1 + x for x in q], lids, len(s1))])
    seen = count_tokens(be)
    got = be.run_shared([([s2 + x for x in q], lids, len(s2))])[0]
    assert seen[0] == be.MIN_CONTINUE
    assert close(reference(m, [s2 + x for x in q], lids), got)


@FORWARDS
def test_a_growing_state_is_continued_only_with_the_same_bits(tmp_path, forward, monkeypatch):
    """With 8-bit weights a state that grows from a cached one continues it only when the cached rows have the bits
    this request's uncached forward gives them (MLXBackend._kernels the same for both row counts); otherwise it is
    computed afresh, reads out as the uncached forward bit for bit and still replaces the entry it extends."""
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    rnd = random.Random(3)
    s1 = [rnd.randrange(1, 64) for _ in range(300)]
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(40)]
    q, lids = [[5, 6, 10, 11], [5, 6, 11, 10]], [[10, 11], [10, 11]]
    plain = mlx_stub(tmp_path, "q8", forward=forward)
    for same, first in ((False, len(s2)), (True, 40)):
        be = mlx_stub(tmp_path, "q8", forward=forward)
        be.state_cache = StateCache(2 ** 30)
        if same:  # the bits of both row counts the same (on the tiny model they are not)
            monkeypatch.setattr(be, "_kernels", lambda ref: ())
        be.run_shared([([s1 + x for x in q], lids, len(s1))])
        assert be._kernels(len(s1) + 2) != be._kernels(len(s2) + 2) or same
        seen = count_tokens(be)
        got = be.run_shared([([s2 + x for x in q], lids, len(s2))])[0]
        assert seen[0] == first
        st = be.state_cache.stats()
        assert (st["prefix_hits"], st["entries"]) == (int(same), 1)  # the entry it extends replaced
        if not same:
            assert got == plain.run_shared([([s2 + x for x in q], lids)])[0]


def test_kernel_rows_finds_where_the_row_count_stops_mattering():
    """_kernel_rows bisects for the fewest rows from which a product gives each row the bits it has at PROBE_ROWS rows
    (a fake kernel that sums in another order below 100 rows), and gives None when no such boundary checks out."""
    mx = pytest.importorskip("mlx.core")
    be = MLXBackend.__new__(MLXBackend)
    be.mx, be.PROBE_ROWS = mx, 512
    w = mx.random.normal((8, 16), key=mx.random.key(1))

    def below_100(x):
        return x @ w.T + (1e-3 if x.shape[0] < 100 else 0.0)

    def odd(x):  # depends on the row count everywhere
        return x @ w[:4].T + (1e-3 if x.shape[0] % 2 else 0.0)
    assert be._kernel_rows(below_100, mx.zeros((1, 16)), w, {"transpose": True}) == 100
    assert be._kernel_rows(odd, mx.zeros((1, 16)), w[:4], {"transpose": True}) is None


def test_uncut_runs_products_at_the_uncached_row_count(monkeypatch):
    """Inside _uncut(rows, ref) a linear layer over `rows` rows (quantised or dense) runs at the fewest rows where its
    kernel stops depending on the row count, at most ref (or at ref when no such count is known), and gives the rows of
    that larger product; products over other row counts run as they are, and both are restored afterwards."""
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    be = MLXBackend.__new__(MLXBackend)
    be.mx = mx
    real, dense, seen = mx.quantized_matmul, nn.Linear.__call__, []

    def spy(x, w, **kw):
        seen.append(x.shape[-2])
        return real(x, w, **kw)

    def dense_spy(mod, x):
        seen.append(x.shape[-2])
        return dense(mod, x)
    monkeypatch.setattr(mx, "quantized_matmul", spy)
    monkeypatch.setattr(nn.Linear, "__call__", dense_spy)
    w, scales, biases = mx.quantize(mx.random.normal((32, 64), key=mx.random.key(0)), group_size=64, bits=8)
    kw = dict(scales=scales, biases=biases, transpose=True, group_size=64, bits=8)
    lin = nn.Linear(64, 24)
    x = mx.random.normal((1, 5, 64), key=mx.random.key(2))
    for fewest, ref, rows in ((16, 40, 16), (16, 12, 12), (None, 40, 40), (4, 40, 5)):
        be._kernel_rows = lambda f, x, w, kw, n=fewest: n
        with be._uncut(5, ref):
            y = mx.quantized_matmul(x, w, **kw)
            mx.quantized_matmul(x[:, :1], w, **kw)
            z = lin(x)
        assert seen[-3:] == [rows, 1, rows] and y.shape == (1, 5, 32) and z.shape == (1, 5, 24)
        xp = mx.concatenate([x, mx.zeros((1, rows - 5, 64))], axis=1)
        assert mx.array_equal(y, real(xp, w, **kw)[:, :5]).item()
        assert mx.array_equal(z, dense(lin, xp)[:, :5]).item()
        assert mx.quantized_matmul is spy and nn.Linear.__call__ is dense_spy


@pytest.mark.parametrize("mask", ["causal", "window"])
def test_uncut_attention_runs_over_the_uncached_key_count(monkeypatch, mask):
    """Inside _uncut(rows, ref) an attention of `rows` queries over fewer than ref keys runs over ref keys (zero
    padding, masked for the real rows) and gives the rows of the unpadded call (fp32, within rounding); other calls
    run as they are. Once per shape it checks whether its rows depend on the key count: the same call over more
    padding keys (KEY_CHECKS), then its first half over its own keys."""
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.base import create_causal_mask
    be = MLXBackend.__new__(MLXBackend)
    be.mx = mx
    real, seen = mx.fast.scaled_dot_product_attention, []

    def spy(q, k, v, **kw):
        seen.append((q.shape[2], k.shape[2]))
        return real(q, k, v, **kw)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    q, k, v = (mx.random.normal((1, h, n, 16), key=mx.random.key(i)) * 0.3
               for i, (h, n) in enumerate([(4, 40), (2, 70), (2, 70)]))
    m = "causal" if mask == "causal" else create_causal_mask(40, 30, window_size=8)
    with be._uncut(40, 96):
        got = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25, mask=m)
        mx.fast.scaled_dot_product_attention(q[:, :, :1], k, v, scale=0.25, mask=None)
    checks = [(66 + e, 96 + e) for e in be.KEY_CHECKS] + [(20, 50)]  # 40 // 3 rows: too few queries to check
    assert seen[0] == (66, 96) and seen[-1] == (1, 70)  # padded; as it is
    assert seen[1:-1] == checks[: len(seen) - 2] and len(seen) > 2
    assert mx.fast.scaled_dot_product_attention is spy and len(be._attn_keys) == 1
    assert mx.allclose(got, real(q, k, v, scale=0.25, mask=m), atol=1e-6).item()


@FORWARDS
def test_cut_forwards_give_the_uncached_bits_with_8_bit_weights(tmp_path, forward):
    """With quantised weights (MLX picks the matmul kernel by the row count) a state computed for the request itself
    gives the uncached probabilities bit for bit, cold and cached: a SOAM group, and a single prompt (the rest of it in
    one forward over the state, read out at its last token, as the uncached path reads it)."""
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    plain, cached = mlx_stub(tmp_path, "q8", forward=forward), mlx_stub(tmp_path, "q8", forward=forward)
    cached.state_cache = StateCache(2 ** 30)
    other = [t + 1 for t in STATE]
    one = [other + [3]]
    for _ in range(2):  # cold, then from the cache
        assert cached.run_shared([(TOKS, LIDS, len(STATE))])[0] == plain.run_shared([(TOKS, LIDS)])[0]
        assert cached.run_shared([(one, LIDS[4:], len(other))])[0] == plain.run_shared([(one, LIDS[4:])])[0]
    st = cached.state_cache.stats()
    assert (st["entries"], st["misses"], st["hits"]) == (2, 2, 2)


@FORWARDS
def test_a_single_prompt_runs_its_rest_in_one_forward(tmp_path, forward):
    """A single prompt with a cut: the state, then everything after it in one forward (not the rest of the prefix and
    then its last token); from the cache, the rest only."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = cached_stub(tmp_path, forward=forward)
    one = [STATE + list(range(7, 26)) + [3]]
    seen = count_tokens(be)
    cold = be.run_shared([(one, [LIDS[4]], len(STATE))])[0]
    assert seen == [len(STATE), 20]
    seen.clear()
    assert be.run_shared([(one, [LIDS[4]], len(STATE))])[0] == cold and seen == [20]
    assert close(reference(m, one, [LIDS[4]]), cold)


@FORWARDS
def test_a_state_cached_by_another_request_gives_the_uncached_bits(tmp_path, forward):
    """Several requests about one state (the usual use of the cache): the state is cached by a single question, then
    read by a request of other questions. With 8-bit weights its answers equal the server's without the cache bit for
    bit (docs/HARDWARE.md, Apple M4 state cache): an entry whose rows were computed at another row count is not
    continued (MLXBackend._kernels), whether the root matches it exactly or only begins with it."""
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    plain, cached = mlx_stub(tmp_path, "q8", forward=forward), mlx_stub(tmp_path, "q8", forward=forward)
    cached.state_cache = StateCache(2 ** 30)
    one = [STATE + list(range(7, 26)) + [3]]
    assert cached.run_shared([(one, LIDS[4:], len(STATE))])[0] == plain.run_shared([(one, LIDS[4:])])[0]
    assert cached._kernels(len(one[0])) != cached._kernels(len(TOKS[0]))  # other bits at the group's row count
    for _ in range(2):  # computed afresh at the group's row count, then from its own entry
        assert cached.run_shared([(TOKS, LIDS, len(STATE))])[0] == plain.run_shared([(TOKS, LIDS)])[0]
    assert cached.run_shared([(one, LIDS[4:], len(STATE))])[0] == plain.run_shared([(one, LIDS[4:])])[0]
    st = cached.state_cache.stats()
    assert (st["entries"], st["misses"], st["hits"]) == (2, 2, 2)  # one entry per row count, each found again
