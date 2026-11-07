"""--state-cache-bits 8 / 4: the state cache keeps keys and values as integers with a step per group (keys per
channel, values per token) and gives them back in their dtype; answers over cached states stay close to the uncached
ones, and a GB holds about 2x / 3.5x as many states."""
import pytest
import torch

from basal.engine import StateCache, kv_dequant, kv_quant
from basal.server import Server, parser

from test_gemma4_graph import Stub, separate, tiny


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("T", [1, 5, 128, 300])
def test_round_trip_within_half_a_step(bits, T):
    torch.manual_seed(0)
    x = (torch.randn(3, T, 128) * torch.linspace(0.1, 20, 128)).bfloat16()  # channels of other sizes (keys)
    for keys in (True, False):
        e = kv_quant(x, bits, keys)
        y = kv_dequant(e, bits, keys)
        assert y.shape == x.shape and y.dtype == x.dtype
        _, step, _, _, _ = e
        bound = step.float().max() * 0.5 + x.float().abs().max() * 2 ** -8  # half a step + bf16 rounding
        assert (y.float() - x.float()).abs().max() <= bound
    assert e[0].numel() == 3 * T * 128 * bits // 8


def test_keys_per_channel_keep_small_channels():
    """A key channel 100x smaller than the others keeps its own step (per channel), so it survives 4 bits (half a
    step of its own range; a step shared with the other channels would be ~15x its size)."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 128)
    x[..., 0] *= 0.01
    y = kv_dequant(kv_quant(x, 4, True), 4, True)
    c = x[..., 0]
    assert (y[..., 0] - c).abs().max() <= (c.max() - c.min()) / 30 * 1.01


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("T", [37, 300])
@pytest.mark.parametrize("chunk", [None, 1])
def test_an_entry_comes_back_as_each_layer_quantised_alone(bits, T, chunk, monkeypatch):
    """The cache quantises the layers of one shape together (one call per KV_STACK_BYTES, not 16-18 kernels per
    tensor and layer); every layer it gives back, all of them, equals that layer's own kv_quant / kv_dequant round
    trip, with layers of two shapes (Gemma 4's layer types) interleaved, in one chunk per shape or (chunk 1: a limit
    of one layer and a half) in several; and longest()'s usable() check reads each layer's own shape."""
    import basal.engine as E
    torch.manual_seed(0)
    shapes = [(2, T, 128), (1, T // 2 + 1, 64)] * 3
    kv = [((torch.randn(s) * (i + 1)).bfloat16(), torch.randn(s).bfloat16()) for i, s in enumerate(shapes)]
    if chunk:
        monkeypatch.setattr(E, "KV_STACK_BYTES", kv[0][0].numel() * 3)  # 1.5 layers of the larger shape (bf16)
    c = StateCache(1 << 30, bits=bits)
    c.put(b"k", kv)
    got = c.get(b"k")
    assert len(got) == len(kv)
    for (k, v), (gk, gv) in zip(kv, got):
        assert torch.equal(gk, kv_dequant(kv_quant(k, bits, True), bits, True))
        assert torch.equal(gv, kv_dequant(kv_quant(v, bits, False), bits, False))
    if chunk:
        assert len(c.entries[b"k"][0][1]) > 2  # more chunks than shapes
    seen, c.BLOCK = [], 4
    c.put(b"t", kv, tokens=list(range(T)))
    c.longest(list(range(T)) + [0], usable=lambda kv, m, n: seen.append([(k.shape, v.shape) for k, v in kv]) or True)
    assert seen == [[(tuple(k.shape), tuple(v.shape)) for k, v in kv]]


def test_low_bit_entries_take_less_of_the_budget():
    kv = [(torch.randn(2, 256, 128).bfloat16(), torch.randn(2, 256, 128).bfloat16()) for _ in range(3)]
    sizes = {}
    for bits in (16, 8, 4):
        c = StateCache(1 << 30, bits=bits)
        c.put(b"k", kv, tokens=list(range(256)))
        sizes[bits] = c.stats()["used_bytes"]
        got = c.get(b"k")
        assert [k.shape for k, _ in got] == [k.shape for k, _ in kv] and got[0][0].dtype == torch.bfloat16
        assert c.longest(list(range(256)) + [7])[2] == 256
    assert sizes[16] / sizes[8] > 1.9 and sizes[16] / sizes[4] > 3.4
    with pytest.raises(ValueError, match="16, 8 or 4"):
        StateCache(1 << 30, bits=2)


@pytest.mark.parametrize("bits", [8, 4])
def test_cached_state_answers_close_to_separate(bits):
    """As test_gemma4_graph's second round: the second request runs pass 2 over the dequantised keys / values of the
    state; its probabilities stay close to separate prompts (8 bits: within 1e-2, 4 bits: within 5e-2)."""
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.state_cache = StateCache(1 << 30, bits=bits)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    first = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    second = [state + [22, 23, 24, 30 + o] for o in range(2)]
    lids = [[7, 8], [7, 8]]
    for toks, tol in ((first, 1e-6), (second, 1e-2 if bits == 8 else 5e-2)):
        res = be.run_shared([(toks, lids, len(state))])[0]
        for r, e in zip(res, separate(m, toks, lids)):
            assert torch.allclose(torch.tensor(r), e, atol=tol)
    s = be.state_cache.stats()
    assert s["entries"] == 1 and s["hits"] == 1


def test_bits_need_a_state_cache():
    assert parser().parse_args([]).state_cache_bits == 16
    with pytest.raises(SystemExit):  # 4 bits failed the accuracy check on max (kept in StateCache, not served)
        parser().parse_args(["--state-cache-bits", "4"])
    with pytest.raises(SystemExit, match="--state-cache-bits 8 needs"):
        Server(parser().parse_args(["--mode", "fast", "--state-cache-bits", "8"]))


def test_one_prompt_cut_at_its_state_is_cached_and_read_back():
    """A group of one prompt with a cut (one option order, as a readout of the calibration set) splits at the cut, so
    its state is stored by the first call and read from the cache by the second."""
    m = tiny("31b", torch.float32)
    be = Stub(m)
    be.state_cache = StateCache(1 << 30, bits=8)
    toks = [[1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16, 20, 21, 30]]
    groups = [(toks, [[7, 8, 9]], 12)]
    first = be.run_shared(groups)[0]
    assert be.state_cache.stats()["entries"] == 1 and be.state_cache.stats()["hits"] == 0
    second = be.run_shared(groups)[0]
    assert be.state_cache.stats()["hits"] == 1
    ref = separate(m, toks, [[7, 8, 9]])[0]
    assert torch.allclose(torch.tensor(first[0]), ref, atol=1e-6)
    assert torch.allclose(torch.tensor(second[0]), ref, atol=1e-2)


def test_longest_dequantises_only_the_winner_outside_the_lock(monkeypatch):
    """--state-cache-bits 8: longest() checks candidates on their stored shapes and dequantises only the entry it
    returns, after releasing the cache's lock (other requests' lookups do not wait on dequantisation)."""
    import torch

    from basal import engine
    cache = engine.StateCache(1 << 24, bits=8)
    cache.BLOCK = 4
    pre = [1, 2, 3, 4]
    kv = lambda T: [(torch.randn(2, T, 8), torch.randn(2, T, 8))]  # noqa: E731
    for tail in ([5, 9, 9, 9], [5, 6, 9, 9], [5, 6, 7, 9]):  # each matches the query further than the one before
        toks = pre + tail
        cache.put(cache.key(toks), kv(len(toks)), tokens=toks)
    calls, real = [], engine.kv_dequant

    def counting(e, bits, keys):
        calls.append(cache.lock.locked())
        return real(e, bits, keys)
    monkeypatch.setattr(engine, "kv_dequant", counting)
    seen = []
    found = cache.longest(pre + [5, 6, 7, 8, 0, 0], usable=lambda kv, m, n: seen.append(kv[0][0].shape) or True)
    assert found is not None and found[2] == 7 and found[1][0][0].shape == (2, 8, 8)
    assert calls == [False, False]  # keys and values of the winner, outside the lock
    assert seen == [(2, 8, 8)] * 3  # every improving candidate checked on its shape


def _layers(name, T):
    """The [(heads, tokens, head size)] of each layer a model's state cache entry holds for a state of T tokens: its own
    KV layers (Gemma 4: the first num_hidden_layers - num_kv_shared_layers; a sliding layer keeps its window + MARGIN
    tokens, Gemma4Graph.kv_lens), from the models' config.json."""
    if name == "basal-2.0-4.5B":
        return [(2, T, 128)] * 60
    if name == "basal-2.0-mini":
        return [(2, T, 128)] * 32
    if name == "basal-omni-1.0-mini":  # 42 layers, 18 share KV: 24 own, every 6th full (head size 512)
        return [(2, T, 512) if i % 6 == 5 else (2, min(T, 512 + 256), 256) for i in range(24)]
    if name == "basal-omni-1.0-max":  # 48 own: sliding 8 KV heads of 256, full 1 KV head of 512
        return [(1, T, 512) if i % 6 == 5 else (8, min(T, 1024 + 256), 256) for i in range(48)]
    if name == "basal-omni-pico":  # Qwen3.5: 6 attention layers (its Gated DeltaNet states are not quantised)
        return [(2, T, 256)] * 6
    raise KeyError(name)


@pytest.mark.parametrize("name, T, bits", [
    ("basal-2.0-4.5B", 300, 8), ("basal-2.0-4.5B", 1500, 8), ("basal-2.0-4.5B", 1500, 4), ("basal-2.0-mini", 1500, 8),
    ("basal-omni-1.0-mini", 300, 8), ("basal-omni-1.0-mini", 1500, 8), ("basal-omni-1.0-mini", 1500, 4),
    ("basal-omni-1.0-max", 1300, 8), ("basal-omni-pico", 1500, 8)])
def test_the_stacked_entry_is_each_layers_own_quantisation_for_every_served_layer_shape(name, T, bits):
    """For the layer shapes of the 4.5B, Gemma 4 (mini, max: two head sizes, sliding layers shorter than a long state)
    and Qwen3.5: the codes, steps and lows the cache stores for a stack of layers are exactly every layer's own
    kv_quant, one after another; every layer comes back bit-identical to its own round trip; usable() reads each
    layer's own shape; and the entry takes the bytes of the per-layer entry from the budget. 16 MB stacks: the 4.5B's
    60 layers of a 1.5k-token state go in 3 per tensor, not 1."""
    import basal.engine as E
    torch.manual_seed(0)
    shapes = _layers(name, T)
    kv = [((torch.randn(s) * (1 + i % 5)).bfloat16(), (torch.randn(s) + i).bfloat16()) for i, s in enumerate(shapes)]
    c = StateCache(1 << 34, bits=bits)
    c.put(b"k", kv)
    n, stacks = c.entries[b"k"][0]
    assert n == len(kv) and sorted(i for idx, _, _ in stacks for i in idx) == list(range(len(kv)))
    for idx, kq, vq in stacks:
        for stacked, keys in ((kq, True), (vq, False)):
            own = [kv_quant(kv[i][0 if keys else 1], bits, keys) for i in idx]
            for part in range(3):  # codes, steps, lows: the layers' own, concatenated
                assert torch.equal(stacked[part].flatten(), torch.cat([o[part].flatten() for o in own]))
    for (k, v), (gk, gv) in zip(kv, c.get(b"k")):
        assert torch.equal(gk, kv_dequant(kv_quant(k, bits, True), bits, True))
        assert torch.equal(gv, kv_dequant(kv_quant(v, bits, False), bits, False))
    assert [(tuple(k.shape), tuple(v.shape)) for k, v in c._shapes(c.entries[b"k"][0])] == [(s, s) for s in shapes]
    own_size = sum(t.numel() * t.element_size() for k, v in kv for e in (kv_quant(k, bits, True), kv_quant(v, bits, False))
                   for t in e[:3])
    assert c.entries[b"k"][1] == own_size
    per = {s: 2 * s[0] * s[1] * s[2] for s in shapes}  # bf16 bytes of one layer's keys
    assert len(stacks) == sum(-(-shapes.count(s) // max(1, E.KV_STACK_BYTES // b)) for s, b in per.items())
