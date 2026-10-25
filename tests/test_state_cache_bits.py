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
    with pytest.raises(SystemExit, match="--state-cache-bits 8 / 4 needs"):
        Server(parser().parse_args(["--mode", "fast", "--state-cache-bits", "8"]))
