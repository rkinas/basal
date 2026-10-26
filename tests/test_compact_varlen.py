"""Compact rows through torch's varlen attention (--batches compact off Hopper): every packed prompt attends causally
to itself only, as a separate SDPA forward of that prompt; with a template prefix, its own tokens attend bottom-right
causally to the prefix and to themselves. The kernel tests need CUDA sm80+; the refusal test runs anywhere."""
import json

import pytest
import torch

from basal.engine import GraphBackend, _varlen_attention, torch_varlen_available

why = torch_varlen_available(torch.float16)
cuda = pytest.mark.skipif(why is not None, reason=f"torch varlen attention: {why}")

H, HKV, D = 8, 2, 128  # basal-2.0: GQA with 4 query heads per key / value head, head size 128
TORCH = "torch varlen"  # GraphBackend.compact_kernel for torch's varlen_attn


def packed(lens, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    q, k, v = (torch.randn(1, h, T, D, device="cuda", dtype=torch.float16, generator=g) for h in (H, HKV, HKV))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    return q, k, v, cu


def sdpa(q, k, v, mask=None):
    """[1, heads, n, d] -> [n, heads, d]; causal (top-left, n == keys) without a mask."""
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None,
                                                         enable_gqa=True)
    return o[0].transpose(0, 1)


@cuda
def test_compact_rows_equal_separate_prompts():
    lens = [5, 17, 64, 1, 33, 130]
    q, k, v, cu = packed(lens)
    cu = torch.cat([cu, cu[-1:].repeat(2)])  # empty prompts at the end, as in the engine's fixed prompt count
    got = _varlen_attention(q, k, v, cu, D ** -0.5, impl=TORCH)[0]
    assert got.shape == (sum(lens), H, D)
    a = 0
    for n in lens:
        ref = sdpa(q[:, :, a:a + n], k[:, :, a:a + n], v[:, :, a:a + n])
        assert (got[a:a + n] - ref).abs().max().item() < 2e-3, n
        a += n


@cuda
def test_template_prefix_bottom_right_causal():
    """Each prompt's keys are its template prefix (computed once, shared) followed by its own tokens; its own tokens
    see the whole prefix and the earlier own tokens."""
    lens, pre = [9, 40, 3], [6, 6, 11]  # two prompts on a 6-token prefix, one on an 11-token one
    q, k, v, cu = packed(lens, seed=1)
    g = torch.Generator(device="cuda").manual_seed(2)
    pk, pv = (torch.randn(17, HKV, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(2))
    starts = [0, 0, 6]  # where each prompt's prefix starts in the shared prefix buffer (17 = 6 + 11 tokens)
    gather, cu_k, a, nk = [], [0], 0, 0
    for n, p, s in zip(lens, pre, starts):
        gather += list(range(s, s + p)) + [17 + a + j for j in range(n)]
        a += n
        nk += p + n
        cu_k.append(nk)
    cu_k = torch.tensor(cu_k, device="cuda", dtype=torch.int32)
    gather = torch.tensor(gather, device="cuda")
    got = _varlen_attention(q, k, v, (cu, cu_k, gather, [(pk, pv)]), D ** -0.5, impl=TORCH)[0]
    a = 0
    for n, p, s in zip(lens, pre, starts):
        kk = torch.cat([pk[s:s + p], k[0, :, a:a + n].transpose(0, 1)]).transpose(0, 1)[None]
        vv = torch.cat([pv[s:s + p], v[0, :, a:a + n].transpose(0, 1)]).transpose(0, 1)[None]
        mask = torch.ones(n, p + n, dtype=torch.bool, device="cuda").tril(p)  # own token j sees keys 0 .. p + j
        ref = sdpa(q[:, :, a:a + n], kk, vv, mask)
        assert (got[a:a + n] - ref).abs().max().item() < 2e-3, (n, p)
        a += n


@cuda
def test_torch_varlen_matches_fa3_where_both_run():
    pytest.importorskip("flash_attn_interface")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashAttention 3 runs on Hopper only")
    lens = [7, 64, 129]
    q, k, v, cu = packed(lens, seed=3)
    fa3 = _varlen_attention(q, k, v, cu, D ** -0.5, impl="fa3")
    tv = _varlen_attention(q, k, v, cu, D ** -0.5, impl=TORCH)
    assert (fa3 - tv).abs().max().item() < 2e-3


def test_unsupported_compact_refused_before_loading(tmp_path, monkeypatch):
    """--batches compact where no compact kernel runs (here: fp32 activations, or no CUDA) names the reason before the
    tokenizer or the weights are loaded."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda *_: pytest.fail("loaded before refusing"))
    monkeypatch.setattr("basal.engine.set_attention", lambda *_, **__: "fa2")
    with pytest.raises(ValueError, match="--batches compact needs .* torch's varlen attention"):
        GraphBackend(str(tmp_path), dtype="float32", batches="compact")


def test_prompts_past_the_largest_compact_graph_run_plain():
    """With little free memory _fit_grid cuts COMPACT_LENS to the token budget, below PLAIN_LENS (basal-omni ultra under
    a 32 GB ballast: compact 1,536, plain 1,664). A single prompt between the two runs in a plain graph; it was packed
    into a compact row and failed ("too long: 1584 > 1536", HTTP 500)."""
    be = object.__new__(GraphBackend)
    be.compact, be.prefix = True, False
    be.PLAIN_LENS, be.COMPACT_LENS = [512, 1024, 1664], [256, 512, 1024, 1536]
    units = [(([1] * n, None, None, None, [0]),) for n in (300, 1584, 700, 2000, 1500)]
    out = be._forwards(units)
    assert ([1], "plain") in out and ([3], "plain") in out
    assert sorted(k for idx, kind in out if kind == "compact" for k in idx) == [0, 2, 4]
    assert all(sum(len(units[k][0][0]) for k in idx) <= 1536 for idx, kind in out if kind == "compact")
