"""Block-scaled FP8 (basal/fp8block.py, --mode fp8 on basal-omni): 128 x 128 weight blocks and 1 x 128
activation tiles with their own scales; on CPU the GEMM's maths dequantised, on CUDA the Triton kernel (equal to it,
and the same output for a row whatever the rows around it)."""
import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from basal import fp8block
from basal.engine import FP8_BF16_LAYERS, fp8_scaling_of
from basal.gemma4_graph import Gemma4Graph
from basal.server import Server, parser

S_, G_ = "sliding_attention", "full_attention"


def test_quantize_keeps_each_block_within_e4m3_rounding():
    torch.manual_seed(0)
    w = torch.randn(256, 384) * torch.tensor([1e-3, 1.0, 50.0]).repeat_interleave(128)[None]  # blocks of other sizes
    q, s = fp8block.quantize(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (2, 3)
    blocks = q.float().view(2, 128, 3, 128).abs().amax((1, 3))
    assert torch.all(blocks == 448.0)  # every block uses the whole e4m3 range
    deq = (q.float().view(2, 128, 3, 128) * s[:, None, :, None]).view(256, 384)
    err = (deq - w).abs()
    bound = w.abs() * 2 ** -4 + s.repeat_interleave(128, 0).repeat_interleave(128, 1) * 2 ** -9
    assert torch.all(err <= bound * 1.0001)


def test_block_gemm_grid_needs_whole_output_tiles():
    """The block GEMM (FP8 and int8) stores whole 128-wide output tiles: one program per 128 x 128 tile, and N not a
    multiple of 128 is refused (its last outputs would stay unwritten; fits() keeps every layer to multiples of 128).
    There is one rule for both kernels, not a tile-width override that could never apply."""
    from basal import int8block
    assert fp8block.grid(1, 384) == (3,) and fp8block.grid(129, 384) == (6,) and fp8block.grid(0, 128) == (0,)
    assert int8block.grid is fp8block.grid and int8block.CONFIG["BN"] == fp8block.CONFIG["BN"] == fp8block.BLOCK
    for n in (200, 64):
        with pytest.raises(ValueError, match="not a multiple of 128"):
            fp8block.grid(4, n)


def test_linear_is_close_to_the_bf16_product():
    torch.manual_seed(0)
    x, w = torch.randn(5, 3, 256), torch.randn(384, 256) / 16
    q, s = fp8block.quantize(w)
    y = fp8block.linear(x, q, s)
    ref = x @ w.t()
    assert y.shape == (5, 3, 384)
    assert (y - ref).norm() / ref.norm() < 0.05
    qa, sa = fp8block.act_quant(x.reshape(15, 256))
    assert sa.shape == (15, 2) and torch.all(qa.float().view(15, 2, 128).abs().amax(-1) == 448.0)


def tiny(kind):
    """A tiny Gemma 4 whose projections are multiples of 128 (head sizes 64 / 128): "e4b" with per-layer inputs and
    KV-shared layers, "31b" with keys = values in the global layers."""
    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, head_dim=64, global_head_dim=128, sliding_window=5,
              layer_types=[S_, S_, G_, S_, S_, G_], final_logit_softcapping=3.0, max_position_embeddings=256)
    if kind == "e4b":
        kw.update(hidden_size_per_layer_input=8, vocab_size_per_layer_input=64, num_kv_shared_layers=2)
    else:
        kw.update(hidden_size_per_layer_input=0, attention_k_eq_v=True, num_global_key_value_heads=1)
    m = Gemma4ForCausalLM(Gemma4TextConfig(**kw)).eval()
    m.config._attn_implementation = "sdpa"
    return m.double()  # (float32: rounding differences of 1e-7 between the two forwards flip FP8 roundings)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("merge", [False, True])
def test_gemma4_block_fp8_runs_close_to_bf16(kind, merge):
    """Every projection of the layers becomes block FP8 (merged: one per merged weight, its parts views of it), the
    per-layer input projections stay; the graph forward equals transformers' forward of the quantised model and stays
    close to the unquantised one."""
    m = tiny(kind)
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m.model(input_ids=ids).last_hidden_state
    g = Gemma4Graph(m, merge=merge, fp8="block")
    own = g.n_kv
    if merge:  # q / k / v (q only in KV-shared layers: not merged) and gate / up per layer, o and down
        assert g.fp8 == own + (6 - own) + 6 + 12
        a = m.model.layers[0].self_attn
        assert a.q_proj.weight.data_ptr() == g.qkv[0][0].weight.data_ptr()
    else:
        assert g.fp8 == 6 * 5 + 2 * own - (2 if kind == "31b" else 0)  # 31b: no v in the 2 global layers
    if kind == "e4b":
        assert type(m.model.layers[0].per_layer_projection) is torch.nn.Linear
    with torch.no_grad():
        hf = m.model(input_ids=ids).last_hidden_state
    out = g.forward(ids)
    assert torch.allclose(out, hf, atol=1e-4, rtol=1e-4)
    # each FP8 product is off by 3-4% (e4m3 rounding of random weights); a random tiny model amplifies it layer after
    # layer (cosine with bf16 0.99 after the first layer, 0.88 after the sixth); a wrong scale would leave ~0
    assert torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), 0) > 0.85


@pytest.mark.parametrize("keep", [1, 2])
def test_gemma4_block_fp8_keeps_the_edge_layers_in_bf16(keep):
    """--fp8-bf16-layers: the first and the last `keep` layers keep their bf16 linears, the others are block FP8."""
    m = tiny("31b")
    g = Gemma4Graph(m, fp8="block", fp8_bf16_layers=keep)
    fp8 = [isinstance(layer.mlp.down_proj, fp8block.BlockFP8Linear) for layer in m.model.layers]
    assert fp8 == [keep <= i < 6 - keep for i in range(6)]
    assert all(type(m.model.layers[i].self_attn.q_proj) is torch.nn.Linear for i in (0, 5))
    every = Gemma4Graph(tiny("31b"), fp8="block").fp8
    assert 0 < g.fp8 < every
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        hf = m.model(input_ids=ids).last_hidden_state
    assert torch.allclose(g.forward(ids), hf, atol=1e-4, rtol=1e-4)


def test_fp8_scaling_auto_is_block_for_basal_omni_only():
    assert fp8_scaling_of("auto", "fp8", True, False) == "block"
    assert fp8_scaling_of("auto", "fp8", False, False) == "row"  # Llama: torchao's per-row FP8
    assert fp8_scaling_of("auto", "fp8", True, True) == "row"  # an -FP8 checkpoint keeps its stored weights
    assert fp8_scaling_of("auto", None, True, False) == "row"
    assert fp8_scaling_of("row", "fp8", True, False) == "row" and fp8_scaling_of("block", "fp8", True, False) == "block"
    a = parser().parse_args(["--mode", "fp8"])
    assert a.fp8_scaling == "auto" and a.fp8_bf16_layers == FP8_BF16_LAYERS


def test_negative_bf16_layers_refused():
    with pytest.raises(SystemExit, match="--fp8-bf16-layers"):
        Server(parser().parse_args(["--mode", "fp8", "--fp8-bf16-layers", "-1"]))


def test_block_scaling_only_with_mode_fp8():
    with pytest.raises(SystemExit, match="--fp8-scaling block needs --mode fp8"):
        Server(parser().parse_args(["--mode", "fast", "--fp8-scaling", "block"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("M", [1, 37, 128, 1000])
def test_kernel_equals_the_maths_and_ignores_the_other_rows(M):
    """The Triton GEMM: the reference maths within fp32 rounding, and a row's output bit-identical alone and among
    other rows (fixed tiles, no split reduction)."""
    torch.manual_seed(0)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    q, s = fp8block.quantize(torch.randn(768, 1024, device="cuda") / 32)
    qa, sa = fp8block.act_quant(x)
    y = fp8block.gemm(qa, sa, q, s, torch.float32)
    ref = fp8block._reference(qa, sa, q, s, torch.float32)
    assert torch.allclose(y, ref, rtol=2e-3, atol=2e-3)  # Hopper's FP8 MMA accumulates in reduced precision
    alone = fp8block.gemm(qa[-1:], sa[-1:], q, s, torch.bfloat16)
    many = fp8block.gemm(torch.cat([qa, qa]), torch.cat([sa, sa]), q, s, torch.bfloat16)
    assert torch.equal(alone[0], many[M - 1]) and torch.equal(many[M - 1], many[2 * M - 1])


@pytest.mark.skipif(not (torch.cuda.is_available() and fp8block.sm90()), reason="Hopper with flashinfer")
def test_sm90_kernel_equals_the_triton_kernel():
    """flashinfer's SM90 kernel (DeepGEMM's) and the Triton kernel compute the same block-scaled product."""
    torch.manual_seed(0)
    x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
    q, s = fp8block.quantize(torch.randn(768, 1024, device="cuda") / 32)
    y = fp8block.linear(x, q, s).float()
    qa, sa = fp8block.act_quant(x)
    ref = fp8block._reference(qa, sa, q, s, torch.float32)
    assert (y - ref).abs().max() <= 2e-2 * ref.abs().max()
