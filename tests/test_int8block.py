"""Block-scaled int8 (basal/int8block.py, --mode int8 on basal-omni): fp8block's 128 x 128 weight blocks and 1 x 128
activation tiles with int8 values; on CPU the GEMM's maths dequantised, on CUDA the Triton kernel (equal to it, and the
same output for a row whatever the rows around it)."""
import pytest
import torch

from basal import int8block, lean
from basal.gemma4_graph import Gemma4Graph
from basal.server import MODES, calibration_name, parser

from test_fp8block import tiny


def test_quantize_keeps_each_block_within_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(256, 384) * torch.tensor([1e-3, 1.0, 50.0]).repeat_interleave(128)[None]  # blocks of other sizes
    q, s = int8block.quantize(w)
    assert q.dtype == torch.int8 and s.shape == (2, 3)
    assert torch.all(q.float().view(2, 128, 3, 128).abs().amax((1, 3)) == 127)  # every block uses the whole range
    deq = (q.float().view(2, 128, 3, 128) * s[:, None, :, None]).view(256, 384)
    step = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.all((deq - w).abs() <= step * 0.5001)


def test_linear_is_closer_to_the_product_than_fp8():
    from basal import fp8block
    torch.manual_seed(0)
    x, w = torch.randn(5, 3, 256), torch.randn(384, 256) / 16
    ref = x @ w.t()
    y = int8block.linear(x, *int8block.quantize(w))
    assert y.shape == (5, 3, 384)
    e8 = (y - ref).norm() / ref.norm()
    efp8 = (fp8block.linear(x, *fp8block.quantize(w)) - ref).norm() / ref.norm()
    assert e8 < 0.02 and e8 < efp8  # Gaussian values: 7 bits of uniform steps beat e4m3's 3 mantissa bits
    qa, sa = int8block.act_quant(x.reshape(15, 256))
    assert qa.dtype == torch.int8 and sa.shape == (15, 2)
    assert lean.linear(x, lean._params(int8block.Int8BlockLinear.of(torch.nn.Linear(256, 384, bias=False)))).shape \
        == (5, 3, 384)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("merge", [False, True])
def test_gemma4_block_int8_runs_close_to_bf16(kind, merge):
    """Every projection of the layers outside the edges becomes block int8 (merged: one per merged weight, its parts
    views of it); the graph forward equals transformers' forward of the quantised model and stays close to the
    unquantised one (closer than block FP8)."""
    m, m8 = tiny(kind), tiny(kind)
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m.model(input_ids=ids).last_hidden_state
    g = Gemma4Graph(m, merge=merge, fp8="int8", fp8_bf16_layers=1)
    assert g.fp8 > 0
    kinds = [type(layer.mlp.down_proj) for layer in m.model.layers]
    assert kinds == [torch.nn.Linear] + [int8block.Int8BlockLinear] * 4 + [torch.nn.Linear]
    assert m.model.layers[2].mlp.down_proj.weight.dtype == torch.int8
    with torch.no_grad():
        hf = m.model(input_ids=ids).last_hidden_state
    out = g.forward(ids)
    assert torch.allclose(out, hf, atol=1e-4, rtol=1e-4)
    out8 = Gemma4Graph(m8, merge=merge, fp8="block", fp8_bf16_layers=1).forward(ids)
    cos = torch.nn.functional.cosine_similarity
    assert cos(out.flatten(), ref.flatten(), 0) > cos(out8.flatten(), ref.flatten(), 0)


def test_mode_int8_is_a_graph_mode_with_its_own_calibration_file():
    assert MODES["int8"] == ("graph", "int8", True)
    assert calibration_name("graph", "int8") == "int8"
    assert parser().parse_args(["--mode", "int8", "--quant-load", "cpu"]).quant_load == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("M", [1, 37, 128, 1000])
def test_kernel_equals_the_maths_and_ignores_the_other_rows(M):
    """The Triton GEMM: the reference maths within fp32 rounding, and a row's output bit-identical alone and among
    other rows (fixed tiles, no split reduction)."""
    torch.manual_seed(0)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    q, s = int8block.quantize(torch.randn(768, 1024, device="cuda") / 32)
    qa, sa = int8block.act_quant(x)
    y = int8block.gemm(qa, sa, q, s, torch.float32)
    ref = int8block._reference(qa.cpu(), sa.cpu(), q.cpu(), s.cpu(), torch.float32)
    assert torch.allclose(y.cpu(), ref, rtol=1e-5, atol=1e-5)
    one = int8block.gemm(qa[-1:], sa[-1:], q, s, torch.float32)
    assert torch.equal(one, y[-1:])


def test_int8_serves_basal_omni_ultra_only_with_its_own_calibration(tmp_path):
    """--mode int8 kept the paired accuracy check on ultra and failed it on max (12B), so other basal-omni shapes are
    refused; it needs CALIBRATION.int8.json (or --calibration), as NVFP4 needs its own file."""
    from types import SimpleNamespace

    from basal.engine import omni_quant_refused
    from basal.server import nvfp4_calibration_refused
    from test_nvfp4 import _omni
    ultra, mx = _omni(tmp_path, 5376, 60), _omni(tmp_path, 3840, 48)
    assert omni_quant_refused(ultra, "int8") is None
    assert "ultra (31B) only" in omni_quant_refused(mx, "int8")
    a = SimpleNamespace(calibration=None, no_calibration=False)
    assert "CALIBRATION.int8.json" in nvfp4_calibration_refused(ultra, "graph", "int8", a)
    (ultra / "CALIBRATION.int8.json").write_text("{}")
    assert nvfp4_calibration_refused(ultra, "graph", "int8", a) is None
    assert "CALIBRATION.nvfp4.json" in nvfp4_calibration_refused(ultra, "graph", "nvfp4", a)
