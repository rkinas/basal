"""One block-quantisation core (basal/blockquant.py): --mode fp8 --fp8-scaling block and --mode int8 are two formats of
it (one Triton GEMM, one layer class, one launch rule), the graph engines merge their projections with one helper, and
NVFP4's rounding is derived from one e2m1 table that its activation kernel follows too. No GPU needed."""
import inspect
import re

import pytest
import torch

from basal import blockquant, fp8block, gdn_graph, gemma4_graph, int8block, lean, nvfp4


@pytest.mark.parametrize("mod, cls, dtype, qmax", [(fp8block, fp8block.BlockFP8Linear, torch.float8_e4m3fn, 448.0),
                                                    (int8block, int8block.Int8BlockLinear, torch.int8, 127.0)])
def test_fp8_and_int8_are_formats_of_one_core(mod, cls, dtype, qmax):
    f = mod.FORMAT
    assert isinstance(f, blockquant.BlockFormat) and (f.dtype, f.qmax, f.rounded) == (dtype, qmax, dtype == torch.int8)
    assert mod.quantize.__self__ is f and mod.act_quant.__self__ is f and mod.gemm.__self__ is f
    assert issubclass(cls, blockquant.BlockLinear) and cls.FORMAT is f
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 384, bias=True)
    m = cls.of(lin)
    assert m.weight.dtype == dtype and m.weight_scale.shape == (3, 2)
    part = m.part(128, 256)
    assert type(part) is cls and torch.equal(part.weight.float(), m.weight[128:].float())
    x = torch.randn(5, 256)
    p = lean._params(m)  # a compiled layer runs the same parameters through the same GEMM
    assert len(p) == 3 and torch.equal(lean.linear(x, p), m(x))
    assert (m(x) - lin(x)).norm() / lin(x).norm() < 0.05


def test_one_gemm_kernel_for_both_formats():
    """The int8 kernel was a copy of the FP8 one but for the product of a slice: one kernel source now, the product
    chosen by a compile-time flag; both ops launch with the same tiles through one grid rule."""
    src = inspect.getsource(blockquant._kernel)
    assert src.count("tl.dot(") == 2 and "if INT:" in src
    assert "def kernel" not in inspect.getsource(fp8block) and "def kernel" not in inspect.getsource(int8block)
    assert fp8block.CONFIG == dict(blockquant.TILES, num_stages=4) and int8block.CONFIG == blockquant.TILES
    assert fp8block.grid is int8block.grid is blockquant.grid


def test_fuse_makes_the_layers_views_of_one_weight():
    torch.manual_seed(0)
    mods = [torch.nn.Linear(16, n, bias=False) for n in (8, 4, 4)]
    before = [m.weight.detach().clone() for m in mods]
    w, sizes = blockquant.fuse(mods)
    assert sizes == [8, 4, 4] and torch.equal(w, torch.cat(before))
    for m, b in zip(mods, before):
        assert torch.equal(m.weight, b) and m.weight.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    lin, sizes = gemma4_graph.Gemma4Graph._fuse([torch.nn.Linear(16, 8, bias=False), torch.nn.Linear(16, 8, bias=False)])
    assert sizes == [8, 8] and lin.weight.shape == (16, 16)
    assert not hasattr(gdn_graph, "_merged")  # (the Qwen3.5 engine's copy of it)


def test_e2m1_rounding_comes_from_one_table():
    """The cuts are the midpoints of the e2m1 table with ties to the even code; _codes rounds every value (exact
    midpoints included) to the nearest table value, ties to even, as an argmin over the table does."""
    assert nvfp4.FP4_MAX == nvfp4.E2M1[-1] == 6.0
    assert [t for t, _ in nvfp4.CUTS] == [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    mids = torch.tensor([t for t, _ in nvfp4.CUTS])
    u = torch.cat([torch.linspace(-7, 7, 20001), mids, -mids, torch.tensor([0.0, -0.0, 6.0, 100.0])])
    table = torch.tensor(nvfp4.E2M1)
    d = (u.abs()[:, None] - table[None]).abs()
    best = d.min(-1, keepdim=True).values
    near = (d == best) & ((torch.arange(8)[None] % 2 == 0) | ((d == best).sum(-1, keepdim=True) == 1))
    want = torch.where(u < 0, -table[near.int().argmax(-1)], table[near.int().argmax(-1)])
    got = nvfp4._values(nvfp4._codes(u))
    assert torch.equal(got.abs(), want.abs()) and torch.equal(got[got != 0] < 0, u[got != 0] < 0)


def test_the_activation_kernel_rounds_with_the_same_cuts():
    """The Triton activation quantiser writes the cuts out (a kernel cannot loop over a Python table): they must be the
    table's, comparison by comparison, and its block scale divides by the table's largest value."""
    src = inspect.getsource(nvfp4._build)
    got = re.findall(r"\(a (>=?) ([0-9.]+)\)\.to\(tl\.int32\)", src)
    assert [(op, float(t)) for op, t in got] == [(">=" if odd else ">", t) for t, odd in nvfp4.CUTS]
    assert f"div_rn(amax, {nvfp4.FP4_MAX})" in src
