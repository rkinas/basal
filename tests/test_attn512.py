"""--global-attn triton (basal/attn512.py): the causal kernel for 512-wide heads in the global layers of basal-omni's plain
rows. CPU: its reference maths, where the graph engine routes it, and what it refuses (the kernel itself runs on a
Hopper GPU only)."""
import pytest
import torch
import torch.nn.functional as F

from basal import attn512
from basal.gemma4_graph import Gemma4Graph

from test_gemma4_graph import S_, G_, tiny


@pytest.mark.parametrize("hq, hkv", [(4, 2), (4, 1), (2, 2)])
def test_reference_is_causal_attention_with_grouped_heads(hq, hkv):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, hq, 9, 16), torch.randn(2, hkv, 9, 16), torch.randn(2, hkv, 9, 16)
    want = F.scaled_dot_product_attention(q, k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1),
                                          is_causal=True, scale=0.5)
    assert torch.allclose(attn512.reference(q, k, v, 0.5), want, atol=1e-6)
    assert torch.allclose(attn512.attention(q, k, k, 1.0), attn512.reference(q, k, k, 1.0))  # keys = values


def test_global_layers_of_plain_rows_take_the_kernel():
    """With --global-attn triton the global layers' masks of plain rows are ("t512",) and the forward is unchanged (on
    CPU the op is its reference); the sliding layers, pass 2 and the readout rows keep their kernels."""
    m = tiny("31b")
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    g = Gemma4Graph(m)
    ref = g.forward(ids)
    g.kernels = {S_: "math", G_: "flex"}
    g.global_attn = "triton"
    pos = torch.arange(13)[None].expand(2, 13)
    m_all, _ = g._masks(g.kernels, 13, pos, None, None, None, {})
    assert m_all[G_] == ("t512",) and isinstance(m_all[S_], torch.Tensor)
    assert torch.allclose(g._run(g.kernels, ids, pos), ref, atol=1e-10)


def test_off_by_default():
    """--global-attn flex is the default: the global layers of plain rows keep FlexAttention's block mask."""
    g = Gemma4Graph(tiny("31b"))
    assert g.global_attn == "flex"
    g.kernels = {S_: "math", G_: "flex"}
    m_all, _ = g._masks(g.kernels, 13, torch.arange(13)[None].expand(2, 13), None, None, None, {})
    assert not isinstance(m_all[G_], tuple)


def test_refused_where_it_cannot_run(monkeypatch):
    m = tiny("e4b")
    monkeypatch.setattr(attn512, "available", lambda: "needs a CUDA GPU")  # the reason of a CPU, on any machine
    with pytest.raises(ValueError, match="--global-attn triton: needs a CUDA GPU"):
        Gemma4Graph(m, kernels="flex", global_attn="triton")
    monkeypatch.setattr(attn512, "available", lambda: None)
    with pytest.raises(ValueError, match="heads are \\[16\\] wide, not 512"):  # the tiny model's global heads
        Gemma4Graph(m, kernels="flex", global_attn="triton")
    with pytest.raises(ValueError, match="unknown --global-attn 'cudnn'"):
        Gemma4Graph(m, global_attn="cudnn")
    assert Gemma4Graph(m).global_attn == "flex"


def test_available_is_hopper_only(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "triton", sys.modules.get("triton") or types.ModuleType("triton"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for cap, ok in (((9, 0), True), ((12, 0), False), ((10, 0), False), ((8, 0), False)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, cap=cap: cap)
        assert (attn512.available() is None) == ok


def test_flag(monkeypatch):
    import sys

    import basal.server as server_module
    assert server_module.parser().parse_args([]).global_attn == "flex"
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "eager", "--global-attn", "triton"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must refuse before loading a model"))
    with pytest.raises(SystemExit, match="--global-attn needs the CUDA graph engine"):
        server_module.main()
