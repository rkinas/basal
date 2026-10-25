"""--quant-load cpu: the bf16 weights of basal-omni stay in host memory and each decoder layer goes to the GPU, is
merged and quantised there before the next (Gemma4Graph(stream=...)); the result is the same weights and the same
forward as quantising after loading everything onto the GPU."""
import pytest
import torch

from basal import fp8block
from basal.engine import quant_load_refused
from basal.gemma4_graph import Gemma4Graph, move_except
from basal.server import Server, parser

from test_fp8block import tiny


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("keep", [0, 2])
def test_layer_by_layer_equals_all_at_once(kind, merge, keep):
    ref, m = tiny(kind), tiny(kind)
    g0 = Gemma4Graph(ref, merge=merge, fp8="block", fp8_bf16_layers=keep)
    g1 = Gemma4Graph(m, merge=merge, fp8="block", fp8_bf16_layers=keep, stream=torch.device("cpu"))
    assert (g1.fp8, g1.merged) == (g0.fp8, g0.merged) and g1.fp8 > 0
    assert same_weights(ref, m)
    assert [type(x.mlp.down_proj) for x in m.model.layers] == [type(x.mlp.down_proj) for x in ref.model.layers]
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    assert torch.equal(g1.forward(ids), g0.forward(ids))


def test_move_except_keeps_tied_weights_and_skips_the_layers():
    m = tiny("31b").float()
    m.lm_head.weight = m.model.embed_tokens.weight  # tied, as the released models
    w = m.model.embed_tokens.weight
    move_except(m, torch.device("cpu"), m.model.layers)
    assert m.lm_head.weight is w and m.model.embed_tokens.weight is w


def test_refused_without_a_load_time_quantisation():
    assert quant_load_refused("gpu", None, "row", None, False, False) is None
    assert quant_load_refused("cpu", "fp8", "block", None, True, False) is None
    assert quant_load_refused("cpu", "nvfp4", "row", {"0.q_proj": 1.0}, True, False) is None
    for args in [("cpu", None, "row", None, True, False), ("cpu", "fp8", "row", None, False, False),
                 ("cpu", "fp8", "block", None, True, True), ("cpu", "nvfp4", "row", None, True, False)]:
        assert "--quant-load cpu" in quant_load_refused(*args)
    assert "gpu or cpu" in quant_load_refused("disk", "fp8", "block", None, True, False)
    assert parser().parse_args([]).quant_load == "gpu"
    with pytest.raises(SystemExit, match="--quant-load cpu needs"):
        Server(parser().parse_args(["--mode", "fast", "--quant-load", "cpu"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("kind", ["e4b", "31b"])
def test_gpu_layer_by_layer_equals_all_at_once(kind):
    """On the GPU: a model left on the CPU, its other parts moved and its layers streamed, gives the same FP8 weights
    and bit-identical forwards as the model moved whole and quantised there; layers outside the edges are FP8."""
    ref, m = tiny(kind).bfloat16().cuda(), tiny(kind).bfloat16()
    dev = torch.device("cuda")
    g0 = Gemma4Graph(ref, merge=True, fp8="block", fp8_bf16_layers=1)
    move_except(m, dev, m.model.layers)
    g1 = Gemma4Graph(m, merge=True, fp8="block", fp8_bf16_layers=1, stream=dev)
    assert all(p.is_cuda for p in m.parameters())
    assert same_weights(ref, m)
    assert isinstance(m.model.layers[2].mlp.down_proj, fp8block.BlockFP8Linear)
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(g1.forward(ids), g0.forward(ids))
