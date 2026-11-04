"""--attn: the kernel of causal forwards without a mask. Off CUDA (Mac, CPU) auto is FlashAttention 2 through torch
SDPA and the others are refused clearly; a direct kernel serves only causal, unmasked, half-precision forwards, while
the branch pass (a prefix and a mask) and fp32 keep SDPA."""
import pytest
import torch

import basal.engine as E
from basal.engine import attention_available, basal_attention, set_attention
from basal.server import Server, parser


def test_choices_off_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert set_attention("auto") == "fa2" and set_attention("fa2") == "fa2"
    for impl in ("fa3", "fa4", "cudnn"):
        assert attention_available(impl) == "needs CUDA"
        with pytest.raises(ValueError, match=f"--attn {impl}: needs CUDA"):
            set_attention(impl)


def test_fa3_needs_hopper_and_the_package(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    assert "Hopper (sm_90) only" in attention_available("fa3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    assert "sm_90+" in attention_available("fa4")
    assert E.auto_attention() == "fa2"                      # not Hopper: the built-in kernel


def test_auto_keeps_fa2_for_fp8_and_fa3_stays_selectable(monkeypatch):
    """FP8 GEMMs (an FP8 checkpoint or --mode fp8): auto stays on FlashAttention 2 where FA3 runs, the path the FP8
    ports were gated on; --attn fa3 still selects FA3."""
    monkeypatch.setattr(E, "attention_available", lambda impl: None)   # Hopper with the flash_attn_3 wheel
    assert E.auto_attention() == "fa3" and E.auto_attention(fp8=True) == "fa2"
    calls = []
    monkeypatch.setattr(torch.nn.attention, "activate_flash_attention_impl", lambda name: calls.append(name))
    monkeypatch.setattr(torch.nn.attention, "restore_flash_attention_impl", lambda: calls.append("restore"))
    assert set_attention("auto", fp8=True) == "fa2" and calls == []
    assert set_attention("fa3", fp8=True) == "fa3" and calls == ["FA3"]
    assert set_attention("auto", fp8=True) == "fa2" and calls == ["FA3", "restore"]


def test_graph_backend_tells_the_attention_choice_about_fp8(monkeypatch):
    seen = []

    def stop(choice, fp8=False):
        seen.append((choice, fp8))
        raise RuntimeError("stop")   # before any model is loaded
    monkeypatch.setattr(E, "resolve_attention", stop)  # (graph_options resolves --attn before anything is loaded)
    monkeypatch.setattr(E, "fp8_refused", lambda: None)  # a GPU that runs torchao FP8 (sm 8.9+)
    for checkpoint, quant in ((True, None), (False, "fp8"), (False, None), (False, "nvfp4")):
        monkeypatch.setattr(E, "modelopt_fp8", lambda d, c=checkpoint: c)
        with pytest.raises(RuntimeError, match="stop"):
            E.GraphBackend("model", quant=quant)
    assert seen == [("auto", True), ("auto", True), ("auto", False), ("auto", False)]


class Module:
    layer_idx, is_causal, num_key_value_groups = 0, True, 1
    training = False

    def __init__(self):
        self.config = type("C", (), {"_attn_implementation": "basal_sdpa"})()


def qkv(L=8, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return [torch.randn(1, 2, L, 16, dtype=dtype) for _ in range(3)]


def test_direct_kernel_only_for_causal_unmasked_half_precision(monkeypatch):
    calls = []

    def direct(q, k, v, scale):
        calls.append(q.shape)
        return q.transpose(1, 2).contiguous()
    monkeypatch.setitem(E._ATTN, "direct", direct)
    q, k, v = qkv()
    out, _ = basal_attention(Module(), q, k, v, None, scaling=0.25)
    assert len(calls) == 1 and out.shape == (1, 8, 2, 16)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16)
    basal_attention(Module(), q, k, v, mask, scaling=0.25)                        # a mask: SDPA
    prefix = [(torch.zeros(1, 2, 4, 16, dtype=torch.bfloat16),) * 2]
    basal_attention(Module(), q, k, v, torch.zeros(1, 1, 8, 12, dtype=torch.bfloat16), scaling=0.25,
                    basal_prefix=prefix)                                          # the branch pass: SDPA
    basal_attention(Module(), *qkv(dtype=torch.float32), None, scaling=0.25)      # fp32: SDPA
    basal_attention(Module(), *qkv(L=1), None, scaling=0.25)                      # one query token: SDPA
    assert len(calls) == 1


def test_server_refuses_a_kernel_outside_the_graph_engine(monkeypatch):
    with pytest.raises(SystemExit, match="--attn fa3 needs the CUDA graph engine"):
        Server(parser().parse_args(["--mode", "eager", "--attn", "fa3"]))
