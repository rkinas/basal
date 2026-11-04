"""EagerBackend reads option codes in fp32 from the output head's input, with the model's final logit softcap (Gemma):
equal to the model's own fp32 logits, and in a bf16 model no longer rounded to bf16 logit steps. Llama, Gemma 4 and a
hybrid model (Qwen3.5: Gated DeltaNet and attention layers, left-padded batches and chunked prefill)."""
import pytest
import torch
from transformers import (Gemma4ForCausalLM, Gemma4TextConfig, LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM,
                          Qwen3_5TextConfig)

from basal.engine import EagerBackend


@pytest.fixture(autouse=True)
def _reference_gated_delta_rule(monkeypatch):
    """transformers takes Qwen3.5's gated delta rule and causal convolution from flash-linear-attention / causal-conv1d
    whenever they are installed, also for CPU tensors, where their Triton kernels fail ("cannot be accessed from
    Triton (cpu tensor?)" on a CUDA machine with fla). These CPU tests use the reference PyTorch functions."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        if hasattr(q, name):
            monkeypatch.setattr(q, name, inspect.unwrap(getattr(q, name)))


class Stub(EagerBackend):
    def __init__(self, model):  # the real __init__ minus loading
        self.model, self.dev, self.tok = model, torch.device("cpu"), type("Tok", (), {"pad_token_id": 0})()
        self.head = model.get_output_embeddings()
        cfg = model.config.get_text_config()
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self._head_in = None
        self.head.register_forward_hook(self._grab)

    def encode(self, prompts):
        return prompts


def models():
    torch.manual_seed(0)
    yield LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
    torch.manual_seed(0)
    yield Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                             final_logit_softcapping=3.0, max_position_embeddings=128)).eval()
    torch.manual_seed(0)
    yield Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})).eval()


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


def test_eager_codes_equal_full_logits_fp32():
    toks, lids = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]], [[11, 12, 13], [14, 15]]
    for m in models():
        if hasattr(m.model, "embed_tokens"):  # a larger hidden state, so the softcap is active
            m.model.embed_tokens.weight.data *= 8
        got = Stub(m).run(toks, lids)
        for g, r in zip(got, reference(m, toks, lids)):
            assert torch.allclose(torch.tensor(g), r, atol=1e-5), (type(m).__name__, g, r)


def test_eager_bf16_codes_are_not_clumped():
    m = next(models()).to(torch.bfloat16)
    m.lm_head.weight.data *= 40  # logits of magnitude ~16-32, where bf16 steps are 0.125
    toks = [[1, 2, 3, i] for i in range(4, 40)]
    got = Stub(m).run(toks, [[11, 12]] * len(toks))
    with torch.no_grad():
        bf = [torch.softmax(m(input_ids=torch.tensor([t])).logits[0, -1, [11, 12]].float(), -1)[0].item() for t in toks]
    assert len({round(g[0], 9) for g in got}) > len({round(x, 9) for x in bf})


def test_eager_long_prompts_chunked_and_batches_capped(monkeypatch):
    """Prompts longer than CHUNK are prefilled in chunks through the KV cache, shorter ones batched up to
    MAX_BATCH_TOKENS: the probabilities equal the one-batch full forward (fp32)."""
    for m in models():
        torch.manual_seed(1)
        toks = [list(torch.randint(1, 60, (n,)).tolist()) for n in (5, 9, 23, 31, 40)]
        lids = [[11, 12, 13]] * len(toks)
        ref = Stub(m).run(toks, lids)
        s = Stub(m)
        monkeypatch.setattr(s, "CHUNK", 8)  # the three longest prompts in 8-token chunks
        monkeypatch.setattr(s, "MAX_BATCH_TOKENS", 12)  # the two short ones in separate batches
        got = s.run(toks, lids)
        for g, r in zip(got, ref):
            assert torch.allclose(torch.tensor(g), torch.tensor(r), atol=1e-5), (type(m).__name__, g, r)


def test_eager_sdpa_kernels_without_memory_efficient_before_2_14_1(monkeypatch):
    """pytorch/pytorch#191937: before torch 2.14.1 the eager backend's transformers forward runs without SDPA's
    memory-efficient kernel on CUDA (wrong attention for expanded keys / values at query lengths = 1 mod 32 / 64)."""
    from torch.nn.attention import SDPBackend

    from basal import engine
    assert engine.eager_sdpa_kernels("cpu") is None
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    kernels = engine.eager_sdpa_kernels("cuda")
    assert SDPBackend.EFFICIENT_ATTENTION not in kernels and SDPBackend.MATH in kernels
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)  # Gemma 4 turns the others off
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)
    assert engine.eager_sdpa_kernels("cuda") == [SDPBackend.MATH]
    monkeypatch.setattr(torch, "__version__", "2.14.1")
    assert engine.eager_sdpa_kernels("cuda") is None
    s = Stub(next(models()))  # the forward runs under those kernels
    seen = []
    import torch.nn.attention as attention
    real = attention.sdpa_kernel
    monkeypatch.setattr(attention, "sdpa_kernel", lambda ks: seen.append(ks) or real([SDPBackend.MATH]))
    s.sdpa_kernels = [SDPBackend.MATH]
    s.run([[1, 2, 3, 4]], [[11, 12]])
    assert seen == [[SDPBackend.MATH]]


def test_eager_padded_batch_at_one_mod_64_equals_math_on_gpu():
    """On CUDA: a left-padded batch of a one-KV-head model (transformers repeats it as an expanded view under a mask)
    at padded lengths = 1 mod 64 gives the math kernel's readouts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    from torch.nn.attention import SDPBackend

    from basal import engine
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                                     max_position_embeddings=4096)).cuda().eval()
    s = Stub(m)
    s.dev = torch.device("cuda")
    s.sdpa_kernels = engine.eager_sdpa_kernels("cuda")
    for L in (65, 1857):
        toks = [torch.randint(1, 60, (n,)).tolist() for n in (L, L - 40)]
        got = s.run(toks, [[11, 12, 13]] * 2)
        s2 = Stub(m)
        s2.dev, s2.sdpa_kernels = torch.device("cuda"), [SDPBackend.MATH]
        ref = s2.run(toks, [[11, 12, 13]] * 2)
        for g, r in zip(got, ref):
            assert max(abs(a - b) for a, b in zip(g, r)) < 1e-4
