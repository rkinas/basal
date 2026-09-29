"""Shared-prefix packing is exactly equivalent to separate forward passes (tiny random model, CPU)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import GraphBackend, SharedBackend


class Stub(GraphBackend):
    def __init__(self, model):  # no loading, no CUDA graphs
        self.model, self.dev = model, torch.device("cpu")


def test_pack():
    ids, pos, seg, last = GraphBackend._pack([[1, 2, 3, 4, 5], [1, 2, 3, 7, 8, 9]])
    assert ids == [1, 2, 3, 4, 5, 7, 8, 9] and pos == [0, 1, 2, 3, 4, 3, 4, 5]
    assert seg == [0, 0, 0, 1, 1, 2, 2, 2] and last == [4, 7]


def test_shared_equals_separate():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    be = Stub(m)
    a, b = [1, 5, 9, 11, 3, 4, 6], [1, 5, 9, 11, 3, 8, 2, 7]
    ids, pos, seg, last = GraphBackend._pack([a, b])
    h = be._forward_masked(torch.tensor([ids + [0] * 3]), be._mask_from_seg(torch.tensor([seg + [-1] * 3])),
                           torch.tensor([pos + [0] * 3]))
    with torch.no_grad():
        ha = m.model(input_ids=torch.tensor([a])).last_hidden_state[0, -1]
        hb = m.model(input_ids=torch.tensor([b])).last_hidden_state[0, -1]
    assert torch.allclose(h[0, last[0]], ha, atol=1e-4) and torch.allclose(h[0, last[1]], hb, atol=1e-4)


class CharTok:  # character-level stand-in for the tokenizer
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [1 + ord(c) % 49 for c in text]})()


@pytest.mark.parametrize("dev", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no Apple Silicon GPU"))])
def test_shared_backend_equals_separate(dev):
    """SharedBackend (the mps mode): packed option orders, several questions of different lengths in one padded batch."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg).eval().to(dev)
    m.config._attn_implementation = "sdpa"
    be = SharedBackend.__new__(SharedBackend)
    be.model, be.dev, be.tok, be.shared = m, torch.device(dev), CharTok(), True
    groups = [(["shared state one: A. x B. y", "shared state one: A. y B. x"], [[3, 4], [3, 4]]),
              (["a much longer shared state, number two: A. p B. q C. r", "a much longer shared state, number two: A. r B. q C. p"],
               [[5, 6, 7], [5, 6, 7]]),
              (["single order"], [[8, 9]])]
    got = be.run_shared(groups)
    for (prompts, ids), g in zip(groups, got):
        for p, i, probs in zip(prompts, ids, g):
            with torch.no_grad():
                lg = m(input_ids=torch.tensor([be.tok(p).input_ids], device=dev)).logits[0, -1].float()
            want = torch.softmax(torch.log_softmax(lg, -1)[i], -1).tolist()
            assert probs == pytest.approx(want, abs=1e-4)
