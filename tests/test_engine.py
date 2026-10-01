"""Shared-prefix packing is exactly equivalent to separate forward passes (tiny random model, CPU)."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal import engine
from basal.engine import GraphBackend


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


def test_resolve_skips_weights_for_file_backed_backends(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(name, revision=None, ignore_patterns=None):
        calls.append((name, revision, ignore_patterns))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    assert engine.resolve("org/model") == tmp_path
    assert engine.resolve("org/model", "rev", weights=False) == tmp_path
    assert calls[0] == ("org/model", None, None)
    assert calls[1][:2] == ("org/model", "rev") and "*.safetensors" in calls[1][2] and "*.gguf" in calls[1][2]
    assert engine.resolve(str(tmp_path), weights=False) == tmp_path and len(calls) == 2  # local directory: no download
