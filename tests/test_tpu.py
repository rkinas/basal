"""TPU backend (JAX; runs on CPU here): the packed prefix-trie forward (state once, ask many) and the letter readout
equal separate PyTorch forwards of the same checkpoint (tiny random Llama with the basal-1.0 architecture options: attention / MLP biases,
GQA, rope_theta 1e6 in rope_parameters)."""
import pytest

pytest.importorskip("jax")

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import GraphBackend
from basal.tpu import TPUBackend, subtree_ends
from basal.tpu_kernels import attention, tree_mask

V = 64
PLATFORM = "tpu" if any(d.platform == "tpu" for d in pytest.importorskip("jax").devices()) else "cpu"  # TPU if present


class Tok:  # token ids given directly as lists
    pad_token, pad_token_id = "<pad>", 0

    def __call__(self, ids, add_special_tokens=False):
        return type("E", (), {"input_ids": list(ids)})()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, attention_bias=True, mlp_bias=True,
                      max_position_embeddings=128, rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():  # random biases (initialised to zero), so that a missing bias would show up
        torch.nn.init.normal_(p, std=0.2)
    d = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(d)
    return d, m


@pytest.fixture(scope="module")
def backend(ckpt):
    mp = pytest.MonkeyPatch()  # stand-in tokenizer instead of a Hugging Face tokenizer
    mp.setattr("basal.tpu.AutoTokenizer.from_pretrained", lambda _: Tok())
    mp.setattr(TPUBackend, "LENS", [16, 32])
    mp.setattr(TPUBackend, "BATCHES", [1, 2, 4])
    mp.setattr(TPUBackend, "LONG_LENS", [48, 64])  # above LENS[-1]: blocked attention
    mp.setattr(TPUBackend, "Q_BLOCK", 16)
    mp.setattr(TPUBackend, "KV_BLOCK", 8)
    mp.setattr(TPUBackend, "WARM_READOUTS", 8)
    yield TPUBackend(ckpt[0], "float32", warm=True, max_len=64, platform=PLATFORM)  # warm: all shapes compile
    mp.undo()


def torch_probs(m, ids, letters):
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([ids])).logits[0, -1].double()
    return torch.softmax(lg[letters], -1).numpy()


A, B = [1, 5, 9, 11, 3, 4, 6, 40, 41, 42], [1, 5, 9, 11, 3, 8, 2, 7, 50]
LETTERS = [10, 20, 30]


# One request over one state: two questions x two option orders, plus a third question that shares only the state.
STATE = [1, 5, 9, 11, 3]
SOAM = [STATE + [4, 6, 40, 41, 42], STATE + [4, 6, 41, 40, 42], STATE + [8, 2, 7, 50], STATE + [8, 2, 50, 7],
        STATE + [4, 12]]


def test_tree_mask_equals_graph_mask():
    class Stub(GraphBackend):
        def __init__(self):  # _mask_from_seg only needs the parameter dtype
            self.model = torch.nn.Linear(1, 1)

    ids, pos, seg, last, parent = GraphBackend._pack(SOAM)
    assert len(parent) > 3  # a real trie: state, shared question prefix, option-order leaves
    L = len(ids) + 3
    ref = Stub()._mask_from_seg(torch.tensor([seg + [-1] * 3]), [parent])[0, 0] == 0
    end = np.array([subtree_ends(seg, parent) + list(range(len(ids) + 1, L + 1))], np.int32)
    assert np.array_equal(np.asarray(tree_mask(end))[0], ref.numpy())


def test_soam_group_equals_separate_torch(backend, ckpt):
    _, m = ckpt
    (probs,) = backend.run_shared([(SOAM, [LETTERS] * len(SOAM))])
    for t, p in zip(SOAM, probs):
        assert np.allclose(p, torch_probs(m, t, LETTERS), atol=1e-5)


def test_run_shared_batch_and_long_prompt(backend, ckpt):
    _, m = ckpt
    rng = np.random.default_rng(0)
    long = [1] + rng.integers(3, V, 40).tolist()  # longer than the largest bucket: bucket 48, blocked attention
    groups = [([A, B], [LETTERS, LETTERS]), ([B[:5] + [12], B[:5] + [13, 14]], [LETTERS, [30, 20]]),
              ([long, long[:-1] + [4]], [LETTERS, LETTERS])]
    res = backend.run_shared(groups)
    for (prompts, lids), probs in zip(groups, res):
        for t, li, p in zip(prompts, lids, probs):
            assert np.allclose(p, torch_probs(m, t, li), atol=1e-5)
    assert np.allclose(backend.run([A], [LETTERS])[0], torch_probs(m, A, LETTERS), atol=1e-5)


def test_one_row_per_order(ckpt, monkeypatch):
    _, m = ckpt
    monkeypatch.setattr("basal.tpu.AutoTokenizer.from_pretrained", lambda _: Tok())
    be = TPUBackend(ckpt[0], "float32", shared=False, warm=False, platform=PLATFORM)
    (pa, pb), = be.run_shared([([A, B], [LETTERS, LETTERS])])
    assert np.allclose(pa, torch_probs(m, A, LETTERS), atol=1e-5)
    assert np.allclose(pb, torch_probs(m, B, LETTERS), atol=1e-5)


def test_blocked_attention_equals_full():
    rng = np.random.default_rng(1)
    b, nh, nkv, L, d = 2, 4, 2, 24, 8
    q, k, v = (rng.standard_normal((b, h, L, d)).astype(np.float32) for h in (nh, nkv, nkv))
    _, _, seg, _, parent = GraphBackend._pack(SOAM)
    n = len(seg)
    end = np.array([subtree_ends(seg, parent) + list(range(n + 1, L + 1)), list(range(1, L + 1))], np.int32)
    full = np.asarray(attention(q, k, v, end, 0.3))
    for qb, kvb in ((4, 4), (8, 4), (12, 8), (24, 6)):
        assert np.allclose(np.asarray(attention(q, k, v, end, 0.3, q_block=qb, kv_block=kvb)), full, atol=1e-5)


def test_long_prompt_refused_above_max_len(backend):
    with pytest.raises(ValueError, match="longer than --max-len 64"):
        backend.run_shared([([[1] + [5] * 70], [LETTERS])])


def test_unsupported_dtype_and_missing_device(ckpt, monkeypatch):
    import jax
    monkeypatch.setattr("basal.tpu.AutoTokenizer.from_pretrained", lambda _: Tok())
    with pytest.raises(ValueError, match="bfloat16 / float32"):
        TPUBackend(ckpt[0], "float16", warm=False, platform="cpu")
    assert TPUBackend(ckpt[0], "float32", warm=False, platform="cpu", max_len=4096).max_len == 128  # the model's
    if not any(dv.platform == "tpu" for dv in jax.devices()):
        with pytest.raises(RuntimeError, match="no tpu device"):
            TPUBackend(ckpt[0], "float32", warm=False)
