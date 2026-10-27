"""The eager backend's own Gemma 4 forward (basal/gemma4.py): shared prefixes once (SOAM), the layers from the last
one with its own keys and values at the answer rows only, media towers once per request. Every readout equals the
transformers forward of the whole prompt (fp32, tiny random models shaped like basal-omni: E4B-like with KV sharing
and per-layer inputs, 31B-like with keys = values and no sharing, a MoE block, and a multimodal model with images,
audio and the larger models' bidirectional image attention)."""
import copy
import json

import pytest
import torch
from transformers import (Gemma4AudioConfig, Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration,
                          Gemma4TextConfig, Gemma4VisionConfig)

from basal import engine
from basal.engine import EagerBackend
from basal import gemma4
from basal.gemma4 import Gemma4Forward, HostRows, offload_ple

ROPE = {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}
IMG, AUD, VID, BOI, EOI, BOA, EOA = 60, 61, 62, 58, 59, 57, 56


def text_config(**kw):
    base = dict(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=32, sliding_window=16,
                layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                vocab_size_per_layer_input=64, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
                final_logit_softcapping=30.0, max_position_embeddings=1024, rope_parameters=ROPE)
    return Gemma4TextConfig(**{**base, **kw})


CONFIGS = {
    "e4b": {},  # KV sharing of the last layers, per-layer inputs, wider global heads
    "31b": dict(num_kv_shared_layers=0, hidden_size_per_layer_input=0, attention_k_eq_v=True,
                use_bidirectional_attention="vision"),
    "moe": dict(num_kv_shared_layers=0, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=4,
                top_k_experts=2, moe_intermediate_size=32),
}


def tiny(name):
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(text_config(**CONFIGS[name])).eval()
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(30.0)  # tied head: code logits large enough for the softcap to bend them
    return m


def groups_of():
    """A state longer than the sliding window and CHUNK with three questions, a short single, a long single, a short
    group, and a group whose prompts are prefixes of each other."""
    g = torch.Generator().manual_seed(1)
    r = lambda n: torch.randint(1, 56, (n,), generator=g).tolist()  # noqa: E731
    state = r(50)
    return [[state + [3, 4, 5], state + [3, 9], state + [7] * 7 + [1]], [r(10)], [r(70)], [[1, 2, 3, 4], [1, 2, 3, 5, 6]],
            [[1, 2, 3], [1, 2, 3, 4]], [r(30) + [5], r(12)]]


def last_hidden(m, t):
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([t])).last_hidden_state[0, -1]


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("answer_rows", [True, False])
def test_forward_equals_transformers(name, answer_rows):
    m = tiny(name)
    g = Gemma4Forward(m, answer_rows=answer_rows)
    g.CHUNK = 24  # the state and the long single run in parts
    groups = groups_of()
    for grp, hs in zip(groups, g.hidden(groups)):
        for t, h in zip(grp, hs):
            r = last_hidden(m, t)
            assert (h - r).abs().max() <= 1e-4 * r.abs().max(), (name, t)


def varlen_reference(q, k, v, cu_q, cu_k, max_q, max_k, scale, window):
    """What a FlashAttention varlen kernel computes (causal aligned at each sequence's end, left window), per
    sequence with SDPA: [total q, heads, d] / [total k, kv heads, d] -> [total q, heads, d]."""
    out = []
    for b in range(len(cu_q) - 1):
        qs, ks, vs = q[cu_q[b]: cu_q[b + 1]], k[cu_k[b]: cu_k[b + 1]], v[cu_k[b]: cu_k[b + 1]]
        i = torch.arange(len(qs))[:, None] + len(ks) - len(qs)
        j = torch.arange(len(ks))[None]
        m = (j <= i) & (j > i - window) if window else j <= i
        o = torch.nn.functional.scaled_dot_product_attention(
            qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None], attn_mask=m, scale=scale,
            enable_gqa=True)
        out.append(o[0].transpose(0, 1))
    return torch.cat(out)


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("answer_rows", [True, False])
def test_varlen_kernels_pack_rows(name, answer_rows, monkeypatch):
    """A varlen kernel (here its reference) over the packed real queries and keys of every row: earlier passes' keys
    (a prefix in parts, sliding layers' last window - 1 of them), padding, the answer rows of G1."""
    monkeypatch.setitem(gemma4.VARLEN_KERNELS, "ref", varlen_reference)
    monkeypatch.setattr(Gemma4Forward, "_kernel", lambda self, t, dtype, blk: self.attn[t])
    m = tiny(name)
    for attn in ({"sliding_attention": "ref", "full_attention": "sdpa"}, {"sliding_attention": "ref",
                                                                           "full_attention": "ref"}):
        g = Gemma4Forward(m, answer_rows=answer_rows, attn=attn)
        g.CHUNK = 24
        groups = groups_of()
        for grp, hs in zip(groups, g.hidden(groups)):
            for t, h in zip(grp, hs):
                r = last_hidden(m, t)
                assert (h - r).abs().max() <= 1e-4 * r.abs().max(), (name, attn, t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the memory-efficient kernel runs on CUDA")
@pytest.mark.parametrize("hd", [256, 512])
@pytest.mark.parametrize("kv_heads", [1, 8])
def test_efficient_kernel_equals_math_on_one_mod_32(hd, kv_heads):
    """pytorch/pytorch#191937 (torch <= 2.13): the memory-efficient kernel returns wrong attention for expanded
    (zero-stride) keys / values at query lengths = 1 mod 32 with head size 256 / 512, as transformers passed max's one
    global KV head. The forward's dispatch copies them per head; every row equals the math kernel there."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    g = Gemma4Forward.__new__(Gemma4Forward)
    a = type("Attn", (), {"scaling": 1.0})()
    gen = torch.Generator(device="cuda").manual_seed(0)
    for L in (33, 289, 1857):
        q = torch.nn.functional.normalize(torch.randn(1, 16, L, hd, device="cuda", generator=gen), dim=-1) * 8
        k = torch.nn.functional.normalize(torch.randn(1, kv_heads, L, hd, device="cuda", generator=gen), dim=-1) * 8
        v = torch.randn(1, kv_heads, L, hd, device="cuda", generator=gen)
        i = torch.arange(L, device="cuda")
        mask = (i[None, :] <= i[:, None])[None, None]
        for dtype in (torch.float32, torch.bfloat16):
            args = [x.to(dtype) for x in (q, k, v)]
            with sdpa_kernel([SDPBackend.MATH]):  # "sdpa" is the math kernel only where EagerBackend made it so
                ref = g._attend("sdpa", a, *args, mask, None, None, None)
            got = g._attend("efficient", a, *args, mask, None, None, None)
            assert (got.float() - ref.float()).abs().max() < (1e-4 if dtype == torch.float32 else 0.1), (hd, L, dtype)


def test_flash_kernels_only_where_they_apply():
    g = Gemma4Forward(tiny("31b"), attn={"sliding_attention": "fa3", "full_attention": "efficient"})
    assert g._kernel("sliding_attention", torch.bfloat16, None) == "fa3"
    assert g._kernel("sliding_attention", torch.float32, None) == "sdpa"  # fp32: no flash kernel
    assert g._kernel("sliding_attention", torch.bfloat16, torch.zeros(1, 2)) == "sdpa"  # image blocks
    assert g._kernel("full_attention", torch.bfloat16, torch.zeros(1, 2)) == "efficient"  # global layers: causal only
    g.attn["full_attention"] = "fa3"
    g.block_types = {"sliding_attention", "full_attention"}  # unified models: image blocks in every layer
    assert g._kernel("full_attention", torch.bfloat16, torch.zeros(1, 2)) == "sdpa"


def test_attention_plan(monkeypatch):
    assert gemma4.attention_plan("math") == {}
    assert gemma4.attention_plan("sliding=fa3,full=efficient") == {"sliding_attention": "fa3",
                                                                   "full_attention": "efficient"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(engine, "attention_available", lambda impl: None)
    assert gemma4.attention_plan("auto") == {"sliding_attention": "fa3", "full_attention": "efficient"}
    monkeypatch.setattr(engine, "attention_available", lambda impl: "flash_attn_3 is not installed")
    assert gemma4.attention_plan("auto") == {"sliding_attention": "efficient", "full_attention": "efficient"}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))  # not measured on Blackwell: math
    assert gemma4.attention_plan("auto") == {}
    for bad in ("flash", "sliding=fa2", "global=fa3"):
        with pytest.raises(ValueError):
            gemma4.attention_plan(bad)


def test_helpers_for_other_backends():
    """text_model and is_gemma4 are imported by the CUDA graph engine's Gemma 4 path (keep their names)."""
    m = tiny("e4b")
    assert gemma4.text_model(m) is m.model and gemma4.is_gemma4(m)
    mm = tiny_mm(False, 2, 8)
    assert gemma4.text_model(mm) is mm.model.language_model and gemma4.is_gemma4(mm)
    assert not gemma4.is_gemma4(type("M", (), {"config": type("C", (), {"model_type": "llama"})()})())


def test_kv_sharing_layout():
    g = Gemma4Forward(tiny("e4b"))
    assert g.last_kv == 3 and g.src == {4: 3, 5: 2} and g.store == {2, 3}  # E4B: the last 18 of 42 layers share
    g = Gemma4Forward(tiny("31b"))
    assert g.last_kv == 5 and g.src == {} and g.bidir  # 12B / 31B: the last layer at the answer rows only


def test_answer_rows_skip_work():
    """Layers after the last one with its own keys and values see one row per prompt; a prefix stops at that layer."""
    m = tiny("e4b")
    seen = []
    for i, layer in enumerate(m.model.layers):
        layer.mlp.register_forward_hook(lambda mod, inp, out, i=i: seen.append((i, inp[0].shape[1])))
    g = Gemma4Forward(m)
    g.hidden([[list(range(1, 41)) + [5], list(range(1, 41)) + [6, 7]]])
    prefix = [(i, n) for i, n in seen if n == 40]
    assert [i for i, _ in prefix] == [0, 1, 2]  # the prefix: layers before the last own one (3), MLPs only there
    assert all(n == 1 for i, n in seen if i >= 3)  # the suffix pass from layer 3 on: the answer row of each prompt
    seen.clear()
    Gemma4Forward(m, answer_rows=False).hidden([[list(range(1, 41)) + [5], list(range(1, 41)) + [6, 7]]])
    assert all(n == 2 for i, n in seen if n != 40)  # without: every suffix token in every layer


def test_budgets_split_forwards(monkeypatch):
    m = tiny("e4b")
    groups = groups_of()
    g = Gemma4Forward(m)
    g.CHUNK = 24
    ref = g.hidden(groups)
    calls = []
    orig = Gemma4Forward._pass
    monkeypatch.setattr(Gemma4Forward, "_pass", lambda self, h, *a, **k: calls.append(h.shape) or orig(self, h, *a, **k))
    g.MAX_BATCH_TOKENS, g.ATTN_PAIRS = 40, 2000
    got = g.hidden(groups)
    assert max(s[0] * s[1] for s in calls) <= 40 and len(calls) > 8
    calls.clear()
    g.MAX_BATCH_TOKENS, g.ATTN_PAIRS, g.KV_BYTES = 16384, 2048 * 16384, 60 * g.kv_row_bytes
    assert g.hidden(groups[:1]) and [s[0] for s in calls[-2:]] == [1, 1]  # 3 suffixes over 50 keys: one per forward
    got2 = g.hidden(groups)
    for a, b, c in zip(ref, got, got2):
        for x, y, z in zip(a, b, c):
            assert torch.allclose(x, y, atol=1e-5) and torch.allclose(x, z, atol=1e-5)


@pytest.mark.parametrize("B,per_call", [(1, 10), (2, 3), (1, 1000)])
def test_split_attention_equals_one_call(B, per_call, monkeypatch):
    """An SDPA call past the budgets of a 16k-token context (only states over 16k tokens reach them) runs per KV head,
    its query heads folded into one block of queries, in parts within SCORES scores; together they equal the one call
    (fp32, a random mask)."""
    g = Gemma4Forward(tiny("31b"))
    a = g.layers[2].self_attn  # a global layer
    torch.manual_seed(2)
    H, G, Sq, Lk, d = 4, 2, 37, 90, 32
    q, k, v = torch.randn(B, H, Sq, d), torch.randn(B, G, Lk, d), torch.randn(B, G, Lk, d)
    mask = torch.rand(B, 1, Sq, Lk) < 0.6
    mask[..., -1] = True  # no empty row
    calls, sdpa = [], gemma4.F.scaled_dot_product_attention
    monkeypatch.setattr(gemma4.F, "scaled_dot_product_attention",
                        lambda q, *x, **kw: calls.append(q.shape) or sdpa(q, *x, **kw))
    one = g._attend("sdpa", a, q, k, v, mask, None, None, None)
    assert calls == [q.shape]
    calls.clear()
    g.ATTN_PAIRS, g.SCORES = 10, B * (H // G) * Lk * per_call  # past the budget; per_call queries a call
    split = g._attend("sdpa", a, q, k, v, mask, None, None, None)
    n = min(per_call, Sq)
    assert len(calls) == G * -(-Sq // n) and all(c[1] == 1 and c[2] <= (H // G) * n for c in calls)
    torch.testing.assert_close(split, one, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_long_states_split_equal_one_call(name, monkeypatch):
    """Every forward with the budgets of a 30-token "16k" context (the 50-token state and the 70-token single past it:
    their attention calls split, the state's global keys in buffers) equals the forward without splits and the
    transformers forward."""
    m = tiny(name)
    g = Gemma4Forward(m)
    g.CHUNK = 24
    groups = groups_of()
    ref = g.hidden(groups)
    n, split = [0], Gemma4Forward._split
    monkeypatch.setattr(Gemma4Forward, "_split", lambda self, *x: n.__setitem__(0, n[0] + 1) or split(self, *x))
    g.ATTN_PAIRS, g.KV_BYTES = 24 * 30, 30 * g.kv_row_bytes
    got = g.hidden(groups)
    assert n[0] > 10
    for grp, a, b in zip(groups, ref, got):
        for t, x, y in zip(grp, a, b):
            r = last_hidden(m, t)
            assert torch.allclose(x, y, atol=1e-5) and (y - r).abs().max() <= 1e-4 * r.abs().max()


class Stub(EagerBackend):
    def __init__(self, model, g4=True, proc=None):  # the real __init__ minus loading
        self.model, self.dev, self.prefill, self.proc = model, torch.device("cpu"), "", proc
        self.tok = type("Tok", (), {"pad_token_id": 0, "apply_chat_template": lambda _s, msgs, **_k: json.dumps(msgs)})()
        self.head = model.get_output_embeddings()
        self.softcap = model.config.get_text_config().final_logit_softcapping
        self._head_in = None
        self.head.register_forward_hook(self._grab)
        self.g4 = Gemma4Forward(model) if g4 else None

    def encode(self, prompts):
        return prompts


def test_eager_backend_soam_equals_transformers_path():
    m = tiny("e4b")
    groups = groups_of()
    lids = [[[10, 11, 12]] * len(g) for g in groups]
    new = Stub(m).run_shared(list(zip(groups, lids)))
    old = Stub(m, g4=False).run_shared(list(zip(groups, lids)))
    for a, b in zip(new, old):
        for x, y in zip(a, b):
            assert max(abs(p - q) for p, q in zip(x, y)) < 1e-5
    flat = [p for g in groups for p in g]
    for x, y in zip(Stub(m).run(flat, [[10, 11]] * len(flat)), Stub(m, g4=False).run(flat, [[10, 11]] * len(flat))):
        assert max(abs(p - q) for p, q in zip(x, y)) < 1e-5


def tiny_mm(bidir, shared, ple):
    torch.manual_seed(0)
    t = text_config(sliding_window=8, hidden_size_per_layer_input=ple, num_kv_shared_layers=shared,
                    use_bidirectional_attention="vision" if bidir else None)
    v = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=16, patch_size=2, pooling_kernel_size=2,
                           position_embedding_size=64)
    a = Gemma4AudioConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, subsampling_conv_channels=[128, 4],
                          output_proj_dims=32)
    c = Gemma4Config(text_config=t, vision_config=v, audio_config=a, image_token_id=IMG, audio_token_id=AUD,
                     video_token_id=VID, boi_token_id=BOI, eoi_token_id=EOI, boa_token_id=BOA, eoa_token_index=EOA)
    return Gemma4ForConditionalGeneration(c).eval()


class Proc:
    """The processor of one request: an image (4 x 4 patches -> 4 soft tokens) and 40 audio frames (10 soft tokens),
    placed where the state goes, before the prompt's own tokens (here the chat template's output is the JSON of the
    messages, a list of token ids)."""
    image_token_id, audio_token_id = IMG, AUD

    def __init__(self, n_audio=10):
        g = torch.Generator().manual_seed(2)
        self.px = torch.randn(1, 16, 12, generator=g)
        self.yx = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), -1).reshape(1, 16, 2)
        self.feats, self.n_audio = torch.randn(1, 40, 128, generator=g), n_audio

    def __call__(self, text, images=None, audio=None, **_):
        ids = [2, 5, 6, BOI] + [IMG] * 4 + [EOI, 7, BOA] + [AUD] * self.n_audio + [EOA, 9] + json.loads(text[0])
        ids = torch.tensor([ids])
        return dict(input_ids=ids, pixel_values=self.px, image_position_ids=self.yx, input_features=self.feats,
                    input_features_mask=torch.ones(1, 40, dtype=torch.bool),
                    mm_token_type_ids=torch.where(ids == IMG, 1, torch.where(ids == AUD, 3, 0)))


@pytest.mark.parametrize("bidir,shared,ple", [(True, 0, 0), (False, 2, 8)])
def test_media_once_equals_transformers_path(bidir, shared, ple, monkeypatch):
    m = tiny_mm(bidir, shared, ple)
    towers = []
    vt = m.model.vision_tower
    vt.register_forward_hook(lambda *_: towers.append(1))
    state = torch.randint(1, 50, (20,), generator=torch.Generator().manual_seed(3)).tolist()
    msgs = [state + [11, 12, 13], state + [11, 14], state + [15, 16, 17, 18]]
    lids = [[20, 21], [20, 21, 22], [23, 24]]
    for chunk in (2048, 6):  # 6: the prefix in parts around the image (never inside it)
        new = Stub(m, proc=Proc())
        new.g4.CHUNK = chunk
        towers.clear()
        got = new.run_media(msgs, lids, ["img"], ["aud"], 280)
        assert len(towers) == 1  # the image tower once for the three prompts
        ref = Stub(m, g4=False, proc=Proc()).run_media(msgs, lids, ["img"], ["aud"], 280)
        assert len(towers) == 4  # the transformers path: once per prompt
        for (p, n, nm), (q, n2, nm2) in zip(got, ref):
            assert (n, nm) == (n2, nm2) == (n, 14)
            assert max(abs(a - b) for a, b in zip(p, q)) < 1e-5
    if bidir:  # the image's soft tokens attend to each other in the sliding-window layers
        new.g4.bidir = False
        off = new.run_media(msgs, lids, ["img"], ["aud"], 280)
        assert max(abs(a - b) for (p, *_), (q, *_) in zip(off, got) for a, b in zip(p, q)) > 1e-4


def test_eager_forward_choice(tmp_path, monkeypatch):
    tiny("e4b").save_pretrained(tmp_path)
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())
    assert isinstance(EagerBackend(tmp_path, "float32", device="cpu").g4, Gemma4Forward)
    assert EagerBackend(tmp_path, "float32", device="cpu").g4.attn == {}  # --eager-attn math by default (2.3)
    import argparse
    from basal.server import add_engine_args
    assert add_engine_args(argparse.ArgumentParser()).get_default("eager_attn") == "math"
    assert EagerBackend(tmp_path, "float32", device="cpu", forward="hf").g4 is None
    monkeypatch.setenv("BASAL_EAGER_FORWARD", "hf")
    assert EagerBackend(tmp_path, "float32", device="cpu").g4 is None
    with pytest.raises(ValueError):
        EagerBackend(tmp_path, "float32", device="cpu", forward="fast")


def test_ple_offload_same_answers(tmp_path, monkeypatch):
    """--ple-offload on: the per-layer embedding table leaves the model's parameters for host memory; both forwards
    read the same rows, so the answers do not change."""
    m = tiny("e4b")
    emb = m.model.embed_tokens_per_layer
    ids = torch.tensor([[1, 5, 63, 0], [7, 7, 2, 9]])
    assert torch.equal(HostRows(emb)(ids), emb(ids))
    e16 = copy.deepcopy(emb).to(torch.bfloat16)
    assert torch.equal(HostRows(e16)(ids), e16(ids))
    m.save_pretrained(tmp_path)
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())
    groups = groups_of()
    lids = [[[10, 11, 12]] * len(g) for g in groups]
    ref = EagerBackend(tmp_path, "float32", device="cpu").run_shared(list(zip(groups, lids)))
    for fwd in ("basal", "hf"):
        be = EagerBackend(tmp_path, "float32", device="cpu", forward=fwd, ple_offload=True)
        assert isinstance(be.model.model.embed_tokens_per_layer, HostRows)
        assert not any("embed_tokens_per_layer" in n for n, _ in be.model.named_parameters())
        for a, b in zip(be.run_shared(list(zip(groups, lids))), ref):
            for x, y in zip(a, b):
                assert max(abs(p - q) for p, q in zip(x, y)) < 1e-5
    assert offload_ple(tiny("31b")) == 0  # no per-layer inputs: nothing to move
