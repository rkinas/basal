"""MLX backend: equals the PyTorch readout (tiny random Llama with the basal layout: attention and MLP biases, untied
head, grouped KV heads, transformers-5 `rope_parameters`), with a shared prefix (SOAM group), for unrelated prompts and
with 8-bit weights made at load time (mlx-q8); through mlx-lm's forward (the default) and basal's own
(basal/mlx_forward.py, --mlx-forward basal)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import MLXBackend, mlx_config_overrides

FORWARDS = pytest.mark.parametrize("forward", MLXBackend.FORWARDS)


def tiny_llama(path, rope=None):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False,
                      rope_parameters=rope or {"rope_type": "default", "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # random biases (initialised to zero), so that a dropped bias would be noticed
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
            elif n.endswith(("q_proj.weight", "k_proj.weight")):
                p.mul_(8.0)  # peaky attention, so that positions (rotary settings) matter
    m.save_pretrained(path)
    return m


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).tolist())
    return out


class _Tok:
    pad_token_id = 0


def mlx_stub(path, quant=None, packed=False, forward="mlx-lm"):
    from concurrent.futures import ThreadPoolExecutor
    be = MLXBackend.__new__(MLXBackend)  # no tokenizer files in the tiny model directory
    be.pool, be.tok, be.max_rows = ThreadPoolExecutor(max_workers=1), _Tok(), 3
    be.pool.submit(be._load, path, quant, packed, forward).result()
    assert be.forward.split()[0] == ("mlx-lm" if be.fwd is None else "basal")
    return be


def test_rope_parameters_override():
    assert mlx_config_overrides({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}) == {"rope_theta": 1e6}
    assert mlx_config_overrides({"rope_theta": 5e5}) == {}
    o = mlx_config_overrides({"rope_parameters": {"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0}})
    assert o["rope_theta"] == 5e5 and o["rope_scaling"]["rope_type"] == "llama3"


STATE = list(range(21, 61)) * 3  # 120 shared tokens
# SOAM group: state | question 1 (two orders) | question 2 (two orders) | question 3, suffixes of unequal length
TOKS = [STATE + [5, 6, 10, 11], STATE + [5, 6, 11, 10], STATE + [7, 8, 9, 10, 11, 12], STATE + [7, 8, 9, 12, 11, 10],
        STATE + [3]]
LIDS = [[10, 11], [10, 11], [10, 11, 12], [10, 11, 12], [10, 11, 12, 13]]


@FORWARDS
def test_mlx_equals_torch(tmp_path, forward):
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path, forward=forward)
    assert (be.fwd is None) is (forward == "mlx-lm")
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    # unrelated prompts of different lengths (right-padded batches, no shared prefix), and a mix of both kinds
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    ref_s = reference(m, singles, sl)
    got_s = be.run(singles, sl)
    for r, g in zip(ref_s, got_s):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    mixed = be.run_shared([([singles[0]], [sl[0]]), (TOKS[:2], LIDS[:2])])
    assert max(abs(x - y) for x, y in zip(mixed[0][0], ref_s[0])) < 2e-4
    assert max(abs(x - y) for x, y in zip(mixed[1][1], ref[1])) < 2e-4


@FORWARDS
def test_mlx_without_override_differs(tmp_path, forward):
    """Guard for the override: with mlx-lm's default rope_theta (10000) the readout changes."""
    pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model
    from basal.mlx_forward import MLXForward
    m = tiny_llama(tmp_path)
    toks, lids = [list(range(21, 61)) * 3 + [5]], [[10, 11, 12, 13]]
    be = mlx_stub(tmp_path, forward=forward)
    be.model, _ = load_model(tmp_path)  # no override
    be.fwd = MLXForward.build(be.lm) if be.fwd else None
    assert max(abs(x - y) for x, y in zip(reference(m, toks, lids)[0], be.run(toks, lids)[0])) > 1e-3


def test_mlx_q8_quantises_decoder_blocks_only(tmp_path):
    pytest.importorskip("mlx.core")
    import mlx.nn as nn
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path, "q8")
    assert be.quantization == {"group_size": 64, "bits": 8}
    assert isinstance(be.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(be.model.model.layers[1].mlp.down_proj, nn.QuantizedLinear)
    assert not isinstance(be.model.lm_head, nn.QuantizedLinear)
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):  # 8-bit weights: close to bf16 / fp32, not equal
        assert max(abs(x - y) for x, y in zip(r, g)) < 0.05
    with pytest.raises(ValueError, match="unknown MLX quantisation"):
        mlx_stub(tmp_path, "q4")


@FORWARDS
def test_mlx_reads_255_format2_codes(tmp_path, forward):
    """Prompt format 2: up to 255 option codes, read from the head's logits at the prefill position (the code ids of the
    255-option test vector, docs/format2_vectors.jsonl), for a SOAM group of a 255-option question in two orders and a
    2-option question over the same state, against the PyTorch readout."""
    pytest.importorskip("mlx.core")
    import json
    from pathlib import Path
    lines = (Path(__file__).parents[1] / "docs/format2_vectors.jsonl").read_text(encoding="utf-8").splitlines()
    vec = next(v for v in map(json.loads, lines) if v["id"] == "en-choice-255")
    codes = vec["prompts"][0]["code_ids"]
    assert len(codes) == 255 and len(set(codes)) == 255
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=1024, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False,
                      rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
        m.lm_head.weight.mul_(20.0)  # spread the code logits, so that a wrong column or order would show
    m.save_pretrained(tmp_path)
    be = mlx_stub(tmp_path, forward=forward)
    state = list(range(100, 400))
    toks = [state + list(range(500, 760)) + [7], state + list(range(760, 500, -1)) + [7], state + [9, 8, 7]]
    lids = [codes, codes, codes[:2]]
    ref = reference(m, toks, lids)
    got = be.run_shared([(toks, lids)])[0]
    assert [len(g) for g in got] == [255, 255, 2]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
        assert abs(sum(g) - 1) < 1e-5


def test_mlx_code_logits_in_fp32(tmp_path):
    """bf16 weights: the code logits are the fp32 product of the hidden state and the head's code rows (as the PyTorch
    backends since the fp32 readout), not the bf16 head output, whose logits of magnitude 16-32 step by 0.125."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    m = tiny_llama(tmp_path)
    with torch.no_grad():
        m.lm_head.weight.mul_(40.0)
    m.to(torch.bfloat16).save_pretrained(tmp_path)
    be = mlx_stub(tmp_path)
    assert be.model.lm_head.weight.dtype == mx.bfloat16
    h = mx.array(np.random.default_rng(0).normal(size=(3, 64)).astype(np.float32)).astype(mx.bfloat16)
    lids = [[10, 11], [10, 11, 12], [13, 14, 15, 16]]
    got = be._letters(h, lids)
    W = np.array(be.model.lm_head.weight.astype(mx.float32))
    hf = np.array(h.astype(mx.float32))
    for g, ids, x in zip(got, lids, hf):
        z = W[ids] @ x
        ref = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        assert max(abs(a - b) for a, b in zip(g, ref)) < 1e-5
    z16 = np.array(be.model.lm_head(h).astype(mx.float32))
    assert np.abs(z16[0, [10, 11]] - W[[10, 11]] @ hf[0]).max() > 1e-3  # the bf16 head output is coarser


def tiny_gemma4(path, **kw):
    """A tiny random Gemma 4 decoder with the parts of basal-omni (Gemma 4 E4B): sliding-window and global layers
    (wider global heads, partial "proportional" rotary), per-layer inputs, KV sharing of the last layers, tied head and
    the final logit softcap; the head is scaled so that the softcap changes the code probabilities. kw: other config
    values (the 31B layout: attention_k_eq_v=True, no per-layer inputs, no KV sharing)."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(**{**dict(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, global_head_dim=32, sliding_window=16,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, num_kv_shared_layers=2, final_logit_softcapping=30.0,
        max_position_embeddings=1024,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                            "rope_theta": 1000000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}), **kw})
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(300.0)  # tied head: large code logits, which the softcap bends
    m.save_pretrained(path)
    return m


GEMMA_31B = dict(attention_k_eq_v=True, hidden_size_per_layer_input=0, num_kv_shared_layers=0)


@FORWARDS
@pytest.mark.parametrize("layout", ["e4b", "31b"])
def test_mlx_gemma4_equals_torch(tmp_path, forward, layout):
    """Gemma 4 (basal-omni): the readout applies the final logit softcap, and a shared prefix longer than the sliding
    window (120 tokens, window 16) keeps every position (mlx-lm's rotating cache would keep the last window only). The
    E4B layout and the 31B one (keys as values in the global layers, no per-layer inputs, no KV sharing)."""
    pytest.importorskip("mlx.core")
    m = tiny_gemma4(tmp_path, **(GEMMA_31B if layout == "31b" else {}))
    be = mlx_stub(tmp_path, forward=forward)
    assert be.softcap == 30.0
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    for r, g in zip(reference(m, singles, sl), be.run(singles, sl)):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    if layout == "31b":
        return
    be.softcap = None  # guard: without the softcap the probabilities differ
    assert max(abs(x - y) for r, g in zip(ref, be.run_shared([(TOKS, LIDS)])[0]) for x, y in zip(r, g)) > 1e-2


def test_gemma4_per_layer_config_override():
    """transformers 5 writes Gemma 4's global head size in per_layer_config (basal-omni-1.0 config.json); mlx-lm
    reads global_head_dim, for the decoder and inside the text_config of the multimodal wrapper."""
    t = {"layer_types": ["sliding_attention", "full_attention"], "head_dim": 256,
         "per_layer_config": {"01": {"head_dim": 512}}, "rope_parameters": {"full_attention": {}, "sliding_attention": {}}}
    assert mlx_config_overrides(t) == {"global_head_dim": 512}
    o = mlx_config_overrides({"model_type": "gemma4", "text_config": t})
    assert o["text_config"]["global_head_dim"] == 512 and o["text_config"]["head_dim"] == 256
    assert mlx_config_overrides({"text_config": {**t, "global_head_dim": 512}}) == {}


@FORWARDS
def test_mlx_packed_rows_equal_kv_cache(tmp_path, forward):
    """--mlx-packed, Llama: a group whose prefix and suffixes fit in PACK_TOKENS runs as one packed row (explicit
    positions, block mask), several groups of different lengths in one padded batch; the readouts equal the KV-cache
    path (PACK_TOKENS = 0) and the PyTorch readout. Without the flag (the default) and for Gemma 4 the KV cache stays."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    assert not mlx_stub(tmp_path, forward=forward).packed  # opt-in
    be = mlx_stub(tmp_path, packed=True, forward=forward)
    assert be.packed
    q2 = [list(range(30, 60)) + [4, 10, 11], list(range(30, 60)) + [4, 11, 10]]  # a shorter question, two orders
    groups = [(TOKS, LIDS), (q2, LIDS[:2]), (TOKS[2:4], LIDS[2:4])]
    calls = []
    packed_fwd = be._packed
    be._packed = lambda g: calls.append(len(g)) or packed_fwd(g)
    got = be.run_shared(groups)
    assert calls == [3]  # one forward for the three groups
    be.PACK_BUDGET = 100  # rows longer than a third of the budget: one forward each, the same readouts
    alone = be.run_shared(groups)
    assert calls == [3, 1, 1, 1]
    assert max(abs(a - b) for g, h in zip(got, alone) for x, y in zip(g, h) for a, b in zip(x, y)) < 1e-5
    calls.clear()
    be.PACK_BUDGET = MLXBackend.PACK_BUDGET
    be.PACK_TOKENS = 64  # between q2's packed row (35 tokens) and the others (over 120): q2 packs, the rest fall back
    mixed = be.run_shared(groups)
    assert calls == [1]
    assert max(abs(a - b) for g, h in zip(got, mixed) for x, y in zip(g, h) for a, b in zip(x, y)) < 1e-5
    calls.clear()
    be.PACK_TOKENS = 0
    cached = be.run_shared(groups)
    assert calls == []  # PACK_TOKENS = 0: every group through _group (_shared_len, KV cache)
    assert len(got) == len(cached) == len(groups)
    for (toks, lids), g, c in zip(groups, got, cached):
        ref = reference(m, toks, lids)
        assert len(g) == len(c) == len(ref) == len(toks)
        for r, x, y, ids in zip(ref, g, c, lids):
            assert len(r) == len(x) == len(y) == len(ids)
            assert max(abs(a - b) for a, b in zip(x, y)) < 1e-5
            assert max(abs(a - b) for a, b in zip(r, x)) < 2e-4
    tiny_gemma4(tmp_path / "g4")
    assert not mlx_stub(tmp_path / "g4", packed=True).packed


@pytest.mark.parametrize("rope, packed", [
    ({"rope_type": "linear", "rope_theta": 1e6, "factor": 2.0}, True),
    ({"rope_type": "llama3", "rope_theta": 1e6, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
      "original_max_position_embeddings": 64}, True),
    ({"rope_type": "dynamic", "rope_theta": 1e6, "factor": 2.0}, False),  # base depends on the length of the call
    ({"rope_type": "yarn", "rope_theta": 1e6, "factor": 2.0, "original_max_position_embeddings": 64}, False),
])
def test_mlx_packed_needs_static_rope(tmp_path, rope, packed):
    """--mlx-packed rotates each token as its own length-1 row: only RoPE layers whose frequencies do not depend on the
    sequence length are packed; the others (dynamic NTK, yarn, longrope) fall back to the KV cache, with the same
    readouts as PyTorch."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path, rope)
    be = mlx_stub(tmp_path, packed=True, forward="basal")
    assert be.packed is packed
    assert (be.fwd is not None) is packed  # basal's forward needs the same static RoPE
    groups = [(TOKS[:2], LIDS[:2])]
    got = be.run_shared(groups)
    for r, x in zip(reference(m, *groups[0]), got[0]):
        assert max(abs(a - b) for a, b in zip(r, x)) < 2e-4


MODELS = {"llama": tiny_llama, "gemma4": tiny_gemma4, "gemma4-31b": lambda path: tiny_gemma4(path, **GEMMA_31B)}


def mlx_decoder(path):
    """The mlx-lm decoder of a model directory, as MLXBackend loads it."""
    import json
    from mlx_lm.utils import load_model
    model, _ = load_model(path, model_config=mlx_config_overrides(json.loads((path / "config.json").read_text())))
    return model, getattr(model, "language_model", model)


SINGLES = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]


def padded(rows):
    import mlx.core as mx
    S = max(map(len, rows))
    return mx.array([r + [0] * (S - len(r)) for r in rows]), mx.array([[len(r) - 1] for r in rows])


@pytest.mark.parametrize("arch", MODELS)
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_mlx_forward_equals_mlx_lm(tmp_path, arch, dtype):
    """basal/mlx_forward.py with separate projections: every layer at every position (full=True) is mlx-lm's forward
    bit for bit, for right-padded rows, for a shared prefix's keys and values and for rows that continue it; the
    readout rows (the last layers at those rows only) are within float rounding of it."""
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from basal.mlx_forward import MLXForward, bits_equal
    m = MODELS[arch](tmp_path)
    if dtype == "bf16":
        m.to(torch.bfloat16).save_pretrained(tmp_path)
    model, lm = mlx_decoder(tmp_path)
    fwd = MLXForward(lm, merge=False)
    assert fwd.last == (3 if arch == "gemma4" else len(fwd.layers) - 1)  # E4B layout: the last 2 of 6 share keys
    tol = 1e-6 if dtype == "fp32" else 2 ** -7  # bf16: one rounding step (equal bits on an M4 Pro)

    def close(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        return mx.all(mx.abs(a - b) <= tol * (1 + mx.abs(b))).item()

    ids, ro = padded(SINGLES)
    ref = lm.model(ids)[mx.arange(len(SINGLES)), ro[:, 0]]
    assert bits_equal(fwd.hidden(ids, ro, full=True)[:, 0], ref)
    assert close(fwd.hidden(ids, ro)[:, 0], ref)
    state = TOKS[0][:120]
    cache = [KVCache() for _ in make_prompt_cache(model)]
    lm.model(mx.array([state]), cache=cache)
    kv_ref = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache]
    kv = fwd.prefix(mx.array([state]))
    assert len(kv) == len(kv_ref) and all(bits_equal(a, b) for x, y in zip(kv, kv_ref) for a, b in zip(x, y))
    ids, ro = padded([t[120:] for t in TOKS])
    B = len(TOKS)
    cache = []
    for k, v in kv_ref:
        c = KVCache()
        c.keys, c.values, c.offset = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0), k.shape[2]
        cache.append(c)
    ref = lm.model(ids, cache=cache)[mx.arange(B), ro[:, 0]]
    assert bits_equal(fwd.hidden(ids, ro, past=kv, full=True)[:, 0], ref)
    assert close(fwd.hidden(ids, ro, past=kv)[:, 0], ref)


@pytest.mark.parametrize("arch", MODELS)
def test_mlx_forward_merges_checked_projections(tmp_path, arch, monkeypatch):
    """Merged projections: the separate layers keep views of the merged arrays (the loaded weights, bit for bit), the
    merged forward equals the separate one bit for bit, and a self-check that finds other bits keeps the separate
    layers (and says so in the start-up line)."""
    pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten
    import basal.mlx_forward as mf
    m = MODELS[arch](tmp_path)
    m.to(torch.bfloat16).save_pretrained(tmp_path)
    _, lm = mlx_decoder(tmp_path)
    loaded = dict(tree_flatten(lm.parameters()))
    fwd = mf.MLXForward(lm)
    assert set(fwd.checks) == {"q/k/v", "k/v", "gate/up"} - ({"k/v"} if arch == "gemma4-31b" else set())
    merged = [g for g in fwd.qkv + fwd.gu if g is not None and g.cat is not None]
    assert merged and fwd.checks["gate/up"] == "merged"
    assert all(mf.bits_equal(loaded[k], v) for k, v in tree_flatten(lm.parameters()))
    for g in merged:
        for mod, (a, b) in zip(g.mods, zip([0] + g.cuts, g.cuts + [None])):
            assert mf.bits_equal(mod.weight, g.cat.weight[a:b])
    _, lm0 = mlx_decoder(tmp_path)
    sep = mf.MLXForward(lm0, merge=False)
    assert sep.checks == {} and "readout rows from layer" in sep.describe()
    ids, ro = padded(SINGLES)
    for full in (False, True):
        assert mf.bits_equal(fwd.hidden(ids, ro, full=full), sep.hidden(ids, ro, full=full))
    monkeypatch.setattr(mf, "bits_equal", lambda a, b: False)
    _, lm1 = mlx_decoder(tmp_path)
    kept = mf.MLXForward(lm1)
    assert all(g is None or g.cat is None for g in kept.qkv + kept.gu)
    assert set(kept.checks.values()) == {"kept (bits differ at 1 x 1)"} and "gate/up kept" in kept.describe()
    monkeypatch.undo()
    assert all(mf.bits_equal(loaded[k], v) for k, v in tree_flatten(lm1.parameters()))


def test_mlx_forward_flag(monkeypatch, tmp_path):
    """--mlx-forward: mlx-lm by default, basal on request (needs --mode mlx); an unknown value is refused at load."""
    import sys
    from basal import server as server_module
    assert server_module.parser().parse_args([]).mlx_forward == "mlx-lm"
    assert server_module.parser().parse_args(["--mlx-forward", "basal"]).mlx_forward == "basal"
    with pytest.raises(SystemExit):
        server_module.parser().parse_args(["--mlx-forward", "fast"])
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "eager", "--mlx-forward", "basal"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--mlx-forward needs --mode mlx"):
        server_module.main()
    pytest.importorskip("mlx.core")
    tiny_llama(tmp_path)
    with pytest.raises(ValueError, match="unknown MLX forward"):
        mlx_stub(tmp_path, forward="fast")


def tiny_qwen35(path, dtype=torch.float32):
    """A tiny Qwen3.5 decoder (Gated DeltaNet and attention layers, tied head as in basal-omni-pico) whose key head size
    (32) takes mlx-lm's Metal kernel of the gated delta rule; random (1 + w) norm weights."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4, full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"] * 2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=16,
        max_position_embeddings=512, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(0, 0.3)
    m.to(dtype).save_pretrained(path)
    return m


def test_qwen35_text_override():
    """transformers' text-only Qwen3.5 (model_type qwen3_5_text) loads as mlx-lm's qwen3_5."""
    o = mlx_config_overrides({"model_type": "qwen3_5_text", "rope_parameters": {"rope_type": "default",
                                                                                "rope_theta": 1e7}})
    assert o == {"model_type": "qwen3_5", "rope_theta": 1e7}
    assert "model_type" not in mlx_config_overrides({"model_type": "llama"})


def test_mlx_refuses_other_recurrent_models_before_loading(tmp_path, monkeypatch):
    """Only Qwen3.5's states are carried through a shared prefix; other recurrent models stop before mlx is imported."""
    import json
    import sys
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_next",
                                                      "layer_types": ["linear_attention", "full_attention"]}))
    monkeypatch.setitem(sys.modules, "mlx", None)  # importing mlx would fail
    be = MLXBackend.__new__(MLXBackend)
    with pytest.raises(ValueError, match="Qwen3.5.*--mode eager"):
        be._load(tmp_path)


def test_qwen35_norms_read_one_plus_w(tmp_path):
    from basal.engine import qwen35_norms
    m = tiny_qwen35(tmp_path, torch.bfloat16)
    got = qwen35_norms(tmp_path)
    sd = m.state_dict()
    assert set(got) == {k for k in sd if k.endswith("norm.weight") and ".linear_attn." not in k}
    for k, v in got.items():
        assert v.dtype == torch.float32 and torch.equal(v, 1.0 + sd[k].float())
    pytest.importorskip("mlx.core")
    be = mlx_stub(tmp_path)  # bf16 weights: (1 + w) exact in float32, not rounded to bf16
    import numpy as np
    w = np.array(be.lm.model.layers[0].input_layernorm.weight)
    assert np.array_equal(w, got["model.layers.0.input_layernorm.weight"].numpy())
    assert not np.array_equal(w, (1.0 + sd["model.layers.0.input_layernorm.weight"]).float().numpy())


def test_mlx_qwen35_equals_torch(tmp_path):
    """Qwen3.5 through mlx-lm's forward: a SOAM group (each suffix continues the prefix's Gated DeltaNet states and the
    attention layers' keys and values), unrelated prompts and a mix of both answer as each prompt's own float32
    forward."""
    pytest.importorskip("mlx.core")
    m = tiny_qwen35(tmp_path)
    ref = reference(m, TOKS, LIDS)
    be = mlx_stub(tmp_path)
    assert be.fwd is None and not be.packed
    for r, g in zip(ref, be.run_shared([(TOKS, LIDS)])[0]):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    ref_s = reference(m, singles, sl)
    for r, g in zip(ref_s, be.run(singles, sl)):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    mixed = be.run_shared([([singles[0]], [sl[0]]), (TOKS[:2], LIDS[:2])])
    assert max(abs(x - y) for x, y in zip(mixed[0][0], ref_s[0])) < 2e-4
    assert max(abs(x - y) for x, y in zip(mixed[1][1], ref[1])) < 2e-4


def test_mlx_qwen35_computes_in_float32(tmp_path):
    """bf16 weights are upcast at load: the readouts follow the float32 forward of the same bf16 weights, not a bf16
    one. Quantised weights (mlx-q8, a quantised export) are refused for Qwen3.5 before any weights are read."""
    import json
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    m = tiny_qwen35(tmp_path, torch.bfloat16).float()
    be = mlx_stub(tmp_path)
    assert {v.dtype for _, v in tree_flatten(be.model.parameters())} == {mx.float32}
    for r, g in zip(reference(m, TOKS, LIDS), be.run_shared([(TOKS, LIDS)])[0]):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    with pytest.raises(ValueError, match="quantised MLX weights are not checked for Qwen3.5"):
        mlx_stub(tmp_path, "q8")
    cfg = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "quantization": {"group_size": 64, "bits": 8}}))
    with pytest.raises(ValueError, match="quantised MLX weights are not checked for Qwen3.5"):
        mlx_stub(tmp_path)
