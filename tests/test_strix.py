"""--mode strix (StrixHaloFastBackend): every Triton kernel against its Hugging Face / PyTorch counterpart, the
interval form of the prefix-trie mask against the block mask it replaces, and the whole backend against separate
forwards of every prompt (tiny random Llama with biases, bf16, on the GPU)."""
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from basal import bench as bench_module
from basal import server as server_module
from basal.engine import GraphBackend

triton = pytest.importorskip("triton")
gpu = pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip), reason="needs a ROCm GPU and Triton")


class Tok:
    pad_token, eos_token, pad_token_id = "<pad>", "</s>", 0


def tiny(layers=2):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, attention_bias=True,
                      mlp_bias=True)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    with torch.no_grad():  # random biases (zero by default), so that a missing bias shows
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.05)
    return m


def make(m, warm=False):
    from basal.strix import StrixHaloFastBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("basal.strix.AutoTokenizer.from_pretrained", lambda _: Tok())
        mp.setattr("basal.strix.AutoModelForCausalLM.from_pretrained", lambda _, dtype: m.to(dtype))
        return StrixHaloFastBackend("tiny", warm=warm)


@pytest.fixture(scope="module")
def be():
    return make(tiny())


def request(state, n_q, first=20):
    """n_q questions over one state, each in two option orders (the server's layout)."""
    toks, lids = [], []
    for q in range(n_q):
        for o in range(2):
            toks.append(state + [first + q, 21, 30 + o, 31 - o, 40 + q % 9, 41])
            lids.append([7, 8, 9] if q % 2 else [7, 8])
    return toks, lids


def packed(be, groups):
    """Units of groups concatenated as one sequence: ids, positions, sub_end, and the ancestor mask of the packed
    tries (GraphBackend's block mask, rows of different units see nothing of each other)."""
    ids, pos, sub, seg_g, par_g, unit = [], [], [], [], [], []
    off = nb = 0
    for u, (pk, *_rest) in enumerate(be._units(groups)):
        t, pp, sg, last, par = pk
        ids += t; pos += pp; sub += be._sub_end(pk, off)
        seg_g += [nb + s for s in sg]
        par_g += [nb + p if p >= 0 else -1 for p in par]
        off += len(t); nb += len(par)
    anc = torch.zeros(nb, nb, dtype=torch.bool)
    for b, p in enumerate(par_g):
        if p >= 0:
            anc[b] = anc[p]
        anc[b, b] = True
    s = torch.tensor(seg_g)
    L = len(ids)
    mask = anc[s[:, None], s[None, :]] & torch.ones(L, L, dtype=torch.bool).tril()
    return ids, pos, sub, mask


def groups_mixed():
    random.seed(1)
    state = [random.randrange(2, 60) for _ in range(70)]
    return [request(state, 5), request(state[:9], 1), ([[3] + state[:40], [3] + state[:40] + [5, 6]], [[7, 8], [7, 8]]),
            request(state[5:60], 3, first=30)]


@gpu
def test_interval_mask_is_the_block_mask(be):
    """k <= q < sub_end[k] (one lookup per key) equals the ancestor mask of the packed prefix tries."""
    ids, pos, sub, mask = packed(be, groups_mixed())
    L = len(ids)
    q, k = torch.arange(L)[:, None], torch.arange(L)[None, :]
    assert torch.equal((k <= q) & (q < torch.tensor(sub)[None, :]), mask)


@gpu
def test_attention_kernel_matches_masked_softmax(be):
    ids, pos, sub, mask = packed(be, groups_mixed())
    L, H, KV, D = len(ids), be.H, be.KV, be.D
    torch.manual_seed(0)
    q = torch.randn(H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(KV, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(KV, L, D, device="cuda", dtype=torch.bfloat16)
    sub_t = torch.tensor(sub, device="cuda", dtype=torch.int32)
    kmin, kmax = be._key_bounds(sub_t)
    got = be._attention(q, k, v, sub_t, kmin, kmax).float()
    kk, vv = k.float().repeat_interleave(H // KV, 0), v.float().repeat_interleave(H // KV, 0)
    s = (q.float() @ kk.transpose(1, 2)) / D ** 0.5
    s = s.masked_fill(~mask.cuda()[None], float("-inf"))
    ref = (torch.softmax(s, -1) @ vv).transpose(0, 1).reshape(L, H * D)
    assert (got - ref).abs().max().item() < 2e-2
    # readout-only queries (the last layer): any ascending subset gives the same rows
    tok = torch.tensor(sorted(random.Random(3).sample(range(L), 23)), device="cuda", dtype=torch.int32)
    sel = be._attention(q, k, v, sub_t, kmin, kmax, tok=tok).float()
    assert torch.equal(sel, got[tok.long()])


@gpu
def test_rotary_matches_the_stock_model(be):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    L, H, KV, D = 90, be.H, be.KV, be.D
    torch.manual_seed(1)
    qkv = torch.randn(L, (H + 2 * KV) * D, device="cuda", dtype=torch.bfloat16)
    pos = torch.tensor(list(range(50)) + list(range(30, 70)), device="cuda", dtype=torch.int32)
    q, k, v = be._qkv_heads(qkv, pos)
    cos, sin = be.model.model.rotary_emb(qkv, pos[None].long())
    rq = qkv[:, : H * D].view(L, H, D).transpose(0, 1)[None]
    rk = qkv[:, H * D: (H + KV) * D].view(L, KV, D).transpose(0, 1)[None]
    eq, ek = apply_rotary_pos_emb(rq, rk, cos, sin)
    assert torch.equal(q, eq[0]) and torch.equal(k, ek[0])
    assert torch.equal(v, qkv[:, (H + KV) * D:].view(L, KV, D).transpose(0, 1))


@gpu
@pytest.mark.parametrize("M", [7, 200, 700, 2500])
def test_projections_match_linear(be, M):
    """qkv / o / down configs for every row-count bucket (row chunks past LINEAR_ROWS included): x @ w^T + b (+ r),
    as F.linear and a bf16 add."""
    lay = be.layers[0]
    torch.manual_seed(M)
    for kind, w, b in (("qkv", lay[1], lay[2]), ("o", lay[3], lay[4]), ("down", lay[10], lay[11])):
        x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
        r = torch.randn(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x, w, b)
        assert torch.allclose(be._linear(kind, x, w, b).float(), ref.float(), atol=2e-2, rtol=1e-2)
        ref_r = (ref + r)
        got = be._linear(kind, x, w, b, r.clone())
        assert torch.allclose(got.float(), ref_r.float(), atol=3e-2, rtol=1e-2)


@gpu
@pytest.mark.parametrize("M", [5, 300, 1500])
def test_ffn_and_norm_match_the_stock_layer(be, M):
    layer = be.model.model.layers[0]
    lay = be.layers[0]
    torch.manual_seed(M)
    x = torch.randn(M, be.hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        mlp_in = layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        norm = layer.input_layernorm(x)
    assert torch.allclose(be._ffn_in(x, lay[6], lay[7], lay[8], lay[9]).float(), mlp_in.float(), atol=2e-2, rtol=1e-2)
    assert torch.allclose(be._norm(x, lay[0]).float(), norm.float(), atol=1e-2, rtol=1e-2)


def separate(m, toks, lids):
    """Reference: one bf16 forward per prompt (the stock model), softmax over the letters."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t], device="cuda")).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).cpu())
    return out


def check(be, groups, atol=2e-2):
    res = be.run_shared(groups)
    for (toks, lids), got in zip(groups, res):
        for g, e in zip(got, separate(be.model, toks, lids)):
            assert torch.allclose(torch.tensor(g), e, atol=atol)


@gpu
@pytest.mark.parametrize("n_q", [1, 3, 8])
def test_questions_over_one_state_equal_separate(be, n_q):
    random.seed(n_q)
    check(be, [request([random.randrange(2, 60) for _ in range(120)], n_q)])


@gpu
def test_requests_in_one_sequence_equal_separate_and_alone(be):
    """Rows of several requests run as one sequence; every answer equals its separate forward and the same request
    served alone (rows of other requests are invisible)."""
    groups = groups_mixed()
    check(be, groups)
    alone = be.run_shared([groups[0]])[0]
    together = be.run_shared(groups)[0]
    for x, y in zip(alone, together):
        assert torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-2)


@gpu
def test_chunks_and_bisection(be, monkeypatch):
    """A token budget smaller than the requests: several forwards per call, a long request bisected (exact)."""
    monkeypatch.setattr(be, "TOKEN_BUDGET", 100)
    groups = groups_mixed()
    units = be._units(groups)
    assert len(be._chunks_tok(units)) > 1 and len(units) > len(groups)
    check(be, groups)


@gpu
def test_deterministic(be):
    g = groups_mixed()
    assert be.run_shared(g) == be.run_shared(g)


@gpu
def test_deterministic_algorithms(be):
    """--deterministic: nothing the backend runs is refused under torch's deterministic algorithms, and the answers do
    not change (torch.empty is then filled with NaN, so an output element no kernel writes would show)."""
    g = groups_mixed()
    ref = be.run_shared(g)
    torch.use_deterministic_algorithms(True)
    try:
        assert be.run_shared(g) == ref
    finally:
        torch.use_deterministic_algorithms(False)


@gpu
def test_warmup_with_a_small_vocabulary():
    """The start-up warmup (token ids up to 2100) runs on a model with 64 tokens."""
    assert make(tiny(), warm=True).run_shared(groups_mixed()[:1])


@gpu
def test_warmup_compiles_every_kernel_copy():
    """After the start-up warmup no row count compiles a kernel. Triton compiles one copy per tile config and per class
    of each integer argument (1, a multiple of 16, other), and the request that needs a new one waits for it (the first
    timed pass of S with 1 question and of M with 2 in the request matrix, before the warmup covered them). Three
    layers: the warmup forwards skip the middle one, and the forwards below run all three."""
    from basal import strix
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    if not hasattr(runtime, "jit_cache_hook"):
        pytest.skip("Triton without knobs.runtime.jit_cache_hook")
    # The hook runs at every miss of a kernel's in-process cache, before Triton looks in its disk cache, so a copy the
    # disk cache holds counts as built too.
    for k in (strix._attn_fwd, strix._swiglu_mm, strix._linear, strix._rmsnorm, strix._rope):
        k.device_caches.clear()  # the copies other tests compiled in this process
    be = make(tiny(layers=3), warm=True)
    assert len(be.layers) == 3
    built = []

    def hook(*, repr, **_):
        built.append(repr)
        return False  # compile as usual

    old, runtime.jit_cache_hook = runtime.jit_cache_hook, hook
    try:
        for L in (1, 20, 48, 208, 512, 513, 720, 1536, 1537, 2160, 2304, 2305, 2320, 4609, 5000):
            for nq in (1, 2, 7, 16, 48, 513, 528):
                if nq <= L:
                    be.forward(*be._warm_row(L, nq))
        be._wait()
    finally:
        runtime.jit_cache_hook = old
    assert built == []


def llama(**kw):
    base = dict(hidden_size=2048, intermediate_size=11008, num_attention_heads=16, num_key_value_heads=2,
                head_dim=128, num_hidden_layers=1, vocab_size=64)  # basal-1.5 / basal-2.0 4.5B shapes
    return LlamaConfig(**{**base, **kw})


def test_supported_shapes():
    from basal.strix import StrixHaloFastBackend as B
    assert B.unsupported(llama()) == [] and B.unsupported(tiny().config) == []
    assert B.unsupported(llama(num_attention_heads=32, num_key_value_heads=1, head_dim=64)) == []


@pytest.mark.parametrize("kw, why", [
    (dict(head_dim=80), "head size 80"),                                                    # tl.arange(0, D)
    (dict(num_attention_heads=24, num_key_value_heads=8), "24 query heads over 8"),         # 3 per kv head
    (dict(num_attention_heads=8, num_key_value_heads=1, head_dim=16), "qkv projection"),    # 160 columns, tiles of 128
    (dict(intermediate_size=5000), "gate / up projection"),
    (dict(hidden_size=3072), "hidden size 3072"),
    (dict(hidden_act="gelu"), "activation gelu"),
    (dict(rope_parameters={"rope_type": "dynamic", "rope_theta": 1e6, "factor": 2.0}), "RoPE type dynamic"),
    (dict(model_type="gemma4_unified"), "model type gemma4_unified"),                       # basal-omni
])
def test_unsupported_shapes_are_refused(kw, why):
    """Plain attributes: LlamaConfig itself refuses some of them (24 heads over a 2048 hidden size)."""
    from basal.strix import StrixHaloFastBackend as B
    assert any(why in p for p in B.unsupported(SimpleNamespace(**{**llama().to_dict(), **kw})))


def test_refused_without_rocm(monkeypatch):
    from basal.strix import StrixHaloFastBackend
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(ValueError, match="ROCm"):
        StrixHaloFastBackend("tiny")


class Made:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.tok = args, kwargs, None


def test_mode_selects_the_backend(monkeypatch, tmp_path):
    import basal.strix as strix_module
    monkeypatch.setattr(strix_module, "StrixHaloFastBackend", type("StrixHaloFastBackend", (Made,), {}))
    monkeypatch.setattr(server_module, "resolve", lambda *_, **kw: tmp_path)
    srv = server_module.Server(server_module.parser().parse_args(["--mode", "strix"]))
    assert type(srv.backend).__name__ == "StrixHaloFastBackend"
    assert type(bench_module.build("strix", "md")).__name__ == "StrixHaloFastBackend"
    assert server_module.MODES["strix"][0] == "strix" and "strix" in server_module.DETERMINISTIC_KINDS


def test_graph_backend_untouched():
    """The new mode lives in basal/strix.py; GraphBackend's packing is what it reuses."""
    assert GraphBackend._pack([[1, 2, 3], [1, 2, 4]])[4] == [-1, 0, 0]
