"""The Gemma 4 forward of the CUDA graph engine (basal/gemma4_graph.py) equals transformers' Gemma 4 (tiny random
models, CPU, float64 / float32): plain rows, the readout rows only, and SOAM's two passes through
GraphBackend.run_shared, with sliding-window and global layers, KV-shared layers and per-layer inputs (E2B / E4B) or
one projection for keys and values (12B / 31B)."""
import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from basal.engine import GraphBackend
from basal.gemma4_graph import Gemma4Graph, supports

S_, G_ = "sliding_attention", "full_attention"


def tiny(kind, dtype=torch.float64):
    """kind "e4b": per-layer inputs, the last 2 of 6 layers reuse keys / values; "31b": keys = values projection in
    the global layers (one KV head there). Window 5, so the states below are longer than it; random norms and layer
    scalars, so every weight matters."""
    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, head_dim=8, global_head_dim=16, sliding_window=5,
              layer_types=[S_, S_, G_, S_, S_, G_], final_logit_softcapping=3.0, max_position_embeddings=256)
    if kind == "e4b":
        kw.update(hidden_size_per_layer_input=8, vocab_size_per_layer_input=64, num_kv_shared_layers=2)
    else:
        kw.update(hidden_size_per_layer_input=0, attention_k_eq_v=True, num_global_key_value_heads=1)
    m = Gemma4ForCausalLM(Gemma4TextConfig(**kw)).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for layer in m.model.layers:
            layer.layer_scalar.fill_(0.9)
    m.config._attn_implementation = "sdpa"
    return m.to(dtype)


class Stub(GraphBackend):
    """GraphBackend without loading or CUDA graphs: every unit is longer than the largest graph (PLAIN_LENS / LENS
    = [4]), so it runs the uncompiled forwards (long single prompts, long roots, pass 2 at the readout rows)."""

    def __init__(self, model):
        self.model, self.dev, self._wrows, self.graphs = model, torch.device("cpu"), {}, {}
        self.gemma = Gemma4Graph(model)
        self.softcap = self.gemma.softcap
        self.tok = type("Tok", (), {"pad_token_id": 0})()
        self.shared, self.compact, self.prefix, self.state_cache = True, False, False, None
        self.PLAIN_LENS = self.LENS = [4]
        self.TOKEN_BUDGET = 64
        self._alloc_kv()

    def encode(self, prompts):
        return prompts


def separate(m, toks, lids):
    """Reference: one transformers forward per prompt, the full log-softmax of its (softcapped) logits, softmax over
    the codes."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("merge", [False, True])
def test_plain_rows_equal_transformers(kind, merge):
    m = tiny(kind)
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m.model(input_ids=ids).last_hidden_state
    g = Gemma4Graph(m, merge=merge)
    assert supports(m) is None and g.n_kv == (4 if kind == "e4b" else 6)
    assert g.merged == (6 if merge else 0) and len(g.qkv) == (g.n_kv if merge else 0)
    if merge:  # the modules' weights are views of the merged ones: transformers' forward is unchanged
        with torch.no_grad():
            assert torch.equal(m.model(input_ids=ids).last_hidden_state, ref)
    assert torch.allclose(g.forward(ids), ref, atol=1e-10)
    ro = torch.tensor([[12, 3], [7, 7]])  # the readout rows only (the KV-shared layers, the last own-key layer)
    assert torch.allclose(g.forward(ids, ro=ro), ref[torch.arange(2)[:, None], ro], atol=1e-10)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
def test_two_passes_equal_separate(kind):
    """Every question x option order of a request over one state (SOAM), a duplicate, and a prompt that is the shared
    prefix itself (its readout in pass 1): pass 1 over the root into per-layer key / value buffers (two head sizes),
    pass 2 over the branches with the trie mask and the window on positions, read at the readout rows only."""
    m = tiny(kind, torch.float32)  # the engine's additive masks are float32 at most
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]  # longer than the window: sliding layers drop its start
    toks = [state + [20 + q, 21, 30 + o, 31 - o, 40 + q, 41] + [50] * q for q in range(3) for o in range(2)]
    toks += [list(toks[0]), list(state)]
    lids = [[7, 8, 9] if q % 2 else [7, 8] for q in range(3) for o in range(2)] + [[7, 8], [7, 9]]
    ref = separate(m, toks, lids)
    res = be.run_shared([(toks, lids), ([toks[3]], [lids[3]])])
    for r, e in zip(res[0] + res[1], ref + [ref[3]]):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)


def test_attention_never_reaches_memory_efficient_sdpa(monkeypatch):
    """Every SDPA call of the Gemma 4 forward (plain rows, readout rows, both passes, the long lane) runs with the math
    backend only, grouped key / value heads unexpanded: memory-efficient SDPA reads wrong keys from expanded
    (zero-stride) heads (every global layer of 12B wrong, the 7 Oct bug); flex and FlashAttention 3 take grouped heads
    natively."""
    import torch.nn.functional as F
    calls, real = [], F.scaled_dot_product_attention

    def spy(q, k, *a, **kw):
        b = torch.backends.cuda
        calls.append((b.mem_efficient_sdp_enabled(), b.flash_sdp_enabled(), b.cudnn_sdp_enabled(),
                      k.shape[1] < q.shape[1] and kw.get("enable_gqa")))
        return real(q, k, *a, **kw)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    m = tiny("31b", torch.float32)
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    toks = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    be.run_shared([(toks, [[7, 8], [7, 8]])])
    be.run_long([(toks, [[7, 8], [7, 8]])])
    be.gemma.forward(torch.tensor([toks[0]]), ro=torch.tensor([[3, 15]]))
    assert calls and all(c == (False, False, False, True) for c in calls)


def test_moe_not_supported():
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16, enable_moe_block=True,
                           num_experts=2, top_k_experts=1, moe_intermediate_size=16, max_position_embeddings=64)
    assert "MoE" in supports(Gemma4ForCausalLM(cfg))


def test_readout_slots():
    """Pass-2 readouts (2, row, column) -> readout positions per row (padded with the row's first) and each readout's
    slot; a duplicate shares its column's slot, pass-1 readouts have none."""
    sel = [(2, 0, 5), (1, 1, 0), (2, 1, 3), (2, 0, 9), (2, 0, 5), (2, 1, 7)]
    ro, slot = GraphBackend._readout_slots(sel, 3)
    assert ro.tolist() == [[5, 9], [3, 7], [0, 0]] and slot == [0, None, 0, 1, 0, 1]


@pytest.mark.parametrize("L", [1, 129, 300, 1000])
@pytest.mark.parametrize("W", [None, 5, 128, 200, 512])
def test_plain_block_mask_is_the_causal_window(L, W):
    """The block mask of plain rows (built from block arithmetic): full blocks + the mask inside partial blocks give
    exactly the causal (sliding-window) mask."""
    g = Gemma4Graph(tiny("e4b"))
    g.window = W or 1
    bm = g._plain_block(L, "sliding_attention" if W else "full_attention", "cpu")
    q, k = torch.arange(L)[:, None], torch.arange(L)[None]
    true = (k <= q) & ((q - k < W) if W else True)
    got = torch.zeros(L, L, dtype=torch.bool)
    B = g.BLOCK
    for i in range(-(-L // B)):
        rows = slice(i * B, (i + 1) * B)
        for c in range(int(bm.full_kv_num_blocks[0, 0, i])):
            j = int(bm.full_kv_indices[0, 0, i, c])
            got[rows, j * B: (j + 1) * B] = True
        for c in range(int(bm.kv_num_blocks[0, 0, i])):
            j = int(bm.kv_indices[0, 0, i, c])
            got[rows, j * B: (j + 1) * B] = true[rows, j * B: (j + 1) * B]
    assert torch.equal(got, true)


def test_state_cache_second_round_equals_separate():
    """A request cut at its state (--state-cache-gb): the first call computes the state and stores its keys and values
    (own-key layers only, two head sizes), a second request over the same state runs pass 2 only over the cached
    ones; both equal separate prompts."""
    from basal.engine import StateCache
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.state_cache = StateCache(1 << 30)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    first = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    second = [state + [22, 23, 24, 30 + o] for o in range(2)]
    lids = [[7, 8], [7, 8]]
    for toks in (first, second):
        res = be.run_shared([(toks, lids, len(state))])[0]
        for r, e in zip(res, separate(m, toks, lids)):
            assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    s = be.state_cache.stats()
    assert s["entries"] == 1 and s["hits"] == 1 and s["misses"] == 1


def test_long_state_keeps_the_window_of_sliding_layers():
    """A root longer than the largest graph gets fresh key / value buffers whose sliding layers keep its last `window`
    tokens only (pass 2 reads no further back). A state cached that way by the long lane is reused by the main lane
    over a graph's buffers, its sliding layers' keys placed where the root ends. Both equal separate prompts."""
    from basal.engine import StateCache
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.gemma.MARGIN = 0  # (256 by default: longer than these states)
    assert [k.shape[2] for k, _ in be._kv_new(12)] == [5, 5, 12, 5]  # own layers: sliding, sliding, global, sliding
    be.state_cache = StateCache(1 << 30)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    first = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    second = [state + [22, 23, 24, 30 + o] for o in range(2)]
    lids = [[7, 8], [7, 8]]
    for r, e in zip(be.run_long([(first, lids, len(state))])[0], separate(m, first, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    be.PLAIN_LENS = be.LENS = [16]  # the state now fits the graphs' buffers: the main lane's pass 2 over them
    for r, e in zip(be.run_shared([(second, lids, len(state))])[0], separate(m, second, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    s = be.state_cache.stats()
    assert s["entries"] == 1 and s["hits"] == 1


def test_one_forward_below_the_threshold_equals_separate():
    """States and prompts beyond the largest graph but within one_forward run as one forward (2.3.0's path)."""
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.one_forward = 1 << 20
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    toks = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    res = be.run_shared([(toks, [[7, 8], [7, 8]]), ([state + [40, 41]], [[7, 9]])])
    for r, e in zip(res[0] + res[1], separate(m, toks, [[7, 8], [7, 8]]) + separate(m, [state + [40, 41]], [[7, 9]])):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)


def test_long_lane_equals_separate():
    """The long lane (run_long): every unit alone, one decoder layer at a time (between(i) before layer i)."""
    m = tiny("31b", torch.float32)
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    toks = [state + [20, 21, 30 + o, 31 - o] for o in range(2)] + [state + [40, 41]]
    lids = [[7, 8], [7, 8], [7, 9]]
    calls = []
    res = be.run_long([(toks[:2], lids[:2]), (toks[2:], lids[2:])], between=calls.append)
    for r, e in zip(res[0] + res[1], separate(m, toks, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    assert calls == list(range(6)) * 2  # the root and the single prompt layer by layer; pass 2 runs whole


def test_modelopt_fp8_gemma4_loads_tied_and_runs(tmp_path):
    """A ModelOpt FP8 Gemma 4 checkpoint (the -FP8 repositories): the decoder linears become FP8Linear, the output head
    is tied to the embedding (not in the checkpoint), and the graph engine's forward equals transformers' forward of
    the loaded model (the FP8 product emulated on CPU)."""
    import json

    from safetensors.torch import save_file

    from basal.engine import FP8Linear, load_modelopt_fp8
    m = tiny("e4b", torch.float32)
    m.save_pretrained(tmp_path)
    sd = {}
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name and "per_layer" not in name:
            w = mod.weight.detach().float()
            ws = w.abs().max() / 448.0
            sd[f"{name}.weight"] = (w / ws).to(torch.float8_e4m3fn)
            sd[f"{name}.weight_scale"], sd[f"{name}.input_scale"] = ws.reshape(()), torch.tensor(0.05)
    for k, v in m.state_dict().items():
        if k != "lm_head.weight" and k.rsplit(".", 1)[0] + ".weight_scale" not in sd:
            sd.setdefault(k, v.detach().clone())
    for f in tmp_path.glob("*.safetensors"):
        f.unlink()
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "model.safetensors"))
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "FP8"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    q = load_modelopt_fp8(tmp_path, "cpu", torch.float32)
    assert q.lm_head.weight.data_ptr() == q.model.embed_tokens.weight.data_ptr()
    assert sum(isinstance(x, FP8Linear) for x in q.modules()) > 0
    ids = torch.randint(0, 64, (1, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = q.model(input_ids=ids).last_hidden_state[0, -1]
    assert torch.allclose(Gemma4Graph(q).forward(ids, ro=torch.tensor([[12]]))[0, 0], ref, atol=1e-5)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
def test_chunked_long_states_and_incremental_states_equal_separate(kind):
    """Long prompts and roots in chunks over the keys / values before them (CHUNK 5, sliding layers keeping their
    window only), and incremental states: a growing state runs only its new tokens over the cached state, an edited
    one only after the common prefix. Every readout equals a separate forward of its prompt."""
    from basal.engine import StateCache
    m = tiny(kind, torch.float32)
    be = Stub(m)
    be.CHUNK, be.gemma.MARGIN = 5, 0
    be.state_cache = StateCache(1 << 30)
    be.state_cache.BLOCK = 4
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16, 17, 18]
    lids = [[7, 8], [7, 8]]
    for st in (state, state + [19, 20, 21, 22, 23, 24, 25], state[:9] + [26, 27, 28, 29, 30]):
        toks = [st + [20, 21, 30 + o, 31 - o] for o in range(2)]
        res = be.run_shared([(toks, lids, len(st)), ([st + [40, 41]], [[7, 9]])])
        for r, e in zip(res[0] + res[1], separate(m, toks, lids) + separate(m, [st + [40, 41]], [[7, 9]])):
            assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    s = be.state_cache.stats()
    assert s["prefix_hits"] == 1 and s["entries"] == 2  # only the extension reused KV; the edit is outside the window


def test_warm_captures_largest_first_then_long_pass2():
    """warm() (Gemma 4, basal 2.3.4): every graph's warm-up calls first, in the grid's order (torch.compile compiles
    there), then the captures, the most tokens first; the pass-2 graphs over roots longer than LENS after those, each
    captured as it is warmed up (_warm_long_sfx measures the graph pool as each one is captured). Replays still
    answer as separate forwards."""
    from basal.engine import _graph_tokens

    class Replay:  # a captured graph's stand-in: replay() recomputes its output from the static inputs, in place
        def __init__(self, fn, static):
            self.fn, self.static = fn, static
            self.out = fn(*static)

        def replay(self):
            self.out.copy_(self.fn(*self.static))

    class Order(Stub):
        def __init__(self, model):
            super().__init__(model)
            self.LENS, self.PLAIN_LENS, self.BATCHES, self.TOKEN_BUDGET = [8], [8, 16], [1, 2], 32
            self.SUFFIX_LENS, self.TREE_ROWS = [4, 8], [1, 2]
            self._alloc_kv()
            self.fwd, self.fwd_sfx = self._forward, self._forward_suffix
            self.events = []

        def _capture(self, key, fn, *static):
            self.events.append(("request", key))
            return GraphBackend._capture(self, key, fn, *static)

        def _warm_up(self, fn, static):
            self.events.append(("warm", None))
            fn(*static)

        def _record(self, key, fn, static):
            self.events.append(("capture", key))
            r = Replay(fn, static)
            self.graphs[key] = (r, *static, r.out)
            return self.graphs[key]

    m = tiny("e4b", torch.float32)
    be = Order(m)
    be.warm()
    ev = be.events
    long = [i for i, (kind, k) in enumerate(ev) if kind == "request" and k[0] == "x" and k[3] > be.LENS[-1]]
    assert long, "no pass-2 graph over a root beyond LENS"
    head = ev[: long[0]]
    kinds = [e[0] for e in head]
    first = kinds.index("capture")
    caps = [k for kind, k in head if kind == "capture"]
    assert "warm" not in kinds[first:] and kinds.count("warm") == len(caps) == len(set(caps)) > 4
    big = [k for k in caps if k[0] != "x"]  # (#41) plain / compact graphs the largest first, then pass 2 the smallest
    assert [_graph_tokens(k) for k in big] == sorted(map(_graph_tokens, big), reverse=True)
    p2 = [k for k in caps if k[0] == "x"]
    assert caps == big + p2 and [_graph_tokens(k) for k in p2] == sorted(map(_graph_tokens, p2))
    for i in long:  # each long pass-2 graph: warmed up, then captured at once
        assert ev[i + 1] == ("warm", None) and ev[i + 2] == ("capture", ev[i][1])
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]  # a root of 12 tokens: the pass-2 graphs over roots of 16
    toks = [state + [20 + q, 21, 30 + o, 31 - o] for q in range(2) for o in range(2)] + [[1, 5, 9, 20, 21]]
    lids = [[7, 8]] * 4 + [[7, 9]]
    res = be.run_shared([(toks[:4], lids[:4]), (toks[4:], lids[4:])])
    for r, e in zip(res[0] + res[1], separate(m, toks, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)


@pytest.mark.parametrize("env, want", [(None, True), ("1", True), ("0", False)])
def test_compiled_layers_in_inductor_deterministic_mode(monkeypatch, env, want):
    """The compiled layer functions ask Inductor for its deterministic mode per compile (no kernel config chosen by
    timing, so a cold and a warm compile cache give the same probabilities); TORCHINDUCTOR_DETERMINISTIC=0 turns it
    off."""
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: seen.append(kw) or fn)
    if env is None:
        monkeypatch.delenv("TORCHINDUCTOR_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("TORCHINDUCTOR_DETERMINISTIC", env)
    Gemma4Graph(tiny("e4b"), compile=True)
    assert len(seen) == 3 and all(kw == {"options": {"deterministic": want}} for kw in seen)
