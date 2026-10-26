"""Fast path routing on CPU (graphs replaced by eager replays of the same static buffers): single prompts in plain
graphs or a plain long forward, branched units in masked graphs or the two passes over the root's KV cache, all units
of all groups batched together and read out at once. Every readout equals a separate forward of its own prompt."""
import contextlib
import random

import pytest
import torch

from basal.engine import GraphBackend, causal_sdpa
from tests.test_engine import separate, tiny


class Replay:
    """Stand-in for a captured CUDA graph: replay() recomputes the output from the static inputs, in place. Its output
    is the tensor the function returned at capture, as on the GPU: for Llama graphs a view of the buffer every graph
    of its family writes (_shared_out)."""

    def __init__(self, fn, static):
        self.fn, self.static = fn, static
        self.out = fn(*static)

    def replay(self):
        self.out.copy_(self.fn(*self.static))


class Stub(GraphBackend):
    LENS, PLAIN_LENS, BATCHES, TOKEN_BUDGET, SUFFIX = [16, 32], [16, 32, 48], [1, 2, 3, 4], 96, 4
    TREE_ROWS = [1, 2, 4]

    def __init__(self, model, shared=True):
        self.model, self.dev, self._wrows, self.graphs, self.shared = causal_sdpa(model), torch.device("cpu"), {}, {}, shared
        self.tok = type("T", (), {"pad_token_id": 0})()
        self.fwd, self.fwd_sfx = self._forward, self._forward_suffix
        self._alloc_kv()
        self.replays = []

    def _capture(self, key, fn, *static):
        r = Replay(fn, static)
        self.graphs[key] = (r, *static, r.out)
        return self.graphs[key]

    def _plain_hidden(self, packs, idx):
        self.replays.append(("plain", len(idx), max(len(packs[k][0]) for k in idx)))
        return super()._plain_hidden(packs, idx)

    def _tree_hidden(self, packs, idx, *cache):
        self.replays.append(("tree", len(idx), max(packs[k][2].index(1) for k in idx)))
        return super()._tree_hidden(packs, idx, *cache)


def prompt(rnd, state, n):
    return state + [rnd.randrange(3, 50) for _ in range(n)]


def groups(rnd):
    s_short, s_long = [1] + [rnd.randrange(3, 50) for _ in range(9)], [1] + [rnd.randrange(3, 50) for _ in range(40)]
    g = []
    g.append(([prompt(rnd, s_short, 3)], [[7, 8]]))                                    # plain, short
    g.append(([prompt(rnd, s_long, 2)], [[7, 8, 9]]))                                   # plain, 43 tokens (graph 48)
    g.append(([prompt(rnd, s_long + s_short, 4)], [[7, 8]]))                            # plain, longer than 48: eager
    q = prompt(rnd, s_short, 2)
    g.append(([q + [10, 11], q + [11, 10]], [[10, 11], [10, 11]]))                       # two orders, masked graph
    q1, q2 = prompt(rnd, s_long, 2), prompt(rnd, s_long, 3)
    g.append(([q1 + [10, 11], q1 + [11, 10], q2 + [12, 13, 14], q2 + [14, 13, 12]],      # SOAM row > 32: two passes
              [[10, 11], [10, 11], [12, 13, 14], [12, 13, 14]]))
    g.append(([prompt(rnd, s_short, 1)] * 2, [[7, 8], [7, 8]]))                          # duplicates: one block
    return g


def check(be, gs):
    m = be.model
    res = be.run_shared(gs)
    assert len(res) == len(gs)
    for (toks, lids, *_), got in zip(gs, res):
        assert len(got) == len(toks)
        for r, e, ids in zip(got, separate(m, toks, lids), lids):
            assert len(r) == len(ids) and torch.allclose(torch.tensor(r), e, atol=1e-5)
    return res


def test_routing_and_identity():
    rnd = random.Random(0)
    be = Stub(tiny())
    gs = groups(rnd)
    check(be, gs)
    kinds = [r[0] for r in be.replays]
    assert kinds.count("tree") >= 1 and "plain" in kinds
    assert any(r[0] == "plain" and r[2] > 48 for r in be.replays)  # the 55-token prompt: no graph
    assert all(key[2] <= 48 for key in be.graphs)
    assert any(r[0] == "tree" and r[2] > 16 for r in be.replays)    # the long SOAM root (41 tokens) in a graph


def test_long_root_beyond_the_graphs():
    """A branched unit whose root is longer than the largest graph: pass 1 without a graph, then pass 2."""
    rnd = random.Random(3)
    be = Stub(tiny())
    calls = []
    be.fwd = lambda *a: calls.append(a[0].shape) or be._forward(*a)  # the (compiled) forward of the graphs
    s = [1] + [rnd.randrange(3, 50) for _ in range(60)]
    toks = [s + [10, 11], s + [11, 10], s + [5, 6, 12, 13, 14]]
    check(be, [(toks, [[10, 11], [10, 11], [12, 13, 14]])])
    assert ("tree", 1, 61) in be.replays and all(key[2] <= 48 for key in be.graphs)
    assert all(shape[1] <= 48 for shape in calls)  # the long root runs uncompiled (no recompile on long requests)


def test_many_groups_batch_across_requests():
    rnd = random.Random(1)
    be = Stub(tiny())
    gs = [g for _ in range(5) for g in groups(rnd)]
    check(be, gs)
    plain_rows = [r[1] for r in be.replays if r[0] == "plain"]
    assert max(plain_rows) > 1  # single prompts of different groups share a forward


def test_without_sharing_every_prompt_is_plain():
    rnd = random.Random(2)
    be = Stub(tiny(), shared=False)
    check(be, groups(rnd))
    assert {r[0] for r in be.replays} == {"plain"}


def test_plain_forward_builds_no_mask(monkeypatch):
    """With the basal attention name transformers creates no mask for a plain forward (SDPA is_causal), and a 4D mask
    passes through unchanged."""
    import torch.nn.functional as F
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, is_causal=False, **kw):
        seen.append((attn_mask is None, bool(is_causal)))
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kw)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    be = Stub(tiny())
    be._forward(torch.tensor([[1, 2, 3, 4]]))
    assert seen and all(s == (True, True) for s in seen)
    seen.clear()
    be._eager_shared(GraphBackend._pack([[1, 2, 3], [1, 2, 4]]), [[5, 6], [5, 6]])
    assert seen and all(s == (False, False) for s in seen)


def test_prompt_that_ends_at_the_root():
    """One prompt is the shared prefix of the others: its readout comes from pass 1."""
    rnd = random.Random(4)
    be = Stub(tiny())
    s = [1] + [rnd.randrange(3, 50) for _ in range(12)]
    toks = [s, s + [10, 11], s + [11, 10]]
    check(be, [(toks, [[7, 8], [10, 11], [10, 11]])])
    assert [r[0] for r in be.replays] == ["tree"]


def ref_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal=False, used=None):
    """basal.engine._varlen (torch's varlen attention) written out: per packed sequence, grouped-query heads, causal
    aligned at the end, the first used[i] keys of sequence i -> (o [Tq, H, d], lse [Tq, H] fp32)."""
    cu_q, cu_k = cu_q.tolist(), cu_k.tolist()
    used = used.tolist() if used is not None else None
    r = q.shape[1] // k.shape[1]
    o, lse = torch.zeros_like(q), torch.zeros(q.shape[:2])
    for i in range(len(cu_q) - 1):
        a, b, c = cu_q[i], cu_q[i + 1], cu_k[i]
        e = c + used[i] if used is not None else cu_k[i + 1]
        assert b - a <= max_q and e - c <= max_k
        s = torch.einsum("qhd,khd->hqk", q[a:b].float(), k[c:e].float().repeat_interleave(r, 1)) * scale
        if causal:
            s = s.masked_fill(torch.ones(b - a, e - c, dtype=torch.bool).triu(e - c - (b - a) + 1), float("-inf"))
        lse[a:b] = s.logsumexp(-1).transpose(0, 1)
        o[a:b] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v[c:e].float().repeat_interleave(r, 1)).to(q.dtype)
    return o, lse


def test_cascade_pass2_equals_separate(monkeypatch):
    """The CUDA cascade pass 2 (_varlen_tree, _cascade_varlen: the root's keys with per-row key counts, each trie block
    over its ancestors, merged by log-sum-exp) on the CPU, torch's varlen attention written out (ref_varlen): every
    readout equals its prompt run alone. Several units per forward with different root and branch lengths (padding,
    stale buffer keys past each root), nested option orders, readouts inside the root; below CASCADE_TOKENS the dense
    pass 2 runs."""
    import basal.engine as E
    calls = []
    monkeypatch.setattr(E, "_varlen", lambda *a, **k: calls.append(k.get("causal", False)) or ref_varlen(*a, **k))
    be = Stub(tiny())
    be._cascade_why, be.CASCADE_TOKENS = None, 0
    rnd = random.Random(8)
    s = [1] + [rnd.randrange(3, 50) for _ in range(12)]
    gs = groups(rnd) + [([s, s + [10, 11], s + [11, 10]], [[7, 8], [10, 11], [10, 11]])] + groups(rnd)
    check(be, gs)
    assert {True, False} <= set(calls) and any(r[0] == "tree" and r[1] > 1 for r in be.replays)
    calls.clear()
    be.CASCADE_TOKENS = 1 << 30
    check(be, gs)
    assert not calls


def test_per_layer_compiled_forward_equals_the_model_and_compiles_once():
    """--compile-scope layer: _forward_by_layer (the plain forward one decoder layer at a time through the compiled
    _layer_call, each layer's K / V buffers passed alone) gives, at every shape, the transformers forward's hidden
    states and fills the same key / value buffers (pass 2 reads them); backend eager: Dynamo's guards without code
    generation, so exact. The layer function runs once per layer and is traced once for every layer and shape, one row
    apart (size 1 is specialised): start-up then compiles one layer, not the model."""
    import torch._dynamo as dynamo
    from transformers import LlamaConfig, LlamaForCausalLM

    from basal.engine import _layer_call
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    be = Stub(m)
    dynamo.reset()
    dynamo.utils.counters.clear()
    compiled, calls = torch.compile(_layer_call, dynamic=True, backend="eager"), []
    be._layer_step = lambda *a: calls.append(1) or compiled(*a)
    shapes = ((1, 16), (2, 16), (3, 32), (2, 64), (4, 24))  # within the stub's 128-token key / value buffer
    for b, L in shapes:
        ids = torch.randint(3, 50, (b, L))
        got = be._forward_by_layer(ids, be._kv_views(b, L))
        kv = [(k.clone(), v.clone()) for k, v in be._kv_views(b, L)]
        ref = be._forward(ids, be._kv_views(b, L))  # the transformers forward rewrites the buffers
        assert torch.equal(got, ref)
        assert all(torch.equal(k, k0) and torch.equal(v, v0) for (k, v), (k0, v0) in zip(kv, be._kv_views(b, L)))
    assert len(calls) == len(shapes) * len(m.model.layers)
    assert 1 <= dynamo.utils.counters["stats"]["unique_graphs"] <= 2 < len(m.model.layers)


def test_per_layer_compile_refused_where_not_measured(tmp_path):
    """--compile-scope layer is the bf16 Llama forward's: FP8 / NVFP4 weights and --lean are refused before anything
    is loaded."""
    for kw, what in ((dict(quant="nvfp4"), "NVFP4"), (dict(lean=True), "--lean")):
        with pytest.raises(ValueError, match=f"compile-scope layer.*{what}"):
            GraphBackend(str(tmp_path), compile="layer", **kw)


def test_cascade_why(monkeypatch):
    """The cascade pass 2 needs CUDA, a model with attention layers only and no --lean; the CPU flash op that the CPU
    cascade needs (cascade_ok) plays no part on CUDA."""
    import basal.engine as E
    be = Stub(tiny())
    assert be._cascade_why is not None  # the CPU
    be = Stub(tiny())
    be.dev, be.lean = torch.device("cuda"), object()
    assert "lean" in be._cascade_why

    class NoFlash:
        def __getattr__(self, name):
            raise AttributeError(name)
    pytest.importorskip("torch.nn.attention.varlen")  # (its first import reads torch.ops.aten)
    monkeypatch.setattr(E.torch.ops, "aten", NoFlash())
    monkeypatch.setattr(E, "torch_varlen_available", lambda dtype: None)
    be = Stub(tiny())
    be.dev = torch.device("cuda")
    assert be._cascade_why is None and not E.cascade_ok(be.model.config)


def test_pass2_graphs():
    """With the pass-2 graphs captured (warm), short branch rows replay a graph; longer ones run without one."""
    class Warm(Stub):
        SUFFIX_LENS = [8, 16]
    rnd = random.Random(5)
    be = Warm(tiny())
    be.warm()
    used = []
    for key, g in be.graphs.items():
        if key[0] == "x":
            g[0].replay = (lambda r, k: lambda: (used.append(k), Replay.replay(r)))(g[0], key)
    gs = [g for _ in range(3) for g in groups(rnd)]
    s = [1] + [rnd.randrange(3, 50) for _ in range(10)]
    gs.append(([s + list(range(3, 30)), s + list(range(30, 3, -1))], [[7, 8], [7, 8]]))  # 27-token branches: no graph
    check(be, gs)
    assert used and all(k[2] in (8, 16) for k in used)


def test_shared_graph_outputs_keep_every_readout():
    """Llama graphs share one output buffer per family (plain / pass 2) and one pass-2 mask buffer: a unit read out
    both inside its root (pass 1's buffer) and below it (a pass-2 graph's), in one call with plain prompts and other
    units, still reads its own forward's values, and the captured graphs keep no output of their own."""
    class Warm(Stub):
        SUFFIX_LENS = [8, 16]
    rnd = random.Random(6)
    be = Warm(tiny())
    be.warm()
    used = []
    for key, g in be.graphs.items():
        if key[0] == "x":
            g[0].replay = (lambda r, k: lambda: (used.append(k), Replay.replay(r)))(g[0], key)
    s = [1] + [rnd.randrange(3, 50) for _ in range(12)]
    q = prompt(rnd, s, 8)  # rows of 10 tokens below the root: a pass-2 graph of 16 that covers the root's 13
    check(be, [([s, q + [10, 11], q + [11, 10]], [[7, 8], [10, 11], [10, 11]]),  # readouts in the root and below
               ([prompt(rnd, s, 5)], [[7, 8, 9]])])
    assert used == [("x", 1, 16, 16)] and ("tree", 1, 13) in be.replays
    check(be, groups(rnd))
    for fam in "px":
        assert len({g[-1].untyped_storage().data_ptr() for k, g in be.graphs.items() if k[0] == fam}) == 1
    assert len({g[2].untyped_storage().data_ptr() for k, g in be.graphs.items() if k[0] == "x"}) == 1


def test_each_graph_warmed_up_once_before_its_capture(monkeypatch):
    """The engine's _capture / _warm_up (torch.cuda's stream calls as no-ops on the CPU): a graph's function runs once
    before its capture, the first call that compiles and sets up what the capture cannot. 3 calls per plain graph
    (2 per pass-2 graph) cost 414 s of basal-2.0-4.5B's 1,094 s warm-up on an RTX 4060 Ti."""
    class Stream:
        def wait_stream(self, other):
            pass
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    calls, recorded = [], []
    be = GraphBackend.__new__(GraphBackend)
    be._side, be._deferred = None, {}
    be._record = lambda key, fn, static: recorded.append(key)
    for key in (("p", 2, 16), ("x", 1, 8, 16), ("p", 2, 16)):  # the last one waits already: not warmed up again
        GraphBackend._capture(be, key, lambda *s, k=key: calls.append((k, s)), "static")
    assert calls == [(("p", 2, 16), ("static",)), (("x", 1, 8, 16), ("static",))] and not recorded
    be._deferred = None  # outside warm(): warmed up once, then captured
    GraphBackend._capture(be, ("c", 64), lambda *s: calls.append((("c", 64), s)), "static")
    assert calls[2:] == [(("c", 64), ("static",))] and recorded == [("c", 64)]


@pytest.mark.parametrize("prefix", [False, True])
def test_warm_captures_largest_first_after_every_warm_up(prefix):
    """warm() (Llama) runs every graph's warm-up calls first, in the grid's order (torch.compile compiles there, so
    the compiled kernels do not change), then captures the plain graphs, the most tokens first, so each capture can
    reuse the pool blocks a larger one freed, then the pass-2 graphs, the smallest first (_capture_deferred's memory
    budget keeps the commonest); replays still answer as separate forwards. With --prefix-cache the two pass-2
    families share shapes: each graph is still warmed up and captured once."""
    from basal.engine import _graph_tokens

    class Order(Stub):
        SUFFIX_LENS = PREFIX_SFX = [8, 16]
        PREFIX_ROWS = [1, 2, 4]

        def _capture(self, key, fn, *static):  # the engine's _capture, its two steps observed
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

    Order.prefix = prefix
    be = Order(tiny())
    be.events = []
    be.warm()
    kinds = [e[0] for e in be.events]
    first = kinds.index("capture")
    caps = [k for kind, k in be.events if kind == "capture"]
    requests = [k for kind, k in be.events if kind == "request"]
    assert "warm" not in kinds[first:] and kinds.count("warm") == len(caps) == len(set(caps)) == len(set(requests)) > 20
    assert (len(requests) > len(set(requests))) == prefix  # the shared shapes asked for twice
    plain, pass2 = [k for k in caps if k[0] != "x"], [k for k in caps if k[0] == "x"]
    assert caps == plain + pass2 and plain and pass2
    assert [_graph_tokens(k) for k in plain] == sorted(map(_graph_tokens, plain), reverse=True)
    assert [_graph_tokens(k) for k in pass2] == sorted(map(_graph_tokens, pass2))
    check(be, groups(random.Random(7)))


@pytest.mark.parametrize("card", [24, 1000])
def test_pass2_graphs_captured_while_memory_stays_free(card):
    """With little GPU memory (simulated: every captured graph takes one unit of a `card`-unit GPU, PASS2_RESERVE 10)
    every plain graph is captured, then the pass-2 graphs, the smallest first, while PASS2_RESERVE stays free; a
    pass 2 whose graph was not captured runs without one, and every readout equals its prompt run alone. With room
    for all of them, all are captured."""
    from basal.engine import _graph_tokens

    class Tight(Stub):
        SUFFIX_LENS, PASS2_RESERVE = [8, 16], 10
        _capture = GraphBackend._capture

        def _warm_up(self, fn, static):
            fn(*static)

        def _record(self, key, fn, static):
            r = Replay(fn, static)
            self.graphs[key] = (r, *static, r.out)
            return self.graphs[key]

        def _free_memory(self):
            return card - len(self.graphs)

    be = Tight(tiny())
    be.warm()
    every = [("x", b, S, L) for b in be.TREE_ROWS for L in be.LENS for S in be.SUFFIX_LENS if be._pass2_ok(b, L)]
    plain = [k for k in be.graphs if k[0] == "p"]
    kept = sorted((k for k in be.graphs if k[0] == "x"), key=_graph_tokens)
    assert len(plain) == len(be.grid(be.PLAIN_LENS))
    assert kept == sorted(every, key=_graph_tokens)[: len(kept)]
    assert len(kept) == (card - len(plain) - be.PASS2_RESERVE if card < 100 else len(every))
    assert card - len(be.graphs) >= be.PASS2_RESERVE  # the reserve stays free after the last capture
    rnd = random.Random(9)
    roots = [[1] + [rnd.randrange(3, 50) for _ in range(11)] for _ in range(4)]
    four = [([r + [10, 11], r + [11, 10]], [[10, 11], [10, 11]]) for r in roots]  # 4 units, 4 tokens below 12
    assert (("x", 4, 8, 16) in be.graphs) == (card >= 100)
    check(be, four + groups(rnd))


def test_gemma_long_root_graphs_keep_the_reserve():
    """Gemma 4's pass-2 graphs over roots longer than LENS (_warm_long_sfx) also stop where PASS2_RESERVE would no
    longer stay free (simulated: one unit per graph), not only at PASS2_POOL, which counts the pool and not the
    memory a graph keeps for its kernels."""
    from types import SimpleNamespace

    class Gemma(GraphBackend):
        LENS, PLAIN_LENS, BATCHES, TOKEN_BUDGET = [16, 32], [16, 32, 64], [1, 2, 4], 128
        SUFFIX_LENS, TREE_ROWS, PASS2_RESERVE = [8, 16], [1, 2], 10

    be = Gemma.__new__(Gemma)
    be.gemma, be.compact, be.shared, be.prefix, be.dev = object(), False, True, False, torch.device("cpu")
    be.model, be.graphs = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=0)), {}
    be._graph = lambda b, L: be._capture(("p", b, L), lambda: None)
    be._graph_sfx = lambda b, S, L: be._capture(("x", b, S, L), lambda: None)
    be._warm_up = lambda fn, static: None
    be._record = lambda key, fn, static: be.graphs.setdefault(key, key)
    grid = len(be.grid(be.PLAIN_LENS)) + 2 * 2 * 2  # plain (b <= 4) + pass 2 within LENS: all fit on this card
    be._free_memory = lambda: grid + 2 + be.PASS2_RESERVE - len(be.graphs)  # room for two more graphs
    be.warm()
    long = [k for k in be.graphs if k[0] == "x" and k[3] > be.LENS[-1]]
    assert len(be.graphs) == grid + 2 and len(long) == 2
    assert grid + 2 + be.PASS2_RESERVE - len(be.graphs) >= be.PASS2_RESERVE



def test_gemma_warm_captures_largest_first_then_the_long_roots():
    """Gemma 4 (basal-omni): every graph of the grid, compact graphs included, is warmed up first, then the compact
    and plain graphs are captured the largest first and the pass-2 graphs within LENS the smallest first; then the
    pass-2 graphs over roots longer than LENS (_warm_long_sfx), each captured right after its warm-up, so the
    PASS2_POOL budget measures only what they add. The graph builders only record."""
    from types import SimpleNamespace

    from basal.engine import _graph_tokens

    class Gemma(GraphBackend):
        LENS, PLAIN_LENS, BATCHES, TOKEN_BUDGET = [16, 32], [16, 32, 64], [1, 2, 4], 128
        SUFFIX_LENS, TREE_ROWS, COMPACT_LENS = [8, 16], [1, 2], [16, 32, 64, 128]

    be, ev = Gemma.__new__(Gemma), []
    be.gemma, be.compact, be.shared, be.prefix, be.dev = object(), True, True, False, torch.device("cpu")
    be.model, be.graphs = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=0)), {}
    be._graph_c = lambda T: be._capture(("c", T), lambda: None)
    be._graph = lambda b, L: be._capture(("p", b, L), lambda: None)
    be._graph_sfx = lambda b, S, L: be._capture(("x", b, S, L), lambda: None)
    be._warm_up = lambda fn, static: ev.append(("warm", None))
    be._record = lambda key, fn, static: ev.append(("capture", key))
    be.warm()
    first = next(i for i, (kind, k) in enumerate(ev) if kind == "capture" and k[0] == "x" and k[3] > be.LENS[-1])
    grid, long = ev[:first - 1], ev[first - 1:]
    caps = [k for kind, k in grid if kind == "capture"]
    assert [kind for kind, _ in grid] == ["warm"] * len(caps) + ["capture"] * len(caps)
    plain, pass2 = [k for k in caps if k[0] != "x"], [k for k in caps if k[0] == "x"]
    assert caps == plain + pass2
    assert [_graph_tokens(k) for k in plain] == sorted(map(_graph_tokens, plain), reverse=True)
    assert [_graph_tokens(k) for k in pass2] == sorted(map(_graph_tokens, pass2))
    want = {("c", T) for T in (16, 32, 64, 128)}  # compact rows: plain graphs of at most TREE_ROWS[-1] rows
    want |= {("p", b, L) for L in (16, 32, 64) for b in (1, 2) if b * L <= 128 or b == 1}
    want |= {("x", b, S, L) for b in (1, 2) for L in (16, 32) for S in (8, 16)}  # rows x root within the budget
    assert len(caps) == len(set(caps)) and set(caps) == want
    assert [kind for kind, _ in long] == ["warm", "capture"] * (len(long) // 2)
    assert {k for kind, k in long if kind == "capture"} == {("x", b, S, 64) for b in (1, 2) for S in (8, 16)}



def test_pass2_graphs_are_the_reachable_shapes():
    """warm() captures pass-2 graphs (rows, root length) exactly for the batches pass 1 can form (_chunks; state-cached
    roots are chunked the same way), and the key / value buffer holds the largest of those, not TREE_ROWS[-1] x
    LENS[-1]. The default grid; the graph builders only record what warm() asks for."""
    from types import SimpleNamespace
    be = GraphBackend.__new__(GraphBackend)
    be.shared, be.model, asked = True, SimpleNamespace(config=SimpleNamespace(max_position_embeddings=0)), set()
    be._graph = lambda b, L: None
    be._graph_sfx = lambda b, S, L: asked.add((b, L))
    be.warm()
    reach = set()
    for L in be.LENS:
        for n in range(1, be.TREE_ROWS[-1] + 1):
            for idx in be._chunks([L] * n, be.PLAIN_LENS, [8] * n, be.TREE_ROWS[-1]):
                reach.add((be._bucket(len(idx), be.TREE_ROWS), L))
    assert asked == reach
    assert be._kv_tokens() == max(be.TOKEN_BUDGET, be.PLAIN_LENS[-1], max(b * L for b, L in reach))
    assert be._kv_tokens() < be.TREE_ROWS[-1] * be.LENS[-1]


def test_layered_forward_is_the_forward():
    """The long lane's forward, one decoder layer at a time, is the plain forward exactly (fp32): the same hidden
    states and the same keys / values written for pass 2."""
    be = Stub(tiny())
    ids = torch.tensor([[1] + [random.Random(5).randrange(3, 50) for _ in range(30)]])
    H, D = be.kv_shape
    kv1 = [(torch.zeros(1, H, 31, D), torch.zeros(1, H, 31, D)) for _ in be._kv]
    kv2 = [(torch.zeros(1, H, 31, D), torch.zeros(1, H, 31, D)) for _ in be._kv]
    seen = []
    assert torch.equal(be._forward_layered(ids, kv1, seen.append), be._forward(ids, kv2))
    assert seen == [0, 1] and all(torch.equal(a, b) for x, y in zip(kv1, kv2) for a, b in zip(x, y))


def test_long_lane_answers_as_the_main_lane():
    """run_long (every unit alone, layered, fresh buffers) gives the answers of run_shared and of separate forwards,
    for plain prompts, two orders and SOAM rows, with and without the state cache (a state stored by one lane is a
    hit for the other)."""
    from basal.engine import StateCache
    from tests.test_state_cache import requests
    rnd = random.Random(4)
    be = Stub(tiny())
    gs = groups(rnd)
    main, long = be.run_shared(gs), be.run_long(gs, between=lambda i: None)
    for (toks, lids, *_), a, b in zip(gs, main, long):
        for x, y, e in zip(a, b, separate(be.model, toks, lids)):
            assert torch.allclose(torch.tensor(y), e, atol=1e-5) and torch.allclose(torch.tensor(x), torch.tensor(y),
                                                                                  atol=1e-5)
    be.state_cache = StateCache(1e9)
    s = [1] + [rnd.randrange(3, 50) for _ in range(60)]                  # a root beyond the graphs
    cached = requests(rnd, s)
    first = be.run_long(cached)                                         # misses: the long lane stores the state
    assert be.state_cache.stats()["entries"] == 1
    for again in (be.run_long(cached), be.run_shared(cached)):          # hits in either lane
        for a, b in zip(first, again):
            assert all(torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-5) for x, y in zip(a, b))
    assert be.state_cache.stats()["hits"] >= 2


def test_code_logits_in_fp32():
    """The option-code logits are computed in fp32 from the (bf16) hidden state, not rounded to bf16 steps."""
    be = Stub(tiny())
    be.model = be.model.to(torch.bfloat16)
    h = (torch.randn(3, be.model.config.hidden_size) * 4).to(torch.bfloat16)
    lids = [[10, 11], [12, 13, 14], [10, 11]]
    W, _, _ = be._rows(lids)
    assert W.dtype == torch.float32
    got = be._readout(h, lids)
    full = h.float() @ be.model.lm_head.weight.float().T
    for row, ids in zip(range(3), lids):
        want = torch.softmax(full[row, ids], -1)
        assert torch.allclose(torch.tensor(got[row]), want, atol=1e-6)


def test_many_long_branches_are_split():
    """A request with many long questions over one state is split into packed rows of at most BRANCH_TOKENS tokens
    below the root (pass 2's dense masks grow with the square of that), and every readout still equals a separate
    forward of its own prompt."""
    rnd = random.Random(6)
    be = Stub(tiny())
    be.BRANCH_TOKENS = 40
    s = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    toks = [s + [rnd.randrange(3, 50) for _ in range(15)] + [10, 11] for _ in range(6)]   # 6 x 17 tokens below the root
    units = be._units([(toks, [[10, 11]] * 6)])
    assert len(units) > 1
    assert all(len(pk[0]) - pk[2].index(1) <= 40 for pk, *_ in units if 1 in pk[2])
    check(be, [(toks, [[10, 11]] * 6)])
