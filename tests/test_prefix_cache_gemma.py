"""--prefix-cache on with a Gemma 4 model whose key / value buffer _fit_grid shrank (a large model on a smaller GPU): the
pass-2 graphs over cached template prefixes and the short requests batched over them stay within the buffer, and every
readout equals the prompt's own forward."""
import torch

from basal.engine import GraphBackend, StateCache
from basal.gemma4_graph import Gemma4Graph
from test_gemma4_graph import separate, tiny


class Replay:
    def __init__(self, fn, static):
        self.fn, self.static = fn, static
        self.out = fn(*static)

    def replay(self):
        self.out.copy_(self.fn(*self.static))


class Shrunk(GraphBackend):
    """A Gemma 4 GraphBackend with --prefix-cache on and a key / value buffer of 64 tokens (TOKEN_BUDGET 64, as _fit_grid
    leaves it), graphs replayed eagerly on the CPU."""
    LENS, PLAIN_LENS, BATCHES, TOKEN_BUDGET, SUFFIX = [16, 32], [16, 32], [1, 2, 4, 8], 64, 4
    TREE_ROWS, SUFFIX_LENS = [1, 2], [8]
    PREFIX_ROWS, PREFIX_SFX, PREFIX_ROOT = [1, 2, 4, 8], [8, 16], 32

    def __init__(self, model):
        self.model, self.dev, self._wrows, self.graphs = model, torch.device("cpu"), {}, {}
        self.gemma = Gemma4Graph(model)
        self.softcap = self.gemma.softcap
        self.fwd, self.fwd_sfx = self.gemma.forward, self.gemma.forward_suffix
        self.tok = type("Tok", (), {"pad_token_id": 0})()
        self.shared, self.compact, self.prefix = True, False, True
        self.state_cache = StateCache(1 << 24)
        self._alloc_kv()

    def encode(self, prompts):
        return prompts

    def _capture(self, key, fn, *static):
        r = Replay(fn, static)
        self.graphs[key] = (r, *static, r.out)
        return self.graphs[key]


def test_prefix_graphs_fit_the_buffer():
    be = Shrunk(tiny("e4b", torch.float32))
    shapes = be._prefix_graph_shapes()
    assert shapes and all(b * L <= be._kv_tokens() for b, S, L in shapes)
    for b, S, L in shapes:  # each can be captured over the buffer
        be._graph_sfx(b, S, L)


def test_short_requests_over_a_cached_template_prefix():
    m = tiny("e4b", torch.float32)
    be = Shrunk(m)
    tpl = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 20 tokens: L = 32
    reqs = [([tpl + [30 + r, 40 + q, 50] for q in range(2)], [[7, 8], [7, 9]], len(tpl)) for r in range(8)]
    be.run_shared(reqs[:1])  # computes the template prefix and caches it
    got = be.run_shared(reqs)  # 8 short requests over the cached prefix: 8 rows x 32 > 64 buffer tokens
    for (toks, lids, _), res in zip(reqs, got):
        for r, e in zip(res, separate(m, toks, lids)):
            assert torch.allclose(torch.tensor(r), e, atol=1e-5)
