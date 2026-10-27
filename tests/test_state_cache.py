"""The state cache of the graph engine (--state-cache-gb), on CPU: a request's prompts are cut after their state, the
root (template + state) of a second request with the same state comes from the cache and only the questions run,
and every readout still equals a separate forward of its own prompt (fp32). LRU eviction within the byte budget, and
the server's state boundary."""
import random

import torch

from basal.engine import GraphBackend, StateCache
from tests.test_engine import tiny
from tests.test_engine16 import Replay, Stub, check
from tests.test_server import FakeServer


def requests(rnd, state, n=3):
    """Groups about one state: one question in one order, two orders, and three questions (SOAM), each with the cut
    after the state."""
    def q(k):
        return [rnd.randrange(3, 50) for _ in range(k)]
    a, b, c = q(3), q(2), q(4)
    cut = len(state)
    return [([state + a + [10, 11]], [[10, 11]], cut),
            ([state + b + [10, 11], state + b + [11, 10]], [[10, 11], [10, 11]], cut),
            ([state + a + [12, 13, 14], state + b + [10, 11], state + c + [11, 10]],
             [[12, 13, 14], [10, 11], [10, 11]], cut)][:n]


def test_pack_ends_the_root_at_the_cut():
    ids, pos, seg, last, par = GraphBackend._pack([[1, 2, 3, 4, 5]], cut=3)
    assert seg == [0, 0, 0, 1, 1] and par == [-1, 0] and last == [4]          # a single prompt becomes two blocks
    ids, pos, seg, last, par = GraphBackend._pack([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], cut=2)
    assert seg[:2] == [0, 0] and par[1] == 0 and len(par) == 4               # root, the shared [3, 4], two leaves
    assert GraphBackend._pack([[1, 2, 3]], cut=3)[2] == [0, 0, 0]             # no token after the cut: no cut


def test_cached_states_give_the_same_answers():
    rnd = random.Random(1)
    be = Stub(tiny())
    be.state_cache = StateCache(1e9)
    states = [[1] + [rnd.randrange(3, 50) for _ in range(n)] for n in (12, 30, 47)]  # the last is longer than 48
    first = [g for s in states for g in requests(rnd, s)]
    check(be, first)
    st = be.state_cache.stats()
    assert st["entries"] == 3 and st["hits"] == 0
    seen = len(be.replays)
    again = [g for s in states for g in requests(rnd, s)]          # the same states, other questions
    check(be, again)
    st = be.state_cache.stats()
    assert st["hits"] == 9 and st["entries"] == 3
    # the cached units skip pass 1: every forward of the second call is a pass 2 over the cached state
    assert all(kind == "tree" for kind, _, _ in be.replays[seen:])


def test_lru_budget():
    c = StateCache(budget_bytes=3 * 2 * 4 * 10)                  # three entries of 2 x 10 float32
    kv = lambda: [(torch.zeros(10), torch.zeros(10))]
    for k in range(4):
        c.put(c.key([k]), kv())
    assert c.stats()["entries"] == 3 and c.stats()["evictions"] == 1 and c.get(c.key([0])) is None
    assert c.get(c.key([3])) is not None
    assert StateCache(1e6, salt=b"tenant-a").key([1, 2]) != StateCache(1e6, salt=b"tenant-b").key([1, 2])


def test_server_finds_the_state_boundary():
    srv = FakeServer([0.5, 0.5])
    from basal.server import to_items
    q = to_items("Customer: the parcel arrived damaged.", {"q": {"type": "noul", "instructions": "Damaged?"}})[0]
    cut = srv.state_cut(q)
    from basal.prompt import render
    text = render(srv.tok, q["state"], q["question"], q["options"], q["lang"], srv.fmt)
    assert "".join(map(chr, srv.tok(text).input_ids[:cut])).endswith("the parcel arrived damaged.")
    q = to_items("You", {"q": {"type": "noul", "instructions": "Damaged?"}})[0]  # also a word of the system prompt
    text = render(srv.tok, q["state"], q["question"], q["options"], q["lang"], srv.fmt)
    assert "".join(map(chr, srv.tok(text).input_ids[:srv.state_cut(q)])).endswith("State:\nYou")


def test_cached_states_through_pass2_graphs():
    """With the pass-2 graphs captured, a cached state is copied where pass 1 would have written it and the branch
    rows replay a pass-2 graph."""
    class Warm(Stub):
        SUFFIX_LENS = [8, 16]
    rnd = random.Random(2)
    be = Warm(tiny())
    be.warm()
    used = []
    for key, g in be.graphs.items():
        if key[0] == "x":
            g[0].replay = (lambda r, k: lambda: (used.append(k), Replay.replay(r)))(g[0], key)
    be.state_cache = StateCache(1e9)
    states = [[1] + [rnd.randrange(3, 50) for _ in range(n)] for n in (10, 20)]
    check(be, [g for s in states for g in requests(rnd, s)])
    n = len(used)
    check(be, [g for s in states for g in requests(rnd, s)])
    assert be.state_cache.stats()["hits"] == 6 and len(used) > n


def test_longest_prefix_extends_and_diverges():
    """longest(): a state that extends a cached one shares all of it, an edited one up to the edit (also inside a
    block); fewer than one common block is a miss; an entry that replaces the one it extends drops it from the
    index; eviction drops an entry's blocks."""
    import torch
    from basal.engine import StateCache
    B = StateCache.BLOCK
    kv = [(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2))]
    c = StateCache(1 << 20)
    base = list(range(3 * B + 10))
    c.put(c.key(base), kv, tokens=base)
    key, _, m, n = c.longest(base + [7, 8, 9])
    assert key == c.key(base) and m == n == len(base)
    edited = base[: 2 * B + 5] + [-1] + base[2 * B + 6:]
    assert c.longest(edited)[2] == 2 * B + 5
    assert c.longest(base[: B - 1] + [1]) is None
    longer = base + [7, 8, 9]
    c.put(c.key(longer), kv, tokens=longer, replaces=c.key(base))
    assert not c.has(c.key(base)) and c.longest(base)[0] == c.key(longer)
    small = StateCache(2 * sum(k.numel() * k.element_size() * 2 for k, _ in kv))
    for i in range(3):
        t = [i] + base
        small.put(small.key(t), kv, tokens=t)
    assert small.stats()["evictions"] == 1 and small.longest([0] + base) is None


def test_growing_and_edited_states_reuse_the_cached_prefix():
    """Incremental states (Llama): a state that extends a cached one runs only its new tokens over the cached keys /
    values (in chunks: CHUNK), replacing the old entry; an edited one reuses the common prefix only. Every readout
    equals a separate forward of its prompt (fp32)."""
    rnd = random.Random(2)
    be = Stub(tiny())
    be.state_cache = StateCache(1e9)
    be.state_cache.BLOCK, be.CHUNK = 4, 7
    s1 = [1] + [rnd.randrange(3, 50) for _ in range(40)]
    check(be, requests(rnd, s1, 1))
    s2 = s1 + [rnd.randrange(3, 50) for _ in range(9)]                  # the trace grows
    check(be, requests(rnd, s2, 2))
    st = be.state_cache.stats()
    assert st["prefix_hits"] >= 1 and st["entries"] == 1               # s2 replaced s1
    s3 = s2[:22] + [rnd.randrange(3, 50) for _ in range(15)]            # edited after token 22
    check(be, requests(rnd, s3, 1))
    assert be.state_cache.stats()["prefix_hits"] >= 2 and be.state_cache.stats()["entries"] == 2
