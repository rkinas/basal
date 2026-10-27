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


def test_longest_chooses_best_cached_prefix():
    cache = StateCache(budget_bytes=1024)
    prefix = list(range(cache.BLOCK))
    a = prefix + [10]
    b = prefix + [20]
    query = a + [99]
    kv = [(torch.zeros(1), torch.zeros(1))]
    cache.put(cache.key(a), kv, tokens=a)
    cache.put(cache.key(b), kv, tokens=b)
    match = cache.longest(query)
    assert match is not None
    assert match[0] == cache.key(a)
    assert match[2] == len(a)


def test_eviction_preserves_shared_prefix():
    # Two cache entries fit, but three do not.
    cache = StateCache(budget_bytes=16)
    prefix = list(range(cache.BLOCK))
    a = prefix + [10]
    b = prefix + [20]
    c = list(range(1000, 1000 + cache.BLOCK)) + [30]
    kv = [(torch.zeros(1), torch.zeros(1))]
    key_a = cache.key(a)
    key_b = cache.key(b)
    cache.put(key_a, kv, tokens=a)
    cache.put(key_b, kv, tokens=b)
    # Refresh A so B becomes the LRU entry.
    assert cache.get(key_a) is not None
    # Adding C must evict B, not A.
    cache.put(cache.key(c), kv, tokens=c)
    assert cache.has(key_a)
    assert not cache.has(key_b)
    # A is still cached and should be discoverable.
    match = cache.longest(a + [99])
    assert match is not None
    assert match[0] == key_a
    assert match[2] == len(a)


def test_shared_prefix_ties_and_replacement():
    cache = StateCache(1024)
    cache.BLOCK = 4
    prefix = [1, 2, 3, 4]
    a, b = prefix + [10, 1], prefix + [20, 1]
    kv = [(torch.zeros(1), torch.zeros(1))]
    for tokens in (a, b):
        cache.put(cache.key(tokens), kv, tokens=tokens)
    cache.get(cache.key(a))
    assert cache.longest(prefix + [99])[0] == cache.key(a)  # insertion order, independent of LRU
    extended = a + [30]
    cache.put(cache.key(extended), kv, tokens=extended, replaces=cache.key(a))
    assert not cache.has(cache.key(a))
    assert cache.longest(prefix + [99])[0] == cache.key(b)
    assert cache.longest(extended + [31])[2] == len(extended)


def test_best_shared_prefix_continues_the_model_from_the_right_token():
    rnd = random.Random(3)
    be = Stub(tiny())
    be.state_cache = StateCache(1 << 20)
    be.state_cache.BLOCK = 4
    prefix = [1] + [rnd.randrange(3, 50) for _ in range(11)]
    a, b = prefix + [10], prefix + [20]
    for state in (a, b, a + [30]):
        check(be, requests(rnd, state, 1))
    assert not be.state_cache.has(be.state_cache.key(a))  # the whole A was reused, so its extension replaces it
    assert be.state_cache.has(be.state_cache.key(b))
    assert be.state_cache.stats()["entries"] == 2


def test_prefix_index_matches_full_scan_after_updates():
    """Independent oracle: compare every live state's tokens, including edits, short queries and LRU churn."""
    from collections import OrderedDict

    for block in (1, 4, 256):
        rnd = random.Random(42)
        cache = StateCache(5 * 8)
        cache.BLOCK = block
        live = OrderedDict()
        kv = [(torch.zeros(1), torch.zeros(1))]
        base = [rnd.randrange(8) for _ in range(3 * block + 3)]
        for _ in range(300):
            replaces = None
            if live and rnd.random() < 0.5:
                key = rnd.choice(list(live))
                own = live[key]
                if own is not None and rnd.random() < 0.5:
                    tokens, replaces = own + [rnd.randrange(8)], key
                else:
                    tokens = base[:rnd.randrange(len(base) + 1)] + [rnd.randrange(8)]
            else:
                tokens = base[:rnd.randrange(len(base) + 1)] + [rnd.randrange(8)]
            key = cache.key(tokens)
            stored = tokens if rnd.random() < 0.9 else None
            cache.put(key, kv, tokens=stored, replaces=replaces)
            live.pop(replaces, None)
            if key not in live:
                live[key] = stored
            while len(live) > 5:
                live.popitem(last=False)
            touch = rnd.choice(list(live))
            assert cache.get(touch) is kv
            live.move_to_end(touch)

            query = list(rnd.choice(list(live.values())) or base)
            if rnd.random() < 0.5:
                query = query[:rnd.randrange(len(query) + 1)]
            query += [rnd.randrange(8) for _ in range(rnd.randrange(block + 2))]
            lengths = {}
            for k, own in live.items():
                if own is not None:
                    lengths[k] = next((i for i, (a, b) in enumerate(zip(own, query)) if a != b),
                                      min(len(own), len(query)))
            longest = max(lengths.values(), default=0)
            match = cache.longest(query)
            if longest < block:
                assert match is None
            else:
                assert match is not None and match[2] == lengths[match[0]] == longest
                assert match[1] is kv and match[3] == len(live[match[0]])
                live.move_to_end(match[0])
            assert list(cache.entries) == list(live)
            assert cache.stats()["used_bytes"] == 8 * len(live)
        for i in range(5):
            cache.put(cache.key([-i - 1]), kv)  # evict all indexed states with entries without token ids
        assert not cache.index


def test_gemma_cache_chooses_a_usable_prefix(monkeypatch):
    from tests.test_gemma4_graph import Stub as GemmaStub, tiny as gemma_tiny, separate as gemma_separate

    be = GemmaStub(gemma_tiny("e4b", torch.float32))
    query = [1] + [3 + i % 40 for i in range(599)]
    short = query[:299]
    short_kv, _ = be._continue(short)
    groups = [([query + [10, 11]], [[7, 8]], len(query))]
    reference = gemma_separate(be.model, groups[0][0], groups[0][1])[0]
    forward, calls = be._continue, []

    def observed(tokens, prefix=None, m=0):
        calls.append(m)
        return forward(tokens, prefix, m)

    monkeypatch.setattr(be, "_continue", observed)
    for raw_match in (300, 550):  # same bucket, then an unusable match in a deeper bucket
        long = query[:raw_match] + [61] + [62] * 299
        long_kv, _ = forward(long)
        cache = be.state_cache = StateCache(1 << 24)
        cache.put(cache.key(long), long_kv, tokens=long)
        cache.put(cache.key(short), short_kv, tokens=short)
        assert cache.longest(query)[2] == raw_match
        assert be._usable(long_kv, raw_match, len(long)) == 0
        before = cache.stats()["prefix_hits"]
        lru = list(cache.entries)
        assert cache.longest(query, lambda *_: False) is None
        assert cache.stats()["prefix_hits"] == before and list(cache.entries) == lru
        calls.clear()
        got = torch.tensor(be.run_shared(groups)[0][0])
        assert calls == [len(short)]
        assert torch.allclose(got, reference, atol=1e-6)


def test_longest_gives_an_8bit_entry_in_its_dtype_to_usable_and_the_caller():
    """--state-cache-bits 8 with the usable() check: the check reads the keys' shape, and the caller continues from the
    keys / values, so both get the entry dequantised (a packed entry is a tuple of its parts)."""
    torch.manual_seed(0)
    c = StateCache(1 << 30, bits=8)
    toks = list(range(600))
    kv = [(torch.randn(2, 600, 16), torch.randn(2, 600, 16))]
    c.put(c.key(toks), kv, tokens=toks)
    seen = []
    found = c.longest(toks + [7] * 10, usable=lambda kv, m, n: seen.append(tuple(kv[0][0].shape)) or m)
    assert found is not None and found[2] == 600 and found[3] == 600 and seen == [(2, 600, 16)]
    k = found[1][0][0]
    assert k.dtype == kv[0][0].dtype and (k - kv[0][0]).abs().max() < 0.1
