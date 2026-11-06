"""The state cache's disk tier (--state-cache-dir): every state cached with its token ids is also a .safetensors file
under the directory, found again (exactly and as the longest prefix) by a later cache over the same directory, in this
process or another one, within a disk budget; files of another model signature, or (without --state-cache-import
any-engine) of another engine, are ignored; a broken file is a miss, never an error. Engines: the graph engine's
Gemma 4 forward (CPU) and the MLX engine (Apple Silicon) answer over a state read back from disk as over the one they
computed; the server's flags, refusals and metrics; tools/state_cache/compare.py."""
import errno
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from basal.engine import DiskTier, StateCache, cache_signature, kv_dequant, kv_quant, mlx_codec, torch_codec

ROOT = Path(__file__).resolve().parents[1]


def model_dir(path):
    d = path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text('{"model_type": "llama"}')
    (d / "model-00001.safetensors").write_bytes(b"x" * 10)
    (d / "model-00002.safetensors").write_bytes(b"y" * 20)
    return d


def kv3(n, seed=0, heads=2, dim=128):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(heads, n, dim, generator=g).bfloat16(), torch.randn(heads, n, dim, generator=g).bfloat16())
            for _ in range(3)]


def equal(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q))


class Maker:
    def __init__(self, path):
        self.dir = path / "sc"
        self.sig = cache_signature(model_dir(path), "tiny", None, None, None, "bf16")
        self.made = []

    def tier(self, engine="graph", accept=None, budget=1 << 30, signature=None, **kw):
        t = DiskTier(self.dir, budget, signature or self.sig, engine, *torch_codec("cpu"), accept_engines=accept, **kw)
        self.made.append(t)
        return t

    def cache(self, bits=16, budget=1 << 30, **kw):
        return StateCache(budget, salt=b"s", bits=bits, disk=self.tier(**kw))


@pytest.fixture
def mk(tmp_path):
    m = Maker(tmp_path)
    yield m
    for t in m.made:
        t.close()


def files(tier):
    return sorted(p for p in tier.root.glob("*/*") if p.suffix == ".safetensors")


def stems(tier):
    """The keys (hex) of a tier's files."""
    return [p.name.split(".")[0] for p in files(tier)]


def stored(mk, n=300, seed=0, start=0, **kw):
    """A cache that has written one state of n tokens to disk -> (cache, key, kv, tokens)."""
    c = mk.cache(**kw)
    toks = list(range(start, start + n))
    kv = kv3(n, seed)
    k = c.key(toks)
    c.put(k, kv, tokens=toks)
    c.disk.flush()
    return c, k, kv, toks


# -- T1: the tier ------------------------------------------------------------------------------------------------------

def test_entry_survives_a_new_cache(mk):
    c, k, kv, _ = stored(mk)
    f = files(c.disk)
    assert len(f) == 1 and f[0].parent.name == k.hex()[0] and f[0].name == f"{k.hex()}.graph.safetensors"
    assert c.disk.root.parent == mk.dir and len(c.disk.root.name) == 16
    c2 = mk.cache()
    assert equal(c2.get(k), kv)
    st = c2.stats()
    assert (st["hits"], st["misses"], st["disk_loads"], st["disk_entries"]) == (1, 0, 1, 1)


def test_file_is_valid_safetensors(mk):
    from safetensors import safe_open
    c, _, _, _ = stored(mk)
    with safe_open(str(files(c.disk)[0]), "pt") as f:
        assert sorted(f.keys()) == ["k.0", "k.1", "k.2", "v.0", "v.1", "v.2"]
        t = f.get_tensor("k.1")
        assert t.dtype == torch.bfloat16 and list(t.shape) == [2, 300, 128]
        md = f.metadata()
    assert md["basal_state_cache"] == "1" and md["signature"] == mk.sig and md["engine"] == "graph"
    assert json.loads(md["tokens"]) == list(range(300)) and md["layers"] == "3"
    assert len(json.loads(md["blocks"])) == 1


def test_prefix_found_on_disk(mk):
    _, k, kv, _ = stored(mk, 600)
    c = mk.cache()
    key, got, m, n = c.longest(list(range(700)))
    assert (key, m, n) == (k, 600, 600) and equal(got, kv)
    assert c.stats()["disk_prefix_loads"] == 1
    assert c.longest(list(range(700)))[2] == 600
    assert c.stats()["disk_prefix_loads"] == 1  # (from memory now)


def test_edited_prefix_found_on_disk(mk):
    stored(mk, 600)
    assert mk.cache().longest(list(range(400)) + [9] * 300)[2] == 400


def test_short_state_has_no_prefix_index(mk):
    _, k, kv, _ = stored(mk, 200)
    c = mk.cache()
    assert c.longest(list(range(200))) is None
    assert equal(c.get(k), kv)


def test_other_signature_is_ignored(mk):
    c, _, _, _ = stored(mk)
    f = files(c.disk)[0]
    other = mk.tier(signature=mk.sig.replace('"tiny"', '"other"'))
    copy = other.root / f.parent.name / f.name  # (the same file under the other signature's directory)
    copy.write_bytes(f.read_bytes())
    other.scan()
    assert other.stats()["disk_entries"] == 0 and copy.exists() and f.exists()


def test_other_engine_needs_any_engine(mk):
    stored(mk)
    assert mk.tier(engine="mlx").stats()["disk_entries"] == 0
    assert mk.tier(engine="mlx", accept="any").stats()["disk_entries"] == 1


def test_tokens_required_to_persist(mk):
    c = mk.cache()
    c.put(b"k" * 16, kv3(300))
    c.disk.flush()
    assert files(c.disk) == [] and c.stats()["disk_writes"] == 0


def test_trimmed_entry_is_not_persisted(mk):
    c = mk.cache()
    toks = list(range(300))
    c.put(c.key(toks), [(k[:, -100:], v[:, -100:]) for k, v in kv3(300)], tokens=toks)
    c.disk.flush()
    assert files(c.disk) == [] and c.stats()["entries"] == 1


def test_corrupt_file_is_skipped_at_scan(mk):
    c, _, _, _ = stored(mk)
    f = files(c.disk)[0]
    os.truncate(f, 100)
    garbage = f.parent / f"{'ab' * 16}.safetensors"
    garbage.write_bytes((1 << 40).to_bytes(8, "little") + b"{}")  # a header length no file has
    st = mk.tier().stats()
    assert (st["disk_entries"], st["disk_errors"]) == (0, 2) and f.exists() and garbage.exists()


def test_truncated_data_is_a_miss_and_removed(mk):
    c, k, _, _ = stored(mk)
    f = files(c.disk)[0]
    n = int.from_bytes(f.read_bytes()[:8], "little")
    os.truncate(f, 8 + n + 10)
    c2 = mk.cache()
    assert c2.stats()["disk_entries"] == 1  # (the header is whole)
    assert c2.get(k) is None and not f.exists()
    st = c2.stats()
    assert (st["disk_errors"], st["disk_entries"], st["misses"]) == (1, 0, 1)


@pytest.mark.parametrize("wrong", ["layers", "tokens"])
def test_wrong_tensor_count_is_a_miss(mk, wrong):
    """A file whose tensors are not what its metadata says is a miss, and removed; one whose token ids are not is
    skipped at start-up (counted, left in place)."""
    t = mk.tier()
    toks = list(range(300))
    k = StateCache(1, salt=b"s").key(toks)
    pack, _ = torch_codec("cpu")
    meta = t._meta(3, toks, [])
    if wrong == "tokens":
        meta["tokens"] = json.dumps(toks[:-1])
    DiskTier._write(t._path(k), pack(kv3(300)[:2] if wrong == "layers" else kv3(300)), meta)
    c = mk.cache()
    assert c.get(k) is None and t._path(k).exists() == (wrong == "tokens") and c.stats()["disk_errors"] == 1


def test_resaving_a_known_state_writes_nothing(mk):
    """A state on disk stored again (a restart computed it anyway, or it extends one) is not rewritten; the state it
    replaces still leaves the disk."""
    _, k, kv, toks = stored(mk)
    c = mk.cache()
    old = list(range(7, 307))
    c.put(c.key(old), kv3(300, 7), tokens=old)
    c.disk.flush()
    c.put(k, kv, tokens=toks, replaces=c.key(old))
    c.disk.flush()
    assert c.stats()["disk_writes"] == 1 and stems(c.disk) == [k.hex()]


def test_file_evicted_by_another_process(mk):
    _, k, _, _ = stored(mk)
    t = mk.tier()
    os.unlink(files(t)[0])
    assert t.load(k) is None and k not in t.entries and t.stats()["disk_errors"] == 1


def test_disk_budget_evicts_oldest(mk):
    c, k1, _, _ = stored(mk)
    c.disk.budget = int(1.5 * c.stats()["disk_bytes"])
    toks = list(range(1, 301))
    k2 = c.key(toks)
    c.put(k2, kv3(300, 1), tokens=toks)
    c.disk.flush()
    f = files(c.disk)
    st = c.stats()
    assert [p.name.split(".")[0] for p in f] == [k2.hex()] and st["disk_evictions"] == 1
    assert st["disk_bytes"] == f[0].stat().st_size


def test_replaced_entry_leaves_the_disk(mk):
    c, k1, _, _ = stored(mk)
    toks = list(range(400))
    k2 = c.key(toks)
    c.put(k2, kv3(400), tokens=toks, replaces=k1)
    c.disk.flush()
    assert stems(c.disk) == [k2.hex()] and c.stats()["disk_entries"] == 1


def test_dropped_write_still_deletes_the_replaced(mk, monkeypatch):
    monkeypatch.setattr(DiskTier, "QUEUE", 1)
    c = mk.cache()
    t = c.disk
    entered, release, write = threading.Event(), threading.Event(), DiskTier._write

    def slow(path, items, meta):
        if not entered.is_set():
            entered.set()
            release.wait(10)
        return write(path, items, meta)
    t._write = slow
    x, a, b = [list(range(s, s + 300)) for s in (5000, 0, 1)]
    c.put(c.key(x), kv3(300, 2), tokens=x)  # the writer holds it
    entered.wait(10)
    c.put(c.key(a), kv3(300), tokens=a)  # queued (the queue is full now)
    c.put(c.key(b), kv3(300, 1), tokens=b, replaces=c.key(a))  # dropped
    assert c.stats()["disk_write_drops"] == 1
    release.set()
    t.flush()
    assert stems(t) == [c.key(x).hex()] and not t._path(c.key(a)).exists()
    assert c.stats()["disk_entries"] == 1


def test_delete_cancels_a_state_queued_twice(mk):
    """A state stored twice before its file is written is queued once, so deleting it (the entry replacing it was
    dropped) leaves no file behind."""
    c = mk.cache()
    t = c.disk
    entered, release, write = threading.Event(), threading.Event(), DiskTier._write

    def slow(path, items, meta):
        if not entered.is_set():
            entered.set()
            release.wait(10)
        return write(path, items, meta)
    t._write = slow
    x, a = list(range(5000, 5300)), list(range(300))
    c.put(c.key(x), kv3(300, 2), tokens=x)  # the writer holds it
    entered.wait(10)
    t.save(c.key(a), kv3(300), a, c._blocks(a))
    t.save(c.key(a), kv3(300), a, c._blocks(a))
    assert t.queue.qsize() == 1
    t.delete(c.key(a))
    release.set()
    t.flush()
    assert stems(t) == [c.key(x).hex()] and not t.has(c.key(a))


def test_two_engines_share_a_directory(mk):
    """The graph engine's and MLX's states of one key are two files: neither overwrites the other, a same-engine tier
    reads only its own, any-engine prefers its own, and a file whose metadata names another engine than its name is
    never served."""
    toks = list(range(300))
    ref = StateCache(1, salt=b"s")
    k, blocks = ref.key(toks), ref._blocks(toks)
    zeros, ones = [(torch.zeros(2, 300, 8).bfloat16(),) * 2], [(torch.ones(2, 300, 8).bfloat16(),) * 2]
    g, m = mk.tier("graph"), mk.tier("mlx")
    g.save(k, zeros, toks, blocks)
    m.save(k, ones, toks, blocks)
    g.flush()
    m.flush()
    assert sorted(p.name.split(".")[1] for p in files(g)) == ["graph", "mlx"]
    assert (g.load(k)[0][0][0] == 0).all() and (m.load(k)[0][0][0] == 1).all()
    assert (mk.tier("graph", accept="any").load(k)[0][0][0] == 0).all()
    assert (mk.tier("mlx", accept="any").load(k)[0][0][0] == 1).all()
    g._path(k).write_bytes(m._path(k).read_bytes())  # an MLX file under the graph engine's name: indexed, then read
    assert g.load(k) is None and g.stats()["disk_errors"] == 1 and not g._path(k).exists()
    g._path(k).write_bytes(m._path(k).read_bytes())  # (again, before a new tier indexes the directory)
    assert mk.tier("graph").stats()["disk_entries"] == 0


def test_prefix_on_disk_takes_the_longest_match(mk):
    """Of the states on disk sharing a request's deepest block, the one sharing the most tokens is read, not the most
    recently written."""
    c = mk.cache()
    a, b = list(range(300)), list(range(256)) + [999] * 4
    c.put(c.key(a), kv3(300), tokens=a)
    c.put(c.key(b), kv3(260, 1), tokens=b)
    c.disk.flush()
    key, _, m, n = mk.cache().longest(list(range(400)))
    assert (key, m, n) == (c.key(a), 300, 300)


def test_prefix_larger_than_memory_is_still_found(mk):
    """A state read from disk that the memory budget cannot hold is still a prefix hit (and an exact one)."""
    _, k, kv, toks = stored(mk)
    c = StateCache(10, salt=b"s", disk=mk.tier())
    key, got, m, n = c.longest(toks + [7] * 100)
    assert (key, m, n) == (k, 300, 300) and equal(got, kv)
    st = c.stats()
    assert (st["entries"], st["prefix_hits"], st["disk_prefix_loads"]) == (0, 1, 1) and equal(c.get(k), kv)


def test_memory_eviction_keeps_the_file(mk):
    one = sum(k.nbytes + v.nbytes for k, v in kv3(300))
    c, k1, kv, _ = stored(mk, budget=one)
    toks = list(range(1, 301))
    c.put(c.key(toks), kv3(300, 1), tokens=toks)
    c.disk.flush()
    st = c.stats()
    assert (st["entries"], st["disk_entries"], st["evictions"]) == (1, 2, 1)
    assert equal(c.get(k1), kv) and c.stats()["disk_loads"] == 1


def test_has_sees_the_disk(mk):
    _, k, _, _ = stored(mk)
    c = mk.cache()
    assert c.has(k) and not c.has(b"x" * 16) and c.stats()["disk_loads"] == 0


def test_bits8_round_trips_through_disk(mk):
    from safetensors import safe_open
    c, k, kv, _ = stored(mk, bits=8)
    with safe_open(str(files(c.disk)[0]), "pt") as f:
        assert f.get_tensor("v.0").dtype == torch.bfloat16
    got = mk.cache(bits=8).get(k)
    want = [(kv_dequant(kv_quant(a, 8, True), 8, True), kv_dequant(kv_quant(b, 8, False), 8, False)) for a, b in kv]
    assert equal(got, want)


def test_stale_tmp_files_are_removed(mk):
    t = mk.tier()
    old, young = t.root / "a" / "x.safetensors.tmp-abc", t.root / "a" / "y.safetensors.tmp-def"
    other = t.root / "a" / "notes.txt"
    for p in (old, young, other):
        p.write_bytes(b"x")
    os.utime(old, (time.time() - 3600,) * 2)
    t.scan()
    assert not old.exists() and young.exists() and other.exists()
    assert t.stats()["disk_entries"] == 0 and t.stats()["disk_errors"] == 0


def test_startup_scan_orders_lru_by_mtime(mk):
    c = mk.cache()
    keys = []
    for s in (1000, 2000, 3000):
        toks = list(range(s, s + 300))
        keys.append(c.key(toks))
        c.put(keys[-1], kv3(300, s), tokens=toks)
    c.disk.flush()
    size = files(c.disk)[0].stat().st_size
    for k, hours in zip(keys, (3, 2, 1)):  # the first written is the oldest
        os.utime(c.disk._path(k), (time.time() - 3600 * hours,) * 2)
    c2 = StateCache(1 << 30, salt=b"s", disk=mk.tier(budget=int(2.5 * size)))
    assert [k for k in keys if c2.disk._path(k).exists()] == keys[1:]  # over the budget at start-up: the 3-h-old out
    toks = list(range(4000, 4300))
    c2.put(c2.key(toks), kv3(300, 4), tokens=toks)
    c2.disk.flush()
    assert [k for k in keys if c2.disk._path(k).exists()] == keys[2:] and c2.stats()["disk_evictions"] == 2


def test_two_tiers_share_a_directory(mk):
    a, _, _, _ = stored(mk)
    b = mk.cache()
    assert b.stats()["disk_entries"] == 1
    toks = list(range(1, 301))
    b.put(b.key(toks), kv3(300, 1), tokens=toks)
    b.disk.flush()
    a.disk.scan()
    assert a.stats()["disk_entries"] == 2
    a.disk.scan()
    assert a.stats()["disk_entries"] == 2


def test_close_drains_the_queue(mk):
    c = mk.cache()
    for s in range(5):
        toks = list(range(s, s + 300))
        c.put(c.key(toks), kv3(300, s), tokens=toks)
    c.close()
    assert len(files(c.disk)) == 5 and not c.disk.thread.is_alive()
    toks = list(range(9, 309))
    c.put(c.key(toks), kv3(300), tokens=toks)
    assert c.stats()["disk_write_drops"] == 1 and len(files(c.disk)) == 5


def test_codec_round_trip_torch_mlx(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    t = torch.randn(2, 50, 8).bfloat16()
    tpack, tunpack = torch_codec("cpu")
    mpack, munpack = mlx_codec(mx)
    DiskTier._write(tmp_path / "t.safetensors", tpack([(t, t * 2)]), {"layers": "1"})
    (k, v), = munpack(tmp_path / "t.safetensors")[0]
    assert k.shape == (1, 2, 50, 8) and k.dtype == mx.bfloat16
    assert (np.array(mx.view(k[0], mx.int16)) == t.view(torch.int16).numpy()).all()
    assert (np.array(mx.view(v[0], mx.int16)) == (t * 2).view(torch.int16).numpy()).all()
    a = mx.random.normal((1, 2, 50, 8)).astype(mx.bfloat16)
    DiskTier._write(tmp_path / "m.safetensors", mpack([(a, a)]), {"layers": "1"})
    (k, _), = tunpack(tmp_path / "m.safetensors")[0]
    assert list(k.shape) == [2, 50, 8] and k.dtype == torch.bfloat16
    assert (k.view(torch.int16).numpy() == np.array(mx.view(a[0], mx.int16))).all()
    DiskTier._write(tmp_path / "w.safetensors", mpack([(a, a)]), {"layers": "2"})
    with pytest.raises(ValueError, match="2 tensors for 2 layers"):
        munpack(tmp_path / "w.safetensors")


def test_signature_changes_with_weights_and_adapter(tmp_path):
    md = model_dir(tmp_path)
    base = cache_signature(md, "tiny", None, None, None, "bf16")
    assert cache_signature(md, "tiny", None, None, None, "bf16") == base
    with open(md / "model-00001.safetensors", "ab") as f:
        f.write(b"z")
    weights = cache_signature(md, "tiny", None, None, None, "bf16")
    sigs = [base, weights, cache_signature(md, "tiny", None, SimpleNamespace(name="a"), None, "bf16"),
            cache_signature(md, "tiny", None, None, "q8", "bf16"), cache_signature(md, "tiny", None, None, None, "f16")]
    assert len(set(sigs)) == 5 and json.loads(base)["cache"] == 1


# -- T2: the graph engine's Gemma 4 forward (CPU) ----------------------------------------------------------------------

def gemma_stub(mk, accept=None, margin=2):
    from tests.test_gemma4_graph import Stub, tiny
    be = Stub(tiny("e4b", torch.float32))
    be.gemma.MARGIN = margin
    be.state_cache = StateCache(1 << 30, salt=b"s", disk=mk.tier(after_load=be._trim, accept=accept))
    return be


STATE12 = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
LIDS2 = [[7, 8], [7, 8]]


def test_gemma4_graph_import_is_trimmed(mk):
    from tests.test_gemma4_graph import Stub, separate, tiny
    src = Stub(tiny("e4b", torch.float32))  # MARGIN 256: every layer holds the 12 tokens
    src.state_cache = StateCache(1 << 30, salt=b"s")
    src.run_shared([([STATE12 + [20, 21, 30 + o, 31 - o] for o in range(2)], LIDS2, 12)])
    key = src.state_cache.key(STATE12)
    full = src.state_cache.entries[key][0]
    assert [k.shape[1] for k, _ in full] == [12] * len(full)
    mlx = mk.tier(engine="mlx")
    mlx.save(key, full, STATE12, src.state_cache._blocks(STATE12))
    mlx.flush()
    assert gemma_stub(mk).state_cache.stats()["disk_entries"] == 0  # (same-engine)
    be = gemma_stub(mk, accept="any")
    assert [k.shape[1] for k, _ in be.state_cache.get(key)] == be.gemma.kv_lens(12) == [7, 7, 12, 7]
    toks = [STATE12 + [22, 23, 24, 30 + o] for o in range(2)]
    for r, e in zip(be.run_shared([(toks, LIDS2, 12)])[0], separate(be.model, toks, LIDS2)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-4)


def test_gemma4_graph_trimmed_entry_not_exported(mk):
    be = gemma_stub(mk)
    be.run_shared([([STATE12 + [20, 21, 30 + o, 31 - o] for o in range(2)], LIDS2, 12)])
    be.state_cache.disk.flush()
    st = be.state_cache.stats()
    assert (st["entries"], st["disk_writes"]) == (1, 0) and files(be.state_cache.disk) == []
    be.run_shared([([STATE12[:6] + [20, 21, 30 + o, 31 - o] for o in range(2)], LIDS2, 6)])
    be.state_cache.disk.flush()
    assert len(files(be.state_cache.disk)) == 1


def test_gemma4_graph_restart_is_bit_identical(mk):
    groups = [([STATE12[:7] + [20, 21, 30 + o, 31 - o] for o in range(2)], LIDS2, 7)]
    be = gemma_stub(mk)
    cold = be.run_shared(groups)
    be.state_cache.disk.flush()
    warm_be = gemma_stub(mk)
    warm = warm_be.run_shared(groups)
    assert warm == cold
    st = warm_be.state_cache.stats()
    assert (st["hits"], st["misses"], st["disk_loads"]) == (1, 0, 1)


# -- T3: the MLX engine (Apple Silicon) --------------------------------------------------------------------------------

def mlx_cached(path, sc, accept=None, forward="mlx-lm"):
    import mlx.core as mx
    from tests.test_mlx_state_cache import cached_stub
    be = cached_stub(path, forward=forward)
    be.state_cache = StateCache(1 << 30, salt=b"s", disk=DiskTier(sc, 1 << 30, "sig", "mlx", *mlx_codec(mx),
                                                                  accept_engines=accept))
    return be


@pytest.mark.parametrize("forward", ["mlx-lm", "basal"])
def test_mlx_restart_is_bit_identical(tmp_path, forward):
    pytest.importorskip("mlx.core")
    from tests.test_mlx import LIDS, STATE, TOKS, reference, tiny_llama
    from tests.test_mlx_state_cache import close
    m = tiny_llama(tmp_path)
    be = mlx_cached(tmp_path, tmp_path / "sc", forward=forward)
    cold = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    be.state_cache.close()
    be2 = mlx_cached(tmp_path, tmp_path / "sc", forward=forward)
    warm = be2.run_shared([(TOKS, LIDS, len(STATE))])[0]
    assert warm == cold and close(reference(m, TOKS, LIDS), warm)
    st = be2.state_cache.stats()
    assert (st["hits"], st["misses"], st["disk_loads"]) == (1, 0, 1)
    be2.state_cache.close()


def test_mlx_prefix_from_disk_recomputes_the_root(tmp_path):
    pytest.importorskip("mlx.core")
    import random
    from tests.test_mlx import tiny_llama
    from tests.test_mlx_state_cache import cached_stub, count_tokens
    tiny_llama(tmp_path)
    rnd = random.Random(0)
    s1 = [rnd.randrange(1, 64) for _ in range(300)]
    s2 = s1 + [rnd.randrange(1, 64) for _ in range(40)]
    q, lids = [[5, 6, 10, 11], [5, 6, 11, 10]], [[10, 11], [10, 11]]
    be = mlx_cached(tmp_path, tmp_path / "sc")
    be.run_shared([([s1 + x for x in q], lids, len(s1))])
    be.state_cache.close()
    be2 = mlx_cached(tmp_path, tmp_path / "sc")
    seen = count_tokens(be2)
    got = be2.run_shared([([s2 + x for x in q], lids, len(s2))])[0]
    assert be2.state_cache.stats()["disk_prefix_loads"] == 1
    assert seen[0] == len(s2)  # the promoted entry has no row count (_refs): the root is computed, not continued
    assert got == cached_stub(tmp_path).run_shared([([s2 + x for x in q], lids, len(s2))])[0]
    be2.state_cache.close()


def test_mlx_imports_graph_entry_with_any_engine(tmp_path):
    pytest.importorskip("mlx.core")
    from tests.test_mlx import LIDS, STATE, TOKS, reference, tiny_llama
    from tests.test_mlx_state_cache import close
    m = tiny_llama(tmp_path)
    with torch.no_grad():
        pkv = m(input_ids=torch.tensor([STATE]), use_cache=True).past_key_values
    kv = [(layer.keys[0], layer.values[0]) for layer in pkv.layers]
    writer = DiskTier(tmp_path / "sc", 1 << 30, "sig", "graph", *torch_codec("cpu"))
    ref = StateCache(1, salt=b"s")
    writer.save(ref.key(STATE), kv, STATE, ref._blocks(STATE))
    writer.close()
    be = mlx_cached(tmp_path, tmp_path / "sc")
    assert be.state_cache.stats()["disk_entries"] == 0
    be = mlx_cached(tmp_path, tmp_path / "sc", accept="any")
    warm = be.run_shared([(TOKS, LIDS, len(STATE))])[0]
    assert be.state_cache.stats()["disk_loads"] == 1
    assert close(reference(m, TOKS, LIDS), warm, tol=2e-3)
    be.state_cache.close()


def test_mlx_single_prompt_cut_hits_disk(tmp_path):
    pytest.importorskip("mlx.core")
    from tests.test_mlx import LIDS, STATE, TOKS, reference, tiny_llama
    from tests.test_mlx_state_cache import close
    m = tiny_llama(tmp_path)
    one = [TOKS[4]]
    be = mlx_cached(tmp_path, tmp_path / "sc")
    cold = be.run_shared([(one, [LIDS[4]], len(STATE))])[0]
    be.state_cache.close()
    be2 = mlx_cached(tmp_path, tmp_path / "sc")
    warm = be2.run_shared([(one, [LIDS[4]], len(STATE))])[0]
    st = be2.state_cache.stats()
    assert (st["hits"], st["disk_loads"]) == (1, 1) and warm == cold and close(reference(m, one, [LIDS[4]]), warm)
    be2.state_cache.close()


# -- T4: the server ----------------------------------------------------------------------------------------------------

def test_dir_flag_needs_a_cache():
    from basal.server import Server, parser
    with pytest.raises(SystemExit, match="--state-cache-dir needs"):
        Server(parser().parse_args(["--mode", "mlx", "--state-cache-dir", "/tmp/x"]))


def test_dir_flag_refused_outside_graph_and_mlx():
    from basal.server import Server, parser
    with pytest.raises(SystemExit, match="works with the CUDA graph engine"):
        Server(parser().parse_args(["--mode", "fast-exit", "--state-cache-gb", "1", "--state-cache-dir", "/tmp/x"]))


def test_dir_flag_refused_for_qwen35():
    from basal import gdn_graph
    assert "--state-cache-dir" in gdn_graph.refused(disk=True, state_cache_gb=1, device=False)
    assert gdn_graph.refused(state_cache_gb=1, device=False) is None


def test_defaults():
    from basal.server import parser
    a = parser().parse_args([])
    assert (a.state_cache_dir, a.state_cache_disk_gb, a.state_cache_import) == (None, 20, "same-engine")


def test_metrics_report_the_disk(mk):
    import asyncio

    from starlette.testclient import TestClient

    from basal import server as S

    def body(cache):
        class Srv:
            name, layout, long_tokens = "basal-test", "canonical", None
            backend = type("B", (), {"state_cache": cache, "policies": {}})()

            def load(self):
                return {"running": 0, "queued": 0}

            async def worker(self):
                await asyncio.sleep(3600)
        with TestClient(S.http_app(Srv(), "fast")) as c:
            return c.get("/metrics").text
    c, _, _, _ = stored(mk)
    on = body(c)
    assert "basal_state_cache_disk_loads_total 0" in on and "basal_state_cache_disk_entries 1" in on
    assert "basal_state_cache_disk_writes_total 1" in on and c.disk.closed  # (closed at shutdown: the queue written)
    off = body(StateCache(1 << 20))
    assert "basal_state_cache_hits_total" in off and "disk" not in off


# -- T5: faults and concurrency ----------------------------------------------------------------------------------------

def test_writer_survives_enospc(mk):
    c = mk.cache()
    t, write, calls = c.disk, DiskTier._write, []

    def full(path, items, meta):
        calls.append(path)
        if len(calls) == 1:
            Path(path).write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "no space")
        return write(path, items, meta)
    t._write = full
    a, b = list(range(300)), list(range(1, 301))
    c.put(c.key(a), kv3(300), tokens=a)
    c.put(c.key(b), kv3(300, 1), tokens=b)
    t.flush()
    assert c.stats()["disk_errors"] == 1 and stems(t) == [c.key(b).hex()]
    assert not list(t.root.glob("*/*.tmp-*"))


def small(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, n, 8, generator=g).bfloat16(), torch.randn(1, n, 8, generator=g).bfloat16())]


def test_concurrent_puts_and_gets(mk, monkeypatch):
    monkeypatch.setattr(DiskTier, "QUEUE", 1000)
    c = mk.cache()
    errors = []

    def work(t):
        try:
            for i in range(50):
                toks = [t * 1000 + i] + list(range(299))
                c.put(c.key(toks), small(300, t * 1000 + i), tokens=toks)
                c.get(c.key(toks))
                c.longest(toks + [1])
        except Exception as e:  # noqa: BLE001  (reported below)
            errors.append(e)
    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    c.disk.flush()
    st = c.stats()
    assert errors == [] and st["disk_entries"] == 400 and st["disk_write_drops"] == 0
    assert st["disk_bytes"] == sum(p.stat().st_size for p in files(c.disk)) and len(c.disk.index) == 400


def test_two_caches_two_threads_one_dir(mk, monkeypatch):
    monkeypatch.setattr(DiskTier, "QUEUE", 1000)
    caches = [mk.cache(), mk.cache()]
    keys = [caches[0].key([i] + list(range(299))) for i in range(50)]

    def work(c):
        for i, k in enumerate(keys):
            c.put(k, small(300, i), tokens=[i] + list(range(299)))
    threads = [threading.Thread(target=work, args=(c,)) for c in caches]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for c in caches:
        c.disk.flush()
    assert len(files(caches[0].disk)) == 50
    for c in caches:
        assert all(c.disk.load(k) is not None for k in keys) and c.stats()["disk_errors"] == 0


# -- tools/state_cache/compare.py ----------------------------------------------------------------------------------------

def test_compare_tool(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("compare", ROOT / "tools/state_cache/compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    def write(name, rows):
        p = tmp_path / name
        p.write_text("".join(json.dumps({"id": i, "answers": {"q": {"probabilities": {"0": a, "1": 1 - a}}},
                                         "latency_ms": ms}) + "\n" for i, (a, ms) in enumerate(rows)))
        return str(p)
    ref = write("ref.jsonl", [(0.7, 100), (0.505, 100), (0.9, 300)])
    cand = write("cand.jsonl", [(0.7, 10), (0.495, 20), (0.1, 30)])
    args = ["--reference", ref, "--candidate", cand]
    assert compare.main(args + ["--require", "agreement>=1"]) == 1
    out = capsys.readouterr().out.splitlines()
    r = json.loads(out[0])
    assert (r["agreement"], r["near_ties"], r["disagreements"], r["identical"]) == (1 / 3, 1, 2, False)
    assert r["agreement_at_margin_0_2"] == 0.5 and out[1] == "FAIL agreement>=1 (0.3333333333333333)"
    assert compare.main(["--reference", ref, "--candidate", ref, "--speed-off", ref, "--speed-on", cand,
                         "--require", "identical=true", "speedup>=5", "--out", str(tmp_path / "row.json"),
                         "--label", "x"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["identical"] is True and json.loads(out[0])["speedup"] == 5.0
    assert out[1].startswith("| x | 100 ms | 20 ms | 5.00x | 3/3 |")
    m = tmp_path / "m.txt"
    m.write_text("basal_state_cache_hits_total 3\nbasal_state_cache_disk_loads_total 2\nbasal_state_cache_bytes 9\n")
    assert compare.main(args[:2] + ["--candidate", ref, "--metrics", str(m), "--require", "disk_loads=2"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["metrics"][0]["hits"] == 3
    write("short.jsonl", [(0.7, 1)])
    assert compare.main(["--reference", ref, "--candidate", str(tmp_path / "short.jsonl")]) == 2
    capsys.readouterr()
    failed = tmp_path / "failed.jsonl"  # (every request of the run failed: nothing to compare)
    failed.write_text(json.dumps({"id": 0, "answers": None, "error": "HTTP 500"}) + "\n")
    assert compare.main(["--reference", str(failed), "--candidate", str(failed), "--require", "identical=true"]) == 2
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["identical"] is False and out[1] == "FAIL readouts>0 (0)"
