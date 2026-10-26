"""--graph-memory-gb: the CUDA graph engine sizes its graph shapes (the grid, the pass-2 graphs captured, the longest
state run as one forward) for a set amount of free GPU memory instead of the measured one, so a GPU with more free
memory runs the shapes of a smaller one. The default (no value) measures, as before. torch.cuda's memory calls are
stubbed: the shapes and answers themselves are checked on an H100."""
from types import SimpleNamespace

import pytest
import torch

from basal.engine import GraphBackend, graph_memory_bytes
from basal.server import Server, parser

GiB = 2 ** 30


def backend(monkeypatch, free, graph_memory=None):
    """A GraphBackend shell on a 'cuda' device with `free` bytes measured (Gemma 4 sized: 31B's 0.9 MB per token)."""
    be = GraphBackend.__new__(GraphBackend)
    be.dev, be.shared, be.graph_memory = torch.device("cuda"), True, graph_memory
    be.gemma = SimpleNamespace(lm=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1,
                                                                                              dtype=torch.bfloat16))))
    be.kv_shapes = [(16, 256)] * 30 + [(4, 512)] * 10  # 0.9 MB per token in bf16
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (free[0], 80 * GiB))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return be


def grid(be):
    return be.TOKEN_BUDGET, be.PLAIN_LENS, be.LENS, be.COMPACT_LENS


def test_graph_memory_bytes():
    assert graph_memory_bytes(None) is None
    assert graph_memory_bytes(12) == 12 * GiB and graph_memory_bytes(0.5) == GiB // 2
    for bad in (0, -1):
        with pytest.raises(ValueError, match="above 0"):
            graph_memory_bytes(bad)


def test_grid_follows_the_set_memory_not_the_measured(monkeypatch):
    """An 80 GB card pinned to 14 GiB runs the grid a card with 14 GiB free measures; unpinned it runs a larger one."""
    small = backend(monkeypatch, [14 * GiB])
    small._fit_grid()
    pinned = backend(monkeypatch, [46 * GiB], graph_memory=14 * GiB)
    pinned._pin_memory()
    pinned._fit_grid()
    big = backend(monkeypatch, [46 * GiB])
    big._fit_grid()
    assert grid(pinned) == grid(small)
    assert grid(big) != grid(small) and big.TOKEN_BUDGET > small.TOKEN_BUDGET


def test_pinned_free_memory_is_measured_less_the_offset(monkeypatch, capsys):
    free = [46 * GiB]
    be = backend(monkeypatch, free, graph_memory=14 * GiB)
    be._pin_memory()
    assert be._memory_offset == 32 * GiB and be._free_memory() == 14 * GiB
    assert "sized for 14 GiB" in capsys.readouterr().out
    free[0] = 40 * GiB  # the engine took 6 GiB since: the shapes see 8
    assert be._free_memory() == 8 * GiB
    be._capture_cost = 0
    assert be._room()
    free[0] = 32 * GiB + be.PASS2_RESERVE - 1  # less than the pass-2 reserve left of the 14 GiB
    assert not be._room()


def test_one_forward_limit_as_on_the_smaller_card(monkeypatch):
    """After the same use of memory, the pinned big card and the small card compute the same one-forward limit."""
    cfg = SimpleNamespace(intermediate_size=21504, hidden_size=5376)
    body = SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)))
    small = backend(monkeypatch, [14 * GiB])
    small.gemma, small.model = None, SimpleNamespace(config=cfg, model=body)
    small_n = small._one_forward_tokens()
    free = [46 * GiB]
    pinned = backend(monkeypatch, free, graph_memory=20 * GiB)
    pinned._pin_memory()
    free[0] = 40 * GiB  # 6 GiB used since the weights: 14 GiB left of the 20
    pinned.gemma, pinned.model = None, SimpleNamespace(config=cfg, model=body)
    assert pinned._one_forward_tokens() == small_n


def test_default_measures(monkeypatch):
    be = backend(monkeypatch, [46 * GiB])
    be._pin_memory()
    assert be._memory_offset == 0 and be._free_memory() == 46 * GiB


def test_more_than_free_is_refused(monkeypatch):
    be = backend(monkeypatch, [13 * GiB], graph_memory=14 * GiB)
    with pytest.raises(ValueError, match="only 13.0 GiB of GPU memory are free after the weights"):
        be._pin_memory()


def test_shapes_report(monkeypatch):
    be = backend(monkeypatch, [14 * GiB])
    be.graphs, be.compact, be.one_forward = {("p", 1, 128): 0, ("p", 2, 128): 0, ("x", 1, 64, 128): 0}, False, 24696
    s = be.shapes()
    assert s["graphs"] == {"p": 2, "x": 1} and s["one_forward"] == 24696 and len(s["hash"]) == 12
    other = backend(monkeypatch, [14 * GiB])
    other.graphs, other.compact, other.one_forward = {("p", 1, 128): 0, ("x", 1, 64, 128): 0}, False, 24696
    assert other.shapes()["hash"] != s["hash"]


def test_server_flag():
    assert parser().parse_args([]).graph_memory_gb is None
    assert parser().parse_args(["--graph-memory-gb", "12.5"]).graph_memory_gb == 12.5
    with pytest.raises(SystemExit, match="needs the CUDA graph engine"):
        Server(parser().parse_args(["--mode", "eager", "--graph-memory-gb", "12"]))
    with pytest.raises(SystemExit, match="above 0"):
        Server(parser().parse_args(["--mode", "fast", "--graph-memory-gb", "0"]))
