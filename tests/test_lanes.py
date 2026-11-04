"""--deterministic --lanes N: N engine calls at once, one request each, every call on a lane of its own; refused where
it cannot keep the answers of one request at a time."""
import asyncio
import sys
import threading
import time
import types

import pytest
import torch

import basal.server as server_module
from basal.engine import GraphBackend
from basal.server import Server, lanes_refused, parser


class Lane:
    """A stub lane: one request per call, slow enough for calls to overlap; records its own overlaps."""
    streams, inflight = True, 2

    def __init__(self, name, live):
        self.name, self.live, self.busy, self.calls = name, live, False, 0

    def run_shared(self, groups, policy=None, done=None):
        assert not self.busy, "two calls on one lane"
        self.busy = True
        with self.live["lock"]:
            self.live["now"] += 1
            self.live["max"] = max(self.live["max"], self.live["now"])
        time.sleep(0.1)
        with self.live["lock"]:
            self.live["now"] -= 1
        self.busy, self.calls = False, self.calls + 1
        out = [[[0.5] * len(ids) for ids in g[1]] for g in groups]
        for k, r in enumerate(out):
            done(k, r)
        return out


def test_lanes_run_requests_at_once_one_per_lane():
    live = {"lock": threading.Lock(), "now": 0, "max": 0}
    lanes = [Lane(i, live) for i in range(3)]
    srv = Server.__new__(Server)
    srv.backend, srv.max_batch, srv.wait, srv.window = lanes[0], 1, 0.0, 0.0
    srv.lanes, srv.free_lanes = lanes, list(lanes)

    async def go():
        srv.queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        futs = [loop.create_future() for _ in range(6)]
        for f in futs:
            await srv.queue.put(([[1, 2]], [[7, 8]], f, None))
        w = loop.create_task(srv.worker())
        res = await asyncio.gather(*futs)
        w.cancel()
        return res
    res = asyncio.run(go())
    assert res == [[[0.5, 0.5]]] * 6
    assert live["max"] == 3 and [x.calls for x in lanes] == [2, 2, 2]  # three at once, never two on one lane
    assert sorted(x.name for x in srv.free_lanes) == [0, 1, 2]          # every lane back in the pool


def args(*extra):
    return parser().parse_args(list(extra))


def test_lanes_refused(monkeypatch):
    monkeypatch.delenv("BASAL_EXPERIMENTAL_LANES", raising=False)
    assert parser().parse_args([]).lanes == 1 and lanes_refused(args(), "graph") is None
    assert "experimental" in lanes_refused(args("--lanes", "4", "--deterministic"), "graph")
    monkeypatch.setenv("BASAL_EXPERIMENTAL_LANES", "1")
    assert lanes_refused(args("--lanes", "4", "--deterministic"), "graph") is None
    assert "at least 1" in lanes_refused(args("--lanes", "0"), "graph")
    assert "needs --deterministic" in lanes_refused(args("--lanes", "2"), "graph")
    assert "CUDA graph engine" in lanes_refused(args("--lanes", "2", "--deterministic"), "eager")
    assert "CUDA graph engine" in lanes_refused(args("--lanes", "2", "--deterministic"), "exit")
    assert "--long-lane" in lanes_refused(args("--lanes", "2", "--deterministic", "--long-lane", "on"), "graph")


def test_lanes_refused_before_a_model_is_read(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "fast", "--lanes", "2"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    monkeypatch.setattr(server_module, "_recurrent", lambda *_, **__: pytest.fail("must reject before reading a model"))
    with pytest.raises(SystemExit, match="--lanes needs --deterministic"):
        server_module.main()


class Engine(GraphBackend):
    """The graph engine's lane state without a model or CUDA graphs."""

    def __init__(self, model=None, dev="cpu"):
        self.model, self.dev, self._wrows, self.graphs = model, torch.device(dev), {}, {}


def test_a_lane_has_its_own_graphs_lock_and_row_cache():
    base = Engine(model=object())
    base._wrows[(1, 2)] = "rows"
    new = base._lane_copy()
    assert new.model is base.model                                      # the weights are shared
    assert new._wrows == {} and new._wrows is not base._wrows           # rows are written on the lane's own stream
    assert new.graphs == {} and new._launch is not base._launch and not new.frozen and new.stream is None
    assert new._eager_lock is base._eager_lock                          # one lock for every lane


def test_a_frozen_lane_keeps_its_eager_forward_per_shape():
    calls = []

    def fn(x):
        calls.append(1)
        return x * 2
    e = Engine()
    e.frozen = True
    x = torch.ones(3)
    g = e._capture(("p", 1, 3), fn, x)
    assert e.graphs[("p", 1, 3)] is g and len(calls) == 1 and torch.equal(g[-1], x * 2)  # kept: the next call finds it
    x.fill_(3)
    g[0].replay()  # (callers fill the static inputs, then replay, as with a graph)
    assert len(calls) == 2 and torch.equal(g[-1], torch.full((3,), 6.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams")
def test_rows_a_lane_reads_are_written_on_its_own_stream():
    """_rows fills its cache on the stream of the call that asks first. With that lane's stream held back, another lane
    (its own stream) must still read the head's rows, not memory the first stream has not written yet."""
    torch.manual_seed(0)
    head = torch.nn.Linear(256, 4096, bias=False).cuda()
    a = Engine(model=types.SimpleNamespace(lm_head=head), dev="cuda")
    b = a._lane_copy()
    sa, sb, lids = torch.cuda.Stream(), torch.cuda.Stream(), [[11, 12, 13], [500, 11]]
    torch.cuda.synchronize()
    with torch.cuda.stream(sa):
        torch.cuda._sleep(int(5e8))  # the first lane's stream is busy (about a quarter of a second)
        a._rows(lids)
    with torch.cuda.stream(sb):
        w, _, _ = b._rows(lids)
        got = w.clone()
    torch.cuda.synchronize()
    assert torch.equal(got, head.weight[[11, 12, 13, 500]].float())  # (the rows of every code id asked, sorted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_lanes_graphs_replayed_at_once_give_their_own_answers():
    """Each lane captures on its own stream, so its graphs keep their own cuBLAS workspace: two lanes' GEMM graphs
    (small M, long K: split-K, which sums partial results in the workspace) replayed at once on two streams, many
    times, each give the output they give alone."""
    torch.manual_seed(0)
    w = torch.randn(4096, 16384, device="cuda", dtype=torch.bfloat16) / 64
    xs = [torch.randn(16, 16384, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    lanes = [Engine(dev="cuda"), None]
    lanes[0].pool = torch.cuda.graph_pool_handle()
    lanes[1] = lanes[0]._lane_copy()
    lanes[1].pool, lanes[1]._capture_stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
    [x @ w.t() for x in xs]  # (sets up cuBLAS before the captures)
    graphs = [e._record(("p", 16, 16384), lambda x: x @ w.t(), (x,)) for e, x in zip(lanes, xs)]
    want = []
    for g in graphs:  # each alone
        g[0].replay()
        torch.cuda.synchronize()
        want.append(g[-1].clone())
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    for _ in range(200):
        for g, s in zip(graphs, streams):
            with torch.cuda.stream(s):
                g[0].replay()
    torch.cuda.synchronize()
    for g, x in zip(graphs, want):
        assert torch.equal(g[-1], x)
