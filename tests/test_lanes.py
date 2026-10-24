"""--deterministic --lanes N: N engine calls at once, one request each, every call on a lane of its own; refused where
it cannot keep the answers of one request at a time."""
import asyncio
import sys
import threading
import time

import pytest

import basal.server as server_module
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
        t = time.perf_counter()
        w = loop.create_task(srv.worker())
        res = await asyncio.gather(*futs)
        w.cancel()
        return res, time.perf_counter() - t
    res, dt = asyncio.run(go())
    assert res == [[[0.5, 0.5]]] * 6
    assert live["max"] == 3 and [x.calls for x in lanes] == [2, 2, 2]  # three at once, never two on one lane
    assert dt < 0.45                                                    # one at a time: 0.6 s
    assert sorted(x.name for x in srv.free_lanes) == [0, 1, 2]          # every lane back in the pool


def args(*extra):
    return parser().parse_args(list(extra))


def test_lanes_refused():
    assert parser().parse_args([]).lanes == 1 and lanes_refused(args(), "graph") is None
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
