"""--long-lane: requests whose longest prompt is longer than --long-tokens go to a lane of their own (HRRN order, one
at a time, one decoder layer at a time), which gives the GPU to the main lane's batches between layers (LaneGate).
The answers are those of the main lane; the engine side (run_long against run_shared) is tested in test_engine16."""
import asyncio
import threading
import time

from basal.server import LaneGate
from tests.test_server import FakeServer

Q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta", "c": "Gamma"}}


class LaneServer(FakeServer):
    """FakeServer with the long lane on: the fake engine's run_long walks three "layers" through the gate."""

    def __init__(self, pos_probs, tokens=500, **kw):
        super().__init__(pos_probs, **kw)
        self.long_lane(tokens, 0.0, cuda=False)
        self.gate.main_add = lambda n: None  # FakeServer's main worker answers without engine calls (gate tested below)
        self.long_calls, self.long_groups = [], []
        srv = self

        class Backend:
            policies, state_cache = {}, None

            def run_long(self, groups, policy=None, between=None):
                for i in range(3):
                    between(i)
                srv.long_calls.append(len(groups[0][0]))
                srv.long_groups.append(len(groups))
                return [[srv.probs_for(p, x) for p, x in zip(*g[:2])] for g in groups]
        self.backend = Backend()

    async def _run(self, body):
        task = asyncio.get_running_loop().create_task(self.long_worker())
        try:
            return await super()._run(body)
        finally:
            task.cancel()


def probs(prompt, ids):
    w = [(sum(prompt) + 3 * k) % 5 + 1 for k in range(len(ids))]  # depends on the prompt (its question and order)
    return [x / sum(w) for x in w]


def test_long_requests_take_the_long_lane_with_the_same_answers():
    short = {"state": "Short state.", "questions": {"q": Q}}
    long = {"state": "Long state. " * 40, "questions": {"q": Q, "r": dict(Q, instructions="And?")}}
    on, off = LaneServer(probs), FakeServer(probs)
    for body in (short, long):
        a, b = asyncio.run(on._run(body)), asyncio.run(off._run(body))
        assert a["answers"] == b["answers"]
        assert a["usage"]["longest_prompt_tokens"] == b["usage"]["longest_prompt_tokens"]
        assert a["usage"]["lane"] == ("long" if body is long else "main") and "lane" not in b["usage"]
    assert on.long_calls == [4]  # the long request as one entry: two questions in two option orders


def test_a_cached_state_stays_in_the_main_lane():
    srv = LaneServer(probs)

    class Cache:
        def key(self, tokens):
            return tuple(tokens)

        def has(self, key):
            return True
    srv.backend.state_cache = Cache()
    srv.state_cut = lambda q, first=None: 30
    body = {"state": "Long state. " * 40, "questions": {"q": Q}}
    assert asyncio.run(srv._run(body))["usage"]["lane"] == "main" and srv.long_calls == []


def test_hrrn_prefers_shorter_long_requests_until_one_has_waited():
    srv = LaneServer(probs)
    now = time.perf_counter()
    srv.long_rate = 1e-3                                   # 1 ms per token
    fresh_big, fresh_small = (None,) * 5 + (now, 10_000), (None,) * 5 + (now, 2_000)
    old_big = (None,) * 5 + (now - 30, 10_000)             # waited 30 s: ratio 4 against the small one's ~1
    ratio = lambda e: ((now - e[5]) / (e[6] * srv.long_rate) + 1, -e[6])  # noqa: E731  (as long_worker)
    assert max([fresh_big, fresh_small], key=ratio) is fresh_small
    assert max([fresh_big, fresh_small, old_big], key=ratio) is old_big


def test_gate_yields_to_the_main_lane_between_layers():
    """With main-lane work pending and the slice used up, the long lane stops before its next layer until the main
    lane finishes an engine call; without main-lane work it never waits."""
    gate = LaneGate(0.0, cuda=False)
    gate.start()
    t = time.perf_counter()
    gate.between(0)                                        # nothing pending: no wait
    assert gate.yields == 0 and time.perf_counter() - t < 0.05
    gate.main_add(2)
    done = []
    th = threading.Thread(target=lambda: (gate.between(1), done.append(time.perf_counter())))
    th.start()
    time.sleep(0.1)
    assert not done                                        # waiting for the main lane
    t_main = time.perf_counter()
    gate.main_done(2)                                      # the main lane's call ends
    th.join(1)
    assert done and done[0] >= t_main and gate.yields == 1
    slow = LaneGate(10.0, cuda=False)                      # a long slice: no yield within it
    slow.start()
    slow.main_add(1)
    slow.between(0)
    assert slow.yields == 0


def test_gate_share_keeps_the_long_lane_running():
    """Below its share of the recent busy time the long lane does not yield, even with main-lane work waiting."""
    gate = LaneGate(0.0, cuda=False, share=0.5)
    gate.start()
    gate.main_add(1)
    gate.busy["main"] = 10.0             # the main lane had the GPU recently: the long lane is far below its share
    gate.between(0)
    assert gate.yields == 0
    gate.busy.update(main=0.0, long=10.0)  # the long lane had it all: now it yields (main finishes its call)
    done = []
    th = threading.Thread(target=lambda: (gate.between(1), done.append(1)))
    th.start()
    time.sleep(0.05)
    gate.main_done(1, 0.01)
    th.join(1)
    assert done and gate.yields == 1


def test_several_long_requests_per_lane_call():
    srv = LaneServer(probs)
    srv.LONG_BATCH = 4
    bodies = [{"state": f"Long state {i}. " * 40, "questions": {"q": Q}} for i in range(3)]

    async def run_all():
        task = asyncio.get_running_loop().create_task(srv.long_worker())
        srv.queue = asyncio.Queue()
        try:
            return await asyncio.gather(*(srv.decide(b) for b in bodies))
        finally:
            task.cancel()
    outs = asyncio.run(run_all())
    assert all(o["usage"]["lane"] == "long" for o in outs)
    assert srv.long_groups == [3]                                   # the three waiting requests in one call
