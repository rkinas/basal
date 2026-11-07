"""--long-lane: requests whose longest prompt is longer than --long-tokens go to a lane of their own (HRRN order, one
at a time, one decoder layer at a time), which gives the GPU to the main lane's batches between layers (LaneGate).
The answers are those of the main lane; the engine side (run_long against run_shared) is tested in test_engine16."""
import asyncio
import json
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


# -- --long-lane auto, the default (2.8) ------------------------------------------------------------------------------
def test_the_long_lane_is_on_by_default_where_it_applies():
    """--long-lane auto is the default; it gives a server a long lane wherever --long-lane on is served and answers may
    depend on load, and none elsewhere (where on is refused, or with --deterministic)."""
    from basal.server import long_lane_auto_off, parser
    a = parser().parse_args([])
    assert a.long_lane == "auto"
    assert long_lane_auto_off("graph", a) is None
    for kind in ("eager", "exit", "mps", "mlx", "vllm", "gguf"):
        assert long_lane_auto_off(kind, a)
    assert long_lane_auto_off("graph", a, two_way={"max_prompt": 4096})
    assert long_lane_auto_off("graph", parser().parse_args(["--lanes", "2"]))
    assert long_lane_auto_off("graph", parser().parse_args(["--deterministic"]))
    assert long_lane_auto_off("graph", a, gdn=True)
    assert long_lane_auto_off("graph", parser().parse_args(["--long-lane-gdn", "on"]), gdn=True) is None


LONG = {"state": "Long state, a document. " * 30, "questions": {"q": Q, "r": dict(Q, instructions="And?")}}
SHORT = {"state": "Short.", "questions": {"q": Q}}


class Timed:
    """An engine whose calls take time as a GPU's would: a long prompt 0.4 s in the main lane's call (which nothing
    interrupts), and eight layers of 0.05 s in the long lane's (between them it gives the GPU to the main lane). The
    long lane answers other probabilities (of the prompt with one token more), so an answer shows the lane it came
    from; fail: its calls raise."""
    policies, state_cache = {}, None

    def __init__(self, long_tokens, fail=False):
        self.long_tokens, self.fail, self.calls = long_tokens, fail, {"main": 0, "long": 0}

    def run_shared(self, groups, policy=None):
        self.calls["main"] += 1
        time.sleep(0.4 if any(len(p) > self.long_tokens for g in groups for p in g[0]) else 0.01)
        return [[probs(p, x) for p, x in zip(*g[:2])] for g in groups]

    def run_long(self, groups, policy=None, between=None):
        self.calls["long"] += 1
        if self.fail:
            raise RuntimeError("a kernel the long lane has no build of")
        for i in range(8):
            between(i)
            time.sleep(0.05)
        return [[probs(list(p) + [7], x) for p, x in zip(*g[:2])] for g in groups]


def lane_server(mode, fail=False):
    """basal-serve's queue, worker and long lane over Timed: --long-lane auto, on or off (the long lane as Server sets
    it up for that mode)."""
    from tests.test_http_status import AppServer
    srv = AppServer(Timed(600, fail))  # (SHORT: 307 prompt tokens, LONG: 1,021)
    if mode != "off":
        srv.long_lane(600, 0.0, cuda=False)
        srv.long_auto = mode == "auto"
    return srv


async def serving(srv, fn):
    tasks = [asyncio.get_running_loop().create_task(srv.worker())]
    if srv.long_tokens is not None:
        tasks.append(asyncio.get_running_loop().create_task(srv.long_worker()))
    try:
        return await fn()
    finally:
        for t in tasks:
            t.cancel()


def shorts_behind_a_long(mode):
    """A short request in flight, a long one that comes meanwhile, then five short requests one after another ->
    (the slowest of the five in seconds, the long one's usage, the engine's calls per lane)."""
    srv = lane_server(mode)

    async def go():
        first = asyncio.create_task(srv.serve(SHORT))
        while srv.clients < 1:
            await asyncio.sleep(0.001)
        long = asyncio.create_task(srv.serve(LONG))  # another request is in flight: auto's long lane takes it
        await first
        await asyncio.sleep(0.05)  # the long request in its engine call
        waits = []
        for _ in range(5):
            t = time.perf_counter()
            await srv.serve(SHORT)
            waits.append(time.perf_counter() - t)
        return max(waits), (await long)["usage"]

    worst, usage = asyncio.run(serving(srv, go))
    return worst, usage, srv.backend.calls


def test_short_requests_behind_a_long_one_wait_for_it_without_the_long_lane():
    """The CPU before / after of the default: with --long-lane off (2.7 and before) the long request holds the main lane
    for its whole engine call and the short requests behind it wait 0.3 s or more; with auto, the default, it runs in
    the long lane and they are answered at once."""
    off, usage_off, calls_off = shorts_behind_a_long("off")
    auto, usage_auto, calls_auto = shorts_behind_a_long("auto")
    assert off >= 0.3 and "lane" not in usage_off and calls_off["long"] == 0
    assert auto < 0.15 and usage_auto["lane"] == "long" and calls_auto["long"] == 1


def sequential(srv, bodies):
    async def go():
        outs = []
        for body in bodies:
            out = await srv.serve(body)
            out["usage"].pop("latency_ms")
            outs.append(out)
        return outs
    return asyncio.run(serving(srv, go))


def test_the_default_answers_sequential_requests_as_without_the_long_lane():
    """A sequential client's requests, long and short: under auto, the default, each one is alone on the server, so
    each takes the main lane, and every response is the one --long-lane off gives, byte for byte (no usage "lane"
    either). Timed's long lane answers other probabilities, so a request it served would differ; --long-lane on shows
    that it does."""
    bodies = [SHORT, LONG, SHORT, LONG, dict(LONG, state=LONG["state"] * 2)]
    off = sequential(lane_server("off"), bodies)
    auto_srv = lane_server("auto")
    auto = sequential(auto_srv, bodies)
    assert json.dumps(auto, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert auto_srv.backend.calls == {"main": 5, "long": 0}
    on = sequential(lane_server("on"), bodies)
    assert on[1]["answers"] != off[1]["answers"] and on[1]["usage"]["lane"] == "long"


def test_a_failing_long_lane_of_the_default_leaves_its_requests_to_the_main_lane(capsys):
    """auto: an engine error in the long lane (one it may meet first as the default) fails no request: the long one is
    answered by the main lane, as with off, the error is logged once with its traceback, and later long requests go
    to the main lane at once. With --long-lane on, the request gets the error, as before."""
    srv = lane_server("auto", fail=True)

    async def contended(body):
        first = asyncio.create_task(srv.serve(SHORT))
        while srv.clients < 1:
            await asyncio.sleep(0.001)
        out = await srv.serve(body)
        await first
        return out

    async def go():
        return [await contended(LONG), await contended(LONG)]

    outs = asyncio.run(serving(srv, go))
    off = sequential(lane_server("off"), [LONG])[0]
    assert all(o["answers"] == off["answers"] and "lane" not in o["usage"] for o in outs)
    assert srv.backend.calls["long"] == 1 and srv.long_failed and srv.load() == {"running": 0, "queued": 0}
    err = capsys.readouterr().err
    assert err.count("the long lane failed (RuntimeError") == 1 and "Traceback" in err
    on = lane_server("on", fail=True)

    async def go_on():
        try:
            await on.serve(LONG)
        except RuntimeError as e:
            return str(e)
    assert asyncio.run(serving(on, go_on)) == "a kernel the long lane has no build of"


# -- GET /health and /metrics by lane (2.8) ----------------------------------------------------------------------------
async def until(cond, seconds=5):
    for _ in range(int(seconds * 200)):
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("timed out")


def metric(text, name, lane):
    line = next(x for x in text.splitlines() if x.startswith(f'{name}{{lane="{lane}"}} '))
    return float(line.split()[-1])


def test_health_and_metrics_report_the_long_lane_apart():
    """A server with a long lane: GET /health keeps running / queued over both lanes and adds "long", the requests
    whose work is the long lane's (one in its engine call, one waiting behind it); /metrics gives the same split in
    families of their own (main + long = the totals). A load balancer then sees the main lane free while only long
    states run. Idle again, every count is 0."""
    import httpx

    from basal.server import http_app
    srv = lane_server("on")

    async def go():
        app = http_app(srv, "fast")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
            idle = (await c.get("/health")).json()
            first = asyncio.create_task(srv.serve(LONG))
            await until(lambda: srv.backend.calls["long"] == 1)  # in the long lane's engine call
            second = asyncio.create_task(srv.serve(dict(LONG, state=LONG["state"] + "And more.")))
            await until(lambda: len(srv.long_entries) == 1)  # waiting behind it
            busy = (await c.get("/health")).json()
            text = (await c.get("/metrics")).text
            await first
            await second
            after = (await c.get("/health")).json()
        return idle, busy, text, after

    idle, busy, text, after = asyncio.run(serving(srv, go))
    assert idle == after == {"status": "ok", "running": 0, "queued": 0, "long": {"running": 0, "queued": 0}}
    assert busy == {"status": "ok", "running": 1, "queued": 1, "long": {"running": 1, "queued": 1}}
    assert metric(text, "basal_lane_requests_waiting", "long") == 1 and metric(
        text, "basal_lane_requests_waiting", "main") == 0
    assert metric(text, "basal_lane_requests_running", "long") == 1 and metric(
        text, "basal_lane_requests_running", "main") == 0
    assert "basal_requests_waiting 1" in text.splitlines() and "basal_requests_running 1" in text.splitlines()


def test_without_a_long_lane_health_and_metrics_are_as_before():
    import httpx

    from basal.server import http_app
    srv = lane_server("off")

    async def go():
        app = http_app(srv, "fast")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
            return (await c.get("/health")).json(), (await c.get("/metrics")).text

    health, text = asyncio.run(serving(srv, go))
    assert health == {"status": "ok", "running": 0, "queued": 0}
    assert "basal_lane_" not in text


def test_the_lane_counts_come_back_to_zero_when_a_long_request_is_cancelled():
    """A long request whose client leaves while it waits in the long lane's queue, and one whose client leaves during
    its long-lane call: GET /health's "long" comes back to 0 (no request counted for ever)."""
    srv = lane_server("on")

    async def go():
        first = asyncio.create_task(srv.serve(LONG))
        await until(lambda: srv.backend.calls["long"] == 1)
        second = asyncio.create_task(srv.serve(dict(LONG, state=LONG["state"] + "And more.")))
        await until(lambda: len(srv.long_entries) == 1)
        second.cancel()
        first.cancel()
        await asyncio.wait({first, second})
        await until(lambda: srv.in_flight == 0 and srv.queued == 0)
        return srv.health()[0]

    assert asyncio.run(serving(srv, go)) == {"status": "ok", "running": 0, "queued": 0,
                                             "long": {"running": 0, "queued": 0}}
