"""A request cancelled while its work waits in the queue (a client gone, an MCP call cancelled) costs the engine
nothing: its queue entries are dropped before the engine call, the entries of a request with several of them too once
one is cancelled, and the queued / in-flight counts come back to zero. Work already in the engine finishes."""
import asyncio

from tests.test_http_status import AppServer, Engine

Q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta"}}


class Counting(Engine):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run_shared(self, groups, policy=None):
        self.calls.append(len(groups))
        return super().run_shared(groups, policy)


async def _scenario(soam, questions):
    eng = Counting()
    srv = AppServer(eng)
    srv.soam, srv.max_batch = soam, 1  # one entry per engine call: the queued ones wait behind the first
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        eng.go.clear()
        first = asyncio.create_task(srv.serve({"state": "first", "questions": {"q": Q}}))
        while not eng.entered.is_set():
            await asyncio.sleep(0.01)
        qs = {f"q{i}": Q for i in range(questions)}
        late = asyncio.create_task(srv.serve({"state": "late", "questions": qs}))
        await asyncio.sleep(0.05)                          # queued behind the first call
        late.cancel()
        await asyncio.sleep(0.05)
        eng.go.set()
        await first
        for _ in range(50):
            await asyncio.sleep(0.01)
        return eng.calls, srv.queued, srv.in_flight, srv.queue.qsize()
    finally:
        worker.cancel()


def test_a_cancelled_queued_request_never_reaches_the_engine():
    calls, queued, in_flight, left = asyncio.run(_scenario(soam=True, questions=1))
    assert calls == [1] and queued == 0 and in_flight == 0 and left == 0


def test_every_entry_of_a_cancelled_request_is_dropped():
    calls, queued, in_flight, left = asyncio.run(_scenario(soam=False, questions=3))
    assert calls == [1] and queued == 0 and in_flight == 0 and left == 0


def test_an_entry_that_failed_cancels_the_rest_of_its_request():
    async def go():
        eng = Counting()
        srv = AppServer(eng)
        srv.soam, srv.max_batch = False, 1
        fails = iter([True, False, False])

        def run_shared(groups, policy=None):
            eng.calls.append(len(groups))
            if next(fails, False):
                raise RuntimeError("engine failed")
            return [[[0.5, 0.5] for _ in g[1]] for g in groups]
        eng.run_shared = run_shared
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            try:
                await srv.serve({"state": "s", "questions": {f"q{i}": Q for i in range(3)}})
            except RuntimeError:
                pass
            for _ in range(50):
                await asyncio.sleep(0.01)
            return eng.calls, srv.queued, srv.in_flight
        finally:
            worker.cancel()
    calls, queued, in_flight = asyncio.run(go())
    assert calls == [1] and queued == 0 and in_flight == 0      # the request had failed: its other entries do not run
