"""Server.forward settles every request of an engine call: a backend that returns fewer results than the groups it
was given left the surplus requests waiting for ever (zip dropped them), so their HTTP requests hung."""
import asyncio
from types import SimpleNamespace

from basal.server import Server


def settled(n_results, groups=2):
    async def run():
        srv = Server.__new__(Server)
        srv.waiting, srv.free_lanes = {}, []
        srv.backend = SimpleNamespace(run_shared=lambda groups, pol, **kw: [[[1.0]]] * n_results)
        loop = asyncio.get_running_loop()
        futs = [loop.create_future() for _ in range(groups)]
        batch = [([[1]], [[1]], f, None, None) for f in futs]
        await srv.forward(batch, SimpleNamespace(release=lambda: None))
        await asyncio.wait(futs, timeout=0.2)
        return [f.done() and (f.exception() is None) for f in futs], [f.done() for f in futs]
    return asyncio.run(run())


def test_every_request_gets_its_result_when_the_backend_keeps_the_contract():
    assert settled(2) == ([True, True], [True, True])


def test_a_short_backend_result_fails_every_request_instead_of_leaving_one_waiting():
    ok, done = settled(1)
    assert done == [True, True]  # none left waiting
    assert ok == [False, False]  # the call failed as a whole (a 500 for each), not a wrong answer for one
