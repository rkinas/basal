"""An explanation and its served comparison must share the model without overlapping the next request."""
import asyncio
import sys
import threading
from types import SimpleNamespace

import httpx
import uvicorn

from basal import viz


def test_concurrent_http_requests_do_not_overlap_model_work(monkeypatch):
    comparing = threading.Event()
    comparison_started, release = asyncio.Event(), asyncio.Event()
    apps = []

    async def decide(body):
        comparing.set()
        comparison_started.set()
        await release.wait()
        comparing.clear()
        return {"answers": {}}

    class Explainer:
        name, dev, info = "stub", "cpu", {}
        srv = SimpleNamespace(decide=decide)

        def __init__(self, _args):
            pass

        def explain(self, body):
            assert not comparing.is_set(), "explanation overlapped a served forward on the shared model"
            return {"questions": []}

    monkeypatch.setattr(viz, "Explainer", Explainer)
    monkeypatch.setattr(viz, "examples", lambda: [])
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.append(app))
    monkeypatch.setattr(sys, "argv", ["basal-viz", "--device", "cpu"])
    viz.main()

    async def requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[0]), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/explain", json={}))
            await asyncio.wait_for(comparison_started.wait(), 5)
            second = asyncio.create_task(client.post("/api/explain", json={}))
            try:
                await asyncio.sleep(0.05)  # let the second request reach the model while the comparison is paused
            finally:
                release.set()
            responses = await asyncio.wait_for(asyncio.gather(first, second), 5)
            assert [r.status_code for r in responses] == [200, 200]

    asyncio.run(requests())
