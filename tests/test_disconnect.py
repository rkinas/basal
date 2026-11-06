"""A request whose HTTP client disconnects (a client timeout, a retry) is cancelled (#129): its entries still in the
queue never reach the engine, it answers 499, and the requests around it are answered as before. Through the app of
basal-serve (http_app) with its batching worker and a stub engine that holds the first batch on the GPU."""
import asyncio
import json

import httpx

from basal.server import CLIENT_GONE, http_app, until_disconnect
from tests.test_http_status import BODY, AppServer, Engine


class CountingEngine(Engine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run_shared(self, groups, policy=None):
        self.calls += 1
        return super().run_shared(groups, policy)


async def raw_post(app, path, body, gone):
    """POST through the ASGI app, the client disconnecting once `gone` is set -> (status, body)."""
    messages = [{"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        await gone.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "http_version": "1.1", "scheme": "http",
             "server": ("basal", 80), "client": ("test", 1), "root_path": ""}
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


def run_abandoned(path, body):
    """One request held on the engine, a second one queued behind it whose client disconnects, then the first one
    released -> (status of the abandoned one, its body, status of the held one, engine calls)."""
    engine = CountingEngine()
    srv = AppServer(engine)
    engine.go.clear()

    async def go():
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            app = http_app(srv, "fast")
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://basal") as c:
                held = asyncio.create_task(c.post("/v1/systemone", json=BODY))
                await asyncio.get_running_loop().run_in_executor(None, engine.entered.wait, 5)
                gone = asyncio.Event()
                abandoned = asyncio.create_task(raw_post(app, path, body, gone))
                for _ in range(200):  # until the second request waits in the queue
                    if srv.queued:
                        break
                    await asyncio.sleep(0.01)
                assert srv.queued == 1
                gone.set()
                status, out = await asyncio.wait_for(abandoned, 5)
                engine.go.set()
                return status, out, (await held).status_code
        finally:
            worker.cancel()

    status, out, held = asyncio.run(go())
    return status, out, held, engine.calls


def test_a_request_abandoned_in_the_queue_never_reaches_the_engine():
    status, out, held, calls = run_abandoned("/v1/systemone", BODY)
    assert (out, status) == CLIENT_GONE
    assert held == 200
    assert calls == 1  # the held request only


def test_decisions_endpoint_drops_an_abandoned_request_too():
    body = {"model": "basal", "input": "Invoice 1240 PLN.",
            "questions": [{"type": "choice", "name": "q", "instructions": "Which?",
                           "choices": [{"value": "a", "description": "Alpha"}, {"value": "b", "description": "Beta"}]}]}
    status, out, held, calls = run_abandoned("/v1/decisions", body)
    assert (out, status) == CLIENT_GONE
    assert held == 200
    assert calls == 1


class Request:
    def __init__(self, gone_after=None):
        self.polls, self.gone_after = 0, gone_after

    async def is_disconnected(self):
        self.polls += 1
        return self.gone_after is not None and self.polls >= self.gone_after


def test_until_disconnect_returns_the_answer_and_raises_its_errors():
    async def answer():
        await asyncio.sleep(0.02)
        return {"ok": 1}

    async def broken():
        raise ValueError("bad")

    assert asyncio.run(until_disconnect(Request(), answer(), poll=0.005)) == {"ok": 1}
    try:
        asyncio.run(until_disconnect(Request(), broken(), poll=0.005))
    except ValueError as e:
        assert str(e) == "bad"
    else:
        raise AssertionError("the error of the serve is raised")


def test_until_disconnect_cancels_the_serve_when_the_client_leaves():
    state = {}

    async def serve():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    assert asyncio.run(until_disconnect(Request(gone_after=2), serve(), poll=0.005)) is None
    assert state == {"cancelled": True}
