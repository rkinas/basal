"""A request whose HTTP client disconnects (a client timeout, a retry) is cancelled (#129): its entries still in the
queue never reach the engine, it answers 499, and the requests around it are answered as before. Through the app of
basal-serve (http_app) with its batching worker and a stub engine that holds the first batch on the GPU."""
import asyncio
import json
import sys

import httpx
import pytest
from starlette.requests import Request

from basal import server
from tests.test_http_status import BODY, AppServer, Engine

GONE = (499, {"error": "client disconnected"})  # the answer of a request whose client left


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
            app = server.http_app(srv, "fast")
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
    assert (status, out) == GONE
    assert held == 200
    assert calls == 1  # the held request only


def test_decisions_endpoint_drops_an_abandoned_request_too():
    body = {"model": "basal", "input": "Invoice 1240 PLN.",
            "questions": [{"type": "choice", "name": "q", "instructions": "Which?",
                           "choices": [{"value": "a", "description": "Alpha"}, {"value": "b", "description": "Beta"}]}]}
    status, out, held, calls = run_abandoned("/v1/decisions", body)
    assert (status, out) == GONE
    assert held == 200
    assert calls == 1


def client(leave_after=None):
    """The Request of a handler whose body has been read: receive() waits as uvicorn's does, and says
    http.disconnect `leave_after` seconds after its first call (never with None). .receives counts its calls."""
    left = asyncio.Event()

    async def receive():
        request.receives += 1
        if leave_after is not None and request.receives == 1:
            asyncio.get_running_loop().call_later(leave_after, left.set)
        await left.wait()
        return {"type": "http.disconnect"}

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    request.receives = 0
    return request


def test_until_disconnect_returns_the_answer_and_raises_its_errors():
    async def answer():
        await asyncio.sleep(0.02)
        return {"ok": 1}

    async def broken():
        raise ValueError("bad")

    assert asyncio.run(server.until_disconnect(client(), answer())) == {"ok": 1}
    try:
        asyncio.run(server.until_disconnect(client(), broken()))
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

    assert asyncio.run(server.until_disconnect(client(leave_after=0.02), serve())) is None
    assert state == {"cancelled": True}


def test_a_cancelled_handler_cancels_its_serve():
    """The handler's own cancellation (a server shutting down) reaches the serve, as when the handler awaited it: a
    serve left running would put its entries on the engine for nobody."""
    state = {}

    async def serve():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def go():
        handler = asyncio.ensure_future(server.until_disconnect(client(), serve()))
        await asyncio.sleep(0.05)
        handler.cancel()
        await asyncio.wait({handler})
        await asyncio.sleep(0.05)
        return handler.cancelled(), dict(state)  # (before asyncio.run cancels whatever is left at its end)

    assert asyncio.run(go()) == (True, {"cancelled": True})


def test_a_waiting_request_costs_one_receive():
    """The wait for the disconnect is one receive() for the whole serve, not a poll every 0.25 s (about 40 us of
    the event loop each: 15% of a core with 1,024 requests waiting)."""
    request = client()

    async def serve():
        await asyncio.sleep(0.6)
        return {"ok": 1}

    assert asyncio.run(server.until_disconnect(request, serve())) == {"ok": 1}
    assert request.receives == 1


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
def test_a_client_that_leaves_while_sending_its_body_answers_499(path, capsys):
    """The connection closed in the middle of the body: 499 as for a client that leaves later, not a 500 with a
    traceback in the log (a server error that is none)."""
    srv = AppServer(Engine())
    app = server.http_app(srv, "fast")
    messages = [{"type": "http.request", "body": json.dumps(BODY).encode()[:10], "more_body": True},
                {"type": "http.disconnect"}]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "http_version": "1.1", "scheme": "http",
             "server": ("basal", 80), "client": ("test", 1), "root_path": ""}
    try:
        asyncio.run(app(scope, receive, send))
    except Exception as e:  # noqa: BLE001  (Starlette raises a handler's error again after its 500)
        print(f"raised {type(e).__name__}", file=sys.stderr)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    out = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
    assert (status, out) == GONE
    err = capsys.readouterr().err
    assert "Traceback" not in err and "raised" not in err and "request failed" not in err
    assert not srv.backend.entered.is_set()
