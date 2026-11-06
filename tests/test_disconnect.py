"""A request whose HTTP client disconnects (a client timeout, a retry) is cancelled (#129): its entries still in the
queue never reach the engine, it answers 499, and the requests around it are answered as before. Through the app of
basal-serve (http_app) with its batching worker and a stub engine that holds the first batch on the GPU."""
import asyncio
import json
import sys
import threading
import time

import httpx
import pytest
from starlette.requests import Request

from basal import server
from basal import think as T
from tests.test_http_status import BODY, AppServer, Engine

GONE = (499, {"error": "client disconnected", "code": "CLIENT_DISCONNECTED"})  # a request whose client left


IDLE = {"running": 0, "queued": 0}  # GET /health of a server with nothing in flight


class CountingEngine(Engine):
    def __init__(self):
        super().__init__()
        self.calls = self.groups = 0

    def run_shared(self, groups, policy=None):
        self.calls += 1
        self.groups += len(groups)  # (one group per request: AppServer packs a request into one entry)
        return super().run_shared(groups, policy)


async def raw_post(app, path, body, gone, headers=()):
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
             "headers": [(b"content-type", b"application/json"), *headers], "http_version": "1.1", "scheme": "http",
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
                held = (await held).status_code
                assert srv.load() == IDLE
                return status, out, held
        finally:
            worker.cancel()

    status, out, held = asyncio.run(go())
    return status, out, held, engine.calls


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/basal"])
def test_a_request_abandoned_in_the_queue_never_reaches_the_engine(path):
    status, out, held, calls = run_abandoned(path, BODY)
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


async def until(cond, seconds=5):
    for _ in range(int(seconds * 100)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


def on_app(srv, fn):
    """fn(app, httpx client) on http_app over srv with its batching worker -> its result."""
    async def go():
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            app = server.http_app(srv, "fast")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
                return await fn(app, c)
        finally:
            worker.cancel()
    return asyncio.run(go())


def test_a_request_abandoned_while_its_batch_forms_never_reaches_the_engine():
    """--wait-ms: the worker has taken the request from the queue and waits for more to join its call when the client
    leaves. The call starts without it (forward skips done futures) and serves the request that joined after it."""
    engine = CountingEngine()
    srv = AppServer(engine)
    srv.wait = 0.3

    async def fn(app, c):
        gone = asyncio.Event()
        abandoned = asyncio.create_task(raw_post(app, "/v1/systemone", BODY, gone))
        await until(lambda: srv.queued == 1 and srv.queue.empty())  # in the forming batch, not yet in a call
        gone.set()
        status, out = await asyncio.wait_for(abandoned, 5)
        assert not engine.entered.is_set()  # (answered before the call started)
        later = await c.post("/v1/systemone", json=BODY)  # joins the same batch
        return status, out, later.status_code, srv.load()

    status, out, later, load = on_app(srv, fn)
    assert (status, out) == GONE
    assert (later, load) == (200, IDLE)
    assert (engine.calls, engine.groups) == (1, 1)  # the later request only


def test_a_request_abandoned_on_the_gpu_answers_at_once_and_the_server_goes_on():
    """The client leaves while its batch is on the engine: 499 without waiting for the engine, whose call finishes
    (its answer for nobody); the next request is served and nothing stays counted in flight."""
    engine = CountingEngine()
    engine.go.clear()
    srv = AppServer(engine)

    async def fn(app, c):
        gone = asyncio.Event()
        abandoned = asyncio.create_task(raw_post(app, "/v1/systemone", BODY, gone))
        await asyncio.get_running_loop().run_in_executor(None, engine.entered.wait, 5)
        gone.set()
        status, out = await asyncio.wait_for(abandoned, 5)  # the engine still holds its batch
        engine.go.set()
        later = await c.post("/v1/systemone", json=BODY)
        return status, out, later.status_code, srv.load()

    status, out, later, load = on_app(srv, fn)
    assert (status, out) == GONE
    assert (later, load) == (200, IDLE)
    assert engine.calls == 2


class Sidecar:
    """The vLLM server of --think auto: its reasoning waits until it is cancelled (in basal-serve, httpx's request to
    vLLM is cancelled with the serve, which closes its connection: vLLM then stops the generation)."""
    max_len = 8192

    def __init__(self):
        self.reasoning, self.stopped = asyncio.Event(), False

    async def reason(self, ids, max_tokens):
        self.reasoning.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.stopped = True
            raise


class Thinking(T.ThinkServer, AppServer):
    """ThinkServer.decide over the batching worker and the stub engine, with the stub sidecar."""

    def __init__(self, engine):
        AppServer.__init__(self, engine)
        self.think, self.think_cal, self.think_cal_name, self.think_log = "auto", {}, "off", None
        self.think_tokens, self.sidecar = 64, Sidecar()


def test_a_think_request_abandoned_while_it_reasons_stops_the_reasoning():
    srv = Thinking(CountingEngine())

    async def fn(app, c):
        gone = asyncio.Event()
        abandoned = asyncio.create_task(raw_post(app, "/v1/systemone", dict(BODY, think="force"), gone))
        await asyncio.wait_for(srv.sidecar.reasoning.wait(), 5)  # readout 1 done, the reasoning under way
        gone.set()
        status, out = await asyncio.wait_for(abandoned, 5)
        return status, out, srv.sidecar.stopped, srv.load()

    status, out, stopped, load = on_app(srv, fn)
    assert (status, out) == GONE
    assert (stopped, load) == (True, IDLE)


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


# -- behind basal-router: the router's own request to the replica is cancelled, so the replica sees the disconnect ----
class Uvicorn:
    """An ASGI app served by uvicorn on a free local port, in a thread of its own (as basal-serve serves it)."""

    def __init__(self, app):
        import socket

        import uvicorn
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(500):
            if self.server.started:
                return self
            time.sleep(0.01)
        raise RuntimeError("the server did not start")

    def __exit__(self, *_):
        self.server.should_exit = True
        self.thread.join(10)


def test_a_request_abandoned_behind_the_router_never_reaches_the_replicas_engine():
    """basal-router in front of a replica (basal-serve's app under uvicorn, over TCP): the replica's engine holds one
    request, the router's request queues behind it, and the router's client leaves. The router answers 499 at once and
    cancels its own request to the replica, which closes that connection: the replica sees the disconnect and drops
    its queued request, which never reaches the engine."""
    from basal import router as R
    engine = CountingEngine()
    engine.go.clear()
    srv = AppServer(engine)
    with Uvicorn(server.http_app(srv, "fast")) as replica:
        rt = R.Router([replica.url])
        rt.ready = {replica.url}

        async def go():
            async with httpx.AsyncClient(base_url=replica.url, timeout=30) as direct:
                held = asyncio.create_task(direct.post("/v1/systemone", json=BODY))
                await asyncio.get_running_loop().run_in_executor(None, engine.entered.wait, 5)
                gone = asyncio.Event()
                abandoned = asyncio.create_task(raw_post(R.app(rt), "/v1/systemone", BODY, gone))
                await until(lambda: srv.queued == 1)  # the router's request waits in the replica's queue
                assert rt.load[replica.url] == 1
                gone.set()
                status, out = await asyncio.wait_for(abandoned, 5)
                await until(lambda: srv.queued == 0)  # the replica saw its client (the router) leave
                engine.go.set()
                held = (await held).status_code
                await until(lambda: srv.load() == IDLE)
                await rt.client.aclose()
                return status, out, held, rt.load[replica.url], rt.failed[replica.url]

        status, out, held, load, failed = asyncio.run(go())
    assert (status, out) == GONE
    assert (held, engine.calls) == (200, 1)  # the held request only
    assert (load, failed) == (0, 0)  # (a client that left is no replica failure)


def test_the_router_cancels_the_call_of_a_request_whose_client_left():
    """The replica's call is cancelled with the router's handler, and a client that leaves while sending its body
    answers 499, not 500."""
    from basal import router as R
    calls = {"started": 0, "cancelled": 0}

    async def replica(request):
        calls["started"] += 1
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            calls["cancelled"] += 1
            raise
        return httpx.Response(200, json={})

    async def go():
        rt = R.Router(["http://r0"], transport=httpx.MockTransport(replica))
        rt.ready = {"http://r0"}
        gone = asyncio.Event()
        abandoned = asyncio.create_task(raw_post(R.app(rt), "/v1/systemone", BODY, gone))
        await until(lambda: calls["started"] == 1)
        gone.set()
        first = await asyncio.wait_for(abandoned, 5)
        messages = [{"type": "http.request", "body": b'{"state": "', "more_body": True}, {"type": "http.disconnect"}]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)
        try:
            await R.app(rt)({"type": "http", "method": "POST", "path": "/v1/systemone", "raw_path": b"/v1/systemone",
                             "query_string": b"", "headers": [], "http_version": "1.1", "scheme": "http",
                             "server": ("basal", 80), "client": ("test", 1), "root_path": ""}, receive, send)
        except Exception:  # noqa: BLE001  (Starlette raises a handler's error again after its 500)
            pass
        second = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return first, second, dict(calls), rt.load["http://r0"]

    first, second, seen, load = asyncio.run(go())
    assert first == GONE and second == 499
    assert seen == {"started": 1, "cancelled": 1} and load == 0


# -- /mcp: a tool call whose client leaves is cancelled with its stateless session ---------------------------------
class Tools:
    """The server behind /mcp: its serve waits until it is cancelled, or answers at once with `quick`."""
    name = "basal"

    def __init__(self, quick=False):
        self.quick, self.started, self.cancelled = quick, None, False

    async def decide(self, body):  # (mcp_server.http serves through serve)
        raise AssertionError("not called")

    async def serve(self, body):
        self.started.set()
        if self.quick:
            return {"model": "basal", "answers": {"answer": {"type": "noul", "noul": 0.9, "confidence": 0.9,
                                                             "probabilities": {"true": 0.9, "false": 0.1}}}}
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


MCP_CALL = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "decide", "arguments": {"state": "Invoice 1240 PLN.", "question": "Is it paid?"}}}
MCP_HEADERS = [(b"accept", b"application/json, text/event-stream"), (b"host", b"127.0.0.1:8000"),
               (b"mcp-protocol-version", b"2025-06-18")]


def mcp_call(tools, leave):
    """A tools/call through the /mcp routes of basal-serve --mcp on (raw ASGI, the session manager running); the client
    leaves once the serve has started when `leave` -> (status, body, whether the serve was cancelled)."""
    from starlette.applications import Starlette

    from basal import mcp_server
    routes, run = mcp_server.http(tools, "127.0.0.1")
    app = Starlette(routes=routes)

    async def go():
        tools.started = asyncio.Event()
        gone = asyncio.Event()
        async with run():
            call = asyncio.create_task(raw_post(app, "/mcp", MCP_CALL, gone, MCP_HEADERS))
            await asyncio.wait_for(tools.started.wait(), 5)
            if leave:
                gone.set()
            status, out = await asyncio.wait_for(call, 5)
        return status, out, tools.cancelled

    return asyncio.run(go())


def test_an_mcp_tool_call_whose_client_leaves_is_cancelled():
    status, out, cancelled = mcp_call(Tools(), leave=True)
    assert (status, out) == GONE
    assert cancelled is True


def test_an_mcp_tool_call_whose_client_stays_is_answered():
    status, out, cancelled = mcp_call(Tools(quick=True), leave=False)
    assert status == 200 and cancelled is False
    assert out["result"]["structuredContent"]["noul"] == 0.9
