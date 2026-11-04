"""Regression tests for fix/router-client (Z-1082)."""
import asyncio
import json

import httpx
import pytest

from basal import client as C
from basal import router as R


class Reply:
    """What the code reads of an httpx response."""
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.headers = status, {k.lower(): v for k, v in (headers or {}).items()}
        self.content = json.dumps(body if body is not None else {}).encode()
        self.text = self.content.decode()

    def json(self):
        return json.loads(self.content)


OK = Reply(200, {"answers": {"q": {"answer": "a", "confidence": 0.9}}, "model": "m", "usage": {}})
STARTING = Reply(503, {"status": "starting"}, {"retry-after": "5"})
NO_REPLICA = Reply(503, {"error": "no basal replica is ready"})
OOM = Reply(503, {"error": "CUDA out of memory", "code": "OUT_OF_MEMORY"})


class Http:
    def __init__(self, replies):
        self.replies, self.calls, self.headers = list(replies), 0, {}

    def request(self, method, url, **kw):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def attempt(monkeypatch, replies, retries=2):
    sleeps = []
    monkeypatch.setattr(C.time, "sleep", sleeps.append)
    http = Http(replies)
    try:
        C.Basal(http=http, retries=retries, backoff=0.5).request({"state": "x", "questions": {}})
        return "ok", http.calls, sleeps
    except C.BasalError as e:
        return e, http.calls, sleeps


def test_i11_a_server_still_starting_is_not_out_of_memory(monkeypatch):
    err, calls, _ = attempt(monkeypatch, [STARTING] * 3)
    assert not isinstance(err, C.OutOfMemory) and calls == 3


def test_i11_the_retries_wait_at_least_retry_after(monkeypatch):
    _, _, sleeps = attempt(monkeypatch, [STARTING] * 3)
    assert sleeps and sleeps[0] >= 5


def test_i11_a_router_with_no_ready_replica_is_not_out_of_memory(monkeypatch):
    err, _, _ = attempt(monkeypatch, [NO_REPLICA] * 3)
    assert not isinstance(err, C.OutOfMemory)


def test_i11_real_out_of_memory_is_still_retried_with_the_backoff(monkeypatch):
    err, calls, sleeps = attempt(monkeypatch, [OOM] * 3)
    assert isinstance(err, C.OutOfMemory) and calls == 3 and sleeps == [0.5, 1.0]
    assert attempt(monkeypatch, [STARTING, OK])[:2] == ("ok", 2)


class Reply:
    """What the code reads of an httpx response."""
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.headers = status, {k.lower(): v for k, v in (headers or {}).items()}
        self.content = json.dumps(body if body is not None else {}).encode()
        self.text = self.content.decode()

    def json(self):
        return json.loads(self.content)

async def call(app, method, path, body=b"", headers=()):
    sent, done = [], False
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path,
             "raw_path": path.encode(), "query_string": b"", "server": ("x", 80), "client": ("c", 1),
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}

    async def receive():
        nonlocal done
        if not done:
            done = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    async def send(m):
        sent.append(m)
    await app(scope, receive, send)
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))

BODY = json.dumps({"state": "x", "questions": {}}).encode()
JSON = [("content-type", "application/json")]


def mock(fn):
    """An httpx transport that answers with what `async fn(url, content, headers)` returns (a Reply) or raises."""
    async def handler(request):
        r = await fn(str(request.url), request.content, dict(request.headers))
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)
    return httpx.MockTransport(handler)


def make(transport, ready=True):
    r = R.Router(["http://r1"], transport=mock(transport) if transport else None)
    if ready:
        r.ready.add("http://r1")
    return r, R.app(r)


def run(coro):
    return asyncio.run(coro)


def test_i12_the_client_request_id_reaches_the_replica_and_the_response():
    seen = {}

    async def transport(url, content, headers):
        seen.update({k: v.decode() if isinstance(v, bytes) else v for k, v in headers.items()})
        return Reply(200, {"answers": {}}, {"content-type": "application/json", "x-request-id": "replica-own"})
    _, app = make(transport)
    _, h, _ = run(call(app, "POST", "/v1/systemone", BODY, [*JSON, ("x-request-id", "client-123")]))
    assert h.get("x-request-id") == "client-123" and seen.get("x-request-id") == "client-123"


def test_i12_a_request_without_an_id_gets_one():
    async def transport(url, content, headers):
        return Reply(200, {"answers": {}}, {"content-type": "application/json"})
    _, app = make(transport)
    _, h, _ = run(call(app, "POST", "/v1/systemone", BODY, JSON))
    assert h.get("x-request-id")


def test_i12_the_routers_own_503_says_when_to_retry():
    async def transport(url, content, headers):
        raise AssertionError("no replica is ready")
    _, app = make(transport, ready=False)
    st, h, _ = run(call(app, "POST", "/v1/systemone", BODY, JSON))
    assert st == 503 and h.get("retry-after")


def test_i13_the_replica_client_has_a_timeout():
    r, _ = make(None)
    timeout = r.client.timeout
    assert getattr(timeout, "read", getattr(timeout, "timeout", None)) is not None


def test_i13_one_read_error_does_not_take_the_replica_out_of_the_pool():
    state = {"n": 0}

    async def flaky(url, content, headers):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ReadError("connection reset")
        return Reply(200, {"answers": {}})
    r, app = make(flaky)
    run(call(app, "POST", "/v1/systemone", BODY, JSON))
    assert run(call(app, "POST", "/v1/systemone", BODY, JSON))[0] == 200


def test_i13_a_refused_connection_still_takes_the_replica_out():
    async def down(url, content, headers):
        raise httpx.ConnectError("refused")
    r, app = make(down)
    run(call(app, "POST", "/v1/systemone", BODY, JSON))
    assert not r.ready
