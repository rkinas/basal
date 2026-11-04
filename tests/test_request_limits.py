"""Request limits of basal-serve, checked before the engine: a body over --max-body-mb (default 256 MiB, above the
largest request the media limits allow) answers 413 whether it gives its length or not, without being read whole;
a request over --max-branches (default 4,096: a question counts 1, a multi question or a tool set one per label)
answers 422."""
import asyncio
import json

import httpx
import pytest
from test_http_status import BODY, AppServer, Engine

from basal import media
from basal.server import MAX_BODY_BYTES, MAX_BRANCHES, check_body, http_app, parser


class Counting(Engine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run_shared(self, groups, policy=None):
        self.calls += 1
        return super().run_shared(groups, policy)


def srv_with(**kw):
    srv = AppServer(Counting())
    srv.engine_calls = lambda: srv.backend.calls
    for k, v in kw.items():
        setattr(srv, k, v)
    return srv


async def _post(srv, path, content, headers=None):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        transport = httpx.ASGITransport(app=http_app(srv, "fast"))
        async with httpx.AsyncClient(transport=transport, base_url="http://basal") as c:
            return await c.post(path, content=content, headers=headers or {"content-type": "application/json"})
    finally:
        worker.cancel()


def post(srv, path, content, headers=None):
    return asyncio.run(_post(srv, path, content, headers))


def test_defaults_take_the_largest_media_request():
    a = parser().parse_args([])
    assert a.max_body_mb * 2**20 == MAX_BODY_BYTES and a.max_branches == MAX_BRANCHES == 4096
    base64_items = media.MAX_ITEMS * (media.MAX_BYTES * 4 // 3 + 4)
    assert MAX_BODY_BYTES > base64_items + 16 * 2**20      # 8 items of 20 MB in base64, and room for the text


@pytest.mark.parametrize("path,shape", [("/v1/systemone", "basal"), ("/v1/decisions", "openai")])
def test_a_body_over_the_limit_is_refused_with_413(path, shape):
    srv = srv_with(max_body=1000)
    body = json.dumps(dict(BODY, state="x" * 2000)).encode()
    r = post(srv, path, body)
    assert r.status_code == 413 and srv.engine_calls() == 0
    err = r.json()["error"]
    assert ("1000 bytes" in (err if shape == "basal" else err["message"]))


def test_a_streamed_body_over_the_limit_is_refused_without_its_length():
    srv = srv_with(max_body=1000)
    sent = []

    async def chunks():
        for _ in range(100):
            sent.append(1)
            yield b" " * 100
    r = post(srv, "/v1/systemone", chunks())
    assert r.status_code == 413 and srv.engine_calls() == 0


def test_a_body_under_the_limit_is_answered():
    srv = srv_with(max_body=10_000)
    assert post(srv, "/v1/systemone", json.dumps(BODY).encode()).status_code == 200


def test_too_many_branches_are_refused_before_the_engine():
    srv = srv_with(max_branches=4)
    q = BODY["questions"]["q"]
    many = {"state": "s", "questions": {f"q{i}": q for i in range(5)}}
    labels = {"state": "s", "questions": {"m": {"type": "multi", "instructions": "Which?",
                                                "criteria": {f"l{i}": "x" for i in range(5)}}}}
    for body in (many, labels):
        r = post(srv, "/v1/systemone", json.dumps(body).encode())
        assert r.status_code == 422 and "--max-branches" in r.json()["error"] and srv.engine_calls() == 0
    ok = {"state": "s", "questions": {"m": dict(labels["questions"]["m"], criteria=["a", "b", "c"]), "q": q}}
    assert post(srv, "/v1/systemone", json.dumps(ok).encode()).status_code == 200


def test_check_body_counts_labels_and_tools():
    tools = [{"name": f"t{i}", "description": "d", "parameters": {}} for i in range(3)]
    check_body({"state": "s", "questions": {"t": {"type": "tool", "tools": tools, "multi": True}}}, 3)
    with pytest.raises(ValueError, match="4 branches"):
        check_body({"state": "s", "questions": {"t": {"type": "tool", "tools": tools, "multi": True},
                                                "q": {"type": "noul"}}}, 3)
