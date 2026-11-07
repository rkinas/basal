"""A lone surrogate ("\\ud800", valid JSON but no Unicode text) anywhere in a request answers 422 (400 on
/v1/decisions) before the engine runs, instead of a 500 from the tokenizer or the answer's UTF-8; an escaped pair
(an emoji) is one character and is decided as before; --ui full can still show a kept request that held one."""
import asyncio

import pytest

from tests.test_http_status import AppServer, Engine, on_app
from tests.test_ui import SECRET, app_of, fresh, keyfile, on

Q = '{"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}'


@pytest.mark.parametrize("content", [
    '{"state": "a\\ud800b", "questions": {"q": %s}}' % Q,
    '{"state": "s", "questions": {"\\udc00": %s}}' % Q,
    '{"state": "s", "questions": {"q": {"type": "choice", "instructions": "\\ud83d", "criteria": ["a", "b"]}}}',
    '{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["\\udfff", "b"]}}}',
    '{"state": ["x", {"k": "\\ud800"}], "questions": {"q": %s}}' % Q,
])
def test_lone_surrogate_answers_422_before_the_engine(content):
    srv = AppServer(Engine())

    async def go(c):
        return await c.post("/v1/systemone", content=content.encode())
    r = asyncio.run(on_app(srv, go))
    assert r.status_code == 422 and "lone surrogate" in r.json()["error"]
    assert not srv.backend.entered.is_set()


def test_lone_surrogate_on_decisions_answers_400_and_a_pair_is_decided():
    srv = AppServer(Engine())

    async def go(c):
        bad = await c.post("/v1/decisions", content=b'{"input": "a\\ud800", "questions": '
                                                     b'[{"type": "predicate", "instructions": "Paid?"}]}')
        pair = await c.post("/v1/systemone", content=('{"state": "ok \\ud83d\\ude00 \\\\ud800", "questions": {"q": %s}}'
                                                      % Q).encode())
        return bad, pair
    bad, pair = asyncio.run(on_app(srv, go))
    assert bad.status_code == 400 and "lone surrogate" in bad.json()["error"]["message"]
    assert pair.status_code == 200 and "q" in pair.json()["answers"]


def test_ui_full_shows_a_kept_request_with_a_lone_surrogate(tmp_path):
    keys = keyfile(tmp_path)
    srv, u, app = app_of(["--api-keys", str(keys)], keys=keys, full=True)
    auth = {"Authorization": f"Bearer {SECRET}"}

    async def go(c):
        r = await c.post("/v1/systemone", content=('{"state": "a\\ud800", "questions": {"q": %s}}' % Q).encode(),
                         headers=auth)
        fresh(u)
        return r, await c.get("/ui/request/1", headers=auth)
    r, view = asyncio.run(on(srv, app, go))
    assert r.status_code == 422
    assert view.status_code == 200 and view.json()["request"]["state"] == "a\ud800" and view.json()["status"] == 422
