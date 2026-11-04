"""basal-router passes POST /v1/decisions (OpenAI's Decisions API) on to a replica, routed by its input as a state is,
with the client's key; the cascade, blend and fallback, which read basal answers, leave its response untouched."""
import asyncio
import json

import httpx
from starlette.testclient import TestClient

from basal import router as R

REPLY = b'{"object": "decision", "output": [{"type": "predicate", "value": true}],  "x": 1}'


def transport(seen):
    def handler(req):
        base = req.url.host
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.append((base, req.url.path, req.headers.get("x-api-key"), json.loads(req.content)))
        if req.url.path == "/v1/decisions":
            return httpx.Response(200, content=REPLY, headers={"content-type": "application/json"})
        return httpx.Response(200, json={"answers": {"q": {"type": "choice", "choice": "a",
                                                           "probabilities": {"a": 0.5, "b": 0.5}}}})
    return httpx.MockTransport(handler)


def test_decisions_pass_through_routed_by_input():
    seen = []
    names = [f"http://r{i}" for i in range(3)]
    rt = R.Router(names, transport=transport(seen))
    body = {"input": "Invoice 17 is overdue.", "questions": [{"type": "predicate", "instructions": "Paid?"}]}
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        got = [c.post("/v1/decisions", json=body, headers={"X-API-Key": "k"}) for _ in range(4)]
    assert all(r.status_code == 200 and r.content == REPLY for r in got)
    assert len({s[0] for s in seen}) == 1 and all(s[1:] == ("/v1/decisions", "k", body) for s in seen)
    assert R.state_key(json.dumps(body)) == "Invoice 17 is overdue."


def test_decisions_skip_the_cascade():
    seen = []
    rt = R.Router(["http://small"], transport=transport(seen), escalate=["http://big"], below=0.99)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/decisions", json={"input": "x", "questions": [{"type": "predicate", "instructions": "?"}]})
    assert r.content == REPLY and [s[0] for s in seen] == ["small"]
