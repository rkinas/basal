"""The memory a request can hold elsewhere than in basal-serve's engine: basal-router refuses a body over --max-body-mb
with 413 before reading it whole (as basal-serve does), and --ui full keeps at most FULL_BYTES of request bodies and
answers (the newest always)."""
import asyncio
import json
import types

import httpx
from starlette.testclient import TestClient
from test_ui import app_of

from basal import router as R
from basal import ui as ui_module


def test_the_router_refuses_a_body_over_its_limit_before_reading_it():
    seen = []

    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.append(len(req.content))
        return httpx.Response(200, json={"answers": {}})
    rt = R.Router(["http://main"], transport=httpx.MockTransport(handler))
    body = json.dumps({"state": "x" * 5000, "questions": {"q": {"type": "choice"}}}).encode()
    with TestClient(R.app(rt, poll_s=0.05, max_body=1000)) as c:
        asyncio.run(rt.poll())
        big = c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
        dec = c.post("/v1/decisions", content=body, headers={"content-type": "application/json"})
        small = c.post("/v1/systemone", content=body[:900], headers={"content-type": "application/json"})
    assert big.status_code == 413 and big.json()["code"] == "PAYLOAD_TOO_LARGE" and "1000 bytes" in big.json()["error"]
    assert dec.status_code == 413 and "1000 bytes" in dec.json()["error"]["message"]
    assert small.status_code == 200 and seen == [900]  # (only the body under the limit reached a replica)


def test_router_flag_defaults_to_the_servers_limit():
    import basal.server as S
    ap_defaults = {a.dest: a.default for a in S.parser()._actions}
    assert ap_defaults["max_body_mb"] * 2**20 == R.MAX_BODY_BYTES == 256 * 2**20


def test_ui_full_keeps_the_newest_requests_within_its_byte_budget(monkeypatch):
    monkeypatch.setattr(ui_module, "FULL_BYTES", 1000)
    _, u, _ = app_of(full=True)

    async def handler(_):
        return types.SimpleNamespace(status_code=200, body=b'{"answers": {}}')
    rec = u.record(handler, "/v1/systemone")

    async def send(state):
        body = {"state": state, "questions": {}}
        await rec(types.SimpleNamespace(_json=body, _body=json.dumps(body).encode()))
    asyncio.run(send("a" * 300))
    asyncio.run(send("b" * 300))
    assert len(u.bodies) == 2 and u.kept_bytes <= 1000
    asyncio.run(send("c" * 700))  # (the two older ones no longer fit with it)
    assert [b[0] for b in u.bodies] == [3] and u.kept_bytes <= 1000
    assert u.request_json(1) is None and json.loads(u.request_json(3))["request"]["state"] == "c" * 700
    asyncio.run(send("d" * 5000))  # larger than the budget: kept alone
    assert [b[0] for b in u.bodies] == [4]
