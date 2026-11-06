"""Which headers and probes basal-serve's --api-keys lets through."""
import asyncio
import hashlib
import json

from basal import auth as A


async def ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def status(tmp_path, method, path, headers):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"keys": [{"name": "k1", "sha256": hashlib.sha256(b"good").hexdigest()}]}),
                         encoding="utf-8")
    out = {}

    async def send(m):
        if m["type"] == "http.response.start":
            out["s"] = m["status"]

    async def receive():
        return {"type": "http.request", "body": b""}
    scope = {"type": "http", "method": method, "path": path,
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(A.protect(ok_app, A.ApiKeys(keys_file))(scope, receive, send))
    return out["s"]


def test_a_valid_bearer_is_accepted_when_a_stale_x_api_key_header_is_also_sent(tmp_path):
    assert status(tmp_path, "POST", "/v1/systemone", {"x-api-key": "stale", "authorization": "Bearer good"}) == 200
    assert status(tmp_path, "POST", "/v1/systemone", {"x-api-key": "stale"}) == 401
    assert status(tmp_path, "POST", "/v1/systemone", {"authorization": "Bearer good"}) == 200
