"""basal-serve --ui full with --api-keys: a kept request (its text and answer, GET /ui/request/<id>) is read by the key
that made it or by an admin key, never by another client's key; without --api-keys the page stays open as before."""
import asyncio
import hashlib
import json

import pytest
from test_ui import app_of, on

from tests.test_http_status import BODY

KEYS = {"alice": ("bsl_alice-secret", False), "bob": ("bsl_bob-secret", False), "ops": ("bsl_ops-secret", True)}


def keyfile(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"keys": [{"name": n, "sha256": hashlib.sha256(s.encode()).hexdigest(), "admin": a}
                                         for n, (s, a) in KEYS.items()]}))
    return path


@pytest.mark.parametrize("form", ["bearer", "x-api-key"])
def test_another_clients_key_cannot_read_a_kept_request(tmp_path, form):
    keys = keyfile(tmp_path)
    srv, u, app = app_of(["--api-keys", str(keys)], keys=keys, full=True)

    def auth(name):
        secret = KEYS[name][0]
        return {"Authorization": f"Bearer {secret}"} if form == "bearer" else {"X-API-Key": secret}

    async def go(c):
        await c.post("/v1/systemone", json=BODY, headers=auth("alice"))
        return {n: await c.get("/ui/request/1", headers=auth(n)) for n in KEYS}
    got = asyncio.run(on(srv, app, go))
    assert got["bob"].status_code == 403 and BODY["state"] not in got["bob"].text
    assert got["alice"].status_code == 200 and got["alice"].json()["request"] == BODY
    assert got["ops"].status_code == 200 and got["ops"].json()["request"] == BODY


def test_without_api_keys_the_kept_requests_stay_open():
    srv, u, app = app_of(full=True)

    async def go(c):
        await c.post("/v1/systemone", json=BODY)
        return await c.get("/ui/request/1")
    r = asyncio.run(on(srv, app, go))
    assert r.status_code == 200 and r.json()["request"] == BODY
