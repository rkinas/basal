import hashlib
import json
import os

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from basal.auth import ApiKeys, protect


def fixture(tmp_path, keys=True, cors=()):
    secret, admin = "bsl_test-secret", "bsl_admin-secret"
    path = tmp_path / "keys.json"
    if keys:
        path.write_text(json.dumps({"keys": [{"name": "user", "sha256": hashlib.sha256(secret.encode()).hexdigest(),
                                              "admin": False},
                                             {"name": "admin", "sha256": hashlib.sha256(admin.encode()).hexdigest(),
                                              "admin": True}]}))

    async def endpoint(_):
        return JSONResponse({"ok": True})

    async def stream(_):
        async def chunks():
            yield b"a"
            yield b"b"
        return StreamingResponse(chunks())

    app = Starlette(routes=[Route("/data", endpoint), Route("/stream", stream), Route("/mcp", stream),
                            Route("/health", endpoint)])
    return TestClient(protect(app, ApiKeys(path) if keys else None, cors)), path, secret, admin


def test_open_and_health(tmp_path):
    client, _, _, _ = fixture(tmp_path, keys=False)
    assert client.get("/data").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/keys").json() == {"keys": []}


def test_keys_authenticate_health_and_stream(tmp_path):
    client, _, key, _ = fixture(tmp_path)
    assert client.get("/data").json() == {"error": "missing or invalid API key", "code": "UNAUTHORIZED"}
    assert client.get("/data", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/data", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/data", headers={"X-API-Key": key}).status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/stream", headers={"X-API-Key": key}).content == b"ab"
    assert client.get("/mcp", headers={"X-API-Key": key}).content == b"ab"


def test_revocation_live_and_admin_usage(tmp_path):
    client, path, key, admin = fixture(tmp_path)
    assert client.get("/v1/keys", headers={"X-API-Key": key}).status_code == 403
    assert client.get("/v1/keys", headers={"X-API-Key": key}).json()["code"] == "FORBIDDEN"
    usage = client.get("/v1/keys", headers={"X-API-Key": admin}).json()["keys"]
    assert next(item for item in usage if item["name"] == "user")["last_used"] is not None
    path.write_text('{"keys": []}')
    assert client.get("/data", headers={"X-API-Key": key}).status_code == 401


def test_key_file_reloads_after_two_seconds_even_with_same_signature(tmp_path, monkeypatch):
    import basal.auth as auth

    path = tmp_path / "keys.json"
    old_key, new_key = "old-secret", "new-secret"
    def contents(key):
        return json.dumps({"keys": [{"name": "user", "sha256": hashlib.sha256(key.encode()).hexdigest()}]})

    path.write_text(contents(old_key))
    keys = ApiKeys(path)
    now = 100.0
    monkeypatch.setattr(auth.time, "monotonic", lambda: now)
    stat = path.stat()
    assert keys.check(old_key)["name"] == "user"

    path.write_text(contents(new_key))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert path.stat().st_ino == stat.st_ino
    now += 2.1
    assert keys.check(old_key) is None
    assert keys.check(new_key)["name"] == "user"


def test_invalid_file_fails_closed(tmp_path):
    client, path, _, _ = fixture(tmp_path)
    path.write_text("invalid")
    assert client.get("/data").status_code == 401
    assert client.get("/health").status_code == 200


def test_cors_preflight_only_allowed_origin(tmp_path):
    client, _, _, _ = fixture(tmp_path, cors=("https://allowed.example",))
    headers = {"Origin": "https://allowed.example", "Access-Control-Request-Method": "POST"}
    response = client.options("/data", headers=headers)
    assert response.headers["access-control-allow-origin"] == "https://allowed.example"
    response = client.options("/data", headers={**headers, "Origin": "https://other.example"})
    assert "access-control-allow-origin" not in response.headers
