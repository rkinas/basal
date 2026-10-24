"""API-key authentication for basal-serve (and the basal-ui API server)."""
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
import uuid


class ApiKeys:
    def __init__(self, path):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._signature = None
        self._loaded_at = None
        self._usage = {}
        self._warned = False

    def _load(self):
        try:
            stat = os.stat(self.path)
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            now = time.monotonic()
            if signature == self._signature and now - self._loaded_at <= 2:
                return self._records
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            records = data["keys"]
            if not isinstance(records, list) or any(not isinstance(k, dict) or
                    not isinstance(k.get("name"), str) or not isinstance(k.get("sha256"), str) or
                    len(k["sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in k["sha256"])
                    or not isinstance(k.get("admin", False), bool) for k in records):
                raise ValueError("invalid API key file")
            self._signature, self._records, self._warned = signature, records, False
            self._loaded_at = now
            return records
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self._signature, self._records = None, []
            if not self._warned:
                print(f"basal: cannot read API key file: {e}", file=sys.stderr, flush=True)
                self._warned = True
            return []

    def check(self, secret):
        with self._lock:
            digest = hashlib.sha256(secret.encode()).hexdigest()
            for record in self._load():
                if hmac.compare_digest(digest, record["sha256"].lower()):
                    return record
        return None

    def touch(self, name):
        with self._lock:
            self._usage[name] = time.time()

    def usage(self):
        with self._lock:
            return [{"name": key["name"], "last_used": self._usage.get(key["name"])}
                    for key in self._load()]


async def _json(send, status, body, headers=()):
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *headers]})
    await send({"type": "http.response.body", "body": data})


def protect(app, keys, cors_origins=()):
    """Protect HTTP routes without consuming request or response streams."""
    async def auth_app(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        path, method = scope["path"], scope["method"]
        open_health = path == "/health" and method == "GET"
        record = None
        if not open_health and keys is not None:
            headers = dict(scope.get("headers", ()))
            authorization = headers.get(b"authorization", b"").decode("latin1")
            secret = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            secret = headers.get(b"x-api-key", b"").decode("latin1") or secret
            record = keys.check(secret) if secret else None
            if record is None:
                if path in ("/v1/decisions", "/decisions"):
                    body = {"error": {"message": "missing or invalid API key", "type": "authentication_error",
                                      "param": None, "code": "UNAUTHORIZED"}}
                else:
                    body = {"error": "missing or invalid API key", "code": "UNAUTHORIZED"}
                await _json(send, 401, body, ((b"www-authenticate", b"Bearer"),))
                return
            keys.touch(record["name"])
            scope["basal.api_key"] = record  # routes behind protect() may check admin rights
        if path == "/v1/keys" and method == "GET":
            if record is not None and not record.get("admin", False):
                await _json(send, 403, {"error": "admin key required", "code": "FORBIDDEN"})
                return
            await _json(send, 200, {"keys": keys.usage() if keys else []})
            return
        await app(scope, receive, send)

    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware
        return CORSMiddleware(auth_app, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                              allow_headers=["Authorization", "Content-Type", "X-API-Key", "Mcp-Session-Id",
                                             "Mcp-Protocol-Version"], expose_headers=["Mcp-Session-Id", "x-request-id",
                                                                                    "x-typesafe-request-id"])
    return auth_app


def request_ids(app):
    """Echo X-Request-Id (or a generated req_<hex>) as x-request-id and x-typesafe-request-id on every response."""
    async def with_request_id(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        supplied = dict(scope.get("headers", ())).get(b"x-request-id", b"").decode("latin1")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied) else f"req_{uuid.uuid4().hex}"

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()), (b"x-typesafe-request-id", request_id.encode())]
            await send(message)
        await app(scope, receive, send_with_id)
    return with_request_id
