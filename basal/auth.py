"""API-key authentication for basal-serve."""
import hashlib
import asyncio
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
        self._lock = threading.Lock()  # the file: a reload may wait for a rewrite in place (never on the event loop)
        self._usage_lock = threading.Lock()
        self._good = None  # (signature, records, loaded_at) of the last good read, replaced as ONE tuple: a reader without the lock never sees a half-set state
        self._bad = None  # (signature, time) of a file that could not be read: not read again for 2 s
        self._usage = {}
        self._warned = False

    def _cached(self):
        """The records while the file is as it was when read (for 2 s), [] while it is as it was when it failed to read
        (fail closed, no second wait), else None: the file has to be read."""
        try:
            stat = os.stat(self.path)
        except OSError:
            return None
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        now = time.monotonic()
        good = self._good  # one read: signature, records and time belong together
        if good is not None and signature == good[0] and now - good[2] <= 2:
            return good[1]
        bad = self._bad
        if bad is not None and signature == bad[0] and now - bad[1] <= 2:
            return []
        return None

    def _load(self):
        signature = None
        try:
            cached = self._cached()
            if cached is not None:
                return cached
            stat = os.stat(self.path)
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            now = time.monotonic()
            from .textio import read_text
            try:
                data = json.loads(read_text(self.path))
            except ValueError:  # JSONDecodeError, UnicodeDecodeError (cut inside a character): the file is being
                # rewritten in place; read it once more after the write. This waits, so check() and usage() reach it
                # from the event loop only through a thread (acheck, to_thread), and a failure is remembered for 2 s.
                time.sleep(0.05)
                data = json.loads(read_text(self.path))
            records = data["keys"]
            if not isinstance(records, list) or any(not isinstance(k, dict) or
                    not isinstance(k.get("name"), str) or not isinstance(k.get("sha256"), str) or
                    len(k["sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in k["sha256"])
                    or not isinstance(k.get("admin", False), bool) for k in records):
                raise ValueError("invalid API key file")
            self._good, self._warned = (signature, records, now), False
            return records
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self._good = None
            if signature is not None:
                self._bad = (signature, time.monotonic())
            if not self._warned:
                print(f"basal: cannot read API key file: {e}", file=sys.stderr, flush=True)
                self._warned = True
            return []

    def _records_now(self):
        records = self._cached()  # without the lock: a reload in another thread does not hold this one up
        if records is None:
            with self._lock:
                records = self._load()
        return records

    def check(self, secret):
        digest = hashlib.sha256(secret.encode()).hexdigest()
        for record in self._records_now():
            if hmac.compare_digest(digest, record["sha256"].lower()):
                return record
        return None

    async def acheck(self, secret):
        """check() for the event loop: a read of the file (which may wait for a rewrite) runs in a thread."""
        if self._cached() is not None:
            return self.check(secret)
        return await asyncio.to_thread(self.check, secret)

    def touch(self, name):
        with self._usage_lock:
            self._usage[name] = time.time()

    def usage(self):
        records = self._records_now()
        with self._usage_lock:
            return [{"name": key["name"], "last_used": self._usage.get(key["name"])} for key in records]


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
        open_health = path in ("/health", "/health/") and method in ("GET", "HEAD")
        record = None
        if not open_health and keys is not None:
            headers = dict(scope.get("headers", ()))
            authorization = headers.get(b"authorization", b"").decode("latin1")
            secret = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            x_key = headers.get(b"x-api-key", b"").decode("latin1")
            for s in (x_key, secret):
                record = await keys.acheck(s) if s else None
                if record is not None:
                    break
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
        if path == "/v1/keys" and method == "GET" and keys is not None:  # without --api-keys: the app's 404
            if record is not None and not record.get("admin", False):
                await _json(send, 403, {"error": "admin key required", "code": "FORBIDDEN"})
                return
            await _json(send, 200, {"keys": await asyncio.to_thread(keys.usage) if keys else []})
            return
        await app(scope, receive, send)

    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware
        return CORSMiddleware(auth_app, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                              allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id", "Mcp-Session-Id",
                                             "Mcp-Protocol-Version"], expose_headers=["Mcp-Session-Id", "x-request-id"])
    return auth_app


def request_ids(app):
    """Echo X-Request-Id (or a generated req_<hex>) as x-request-id on every response."""
    async def with_request_id(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        supplied = dict(scope.get("headers", ())).get(b"x-request-id", b"").decode("latin1")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied) else f"req_{uuid.uuid4().hex}"

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode())]
            await send(message)
        await app(scope, receive, send_with_id)
    return with_request_id
