"""API-key authentication for basal-serve."""
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
            from .textio import read_text
            data = json.loads(read_text(self.path))
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
        open_health = path in ("/health", "/health/") and method in ("GET", "HEAD")
        record = None
        if not open_health and keys is not None:
            headers = dict(scope.get("headers", ()))
            authorization = headers.get(b"authorization", b"").decode("latin1")
            secret = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            x_key = headers.get(b"x-api-key", b"").decode("latin1")
            record = next((r for r in (keys.check(s) for s in (x_key, secret) if s) if r is not None), None)
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
            await _json(send, 200, {"keys": keys.usage() if keys else []})
            return
        await app(scope, receive, send)

    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware
        return CORSMiddleware(auth_app, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                              allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id", "Mcp-Session-Id",
                                             "Mcp-Protocol-Version"], expose_headers=["Mcp-Session-Id", "x-request-id"])
    return auth_app


def request_ids(app):
    """Echo X-Request-Id (or a generated req_<hex>) as x-request-id on every response; the app finds it in
    scope["basal.request_id"] (basal-router passes it on to the replicas)."""
    async def with_request_id(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        supplied = dict(scope.get("headers", ())).get(b"x-request-id", b"").decode("latin1")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied) else f"req_{uuid.uuid4().hex}"
        scope["basal.request_id"] = request_id

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode())]
            await send(message)
        await app(scope, receive, send_with_id)
    return with_request_id


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")


def loopback(host):
    """A loopback host: localhost or an address of 127.0.0.0/8 / ::1."""
    import ipaddress
    host = (host or "").strip().strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def allowed_hosts(bind, allowed=None, api_keys=None):
    """The Host names a request may give (host_check), or None for no check. --allowed-hosts "*": none; a list: those
    and the loopback names; without it, the loopback names when the server is bound to a loopback host and has no
    --api-keys (a web page cannot then reach it through DNS rebinding), else no check (a server bound to the network
    is reached by name or address; one with --api-keys needs a key a page does not have)."""
    if allowed is not None:
        names = [h.strip().lower() for h in allowed.split(",") if h.strip()]
        return None if "*" in names else set(names) | set(LOOPBACK_NAMES)
    return set(LOOPBACK_NAMES) if loopback(bind) and not api_keys else None


def _host_of(value):
    """The host part of a Host header value ("127.0.0.1:8000", "[::1]:8000", "localhost")."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value[1:]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def host_check(app, hosts, cors_origins=()):
    """ASGI: a request whose Host header names a host outside `hosts`, or that comes from a web page of another origin
    (an Origin header outside `hosts` and --cors-origins), answers 403 before the app reads it: a page cannot reach a
    loopback server through DNS rebinding, nor send it requests from another site. GET and HEAD of /health and /health/
    stay open, as protect leaves them (a load balancer or a proxy probes with a host of its own)."""
    from urllib.parse import urlsplit
    origins = {o.rstrip("/").lower() for o in cors_origins}

    async def checked(scope, receive, send):
        if scope["type"] != "http" or (scope["path"] in ("/health", "/health/") and scope["method"] in ("GET", "HEAD")):
            await app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        host = headers.get(b"host", b"").decode("latin1")
        origin = headers.get(b"origin", b"").decode("latin1").strip()
        why = None
        if host and _host_of(host) not in hosts:
            why = f"Host {host!r} is not one this server answers (--allowed-hosts)"
        elif origin and origin.rstrip("/").lower() not in origins:
            try:
                name = urlsplit(origin).hostname
            except ValueError:
                name = None
            if name is None or name.lower() not in hosts:
                why = f"Origin {origin!r} is not allowed (--allowed-hosts, --cors-origins)"
        if why is not None:
            await _json(send, 403, {"error": why, "code": "FORBIDDEN"})
            return
        await app(scope, receive, send)
    return checked
