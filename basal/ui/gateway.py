"""basal-ui's API server: one port for every loaded model. Each request names its model (`"model"` in the body); the
gateway checks API keys, CORS and serves MCP, then forwards to that model's private basal-serve and counts the
decision in the statistics and the history."""
import socket
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager

import httpx

SYSTEMONE = ("/v1/systemone", "/v1/basal")
DECISIONS = ("/v1/decisions", "/decisions")
SNAPSHOT_KEYS = ("host", "port", "require_key", "cors_origins", "mcp")


def build_app(fleet, stats, keys=None, cors_origins=(), mcp=False, host="127.0.0.1"):
    """The gateway's ASGI app. keys: basal.auth.ApiKeys or None (no key required)."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from basal.auth import protect, request_ids
    from basal.compat import openai_error

    from .app import host_middleware

    allowed_hosts = ["*"] if host in ("0.0.0.0", "::") else {host, "127.0.0.1", "localhost", "::1"}
    client: dict = {}

    def error(path, status, message, code):
        if path in DECISIONS:
            return openai_error(status, message, code)
        return JSONResponse({"error": message, "code": code}, status_code=status)

    def loaded_text(loaded):
        return ", ".join(loaded) or "none"

    async def forward(request):
        path = request.url.path
        try:
            body = await request.json()
        except Exception:
            return error(path, 400, "malformed JSON", "INVALID_JSON")
        if not isinstance(body, dict):
            return error(path, 400, "request body must be a JSON object", "INVALID_JSON")
        loaded = fleet.loaded_ids()
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return error(path, 400, f"model is required; loaded models: {loaded_text(loaded)}", "MODEL_REQUIRED")
        if model not in loaded:
            return error(path, 404, f"model '{model}' is not loaded; loaded models: {loaded_text(loaded)}",
                         "MODEL_NOT_FOUND")
        if path in SYSTEMONE:  # the child's own API has no model field; /v1/decisions validates it itself
            body = {k: v for k, v in body.items() if k != "model"}
        upstream = "/v1/systemone" if path in SYSTEMONE else "/v1/decisions"
        try:
            r = await fleet.adecide(model, body, path=upstream, source="api", client=client["http"])
        except (KeyError, RuntimeError) as exc:  # unloaded or restarted since the check above
            return error(path, 409, f"model '{model}' is not ready: {exc}", "MODEL_NOT_READY")
        except httpx.HTTPError as exc:
            return error(path, 502, f"basal-serve: {exc}", "UPSTREAM_ERROR")
        return Response(r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    async def models(_):
        entries = fleet.model_entries()
        return JSONResponse({"object": "list",
                             "data": [{"id": e["id"], "object": "model", "created": e["created"], "owned_by": "basal"}
                                      for e in entries],
                             "models": [e["entry"] or {"name": e["id"]} for e in entries]})

    async def health(_):
        return JSONResponse({"status": "ok", "models": fleet.loaded_ids()})

    async def stats_route(request):
        record = request.scope.get("basal.api_key")
        if record is not None and not record.get("admin", False):
            return JSONResponse({"error": "admin key required", "code": "FORBIDDEN"}, status_code=403)
        return JSONResponse(stats.snapshot())

    mcp_routes, mcp_run = [], None
    if mcp:
        from types import SimpleNamespace

        from basal.mcp_server import http

        async def mcp_decide(body):
            model = body.pop("model", None)
            loaded = fleet.loaded_ids()
            if model not in loaded:
                raise ValueError(f"model '{model}' is not loaded; loaded models: {loaded_text(loaded)}")
            r = await fleet.adecide(model, body, source="mcp", client=client["http"])
            if r.status_code != 200:
                try:
                    message = r.json()["error"]
                except Exception:  # noqa: BLE001
                    message = r.text
                raise ValueError(f"basal server {r.status_code}: {message}")
            return r.json()

        mcp_routes, mcp_run = http(SimpleNamespace(decide=mcp_decide, name="basal-ui"), host, fleet.loaded_ids)

    @asynccontextmanager
    async def lifespan(_app):
        async with AsyncExitStack() as stack:
            client["http"] = await stack.enter_async_context(httpx.AsyncClient(timeout=120))
            if mcp_run:
                await stack.enter_async_context(mcp_run())
            yield

    app = Starlette(routes=[
        *[Route(p, forward, methods=["POST"]) for p in SYSTEMONE + DECISIONS],
        Route("/v1/models", models), Route("/health", health), Route("/v1/stats", stats_route), *mcp_routes,
    ], lifespan=lifespan)
    return request_ids(protect(host_middleware(app, allowed_hosts), keys, cors_origins))


class Gateway:
    """Runs build_app with uvicorn in a daemon thread on the configured host and port while serving is enabled."""

    def __init__(self, fleet, settings_mod, stats):
        self.fleet, self.settings_mod, self.stats = fleet, settings_mod, stats
        self.lock = threading.Lock()
        self.server = self.thread = self.snapshot = self.keys = None
        self.error = None

    def start(self):
        import uvicorn

        from basal.auth import ApiKeys

        with self.lock:
            if self.server is not None:
                if self.thread.is_alive():
                    return self.status()
                self.server = self.thread = self.snapshot = self.keys = None  # exited after a timed-out stop
            config = self.settings_mod.serve_settings()
            self.error = None
            if not config["enabled"]:
                return self.status()
            host, port = config["host"], config["port"]
            with socket.socket() as s:
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((host, port))
                except OSError:
                    self.error = f"port {port} on {host} is in use"
                    self.fleet.event("error", f"API server: {self.error}")
                    return self.status()
            if config["mcp"]:
                from basal.mcp_server import available
                missing = available()
                if missing:
                    self.error = missing
                    self.fleet.event("error", f"API server: {missing}")
                    return self.status()
            keys = None
            if config["require_key"]:
                keys = ApiKeys(self.settings_mod.keys_file(self.fleet.internal_sha256()))
            app = build_app(self.fleet, self.stats, keys, config["cors_origins"], config["mcp"], host)
            server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
            thread = threading.Thread(target=server.run, name="basal-ui-gateway", daemon=True)
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not server.started:
                server.should_exit = True
                thread.join(timeout=5)
                self.error = f"the API server did not start on {host}:{port}"
                self.fleet.event("error", f"API server: {self.error}")
                return self.status()
            self.server, self.thread, self.keys = server, thread, keys
            self.snapshot = {k: config[k] for k in SNAPSHOT_KEYS}
            self.fleet.event("info", f"API server listening on {host}:{port}")
            return self.status()

    def stop(self):
        with self.lock:
            server, thread = self.server, self.thread
            if server is None:
                return True
            server.should_exit = True
            thread.join(timeout=5)
            if thread.is_alive():  # a long decision keeps the old server alive: the port is not free yet
                self.error = "the API server is still finishing a request; try again in a moment"
                self.fleet.event("warn", f"API server: {self.error}")
                return False
            self.server = self.thread = self.snapshot = self.keys = None
        self.fleet.event("info", "API server stopped")
        return True

    def restart(self):
        if not self.stop():
            return self.status()
        return self.start()

    def status(self):
        config = self.settings_mod.serve_settings()
        snapshot = self.snapshot
        if snapshot is None and self.error is None and not config["enabled"]:
            return None
        current = {k: config[k] for k in SNAPSHOT_KEYS}
        shown = snapshot or current
        host = shown["host"]
        url_host = host if host != "0.0.0.0" else self.fleet.lan_ip() or "127.0.0.1"
        return {"running": snapshot is not None, "url": f"http://{url_host}:{shown['port']}" if snapshot else None,
                "host": host, "port": shown["port"], "require_key": shown["require_key"], "mcp": shown["mcp"],
                "models": self.fleet.loaded_ids(),
                "restart_required": (snapshot != current) if config["enabled"] else snapshot is not None,
                "error": self.error}

    def key_usage(self) -> dict:
        keys = self.keys
        return {item["name"]: item["last_used"] for item in keys.usage()} if keys else {}
