"""A request whose HTTP client disconnects is cancelled (basal-serve, basal-router, /mcp): uvicorn does not cancel a
handler when its client goes away, so without this its work ran in full for nobody. No torch here: basal-router
imports this module alone, as it does basal.metrics."""
import asyncio
import json

CLIENT_GONE = ({"error": "client disconnected", "code": "CLIENT_DISCONNECTED"}, 499)  # (nobody reads it)


async def until_disconnect(request, coro):
    """The result of `coro` (the request's serve), or None when the HTTP client disconnects first. The serve task is
    then cancelled, so the request's entries still in the queue are dropped before they reach the engine (ask cancels
    their futures, the workers skip done futures); a batch already on the GPU finishes. Without it a client that timed
    out left its request to run in full: 48 abandoned 6k-token requests kept the GPU busy for 74 s (#129).
    The disconnect is the server's http.disconnect message, awaited by a task of its own (the handler has read the
    body, so nothing else receives): a waiting request costs the event loop nothing and is cancelled at once, where
    polling is_disconnected() every 0.25 s cost 15% of a core with 1,024 requests waiting. A serve that has ended
    answers as it ended (its error raised, not lost); a cancelled handler cancels its serve too."""
    task = asyncio.ensure_future(coro)
    gone = asyncio.ensure_future(_disconnected(request.receive))
    try:
        await asyncio.wait({task, gone}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()
        task.cancel()  # (nothing once it has ended) the client is gone, or this handler was cancelled
    await asyncio.wait({task})  # the serve's own clean-up first: ask drops the request's queued entries
    return None if task.cancelled() else task.result()


async def _disconnected(receive):
    """Returns when the server says the request's client is gone (http.disconnect). A receive that fails says
    nothing: the request is then served to the end, as without the watch."""
    try:
        while (await receive())["type"] != "http.disconnect":
            pass
    except Exception:  # noqa: BLE001
        await asyncio.get_running_loop().create_future()  # (never set: cancelled when the serve ends)


class CancelOnDisconnect:
    """An ASGI app that reads its own request (the MCP endpoint), cancelled when its client disconnects before its
    response is complete: the server's messages reach it through a task of ours, which sees the http.disconnect.
    A stateless MCP request's session ends with its handler, and the tool call with it. When no response has started,
    the 499 of a request whose client left is sent (nobody reads it)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbox, state = asyncio.Queue(), {"started": False, "complete": False}

        async def sent(message):
            if message["type"] == "http.response.start":
                state["started"] = True
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                state["complete"] = True  # (the server's http.disconnect after it says nothing about the client)
            await send(message)

        task = asyncio.ensure_future(self.app(scope, inbox.get, sent))

        async def relay():
            while True:
                message = await receive()
                inbox.put_nowait(message)
                if message["type"] == "http.disconnect":
                    if not state["complete"]:
                        task.cancel()
                    return

        pump = asyncio.ensure_future(relay())
        try:
            await asyncio.wait({task})
        finally:
            pump.cancel()
            task.cancel()  # (nothing once it has ended) or this handler was cancelled
        if not task.cancelled():
            task.result()
        elif not state["started"]:
            body, status = CLIENT_GONE
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": json.dumps(body).encode()})
