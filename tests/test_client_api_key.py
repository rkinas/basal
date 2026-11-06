"""The API key of a Basal / AsyncBasal client goes with each of its requests, never into the headers of an httpx
client it shares: there it went with every request of every client on that http, to any host."""
import asyncio

import httpx

from basal.client import AsyncBasal, Basal


def seen_with(auth):
    def handler(request):
        auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"answers": {}})
    return handler


def test_two_clients_on_one_http_each_send_their_own_key():
    auth = []
    http = httpx.Client(transport=httpx.MockTransport(seen_with(auth)))
    a = Basal("http://basal/", http=http, api_key="key-a", backoff=0)
    b = Basal("http://basal/", http=http, api_key="key-b", backoff=0)
    plain = Basal("http://basal/", http=http, backoff=0)
    a.request({"state": "s", "questions": {}})
    b.request({"state": "s", "questions": {}})
    plain.request({"state": "s", "questions": {}})
    assert auth == ["Bearer key-a", "Bearer key-b", None]
    assert "authorization" not in http.headers


def test_two_async_clients_on_one_http_each_send_their_own_key():
    auth = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(seen_with(auth)))
    a = AsyncBasal("http://basal/", http=http, api_key="key-a", backoff=0)
    b = AsyncBasal("http://basal/", http=http, api_key="key-b", backoff=0)
    asyncio.run(a.request({"state": "s", "questions": {}}))
    asyncio.run(b.request({"state": "s", "questions": {}}))
    assert auth == ["Bearer key-a", "Bearer key-b"]
    assert "authorization" not in http.headers
