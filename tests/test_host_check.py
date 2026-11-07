"""DNS rebinding: a server bound to a loopback host without --api-keys answers only requests that name a loopback host
(or one of --allowed-hosts) and that no other site's page sent; 403 before the app reads them. A server bound to the
network, or with --api-keys, is not checked unless --allowed-hosts says so."""
import asyncio

import httpx
import pytest
from starlette.testclient import TestClient
from test_http_status import BODY, AppServer, Engine

from basal import router as R
from basal.auth import LOOPBACK_NAMES, allowed_hosts, loopback
from basal.server import parser, served_app


def test_which_servers_are_checked():
    assert loopback("127.0.0.1") and loopback("127.0.0.2") and loopback("::1") and loopback("localhost")
    assert not loopback("0.0.0.0") and not loopback("192.168.0.10") and not loopback("example.com")
    assert allowed_hosts("127.0.0.1") == set(LOOPBACK_NAMES) and allowed_hosts("0.0.0.0") is None
    assert allowed_hosts("127.0.0.1", api_keys="keys.json") is None              # a page has no key
    assert allowed_hosts("0.0.0.0", "basal.example, B.example") == {"basal.example", "b.example", *LOOPBACK_NAMES}
    assert allowed_hosts("127.0.0.1", "*") is None
    assert parser().parse_args([]).allowed_hosts is None


def requests(args, calls):
    a = parser().parse_args(["--mode", "eager", *args])
    eng = Engine()
    real = eng.run_shared
    eng.run_shared = lambda groups, policy=None: calls.append(1) or real(groups, policy)
    srv = AppServer(eng)
    app = served_app(srv, a)

    async def go(headers):
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as c:
                return (await c.post("/v1/systemone", json=BODY, headers=headers)).status_code, \
                    (await c.get("/health", headers=headers)).status_code
        finally:
            worker.cancel()
    return lambda headers: asyncio.run(go(headers))


@pytest.mark.parametrize("headers,status", [
    ({}, 200),
    ({"host": "localhost:8000"}, 200),
    ({"host": "[::1]:8000"}, 200),
    ({"host": "attacker.example"}, 403),                                     # a rebound name
    ({"host": "attacker.example:8000"}, 403),
    ({"origin": "http://attacker.example"}, 403),                            # another site's page
    ({"origin": "null"}, 403),
    ({"origin": "http://localhost:3000"}, 200),                              # a local page
])
def test_a_loopback_server_without_keys_answers_only_its_own_host(headers, status):
    calls = []
    post = requests(["--host", "127.0.0.1"], calls)
    got, health = post(headers)
    assert got == status and health == 200 and len(calls) == (1 if status == 200 else 0)


def test_a_probe_of_health_is_answered_whatever_host_it_names():
    """GET and HEAD of /health and /health/ stay open, as the API keys leave them (a load balancer or a proxy probes
    with a host of its own); every other route is still checked."""
    a = parser().parse_args(["--mode", "eager", "--host", "127.0.0.1"])
    srv = AppServer(Engine())
    app = served_app(srv, a)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as c:
            h = {"host": "basal.internal"}
            return [(await c.request(m, p, headers=h)).status_code for m in ("GET", "HEAD") for p in ("/health",
                                                                                                    "/health/")], \
                (await c.get("/v1/models", headers=h)).status_code
    probes, other = asyncio.run(go())
    assert probes == [200, 200, 200, 200] and other == 403


def test_allowed_hosts_and_cors_origins_widen_it_and_a_network_bind_is_not_checked():
    calls = []
    assert requests(["--host", "127.0.0.1", "--allowed-hosts", "basal.lan"], calls)({"host": "basal.lan"})[0] == 200
    assert requests(["--host", "127.0.0.1", "--cors-origins", "http://app.example"], calls)(
        {"origin": "http://app.example"})[0] == 200
    assert requests(["--host", "0.0.0.0"], calls)({"host": "attacker.example"})[0] == 200
    assert requests(["--host", "127.0.0.1", "--allowed-hosts", "*"], calls)({"host": "attacker.example"})[0] == 200


def test_the_router_checks_the_host_too():
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"answers": {}})
    rt = R.Router(["http://main"], transport=httpx.MockTransport(handler))
    with TestClient(R.app(rt, poll_s=0.05, hosts=allowed_hosts("127.0.0.1")), base_url="http://127.0.0.1:8000") as c:
        asyncio.run(rt.poll())
        ok = c.post("/v1/systemone", json=BODY)
        bad = c.post("/v1/systemone", json=BODY, headers={"host": "attacker.example"})
    assert ok.status_code == 200 and bad.status_code == 403 and "--allowed-hosts" in bad.json()["error"]
