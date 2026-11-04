"""Regression tests for fix/client-responses (Z-1089)."""
import asyncio

import httpx
import pytest

from basal import tasks
from basal.client import AsyncBasal, Basal, ServerError


def sync(handler):
    return Basal("http://basal/", http=httpx.Client(transport=httpx.MockTransport(handler)), backoff=0)


def test_cl3_a_200_reply_that_is_not_json_is_a_server_error():
    c = sync(lambda request: httpx.Response(200, text="<html>502 bad gateway</html>"))
    with pytest.raises(ServerError, match="bad gateway"):
        c.request({"state": "s", "questions": {}})


@pytest.mark.parametrize("reply", [{}, {"answers": []}, {"answers": None}, []])
def test_cl3_a_reply_without_answers_is_a_server_error(reply):
    c = sync(lambda request: httpx.Response(200, json=reply))
    with pytest.raises(ServerError):
        c.decide("s", {"q": {"type": "yesno", "instructions": "Late?"}})


def test_cl3_the_async_client_does_the_same():
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>no</html>")))
    c = AsyncBasal("http://basal/", http=http, backoff=0)
    with pytest.raises(ServerError):
        asyncio.run(c.request({"state": "s", "questions": {}}))


def test_p1_the_typeset_minus_is_a_minus():
    assert tasks.parse_number("Temperatura: \u22125 stopni") == -5
    assert tasks.parse_number("\u22123,5") == -3.5


def test_p1_the_ascii_minus_and_a_plain_number_are_unchanged():
    assert tasks.parse_number("-5") == -5 and tasks.parse_number("5") == 5
