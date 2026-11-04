"""Regression tests for fix/router-client (Z-1082)."""
import json

import pytest

from basal import client as C


class Reply:
    """What the code reads of an httpx response."""
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.headers = status, {k.lower(): v for k, v in (headers or {}).items()}
        self.content = json.dumps(body if body is not None else {}).encode()
        self.text = self.content.decode()

    def json(self):
        return json.loads(self.content)


OK = Reply(200, {"answers": {"q": {"answer": "a", "confidence": 0.9}}, "model": "m", "usage": {}})
STARTING = Reply(503, {"status": "starting"}, {"retry-after": "5"})
NO_REPLICA = Reply(503, {"error": "no basal replica is ready"})
OOM = Reply(503, {"error": "CUDA out of memory", "code": "OUT_OF_MEMORY"})


class Http:
    def __init__(self, replies):
        self.replies, self.calls, self.headers = list(replies), 0, {}

    def request(self, method, url, **kw):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def attempt(monkeypatch, replies, retries=2):
    sleeps = []
    monkeypatch.setattr(C.time, "sleep", sleeps.append)
    http = Http(replies)
    try:
        C.Basal(http=http, retries=retries, backoff=0.5).request({"state": "x", "questions": {}})
        return "ok", http.calls, sleeps
    except C.BasalError as e:
        return e, http.calls, sleeps


def test_i11_a_server_still_starting_is_not_out_of_memory(monkeypatch):
    err, calls, _ = attempt(monkeypatch, [STARTING] * 3)
    assert not isinstance(err, C.OutOfMemory) and calls == 3


def test_i11_the_retries_wait_at_least_retry_after(monkeypatch):
    _, _, sleeps = attempt(monkeypatch, [STARTING] * 3)
    assert sleeps and sleeps[0] >= 5


def test_i11_a_router_with_no_ready_replica_is_not_out_of_memory(monkeypatch):
    err, _, _ = attempt(monkeypatch, [NO_REPLICA] * 3)
    assert not isinstance(err, C.OutOfMemory)


def test_i11_real_out_of_memory_is_still_retried_with_the_backoff(monkeypatch):
    err, calls, sleeps = attempt(monkeypatch, [OOM] * 3)
    assert isinstance(err, C.OutOfMemory) and calls == 3 and sleeps == [0.5, 1.0]
    assert attempt(monkeypatch, [STARTING, OK])[:2] == ("ok", 2)
