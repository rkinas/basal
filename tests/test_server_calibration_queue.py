"""Regression tests for fix/server-calibration-queue (Z-1082)."""
import asyncio
import json
import math
import sys
import threading

import pytest

from basal import server as S
from basal.server import Server, http_app
from basal.server import check_calibration


def refused(cal):
    try:
        check_calibration(cal, 1, "CAL.json")
        return False
    except SystemExit:
        return True


@pytest.mark.parametrize("T", [0, 0.0, -1.0, float("nan"), float("inf"), True, "2"])
def test_i09_a_temperature_that_is_not_a_finite_number_above_zero_is_refused_at_start_up(T):
    assert refused({"prompt": 1, "temperature_per_prim": {"choice": T}})
    assert refused({"prompt": 1, "temperature_per_bucket": {"choice": {"2-10": T}}})


@pytest.mark.parametrize("cal", [{"temperature_per_prim": {"choice": 1.5}}, {"temperature_per_prim": {"choice": 2}},
                                 {"temperature_per_bucket": {"choice": {"2-10": 2.0}}}, {}])
def test_i09_a_valid_calibration_is_accepted(cal):
    assert not refused({"prompt": 1, **cal})


def test_p6_a_missing_api_keys_file_is_warned_about_at_start_up(monkeypatch, tmp_path, capsys):
    def stop(*args):
        raise SystemExit("stop after the warning")
    monkeypatch.setattr(S, "refuse_lean_flex", stop)
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--api-keys", str(tmp_path / "keys.json")])
    try:
        S.main()
    except SystemExit:
        pass
    assert "no such file" in capsys.readouterr().err


class Tok:
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        ids = [[ord(c) for c in s] for s in text] if isinstance(text, list) else [ord(c) for c in text]
        return type("E", (), {"input_ids": ids})()


class CountingEngine:
    policies = {}

    def __init__(self):
        self.groups, self.entered, self.go = 0, threading.Event(), threading.Event()
        self.go.set()

    def run_shared(self, groups, policy=None, **kw):
        self.entered.set()
        self.go.wait(10)
        self.groups += len(groups)
        return [[[1 / len(x)] * len(x) for x in g[1]] for g in groups]


class AppServer(Server):
    def __init__(self, engine):
        self.name, self.temps, self.tok, self.letters, self.pos_probs = "basal-test", {}, Tok(), {}, [0.5, 0.5]
        self.orders, self.soam, self.log, self.cal = 2, True, None, {}
        self.fmt, self.max_options = 1, 10
        self.tau, self.band = 0.6, (0.2, 0.8)
        self.entries = 0
        self.backend, self.max_batch, self.wait, self.long_tokens = engine, 8, 0.0, None
        self.context, self.overflow, self.trunc_head = (8192, "model context"), "refuse", 1 / 3
        self.cal_mismatch = None
        self.queue = asyncio.Queue()

    def probs_for(self, p, i):
        return self.pos_probs[:len(i)]

async def post(app, body):
    sent, done = [], False
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "path": "/v1/systemone",
             "raw_path": b"/v1/systemone", "query_string": b"", "server": ("x", 80), "client": ("c", 1),
             "headers": [(b"content-type", b"application/json")]}

    async def receive():
        nonlocal done
        if not done:
            done = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    async def send(m):
        sent.append(m)
    await app(scope, receive, send)
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


BODY = {"state": "Invoice 1240 PLN.", "questions": {"q": {"type": "choice", "instructions": "Which?",
                                                           "criteria": {"a": "Alpha", "b": "Beta"}}}}


def test_i14_a_request_abandoned_in_the_queue_is_not_run_on_the_engine():
    srv = AppServer(CountingEngine())
    body = json.dumps(BODY).encode()

    async def scenario():
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            app = http_app(srv, "fast")
            srv.backend.go.clear()
            first = asyncio.create_task(post(app, body))
            for _ in range(100):
                await asyncio.sleep(0.02)
                if srv.backend.entered.is_set():
                    break
            dead = [asyncio.create_task(post(app, body)) for _ in range(10)]
            live = [asyncio.create_task(post(app, body)) for _ in range(3)]
            await asyncio.sleep(0.3)
            for t in dead:
                t.cancel()
            await asyncio.sleep(0.1)
            srv.backend.go.set()
            res = await asyncio.gather(first, *live)
            await asyncio.sleep(0.3)
            return [r[0] for r in res]
        finally:
            worker.cancel()
    assert asyncio.run(scenario()) == [200] * 4
    assert srv.backend.groups == 4  # the first and the three that waited; not the ten that gave up


class Log:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


OUT = {"answers": {"q": {"type": "noul", "probabilities": [0.5, 0.5]}}}
Q = {"q": {"type": "noul", "question": "ok?"}}


def logged(body, out=OUT):
    srv = Server.__new__(Server)
    srv.name, srv.log = "m", Log()
    srv.log_decision(body, out)
    return srv.log.lines[-1]


def key(body):
    return json.loads(logged(body))["request_sha256"]


def test_s21_two_images_with_the_same_text_have_different_keys():
    a = key({"state": "same", "questions": Q, "media": [{"type": "image", "data": "AAAA"}]})
    b = key({"state": "same", "questions": Q, "media": [{"type": "image", "data": "BBBB"}]})
    assert a != b


def test_s21_a_different_language_is_a_different_key():
    assert key({"state": "same", "questions": Q, "lang": "pl"}) != key({"state": "same", "questions": Q, "lang": "en"})


def test_s21_a_nan_probability_is_logged_as_strict_json():
    nan = {"answers": {"q": {"type": "noul", "probabilities": [math.nan, 1.0]}}}
    json.loads(logged({"state": "s", "questions": Q}, nan), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_s21_a_request_without_media_or_language_keeps_its_key():
    assert key({"state": "s", "questions": Q}) == key({"state": "s", "questions": Q})
