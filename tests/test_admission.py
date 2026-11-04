"""Admission (Server.admit): every check of a request that needs no engine work runs before any of it, so an invalid
request answers 422 (400 on /v1/decisions) with the engine, the facts and the media decoder untouched, never a 500 or
a 422 after the run. Through basal-serve's app with a counting stub engine."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from test_http_status import BODY, AppServer, Engine

from basal import server as server_module
from basal.request import MAX_DEPTH, check_json
from basal.prompt import RequestError
from basal.server import http_app


class Counting(Engine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def run_shared(self, groups, policy=None):
        self.calls += 1
        return super().run_shared(groups, policy)


def srv_with(**kw):
    srv = AppServer(Counting())
    for k, v in kw.items():
        setattr(srv, k, v)
    return srv


async def _post(srv, path, content):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        transport = httpx.ASGITransport(app=http_app(srv, "fast"))
        async with httpx.AsyncClient(transport=transport, base_url="http://basal") as c:
            return await c.post(path, content=content, headers={"content-type": "application/json"})
    finally:
        worker.cancel()


def post(srv, body, path="/v1/systemone"):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return asyncio.run(_post(srv, path, content))


def raw(body_text):
    return body_text.encode("ascii")


CHOICE = '{"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}'

# (a JSON body as sent, what the 422 says): each was a 500 (the tokenizer, the response's JSON) or ran the engine first
BAD = [
    (raw('{"state": "Invoice \\udead 1240", "questions": {"q": ' + CHOICE + '}}'), "lone surrogate"),
    (raw('{"state": "x", "questions": {"q\\ud800": ' + CHOICE + '}}'), "lone surrogate"),
    (raw('{"state": "x", "questions": {"q": {"type": "choice", "instructions": "\\udfff?", "criteria": ["a", "b"]}}}'),
     "lone surrogate"),
    (raw('{"state": "x", "questions": {"q": {"type": "choice", "criteria": ["ok", Infinity]}}}'), "not a finite"),
    (raw('{"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": NaN}}}}'), "not a finite"),
    (raw('{"state": ' + "[" * (MAX_DEPTH + 1) + "]" * (MAX_DEPTH + 1) + ', "questions": {"q": ' + CHOICE + '}}'),
     "nests"),
]


@pytest.mark.parametrize("body,said", BAD)
def test_a_body_the_engine_cannot_take_is_refused_before_the_run(body, said):
    srv = srv_with()
    r = post(srv, body)
    assert r.status_code == 422 and said in r.json()["error"]
    assert srv.backend.calls == 0


def test_a_decisions_body_nested_beyond_the_readers_answers_400():
    srv = srv_with()
    deep = "[" * 900 + '"x"' + "]" * 900  # (Python's parser reads it; the recursive reading after it did not)
    r = post(srv, raw('{"input": ' + deep + ', "questions": [{"type": "predicate", "instructions": "Q?"}]}'),
             "/v1/decisions")
    assert r.status_code == 400 and "nests" in r.json()["error"]["message"] and srv.backend.calls == 0


def test_numbers_in_the_state_stay_as_they_are():
    srv = srv_with()
    r = post(srv, raw('{"state": {"x": NaN, "y": Infinity}, "questions": {"q": ' + CHOICE + '}}'))
    assert r.status_code == 200 and srv.backend.calls == 1


def test_a_valid_request_is_answered_as_before():
    srv = srv_with()
    r = post(srv, BODY)
    assert r.status_code == 200 and r.json()["answers"]["q"]["choice"] in ("a", "b") and srv.backend.calls == 1


@pytest.mark.parametrize("q,said", [
    ({"type": "multi", "criteria": ["a", "b"], "evidence": True}, "not supported for multi"),
    ({"type": "noul", "evidence": True}, "evidence is not available"),
])
def test_evidence_that_cannot_be_given_is_refused_before_the_run(q, said):
    srv = srv_with(evidence=SimpleNamespace(available=True) if q["type"] == "multi" else None)
    r = post(srv, {"state": "x", "questions": {"q": q}})
    assert r.status_code == 422 and said in r.json()["error"] and srv.backend.calls == 0


def test_task_questions_count_their_branches_before_the_run():
    srv = srv_with(max_branches=4)
    claims = [f"claim {i}" for i in range(5)]  # verify: one question per claim
    r = post(srv, {"state": "x", "questions": {"v": {"type": "verify", "claims": claims}}})
    assert r.status_code == 422 and "--max-branches 4" in r.json()["error"] and srv.backend.calls == 0
    r = post(srv, {"state": "x", "questions": {"v": {"type": "verify", "claims": claims[:4]}}})
    assert r.status_code == 200 and srv.backend.calls > 0


def test_a_rank_request_with_an_invalid_question_runs_nothing():
    srv = srv_with()
    rank_q = {"type": "rank", "instructions": "Does {candidate} answer it?", "candidates": ["a", "b", "c"]}
    r = post(srv, {"state": "x", "questions": {"best": rank_q, "bad": {"type": "choice", "criteria": ["only"]}}})
    assert r.status_code == 422 and "'bad'" in r.json()["error"] and srv.backend.calls == 0


def test_a_bad_field_is_refused_before_the_facts_and_the_media(monkeypatch):
    calls = []
    monkeypatch.setattr(server_module, "inject_facts", lambda text: calls.append("facts") or text)
    import basal.media as media
    monkeypatch.setattr(media, "decode", lambda *a, **k: calls.append("decode") or ([], [], []))
    srv = srv_with()
    srv.backend.media_kinds = ("image",)
    image = [{"type": "image", "data": "aGVsbG8="}]
    for body in ({"state": "x", "facts": "auto", "lang": "de", "questions": {"q": json.loads(CHOICE)}},
                 {"state": "x", "media": image, "lang": "de", "questions": {"q": json.loads(CHOICE)}},
                 {"state": "x", "media": image, "questions": {"q": json.loads(CHOICE), "r": {"type": "choice"}}}):
        r = post(srv, body)
        assert r.status_code == 422
    srv.layout = "qfirst"
    r = post(srv, {"state": "x", "media": image, "questions": {"q": json.loads(CHOICE)}})
    assert r.status_code == 422 and "canonical layout" in r.json()["error"]
    assert calls == [] and srv.backend.calls == 0


def test_check_json_is_iterative_and_names_the_place():
    deep = []
    for _ in range(5000):
        deep = [deep]
    with pytest.raises(RequestError, match="nests"):
        check_json({"state": deep})
    with pytest.raises(RequestError, match=r"questions\['q'\]\['criteria'\]\[1\]"):
        check_json({"state": "x", "questions": {"q": {"criteria": ["a", float("inf")]}}})
    check_json({"state": float("nan"), "questions": {"q": {"criteria": ["a", "b"]}}})  # the state is data
