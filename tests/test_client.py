"""basal.client: the request body of every question type, client defaults and request fields, the server's status codes
as exceptions (a retry on 503 only), and AsyncBasal end to end through the app of basal-serve (http_app) with its
batching worker and a stub engine. Then against other /v1/systemone servers of the same request shape (answers in the
shapes of vLLM's structured-diffusion example): the request bodies, the bearer token, noul answers that carry only
"noul", and that basal-serve's answers come back unchanged."""
import asyncio
import json

import httpx
import pytest
import torch

from basal import client as C
from basal.client import AsyncBasal, Basal, InvalidRequest, OutOfMemory, ServerError, complete
from basal.server import to_items
from tests.test_http_status import AppServer, Engine
from tests.test_server import FakeServer

STATE = "Invoice 1240 PLN, due 14 days after delivery."


def recording(answers=None, statuses=()):
    """A Basal on a MockTransport: the bodies it sent, and the given statuses (then 200) as responses."""
    sent, statuses = [], list(statuses)

    def handler(request):
        sent.append(json.loads(request.content) if request.content else request.url.path)
        if statuses:
            status = statuses.pop(0)
            return httpx.Response(status, json={"error": f"status {status}"})
        body = sent[-1]
        names = body["questions"] if isinstance(body, dict) else []
        return httpx.Response(200, json={"model": "m", "usage": {}, "answers": answers or {n: {"n": n} for n in names}})
    return sent, handler


def client(handler, **kw):
    return Basal("http://basal/", http=httpx.Client(transport=httpx.MockTransport(handler)), backoff=0, **kw)


@pytest.mark.parametrize("call,question", [
    (lambda b: b.choice(STATE, "Which?", {"a": "A", "b": "B"}, option_keys="hide"),
     {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}, "option_keys": "hide"}),
    (lambda b: b.yes_no(STATE, "Late?"), {"type": "noul", "instructions": "Late?"}),
    (lambda b: b.yes_no(STATE, "Late?", true="after the due date"),
     {"type": "noul", "instructions": "Late?", "criteria": {"true": "after the due date"}}),
    (lambda b: b.score(STATE, "How urgent?", ["low", "high"]),
     {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}),
    (lambda b: b.multi(STATE, "Which apply?", ["vat", "late"], max=1),
     {"type": "multi", "instructions": "Which apply?", "criteria": ["vat", "late"], "max": 1}),
    (lambda b: b.act(STATE, "Pay?", {"true": "valid", "false": "invalid"}, {"wrong": 99, "defer": 1}, max_error=0.05),
     {"type": "act", "instructions": "Pay?", "criteria": {"true": "valid", "false": "invalid"},
      "costs": {"wrong": 99, "defer": 1}, "max_error": 0.05}),
    (lambda b: b.tool(STATE, [{"name": "pay", "inputSchema": {}}], rank=2),
     {"type": "tool", "tools": [{"name": "pay", "inputSchema": {}}], "rank": 2}),
    (lambda b: b.classify(STATE, {"invoice": "an invoice"}, none=True),
     {"type": "classify", "labels": {"invoice": "an invoice"}, "none": True}),
    (lambda b: b.verify(STATE, {"total": "The total is 1240 PLN."}),
     {"type": "verify", "claims": {"total": "The total is 1240 PLN."}}),
    (lambda b: b.compress(STATE, "Payment terms?", unit="sentence"),
     {"type": "compress", "instructions": "Payment terms?", "unit": "sentence"}),
    (lambda b: b.extract(STATE, {"type": "object", "properties": {"total": {"type": "number"}}}),
     {"type": "extract", "schema": {"type": "object", "properties": {"total": {"type": "number"}}}}),
])
def test_each_method_sends_its_question_and_returns_its_answer(call, question):
    sent, handler = recording()
    assert call(client(handler)) == {"n": "q"}
    assert sent == [{"state": STATE, "questions": {"q": question}}]


def test_defaults_and_fields_of_decide():
    sent, handler = recording()
    b = client(handler, facts="auto", lang="pl", early_exit="0.99")
    qs = {"dept": C.choice("Which?", ["a", "b"]), "late": C.yes_no("Late?")}
    assert b.decide(STATE, qs, lang="en", think=None, media=[{"type": "image", "data": "x"}]) == {
        "dept": {"n": "dept"}, "late": {"n": "late"}}
    assert sent[0] == {"state": STATE, "questions": qs, "facts": "auto", "lang": "en", "early_exit": "0.99",
                       "media": [{"type": "image", "data": "x"}]}
    assert b.request({"state": STATE, "questions": qs})["usage"] == {}


@pytest.mark.parametrize("status,cls", [(422, InvalidRequest), (503, OutOfMemory), (500, ServerError),
                                        (404, ServerError)])
def test_status_codes_are_exceptions(status, cls):
    _, handler = recording(statuses=[status])
    with pytest.raises(cls) as e:
        client(handler).yes_no(STATE, "Late?")
    assert str(e.value) == f"status {status}"  # the server's "error" text, not the JSON body
    assert e.value.status == status and isinstance(e.value, RuntimeError)  # the 2.4 client raised RuntimeError


def test_only_out_of_memory_is_retried():
    sent, handler = recording(statuses=[503, 503])
    assert client(handler, retries=2).yes_no(STATE, "Late?") == {"n": "q"} and len(sent) == 3
    sent, handler = recording(statuses=[503, 503])
    with pytest.raises(OutOfMemory):
        client(handler, retries=1).yes_no(STATE, "Late?")
    assert len(sent) == 2
    sent, handler = recording(statuses=[422])
    with pytest.raises(InvalidRequest):
        client(handler, retries=3).yes_no(STATE, "Late?")
    assert len(sent) == 1


def test_a_body_that_is_not_json_is_a_server_error():
    b = Basal("http://basal", http=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(502, text="Bad Gateway"))))
    with pytest.raises(ServerError) as e:
        b.decide(STATE, {"q": C.yes_no("Late?")})
    assert str(e.value) == "Bad Gateway" and e.value.status == 502


def on_app(srv, fn, **kw):
    async def go():
        worker = asyncio.get_running_loop().create_task(srv.worker())
        from basal.server import http_app
        try:
            async with AsyncBasal("http://basal", http=httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=http_app(srv, "fast")), base_url="http://basal"), **kw) as b:
                return await fn(b)
        finally:
            worker.cancel()
    return asyncio.run(go())


def test_async_client_through_the_server():
    async def go(b):
        one = await b.choice(STATE, "Which?", {"a": "Alpha", "b": "Beta"})
        both = await b.decide(STATE, {"dept": C.choice("Which?", ["a", "b"]), "late": C.yes_no("Late?")})
        return one, both, await b.health()
    one, both, health = on_app(AppServer(Engine()), go)
    assert one["type"] == "choice" and one["choice"] in ("a", "b") and set(one["probabilities"]) == {"a", "b"}
    assert both["dept"]["type"] == "choice" and both["late"]["type"] == "noul"
    assert health == {"status": "ok", "running": 0, "queued": 0}


EVERY_TYPE = {
    "choice": C.choice("Which?", {"a": "A", "b": "B"}),
    "noul": C.yes_no("Late?", true="after the due date"),
    "score": C.score("How urgent?", ["low", "mid", "high"]),
    "multi": C.multi("Which apply?", ["vat", "late", "other"], max=2),
    "act": C.act("Pay?", {"true": "valid", "false": "invalid"}, {"wrong": 99, "defer": 1}),
    "tool": C.tool([{"name": "pay", "description": "Pay it", "inputSchema": {"type": "object", "properties": {}}}],
                   "Next step?"),
    "classify": C.classify({"invoice": "an invoice", "letter": "a letter"}, none=True),
    "verify": C.verify({"total": "The total is 1240 PLN."}),
    "compress": C.compress("Payment terms?", unit="sentence"),
    "extract": C.extract({"type": "object", "properties": {"total": {"type": "number"}}}),
}


def test_every_question_type_is_a_request_the_server_answers():
    async def go(b):
        return {k: (await b.decide(STATE, {k: q}))[k]["type"] for k, q in EVERY_TYPE.items()}, await b.models()
    types, models = on_app(AppServer(Engine()), go)
    assert types == {k: k for k in EVERY_TYPE}
    assert models[0]["name"] == "basal-test"


def test_async_client_errors_through_the_server():
    async def invalid(b):
        return await b.choice(STATE, "Which?", {"a": "Alpha"})
    with pytest.raises(InvalidRequest) as e:
        on_app(AppServer(Engine()), invalid)
    assert "1 options" in str(e.value) and not str(e.value).startswith("{")

    async def oom(b):
        return await b.yes_no(STATE, "Late?")
    with pytest.raises(OutOfMemory) as e:
        on_app(AppServer(Engine(torch.OutOfMemoryError("CUDA out of memory"))), oom, retries=1, backoff=0)
    assert str(e.value) == "CUDA out of memory" and e.value.status == 503


@pytest.mark.parametrize("retries", [-1, 1.5, None])
def test_retries_must_be_a_count(retries):
    for cls in (Basal, AsyncBasal):
        with pytest.raises(ValueError, match="retries"):
            cls("http://basal", retries=retries)


BASAL = {"q": {"type": "noul", "noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}, "confidence": 0.8}}
OTHER = {  # the answer shapes of vLLM's examples/features/structured_diffusion/structured_server.py (jev_answer)
    "noul": {"type": "noul", "noul": 0.3},
    "choice": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.25, "tech": 0.75}, "confidence": 0.75},
    "score": {"type": "score", "score": 1.2, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
              "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.6},
}


def served(answers, seen, **kw):
    b = Basal("http://server", **kw)

    def handle(request):
        seen.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={"model": "m", "answers": answers})
    b.http = httpx.Client(transport=httpx.MockTransport(handle), headers=b.http.headers)
    return b


def test_basal_answers_come_back_unchanged_and_no_token_by_default():
    seen = []
    assert served(BASAL, seen).yes_no("x", "Urgent?") == BASAL["q"]
    assert "authorization" not in seen[0][0]
    # yes_no without descriptions sends no "criteria" (#58; 2.4 sent an empty one)
    assert seen[0][1] == {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}


def test_bearer_token():
    seen = []
    served(BASAL, seen, api_key="k1").yes_no("x", "Urgent?")
    assert seen[0][0]["authorization"] == "Bearer k1"


def test_noul_with_only_p_gets_probabilities_and_confidence():
    seen = []
    ans = served({"q": OTHER["noul"]}, seen).yes_no("x", "Urgent?")
    assert ans == {"type": "noul", "noul": 0.3, "probabilities": {"true": 0.3, "false": 0.7}, "confidence": 0.7}
    assert complete(None) is None and complete(OTHER["choice"]) is OTHER["choice"]  # a skipped question, a full answer
    assert served({"q": OTHER["score"]}, []).score("x", "Tone?", ["calm", "annoyed", "furious"]) == OTHER["score"]


def test_choice_list_criteria_are_sent_as_a_map():
    seen = []
    served({"q": OTHER["choice"]}, seen).choice("x", "Which team?", ["billing", "tech"])
    assert seen[0][1]["questions"]["q"]["criteria"] == {"billing": None, "tech": None}
    # basal-serve reads both forms as the same options
    same = [to_items("x", {"q": {"type": "choice", "instructions": "Which team?", "criteria": c}})[0]
            for c in (["billing", "tech"], {"billing": None, "tech": None})]
    assert same[0] == same[1]


def test_basal_serve_answers_the_other_servers_request_example():
    """The /v1/systemone request of vLLM's example (a "model" field and a state object): basal-serve answers it, with
    the noul fields that example returns ("type", "noul") and more."""
    body = {"model": "any", "state": {"ticket": "Everything is down and we have a demo at noon."},
            "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}
    ans = asyncio.run(FakeServer([0.9, 0.1])._run(body))["answers"]["urgent"]
    assert ans["type"] == "noul" and 0 <= ans["noul"] <= 1 and set(ans["probabilities"]) == {"true", "false"}


def test_errors_raise_with_the_body():
    b = Basal("http://server")
    b.http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(422, json={"error": {"message": "bad", "type": "validation_error"}})))
    with pytest.raises(RuntimeError, match="validation_error"):
        b.yes_no("x", "Urgent?")
