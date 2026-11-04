"""POST /v1/decisions (OpenAI's Decisions API): every question is decided as the System One request it stands for, so
the probabilities equal /v1/systemone's to the bit; answers come back in question order with typed choice values;
errors have OpenAI's shape. Through basal-serve's app with its batching worker and a stub engine."""
import asyncio
import re

import pytest

from tests.test_http_status import AppServer, on_app


class Weights:
    """run_shared of a backend: each option's weight is the "~w" written after it in the prompt (else 1), so answers
    depend on the options and their order."""
    policies = {}

    def __init__(self, error=None):
        self.error = error

    def run_shared(self, groups, policy=None):
        if self.error is not None:
            raise self.error
        out = []
        for prompts, ids in groups:
            rows = []
            for p, i in zip(prompts, ids):
                text = "".join(map(chr, p))
                w = [float(x) for x in re.findall(r"^[A-J]\. .*?~([0-9.]+)", text, re.M)][: len(i)]
                w = (w + [1.0] * len(i))[: len(i)]
                rows.append([x / sum(w) for x in w])
            out.append(rows)
        return out


STATE = "Klient: zapłaciłem dwa razy za zamówienie 4471."
OPENAI = {"model": "basal", "input": STATE, "questions": [
    {"type": "predicate", "name": "angry", "instructions": "Is the customer angry?"},
    {"type": "choice", "name": "team", "instructions": "Which team?",
     "choices": [{"value": "billing", "description": "payments ~3"}, {"value": "shipping ~1"}, {"value": True}]},
    {"type": "score", "instructions": "How urgent?",
     "levels": [{"label": "low ~1"}, {"label": "high", "description": "today ~2"}]}]}
SYSTEMONE = {"state": STATE, "questions": {
    "a": {"type": "noul", "instructions": "Is the customer angry?"},
    "t": {"type": "choice", "instructions": "Which team?",
          "criteria": {"billing": "payments ~3", "shipping ~1": None, "true": None}},
    "s": {"type": "score", "instructions": "How urgent?", "criteria": ["low ~1", "high: today ~2"]}}}


def both(srv, decisions, systemone):
    async def go(c):
        return (await c.post("/v1/decisions", json=decisions)), (await c.post("/v1/systemone", json=systemone))
    return asyncio.run(on_app(srv, go))


def post_decisions(srv, body=None, content=None):
    async def go(c):
        return await (c.post("/v1/decisions", content=content) if content is not None
                      else c.post("/v1/decisions", json=body))
    return asyncio.run(on_app(srv, go))


def test_the_same_probabilities_as_systemone_to_the_bit():
    d, s = both(AppServer(Weights()), OPENAI, SYSTEMONE)
    assert d.status_code == s.status_code == 200
    got, want = d.json(), s.json()["answers"]
    a, t, sc = got["answers"]
    assert a == {"type": "predicate", "name": "angry", "probability": want["a"]["noul"]}
    assert t["type"] == "choice" and t["name"] == "team" and t["choice"] == "billing"
    assert t["probabilities"] == [{"value": "billing", "probability": want["t"]["probabilities"]["billing"]},
                                  {"value": "shipping ~1", "probability": want["t"]["probabilities"]["shipping ~1"]},
                                  {"value": True, "probability": want["t"]["probabilities"]["true"]}]
    assert t["confidence"] == want["t"]["confidence"]
    assert sc == {"type": "score", "name": None, "score": want["s"]["score"], "confidence": want["s"]["confidence"],
                  "probabilities": [{"label": "low ~1", "probability": want["s"]["probabilities"]["0"], "value": 0},
                                    {"label": "high", "probability": want["s"]["probabilities"]["1"], "value": 1}]}
    assert got["model"] == "basal-test" and got["usage"]["input_tokens"] == s.json()["usage"]["input_tokens"] > 0
    assert got["usage"]["output_tokens"] == 0 and got["usage"]["total_tokens"] == got["usage"]["input_tokens"]


def test_a_boolean_choice_wins_as_a_boolean():
    body = {"input": "x", "questions": [{"type": "choice", "instructions": "Pay?",
                                         "choices": [{"value": True, "description": "yes ~9"}, {"value": False}]}]}
    r = post_decisions(AppServer(Weights()), body)
    assert r.status_code == 200 and r.json()["answers"][0]["choice"] is True


def test_a_message_list_input_is_the_json_state():
    msgs = [{"role": "user", "content": "I was charged twice."}, {"role": "assistant", "content": "Sorry!"}]
    d, s = both(AppServer(Weights()), dict(OPENAI, input=msgs), dict(SYSTEMONE, state=msgs))
    assert d.json()["answers"][0]["probability"] == s.json()["answers"]["a"]["noul"]


@pytest.mark.parametrize("body,content,param,message", [
    (None, b"{not json", None, "not valid JSON"),
    (["a list"], None, None, "must be a JSON object"),
    ({"questions": OPENAI["questions"]}, None, "input", "'input'"),
    ({"input": "x"}, None, "questions", "1 to 400"),
    ({"input": "x", "questions": []}, None, "questions", "1 to 400"),
    ({"input": "x", "questions": [{"type": "multi", "instructions": "?"}]}, None, "questions[0].type", "one of"),
    ({"input": "x", "questions": [{"type": "predicate"}]}, None, "questions[0].instructions", "'instructions'"),
    ({"input": "x", "questions": [{"type": "choice", "instructions": "?", "choices": [{"value": "a"}]}]}, None,
     "questions[0].choices", "2 to 255"),
    ({"input": "x", "questions": [{"type": "choice", "instructions": "?",
                                   "choices": [{"value": "true"}, {"value": True}]}]}, None,
     "questions[0].choices[1]", "unique"),
    ({"input": "x", "questions": [{"type": "choice", "instructions": "?",
                                   "choices": [{"value": 1}, {"value": "b"}]}]}, None, "questions[0].choices[0]",
     "string or boolean"),
    ({"input": "x", "questions": [{"type": "score", "instructions": "?", "levels": [{"label": "a"}, {"x": 1}]}]},
     None, "questions[0].levels[1]", "'label'"),
    ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}],
      "questions": OPENAI["questions"]}, None, "input", "images"),
    ({"input": "x", "questions": [{"type": "choice", "instructions": "?",
                                   "choices": [{"value": f"o{i}"} for i in range(11)]}]}, None, None,
     "11 options (supported: 2..10)"),  # the model's own limit (a prompt-format-1 stub here)
])
def test_invalid_requests_answer_400_in_openai_shape(body, content, param, message):
    srv = AppServer(Weights())
    r = post_decisions(srv, body, content)
    e = r.json()["error"]
    assert r.status_code == 400 and e["type"] == "invalid_request_error" and message in e["message"]
    assert e["param"] == param and e["code"] is None and set(e) == {"message", "type", "param", "code"}
    assert srv.load() == {"running": 0, "queued": 0}


def test_server_errors_answer_500_in_openai_shape():
    r = post_decisions(AppServer(Weights(RuntimeError("engine failed"))), OPENAI)
    assert r.status_code == 500
    assert r.json() == {"error": {"message": "engine failed", "type": "server_error", "param": None, "code": None}}


def test_the_openai_sdk_reads_the_answer():
    openai = pytest.importorskip("openai")
    decision = getattr(openai.types, "Decision", None)
    if decision is None:
        pytest.skip("this openai package has no Decisions API")
    r = post_decisions(AppServer(Weights()), OPENAI)
    d = decision.model_validate(r.json())
    assert [a.type for a in d.answers] == ["predicate", "choice", "score"] and d.answers[1].choice == "billing"


def test_a_deeply_nested_input_is_no_server_error():
    """Valid JSON nested 900 levels (under the parser's limit): the image check walked it recursively and failed with
    a 500; it is decided (the state is the input's JSON), and an image part at the bottom is still found."""
    deep = b'{"a":' * 900 + b"1" + b"}" * 900
    q = b', "questions": [{"type": "predicate", "instructions": "Paid?"}]}'
    ok = post_decisions(AppServer(Weights()), content=b'{"input": [' + deep + b"]" + q)
    assert ok.status_code == 200 and len(ok.json()["answers"]) == 1
    image = b'{"type": "input_image", "image_url": "data:image/png;base64,AA"}'
    bad = post_decisions(AppServer(Weights()), content=b'{"input": [' + b"[" * 900 + image + b"]" * 900 + b"]" + q)
    assert bad.status_code == 400 and "images" in bad.json()["error"]["message"]
