import argparse
import base64
import hashlib

import pytest
from starlette.testclient import TestClient

from basal.server import build_app


class FakeServer:
    name, backend, temps, fmt, orders = "served-model", type("Backend", (), {"policies": {}, "media_kinds": ()})(), {}, 2, "auto"
    long_tokens = None

    def __init__(self, fail=None):
        self.fail = fail
        self.requests = []

    async def decide(self, body):
        self.requests.append(body)
        if self.fail == "value":
            raise ValueError("bad request")
        if self.fail == "key":
            raise KeyError("questions")
        if self.fail == "type":
            raise TypeError("bad criteria")
        if self.fail:
            raise RuntimeError("broken")
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"noul": .8, "probabilities": {"true": .8, "false": .2}, "confidence": .8}
            elif q["type"] == "score":
                keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]
                answers[name] = {"score": .7, "probabilities": {key: .5 for key in keys}, "confidence": .5}
            else:
                keys = list(q["criteria"])
                answers[name] = {"choice": keys[0], "probabilities": {k: .75 if i == 0 else .25 for i, k in enumerate(keys)}, "confidence": .75}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 0}}

    async def worker(self):
        return


def client(srv=None, args=None):
    srv = srv or FakeServer()
    args = args or argparse.Namespace(mode="eager", api_keys=None, cors_origins="", mcp="off", host="127.0.0.1", port=8001)
    return TestClient(build_app(srv, args)), srv


def test_systemone_codes_request_ids_and_alias_media():
    c, srv = client()
    response = c.post("/v1/systemone", json={"state": "hi", "questions": {"x": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}, "images": ["data:image/png;base64,aGk="]}, headers={"x-request-id": "client:42"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == response.headers["x-typesafe-request-id"] == "client:42"
    assert srv.requests[0]["media"] == [{"type": "image", "data": "aGk="}]
    assert c.post("/v1/systemone", content="{").json()["code"] == "INVALID_JSON"
    assert c.post("/v1/systemone", content="[]", headers={"content-type": "application/json"}).status_code == 400
    invalid, _ = client(FakeServer("value"))
    rejected = invalid.post("/v1/systemone", json={"state": "hi", "questions": {}})
    assert rejected.status_code == 422 and rejected.json()["code"] == "INVALID_REQUEST"
    failed, _ = client(FakeServer("runtime"))
    assert failed.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul"}}}).status_code == 500
    malformed_id = c.get("/health", headers={"x-request-id": "!bad"})
    assert malformed_id.headers["x-request-id"].startswith("req_")
    assert malformed_id.headers["x-request-id"] == malformed_id.headers["x-typesafe-request-id"]
    assert c.post("/v1/systemone", json={"images": ["https://example.test/a"]}).status_code == 422
    native = {"state": "hi", "questions": {"x": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
              "media": [{"type": "image", "data": "data:image/png;base64,aGk="}]}
    assert c.post("/v1/systemone", json=native).status_code == 200
    assert srv.requests[-1]["media"] == native["media"]


@pytest.mark.parametrize("body", [
    {"state": "x"},
    {"state": "x", "questions": []},
    {"state": "x", "questions": {"bad": "question"}},
    {"state": "x", "questions": {"bad": {"type": "choice", "criteria": "not a mapping"}}},
])
def test_invalid_systemone_shapes_are_request_errors(body):
    c, _ = client()
    response = c.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("failure", ["key", "type", "runtime"])
@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
def test_decide_errors_are_mapped_by_exception(failure, path):
    c, _ = client(FakeServer(failure))
    body = ({"state": "x", "questions": {"q": {"type": "noul"}}} if path == "/v1/systemone"
            else {"input": "x", "questions": [{"type": "predicate", "instructions": "valid?"}]})
    response = c.post(path, json=body)
    code = "INTERNAL_ERROR"
    assert response.status_code == 500
    if path == "/v1/systemone":
        assert response.json()["code"] == code
    else:
        assert response.json()["error"]["code"] == code


def test_decisions_all_question_types_and_order_and_alias():
    c, srv = client()
    request = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "one"}, {"type": "input_text", "text": "two"}, {"type": "input_image", "image_url": "data:image/png;base64,aGk="}]}], "questions": [
        {"type": "predicate", "instructions": "valid?"},
        {"type": "choice", "name": "pick", "instructions": "choose", "choices": [{"value": True}, {"value": "no"}]},
        {"type": "score", "name": "rating", "instructions": "rate", "levels": [{"label": "low", "description": "Poor"}, {"label": "high"}]}]}
    response = c.post("/v1/decisions", json=request)
    assert response.status_code == 200
    payload = response.json()
    assert [a["type"] for a in payload["answers"]] == ["predicate", "choice", "score"]
    assert [a["name"] for a in payload["answers"]] == [None, "pick", "rating"]
    assert payload["answers"][1]["choice"] is True
    assert payload["model"] == "served-model" and payload["usage"]["input_tokens"] == 12
    assert srv.requests[0]["state"] == "one\n\ntwo"
    assert srv.requests[0]["media"] == [{"type": "image", "data": "aGk="}]


@pytest.mark.parametrize("questions", [[], [{}] * 201])
def test_decision_question_count_limits(questions):
    c, _ = client()
    assert c.post("/decisions", json={"input": "x", "questions": questions}).status_code == 400

@pytest.mark.parametrize("count", [1, 256])
def test_decision_choice_limits(count):
    c, _ = client()
    question = {"type": "choice", "instructions": "choose", "choices": [{"value": str(i)} for i in range(count)]}
    assert c.post("/v1/decisions", json={"input": "x", "questions": [question]}).status_code == 400

@pytest.mark.parametrize("count", [1, 11])
def test_decision_score_limits(count):
    c, _ = client()
    question = {"type": "score", "instructions": "rate", "levels": [{"label": str(i)} for i in range(count)]}
    assert c.post("/v1/decisions", json={"input": "x", "questions": [question]}).status_code == 400



def test_decision_score_level_labels_must_be_unique_and_described_levels_round_trip():
    c, srv = client()
    question = {"type": "score", "name": "rating", "instructions": "rate",
                "levels": [{"label": "a"}, {"label": "a"}, {"label": "b"}]}
    response = c.post("/v1/decisions", json={"input": "x", "questions": [question]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert response.json()["error"]["message"] == "score level labels must be unique"

    question["levels"] = [{"label": "low", "description": "Poor"},
                          {"label": "high", "description": "Good"}]
    response = c.post("/v1/decisions", json={"input": "x", "questions": [question]})
    assert response.status_code == 200
    assert [item["label"] for item in response.json()["answers"][0]["probabilities"]] == ["low", "high"]
    assert [item["probability"] for item in response.json()["answers"][0]["probabilities"]] == [.5, .5]
    assert srv.requests[-1]["questions"]["rating"]["criteria"] == {"low": "Poor", "high": "Good"}


def test_systemone_oversized_alias_media_is_rejected_before_decoding(monkeypatch):
    from basal import compat

    def fail_decode(*args, **kwargs):
        raise AssertionError("oversized media must be rejected before decoding")

    monkeypatch.setattr(compat.base64, "b64decode", fail_decode)
    encoded = "A" * ((20 * 1024 * 1024 * 4 // 3) + 4)
    c, _ = client()
    response = c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "choice",
                     "criteria": {"a": "A", "b": "B"}}}, "images": [f"data:image/png;base64,{encoded}"]})
    assert response.status_code == 422
    native = {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
              "media": [{"type": "image", "data": "aGk="}]}
    too_many = {**native, "images": ["data:image/png;base64,aGk="] * 8}
    assert c.post("/v1/systemone", json=too_many).status_code == 422


def test_decision_duplicates_unsupported_and_http_image():
    c, _ = client()
    base = {"input": "x", "questions": [{"type": "choice", "instructions": "choose", "choices": [{"value": True}, {"value": "true"}]}]}
    assert c.post("/v1/decisions", json=base).status_code == 400
    response = c.post("/v1/decisions", json={**base, "questions": [{"type": "predicate", "instructions": "x"}], "bad": 1})
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "bad"
    http = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/a"}]}], "questions": [{"type": "predicate", "instructions": "x"}]}
    assert c.post("/v1/decisions", json=http).status_code == 400


def test_auth_envelopes_request_ids_and_cors(tmp_path):
    secret = "secret"
    keys = tmp_path / "keys.json"
    keys.write_text('{"keys": [{"name": "user", "sha256": "%s"}]}' % hashlib.sha256(secret.encode()).hexdigest())
    args = argparse.Namespace(mode="eager", api_keys=str(keys), cors_origins="https://allowed.example", mcp="off", host="127.0.0.1", port=1)
    c = TestClient(build_app(FakeServer(), args))
    one = c.post("/v1/systemone", json={})
    assert one.status_code == 401 and one.json()["code"] == "UNAUTHORIZED"
    openai = c.post("/v1/decisions", json={})
    assert openai.status_code == 401 and openai.json()["error"]["type"] == "authentication_error"
    assert openai.headers["x-request-id"] == openai.headers["x-typesafe-request-id"]
    assert c.get("/health").headers.get("x-request-id")
    cors = c.options("/v1/systemone", headers={"Origin": "https://allowed.example", "Access-Control-Request-Method": "POST"})
    cors = c.post("/v1/systemone", json={"state": "x", "questions": {}}, headers={"Origin": "https://allowed.example"})
    assert "x-request-id" in cors.headers["access-control-expose-headers"].lower()
