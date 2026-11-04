"""Request validation of basal-serve through the app (no model): a body nested too deeply, base of an act question, a
threshold that is not finite."""
import json

import pytest
from starlette.testclient import TestClient
from test_server import FakeServer

from basal.server import check_body, http_app, to_items


class AppServer(FakeServer):
    async def serve(self, body):
        return await self._run(body)


@pytest.fixture
def client():
    return TestClient(http_app(AppServer([0.5, 0.5]), "fast"), raise_server_exceptions=False)


def post(client, path, body):
    data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=True).encode()
    return client.post(path, content=data, headers={"content-type": "application/json"})


def one(q):
    return {"state": "s", "questions": {"q": q}}


def test_deeply_nested_body_is_a_request_error(client):
    deep = b"[" * 100000 + b"]" * 100000
    assert post(client, "/v1/systemone", deep).status_code == 422
    assert post(client, "/v1/decisions", deep).status_code == 400


@pytest.mark.parametrize("base", [[], ["a"], {"a": 1}])
def test_act_base_that_is_not_a_string_is_refused_before_the_engine(client, base):
    q = {"type": "act", "instructions": "What?", "criteria": {"a": "A", "b": "B"},
         "costs": {"wrong": 1, "defer": 0.2}, "max_error": 0.1, "base": base}
    assert post(client, "/v1/systemone", one(q)).status_code == 422


@pytest.mark.parametrize("base", ["banana", ""])
def test_act_base_that_is_not_choice_or_noul_is_refused_with_the_allowed_values(client, base):
    q = {"type": "act", "instructions": "What?", "criteria": {"a": "A", "b": "B"},
         "costs": {"wrong": 1, "defer": 0.2}, "max_error": 0.1, "base": base}
    r = post(client, "/v1/systemone", one(q))
    assert r.status_code == 422 and '"choice" or "noul"' in r.json()["error"]


@pytest.mark.parametrize("base", ["choice", "noul"])
def test_act_base_choice_and_noul_are_accepted(base):
    q = {"type": "act", "instructions": "What?", "criteria": {"a": "A", "b": "B"},
         "costs": {"wrong": 1, "defer": 0.2}, "max_error": 0.1, "base": base}
    assert len(to_items("s", {"q": q})) == 1


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), "nan", "inf"])
def test_multi_threshold_that_is_not_finite_is_refused_before_the_engine(client, threshold):
    q = {"type": "multi", "instructions": "Which apply?", "criteria": {"a": "A", "b": "B"}, "threshold": threshold}
    assert post(client, "/v1/systemone", one(q)).status_code == 422


@pytest.mark.parametrize("threshold", [0.5, 0, 1, "0.25"])
def test_multi_threshold_that_is_a_number_is_accepted(threshold):
    q = {"type": "multi", "instructions": "Which apply?", "criteria": {"a": "A", "b": "B"}, "threshold": threshold}
    assert len(to_items("s", {"q": q})) == 1


@pytest.mark.parametrize("threshold", [10**400, -10**400, 2 * 10**308], ids=["1e400", "-1e400", "2e308"])
def test_multi_threshold_integer_beyond_the_largest_float_is_refused_before_the_engine(client, threshold):
    q = {"type": "multi", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}, "threshold": threshold}
    r = post(client, "/v1/systemone", one(q))
    assert r.status_code == 422 and "finite" in r.json()["error"]


def test_multi_threshold_big_but_representable_integer_and_text_behave_as_before(client):
    q = {"type": "multi", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}
    assert post(client, "/v1/systemone", one(dict(q, threshold=10**300))).status_code == 200
    assert post(client, "/v1/systemone", one(dict(q, threshold="banana"))).status_code == 422


class CheckedServer(FakeServer):
    """serve() as Server.serve is: the shape of the body is checked before decide."""

    async def serve(self, body):
        check_body(body)
        return await self._run(body)


@pytest.mark.parametrize("questions", [[], [1], [{"type": "noul"}], "x", 5, None, True, {}, 1.5, [[[]]], {"q": 1},
                                       {"q": [1]}])
def test_questions_that_are_not_an_object_of_question_objects_are_a_request_error(questions):
    c = TestClient(http_app(CheckedServer([0.5, 0.5]), "fast"), raise_server_exceptions=False)
    r = post(c, "/v1/systemone", {"state": "s", "questions": questions})
    assert r.status_code == 422 and "questions" in r.json()["error"]


def test_questions_nested_too_deeply_are_a_request_error():
    c = TestClient(http_app(CheckedServer([0.5, 0.5]), "fast"), raise_server_exceptions=False)
    deep = b'{"state": "s", "questions": ' + b"[" * 9000 + b"]" * 9000 + b"}"
    assert post(c, "/v1/systemone", deep).status_code == 422
