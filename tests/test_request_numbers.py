"""Numbers and sizes a request gives are checked before the engine runs (422, the engine never called): an act cost
that is not a finite number (NaN, infinite, beyond the largest float), and more labels than the iou / f1 objectives
take (256)."""
import json

import pytest
from starlette.testclient import TestClient
from test_server import FakeServer

from basal.server import check_body, http_app


class CheckedServer(FakeServer):
    async def serve(self, body):
        check_body(body)
        return await self._run(body)


def post(body):
    srv = CheckedServer([0.5, 0.5])
    c = TestClient(http_app(srv, "fast"), raise_server_exceptions=False)
    data = json.dumps(body, allow_nan=True).encode()
    r = c.post("/v1/systemone", content=data, headers={"content-type": "application/json"})
    return r, srv.entries


def act(costs):
    return {"state": "s", "questions": {"q": {"type": "act", "instructions": "What?", "criteria": {"a": "A", "b": "B"},
                                              "costs": costs}}}


BAD = [float("nan"), float("inf"), -float("inf"), "NaN", "inf", "1e400", 10**400]


@pytest.mark.parametrize("v", BAD, ids=[repr(v)[:12] for v in BAD])
@pytest.mark.parametrize("form", ["wrong", "defer", "full"])
def test_act_cost_that_is_not_a_finite_number_is_refused_before_the_engine(v, form):
    costs = {"wrong": {"wrong": v, "defer": 1}, "defer": {"wrong": 9, "defer": v},
             "full": {"go": {"a": 0, "b": v}, "stop": {"a": 1, "b": 0}}}[form]
    r, ran = post(act(costs))
    assert r.status_code == 422 and "finite" in r.json()["error"] and ran == 0


@pytest.mark.parametrize("costs", [{"wrong": 9, "defer": 1}, {"wrong": "9", "defer": 0.5},
                                   {"go": {"a": 0, "b": 10**300}, "stop": {"a": 1, "b": 0}}])
def test_act_costs_that_are_numbers_are_accepted(costs):
    r, ran = post(act(costs))
    assert r.status_code == 200 and ran == 1


@pytest.mark.parametrize("objective", ["iou", "f1"])
def test_more_labels_than_the_objective_takes_is_refused_before_the_engine(objective):
    labels = {f"l{i}": f"Label {i}" for i in range(257)}
    q = {"type": "multi", "instructions": "Which?", "criteria": labels, "objective": objective}
    r, ran = post({"state": "s", "questions": {"q": q}})
    assert r.status_code == 422 and "256" in r.json()["error"] and ran == 0


def test_256_labels_and_other_objectives_are_accepted():
    from basal.sets import validate
    keys = [f"l{i}" for i in range(300)]
    validate("q", "f1", keys[:256])
    validate("q", "exact", keys)
    validate("q", "threshold", keys)


N = 20_000
TOO_MANY = {
    "choice": {"type": "choice", "instructions": "x", "criteria": {f"k{i}": f"o{i}" for i in range(N)}},
    "choice list": {"type": "choice", "instructions": "x", "criteria": [f"o{i}" for i in range(N)]},
    "act": {"type": "act", "instructions": "x", "criteria": {f"k{i}": f"o{i}" for i in range(N)},
            "costs": {"wrong": 1, "defer": 0.5}},
    "tool": {"type": "tool", "tools": [{"type": "function", "function": {"name": f"f{i}", "description": "d"}}
                                       for i in range(N)]},
}


@pytest.mark.parametrize("kind", list(TOO_MANY))
def test_a_question_with_far_more_options_than_the_cap_is_refused_in_linear_time(kind):
    """The options of a question are read in Server.admit, on the event loop, before the cap (255) refuses them: in
    linear time (each option's text counted in the whole list, each tool's name too, and act's outcome x outcome
    cost matrix built before the count took 2-12 s here, 30 s for 32,000 outcomes)."""
    import time

    from basal.prompt import RequestError
    from basal.server import to_items
    t = time.perf_counter()
    with pytest.raises(RequestError, match=r"question 'q': 2000\d options"):  # (a tool question: and 3 extra options)
        to_items("", {"q": TOO_MANY[kind]}, 255, force_lang="en")
    assert time.perf_counter() - t < 1.0
