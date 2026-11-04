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
