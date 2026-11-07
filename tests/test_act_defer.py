"""act costs and their defer action.

The short form {"wrong": W, "defer": D}: defer_action and an outcome named defer. Each test has a control (a correct
input is accepted and gives the same result before and after).

act with max_error needs a defer action (API.md): a request without one is refused before the engine runs, whatever
the model's confidence (it answered 200 when the decision was confident and 422, the whole response lost, when it was
not); a defer_action that is not among the cost rows is refused too."""
import json

import pytest
from starlette.testclient import TestClient
from test_server import FakeServer

from basal.prompt import RequestError
from basal.server import act_spec, check_body, cost_rows, http_app

CAL = {"temperature_per_prim": {"choice": 1.0},
       "thresholds": {"0.05": {"confidence": 0.9, "per_type": {"choice": {"confidence": 0.9}}}}}


class CheckedServer(FakeServer):
    async def serve(self, body):
        check_body(body)
        return await self._run(body)


def post(q, p_a):
    srv = CheckedServer([p_a, 1 - p_a], cal=CAL)
    c = TestClient(http_app(srv, "fast"), raise_server_exceptions=False)
    r = c.post("/v1/systemone", content=json.dumps({"state": "s", "questions": {"q": q}}).encode(),
               headers={"content-type": "application/json"})
    return r, srv.entries


ACT = {"type": "act", "instructions": "What?", "criteria": {"a": "A", "b": "B"}, "max_error": 0.05}


@pytest.mark.parametrize("p_a", [0.99, 0.6])  # confident (no refusal needed) and not
@pytest.mark.parametrize("costs", [{"wrong": 9}, {"go": {"a": 0, "b": 9}, "stop": {"a": 9, "b": 0}}])
def test_max_error_without_a_defer_action_is_refused_before_the_engine(costs, p_a):
    r, ran = post(dict(ACT, costs=costs), p_a)
    assert r.status_code == 422 and "defer action" in r.json()["error"] and ran == 0


def test_defer_action_that_is_not_a_cost_row_is_refused():
    costs = {"go": {"a": 0, "b": 9}, "stop": {"a": 9, "b": 0}}
    r, ran = post(dict(ACT, costs=costs, defer_action="ask"), 0.6)
    assert r.status_code == 422 and "'ask'" in r.json()["error"] and ran == 0


@pytest.mark.parametrize("q", [dict(ACT, costs={"wrong": 9, "defer": 1}),
                               dict(ACT, costs={"go": {"a": 0, "b": 9}, "human": {"a": 1, "b": 1}}),
                               dict(ACT, costs={"go": {"a": 0, "b": 9}, "ask": {"a": 1, "b": 1}}, defer_action="ask"),
                               {k: v for k, v in dict(ACT, costs={"go": {"a": 0, "b": 9}, "stop": {"a": 9, "b": 0}}
                                                      ).items() if k != "max_error"}])
def test_requests_with_a_defer_action_or_without_max_error_are_answered(q):
    for p_a in (0.99, 0.6):
        r, ran = post(q, p_a)
        assert r.status_code == 200 and ran == 1


def test_defer_action_is_not_silently_ignored_by_the_wrong_defer_form():
    short = {"type": "act", "costs": {"wrong": 1, "defer": 0.2}}
    for bad in ("human", "yes", "not_an_action"):
        with pytest.raises(RequestError, match="defer_action"):
            act_spec("q", dict(short, defer_action=bad), ["yes", "no"])
    with pytest.raises(RequestError, match="defer_action"):          # no "defer" cost: nothing for defer_action to name
        act_spec("q", {"type": "act", "costs": {"wrong": 1}, "defer_action": "defer"}, ["yes", "no"])
    assert act_spec("q", dict(short, defer_action="defer"), ["yes", "no"])["defer"] == "defer"   # controls
    assert act_spec("q", short, ["yes", "no"])["defer"] == "defer"
    assert act_spec("q", {"type": "act", "costs": {"wrong": 1}}, ["yes", "no"])["defer"] is None


def test_an_outcome_named_defer_does_not_overwrite_the_defer_costs():
    with pytest.raises(RequestError, match="defer"):
        act_spec("q", {"type": "act", "costs": {"wrong": 1, "defer": 0.2}}, ["defer", "accept"])
    spec = act_spec("q", {"type": "act", "costs": {"wrong": 1, "defer": 0.2}}, ["yes", "no"])   # control
    rows = cost_rows(spec)
    assert rows["defer"] == {"yes": 0.2, "no": 0.2} and rows["yes"] == {"yes": 0.0, "no": 1.0}
