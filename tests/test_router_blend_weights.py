"""basal-router --blend: a member with weight 0 never supplies the answer; the answer and its "blend" field name the
same models."""
import asyncio

import pytest
from starlette.testclient import TestClient
from test_router import BIG, MID, SMALL, _choice, level_transport

from basal import router as R

CONF = {SMALL: {"c": _choice({"a": 0.8, "b": 0.2})}, MID: {"c": _choice({"a": 0.1, "b": 0.9})},
        BIG: {"c": _choice({"a": 0.4, "b": 0.6})}}


def _blend(first, weights, post=None):
    rt = R.Router([SMALL], transport=level_transport(CONF, {}, post=post),
                  blend=[([MID], weights[0]), ([BIG], weights[1])], blend_weight=first)
    with TestClient(R.app(rt, poll_s=0.05)) as c:
        asyncio.run(rt.poll())
        return c.post("/v1/systemone", json={"state": "Doc", "questions": {"c": {"type": "choice"}}}).json()["answers"]["c"]


def test_zero_weight_first_model_does_not_supply_the_answer():
    a = _blend(0.0, (1.0, 0.0))
    assert a["choice"] == "b" and a["probabilities"] == CONF[MID]["c"]["probabilities"]
    assert a["blend"] == {"models": ["model-mid"], "weights": [1.0]}


def test_only_the_first_model_with_weight_keeps_its_answer():
    a = _blend(1.0, (0.0, 0.0))
    assert a == dict(CONF[SMALL]["c"], blend={"models": ["model-small"], "weights": [1.0]})


@pytest.mark.parametrize("weights,model", [((0.0, 1.0), "model-big"), ((1.0, 0.0), "model-mid")])
def test_the_one_weighted_member_left_answers(weights, model):
    a = _blend(0.0, weights)
    host = model.split("-")[1]
    assert a["probabilities"] == CONF[f"http://{host}"]["c"]["probabilities"]
    assert a["blend"]["models"] == [model]


def test_failed_member_leaves_the_weighted_one():
    a = _blend(0.0, (1.0, 1.0), post={MID: 503})
    assert a["probabilities"] == CONF[BIG]["c"]["probabilities"] and a["blend"]["models"] == ["model-big"]


def test_two_weighted_members_blend_without_the_zero_one():
    a = _blend(0.0, (1.0, 1.0))
    assert a["probabilities"] == pytest.approx({"a": 0.25, "b": 0.75})
    assert a["blend"] == {"models": ["model-mid", "model-big"], "weights": [1.0, 1.0]}
