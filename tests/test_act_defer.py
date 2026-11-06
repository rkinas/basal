"""act costs in the short form {"wrong": W, "defer": D}: defer_action and an outcome named defer. Each test has a
control (a correct input is accepted and gives the same result before and after)."""
import pytest

from basal.prompt import RequestError
from basal.server import act_spec


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
    assert spec["costs"]["defer"] == {"yes": 0.2, "no": 0.2} and spec["costs"]["yes"] == {"yes": 0.0, "no": 1.0}
