"""act costs (B-093, B-094) and extract schemas (B-095, B-096): each test fails on 8e80d3e plus the earlier packages and passes
after the fix; each has a control (a correct input is accepted and gives the same result before and after)."""
import pytest

from basal.prompt import RequestError
from basal.server import act_spec

FULL = {"type": "act", "max_error": 0.05, "costs": {"approve": {"y": 0, "n": 1}, "reject": {"y": 1, "n": 0}}}


# B-093: defer_action that names no action of costs
def test_defer_action_must_be_one_of_the_costs_actions():
    with pytest.raises(RequestError, match="defer_action"):
        act_spec("q", dict(FULL, defer_action="not_an_action"), ["y", "n"])
    assert act_spec("q", dict(FULL, defer_action="reject"), ["y", "n"])["defer"] == "reject"   # control: a declared action


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
