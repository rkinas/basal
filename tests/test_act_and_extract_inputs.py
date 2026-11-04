"""act costs (B-093, B-094) and extract schemas (B-095, B-096): each test fails on 8e80d3e plus the earlier packages and passes
after the fix; each has a control (a correct input is accepted and gives the same result before and after)."""
import pytest

from basal import tasks
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


# B-094: an outcome named defer
def test_an_outcome_named_defer_does_not_overwrite_the_defer_costs():
    with pytest.raises(RequestError, match="defer"):
        act_spec("q", {"type": "act", "costs": {"wrong": 1, "defer": 0.2}}, ["defer", "accept"])
    spec = act_spec("q", {"type": "act", "costs": {"wrong": 1, "defer": 0.2}}, ["yes", "no"])   # control
    assert spec["costs"]["defer"] == {"yes": 0.2, "no": 0.2} and spec["costs"]["yes"] == {"yes": 0.0, "no": 1.0}


# B-095: a dot in a property name
def test_extract_rejects_a_property_name_with_a_dot():
    schema = {"type": "object", "properties": {"a.b": {"type": "boolean"},
                                               "a": {"type": "object", "properties": {"b": {"type": "boolean"}}}}}
    with pytest.raises(RequestError):
        tasks.expand({"state": "s", "questions": {"q": {"type": "extract", "schema": schema}}})
    ok = {"type": "object", "properties": {"a_b": {"type": "boolean"},
                                           "a": {"type": "object", "properties": {"b": {"type": "boolean"}}}}}
    assert len(tasks.expand({"state": "s", "questions": {"q": {"type": "extract", "schema": ok}}})[0]["questions"]) == 2


# B-096: enum values that share an answer key
@pytest.mark.parametrize("values", [[1, "1"], ["__missing__", "x"]])
def test_extract_rejects_enum_values_that_share_an_answer_key(values):
    with pytest.raises(RequestError):
        tasks.expand({"state": "s", "questions": {"q": {"type": "extract", "schema": {
            "type": "object", "properties": {"f": {"enum": values}}}}}})
    tasks.expand({"state": "s", "questions": {"q": {"type": "extract", "schema": {            # control
        "type": "object", "properties": {"f": {"enum": ["a", "b"]}}}}}})
