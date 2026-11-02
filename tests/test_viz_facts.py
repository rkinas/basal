"""basal-viz explains the prompt basal-serve serves: with "facts": "auto" the state carries the computed facts."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from basal import viz
from basal.facts import inject
from basal.prompt import RequestError

STATE = "Faktura wystawiona 15 marca 2024 r., termin płatności 30 dni."


@pytest.fixture
def explainer(monkeypatch):
    """An Explainer without a model: to_items records the state it is given, no question is explained."""
    seen = []
    monkeypatch.setattr(viz, "to_items", lambda state, *_, **__: seen.append(state) or [])
    ex = object.__new__(viz.Explainer)
    ex.lock, ex.srv, ex.name = threading.Lock(), SimpleNamespace(max_options=26), "stub"
    return ex, seen


@pytest.mark.parametrize("state", [STATE, {"dokument": STATE}])
def test_facts_auto_explains_the_state_with_the_facts_block(explainer, state):
    ex, seen = explainer
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    ex.explain({"state": state, "questions": {}, "facts": "auto"})
    assert seen == [inject(text)] and seen[0] != text  # the facts basal-serve appends (Server.decide)


@pytest.mark.parametrize("facts", [None, "off"])
def test_without_facts_the_state_is_explained_as_given(explainer, facts):
    ex, seen = explainer
    ex.explain({"state": STATE, "questions": {}} | ({"facts": facts} if facts else {}))
    assert seen == [STATE]


@pytest.mark.parametrize("facts", ["on", "AUTO", None, False, 1, {}, []])
def test_invalid_facts_are_rejected_before_building_prompts(explainer, facts):
    ex, seen = explainer
    body = {"state": STATE, "questions": {}, "facts": facts}
    with pytest.raises(RequestError) as served:
        asyncio.run(viz.Server.decide(ex.srv, body))
    with pytest.raises(RequestError) as explained:
        ex.explain(body)
    assert str(explained.value) == str(served.value)
    assert viz.http_status(explained.value) == 422
    assert seen == []
