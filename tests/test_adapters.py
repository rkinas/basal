"""Presets (basal-run --preset, Basal.preset) and the agent-framework adapters, against a fake server (the frameworks
themselves are stubbed: they are not dependencies)."""
import json
import sys
import types

import httpx
import pytest

from basal import adapters, presets
from basal.client import Basal
from basal.run import to_request

TYPES = {"choice", "noul", "score", "multi"}


def fake_server(seen):
    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"answers": {q: {"type": v["type"], "probabilities": {"a": 1.0}}
                                                     for q, v in body["questions"].items()}})
    return Basal("http://basal", transport=httpx.MockTransport(handler))


def test_presets_are_valid_questions_in_both_languages():
    for name in presets.names():
        pl, en = presets.questions(name, "pl"), presets.questions(name, "en")
        assert set(pl) == set(en) and pl
        for q in list(pl.values()) + list(en.values()):
            assert q["type"] in TYPES and q["instructions"].strip()
    q = presets.questions("triage", "en")
    q["urgency"]["criteria"].append("x")                       # a copy: editing it changes nothing
    assert presets.questions("triage", "en")["urgency"]["criteria"] == ["low", "medium", "high"]
    with pytest.raises(ValueError, match="unknown preset"):
        presets.questions("nope")


def test_preset_language_follows_the_state():
    assert presets.lang_of("Klient zgłasza, że nie działa logowanie do konta") == "pl"
    assert presets.lang_of("The customer cannot log in to the account") == "en"
    body = to_request({"state": "Klient zgłasza, że nie działa logowanie"}, preset="triage")
    assert body["questions"]["urgency"]["instructions"] == "Jak pilna jest sprawa?"
    assert to_request({"state": "x", "questions": {"q": {}}}, preset="triage")["questions"] == {"q": {}}  # own wins
    assert to_request({"state": "s"}, preset="triage", lang="en")["questions"]["needs_human"]["type"] == "noul"


def test_client_preset():
    seen = []
    ans = fake_server(seen).preset("Spam spam, buy now", "moderation", "en")
    assert set(ans) == {"harmful", "spam", "personal_data"} and seen[0]["state"] == "Spam spam, buy now"


@pytest.fixture
def frameworks(monkeypatch):
    """Stub langchain_core / llama_index / crewai with the factory each adapter uses."""
    made = {}

    class StructuredTool:
        @classmethod
        def from_function(cls, func, name, description):
            made["langchain"] = dict(func=func, name=name, description=description)
            return func

    class FunctionTool:
        @classmethod
        def from_defaults(cls, fn, name, description):
            made["llamaindex"] = dict(func=fn, name=name, description=description)
            return fn

    class BaseTool:
        pass

    for mod, attrs in {"langchain_core": {}, "langchain_core.tools": {"StructuredTool": StructuredTool},
                       "llama_index": {}, "llama_index.core": {}, "llama_index.core.tools": {"FunctionTool": FunctionTool},
                       "crewai": {}, "crewai.tools": {"BaseTool": BaseTool}}.items():
        m = types.ModuleType(mod)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, mod, m)
    return made


def test_adapters_call_the_http_api(frameworks):
    seen = []
    client = fake_server(seen)
    q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}
    assert set(adapters.langchain_tool(client=client)("state A", q)) == {"q"}
    assert set(adapters.llamaindex_tool(client=client)("state B", json.dumps(q))) == {"q"}  # JSON text works too
    crew = adapters.crewai_tool(client=client, preset="triage", lang="en")
    out = json.loads(crew._run("state C"))
    assert set(out) == {"urgency", "category", "needs_human"} and crew.name == "basal_decide"
    assert [b["state"] for b in seen] == ["state A", "state B", "state C"]
    assert frameworks["langchain"]["name"] == "basal_decide" and "preset" not in frameworks["langchain"]["description"]
    with pytest.raises(ValueError, match="questions are needed"):
        adapters.langchain_tool(client=client)("state D")


def test_adapter_without_its_framework_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai.tools", None)
    with pytest.raises(ImportError, match="uv pip install crewai"):
        adapters.crewai_tool()
