"""--think reads the questions as the first readout read them (Server.decide hands them to second_readout): the state
with its facts, computed once off the event loop, not again on it; task sub-questions never think."""
import asyncio

from test_think_requests import Probe, Q

from basal import server as S
from basal import think as T


def test_think_reads_the_state_readout_1_saw_and_the_facts_are_computed_once(monkeypatch):
    calls = []

    def facts(text):
        calls.append(text)
        return text + "\n[facts]"
    monkeypatch.setattr(S, "inject_facts", facts)
    monkeypatch.setattr(T, "inject_facts", facts, raising=False)  # (where the think pass computed them again)
    s, seen = Probe(), []

    async def think_one(q):
        seen.append((q["name"], q["state"]))
        return ([0.9, 0.1], "note", 1, 0), None
    s.think_one = think_one
    body = {"state": "Faktura z 15 marca 2024.", "facts": "auto", "think": "force", "questions": {"q": Q}}
    out = asyncio.run(s._run(body))
    assert calls == ["Faktura z 15 marca 2024."] and seen == [("q", "Faktura z 15 marca 2024.\n[facts]")]
    assert out["answers"]["q"]["think"]["tokens"] == 1


def test_only_the_plain_questions_of_a_task_request_think():
    s, seen = Probe(), []

    async def think_one(q):
        seen.append(q["name"])
        return ([0.9, 0.1], "note", 1, 0), None
    s.think_one = think_one
    body = {"state": "Alice works here.", "think": "force",
            "questions": {"q": Q, "v": {"type": "verify", "claims": ["Alice works here."]}}}
    out = asyncio.run(s._run(body))
    assert seen == ["q"] and out["answers"]["q"]["think"]["tokens"] == 1 and "think" not in out["answers"]["v"]
