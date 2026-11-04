"""guard_input / guard_output / route_intent presets and their decision steps (basal/presets.py), no model."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from basal import presets
from basal.client import AsyncBasal, Basal


def noul(p, abstain=None):
    a = {"type": "noul", "noul": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": max(p, 1 - p)}
    return a if abstain is None else dict(a, abstain=abstain)


def sev(level, abstain=None):
    probs = {str(i): 0.01 for i in range(4)}
    probs[str(level)] = 0.97
    a = {"type": "score", "score": float(level), "probabilities": probs, "confidence": 0.97}
    return a if abstain is None else dict(a, abstain=abstain)


def battery(jb=0.0, harmful=0.0, medical=0.0, self_harm=0.0, level=0, who="input", a=None):
    first = "jailbreak" if who == "input" else "policy_break"
    return {first: noul(jb, a), "harmful": noul(harmful, a), "medical": noul(medical, a),
            "self_harm": noul(self_harm, a), "severity": sev(level, a)}


def test_guard_batteries_have_the_listed_questions_in_both_languages():
    for name, first in (("guard_input", "jailbreak"), ("guard_output", "policy_break")):
        for lang in ("pl", "en"):
            q = presets.questions(name, lang)
            assert list(q) == [first, "harmful", "medical", "self_harm", "severity"]
            assert [q[k]["type"] for k in q] == ["noul"] * 4 + ["score"]
            assert len(q["severity"]["criteria"]) == 4
    q = presets.questions("route_intent", "en")
    assert q["intent"]["type"] == "choice" and q["complexity"]["type"] == "score"
    assert "max_error" not in q["intent"]
    assert all(v["max_error"] == 0.05 for v in presets.questions("guard_input", "en", max_error=0.05).values())


def test_guard_policies_and_precedence():
    assert presets.guard(battery())["action"] == "pass"
    assert presets.guard(battery(jb=0.9))["action"] == "block"
    assert presets.guard(battery(medical=0.7))["action"] == "review"
    assert presets.guard(battery(medical=0.7), "permissive")["action"] == "pass"
    assert presets.guard(battery(harmful=0.3))["action"] == "review"             # strict reviews from 0.15
    assert presets.guard(battery(harmful=0.3), "permissive")["action"] == "pass"
    assert presets.guard(battery(level=2), "permissive")["action"] == "review"
    assert presets.guard(battery(level=2))["action"] == "block"
    r = presets.guard(battery(jb=0.99, self_harm=0.9, medical=0.9))             # support > block > review > pass
    assert r["action"] == "support" and r["reason"] == "self_harm"
    assert presets.guard(battery(jb=0.99, medical=0.9))["action"] == "block"
    a = battery(medical=0.9)
    a["jailbreak"] = None                                                       # a missing answer is a review
    assert presets.guard(a)["action"] == "review" and presets.guard(a)["checks"]["jailbreak"]["why"] == "no answer"
    own = {"act_at": 0.99, "review_at": 0.98, "severity": {"review": 3, "block": 4}, "hazards": {"harmful": "block"}}
    assert presets.guard(battery(harmful=0.9), own)["action"] == "pass"
    with pytest.raises(ValueError, match="unknown policy"):
        presets.guard(battery(), "lax")


def test_guard_certified_mode_takes_the_thresholds_from_the_abstain_flag():
    ok = presets.guard(battery(a=False), max_error=0.05)
    assert ok["action"] == "pass" and ok["mode"] == "certified"
    assert presets.guard(battery(jb=0.8, a=False), max_error=0.05)["action"] == "block"
    assert presets.guard(battery(jb=0.8, a=True), max_error=0.05)["action"] == "review"
    assert presets.guard(battery(jb=0.2, a=True), max_error=0.05)["action"] == "review"
    assert presets.guard(battery(self_harm=0.9, a=True), max_error=0.05)["action"] == "review"
    assert presets.guard(battery(self_harm=0.9, a=False), max_error=0.05)["action"] == "support"
    assert presets.guard(battery(), max_error=0.05)["action"] == "review"        # no flag: fail closed


def test_route_intent_sends_low_confidence_and_abstain_to_a_person():
    intent = {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1}, "confidence": 0.9}
    comp = {"type": "score", "score": 1.0, "legend": {"0": "simple", "1": "moderate", "2": "complex"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}, "confidence": 0.8}
    r = presets.route({"intent": intent, "complexity": comp})
    assert r["route"] == "billing" and r["complexity"] == "moderate" and r["level"] == 1
    assert presets.route({"intent": dict(intent, confidence=0.4), "complexity": comp})["route"] == "human"
    assert presets.route({"intent": None, "complexity": comp})["route"] == "human"
    ok = {"intent": dict(intent, abstain=False), "complexity": dict(comp, abstain=False)}
    assert presets.route(ok, max_error=0.05)["route"] == "billing"
    ab = {"intent": dict(intent, abstain=True), "complexity": dict(comp, abstain=False)}
    assert presets.route(ab, max_error=0.05)["why"] == "abstain"
    assert presets.route({"intent": intent, "complexity": comp}, max_error=0.05)["route"] == "human"


def test_client_guard_asks_the_battery_with_max_error():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"answers": {k: noul(0.0, False) if v["type"] == "noul" else sev(0, False)
                                                     for k, v in body["questions"].items()}})
    b = Basal("http://basal", transport=httpx.MockTransport(handler))
    r = b.guard("Hello there", "output", "permissive", "en", max_error=0.05)
    assert r["action"] == "pass" and set(r["answers"]) == {"policy_break", "harmful", "medical", "self_harm", "severity"}
    assert seen[0]["questions"]["harmful"]["max_error"] == 0.05


def test_async_client_presets_send_and_return_what_basal_does():
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"answers": {k: noul(0.0, False) if v["type"] == "noul" else sev(0, False)
                                                     for k, v in body["questions"].items()}})

    async def go():
        async with AsyncBasal("http://basal", transport=httpx.MockTransport(handler)) as b:
            return await b.guard("Hello there", "output", "permissive", "en", max_error=0.05)
    seen = []
    want = Basal("http://basal", transport=httpx.MockTransport(handler)).guard("Hello there", "output", "permissive",
                                                                              "en", max_error=0.05)
    got = asyncio.run(go())
    assert got == want and seen[0] == seen[1]


FIXTURE = Path(__file__).resolve().parents[1] / "clients" / "typescript" / "test" / "presets_fixture.json"


def test_guard_and_route_decide_the_shared_cases_of_the_typescript_client():
    """The cases the TypeScript client checks its guard() / route() with are Python's decisions."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for c in fixture["guard"]:
        assert presets.guard(c["answers"], c["policy"], c["max_error"], c.get("battery")) == c["expect"], c
    for c in fixture["route"]:
        assert presets.route(c["answers"], 0.6, c["max_error"]) == c["expect"], c


@pytest.mark.parametrize("answers", [{}, {"harmful": noul(0.0)}, {"jailbreak": None}])
def test_an_incomplete_battery_is_never_a_pass(answers):
    r = presets.guard(answers)
    assert r["action"] == "review" and set(r["checks"]) >= {"jailbreak", "harmful", "medical", "self_harm", "severity"}
    assert r["checks"]["self_harm"]["why"] == "no answer"
    assert presets.guard(battery(), expect=["harmful"])["action"] == "pass"     # an explicit (partial) battery
    with pytest.raises(ValueError, match="unknown battery"):
        presets.guard(battery(), expect="middle")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, None, "0.9", True])
def test_a_malformed_probability_is_a_review(bad):
    for name in ("jailbreak", "harmful", "medical"):
        a = battery()
        a[name] = dict(a[name], noul=bad)
        r = presets.guard(a, "permissive")
        assert r["action"] == "review" and r["checks"][name]["why"] == "malformed answer"
    a = battery()
    a["severity"] = dict(a["severity"], probabilities={"0": bad, "1": 0.1, "2": 0.0, "3": 0.0})
    assert presets.guard(a)["checks"]["severity"]["why"] == "malformed answer"
    a = battery(a=False)
    a["harmful"] = dict(a["harmful"], abstain=None if bad is True else bad)
    assert presets.guard(a, max_error=0.05)["action"] == "review"
    intent = {"choice": "billing", "confidence": bad}
    comp = {"probabilities": {"0": 0.2, "1": 0.8}}
    assert presets.route({"intent": intent, "complexity": comp})["route"] == "human"
