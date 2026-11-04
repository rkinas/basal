"""basal-router: ask_if after an escalation or a blend (B-090), the images alias (B-091), a zero-weight base (B-092).
Each test fails on 8e80d3e plus the earlier packages and passes after the fix; each has a control (correct input, same result)."""
import asyncio
import json

import httpx
import pytest

from basal import router

CHOICE = {"type": "choice", "criteria": ["yes", "no"]}


def _ans(choice, p):
    py = p if choice == "yes" else 1 - p
    return {"type": "choice", "choice": choice, "confidence": p, "probabilities": {"yes": py, "no": 1 - py}}


def _noul(true, p):
    return {"type": "noul", "noul": true, "confidence": p, "probabilities": {"true": p if true else 1 - p, "false": 1 - p if true else p}}


def _score(top, p):
    other = "0" if top == "1" else "1"
    return {"type": "score", "score": float(top), "confidence": p, "probabilities": {top: p, other: 1 - p}}


def _forward(req, handler, **kw):
    async def go():
        r = router.Router(kw.pop("replicas", ["http://small"]), transport=httpx.MockTransport(handler), **kw)
        r.ready.update(r.replicas)
        try:
            status, raw, _ = await r.forward("/v1/systemone", json.dumps(req).encode(), {})
            return status, json.loads(raw)
        finally:
            await r.client.aclose()
    return asyncio.run(go())


def _like_server(request, make):
    """A replica that refuses (422) a request whose ask_if / depends_on names a question the request does not have."""
    qs = json.loads(request.content)["questions"]
    for name, spec in qs.items():
        for x in list((spec.get("ask_if") or {})) + list(spec.get("depends_on") or []):
            if x not in qs:
                return httpx.Response(422, json={"error": f"question {name!r}: depends on unknown question {x!r}"})
    return httpx.Response(200, json={"model": request.url.host, "answers": {n: make(request.url.host, n) for n in qs}})


# parent type -> (question, answer the first level gives, answer the escalation gives, ask_if value that holds first)
PARENTS = {
    "choice": (CHOICE, lambda: _ans("yes", .6), lambda: _ans("no", .99), ["yes"]),
    "noul": ({"type": "noul"}, lambda: _noul(True, .6), lambda: _noul(False, .99), ["yes"]),
    "score": ({"type": "score", "criteria": ["low", "high"]}, lambda: _score("1", .6), lambda: _score("0", .99), ["high"]),
}


@pytest.mark.parametrize("kind", sorted(PARENTS))
def test_cascade_nulls_a_child_whose_gate_changed_in_the_escalation(kind):
    spec, first, big, ask = PARENTS[kind]
    req = {"state": "s", "questions": {"gate": spec, "child": dict(CHOICE, ask_if={"gate": ask})}}

    def make(host, name):
        if name == "gate":
            return big() if host == "large" else first()
        return _ans("yes", .99)
    status, out = _forward(req, lambda r: _like_server(r, make), escalate=["http://large"], below=.8)
    assert status == 200 and out["answers"]["gate"]["probabilities"] == big()["probabilities"]
    assert out["answers"]["child"] is None
    assert out["usage"]["skipped"] == ["child"]                              # docs/API.md: usage.skipped names it
    assert out["cascade"]["escalated"] == ["gate"]
    # control: the escalated parent still satisfies the gate -> the child keeps its answer
    status, out = _forward(req, lambda r: _like_server(r, lambda h, n: first() if n == "gate" else _ans("yes", .99)),
                           escalate=["http://large"], below=.8)
    assert out["answers"]["child"]["choice"] == "yes" and "skipped" not in out.get("usage", {})


def test_cascade_escalates_a_child_whose_parent_is_not_in_the_subset():
    req = {"state": "s", "questions": {"gate": CHOICE, "child": dict(CHOICE, ask_if={"gate": ["yes"]}),
                                       "dep": dict(CHOICE, depends_on=["gate"])}}

    def make(host, name):
        if name == "gate":
            return _ans("yes", .99)
        return _ans("yes", .99) if host == "large" else _ans("yes", .6)
    status, out = _forward(req, lambda r: _like_server(r, make), escalate=["http://large"], below=.8)
    # main: the escalation request names a gate it does not contain, the replica answers 422 and nothing is escalated
    assert status == 200 and out["cascade"]["escalated"] == ["child", "dep"]
    assert out["answers"]["child"]["confidence"] == .99 and out["answers"]["child"]["choice"] == "yes"
    assert out["answers"]["gate"]["confidence"] == .99                      # control: the parent was not asked again


def test_blend_of_a_child_without_its_parent_in_the_subset():
    # the parent has max_error, so it is not blended: the blend request holds the child alone
    gate = dict(CHOICE, max_error=.1)
    req = {"state": "s", "questions": {"gate": gate, "child": dict(CHOICE, ask_if={"gate": ["yes"]})}}

    def make(host, name):
        return _ans("yes", .9) if host != "other" or name != "child" else _ans("yes", .7)
    status, out = _forward(req, lambda r: _like_server(r, make), blend=[(["http://other"], 1.)], replicas=["http://main"])
    assert status == 200 and len(out["answers"]["child"]["blend"]["models"]) == 2     # main: 422, no blend
    assert "blend" not in out["answers"]["gate"]                                      # control


def test_blend_nulls_a_child_whose_gate_flipped_in_the_blend():
    req = {"state": "s", "questions": {"gate": CHOICE, "child": dict(CHOICE, ask_if={"gate": ["yes"]})}}

    def make(host, name):
        if name == "gate":
            return _ans("no", .9) if host == "other" else _ans("yes", .6)
        return _ans("yes", .99)
    _, out = _forward(req, lambda r: _like_server(r, make), blend=[(["http://other"], 1.)], replicas=["http://main"])
    assert out["answers"]["gate"]["choice"] == "no" and out["answers"]["child"] is None
    assert out["usage"]["skipped"] == ["child"]
    _, out = _forward(req, lambda r: _like_server(r, lambda h, n: _ans("yes", .9)), blend=[(["http://other"], 1.)],
                      replicas=["http://main"])                                       # control: the gate holds
    assert out["answers"]["gate"]["choice"] == "yes" and out["answers"]["child"]["choice"] == "yes"
    assert "skipped" not in out.get("usage", {})


def test_router_adds_to_the_skipped_names_the_replica_already_gave():
    assert router.note_skipped({"usage": {"skipped": ["a"], "output_tokens": 1}}, ["b", "a", "b"]) is None
    resp = {"usage": {"skipped": ["a"]}}
    router.note_skipped(resp, ["b", "a", "b"])
    assert resp["usage"]["skipped"] == ["a", "b"]
    resp = {}
    router.note_skipped(resp, [])
    assert resp == {}                                                                # nothing nulled: nothing added
