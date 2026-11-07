"""basal-router with ask_if / depends_on: the questions a cascade level, a blend model or the fallback asks again go
without their gates (a replica refused a sub-request that names questions it does not hold), and the gates are
applied again to the final answers: a question whose gate a replaced answer now fails is null, one that it now passes
is asked (and goes through the same levels), so null and non-null answers match the final parents."""
import asyncio
import json

import httpx
import pytest
from starlette.testclient import TestClient
from test_router import FB, LLM, llm_reply

from basal import router as R
from basal import server as S

SMALL, MID, BIG = "http://small", "http://mid", "http://big"
AB = {"a": "A", "b": "B"}


def _choice(p_a):
    p = {"a": p_a, "b": round(1 - p_a, 6)}
    return {"type": "choice", "choice": max(p, key=p.get), "probabilities": p, "confidence": max(p.values())}


def replica(conf, asked):
    """Replicas that check the gates and skip as basal-serve does: conf[replica][question] = P(a)."""
    def handler(req):
        base = f"{req.url.scheme}://{req.url.host}"
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if base == LLM.rsplit("/", 1)[0]:
            return httpx.Response(200, json=llm_reply("B"))
        body = json.loads(req.content)
        qs = body["questions"]
        asked.setdefault(base, []).append(sorted(qs))
        try:
            items = S.to_items(body["state"], qs)
            gates = S.ask_gates(qs, items)
        except S.RequestError as e:
            return httpx.Response(422, json={"error": str(e)})
        answers = {q: _choice(conf[base][q]) for q in qs}
        out = {"model": "model-" + req.url.host, "answers": answers, "usage": {"questions": len(qs)}}
        skipped = S.skip_answers(answers, gates, items)
        if skipped:
            out["usage"]["skipped"] = skipped
        return httpx.Response(200, json=out)
    return httpx.MockTransport(handler)


def ask(rt, qs):
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "Doc", "questions": qs})
    assert r.status_code == 200, r.text
    return r.json()


def q(**kw):
    return {"type": "choice", "criteria": AB, **kw}


def cascade(conf, asked):
    return R.Router([SMALL], transport=replica(conf, asked), escalate=[MID], below=0.8)


def test_child_of_a_parent_that_changed_away_is_null():
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.9}, MID: {"p": 0.1, "c": 0.9}}            # p: a at level 1, b after escalation
    out = ask(cascade(conf, asked), {"p": q(), "c": q(ask_if={"p": ["a"]})})
    assert out["answers"]["p"]["choice"] == "b" and out["answers"]["c"] is None
    assert out["usage"]["skipped"] == ["c"]


def test_child_of_a_parent_that_changed_to_its_value_is_asked():
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.95}, MID: {"p": 0.1, "c": 0.3}}            # c gated on p = b: skipped at level 1
    out = ask(cascade(conf, asked), {"p": q(), "c": q(ask_if={"p": ["b"]})})
    assert out["answers"]["p"]["choice"] == "b"
    assert out["answers"]["c"]["choice"] == "a" and out["answers"]["c"]["answered_by"] == "first"
    assert "skipped" not in out["usage"]
    assert asked[SMALL] == [["c", "p"], ["c"]] and asked[MID] == [["p"]]       # c asked without its gate, alone


@pytest.mark.parametrize("status", [503, 504, 422])
def test_a_child_whose_re_ask_fails_fails_the_request(status):
    """The parent changed to the child's value, so the child is asked again; that call fails. The request fails with
    the call's status and error (it answered 200 with the child null and listed as skipped, as if its condition had
    failed), as basal-serve fails a request when one of its engine calls fails."""
    asked, conf = {}, {SMALL: {"p": 0.6, "c": 0.95}, MID: {"p": 0.1, "c": 0.3}}
    inner = replica(conf, asked)

    def handler(req):
        if req.url.path != "/health" and req.url.host == "small" and sorted(json.loads(req.content)["questions"]) == ["c"]:
            return httpx.Response(status, json={"error": f"replica says {status}"})
        return inner.handler(req)
    rt = R.Router([SMALL], transport=httpx.MockTransport(handler), escalate=[MID], below=0.8)
    with TestClient(R.app(rt, poll_s=60)) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": "Doc", "questions": {"p": q(), "c": q(ask_if={"p": ["b"]})}})
    assert r.status_code == status and r.json() == {"error": f"replica says {status}"}


def test_low_confidence_child_of_a_confident_parent_is_escalated():
    asked = {}
    conf = {SMALL: {"p": 0.95, "c": 0.55}, MID: {"p": 0.95, "c": 0.1}}
    out = ask(cascade(conf, asked), {"p": q(), "c": q(ask_if={"p": ["a"]}, depends_on=["p"])})
    assert out["answers"]["c"]["choice"] == "b" and out["answers"]["c"]["answered_by"] == "escalated"
    assert out["cascade"]["escalated"] == ["c"] and asked[MID] == [["c"]]


def test_two_levels_of_gates_follow_the_final_answers():
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.95, "g": 0.95}, MID: {"p": 0.1, "c": 0.95, "g": 0.95}}
    qs = {"p": q(), "c": q(ask_if={"p": ["b"]}), "g": q(ask_if={"c": ["a"]})}
    out = ask(cascade(conf, asked), qs)
    assert [out["answers"][k]["choice"] for k in "pcg"] == ["b", "a", "a"]   # c and then g become eligible
    assert asked[SMALL] == [["c", "g", "p"], ["c"], ["g"]]


def test_escalated_child_that_its_parent_skips_is_null():
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.55}, MID: {"p": 0.1, "c": 0.1}}
    out = ask(cascade(conf, asked), {"p": q(), "c": q(ask_if={"p": ["a"]})})
    assert out["answers"]["c"] is None and out["usage"]["skipped"] == ["c"]


@pytest.mark.parametrize("down", [True, False])
def test_level_down_keeps_the_first_answers_consistent(down):
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.9}, MID: {"p": 0.6, "c": 0.9}}
    transport = replica(conf, asked)
    if down:
        base = transport

        def handler(req):
            if req.url.host == "mid" and req.url.path != "/health":
                return httpx.Response(503, json={"error": "x"})
            return base.handle_request(req)
        transport = httpx.MockTransport(handler)
    rt = R.Router([SMALL], transport=transport, escalate=[MID], below=0.8)
    out = ask(rt, {"p": q(), "c": q(ask_if={"p": ["a"]})})
    assert out["answers"]["p"]["choice"] == "a" and out["answers"]["c"]["choice"] == "a"


def test_blend_asks_gated_questions_without_their_gates():
    asked = {}
    conf = {SMALL: {"p": 0.6, "c": 0.9}, MID: {"p": 0.0, "c": 0.5}}
    rt = R.Router([SMALL], transport=replica(conf, asked), blend=[([MID], 1.0)], blend_weight=1.0)
    out = ask(rt, {"p": q(), "c": q(ask_if={"p": ["a"]})})
    assert out["answers"]["p"]["choice"] == "b" and out["answers"]["p"]["probabilities"]["a"] == pytest.approx(0.3)
    assert out["answers"]["c"] is None                                          # p is b in the blend
    assert asked[MID] == [["c", "p"]]                                           # asked, not refused


def test_fallback_answer_regates_its_children():
    asked = {}
    conf = {SMALL: {"p": 0.55, "c": 0.9}}
    rt = R.Router([SMALL], transport=replica(conf, asked), fallback=dict(FB, below=0.6))
    out = ask(rt, {"p": q(), "c": q(ask_if={"p": ["a"]})})
    assert out["answers"]["p"]["answered_by"] == "fallback" and out["answers"]["p"]["choice"] == "b"
    assert out["answers"]["c"] is None and out["usage"]["skipped"] == ["c"]


def test_noul_and_score_parents_are_read_as_basal_serve_reads_them():
    def ans(t, p):
        if t == "noul":
            return {"type": "noul", "noul": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": max(p, 1 - p)}
        return {"type": "score", "score": p, "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 1 - p, "1": p}, "confidence": max(p, 1 - p)}
    for t, vals in (("noul", ["yes"]), ("noul", [True]), ("score", ["high"]), ("score", ["1"])):
        resp = {"answers": {"p": ans(t, 0.7), "c": _choice(0.9)}}
        order = R.gate_order({"p": {"type": t}, "c": q(ask_if={"p": vals})})
        assert R.gates_pass(resp["answers"], order[0][1])
        resp["answers"]["p"] = ans(t, 0.2)
        assert not R.gates_pass(resp["answers"], order[0][1])


def test_requests_without_gates_are_answered_as_before():
    asked = {}
    conf = {SMALL: {"p": 0.6}, MID: {"p": 0.1}}
    out = ask(cascade(conf, asked), {"p": q()})
    assert out["answers"]["p"]["answered_by"] == "escalated" and "skipped" not in out["usage"]
    assert asked == {SMALL: [["p"]], MID: [["p"]]}
