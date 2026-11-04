"""--think auto keeps the contracts of an answer (a stub readout 2, no vLLM): readout 2's answer carries the abstain flag
and answer set the question asks for (max_error, coverage), under readout 2's calibration; ask_if is decided by the
answers after thinking; a request with images or audio does not think (readout 2 reads text only)."""
import asyncio

from test_server import FakeServer
from test_think_requests import Probe

from basal.server import NO_CERTIFICATE, NO_CONFORMAL

Q = {"type": "choice", "criteria": ["a", "b"]}


class Leaning(Probe):
    """Readout 1: a 0.4, b 0.6 (one option order); readout 2: a 0.9, b 0.1."""

    def __init__(self):
        super().__init__()
        FakeServer.__init__(self, [0.4, 0.6])
        self.orders = 1

    def media_of(self, body, pol):
        return {"kinds": ["image"], "images": [None], "audio": [], "detail": "default"} if body.get("media") else None

    async def ask_media(self, qs, branches, m):
        return [[self.probs_for(p, i) for p, i in zip(b[3], b[4])] for b in branches], 0, [0] * len(branches), 0


def run(s, body):
    return asyncio.run(s._run(body))


def test_readout_2_answer_keeps_the_limits_asked_for_without_a_calibration():
    out = run(Leaning(), {"state": "text", "think": "force", "questions": {"q": dict(Q, max_error=0.05, coverage=0.95)}})
    a = out["answers"]["q"]
    assert a["think"]["tokens"] == 1 and a["choice"] == "a"
    assert a["abstain"] is True and a["max_error"] == NO_CERTIFICATE           # readout 2 has no certificate
    assert a["answer_set"] == ["a", "b"] and a["coverage"] == 0.95 and a["coverage_note"] == NO_CONFORMAL


def test_readout_2_answer_uses_its_own_calibration():
    s = Leaning()
    s.think_cal = {"thresholds": {"0.05": {"confidence": 0.8, "per_type": {"choice": {"confidence": 0.8}}}},
                   "conformal": {"levels": {"0.95": {"per_type": {"choice": {"p_min": 0.5}}}}}}
    a = run(s, {"state": "text", "think": "force", "questions": {"q": dict(Q, max_error=0.05, coverage=0.95)}})
    a = a["answers"]["q"]
    assert a["abstain"] is False and "max_error" not in a and a["answer_set"] == ["a"]


def test_ask_if_follows_the_answer_after_thinking():
    qs = {"parent": Q, "child": dict(Q, ask_if={"parent": ["a"]}), "other": dict(Q, ask_if={"parent": ["b"]})}
    out = run(Leaning(), {"state": "text", "think": "force", "questions": qs})
    assert out["answers"]["parent"]["choice"] == "a"                           # b before thinking
    assert out["answers"]["child"] is not None and out["answers"]["other"] is None
    assert out["usage"]["skipped"] == ["other"]
    off = run(Leaning(), {"state": "text", "think": "off", "questions": qs})
    assert off["answers"]["parent"]["choice"] == "b" and off["answers"]["child"] is None


def test_a_request_with_media_does_not_think():
    s = Leaning()
    out = run(s, {"state": "text", "think": "force", "questions": {"q": Q}, "media": [{"type": "image", "data": "x"}]})
    assert s.thought == [] and out["answers"]["q"]["choice"] == "b"
    assert out["answers"]["q"]["meta"]["think"] == {"used": False, "tokens": 0}
