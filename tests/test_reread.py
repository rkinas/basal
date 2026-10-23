"""--reread (2.5): a choice or yes / no answer of a short request whose calibrated top probability is below the threshold
is read again with the state twice (the state_twice layout) and the two readouts are combined; off, nothing changes.
The stub backend of test_server answers differently for the second layout."""
import asyncio
import math

import pytest

from basal.server import parser, reread_refused
from test_server import FakeServer

FIRST = {"x": [0.55, 0.45], "sure": [0.95, 0.05], "yn": [0.6, 0.4]}
SECOND = [0.2, 0.8]


def server(reread=None, combine="mean", max_tokens=2048):
    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        if "ponownie" in text or "(again)" in text:
            return SECOND[: len(ids)]
        for key, p in FIRST.items():
            if f"q-{key}" in text:
                return p[: len(ids)]
        return [1 / len(ids)] * len(ids)
    srv = FakeServer(probs, soam=True)
    srv.fmt, srv.max_options, srv.orders, srv.layout = 2, 255, 1, "canonical"
    srv.reread, srv.reread_combine, srv.reread_max = reread, combine, max_tokens
    return srv


BODY = {"state": "Zamówienie 7 zostało wysłane w poniedziałek.",
        "questions": {"x": {"instructions": "q-x czy wysłano?", "criteria": {"a": "tak, wysłano", "b": "nie"}},
                      "sure": {"instructions": "q-sure kiedy?", "criteria": {"pn": "w poniedziałek", "wt": "we wtorek"}},
                      "yn": {"type": "noul", "instructions": "q-yn czy na czas?"}}}


def run(srv, body=BODY):
    return asyncio.run(srv._run(body))


def test_off_changes_nothing():
    out = run(server())
    assert all("reread" not in a for a in out["answers"].values()) and "reread_branches" not in out["usage"]
    assert out["answers"]["x"]["probabilities"] == pytest.approx({"a": 0.55, "b": 0.45})


def test_uncertain_answers_are_read_again_and_combined():
    plain = run(server())
    srv = server(0.9)
    out = run(srv)
    a = out["answers"]
    assert a["x"].get("reread") and a["yn"].get("reread") and "reread" not in a["sure"]
    assert out["usage"]["reread_branches"] == 2
    assert out["usage"]["input_tokens"] > plain["usage"]["input_tokens"]  # the second reads' prompts
    g = [math.sqrt(0.55 * 0.2), math.sqrt(0.45 * 0.8)]  # mean of the log-probabilities, renormalised
    want = [x / sum(g) for x in g]
    assert list(a["x"]["probabilities"].values()) == pytest.approx(want, abs=1e-6)
    assert a["sure"]["probabilities"] == plain["answers"]["sure"]["probabilities"]  # confident: untouched
    second = run(server(0.9, "second"))["answers"]["x"]["probabilities"]
    assert list(second.values()) == pytest.approx(SECOND, abs=1e-6)


def test_long_requests_are_not_read_again():
    out = run(server(0.9, max_tokens=20))
    assert out["usage"]["reread_branches"] == 0 and all("reread" not in a for a in out["answers"].values())


def test_only_choice_and_yes_no():
    body = {"state": BODY["state"],
            "questions": {"s": {"type": "score", "instructions": "q-x jak bardzo?", "levels": ["nisko", "wysoko"]},
                          "m": {"type": "multi", "instructions": "q-x które?", "criteria": {"j": "jedno", "d": "drugie"}}}}
    out = run(server(0.99), body)
    assert out["usage"]["reread_branches"] == 0


def test_refusals_and_flags():
    assert reread_refused(None, 1) is None and reread_refused(0.9, 2) is None
    assert "format 1" in reread_refused(0.9, 1)
    assert "(0, 1]" in reread_refused(1.5, 2)
    a = parser().parse_args(["--model", "m"])
    assert a.reread is None and a.reread_combine == "mean" and a.reread_max_tokens == 2048
    a = parser().parse_args(["--model", "m", "--reread", "0.9", "--reread-combine", "second"])
    assert a.reread == 0.9 and a.reread_combine == "second"
