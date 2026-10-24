"""basal.client against basal-serve and against other /v1/systemone servers of the same request shape (answers in
the shapes of vLLM's structured-diffusion example): the request bodies, the bearer token, noul answers that carry only
"noul", and that basal-serve's answers come back unchanged."""
import asyncio
import json

import httpx
import pytest

from basal.client import Basal, complete
from basal.server import to_items
from tests.test_server import FakeServer

BASAL = {"q": {"type": "noul", "noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}, "confidence": 0.8}}
OTHER = {  # the answer shapes of vLLM's examples/features/structured_diffusion/structured_server.py (jev_answer)
    "noul": {"type": "noul", "noul": 0.3},
    "choice": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.25, "tech": 0.75}, "confidence": 0.75},
    "score": {"type": "score", "score": 1.2, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
              "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.6},
}


def client(answers, seen, **kw):
    b = Basal("http://server", **kw)

    def handle(request):
        seen.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={"model": "m", "answers": answers})
    b.http = httpx.Client(transport=httpx.MockTransport(handle), headers=b.http.headers)
    return b


def test_basal_answers_come_back_unchanged_and_no_token_by_default():
    seen = []
    assert client(BASAL, seen).yes_no("x", "Urgent?") == BASAL["q"]
    assert "authorization" not in seen[0][0]
    assert seen[0][1] == {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?", "criteria": {}}}}


def test_bearer_token():
    seen = []
    client(BASAL, seen, api_key="k1").yes_no("x", "Urgent?")
    assert seen[0][0]["authorization"] == "Bearer k1"


def test_noul_with_only_p_gets_probabilities_and_confidence():
    seen = []
    ans = client({"q": OTHER["noul"]}, seen).yes_no("x", "Urgent?")
    assert ans == {"type": "noul", "noul": 0.3, "probabilities": {"true": 0.3, "false": 0.7}, "confidence": 0.7}
    assert complete(None) is None and complete(OTHER["choice"]) is OTHER["choice"]  # a skipped question, a full answer
    assert client({"q": OTHER["score"]}, []).score("x", "Tone?", ["calm", "annoyed", "furious"]) == OTHER["score"]


def test_choice_list_criteria_are_sent_as_a_map():
    seen = []
    client({"q": OTHER["choice"]}, seen).choice("x", "Which team?", ["billing", "tech"])
    assert seen[0][1]["questions"]["q"]["criteria"] == {"billing": None, "tech": None}
    # basal-serve reads both forms as the same options
    same = [to_items("x", {"q": {"type": "choice", "instructions": "Which team?", "criteria": c}})[0]
            for c in (["billing", "tech"], {"billing": None, "tech": None})]
    assert same[0] == same[1]


def test_basal_serve_answers_the_other_servers_request_example():
    """The /v1/systemone request of vLLM's example (a "model" field and a state object): basal-serve answers it, with
    the noul fields that example returns ("type", "noul") and more."""
    body = {"model": "any", "state": {"ticket": "Everything is down and we have a demo at noon."},
            "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}
    ans = asyncio.run(FakeServer([0.9, 0.1])._run(body))["answers"]["urgent"]
    assert ans["type"] == "noul" and 0 <= ans["noul"] <= 1 and set(ans["probabilities"]) == {"true", "false"}


def test_errors_raise_with_the_body():
    b = Basal("http://server")
    b.http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(422, json={"error": {"message": "bad", "type": "validation_error"}})))
    with pytest.raises(RuntimeError, match="validation_error"):
        b.yes_no("x", "Urgent?")
