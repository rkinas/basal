"""The rank question type (basal/rank.py) through Server.decide with a stub backend: requests of `chunk` candidates
over one shared state each, groups kept together, the prior, the final choice over the shortlist (in the screening
state or in its own), the answer, the usage and the refusals."""
import asyncio
import re

import pytest

from basal import rank
from basal.prompt import RequestError

from test_server import FakeServer


def relevance(weight):
    """Position probabilities from the candidates' texts: a yes / no question about "<Label> n" answers yes with
    weight(text) / (weight(text) + 1); a choice over "<Label> n" options picks in proportion to weight(text)."""
    def f(prompt, ids):
        s = "".join(map(chr, prompt))
        state = re.split(r"(?:State|Stan):\n", s, maxsplit=1)[1]
        state = re.split(r"\n\n(?:Question|Pytanie): ", state, maxsplit=1)[0]
        texts = dict(re.findall(r"^(\w+ \d+): (.*)$", state, re.M))
        q = re.split(r"(?:Question|Pytanie): ", s)[-1].split("\nOp", 1)[0]
        opts = re.findall(r"^[A-J]\. (.*)$", s, re.M)[-len(ids):]
        if set(opts) in ({"Yes", "No"}, {"Tak", "Nie"}):
            name = re.search(r"(\w+ \d+)", q)
            p = weight(texts[name.group(1)]) / (weight(texts[name.group(1)]) + 1) if name else 0.5
            return [p if o in ("Yes", "Tak") else 1 - p for o in opts]
        w = [weight(texts[o]) for o in opts]
        return [x / sum(w) for x in w]
    return f


def run(srv, body):
    return asyncio.run(srv._run(body))


def server(weight):
    srv = FakeServer(relevance(weight))
    srv.calls = []
    decide = srv.decide

    async def spy(body):
        if not any(rank.is_rank(q) for q in body["questions"].values()):
            srv.calls.append(body)
        return await decide(body)
    srv.decide = spy
    return srv


def scores(text):
    return float(re.search(r"\d+", text).group()) if re.search(r"\d+", text) else 0.5


INSTR = "Does {candidate} answer the question?"


def test_rank_more_candidates_than_options_in_requests_of_chunk():
    srv = server(scores)
    cands = [f"doc {i % 37}" for i in range(300)]  # 300 candidates, more than any choice question takes
    out = run(srv, {"state": "Which document?", "questions": {"r": {
        "type": "rank", "instructions": INSTR, "candidates": cands, "label": "Doc", "chunk": 50, "top": 5}}})
    a = out["answers"]["r"]
    assert a["type"] == "rank" and a["candidates"] == 300 and a["experimental"]
    screens = [b for b in srv.calls if "choice" not in b["questions"]]
    assert len(screens) == 6 and all(len(b["questions"]) == 50 for b in screens)
    assert all(q["type"] == "noul" for b in screens for q in b["questions"].values())
    assert screens[0]["state"].startswith("Which document?\n\nDoc 1: doc 0\nDoc 2: doc 1")
    assert screens[1]["questions"]["0"]["instructions"] == "Does Doc 1 answer the question?"
    assert len(a["ranking"]) == 300 and a["requests"] == 7 and out["usage"]["requests"] == 7
    top = [e["key"] for e in a["ranking"][:5]]
    assert {cands[int(k)] for k in top} == {"doc 36"} and all("choice" in e for e in a["ranking"][:5])
    rest = [e["score"] for e in a["ranking"][5:]]
    assert rest == sorted(rest, reverse=True) and "choice" not in a["ranking"][5]
    assert out["usage"]["branches"] == 301  # one per question: 300 yes / no and the choice


def test_rank_groups_stay_in_one_request_and_are_shown():
    srv = server(lambda t: 9 if "tax" in t else 1)
    cands = [{"key": f"p{g}.{i}", "text": f"page {g} part {i}" + (" tax" if (g, i) == (2, 1) else ""),
              "group": f"Page {g}"} for g in range(4) for i in range(3)]
    out = run(srv, {"state": "How do I pay the tax?", "questions": {"r": {
        "type": "rank", "instructions": INSTR, "candidates": cands, "label": "Passage", "list_title": "Passages",
        "chunk": 7, "top": 0}}})
    screens = srv.calls
    assert [len(b["questions"]) for b in screens] == [6, 6]  # 3 + 3 per request: a group is not split
    assert "Passages:\n[Page 0]\nPassage 1: page 0 part 0" in screens[0]["state"]
    assert "[Page 2]\nPassage 1: page 2 part 0" in screens[1]["state"]
    a = out["answers"]["r"]
    assert a["top"] == "p2.1" and a["requests"] == 2 and "choice" not in a["ranking"][0]
    assert rank.chunks([("a", "x", "g")] * 5, 2) == [[0, 1], [2, 3], [4]]  # a group larger than a request is split


def test_rank_choice_in_the_screening_state_or_a_shortlist():
    srv = server(scores)
    cands = {f"k{i}": f"item {i}" for i in range(12)}
    q = {"type": "rank", "instructions": INSTR, "candidates": cands, "chunk": 0, "top": 3}
    out = run(srv, {"state": "Pick one.", "questions": {"r": q}})
    screen, choice = srv.calls
    assert choice["state"] == screen["state"]  # one request held every candidate: the choice reads the same state
    c = choice["questions"]["choice"]
    assert c["option_keys"] == "hide" and list(c["criteria"].values()) == ["Candidate 10", "Candidate 11",
                                                                           "Candidate 12"]
    assert c["instructions"] == "Which candidate is the best match?"
    a = out["answers"]["r"]
    assert [e["key"] for e in a["ranking"][:3]] == ["k11", "k10", "k9"] and a["top"] == "k11"
    assert abs(sum(e["choice"] for e in a["ranking"][:3]) - 1) < 1e-6
    srv = server(scores)
    run(srv, {"state": "Pick one.", "questions": {"r": dict(q, choice_context="shortlist")}})
    assert srv.calls[1]["state"] == "Pick one.\n\nCandidate 1: item 9\nCandidate 2: item 10\nCandidate 3: item 11"
    srv = server(scores)
    run(srv, {"state": "Pick one.", "questions": {"r": dict(q, chunk=5)}})  # auto with 3 requests: a shortlist
    assert len(srv.calls) == 4 and srv.calls[3]["state"].endswith("Candidate 3: item 11")
    with pytest.raises(RequestError, match="screen needs every candidate in one request"):
        run(server(scores), {"state": "Pick one.", "questions": {"r": dict(q, chunk=5, choice_context="screen")}})


def test_rank_prior_candidates_first_limit_and_other_questions():
    srv = server(lambda t: 2.0)
    cands = {"a": "alpha", "b": "beta", "c": "gamma"}
    out = run(srv, {"state": "Q?", "questions": {
        "first": {"type": "noul", "instructions": "Is it a question?"},
        "r": {"type": "rank", "instructions": INSTR, "candidates": cands, "prior": {"b": 0.5, "c": 1.5}, "top": 0,
              "order": "candidates_first", "limit": 2}}})
    assert list(out["answers"]) == ["first", "r"] and out["answers"]["first"]["type"] == "noul"
    a = out["answers"]["r"]
    assert [e["key"] for e in a["ranking"]] == ["c", "a"] and abs(a["ranking"][0]["yes"] - 2 / 3) < 1e-6
    assert abs(a["ranking"][0]["score"] - 1.0) < 1e-6
    screen = next(b for b in srv.calls if "0" in b["questions"])
    assert screen["state"] == "Candidate 1: alpha\nCandidate 2: beta\nCandidate 3: gamma\n\nQ?"
    assert out["usage"]["questions"] == 2 and out["usage"]["requests"] == 2


def test_rank_chunk_zero_falls_back_to_requests_of_50_beyond_the_context():
    srv = server(lambda t: 1.0)
    srv.context = (3000, "the model's context")
    run(srv, {"state": "Q?", "questions": {"r": {"type": "rank", "instructions": INSTR, "top": 0, "chunk": 0,
                                                 "candidates": [f"candidate text number {i}" for i in range(120)]}}})
    assert [len(b["questions"]) for b in srv.calls] == [50, 50, 20]
    srv = server(lambda t: 1.0)
    srv.context = (20000, "the model's context")
    run(srv, {"state": "Q?", "questions": {"r": {"type": "rank", "instructions": INSTR, "top": 0, "chunk": 0,
                                                 "candidates": [f"candidate text number {i}" for i in range(120)]}}})
    assert [len(b["questions"]) for b in srv.calls] == [120]


def test_rank_in_flight_counts_its_requests():
    srv = server(lambda t: 1.0)
    seen = []
    decide = srv.decide

    async def spy(body):
        seen.append((srv.in_flight, srv.queued))
        return await decide(body)
    srv.decide = spy
    run(srv, {"state": "Q?", "questions": {"r": {"type": "rank", "instructions": INSTR, "top": 2, "chunk": 2,
                                                 "candidates": ["a", "b", "c", "d"]}}})
    assert srv.in_flight == 0 and all(f >= q for f, q in seen) and max(f for f, _ in seen) >= 2


@pytest.mark.parametrize("q, msg", [
    ({"instructions": "Is it relevant?", "candidates": ["a", "b"]}, "{candidate}"),
    ({"candidates": "a b"}, "candidates must be"),
    ({"candidates": []}, "no candidates"),
    ({"candidates": {"a": "x"}, "top": 1}, "top must be 0"),
    ({"candidates": ["a", "b"], "top": 11}, "top must be an integer from 0 to 10"),
    ({"candidates": ["a", "b"], "chunk": -1}, "chunk must be"),
    ({"candidates": ["a", "b"], "choice_context": "all"}, "choice_context"),
    ({"candidates": ["a", "b"], "order": "last"}, "order must be"),
    ({"candidates": ["a", "b"], "prior": {"7": 1}}, "unknown candidates"),
    ({"candidates": ["a", "b"], "prior": {"0": -1}}, "prior must be"),
    ({"candidates": ["a", "b"], "prior": {"0": float("inf")}}, "prior must be"),
    ({"candidates": ["a", "b"], "prior": {"0": float("nan")}}, "prior must be"),
    ({"candidates": ["a", "b"], "prior": {"0": 10**400}}, "prior must be"),
    ({"candidates": [{"key": "a", "text": "x"}, {"key": "a", "text": "y"}]}, "unique"),
    ({"candidates": [{"key": "a"}]}, "has no text"),
    ({"candidates": ["a", "b"], "max_error": 0.05}, "max_error is not available"),
    ({"candidates": ["a", "b"], "label": ""}, "label must be"),
])
def test_rank_refusals(q, msg):
    body = {"state": "Q?", "questions": {"r": {"type": "rank", "instructions": INSTR, **q}}}
    with pytest.raises(RequestError, match=re.escape(msg)):
        run(server(lambda t: 1.0), body)


def test_rank_media_refused_and_polish_defaults():
    with pytest.raises(RequestError, match="media"):
        run(server(lambda t: 1.0), {"state": "Q?", "media": [], "questions": {"r": {
            "type": "rank", "instructions": INSTR, "candidates": ["a", "b"]}}})
    spec = rank.parse("r", {"instructions": "Czy {candidate} pasuje?", "candidates": ["a", "b"]}, 255, "pl")
    assert spec["label"] == "Kandydat" and spec["choice"] == "Która pozycja najlepiej pasuje?" and spec["top"] == 20
    assert rank.parse("r", {"instructions": INSTR, "candidates": ["a", "b"]}, 10, "en")["top"] == 10


def test_rank_candidates_per_request_capped():
    """Every 50 candidates are one engine request through the queue all clients share: a server takes at most
    --rank-max-candidates (default 10,000) per request, all its rank questions together."""
    from basal.server import parser
    assert parser().parse_args([]).rank_max_candidates == rank.MAX_CANDIDATES == 10_000
    srv = server(lambda t: 1.0)
    srv.rank_max = 100
    q = {"type": "rank", "instructions": INSTR, "top": 0}
    with pytest.raises(RequestError, match="101 candidates, more than the 100 this server takes per request"):
        run(srv, {"state": "Q?", "questions": {"r": dict(q, candidates=["x"] * 101)}})
    with pytest.raises(RequestError, match="120 rank candidates in this request, more than the 100"):
        run(srv, {"state": "Q?", "questions": {"a": dict(q, candidates=["x"] * 60), "b": dict(q, candidates=["y"] * 60)}})
    assert srv.calls == []  # refused before any engine request
    out = run(srv, {"state": "Q?", "questions": {"r": dict(q, candidates=["x"] * 100)}})
    assert out["answers"]["r"]["candidates"] == 100 and out["usage"]["requests"] == 2
