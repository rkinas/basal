"""--context-overflow truncate: the questions rebuilt over the cut state keep the language of the whole request."""
import asyncio

from basal.server import Server, to_items

from test_server import Tok


class _DecTok(Tok):
    def decode(self, ids):
        return "".join(map(chr, ids))


class _CharServer(Server):
    TRUNC_MARK, TRUNC_MARGIN = "\n[...]\n", 8

    def __init__(self):
        self.tok, self.trunc_head, self.cal = _DecTok(), 0.5, {}
        self.context, self.overflow, self.max_options = (300, "context"), "truncate", 10

    def branch(self, i, j, q):
        return (i, 0, j, [ord(c) for c in q["state"] + q["branches"][j][0]])

    def encode(self, branches):
        longest = [len(b[3]) for b in branches]
        return sum(longest), longest

    def check_context(self, *a):
        pass

    def cut_of(self, *a):
        return None

    def lane_of(self, *a):
        return "main"

    async def ask(self, *a):
        return [None] * len(a[0])


def _truncated(force=None):
    srv = _CharServer()
    body = {"state": "x" * 250 + "ą" + "x" * 250,
            "questions": {"n": {"type": "noul", "instructions": "Is it valid?"},
                          "m": {"type": "multi", "criteria": ["red", "blue"], "instructions": "Select labels"}}}
    if force:
        body["lang"] = force
    qs = to_items(body["state"], body["questions"], force_lang=force)
    branches = [srv.branch(i, j, q) for i, q in enumerate(qs) for j in range(len(q["branches"]))]
    out = asyncio.run(srv.ask_text(body, qs, branches, None))
    assert out[5] and "ą" not in out[0][0]["state"]  # it was truncated and the only Polish letter is gone
    return out[0]


def test_truncation_keeps_the_language_of_the_question_branches():
    control = _truncated("pl")
    assert all(q["branches"][0][1] == ["Tak", "Nie"] for q in control)
    for q in _truncated():
        assert q["lang"] == "pl" and q["branches"][0][1] == ["Tak", "Nie"], q["branches"]


def _served(body, cut_middle):
    """Server.decide over the stub backend with --context-overflow truncate and a context that cuts the state's
    middle (cut_middle: the length of its middle) -> (answers, usage, every prompt the engine read)."""
    from test_server import FakeServer
    seen = []

    def probs(prompt, ids):
        seen.append("".join(map(chr, prompt)))
        return [0.7] + [0.3 / (len(ids) - 1)] * (len(ids) - 1)
    srv = FakeServer(probs)
    srv.tok, srv.overflow, srv.trunc_head = _DecTok(), "truncate", 0.5
    if cut_middle:
        longest = max(srv.encode([srv.branch(i, j, q) for i, q in enumerate(to_items(
            body["state"], body["questions"], force_lang=body.get("lang"))) for j in range(len(q["branches"]))])[1])
        srv.context = (longest - cut_middle + srv.TRUNC_MARGIN, "the test's context")
    out = asyncio.run(srv._run(body))
    return out["answers"], out["usage"], seen


def test_a_truncated_request_reads_as_the_same_request_with_its_language_named():
    """The served path: a Polish state whose Polish letters are all in the middle the truncation drops, and
    questions without Polish letters. Every prompt of the cut request is the one of the same request with "lang": "pl"
    (before: noul and multi read the English Yes / No and the English multi text under the Polish prompt)."""
    head, middle = "Klient pisze w sprawie karty. " * 4, "Zgubiłem kartę wczoraj wieczorem. " * 12
    tail = ("Prosze o pilny kontakt dzisiaj. " * 4)[: len(head)]
    qs = {"pilne": {"type": "noul", "instructions": "Czy sprawa jest pilna?"},
          "dzialy": {"type": "multi", "instructions": "Ktore dzialy?", "criteria": ["karty", "kredyty"]},
          "kanal": {"type": "choice", "instructions": "Jaki kanal?", "criteria": ["e-mail", "telefon"]}}
    body = {"state": head + middle + tail, "questions": qs}
    answers, usage, prompts = _served(body, len(middle))
    assert usage["truncated"] and not any("Zgubi" in p for p in prompts)
    named_answers, named_usage, named_prompts = _served(dict(body, lang="pl"), len(middle))
    assert named_usage["truncated"] == usage["truncated"]
    assert prompts == named_prompts and answers == named_answers
    assert sum("A. Tak\nB. Nie" in p or "A. Nie\nB. Tak" in p for p in prompts) >= 3  # noul + the two multi labels
    assert not any("Yes" in p for p in prompts)


def test_an_untruncated_request_is_as_before():
    """The control: the same kind of request in a context it fits is not cut, and a question without Polish letters
    keeps the Polish of its state."""
    body = {"state": "Zgubiłem kartę. Prosze o pilny kontakt.", "questions": {
        "pilne": {"type": "noul", "instructions": "Czy sprawa jest pilna?"}}}
    answers, usage, prompts = _served(body, 0)
    assert not usage.get("truncated") and all("Tak" in p and "Yes" not in p for p in prompts)
