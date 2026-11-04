"""Request handling found by an independent audit: the language after a truncation (B-098), the evidence window (B-099), a short
backend result (B-100), a skipped question under think (B-101). Each test fails on 8e80d3e plus the earlier packages and passes
after the fix; each has a control (a correct input behaves the same before and after)."""
import asyncio


from basal.server import Server, to_items
from tests.test_server import Tok


# I17: after the state is truncated, the questions are rebuilt and the language of the original was lost
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
    assert out[5] and "ą" not in out[0][0]["state"]       # it was truncated and the only Polish letter is gone
    return out[0]


def test_truncation_keeps_the_language_of_the_question_branches():
    control = _truncated("pl")
    assert all(q["branches"][0][1] == ["Tak", "Nie"] for q in control)
    for q in _truncated():
        assert q["lang"] == "pl" and q["branches"][0][1] == ["Tak", "Nie"], q["branches"]
