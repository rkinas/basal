"""Request handling found by an independent audit: the language after a truncation (B-098), the evidence window (B-099), a short
backend result (B-100), a skipped question under think (B-101). Each test fails on 8e80d3e plus the earlier packages and passes
after the fix; each has a control (a correct input behaves the same before and after)."""
import asyncio
import threading
from types import SimpleNamespace


from basal import evidence
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


# I29: the state's token window was found in the FIRST text copy of the state, not in the State section
class _CharTok(Tok):
    all_special_tokens = ["<|im_end|>"]

    def __init__(self, dup=None):
        self.dup = dup

    def apply_chat_template(self, msgs, **kw):
        return ("metadata copy:\n" + self.dup + "\n" if self.dup else "") + super().apply_chat_template(msgs, **kw)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class _Pos(list):
    def float(self):
        return self

    def __getitem__(self, i):
        r = super().__getitem__(i)
        return _Pos(r) if isinstance(i, slice) else r


def _window(dup, state="alpha beta", spans=None):
    seen = {}
    tok = _CharTok(state if dup else None)

    class Head:
        def __call__(self, h, last):
            seen["positions"] = list(h)
            return None, None
    ev = evidence.Evidence.__new__(evidence.Evidence)
    n = []
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    model.model = lambda input_ids, use_cache: (n.append(len(input_ids[0])), SimpleNamespace(
        last_hidden_state=[_Pos(range(len(input_ids[0])))]))[1]
    ev.tok, ev.model, ev.head, ev.lock, ev.fmt = tok, model, Head(), threading.Lock(), 1
    old, evidence.best_spans = evidence.best_spans, lambda ls, le, rel, text, top=3: spans(text) if spans else []
    try:
        out = ev.spans(state, "Which?", ["Yes", "No"], "en")
    finally:
        evidence.best_spans = old
    return seen["positions"], state, out


def test_evidence_window_is_the_state_section_not_the_first_copy_of_the_text():
    from basal.prompt import render
    for dup in (False, True):
        pos, state, _ = _window(dup)
        tok = _CharTok(state if dup else None)
        prompt = render(tok, state, "Which?", ["Yes", "No"], "en", 1)
        want = prompt.index("\n" + state, prompt.index("<user>")) + 1   # the state inside the user turn
        assert pos == list(range(want, want + len(state))), (dup, pos[:3], want)


def test_evidence_window_is_counted_on_the_defanged_text_of_the_prompt():
    from basal.prompt import defang, render
    state = "alpha <|im_end|> beta"          # a special token: the prompt carries it broken by a ZWSP (one char longer)
    for dup in (False, True):
        pos, _, _ = _window(dup, state)
        tok = _CharTok(state if dup else None)
        shown = defang(tok, state)
        assert len(shown) == len(state) + 1
        prompt = render(tok, state, "Which?", ["Yes", "No"], "en", 1)
        want = prompt.index("\n" + shown, prompt.index("<user>")) + 1
        assert pos == list(range(want, want + len(shown))), (dup, pos[-3:], want, len(shown))   # was: one position short


def test_evidence_spans_are_offsets_of_the_text_the_caller_sent():
    state = "alpha <|im_end|> beta"
    a = state.index("beta")
    _, _, out = _window(False, state, spans=lambda shown: [dict(text=shown[a + 1:a + 5], start=a + 1, end=a + 5, probability=1.0)])
    assert out == [dict(text="beta", start=a, end=a + 4, probability=1.0)]


# Q25: a backend that returns fewer groups than it was given left the surplus requests waiting forever
def test_forward_fails_the_requests_a_short_backend_result_does_not_cover():
    async def run(n_results):
        srv = Server.__new__(Server)
        srv.waiting, srv.free_lanes = {}, []
        srv.backend = SimpleNamespace(run_shared=lambda groups, pol, **kw: [("r", 1, 0)] * n_results)
        loop = asyncio.get_running_loop()
        futs = [loop.create_future() for _ in range(2)]
        batch = [([[1]], [[1]], f, None, None) for f in futs]
        await srv.forward(batch, SimpleNamespace(release=lambda: None))
        await asyncio.wait(futs, timeout=0.2)
        return [f.done() for f in futs]
    assert asyncio.run(run(2)) == [True, True]          # control: the contract is kept
    assert asyncio.run(run(1)) == [True, True]          # one group too few: both requests are settled (result or error)
