"""Policy defaults kept and pinned: a multi question with min above max returns the min most probable labels (the
legacy coercion since 1.5, documented in API.md); the think pass's readout 2 reads the canonical prompt layout, as
its weights and calibration were made, whatever layout readout 1 serves."""
import asyncio

from test_server import FakeServer
from test_think_requests import Probe

from basal import think as T
from basal.prompt import render
from basal.sets import select, validate


def test_min_above_max_returns_the_min_most_probable_labels():
    probs = {"a": 0.9, "b": 0.6, "c": 0.4, "d": 0.1}
    validate("q", "threshold", list(probs), 0.5, 3, 1)  # (accepted, as since 1.5)
    assert sorted(select(probs, "threshold", 0.5, min=3, max=1)[0]) == ["a", "b", "c"]


class Sidecar:
    max_len = 10**6

    def __init__(self):
        self.prompts = []

    async def reason(self, ids, max_tokens):
        self.prompts.append("".join(map(chr, ids)))
        return "note", 1

    async def readout(self, ids, codes):
        import torch
        return torch.tensor([0.9, 0.1])


class Layouted(Probe):
    """The real think_one (readout 2 over a stand-in sidecar) on a server serving the qfirst layout."""
    think_one = T.ThinkServer.think_one

    def __init__(self):
        super().__init__()
        self.fmt, self.max_options, self.layout, self.think_tokens = 2, 255, "qfirst", 8
        self.sidecar = Sidecar()


def test_readout_2_reads_the_canonical_layout():
    s = Layouted()
    q = {"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}
    out = asyncio.run(FakeServer._run(s, {"state": "Invoice 1240.", "think": "force", "questions": {"q": q}}))
    assert out["answers"]["q"]["think"]["tokens"] == 1
    canonical = render(s.tok, "Invoice 1240.", "Which?", ["a", "b"], "en", 2)
    sent = s.sidecar.prompts[0][:len(canonical) - 40]  # (its head: the prompt without the answer prefill)
    assert canonical.startswith(sent)
    assert not render(s.tok, "Invoice 1240.", "Which?", ["a", "b"], "en", 2, "qfirst").startswith(sent)
