"""Evidence output: `"evidence": true` on a question returns the span(s) of the state that support the answer.

A pointer head over the state's tokens, trained after the decision model on its frozen final-layer states (so it does
not change any decision): start / end scores of a state token are bilinear in that token's state and the state at the
answer position, so the span depends on the question. The head ships as `evidence_head.pt` next to the weights; it was
trained on verified verbatim evidence quotes from real Polish documents. Spans are character offsets into the state
text, so the quote is always verbatim; each comes with a probability (p_start * p_end of the span).

This version reads the head on a separate pass of the question's prompt (original option order); it runs only for
questions that ask for evidence. EvidenceHead and best_spans must stay identical to the copy used to train the head.
"""
import math
import threading
from pathlib import Path

import torch
import torch.nn as nn

from .prompt import defang, render

MAX_SPAN = 80


class EvidenceHead(nn.Module):
    def __init__(self, d, k=256):
        super().__init__()
        self.k = k
        self.ws, self.us = nn.Linear(d, k, bias=False), nn.Linear(d, k, bias=False)
        self.we, self.ue = nn.Linear(d, k, bias=False), nn.Linear(d, k, bias=False)
        self.bs, self.be = nn.Linear(d, 1, bias=False), nn.Linear(d, 1, bias=False)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)  # final-layer states have large norms

    def forward(self, h, q):
        """h: (n, d) state token states, q: (d,) answer-position state -> start / end log-probs over the n tokens."""
        h, q = self.norm(h), self.norm(q)
        s = self.ws(h) @ self.us(q) / math.sqrt(self.k) + self.bs(h).squeeze(-1)
        e = self.we(h) @ self.ue(q) / math.sqrt(self.k) + self.be(h).squeeze(-1)
        return s.log_softmax(-1), e.log_softmax(-1)


def best_spans(ls, le, rel, state, top=3):
    """The top non-overlapping spans (start token i <= end token j < i + MAX_SPAN) by log p_start + log p_end, from
    the top * 4 best candidates. Only the band of MAX_SPAN ends per start is scored ([n, MAX_SPAN], not [n, n]: a
    32k-token state took 4 GB per score matrix), and only valid spans are candidates (a state of a few tokens has fewer
    than top * 4: no inverted or empty span is returned)."""
    n = ls.shape[0]
    w = min(MAX_SPAN, n)
    end = torch.arange(n, device=ls.device)[:, None] + torch.arange(w, device=ls.device)[None]  # [n, w]
    ok = end < n
    sc = (ls[:, None] + le[end.clamp(max=n - 1)]).masked_fill(~ok, float("-inf"))
    out, flat = [], sc.flatten().topk(min(top * 4, int(ok.sum())))
    for v, k in zip(flat.values.tolist(), flat.indices.tolist()):
        i, d = divmod(k, w)
        a, b = rel[i][0], rel[i + d][1]
        if any(not (b <= x["start"] or a >= x["end"]) for x in out):
            continue  # no overlapping spans
        out.append(dict(text=state[a:b], start=a, end=b, probability=math.exp(v)))
        if len(out) == top:
            break
    return out


def shown_to_state(state, shown):
    """Offsets into `shown` (the state with zero-width spaces inserted by prompt.defang) -> offsets into `state`: an
    inserted character maps to the offset of the character after it."""
    back, i = [], 0
    for c in shown:
        back.append(i)
        if i < len(state) and c == state[i]:
            i += 1
    back.append(len(state))
    return back


class Evidence:
    """Loads `evidence_head.pt` from the model directory (absent: evidence requests are refused)."""

    def __init__(self, model_dir, model, tok, fmt=1):
        p = Path(model_dir) / "evidence_head.pt"
        self.available = p.exists()
        self.model, self.tok, self.fmt, self.lock = model, tok, fmt, threading.Lock()
        if self.available:
            dev = next(model.parameters()).device
            ck = torch.load(p, map_location=dev)
            self.head = EvidenceHead(ck["d"], ck["k"]).to(dev).float().eval()
            self.head.load_state_dict(ck["state_dict"])

    @torch.no_grad()
    def spans(self, state, question, options, lang, limit=None, top=3):
        """Top non-overlapping spans of `state` (character offsets); spans starting at or after `limit` (e.g. in the
        appended facts block) are dropped."""
        prompt = render(self.tok, state, question, options, lang, self.fmt)
        shown = defang(self.tok, state)  # the state as the prompt shows it (special tokens broken)
        s0 = prompt.index("\n" + shown) + 1
        s1 = s0 + len(shown)
        enc = self.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= s0 and b <= s1 and b > a]
        if not idx:
            return []
        t0, t1 = idx[0], idx[-1] + 1
        rel = [(max(a - s0, 0), max(b - s0, 0)) for a, b in enc["offset_mapping"][t0:t1]]
        if shown != state:  # offsets into the shown state -> offsets into the request's state
            back = shown_to_state(state, shown)
            rel = [(back[a], back[b]) for a, b in rel]
        dev = next(self.model.parameters()).device
        with self.lock:
            h = self.model.model(input_ids=torch.tensor([enc["input_ids"]], device=dev),
                                 use_cache=False).last_hidden_state[0].float()
            ls, le = self.head(h[t0:t1], h[-1])
            out = best_spans(ls, le, rel, state, top=top + 2)
        if limit is not None:
            out = [x for x in out if x["start"] < limit]
            for x in out:
                x["end"] = min(x["end"], limit)
                x["text"] = state[x["start"]:x["end"]]
        return out[:top]
