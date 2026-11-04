"""Evidence spans (basal/evidence.best_spans): the band of MAX_SPAN ends per start gives exactly the spans of the full
[n, n] score matrix (2.5.3's selection, the reference below), and a state of a few tokens returns only valid spans."""
import math

import pytest
import torch

from basal.evidence import MAX_SPAN, best_spans


def reference(ls, le, rel, state, top=3):
    """2.5.3's selection over the full matrix, masked entries skipped."""
    n = ls.shape[0]
    sc = ls[:, None] + le[None, :]
    mask = torch.ones(n, n, dtype=torch.bool).triu().logical_and(~torch.ones(n, n, dtype=torch.bool).triu(MAX_SPAN))
    sc = sc.masked_fill(~mask, -1e9)
    out, flat = [], sc.flatten().topk(min(top * 4, n * n))
    for v, k in zip(flat.values.tolist(), flat.indices.tolist()):
        i, j = divmod(k, n)
        if v <= -1e8:
            continue
        a, b = rel[i][0], rel[j][1]
        if any(not (b <= x["start"] or a >= x["end"]) for x in out):
            continue
        out.append(dict(text=state[a:b], start=a, end=b, probability=math.exp(v)))
        if len(out) == top:
            break
    return out


def case(n, seed):
    g = torch.Generator().manual_seed(seed)
    ls = torch.randn(n, generator=g).log_softmax(-1)
    le = torch.randn(n, generator=g).log_softmax(-1)
    rel = [(3 * i, 3 * i + 2) for i in range(n)]
    return ls, le, rel, "x" * (3 * n)


@pytest.mark.parametrize("n", [6, 9, 12, 40, 79, 80, 81, 200, 1000])
@pytest.mark.parametrize("top", [1, 3, 5])
def test_band_equals_the_full_matrix(n, top):
    for seed in range(5):
        ls, le, rel, state = case(n, seed)
        assert best_spans(ls, le, rel, state, top) == reference(ls, le, rel, state, top), (n, seed)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_short_states_return_valid_spans_only(n):
    ls, le, rel, state = case(n, 0)
    got = best_spans(ls, le, rel, state, top=3)
    assert got and all(x["start"] < x["end"] and x["probability"] > 0 for x in got)
    assert got == reference(ls, le, rel, state, top=3)


def test_two_token_state_has_no_inverted_span():
    """Two tokens: 3 valid spans, fewer than top * 4 candidates; the masked one (end before start) was returned as a
    fourth candidate with probability 0."""
    ls = le = torch.tensor([0.9, 0.1]).log()
    got = best_spans(ls, le, [(0, 2), (3, 5)], "ab cd", top=3)
    assert [(x["start"], x["end"]) for x in got] == [(0, 2), (3, 5)]
