"""Set selection for `multi`: which labels to return, from one independent probability per label.

  threshold  labels with p >= threshold (default 0.5), at most `max`, at least `min` (the 1.5 behaviour)
  exact      labels with p >= 0.5: the most probable set, i.e. the highest chance that the selected set is exactly right
  iou        the set with the highest expected IoU (Jaccard index) with the true set
  f1         the set with the highest expected F1 with the true set

Under independence the best set of a given size is the top-k labels by probability (swapping a less probable label in
for a more probable one never raises the expected score), so the engine scores every prefix k = min..max of the labels
sorted by probability and returns the best one with its expected score. Both empty (nothing selected, nothing applies)
counts as a perfect match.

The expectations are exact. With A ~ the positives inside the prefix and B ~ the positives outside it (independent
Poisson-binomial variables), IoU = A / (k + B) and F1 = 2A / (k + A + B), and their expectations are integrals of the
Poisson-binomial generating functions, G_B(t) = prod_j (1 - p_j + p_j t):

  E[IoU_k] = E[A] * E[1 / (k + B)] = E[A] * int_0^1 t^(k-1) G_B(t) dt
  E[F1_k]  = 2 sum_{i <= k} p_i E[1 / (k + 1 + N_-i)] = 2 int_0^1 t^k sum_{i <= k} p_i prod_{j != i} (1 - p_j + p_j t) dt

(N_-i: the positives among all labels except i). The integrands are polynomials of degree < 2n, so Gauss-Legendre
quadrature with M >= n nodes integrates them exactly up to rounding; all prefixes together cost O(n M) in log space
(255 labels: about 0.1 ms on a CPU).
"""
import builtins
import hashlib
import json
from functools import lru_cache

import numpy as np

OBJECTIVES = ("threshold", "exact", "iou", "f1")


@lru_cache(maxsize=None)
def _nodes(m):
    """Gauss-Legendre quadrature on [0, 1] with m nodes (exact for polynomials of degree <= 2m - 1): the weighted
    powers w_m t_m^j, j = 0..256, as a [257, m] table."""
    x, w = np.polynomial.legendre.leggauss(m)
    t = (x + 1) / 2
    return t, w / 2 * t[None, :] ** np.arange(257)[:, None]


def _running(x, op, reverse=False):
    """Running product / sum over the labels (axis 0), one row of nodes per step: much faster than numpy's
    accumulate along an axis for these shapes."""
    out = np.empty_like(x)
    rows = range(len(x) - 1, -1, -1) if reverse else range(len(x))
    prev = None
    for i in rows:
        out[i] = x[i] if prev is None else op(out[prev], x[i])
        prev = i
    return out


def expected_scores(p, objective):
    """p: probabilities sorted in descending order -> expected score of every prefix k = 0..n (array of n + 1)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    n = len(p)
    if objective == "exact":
        with np.errstate(divide="ignore"):
            lp, lq = np.log(p), np.log1p(-p)
        return np.exp(np.concatenate([[0.0], np.cumsum(lp)]) + np.concatenate([np.cumsum(lq[::-1])[::-1], [0.0]]))
    if objective not in ("iou", "f1"):
        raise ValueError(f"unknown objective {objective!r}")
    if n == 0:
        return np.array([1.0])
    if n > 256:
        raise ValueError(f"{n} labels (at most 256)")
    # nodes: IoU's integrands have degree n - 1, F1's at most 2n - 1; m is rounded up to a power of two (cached)
    m = 8
    while 2 * m - 1 < (n - 1 if objective == "iou" else 2 * n - 1):
        m *= 2
    t, wt = _nodes(m)
    # factors 1 - p_j + p_j t lie in [t, 1]: products in linear space can only underflow, where the integrand is
    # negligible next to its values near t = 1 (every term is positive)
    f = 1 + np.multiply.outer(p, t - 1)                                         # [n, m]
    if objective == "iou":
        suf = _running(f, np.multiply, reverse=True)                            # prod_{j >= k}, k = 0..n-1
        inv = np.einsum("km,km->k", wt[:n], np.vstack([suf[1:], np.ones((1, m))]))  # E[1 / (k + B_k)], k = 1..n
        out = np.cumsum(p) * inv
    else:
        loo = p[:, None] / f * np.prod(f, 0)                                    # p_i prod_{j != i} (f_i >= t > 0)
        out = 2 * np.einsum("km,km->k", wt[1: n + 1], _running(loo, np.add))   # k = 1..n
    return np.concatenate([[np.prod(1 - p)], np.minimum(out, 1.0)])  # k = 0: right only if no label applies


def _alone(probs, key, objective):
    """Expected score of selecting exactly {key} (every other label outside the set)."""
    rest = [v for k, v in probs.items() if k != key]
    return float(expected_scores([probs[key]] + rest, objective)[1])


def select(probs, objective="threshold", threshold=0.5, min=None, max=None, exclusive=(), fallback=None,
           fallback_threshold=None):
    """probs: {key: P(label applies)} -> (selected keys, expected score or None for threshold, constraints applied or
    None).

    Label constraints (keys of probs):
      exclusive  labels that can only be selected alone (a "general / no narrowing" category): the best set of the
                 other labels competes with each of them alone, under the objective (threshold: an exclusive label
                 with p >= threshold wins when it is more probable than every label of the thresholded set)
      fallback   a label selected only when nothing else is (implicitly exclusive): under threshold when its p >=
                 fallback_threshold (default: threshold), under iou / f1 / exact when it scores higher alone than the
                 empty set; an explicit fallback_threshold decides under every objective
    The empty set stays a candidate unless min >= 1 (both empty counts as a perfect match)."""
    excl = list(dict.fromkeys(list(exclusive or ()) + ([fallback] if fallback is not None else [])))
    alone = [e for e in excl if e != fallback]
    normal = {k: v for k, v in probs.items() if k not in excl}
    order = sorted(normal, key=normal.get, reverse=True)
    n = len(order)
    one_ok = int(min or 0) <= 1 and (max is None or int(max) >= 1)  # may a single label be the whole set?
    chosen = "labels"
    if objective == "threshold":
        sel = [k for k in order if probs[k] >= threshold]
        if max is not None:
            sel = sel[: int(max)]
        if min is not None and len(sel) < int(min):
            sel = order[: int(min)]
        score = None
        cand = [e for e in alone if probs[e] >= threshold]
        if cand and one_ok:
            e = builtins.max(cand, key=probs.get)
            if not sel or probs[e] > builtins.max(probs[k] for k in sel):
                sel, chosen = [e], "exclusive"
        if fallback is not None and not sel and one_ok:
            if probs[fallback] >= (threshold if fallback_threshold is None else fallback_threshold):
                sel, chosen = [fallback], "fallback"
    else:
        lo = builtins.min(int(min or 0), n)
        hi = n if max is None else int(builtins.min(max, n))
        p = [probs[k] for k in order]
        # the constrained labels sit outside every prefix: the scores of k = 0..n are those of the normal sets
        scores = expected_scores(p + [probs[e] for e in excl], objective)
        if objective == "exact":
            best = int(np.clip(sum(x >= 0.5 for x in p), lo, hi))
        else:
            best = lo + int(np.argmax(scores[lo: hi + 1]))
        sel, score = order[:best], float(scores[best])
        if one_ok:
            for e in alone:
                s = _alone(probs, e, objective)
                if s > score:
                    sel, score, chosen = [e], s, "exclusive"
            if fallback is not None and not sel:
                s = _alone(probs, fallback, objective)
                if (s > score) if fallback_threshold is None else probs[fallback] >= fallback_threshold:
                    sel, score, chosen = [fallback], s, "fallback"
    if not excl:
        return sel, score, None
    applied = dict(exclusive=alone, fallback=fallback, chosen=chosen if sel else "empty")
    if fallback is not None:
        applied["fallback_threshold"] = threshold if fallback_threshold is None and objective == "threshold" \
            else fallback_threshold
    return sel, score, applied


def validate(name, objective, keys, threshold=None, min=None, max=None, exclusive=None, fallback=None,
             fallback_threshold=None):
    """Request checks (the threshold objective keeps the 1.5 handling of min / max)."""
    n = len(keys)
    if objective not in OBJECTIVES:
        raise ValueError(f"question {name!r}: objective must be one of {list(OBJECTIVES)}, got {objective!r}")
    if exclusive is not None:
        if not isinstance(exclusive, list) or not all(isinstance(e, str) for e in exclusive):
            raise ValueError(f"question {name!r}: exclusive must be a list of label keys")
        bad = [e for e in exclusive if e not in keys]
        if bad:
            raise ValueError(f"question {name!r}: exclusive labels {bad} are not among the criteria")
    if fallback is not None and fallback not in keys:
        raise ValueError(f"question {name!r}: fallback {fallback!r} is not among the criteria")
    if fallback_threshold is not None:
        if fallback is None:
            raise ValueError(f"question {name!r}: fallback_threshold needs a fallback label")
        if isinstance(fallback_threshold, bool) or not isinstance(fallback_threshold, (int, float)) \
                or not 0 <= fallback_threshold <= 1:
            raise ValueError(f"question {name!r}: fallback_threshold must be a number in 0..1")
    if objective == "threshold":
        for k, v in (("min", min), ("max", max)):
            if v is not None:
                try:
                    int(v)  # keep the legacy coercion, but reject bad bounds before inference
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(f"question {name!r}: {k} must convert to an integer, got {v!r}") from None
        return
    if threshold is not None:
        raise ValueError(f"question {name!r}: threshold applies only to objective \"threshold\"")
    for k, v in (("min", min), ("max", max)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError(f"question {name!r}: {k} must be a non-negative integer, got {v!r}")
    if min is not None and min > n:
        raise ValueError(f"question {name!r}: min {min} is more than the {n} labels")
    if min is not None and max is not None and min > max:
        raise ValueError(f"question {name!r}: min {min} > max {max}")


def label_set_id(keys, labels, fallback):
    """Stable id of a label set with its fallback (the key of calibration file 2's `multi` section): a hash of the
    sorted (key, label text) pairs and the fallback key, so a fitted fallback_threshold applies only to the same
    labels."""
    pairs = sorted([str(k), str(t)] for k, t in zip(keys, labels))
    blob = json.dumps([pairs, fallback], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
