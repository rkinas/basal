"""The certified threshold is a fixed-sequence test (basal/calibrate.py threshold): under the null, where every set of
accepted decisions has an error above the target, it certifies a threshold at most alpha of the time; the scan that
went on past failing levels until one passed did not. Seeded Monte Carlo, and the exact chance of the first test."""
import math
import time

import numpy as np
import pytest

from basal import calibrate
from basal.calibrate import (ALPHA, accepted_needed, cp_upper, fit, fit_tasks, fit_temperature, levels,
                             no_threshold_reason, per_type_thresholds, served, threshold)


def tail(e, n, q):
    """P(X <= e) for X ~ Binomial(n, q), computed here on its own (not with the code under test)."""
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(q)
                        + (n - k) * math.log1p(-q)) for k in range(e + 1))


def table(n_max, risk):
    """e_max[n]: the most errors among n accepted decisions that still meet the bound (-1: none)."""
    out, e = [-1] * (n_max + 1), -1
    for n in range(1, n_max + 1):
        while e + 1 <= risk * n and tail(e + 1, n, risk) <= ALPHA:
            e += 1
        out[n] = e
    return np.array(out)


def scan_past_failures(wrong, e_max, min_n=50):
    """The scan before the fixed sequence (distinct confidences, most confident first): failing levels are skipped
    until one passes, then it stops at the first failure."""
    cum, thr = np.cumsum(wrong), None
    for n in range(min_n, len(wrong) + 1):
        if cum[n - 1] <= e_max[n]:
            thr = n
        elif thr is not None:
            break
    return thr


@pytest.mark.parametrize("risk,p,n", [(0.05, 0.051, 800), (0.05, 0.051, 150), (0.01, 0.0105, 800),
                                       (0.01, 0.0105, 2000)])
def test_false_certification_under_the_null_is_at_most_alpha(risk, p, n):
    trials = 1500
    rng = np.random.default_rng(20261104)
    conf = np.linspace(0.999, 0.501, n).tolist()
    e_max = table(n, risk)
    hits = old = 0
    for _ in range(trials):
        wrong = rng.random(n) < p          # every set of accepted decisions has an error rate p > risk
        hits += threshold(list(zip(conf, (~wrong).tolist())), risk) is not None
        old += scan_past_failures(wrong, e_max) is not None
    assert hits / trials <= ALPHA
    if n >= 800:
        assert old / trials > ALPHA        # the same data: scanning on past failures certifies too often


@pytest.mark.parametrize("risk,n", [(0.05, 800), (0.05, 150), (0.01, 800), (0.01, 2000)])
def test_only_the_first_level_tested_can_open_the_sequence(risk, n):
    """Exact: the first level tested is set by the counts, and at any error rate >= risk it passes at most alpha of
    the time; every later level is tested only after it passed."""
    conf = np.linspace(0.999, 0.501, n).tolist()
    lv = levels([(c, True) for c in conf])
    first = lv[calibrate.start_level(lv, risk)][1]
    assert first in (accepted_needed(risk), accepted_needed(risk, 1))
    e_max = table(first, risk)
    for q in (risk, 1.02 * risk, 2 * risk):
        assert tail(int(e_max[first]), first, q) <= ALPHA


def test_ties_and_an_error_rate_that_grows_towards_the_top_stay_under_alpha():
    rng = np.random.default_rng(7)
    conf = np.round(np.linspace(0.999, 0.6, 600), 2).tolist()   # 40 levels of equal confidences
    p = np.where(np.arange(600) < 100, 0.08, 0.051)             # the top is worse than the rest
    hits = sum(threshold(list(zip(conf, (rng.random(600) >= p).tolist())), 0.05) is not None for _ in range(2000))
    assert hits / 2000 <= ALPHA


def test_the_first_level_is_set_by_the_counts_and_its_failure_ends_the_test():
    conf = np.linspace(0.999, 0.5, 2000).tolist()
    lv, start_level = levels([(c, True) for c in conf]), calibrate.start_level
    assert lv[start_level(lv, 0.05)][1] == accepted_needed(0.05, 1) == 93       # 2000 >= 2 x 93: one error allowed
    small = levels([(c, True) for c in conf[:150]])
    assert small[start_level(small, 0.05)][1] == accepted_needed(0.05) == 59    # 150 < 186: none allowed
    cs = [(c, i not in (10, 20)) for i, c in enumerate(conf)]                    # two errors among the top 93
    assert start_level(levels(cs), 0.05) == start_level(lv, 0.05)                # outcomes do not move the start
    assert threshold(cs, 0.05) is None                      # 124 accepted with 2 errors would pass: never tested
    assert cp_upper(2, 124) <= 0.05 < cp_upper(2, 93)
    reason = no_threshold_reason(cs, 0.05)
    assert "first level tested" in reason and "top 93 FIT decisions, 2 errors" in reason
    assert threshold([(c, i != 10) for i, c in enumerate(conf)], 0.05) == conf[-1]   # one error there: goes on
    assert threshold([(c, i != 10) for i, c in enumerate(conf[:150])], 0.05) is None  # the small set: it ends


def test_within_is_the_clopper_pearson_bound():
    for e, n in ((0, 59), (0, 58), (1, 93), (1, 92), (10, 450), (12, 3240), (3, 50)):
        for risk in (0.01, 0.05, 0.2):
            assert calibrate.within(e, n, risk) == (cp_upper(e, n) <= risk) == (e <= risk * n and tail(e, n, risk)
                                                                                 <= ALPHA)


def test_a_scan_over_fifty_thousand_decisions_takes_well_under_a_second():
    """The bound of every level reused one O(n) table copy, so a type or task with tens of thousands of decisions that
    passes far down took minutes; levels that add no error need no bound at all."""
    rng = np.random.default_rng(3)
    c = rng.uniform(0.5, 1.0, 50000).round(6)
    cs = list(zip(c.tolist(), (~((c < 0.7) & (rng.random(50000) < 0.02))).tolist()))  # errors below 0.7 only
    t = time.perf_counter()
    assert threshold(cs, 0.05) is not None and threshold(cs, 0.01) is not None
    assert time.perf_counter() - t < 5


def test_learned_temperatures_keep_false_certification_under_alpha():
    """For choice decisions (three to five options) the temperature, fitted on the same FIT decisions, can reorder
    their confidences. With every decision wrong with the same chance just above the target (so every order is a
    null), the certificate on the confidences as served is still issued at most alpha of the time."""
    rng = np.random.default_rng(11)
    trials, n, risk, p = 400, 300, 0.05, 0.055
    hits = moved = 0
    for _ in range(trials):
        rows, logits = [], []
        for i in range(n):
            z = rng.normal(0.0, 2.5, rng.integers(3, 6))
            q = np.exp(z - z.max())
            q /= q.sum()
            top = int(q.argmax())
            gold = top if rng.random() >= p else int(rng.choice([j for j in range(len(q)) if j != top]))
            rows.append(dict(prim="choice", p=q.tolist(), gold=gold, group=f"g{i}"))
            logits.append(np.log(q))
        T = fit_temperature(rows)
        conf = [float(np.exp(z / T - np.logaddexp.reduce(z / T)).max()) for z in logits]
        ok = [int(np.argmax(z)) == r["gold"] for z, r in zip(logits, rows)]
        hits += threshold(list(zip(conf, ok)), risk) is not None
        raw = [float(np.exp(z).max()) for z in logits]
        moved += np.argsort(conf, kind="stable").tolist() != np.argsort(raw, kind="stable").tolist()
    assert moved > trials / 2                  # the learned temperature does reorder the confidences
    assert hits / trials <= ALPHA


def test_choice_thresholds_are_certified_on_the_served_confidences():
    """A temperature can swap two choice decisions (three options or more): the levels are those basal-serve gates."""
    a, b = np.exp([0.0, -3.0, -3.0]), np.exp([0.0, -2.0, -20.0])
    a, b = (a / a.sum()).tolist(), (b / b.sum()).tolist()
    rows = [dict(prim="choice", p=a, gold=0, group=f"a{i}") for i in range(120)] + \
           [dict(prim="choice", p=b, gold=1, group=f"b{i}") for i in range(120)]
    (ca, _), (cb, _) = served([rows[0]], {"choice": 1.0})[0], served([rows[-1]], {"choice": 1.0})[0]
    (ta, _), (tb, _) = served([rows[0]], {"choice": 10.0})[0], served([rows[-1]], {"choice": 10.0})[0]
    assert ca > cb and ta < tb                              # the order of the two confidences swaps
    one = per_type_thresholds(rows, [], {"choice": 1.0}, {}, 0.05)["choice"]
    ten = per_type_thresholds(rows, [], {"choice": 10.0}, {}, 0.05)["choice"]
    assert one["confidence"] is not None and one["fit_error"] == 0.0     # the correct ones first at T = 1
    assert ten["confidence"] is None                                     # the wrong ones first at T = 10
    assert fit(rows, {}, [0.05], min_per_type=10 ** 9, min_blend=0)[0]["thresholds"]["0.05"]["per_type"]["choice"][
        "confidence"] == one["confidence"]


def test_fit_tasks_uses_the_same_test():
    conf = np.linspace(0.999, 0.5, 2000).tolist()
    rows = [dict(task="verify", group=f"g{i}", c=c, ok=i not in (10, 20)) for i, c in enumerate(conf)]
    thr, rep = fit_tasks(rows, [0.05])
    assert thr == {} and "first level tested" in rep["verify"]["targets"]["0.05"]["reason"]
