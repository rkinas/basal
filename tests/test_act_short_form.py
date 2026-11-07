"""act's short-form costs {"wrong": W, "defer": D} kept as their two numbers (server.act_spec, expected_costs): no
outcome x outcome matrix at admission (255 outcomes took 1.6 MiB per question, 6.7 GB at the caps), and the expected
costs, the chosen action and its tie order the same floats as from the matrix, term for term."""
import random
import tracemalloc

import pytest

from basal.server import act_answer, act_spec, expected_costs, to_items


def matrix_costs(outcomes, w, d):
    """The short form's matrix as act_spec built it before (the reference)."""
    C = {y: {z: 0.0 if z == y else float(w) for z in outcomes} for y in outcomes}
    if d is not None:
        C["defer"] = {z: float(d) for z in outcomes}
    return C


def matrix_answer(q, probs):
    """act_answer's expected costs and action from the matrix (the code before)."""
    C = matrix_costs(q["keys"], *q["short_ref"])
    exp = {a: sum(probs[z] * c for z, c in row.items()) for a, row in C.items()}
    return exp, min(exp, key=exp.get)


def probabilities(rng, n, how):
    if how == "random":
        p = [rng.random() for _ in range(n)]
    elif how == "ties":
        p = [rng.choice([0.25, 0.5, 0.125]) for _ in range(n)]
    elif how == "one":
        p = [0.0] * n
        p[rng.randrange(n)] = 1.0
    elif how == "uniform":
        p = [1.0] * n
    else:  # tiny and huge spreads
        p = [rng.choice([1e-300, 1e-12, 0.3, 1 - 1e-12]) for _ in range(n)]
    s = sum(p)
    return [x / s for x in p]


COSTS = [(1, None), (2, 1), (1, 0.5), (0.3, 0.1), (5, 5), (0, 0), (1e-9, 3e-10), (7, 0.0), ("2", "1"), (3.5, 2.25),
         (1, 1e-300), (10**6, 1)]


@pytest.mark.parametrize("w, d", COSTS)
def test_expected_costs_and_the_action_are_the_matrix_ones_bit_for_bit(w, d):
    rng = random.Random(f"{w}:{d}")
    for n in (2, 3, 5, 10, 26, 100, 255):
        keys = [f"o{i}" for i in range(n)] if n > 2 else ["true", "false"]
        costs = {"wrong": w} if d is None else {"wrong": w, "defer": d}
        spec = act_spec("q", {"type": "act", "costs": costs}, keys)
        assert spec["costs"] is None and spec["short"] == (float(w), None if d is None else float(d))
        q = {"act": spec, "base": "choice", "keys": keys, "short_ref": (w, d)}
        for how in ("random", "ties", "one", "uniform", "spread"):
            for _ in range(5):
                probs = dict(zip(keys, probabilities(rng, n, how)))
                exp, action = matrix_answer(q, probs)
                ans = act_answer(q, probs, max(probs.values()), {})
                assert ans["action"] == action
                assert ans["expected_costs"] == {a: round(v, 6) for a, v in exp.items()}
                assert list(ans["expected_costs"]) == list(exp)  # (the tie order: the outcomes, then defer)
                got = expected_costs(spec, probs)
                assert all(got[a] == exp[a] and repr(got[a]) == repr(exp[a]) for a in exp)


def test_the_full_form_is_kept_as_given():
    costs = {"pay": {"a": 0, "b": 3}, "hold": {"a": 1, "b": 0}, "human": {"a": 0.5, "b": 0.5}}
    spec = act_spec("q", {"type": "act", "costs": costs}, ["a", "b"])
    assert spec["short"] is None and spec["costs"] == {k: {z: float(c) for z, c in r.items()} for k, r in costs.items()}
    assert spec["defer"] == "human"


def test_a_request_of_many_act_questions_is_read_without_their_matrices():
    crit = {f"k{i}": f"o{i}" for i in range(255)}
    qs = {f"q{j}": {"type": "act", "instructions": "x", "criteria": crit, "costs": {"wrong": 1, "defer": 0.5}}
          for j in range(64)}
    tracemalloc.start()
    try:
        to_items("", qs, 255, force_lang="en")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 * 2**20, peak  # (the matrices: 104 MiB for these 64 questions)
