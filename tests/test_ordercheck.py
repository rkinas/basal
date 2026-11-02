"""basal-bench --order-check (basal/ordercheck.py) with stand-in models: one that reads the options' content, one that
always prefers the first position."""
import re
import sys

import basal.bench as bench_module
from basal import ordercheck
from tests.test_server import Tok

QS = [{"state": "Customer wants a refund.", "question": "Dept?", "options": ["Refunds", "IT", "HR"], "gold": 0},
      {"state": "Customer wants a refund.", "question": "Team?", "options": ["Alpha", "Beta"], "gold": 1}]
WEIGHT = {"Refunds": 0.7, "IT": 0.2, "HR": 0.1, "Alpha": 0.4, "Beta": 0.6}


class Fake:
    tok = Tok()

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def run_shared(self, groups, policy=None):
        self.calls.append([len(g[0]) for g in groups])
        return [[self.fn(p, len(ids)) for p, ids in zip(*g)] for g in groups]


def by_content(prompt, k):
    shown = [re.search(rf"\n{chr(65 + i)}\. (\w+)", prompt).group(1) for i in range(k)]
    w = [WEIGHT[x] for x in shown]
    return [x / sum(w) for x in w]


def first_position(prompt, k):
    return [0.6, 0.3, 0.1][:k] if k == 3 else [0.7, 0.3]


def test_a_model_that_reads_the_content_never_flips():
    be = Fake(by_content)
    r = ordercheck.run(be, QS, 2, [1, 2, 4], lat_n=0)
    assert r["n"] == 2 and r["flip_reversed"] == r["flip_rotated"] == r["flip_any"] == 0
    assert r["acc_given"] == r["acc_mean4"] == 1 and r["agree_mean2"] == r["agree_mean4"] == 1
    assert be.calls == [[4, 2]]  # every order of a question in one group: 3 options -> 4 orders, 2 options -> 2


def test_a_first_position_model_flips_and_averaging_helps():
    r = ordercheck.run(Fake(first_position), QS, 2, [1, 2, 4], lat_n=0)
    assert r["flip_reversed"] == r["flip_rotated"] == r["flip_any"] == 1 and r["first_share"] == 1
    assert r["acc_given"] == r["acc_reversed"] == 0.5 and r["acc_mean1"] == r["acc_given"]
    assert r["acc_mean4"] == 0.5  # Dept?: Refunds 1.4 / 4 over 4 orders; Team?: a tie of both orders, Alpha first
    assert set(r["by_k"]) == {"2", "3-4"} and r["by_k"]["3-4"]["flip_rotated"] == 1


def test_latencies_per_k():
    r = ordercheck.run(Fake(by_content), QS, 2, [1, 4], lat_n=1)
    assert r["lat_mean1_ms"] >= 0 and r["lat_mean4_ms"] >= 0


def test_bench_routes_order_check(monkeypatch):
    seen = []
    monkeypatch.setattr(ordercheck, "main", lambda argv: seen.append(argv))
    monkeypatch.setattr(sys, "argv", ["basal-bench", "--order-check", "--model", "m", "--average", "2"])
    assert bench_module.main() == 0 and seen == [["--model", "m", "--average", "2"]]
