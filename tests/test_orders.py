"""--orders 1 | 2 | auto and --order-average K: which option orders are asked, and how they are combined."""
import asyncio
import itertools
from types import SimpleNamespace

import pytest

from basal.server import default_orders, option_orders, orders_for, parser, served_orders
from tests.test_server import FakeServer


def server(orders, probs):
    srv = FakeServer(probs)
    srv.orders, srv.seen = orders, []
    inner = srv.probs_for

    def record(prompt, ids):
        srv.seen.append("".join(map(chr, prompt)))
        return inner(prompt, ids)
    srv.probs_for = record
    return srv


def by_first_option(table):
    """Position probabilities by the text of option A (so each order of each question can differ)."""
    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        return next(v for k, v in table.items() if f"A. {k}\n" in text)
    return probs


BODY = {"state": "Customer wants a refund.", "questions": {
    "sure": {"type": "choice", "instructions": "Dept?", "criteria": {"r": "Refunds", "i": "IT", "h": "HR"},
             "option_keys": "hide"},
    "unsure": {"type": "choice", "instructions": "Team?", "criteria": {"a": "Alpha", "b": "Beta"}, "option_keys": "hide"},
    "yn": {"type": "noul", "instructions": "Refund due?"}}}
TABLE = {"Refunds": [0.9, 0.05, 0.05], "HR": [0.05, 0.05, 0.9],      # order 1 / reversed of "sure"
         "Alpha": [0.55, 0.45], "Beta": [0.3, 0.7],                   # "unsure": top p 0.55 < 0.6 in order 1
         "Yes": [0.7, 0.3], "No": [0.4, 0.6]}                         # noul: P(true) 0.7 inside the band


def test_defaults_and_flags():
    assert default_orders(None, 1) == 2 and default_orders(None, 2) == 1
    assert default_orders("1", 1) == 1 and default_orders("auto", 2) == "auto"
    a = parser().parse_args(["--orders", "auto", "--auto-tau", "0.7", "--auto-band", "0.1", "0.9"])
    assert a.orders == "auto" and a.auto_tau == 0.7 and a.auto_band == [0.1, 0.9]
    assert parser().parse_args([]).orders is None


def test_orders_1_asks_the_given_order_only():
    srv = server(1, by_first_option(TABLE))
    ans = asyncio.run(srv._run(BODY))["answers"]
    assert len(srv.seen) == 3 and all("A. Refunds" in s or "A. Alpha" in s or "A. Yes" in s for s in srv.seen)
    assert abs(ans["unsure"]["probabilities"]["a"] - 0.55) < 1e-6 and abs(ans["yn"]["noul"] - 0.7) < 1e-6


def test_orders_2_averages_both_orders():
    srv = server(2, by_first_option(TABLE))
    ans = asyncio.run(srv._run(BODY))["answers"]
    assert len(srv.seen) == 6
    assert abs(ans["unsure"]["probabilities"]["a"] - (0.55 + 0.7) / 2) < 1e-6
    assert abs(ans["yn"]["noul"] - (0.7 + 0.6) / 2) < 1e-6


def test_orders_auto_asks_the_second_order_where_uncertain():
    srv = server("auto", by_first_option(TABLE))
    r = asyncio.run(srv._run(BODY))
    ans = r["answers"]
    assert len(srv.seen) == 5 and r["usage"]["second_order_branches"] == 2
    assert not any("A. HR" in s for s in srv.seen)                      # "sure" is confident in order 1
    assert abs(ans["sure"]["probabilities"]["r"] - 0.9) < 1e-6
    assert abs(ans["unsure"]["probabilities"]["a"] - (0.55 + 0.7) / 2) < 1e-6
    assert abs(ans["yn"]["noul"] - (0.7 + 0.6) / 2) < 1e-6
    assert srv.entries == 2  # SOAM: one packed entry per pass


def test_orders_auto_on_multi_labels():
    p_yes = {"Uszkodzenie": 0.95, "Zwrot": 0.5, "Faktura": 0.05}

    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]
    srv = server("auto", probs)
    body = {"state": "Paczka uszkodzona, chcę zwrotu.", "questions": {"t": {"type": "multi", "instructions": "Które?",
                                                                            "criteria": {"d": "Uszkodzenie", "r": "Zwrot", "i": "Faktura"}}}}
    r = asyncio.run(srv._run(body))
    assert r["usage"]["second_order_branches"] == 1 and len(srv.seen) == 4
    assert any("„Zwrot”" in s and "A. Nie" in s for s in srv.seen)


def test_option_orders_given_reversed_then_rotations():
    assert option_orders(3, 2) == [[0, 1, 2], [2, 1, 0]]
    assert option_orders(4, 4) == [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0], [0, 3, 2, 1]]
    assert sorted(map(tuple, option_orders(3, 99))) == sorted(itertools.permutations(range(3)))  # every one, once
    for k in (3, 4, 5):  # 2k orders: every option at every position twice
        perms = option_orders(k, 2 * k)
        assert len({tuple(p) for p in perms}) == 2 * k
        assert all(sorted(p[pos] for p in perms) == sorted(list(range(k)) * 2) for pos in range(k))
    assert option_orders(4, 12)[8:] == [[0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2]]  # then lexicographic
    assert orders_for(6, 2) == 2 and orders_for(6, 3) == 6 and orders_for(4, 5) == 4 and orders_for(2, 5) == 2


def test_order_average_flag():
    assert parser().parse_args(["--order-average", "4"]).order_average == 4
    assert parser().parse_args([]).order_average is None
    for bad in ("1", "0", "x"):
        with pytest.raises(SystemExit):
            parser().parse_args(["--order-average", bad])
    assert served_orders(SimpleNamespace(orders=None, order_average=None), 2) == 1  # off: as before
    assert served_orders(SimpleNamespace(orders=None, order_average=4), 2) == 4
    with pytest.raises(SystemExit, match="replaces --orders"):
        served_orders(SimpleNamespace(orders="2", order_average=4), 2)


def test_order_average_means_over_k_orders():
    table = {**TABLE, "IT": [0.1, 0.1, 0.8]}
    srv = server(4, by_first_option(table))
    ans = asyncio.run(srv._run(BODY))["answers"]
    assert len(srv.seen) == 4 + 2 + 2  # 3 options: 4 of the 6 orders; 2 options: both orders
    # sure, P(r) in each order: 0.9 (given), 0.9 (reversed), 0.8 (rotation 1, IT first), 0.9 (its reverse)
    assert abs(ans["sure"]["probabilities"]["r"] - (0.9 + 0.9 + 0.8 + 0.9) / 4) < 1e-6
    assert abs(ans["unsure"]["probabilities"]["a"] - (0.55 + 0.7) / 2) < 1e-6
    two = server(2, by_first_option(table))
    assert asyncio.run(two._run(BODY))["answers"]["unsure"] == ans["unsure"]
