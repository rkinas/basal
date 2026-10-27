"""Conformal answer sets (basal-calibrate --conformal, "coverage" on a question) and the abstain flag ("max_error" on a
choice / noul / score question): the split-conformal p_min, the set rule, coverage on held-out decisions, and that both
are off unless asked for."""
import asyncio
import json

import numpy as np
import pytest

from basal.calibrate import conformal_needed, conformal_p_min, fit, main
from basal.server import NO_CERTIFICATE, NO_CONFORMAL, Server, answer_set, load_calibration, parser
from tests.test_calibrate import SHIPPED, engine, synthetic  # noqa: F401  (engine: a pytest fixture)


def rows_of(n, seed=0, k=4, prim="choice"):
    """Calibrated rows: the gold answer is drawn from the served probabilities themselves."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = rng.dirichlet([0.5] * (2 if prim == "noul" else k))
        out.append(dict(p=[float(x) for x in p], gold=int(rng.choice(len(p), p=p)), prim=prim, group=f"g{seed}-{i}"))
    return out


def test_p_min_is_the_conformal_quantile():
    p = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]           # n = 9
    assert conformal_p_min(p, 0.9) == 0.1                         # k = ceil(10 * 0.9) = 9: the 9th largest
    assert conformal_p_min(p, 0.8) == 0.2                         # k = 8
    assert conformal_p_min(p, 0.5) == 0.5                         # k = 5
    assert conformal_p_min(p[:8], 0.9) is None                    # k = 9 > 8: no p_min, the set is every option
    assert (conformal_needed(0.8), conformal_needed(0.9), conformal_needed(0.95)) == (4, 9, 19)


def test_answer_set_rule():
    assert answer_set([0.2, 0.5, 0.3], 0.25) == [1, 2]            # most probable first
    assert answer_set([0.2, 0.5, 0.3], 0.9) == [1]                # the top answer is always in
    assert answer_set([0.2, 0.5, 0.3], None) == [1, 2, 0]         # no p_min: every option
    assert answer_set([0.4, 0.2, 0.4], 0.3) == [0, 2]             # ties in option order


def test_fit_conformal_holds_its_coverage_on_held_out_decisions():
    rows = rows_of(6000, 1) + rows_of(3000, 2, prim="noul") + rows_of(3000, 3, k=5, prim="score")
    off, _ = fit(rows, SHIPPED, [0.05], min_per_type=50)
    on, report = fit(rows, SHIPPED, [0.05], min_per_type=50, conformal=[0.8, 0.9, 0.95])
    assert "conformal" not in off and {k: v for k, v in on.items() if k != "conformal"} == off  # off by default
    assert report["conformal"] is on["conformal"] and on["conformal"]["method"].startswith("split conformal on CERT")
    sizes = []
    for level in (0.8, 0.9, 0.95):
        per = on["conformal"]["levels"][str(level)]["per_type"]
        assert set(per) == {"choice", "noul", "score"}
        for t, e in per.items():
            assert e["cert_coverage"] >= level                     # in-sample on CERT: >= L by construction
            assert level - 0.03 <= e["fit_coverage"] <= level + 0.04, (level, t, e)  # held out from p_min
            assert e["n"] + e["fit_n"] == sum(r["prim"] == t for r in rows)
        sizes.append(per["choice"]["fit_mean_size"])
    assert sizes == sorted(sizes) and sizes[0] > 1                 # a higher coverage asks for larger sets


def test_too_few_cert_decisions_give_every_option():
    rows = rows_of(20, 4)                                          # ~6 CERT decisions: none reaches 0.9 (needs 9)
    cal, _ = fit(rows, SHIPPED, [0.05], min_per_type=1000, min_blend=0, conformal=[0.9])
    e = cal["conformal"]["levels"]["0.9"]["per_type"]["choice"]
    assert e["p_min"] is None and "needs >= 9" in e["reason"] and e["fit_coverage"] == 1.0


@pytest.fixture
def served(tmp_path, engine):  # noqa: F811
    """basal-serve (stub engine) with a calibration that has answer sets and per-type thresholds."""
    cal = {"temperature_per_prim": {"choice": 1.0, "noul": 1.0, "score": 1.0},
           "thresholds": {"0.05": {"confidence": 0.7, "per_type": {"choice": {"confidence": 0.7},
                                                                    "noul": {"confidence": 0.9}}}},
           "conformal": {"levels": {"0.9": {"per_type": {"choice": {"p_min": 0.15}, "noul": {"p_min": 0.05},
                                                         "score": {"p_min": None}}}}}}
    path = tmp_path / "CAL.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    srv = Server(parser().parse_args(["--mode", "eager", "--calibration", str(path)]))

    def ask(**q):
        async def go():
            w = asyncio.get_running_loop().create_task(srv.worker())
            try:
                return await srv.decide({"state": "x", "questions": {"r": q}})
            finally:
                w.cancel()
        return asyncio.run(go())["answers"]["r"]
    return ask


CHOICE = dict(type="choice", instructions="Which?", option_keys="hide",
              criteria={"a": "first ~0.6", "b": "second ~0.3", "c": "third ~0.1"})


def test_serve_answer_set_and_abstain(served):
    plain = served(**CHOICE)
    assert not {"answer_set", "coverage", "abstain", "max_error"} & set(plain)   # nothing asked, nothing added
    ans = served(**CHOICE, coverage=0.9)
    assert ans["answer_set"] == ["a", "b"] and ans["coverage"] == 0.9 and "coverage_note" not in ans
    assert {k: v for k, v in ans.items() if k not in ("answer_set", "coverage")} == plain
    noul = served(type="noul", instructions="Yes?", criteria={"true": "yes ~0.97", "false": "no ~0.03"}, coverage=0.9)
    assert noul["answer_set"] == ["true"] and abs(noul["probabilities"]["true"] - 0.97) < 1e-6  # no: 0.03 < 0.05
    score = served(type="score", instructions="How much?", criteria=["low ~0.2", "mid ~0.5", "high ~0.3"], coverage=0.9)
    assert score["answer_set"] == ["1", "2", "0"] and score["coverage_note"] == NO_CONFORMAL  # p_min null
    other = served(**CHOICE, coverage=0.8)                          # a level the file does not have: every option
    assert other["answer_set"] == ["a", "b", "c"] and other["coverage_note"] == NO_CONFORMAL
    # abstain: below the type's certified threshold, or no certificate for the target
    assert served(**CHOICE, max_error=0.05)["abstain"] is True       # 0.6 < 0.7
    top = dict(CHOICE, criteria={"a": "first ~0.8", "b": "second ~0.15", "c": "third ~0.05"})
    ok = served(**top, max_error=0.05)
    assert ok["abstain"] is False and "max_error" not in ok
    none = served(**top, max_error=0.01)
    assert none["abstain"] is True and none["max_error"] == NO_CERTIFICATE
    assert served(type="score", instructions="How much?", criteria=["low ~0.95", "mid ~0.04", "high ~0.01"],
                  max_error=0.05)["abstain"] is True                 # no per-type score threshold in the file


def test_serve_refuses_bad_limits(served):
    for bad in (1.0, 0, -0.1, True, "0.9"):
        with pytest.raises(ValueError, match="coverage must be a number between 0 and 1"):
            served(**CHOICE, coverage=bad)
    with pytest.raises(ValueError, match="max_error must be a number between 0 and 1"):
        served(**CHOICE, max_error=2)
    act = {"type": "act", "instructions": "Covered?", "criteria": {"true": "Yes ~0.9", "false": "No ~0.1"},
           "costs": {"wrong": 2, "defer": 1}}
    with pytest.raises(ValueError, match="coverage is supported on choice, noul and score"):
        served(**act, coverage=0.9)


def test_conformal_levels_are_dropped_with_an_unvalidated_calibration(tmp_path):
    cal = {"fitted_on": {"orders": 1}, "temperature_per_prim": {"choice": 1.2}, "conformal": {"levels": {}},
           "thresholds": {}}
    path = tmp_path / "CAL.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    data, _, why = load_calibration(tmp_path, "eager", str(path), None, 2)
    assert why and "conformal" not in data and "thresholds" not in data and data["temperature_per_prim"]


def test_calibrate_conformal_round_trips_through_the_server(tmp_path, engine, capsys):  # noqa: F811
    synthetic(tmp_path / "labelled.jsonl", n=1200)
    out = tmp_path / "CAL.json"
    main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out),
          "--conformal", "0.9,0.8"])
    printed = capsys.readouterr().out
    assert "answer sets" in printed and "coverage 0.9" in printed
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert list(cal["conformal"]["levels"]) == ["0.8", "0.9"]
    p_min = cal["conformal"]["levels"]["0.9"]["per_type"]["choice"]["p_min"]
    srv = Server(parser().parse_args(["--mode", "eager", "--calibration", str(out)]))
    q = dict(CHOICE, coverage=0.9)

    async def go():
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide({"state": "x", "questions": {"r": q}})
        finally:
            w.cancel()
    ans = asyncio.run(go())["answers"]["r"]
    p = ans["probabilities"]
    assert ans["answer_set"] == [k for k in sorted(p, key=p.get, reverse=True)
                                 if k == ans["choice"] or p[k] >= p_min]
    main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out)])
    assert "conformal" not in json.loads(out.read_text(encoding="utf-8"))   # off unless asked for
    with pytest.raises(SystemExit, match="--conformal levels must lie between 0 and 1"):
        main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", "stub", "--conformal", "0.9,1.2"])
