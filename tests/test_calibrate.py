"""basal-calibrate without a model: input validation, the temperature fit, the certified thresholds (and their absence),
and the written file served back through basal-serve's calibration loader."""
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest

import basal.server as server_module
from basal.calibrate import (InputError, cp_upper, fit, fit_temperature, gold_index, load, main, threshold,
                             uncalibrated, warnings)
from basal.server import Server, models_payload, parser, temper, to_items
from tests.test_server import FakeServer, Tok

WEIGHT = re.compile(r"^[A-J]\. .*~([0-9.e-]+)$", re.M)


class StubBackend:
    """Engine stand-in: the position probabilities are the weights written at the end of each option ("... ~0.7"), so
    both option orders give the same canonical distribution."""
    policies = {}

    def __init__(self, *_, **__):
        self.tok = Tok()

    def run_shared(self, groups, policy=None):
        out = []
        for prompts, lids in groups:
            res = []
            for p, ids in zip(prompts, lids):
                w = [float(x) for x in WEIGHT.findall("".join(map(chr, p)))][: len(ids)]
                res.append([x / sum(w) for x in w])
            out.append(res)
        return out


SHIPPED = {"temperature_per_prim": {"choice": 1.04, "noul": 1.56, "score": 1.33},
           "thresholds": {"0.01": {"confidence": 0.97}, "0.05": {"confidence": 0.76}}, "source_tag": "shipped"}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """A model directory with a shipped CALIBRATION.json and the stub engine behind --mode eager."""
    md = tmp_path / "model"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md


def synthetic(path, n=900, sharpen=2.0, seed=0):
    """Labelled requests whose engine is over-confident: it serves q ** sharpen (normalised) while the gold answer is
    drawn from q, so the fitted temperatures should be close to `sharpen`. Two consecutive lines share a group."""
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            t = ("choice", "noul", "score")[i % 3]
            k = 2 if t == "noul" else 3 if t == "score" else int(rng.integers(2, 5))
            q = rng.dirichlet([0.3] * k)
            p = np.maximum(q ** sharpen, 1e-12)
            p /= p.sum()
            g = int(rng.choice(k, p=q))
            w = [f"~{x:.12g}" for x in p]
            if t == "choice":
                crit = {f"k{j}": f"option {j} {w[j]}" for j in range(k)}
                gold = f"k{g}" if i % 2 else crit[f"k{g}"]  # by key or by description
                question = dict(type=t, criteria=crit, option_keys="hide", gold=gold)
            elif t == "noul":
                question = dict(type=t, criteria={"true": f"Yes {w[0]}", "false": f"No {w[1]}"},
                                gold="yes" if g == 0 else "no")
            else:
                question = dict(type=t, criteria=[f"level {j} {w[j]}" for j in range(k)], gold=g)
            rec = {"id": f"c{i}", "group": f"g{i // 2}", "state": f"Case {i}.",
                   "questions": {"q": dict(question, instructions="Which one?")}}
            f.write(json.dumps(rec) + "\n")


def test_load_reports_every_problem_with_its_line(tmp_path):
    lines = [
        "{not json",
        json.dumps({"state": "x"}),
        json.dumps({"id": "a", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Late?"}}}),
        json.dumps({"state": "x", "questions": {"q": {"type": "multi", "criteria": ["a", "b"], "gold": "a"}}}),
        json.dumps({"state": "x", "questions": {"q": {"type": "noul", "gold": "maybe"}}}),
        json.dumps({"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"},
                                                       "gold": "c"}}}),
        json.dumps({"state": "x", "questions": {"q": {"type": "choice", "criteria": ["a"], "gold": "a"}}}),
        json.dumps({"state": "x", "facts": "yes", "questions": {"q": {"type": "noul", "gold": "yes"}}}),
        json.dumps({"state": "x", "questions": {"q": {"type": "score", "criteria": ["lo", "hi"], "gold": 2}}}),
    ]
    (tmp_path / "bad.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(InputError) as e:
        load(tmp_path / "bad.jsonl")
    msg = str(e.value)
    assert "9 problem(s)" in msg
    for needle in ("line 1: not JSON", 'line 2: a request needs "state"', "line 3 (id a), question 'q': no \"gold\"",
                   "line 4, question 'q': multi needs a \"fallback\" label", "line 5, question 'q': gold must be \"yes\"",
                   "line 6, question 'q': gold 'c' is not an option", "line 7, question 'q': 1 options (supported",
                   'line 8: facts must be "off" or "auto"', "line 9, question 'q': gold level 2 is out of range"):
        assert needle in msg, needle


def test_gold_forms_and_groups(tmp_path):
    def idx(q):
        return gold_index(q, to_items("State.", {"q": q})[0])
    choice = {"type": "choice", "criteria": {"ret": "Returns", "it": "IT"}}
    assert idx(dict(choice, gold="it")) == 1 and idx(dict(choice, gold="Returns")) == 0
    assert idx(dict(choice, gold="it: IT")) == 1                              # the shown option text
    assert idx({"type": "choice", "criteria": ["approve", "reject"], "gold": "reject"}) == 1
    for g, want in (("yes", 0), ("No", 1), (True, 0), (False, 1), ("true", 0), ("tak", 0)):
        assert idx({"type": "noul", "gold": g}) == want
    score = {"type": "score", "criteria": ["low", "mid", "high"]}
    assert idx(dict(score, gold=2)) == 2 and idx(dict(score, gold="mid")) == 1 and idx(dict(score, gold="0")) == 0
    assert idx({"type": "score", "criteria": {"lo": "minor", "hi": "severe"}, "gold": "severe"}) == 1
    act = {"type": "act", "criteria": {"true": "Covered", "false": "Not covered"}, "costs": {"wrong": 9, "defer": 1}}
    assert idx(dict(act, gold="no")) == 1
    with pytest.raises(ValueError, match="several options"):
        idx({"type": "choice", "criteria": {"a": "same", "b": "same"}, "gold": "same"})

    (tmp_path / "ok.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"state": "s", "questions": {"a": {"type": "noul", "gold": "yes", "evidence": True},
                                     "b": {"type": "noul", "gold": "no", "group": "twin-1"}}},
        {"id": "x", "group": "twin-1", "state": "s", "questions": {"c": {"type": "noul", "gold": "no"}}}]) + "\n",
                                     encoding="utf-8")
    bodies, decisions = load(tmp_path / "ok.jsonl")
    examples = Path(server_module.__file__).parent / "examples"                           # basal-run simple items
    assert [len(load(examples / f"{t}.jsonl")[1]) for t in ("choice", "noul", "score")] == [8, 8, 8]
    assert [d["gold"] for d in load(examples / "noul.jsonl")[1]][:2] == [0, 1]           # gold index 0 = yes
    assert [d["group"] for d in decisions] == ["1", "twin-1", "twin-1"]       # default: one group per line
    assert bodies[0]["questions"]["a"] == {"type": "noul"}                     # gold, group, evidence are not sent
    assert "group" not in bodies[1] and "id" not in bodies[1]


def test_temperature_fit_recovers_over_and_under_confidence():
    rng = np.random.default_rng(1)
    for true_T in (2.0, 0.5):
        rows = []
        for _ in range(3000):
            z = rng.normal(0, 1.5, 4)
            q = np.exp(z) / np.exp(z).sum()
            p = q ** true_T / (q ** true_T).sum()
            rows.append({"p": list(p), "gold": int(rng.choice(4, p=q))})
        assert abs(fit_temperature(rows) / true_T - 1) < 0.08


def test_clopper_pearson_bound():
    for n in (50, 298, 299, 1000):
        assert abs(cp_upper(0, n) - (1 - 0.05 ** (1 / n))) < 1e-9
    assert cp_upper(0, 299) <= 0.01 < cp_upper(0, 298)
    assert abs(cp_upper(1, 100) - 0.04656) < 1e-4          # Beta(0.95; 2, 99)
    assert cp_upper(5, 400) > 5 / 400


def test_threshold_is_certified_and_keeps_ties_together():
    # 400 correct decisions at 0.99, then a tie of 50 at 0.9 holding 10 errors, then a long correct tail at 0.8
    cs = [(0.99, True)] * 400 + [(0.9, i >= 10) for i in range(50)] + [(0.8, True)] * 600
    assert threshold(cs, 0.01) == 0.99                     # including 0.9 breaks the bound; the tail is not reached
    assert cp_upper(10, 450) > 0.01
    assert threshold(cs, 0.05) == 0.8                      # 10 / 1050 errors: the bound holds all the way down
    ties = [(0.95, True)] * 60 + [(0.95, False)] * 4       # one confidence level: accepted or refused as a whole
    assert threshold(ties, 0.05) is None and threshold(ties, 0.2) == 0.95


def test_no_certified_threshold_when_the_bound_cannot_be_met():
    rows = [{"p": [0.999, 0.001], "gold": 0, "prim": "noul", "group": f"g{i}"} for i in range(200)]
    cal, _ = fit(rows, SHIPPED, [0.01, 0.05], min_per_type=1000, min_blend=0)
    assert cal["temperature_per_prim"] == SHIPPED["temperature_per_prim"]
    assert cal["temperature_source"]["noul"].startswith("shipped") and "choice" in cal["temperature_source"]
    edge = fit(rows, SHIPPED, [0.05], min_per_type=10)[1]          # no errors: the NLL wants the lowest temperature
    assert any("edge of the search (0.25)" in w for w in warnings(edge, [0.05]))
    none, five = cal["thresholds"]["0.01"], cal["thresholds"]["0.05"]
    assert none["confidence"] is None and "at least 299 accepted FIT decisions" in none["reason"]
    assert five["confidence"] is not None and five["fit_upper95"] <= 0.05
    # served: the same position probabilities in both orders -> P = 0.5; answering (cost 1) beats deferring (1.5)
    q = {"type": "act", "instructions": "Covered?", "criteria": {"true": "Yes", "false": "No"},
         "costs": {"wrong": 2, "defer": 1.5}}
    ask = lambda r: asyncio.run(FakeServer([0.9, 0.1], cal=cal)._run(
        {"state": "x", "questions": {"r": dict(q, max_error=r)}}))["answers"]["r"]
    for r in (0.01, 0.05):  # no certified threshold, or P = 0.5 below the certified one: both defer
        assert ask(r)["action"] == "defer" and ask(r)["refused"] in ("true", "false")
    assert ask(0.01)["max_error"] == "no certified threshold for this type at this max_error" and "max_error" not in ask(0.05)
    assert ask(0.01)["deferred"] is True and ask(0.05)["deferred"] is True


def test_calibrate_end_to_end_round_trips_through_the_server(tmp_path, engine, capsys):
    synthetic(tmp_path / "labelled.jsonl")
    out = tmp_path / "CALIBRATION.custom.json"
    main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "max_error on CERT" in printed and "basal-serve --model stub --mode eager --calibration" in printed
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["fitted_on"]["decisions"] == 900 and cal["fitted_on"]["engine"] == "eager"
    assert cal["fitted_on"]["fit"] + cal["fitted_on"]["cert"] == 900
    assert cal["baseline"] == "stub/CALIBRATION.json (shipped)"
    for t in ("choice", "noul", "score"):
        assert cal["temperature_source"][t].startswith("fitted")
        assert 1.5 < cal["temperature_per_prim"][t] < 2.7    # true 2.0 (shipped 1.04-1.56), ~200 FIT decisions
    thr = cal["thresholds"]["0.05"]["confidence"]
    assert thr is not None and cal["thresholds"]["0.05"]["fit_upper95"] <= 0.05

    srv = Server(parser().parse_args(["--mode", "eager", "--calibration", str(out)]))  # basal-serve's loader
    assert srv.cal == cal and srv.temps == cal["temperature_per_prim"]

    async def decide(body):
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    for p_yes in (0.6, 0.99999):  # answering is cheaper than deferring in both; the threshold decides
        q = {"type": "act", "instructions": "Covered?", "costs": {"wrong": 2, "defer": 1}, "max_error": 0.05,
             "criteria": {"true": f"Yes ~{p_yes}", "false": f"No ~{1 - p_yes}"}}
        ans = asyncio.run(decide({"state": "x", "questions": {"r": q}}))["answers"]["r"]
        p = temper(server_module.torch.tensor([p_yes, 1 - p_yes]), cal["temperature_per_prim"]["noul"])
        conf = max(p.tolist())
        assert abs(ans["confidence"] - conf) < 1e-7
        assert ("refused" in ans) == (conf < thr) and ans["calibration"] == "validated"


@pytest.mark.parametrize("snapshot", [True, False])
def test_a_cache_snapshot_is_recorded_as_its_repo_id_and_commit(tmp_path, monkeypatch, capsys, snapshot):
    """--model <Hugging Face cache snapshot> is named as basal-serve names it (repo id and commit), so the file is
    served for that snapshot under --strict-calibration; a plain directory keeps its path and has no revision."""
    sha = "0123456789abcdef0123456789abcdef01234567"
    md = tmp_path / (f"models--Remek--x/snapshots/{sha}" if snapshot else "plain")
    md.mkdir(parents=True)
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    synthetic(tmp_path / "labelled.jsonl")
    out = tmp_path / "CALIBRATION.custom.json"
    main([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", str(md), "--out", str(out)])
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    if snapshot:
        assert (fitted["model"], fitted["revision"]) == ("Remek/x", sha)
    else:
        assert fitted["model"] == str(md) and "revision" not in fitted
    capsys.readouterr()
    srv = Server(parser().parse_args(["--mode", "eager", "--model", str(md), "--calibration", str(out),
                                      "--strict-calibration"]))
    assert srv.cal_mismatch is None and "warning" not in capsys.readouterr().out


def test_url_mode_needs_an_uncalibrated_server(capsys):
    entry = lambda temps: models_payload("basal-1.5-mini", "mlx", {}, temps)["models"][0]
    with pytest.raises(SystemExit, match="--no-calibration"):
        uncalibrated(entry({"choice": 1.04, "noul": 1.56}), "http://h")
    assert uncalibrated(entry({}), "http://h")["mode"] == "mlx"
    uncalibrated({"name": "basal-1.5-mini", "mode": "mlx"}, "http://h")         # a 1.5.0 server: warned, not refused
    assert "make sure it runs with --no-calibration" in capsys.readouterr().err
