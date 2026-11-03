"""Calibration file 2's gates (leave-one-family-out ECE next to a reference, coverage at the certified thresholds) and
its `multi` section (fallback_threshold fitted per label set), and basal-serve using the fitted threshold."""
import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import basal.calibrate as calibrate_module
import basal.server as server_module
from basal.calibrate import (REFERENCE, accepted_needed, cert_note, fit, fit_fallback, is_cert, lofo, main,
                             sample_size, set_metrics, set_probs)
from basal.sets import label_set_id
from basal.server import to_items
from tests.test_server import FakeServer, Tok

WEIGHT = re.compile(r"^[A-J]\. .*~([0-9.e-]+)$", re.M)
LABEL = re.compile(r'Does this apply: "(.*?)"\?')


def rows(n, k, sharpen, family, prim="choice", seed=0):
    """Rows of an engine that serves q ** sharpen while the gold answer is drawn from q (sharpen 1: calibrated)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = rng.dirichlet([0.5] * k)
        p = np.maximum(q ** sharpen, 1e-12)
        out.append(dict(prim=prim, p=list(p / p.sum()), gold=int(rng.choice(k, p=q)), group=f"{family}-{seed}-{i}",
                        family=family))
    return out


def split(rs):
    return [r for r in rs if not is_cert(r["group"])], [r for r in rs if is_cert(r["group"])]


def test_lofo_holds_each_family_out_of_its_own_temperature():
    """Two over-confident families and a calibrated one: the calibrated family, scored with the temperature of the
    other two, is the worst; a family's own decisions never fit its temperature."""
    data = rows(600, 3, 2.0, "a") + rows(600, 3, 2.0, "b", seed=1) + rows(600, 3, 1.0, "c", seed=2)
    lf = lofo(*split(data), {}, {}, 50, 1, reference=REFERENCE["basal-1.5"])["choice"]
    assert lf["n_families"] == 3 and lf["worst_family"] == "c" and lf["n"] == len(split(data)[1])
    assert lf["max"] > lf["mean_floor"] and 0 < lf["pooled"] < lf["max"]
    assert lf["reference"] == 0.082 and lf["regression"] == (lf["mean"] > 0.082)
    same = lofo(*split(rows(600, 3, 2.0, "a") + rows(600, 3, 2.0, "b", seed=1)), {}, {}, 50, 1)["choice"]
    assert same["mean"] < lf["mean"] and same["regression"] is None   # no reference: not evaluated
    one = lofo(*split(rows(300, 3, 2.0, "a")), {}, {}, 50, 1)["choice"]
    assert one["mean"] is None and one["reason"].startswith("one family")
    small = lofo(*split(rows(300, 3, 2.0, "a") + rows(40, 3, 2.0, "b", seed=1)), {}, {}, 50, 1, min_family=30)["choice"]
    assert small["n_families"] == 1 and small["n"] > 0                # family b has < 30 CERT decisions: pooled only


def test_gates_in_the_file():
    data = rows(900, 3, 2.0, "a") + rows(900, 3, 2.0, "b", seed=1) + rows(600, 2, 1.5, "a", "noul", 2)
    cal, report = fit(data, {}, [0.01, 0.05], 50, fmt=2, reference={"choice": 0.5, "noul": 0.0001})
    g = cal["gates"]
    assert set(g["ece_lofo"]) == {"choice", "noul"} and g["ece_lofo"]["choice"]["regression"] is False
    assert g["ece_lofo"]["noul"]["regression"] is True
    for risk in ("0.01", "0.05"):
        thr = cal["thresholds"][risk]["confidence"]
        want = report["targets"][float(risk)]["fitted_cert"]["coverage"] if thr is not None else 0.0
        assert g["automated_at_pooled"][risk] == want
        assert set(cal["thresholds"][risk]["per_type"]) == {"choice", "noul"}
    assert g["automated_at"]["0.05"] > 0 and g["automated_error"]["0.05"] is not None
    one, _ = fit(data, {}, [0.05], 50, fmt=1)   # format 1 files get the gates too (ignored by older servers)
    assert "gates" in one and "multi" not in one


# -- multi fallback ----------------------------------------------------------------------------------------------------

KEYS = ["d", "r", "i", "g"]


def set_rows(n, seed=0, sid="ls"):
    """A label set whose fallback "g" (general) is the right answer in 40% of the questions but rarely reaches 0.5:
    the default rule leaves those questions empty."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = rng.choice(["general", "specific", "empty"], p=[0.4, 0.45, 0.15])
        p = rng.uniform(0.01, 0.15, 4)
        if kind == "general":
            p[3], gold = rng.uniform(0.05, 0.45), [3]
        elif kind == "specific":
            j = int(rng.integers(0, 3))
            p[j], gold = rng.uniform(0.55, 0.99), [j]
        else:
            gold = []
        out.append(dict(prim="multi", p=list(p), gold=gold, keys=KEYS, group=f"s{seed}-{i}", family="m",
                        set=dict(id=sid, objective="threshold", threshold=0.5, min=None, max=None, exclusive=[],
                                 fallback="g")))
    return out


def test_fallback_threshold_fit_trades_empty_answers_for_iou():
    fitted = fit_fallback(set_rows(600), 1.0)["ls"]
    thr, c = fitted["fallback_threshold"], fitted["cert"]
    assert thr is not None and 0.01 <= thr <= 0.2
    assert c["iou"] > c["iou_default"] + 0.2 and c["false_empty"] < c["false_empty_default"]
    assert c["empty_accuracy"] <= c["empty_accuracy_default"]          # the price: empty answers become general
    assert fitted["n_fit"] + fitted["n_cert"] == 600 and fitted["labels"] == KEYS
    few = fit_fallback(set_rows(20), 1.0)["ls"]
    assert few["fallback_threshold"] is None and "FIT questions" in few["reason"]
    # the probabilities are those served with the noul temperature of the same file
    r = set_rows(1)[0]
    assert set_probs(r, 1.0)["g"] == pytest.approx(r["p"][3], abs=1e-6)
    assert set_probs(r, 2.0)["g"] > r["p"][3]                           # T > 1 pulls toward 0.5
    assert set_metrics([(r, set_probs(r, 1.0))], 0.0)["iou"] in (0.0, 1.0)


def test_fit_writes_the_multi_section():
    data = rows(300, 2, 1.5, "a", "noul") + set_rows(600) + set_rows(10, seed=1, sid="small")
    cal, report = fit(data, {}, [0.05], 50, fmt=2)
    assert set(cal["multi"]) == {"ls"} and set(report["multi"]) == {"ls", "small"}   # too small: reported, not written
    assert all(r["prim"] != "multi" for r in data[:300]) and report["types"]["noul"]["n"] == 300


def test_label_set_id_is_stable():
    a = label_set_id(["d", "g"], ["Damage", "General"], "g")
    assert a == label_set_id(["g", "d"], ["General", "Damage"], "g") and len(a) == 16
    assert a != label_set_id(["d", "g"], ["Damage", "General"], "d")
    assert a != label_set_id(["d", "g"], ["Damaged goods", "General"], "g")


def test_server_uses_the_fitted_fallback_threshold():
    crit = {"d": "Damage", "r": "Refund", "g": "General"}
    sid = label_set_id(*[to_items("x", {"t": {"type": "multi", "criteria": crit, "fallback": "g"}})[0][k]
                         for k in ("keys", "labels", "fallback")])
    p_yes = {"Damage": 0.1, "Refund": 0.1, "General": 0.2}

    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]

    def ask(cal, **q):
        body = {"state": "Paczka.", "questions": {"t": dict(type="multi", instructions="Które?", criteria=crit,
                                                            fallback="g", **q)}}
        return asyncio.run(FakeServer(probs, cal=cal)._run(body))["answers"]["t"]
    cal = {"multi": {sid: {"fallback": "g", "fallback_threshold": 0.15}}}
    ans = ask(cal)
    assert ans["selected"] == ["g"] and ans["constraints_applied"]["fallback_threshold"] == 0.15
    assert ans["constraints_applied"]["fallback_threshold_source"] == "calibration"
    ans = ask(cal, fallback_threshold=0.3)                              # an explicit value wins
    assert ans["selected"] == [] and ans["constraints_applied"]["fallback_threshold_source"] == "request"
    ans = ask({})                                                       # no fit: the default (threshold 0.5)
    assert ans["selected"] == [] and ans["constraints_applied"]["fallback_threshold_source"] == "default"
    other = {"multi": {label_set_id(["d", "g"], ["Damage", "General"], "g"): {"fallback_threshold": 0.15}}}
    assert ask(other)["selected"] == []                                 # another label set: not used
    assert ask({"multi": {sid: {"fallback_threshold": None}}})["selected"] == []   # fitted: the default wins


# -- end to end --------------------------------------------------------------------------------------------------------


class StubBackend:
    """Choice / noul: the weights at the end of the option lines ("... ~0.7"); a multi label: P(yes) written in the
    state after the label ("Damage~0.7"), so that the label texts are the same in every question."""
    policies = {}

    def __init__(self, *_, **__):
        self.tok = Tok()

    def run_shared(self, groups, policy=None):
        out = []
        for prompts, lids in groups:
            res = []
            for p, ids in zip(prompts, lids):
                text = "".join(map(chr, p))
                m = LABEL.search(text)
                if m:
                    py = float(re.search(re.escape(m.group(1)) + r"~([0-9.e-]+)", text).group(1))
                    res.append([py, 1 - py] if "A. Yes" in text else [1 - py, py])
                    continue
                w = [float(x) for x in WEIGHT.findall(text)][: len(ids)]
                res.append([x / sum(w) for x in w])
            out.append(res)
        return out


@pytest.fixture
def engine(tmp_path, monkeypatch):
    md = tmp_path / "model"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps({"temperature_per_prim": {"noul": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md


def labelled(path, n=1500, seed=0):
    """Yes/no questions of three families (two over-confident, one calibrated) and multi questions with a fallback."""
    rng = np.random.default_rng(seed)
    crit = {"d": "Damage", "r": "Refund", "i": "Invoice", "g": "General"}
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            fam = ("late", "covered", "signed")[i % 3]
            q = float(rng.uniform(0.02, 0.98))
            sharpen = 1.0 if fam == "signed" else 2.5
            p = q ** sharpen / (q ** sharpen + (1 - q) ** sharpen)
            noul = dict(type="noul", instructions="Applies?", family=fam, gold="yes" if rng.random() < q else "no",
                        criteria={"true": f"Yes ~{p:.12g}", "false": f"No ~{1 - p:.12g}"})
            r = set_rows(1, seed=seed * 100000 + i)[0]
            weights = " ".join(f"{crit[k]}~{x:.12g}" for k, x in zip(KEYS, r["p"]))
            multi = dict(type="multi", instructions="Which?", criteria=crit, fallback="g",
                         gold=[KEYS[j] for j in r["gold"]])
            rec = {"id": f"c{i}", "state": f"Case {i}. {weights}", "questions": {"q": noul}}
            if i % 2 == 0:
                rec["questions"]["tags"] = multi
            f.write(json.dumps(rec) + "\n")


def test_end_to_end_gates_strict_and_multi(tmp_path, engine, capsys):
    labelled(tmp_path / "l.jsonl")
    out = tmp_path / "cal.json"
    argv = [str(tmp_path / "l.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out)]
    main(argv + ["--reference"])
    printed = capsys.readouterr().out
    assert "LOFO ECE on CERT" in printed and "automated on CERT" in printed and "multi fallback" in printed
    cal = json.loads(out.read_text(encoding="utf-8"))
    lf = cal["gates"]["ece_lofo"]["noul"]
    assert lf["n_families"] == 3 and lf["worst_family"] == "signed" and lf["reference"] == 0.062
    assert cal["gates"]["ece_reference"].startswith("basal-1.5") and cal["fitted_on"]["multi"] == 750
    assert cal["fitted_on"]["decisions"] == 1500
    (entry,) = cal["multi"].values()
    assert entry["fallback"] == "g" and entry["fallback_threshold"] is not None
    assert entry["cert"]["iou"] > entry["cert"]["iou_default"]
    strict = argv + ["--strict", "--reference"]
    if lf["regression"]:
        with pytest.raises(SystemExit) as e:
            main(strict)
        assert e.value.code == 1
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"gates": {"ece_lofo": {"noul": {"mean": 0.9}}}}), encoding="utf-8")
    main(argv + ["--strict", "--reference", str(ref)])                 # far above: no regression, exit 0
    ref.write_text(json.dumps({"gates": {"ece_lofo": {"noul": {"mean": 0.0}}}}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(argv + ["--strict", "--reference", str(ref)])
    assert e.value.code == 1 and "above the reference for noul" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="--strict needs --reference"):
        main(argv + ["--strict"])


def test_accepted_needed():
    assert [accepted_needed(0.01, e) for e in range(3)] == [299, 473, 628]
    assert [accepted_needed(0.05, e) for e in range(3)] == [59, 93, 124]


def test_sample_size_warns_before_the_readouts():
    """Feedback from BRAUN SI: 275 verify claims cannot certify 1% (299 accepted FIT decisions without an error are
    needed, the FIT split holds about 70% of them); 5% can be. Said up front, per type and task and target."""
    units = [dict(task="verify", group=f"g{i}") for i in range(275)]
    decisions = [dict(prim="noul", group=f"n{i}") for i in range(1000)] + [dict(prim="multi", group="m")]
    n_fit = sum(not is_cert(u["group"]) for u in units)
    (w,) = sample_size(decisions, units, [0.01, 0.05])
    assert w.startswith(f"verify: max_error 0.01 needs >= 299 accepted FIT decisions without an error (473 with 1 "
                        f"error, 628 with 2); this file gives {n_fit} FIT decisions of 275")
    assert "label about 153 more verify claims. verify will not be certified at max_error 0.01" in w


def test_cert_note_when_cert_cannot_confirm_a_written_threshold():
    """Feedback from BRAUN SI / NovaraX: a threshold certified on FIT whose CERT upper bound is above the target is
    kept, but the report says CERT cannot confirm it; an observed CERT error above the target is said more strongly."""
    x = dict(confidence=0.97, cert_accepted=41, cert_error=0.0, cert_upper95=0.0705)
    assert cert_note("verify", 0.05, x) == ("verify at max_error 0.05: certified on FIT; CERT has 41 accepted, 0 "
                                            "errors, upper bound 7.05% > 5%, too few to confirm on CERT (0 errors "
                                            "would need >= 59 accepted CERT decisions)")
    assert "does not hold on CERT" in cert_note("noul", 0.01, dict(x, cert_accepted=100, cert_error=0.02))
    assert "accepts no decision" in cert_note("noul", 0.01, dict(x, cert_accepted=0, cert_error=None,
                                                                  cert_upper95=None))
    assert cert_note("noul", 0.05, dict(x, cert_accepted=200, cert_upper95=0.0149)) is None  # confirmed
    assert cert_note("noul", 0.05, dict(confidence=None)) is None


def test_sample_size_and_cert_notes_on_stderr(tmp_path, engine, capsys):
    """Both warnings reach stderr: the sample size before the readouts and again in the end-of-run summary, with the
    --from-rows refit too; the exit status is unchanged."""
    labelled(tmp_path / "l.jsonl", n=300)
    rows = tmp_path / "rows.jsonl"
    main([str(tmp_path / "l.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(tmp_path / "cal.json"),
          "--report-only", "--save-rows", str(rows)])
    err = capsys.readouterr().err
    first = err.index("noul: max_error 0.01 needs >= 299 accepted FIT decisions")
    assert first < err.index("basal-calibrate: summary") < err.rindex("noul: max_error 0.01 needs >= 299")
    main([str(rows), "--from-rows", "--report-only"])
    err = capsys.readouterr().err
    assert err.count("noul: max_error 0.01 needs >= 299") == 2 and "basal-calibrate: summary" in err


def test_strict_says_when_no_family_was_tested(tmp_path, engine, capsys):
    """#19: when no family has --min-family CERT decisions of a type, --strict has nothing to test for it. It still
    passes (status 0), and says so on stderr and in the report."""
    labelled(tmp_path / "l.jsonl", n=300)
    main([str(tmp_path / "l.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(tmp_path / "cal.json"),
          "--strict", "--reference", "--min-family", "1000"])
    cap = capsys.readouterr()
    assert ("--strict: noul: 0 families tested (each needs >= 1000 CERT decisions of the type); the family check did "
            "not run") in cap.err
    assert "noul: no leave-one-family-out ECE, 0 families tested" in cap.out
    assert "--strict does not check noul" in cap.out


# -- certified thresholds per type ------------------------------------------------------------------------------------


def binary(n, family, seed, skill):
    """Yes/no rows whose raw confidence is informative: P(correct) rises with it."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = float(rng.uniform(0.5, 0.999))
        right = rng.random() < min(1.0, c + skill)
        out.append(dict(prim="noul", p=[c, 1 - c], gold=0 if right else 1, group=f"{family}{seed}-{i}", family=family))
    return out


def test_per_type_coverage_does_not_depend_on_the_temperature():
    """A temperature is monotone on yes/no confidences, so each type's certified coverage is the same at any T, while
    one threshold pooled with another type moves with T (the loss this removes)."""
    data = binary(3000, "n", 0, 0.1) + rows(3000, 2, 1.0, "c", "choice", 1)
    per, pooled = {}, {}
    for T in (0.7, 1.0, 1.73, 3.0):
        base = {"temperature_per_prim": {"noul": T, "choice": 1.0}}
        cal, _ = fit(data, base, [0.01, 0.05], min_per_type=10 ** 9, fmt=2, min_blend=0)  # shipped, not fitted
        assert cal["temperature_per_prim"]["noul"] == T
        for risk in ("0.01", "0.05"):
            x = cal["thresholds"][risk]["per_type"]["noul"]
            per.setdefault(risk, set()).add((x["fit_coverage"], x["cert_coverage"]))
            pooled.setdefault(risk, set()).add(cal["gates"]["automated_at_pooled"][risk])
    assert all(len(v) == 1 for v in per.values())                     # identical at every temperature
    assert any(len(v) > 1 for v in pooled.values())                   # the pooled threshold is not
    assert next(iter(per["0.05"]))[0] > 0


def test_the_server_applies_only_the_type_threshold():
    """A type's own certified threshold, never the pooled one: a type without a certificate defers (under the pooled
    threshold an uncertified type can run above the target). Only a format-1 file without per_type (basal-1.5) keeps
    its pooled threshold, its only certificate."""
    from basal.server import certified
    cal = {"prompt": 2, "thresholds": {"0.01": {"confidence": 0.9, "per_type": {"noul": {"confidence": 0.95},
                                                                                "score": {"confidence": None}}}}}
    assert certified(cal, 0.01, "noul") == 0.95
    assert certified(cal, 0.01, "score") is None and certified(cal, 0.01, "choice") is None
    assert certified({k: v for k, v in cal.items() if k != "prompt"}, 0.01, "choice") is None
    assert certified({"thresholds": {"0.01": {"confidence": 0.98}}}, 0.01, "noul") == 0.98      # basal-1.5 files
    assert certified({"prompt": 2, "thresholds": {"0.01": {"confidence": 0.98}}}, 0.01, "noul") is None
    assert certified({}, 0.01, "noul") is None and certified(cal, 0.05, "noul") is None


def test_automated_share_and_warnings_follow_the_server():
    """basal-calibrate's automated_at counts a type only at its own threshold (an uncertified type is deferred, as
    basal-serve does), writes the pooled threshold as informational, and warns about every uncertified type."""
    from basal.calibrate import POOLED_NOTE, effective_gate, warnings
    rows = [{"p": [0.9, 0.1], "gold": 0, "prim": "noul", "group": "a"},
            {"p": [0.9, 0.1, 0.0], "gold": 0, "prim": "score", "group": "a"}]
    entry = {"confidence": 0.5, "per_type": {"noul": {"confidence": 0.8}, "score": {"confidence": None}}}
    assert effective_gate(rows, {}, {}, entry) == dict(coverage=0.5, error=0.0)
    rows = [{"p": [0.999, 0.001], "gold": 0, "prim": "noul", "group": f"g{i}"} for i in range(400)] + \
           [{"p": [0.6, 0.3, 0.1], "gold": i % 2, "prim": "score", "group": f"s{i}"} for i in range(100)]
    cal, report = fit(rows, {"temperature_per_prim": {}}, [0.05], min_per_type=10 ** 9, fmt=2)
    entry = cal["thresholds"]["0.05"]
    assert entry["pooled"] == POOLED_NOTE and entry["per_type"]["score"]["confidence"] is None
    assert any("score has no certified threshold" in w for w in warnings(report, [0.05]))


def test_saved_rows_refit_without_an_engine(tmp_path, engine):
    labelled(tmp_path / "l.jsonl", n=600)
    argv = [str(tmp_path / "l.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(tmp_path / "a.json"),
            "--save-rows", str(tmp_path / "rows.jsonl")]
    main(argv)
    main([str(tmp_path / "rows.jsonl"), "--from-rows", "--out", str(tmp_path / "b.json")])
    a, b = json.loads((tmp_path / "a.json").read_text(encoding="utf-8")), json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    for k in ("temperature_per_prim", "thresholds", "gates", "multi"):
        assert a[k] == b[k], k
    with pytest.raises(SystemExit, match="not a file written by"):
        main([str(tmp_path / "l.jsonl"), "--from-rows", "--out", str(tmp_path / "c.json")])


def saved_rows(tmp_path, baseline, **head):
    first = dict(basal_calibrate_rows=1, fmt=2, kind="eager", quant=None, model="stub", engine="eager",
                 data="labels.jsonl", requests=1, baseline=str(baseline), orders=1, **head)
    row = dict(prim="choice", group="g", family="f", gold=0, p=[0.8, 0.2])
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_saved_rows_refit_honours_no_calibration(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"temperature_per_prim": {"choice": 4.0}}), encoding="utf-8")
    rows, out = saved_rows(tmp_path, baseline), tmp_path / "out.json"
    main([str(rows), "--from-rows", "--no-calibration", "--out", str(out)])
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["baseline"] == "none" and cal["temperature_per_prim"].get("choice", 1.0) != 4.0


def test_saved_rows_keep_the_serving_identity_of_the_fit(tmp_path, monkeypatch):
    """--save-rows stores layout, adapter and revision; the --from-rows refit writes them to fitted_on, where
    basal-serve checks them."""
    md = tmp_path / "models--owner--stub" / "snapshots" / ("a" * 40)
    md.mkdir(parents=True)
    (md / "basal.json").write_text(json.dumps({"prompt": 2, "layout": "qfirst"}), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"state": "s", "questions": {"q": {
        "type": "choice", "criteria": ["a", "b"], "gold": "a"}}}) + "\n", encoding="utf-8")
    entry = dict(name="stub", mode="eager", prompt_format=2, orders=1, layout="qfirst",
                 adapter="stub-adapter@sha256:abc", temperature_per_prim={})
    monkeypatch.setattr(httpx, "get", lambda *args, **kw: SimpleNamespace(json=lambda: {"models": [entry]}))
    monkeypatch.setattr(calibrate_module, "resolve", lambda *args, **kw: md)

    async def readouts(*args):
        return [{"answers": {"q": {"probabilities": {"a": 0.8, "b": 0.2}}}}]
    monkeypatch.setattr(calibrate_module, "over_http", readouts)
    rows, first, again = tmp_path / "rows.jsonl", tmp_path / "first.json", tmp_path / "again.json"
    main([str(labels), "--url", "http://h", "--model", "owner/stub", "--no-calibration",
          "--save-rows", str(rows), "--out", str(first)])
    main([str(rows), "--from-rows", "--out", str(again)])
    fitted = [json.loads(p.read_text(encoding="utf-8"))["fitted_on"] for p in (first, again)]
    assert fitted[0]["adapter"] == "stub-adapter@sha256:abc" and fitted[0]["revision"] == "a" * 40
    for field in ("layout", "adapter", "revision"):
        assert fitted[1][field] == fitted[0][field], field
