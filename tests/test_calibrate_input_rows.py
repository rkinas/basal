"""basal-calibrate's input boundary: conditional questions are asked unconditionally (and a missing answer is a failed
row, not a crash after the run), a null group or id is not one shared group, an integer score gold is the level index
and refused when a numeric key says otherwise, and rows with non-finite probabilities or a gold out of range are left
out of the fit and reported."""
import json
import math

import pytest

from basal.calibrate import fit, gold_index, raw_rows, read, task_rows, warnings
from basal.server import to_items

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta"}}


def write(path, lines):
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def test_conditional_questions_are_asked_unconditionally(tmp_path):
    p = write(tmp_path / "l.jsonl", [{"state": "s", "questions": {
        "kind": dict(CHOICE, gold="a"),
        "late": {"type": "noul", "instructions": "Late?", "gold": "no", "ask_if": {"kind": ["b"]},
                 "depends_on": ["kind"]},
        "cls": {"type": "classify", "labels": ["x", "y"], "gold": "x", "ask_if": {"kind": ["a"]}}}}])
    bodies, decisions, task_bodies, units = read(p)
    assert not {"ask_if", "depends_on"} & set(bodies[0]["questions"]["late"])
    assert "ask_if" not in task_bodies[0]["questions"]["cls"]
    # a response without an answer (a server that skipped it) is a failed row, not a TypeError after the run
    resp = [{"answers": {"kind": {"probabilities": {"a": 0.9, "b": 0.1}}, "late": None}}]
    rows, failed = raw_rows(decisions, resp)
    assert [r["name"] for r in rows] == ["kind"] and failed == ["line 1, question 'late': no answer"]
    trows, tfailed = task_rows(units, [{"answers": {"cls": None}}])
    assert trows == [] and tfailed == ["line 1, question 'cls': no answer"]


def test_a_null_group_or_id_is_not_one_shared_group(tmp_path):
    lines = [{"id": f"t{i}", "group": None, "state": "s", "questions": {"q": dict(CHOICE, gold="a")}} for i in range(3)]
    lines += [{"id": None, "group": None, "state": "s", "questions": {"q": dict(CHOICE, gold="a", group=None)}}
              for _ in range(2)]
    _, decisions, _, _ = read(write(tmp_path / "l.jsonl", lines))
    assert [d["group"] for d in decisions] == ["t0", "t1", "t2", "4", "5"]
    assert "None" not in {d["family"] for d in decisions}


def score_item(criteria):
    q = {"type": "score", "instructions": "How?", "criteria": criteria}
    return q, to_items("s", {"q": q})[0]


def test_an_integer_score_gold_is_the_level_index():
    q, item = score_item({"low": "low", "mid": "mid", "high": "high"})
    assert gold_index(dict(q, gold=2), item) == 2 and gold_index(dict(q, gold="mid"), item) == 1
    q, item = score_item(["low", "mid", "high"])
    assert gold_index(dict(q, gold=2), item) == 2
    q, item = score_item({"0": "none", "1": "some", "2": "all"})
    assert gold_index(dict(q, gold=1), item) == 1                       # key and index agree
    q, item = score_item({"1": "low", "2": "mid", "3": "high"})
    assert gold_index(dict(q, gold="2"), item) == 1                     # a string is the key
    with pytest.raises(ValueError, match="ambiguous"):
        gold_index(dict(q, gold=2), item)                               # index 2 is "3", key "2" is level 1


def test_rows_that_cannot_enter_the_fit_are_left_out_and_reported():
    good = [dict(prim="noul", p=[0.9, 0.1], gold=i % 7 == 0, group=f"g{i}", line=i, keys=["true", "false"])
            for i in range(300)]
    good = [dict(r, gold=int(r["gold"])) for r in good]
    bad = [dict(good[0], p=[math.nan, 0.5], line=1001), dict(good[0], p=[math.inf, 0.0], line=1002),
           dict(good[0], gold=2, line=1003), dict(good[0], p=[0.5, 0.3, 0.2], line=1004)]
    cal, report = fit(good, {}, [0.05], min_per_type=10)
    cal2, report2 = fit(good + bad, {}, [0.05], min_per_type=10)
    assert cal2["temperature_per_prim"] == cal["temperature_per_prim"] and math.isfinite(cal2["temperature_per_prim"]["noul"])
    assert len(report2["dropped"]) == 4 and report2["dropped"][0].startswith("line 1001: probabilities")
    assert any(w.startswith("4 row(s) left out of the fit") for w in warnings(report2, [0.05]))
    with pytest.raises(SystemExit, match="no decisions to fit"):
        fit(bad, {}, [0.05])

