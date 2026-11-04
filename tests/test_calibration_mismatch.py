"""A --calibration file fitted on another model (or revision, or adapter) keeps its temperatures only: its thresholds,
task thresholds, conformal levels and fitted fallback thresholds are not applied, every answer is marked unvalidated,
and GET /v1/models agrees. Identity compares whole repo ids when both sides are Hugging Face snapshots. basal-calibrate
--from-rows applies basal-serve's rule for orders and layout to its baseline."""
import asyncio
import json

import pytest

from basal import server
from basal.calibrate import main
from basal.server import Server, calibration_mismatch, load_calibration, parser
from tests.test_calibrate import engine  # noqa: F401 (fixture)

CAL = {"prompt": 1, "temperature_per_prim": {"choice": 1.0, "noul": 1.0, "score": 1.0},
       "thresholds": {"0.05": {"confidence": 0.6, "per_type": {"noul": {"confidence": 0.6}}}},
       "task_thresholds": {"classify": {"0.05": 0.5}},
       "conformal": {"levels": {"0.9": {"per_type": {"noul": {"p_min": 0.2}}}}}}
NOUL = {"type": "noul", "instructions": "Covered?", "criteria": {"true": "Yes ~0.95", "false": "No ~0.05"},
        "max_error": 0.05}


def ask(srv, body):
    async def go():
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    return asyncio.run(go())["answers"]


def serve(path, model):
    return Server(parser().parse_args(["--mode", "eager", "--model", model, "--calibration", str(path)]))


def test_a_file_of_another_model_certifies_nothing(tmp_path, engine, capsys):  # noqa: F811
    f = tmp_path / "C.json"
    f.write_text(json.dumps(dict(CAL, fitted_on={"model": "Remek/basal-omni-1.0-max"})), encoding="utf-8")
    own = serve(f, "Remek/basal-omni-1.0-max")
    assert own.cal_mismatch is None and own.cal_unvalidated is None
    a = ask(own, {"state": "x", "questions": {"q": NOUL}})["q"]
    assert a["abstain"] is False and "calibration" not in a        # the same model: certified at 0.6
    cls = {"type": "classify", "labels": {"a": "Alpha ~0.9", "b": "Beta ~0.1"}, "max_error": 0.05}
    assert ask(own, {"state": "x", "questions": {"c": cls}})["c"]["automated"] == {"0.05": True}
    srv = serve(f, "Remek/basal-omni-1.1-max")
    assert "fitted on Remek/basal-omni-1.0-max" in srv.cal_mismatch
    assert "they are not applied" in capsys.readouterr().out
    assert srv.cal["temperature_per_prim"] == CAL["temperature_per_prim"]
    assert not {"thresholds", "task_thresholds", "conformal", "multi"} & set(srv.cal)
    assert server.certified_table(srv.cal) == {}                   # GET /v1/models and the answers agree: nothing
    assert ask(srv, {"state": "x", "questions": {"c": cls}})["c"]["automated"] == {"0.05": None}
    act = dict(NOUL, type="act", costs={"wrong": 2, "defer": 1})
    ans = ask(srv, {"state": "x", "questions": {"q": NOUL, "s": dict(NOUL, coverage=0.9), "a": act}})
    assert ans["q"]["abstain"] is True and ans["q"]["calibration"] == "unvalidated"
    assert ans["s"]["answer_set"] == ["true", "false"]             # no conformal level: every option
    assert ans["a"]["deferred"] is True and ans["a"]["calibration"] == "unvalidated"


def test_identity_compares_whole_repo_ids_of_snapshots(tmp_path):
    a = tmp_path / "models--OrgA--basal-x" / "snapshots" / "aaa"
    b = tmp_path / "models--OrgB--basal-x" / "snapshots" / "aaa"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    cal = {"fitted_on": {"model": "OrgA/basal-x", "revision": "aaa"}}
    assert calibration_mismatch(cal, "OrgA/basal-x", a) is None
    assert calibration_mismatch(cal, "orga/BASAL-X", a) is None                     # case does not matter
    assert "fitted on OrgA/basal-x, the served model is OrgB/basal-x" in calibration_mismatch(cal, "OrgB/basal-x", b)
    assert calibration_mismatch(cal, "/data/basal-x", tmp_path) is None             # a local copy: by its last part
    assert calibration_mismatch({"fitted_on": {"model": "/data/basal-x"}}, "OrgB/basal-x", b) is None


def test_load_calibration_and_the_shared_rule_agree(tmp_path):
    f = tmp_path / "C.json"
    for data, orders, layout, why in (({"orders": 2}, 1, "canonical", "fitted on 2 option order(s), served with 1"),
                                      ({"layout": "qfirst"}, 1, "canonical", "fitted on the qfirst layout"),
                                      ({"orders": 1}, 1, "canonical", None)):
        f.write_text(json.dumps(dict(CAL, **data)), encoding="utf-8")
        _, _, got = load_calibration(tmp_path, "eager", f, None, orders, layout=layout)
        assert got == server.unvalidated_reason(dict(CAL, **data), "C.json", orders, layout)
        assert (got is None) == (why is None) and (why is None or why in got)


@pytest.mark.parametrize("extra, why", [({"layout": "qfirst"}, "fitted on the qfirst layout"),
                                        ({"orders": 2}, "fitted on 2 option order(s), served with 1")])
def test_refit_drops_the_thresholds_of_a_baseline_basal_serve_would_not_apply(tmp_path, capsys, extra, why):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(CAL, **extra)), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=1, baseline=str(base), orders=1, layout="canonical")
    rows = [dict(prim="noul", group=f"g{i}", family="f", gold=0, p=[0.9, 0.1], line=i, name="q", keys=["true", "false"])
            for i in range(20)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [head] + rows) + "\n", encoding="utf-8")
    main([str(path), "--from-rows", "--report-only", "--targets", "0.05"])
    out = capsys.readouterr()
    assert why in out.err and "not compared or kept" in out.err
