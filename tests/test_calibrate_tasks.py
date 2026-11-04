"""basal-calibrate on task questions (2.2): gold forms, task thresholds certified on the answers as served (with the
kept or the fitted temperatures), the model a file was fitted on, --report-only, and extract's `automated` flag."""
import asyncio
import json

import numpy as np
import pytest

import basal.server as server_module
from basal.calibrate import field_correct, fit, fit_tasks, is_cert, main, read, served, task_rows, task_units
from basal.server import Server, calibration_mismatch, load_calibration, model_identity, parser, temper
from tests.test_calibrate import StubBackend, synthetic
from tests.test_server import FakeServer

SHIPPED = {"temperature_per_prim": {"choice": 1.5, "noul": 1.2, "score": 1.0},
           "thresholds": {"0.05": {"confidence": 0.8, "per_type": {"choice": {"confidence": 0.8}}}},
           "task_thresholds": {"verify": {"0.05": 0.9}}}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    md = tmp_path / "model"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md


def classify_lines(n=1000, sharpen=2.0, seed=1):
    """Classify questions whose engine is over-confident (serves q ** sharpen while the gold is drawn from q); q is
    high enough for a 5% threshold to be certifiable."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = rng.uniform(0.9, 1.0)
        w = np.array([q, 1 - q]) ** sharpen
        w = w / w.sum()
        gold = "refund" if rng.random() < q else "track"
        out.append({"id": f"c{i}", "state": f"ticket {i}", "questions": {"intent": {
            "type": "classify", "labels": {"refund": f"asks for a refund ~{w[0]:.6f}",
                                           "track": f"asks where the parcel is ~{w[1]:.6f}"}, "gold": gold}}})
    return out


def write(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def test_task_gold_forms_and_errors(tmp_path):
    cls = {"type": "classify", "labels": ["refund", "track"], "none": True}
    assert task_units("i", cls, "none", "x") == [(None, "none")]
    with pytest.raises(ValueError, match="not a label"):
        task_units("i", cls, "cancel", "x")
    with pytest.raises(ValueError, match="multi-label classify"):
        task_units("i", dict(cls, multi_label=True), ["refund"], "x")
    with pytest.raises(ValueError, match="compress"):
        task_units("c", {"type": "compress"}, [], "x")
    ver = {"type": "verify", "claims": ["Total 120.", "Due in 30 days."]}
    assert task_units("v", ver, ["supported", None], "x") == [("0", "supported")]
    assert task_units("v", ver, {"1": "Not stated"}, "x") == [("1", "not_stated")]
    assert task_units("v", dict(ver, binary=True), {"0": False}, "x") == [("0", "not_supported")]
    with pytest.raises(ValueError, match="not one of"):
        task_units("v", ver, {"0": "maybe"}, "x")
    with pytest.raises(ValueError, match="claim '7'"):
        task_units("v", ver, {"7": "supported"}, "x")
    schema = {"type": "object", "properties": {
        "method": {"enum": ["transfer", "card"]}, "paid": {"type": "boolean"}, "total": {"type": "number"},
        "seller": {"type": "object", "properties": {"name": {"type": "string"}}},
        "tags": {"type": "array", "items": {"enum": ["urgent"]}}}}
    ext = {"type": "extract", "schema": schema}
    assert task_units("e", ext, {"method": "card", "paid": "yes", "total": "1 230,50", "seller": {"name": None},
                                 "tags": ["urgent"]}, "Invoice") == [
        ("method", ("enum", "card")), ("paid", ("boolean", True)), ("total", ("number", 1230.5)),
        ("seller.name", ("string", None))]  # arrays of enum are not certified; a field left out is not labelled
    with pytest.raises(ValueError, match="not one of its values"):
        task_units("e", ext, {"method": "cash"}, "Invoice")
    with pytest.raises(ValueError, match="labels no field"):
        task_units("e", ext, {"tags": ["urgent"]}, "Invoice")

    mixed = {"id": "m1", "state": "Refund the kettle.", "questions": {
        "dept": {"type": "choice", "criteria": {"returns": "returns", "it": "IT"}, "gold": "returns"},
        "intent": dict(cls, gold="refund"), "check": dict(ver, gold=["supported", "contradicted"])}}
    write(tmp_path / "m.jsonl", [mixed, {"state": "x", "questions": {"i": dict(cls, gold="bad")}}])
    with pytest.raises(ValueError, match="line 2.*not a label"):
        read(tmp_path / "m.jsonl")
    write(tmp_path / "m.jsonl", [mixed])
    bodies, decisions, task_bodies, units = read(tmp_path / "m.jsonl")
    assert list(bodies[0]["questions"]) == ["dept"] and list(task_bodies[0]["questions"]) == ["intent", "check"]
    assert "gold" not in task_bodies[0]["questions"]["intent"] and len(decisions) == 1
    assert [(u["task"], u["part"], u["gold"], u["group"]) for u in units] == [
        ("classify", None, "refund", "m1"), ("verify", "0", "supported", "m1"), ("verify", "1", "contradicted", "m1")]


def test_field_correct_and_task_rows():
    assert field_correct("string", " ACME  ltd", "acme Ltd") and not field_correct("string", "ACME", None)
    assert field_correct("number", 1230.5, 1230.50) and not field_correct("integer", 7, 8)
    assert field_correct("enum", None, None) and field_correct("boolean", False, False)
    units = [dict(line=1, req=0, name="v", task="verify", part="0", gold="supported", group="g"),
             dict(line=1, req=0, name="e", task="extract", part="total", gold=("number", 120.0), group="g"),
             dict(line=1, req=0, name="e", task="extract", part="n", gold=("integer", 3), group="g"),
             dict(line=2, req=1, name="i", task="classify", part=None, gold="a", group="h")]
    resp = [{"answers": {"v": {"claims": {"0": {"verdict": "contradicted", "confidence": 0.7}}},
                         "e": {"fields": {"total": {"value": 120, "confidence": 0.95},
                                          "n": {"value": None, "error": "not a integer", "confidence": 0.99}}}}},
            RuntimeError("boom")]
    rows, failed = task_rows(units, resp)
    assert rows == [dict(task="verify", group="g", c=0.7, ok=False), dict(task="extract", group="g", c=0.95, ok=True)]
    assert failed == ["line 2: boom"]


def test_fit_tasks_certifies_per_task():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3000):  # errors thin out at the top (the fixed sequence ends at an error in its first level)
        c = rng.uniform(0.5, 1.0)
        rows.append(dict(task="classify", group=f"g{i}", c=c, ok=bool(rng.random() < min(1.0, c + 0.01))))
    thr, rep = fit_tasks(rows + [dict(task="verify", group="v", c=0.99, ok=True)], [0.05],
                         shipped={"classify": {"0.05": 0.6}})
    assert set(thr) == {"classify"} and 0.9 < thr["classify"]["0.05"] < 1.0
    x = rep["classify"]["targets"]["0.05"]
    assert x["fit_upper95"] <= 0.05 and x["cert_error"] < 0.08 and rep["verify"]["targets"]["0.05"]["confidence"] is None
    old = rep["classify"]["shipped"]["0.05"]["cert"]        # a too low threshold, measured on your CERT decisions
    assert old["coverage"] > x["cert_coverage"] and old["error"] > 0.05


def test_task_only_input_keeps_the_decision_calibration(tmp_path, engine, capsys):
    write(tmp_path / "t.jsonl", classify_lines())
    out = tmp_path / "C.json"
    main([str(tmp_path / "t.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out), "--targets", "0.05"])
    printed = capsys.readouterr().out
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["temperature_per_prim"] == SHIPPED["temperature_per_prim"]
    assert cal["thresholds"] == SHIPPED["thresholds"] and cal["decision_calibration"].startswith("kept from")
    assert cal["task_thresholds"]["verify"] == {"0.05": 0.9}            # shipped, temperatures unchanged: kept
    thr = cal["task_thresholds"]["classify"]["0.05"]
    assert cal["fitted_on"]["tasks"] == {"classify": 1000} and cal["fitted_on"]["decisions"] == 0
    assert "classify" in printed and "decision questions: none" in printed

    srv = Server(parser().parse_args(["--mode", "eager", "--model", "stub", "--calibration", str(out)]))
    assert srv.cal_mismatch is None

    async def decide(body):
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    for p in (0.7, 0.999):
        ans = asyncio.run(decide({"state": "x", "questions": {"i": {
            "type": "classify", "labels": {"refund": f"r ~{p}", "track": f"t ~{1 - p}"}, "max_error": 0.05}}}))
        conf = max(temper(server_module.torch.tensor([p, 1 - p]), 1.5).tolist())  # the shipped choice temperature
        a = ans["answers"]["i"]
        assert abs(a["confidence"] - conf) < 1e-6 and a["automated"] == {"0.05": conf >= thr}


def test_mixed_input_certifies_tasks_with_the_fitted_temperatures(tmp_path, engine, capsys):
    synthetic(tmp_path / "d.jsonl", n=600)
    lines = [json.loads(x) for x in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()] + classify_lines()
    write(tmp_path / "m.jsonl", lines)
    out = tmp_path / "C.json"
    main([str(tmp_path / "m.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out), "--targets", "0.05"])
    printed = capsys.readouterr().out
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["temperature_source"]["choice"].startswith("fitted") and "decision_calibration" not in cal
    assert set(cal["task_thresholds"]) == {"classify"}                  # the shipped verify threshold is dropped
    assert "verify: the shipped task threshold" in printed
    # certified on the answers as served with the fitted temperatures: the served CERT error holds the target
    g = cal["task_gates"]["classify"]["targets"]["0.05"]
    assert g["fit_upper95"] <= 0.05 and g["cert_error"] < 0.08


def test_url_mode_refuses_task_questions_and_report_only_writes_nothing(tmp_path, engine, capsys):
    write(tmp_path / "t.jsonl", classify_lines(n=100))
    with pytest.raises(SystemExit, match="need the engine in process"):
        main([str(tmp_path / "t.jsonl"), "--url", "http://127.0.0.1:9", "--model", "stub"])
    out = tmp_path / "C.json"
    main([str(tmp_path / "t.jsonl"), "--mode", "eager", "--model", "stub", "--out", str(out), "--report-only"])
    assert not out.exists() and "--report-only: no file written" in capsys.readouterr().out


def test_model_identity_and_mismatch(tmp_path):
    snap = tmp_path / "models--Remek--basal-omni-1.0-max" / "snapshots" / "abc123def4567890"
    snap.mkdir(parents=True)
    assert model_identity("Remek/basal-omni-1.0-max", snap) == ("Remek/basal-omni-1.0-max", "abc123def4567890")
    assert model_identity("/data/basal-omni-1.0-max", tmp_path) == ("/data/basal-omni-1.0-max", None)
    cal = {"fitted_on": {"model": "Remek/basal-omni-1.0-max", "revision": "abc123def4567890"}}
    assert calibration_mismatch(cal, "Remek/basal-omni-1.0-max", snap) is None
    assert calibration_mismatch(cal, "/data/basal-omni-1.0-max", tmp_path) is None   # a local copy, no revision
    assert "basal-omni-1.1-max" in calibration_mismatch(cal, "Remek/basal-omni-1.1-max", tmp_path)
    other = tmp_path / "models--Remek--basal-omni-1.0-max" / "snapshots" / "fff"
    other.mkdir()
    assert "revision" in calibration_mismatch(cal, "Remek/basal-omni-1.0-max", other)
    assert calibration_mismatch({"fitted_on": "our calibration split"}, "x", tmp_path) is None  # shipped files


def test_server_warns_or_refuses_a_file_of_another_model(tmp_path, engine, capsys):
    f = tmp_path / "C.json"
    f.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "Remek/basal-omni-1.0-max"})), encoding="utf-8")
    srv = Server(parser().parse_args(["--mode", "eager", "--model", "Remek/basal-omni-1.1-max", "--calibration",
                                      str(f)]))
    assert "fitted on Remek/basal-omni-1.0-max" in srv.cal_mismatch and "warning: C.json" in capsys.readouterr().out
    payload = server_module.models_payload("m", "eager", {}, {}, calibration_warning=srv.cal_mismatch)
    assert payload["models"][0]["calibration_warning"] == srv.cal_mismatch
    with pytest.raises(SystemExit, match="--strict-calibration"):
        Server(parser().parse_args(["--mode", "eager", "--model", "Remek/basal-omni-1.1-max", "--calibration", str(f),
                                    "--strict-calibration"]))


def test_extract_fields_report_automated():
    def probs(prompt, ids):
        s = "".join(map(chr, prompt))
        q = s.split("<user>")[-1]
        if "method" in q:
            return [0.97, 0.02, 0.01][: len(ids)]
        if "count" in q and "Which line" in q:
            return [0.95, 0.05][: len(ids)]
        return [1.0 / len(ids)] * len(ids)
    schema = {"type": "object", "properties": {"method": {"enum": ["transfer", "card"]},
                                               "count": {"type": "integer"},
                                               "tags": {"type": "array", "items": {"enum": ["urgent", "late"]}}}}
    srv = FakeServer(probs, cal={"task_thresholds": {"extract": {"0.05": 0.9}}})
    srv.orders = 1
    x = asyncio.run(srv._run({"state": "Pay by transfer, two items", "questions": {"x": {
        "type": "extract", "schema": schema, "max_error": 0.05}}}))["answers"]["x"]["fields"]
    assert x["method"]["automated"] == {"0.05": x["method"]["confidence"] >= 0.9}
    assert x["tags"]["automated"] == {"0.05": None}
    assert x["count"]["value"] is None and x["count"]["automated"] == {"0.05": False}  # "two" does not parse


def test_the_compared_file_is_measured_as_served():
    """The report's compared column uses the threshold basal-serve applies to each type (its own), not the pooled one."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(2000):
        prim = ("choice", "noul")[i % 2]
        q = rng.uniform(0.5, 1.0)
        rows.append(dict(prim=prim, p=[q, 1 - q], gold=0 if rng.random() < q else 1, group=f"g{i}", name="x"))
    base = {"prompt": 2, "temperature_per_prim": {"choice": 1.0, "noul": 1.0},
            "thresholds": {"0.05": {"confidence": 0.5, "per_type": {"choice": {"confidence": 0.97}}}}}
    _, report = fit(rows, base, [0.05], fmt=2)
    cert = [r for r in rows if is_cert(r["group"])]
    want = [ok for r, (c, ok) in zip(cert, served(cert, {"choice": 1.0, "noul": 1.0})) if r["prim"] == "choice" and c >= 0.97]
    got = report["compared"]["0.05"]
    assert abs(got["coverage"] - len(want) / len(cert)) < 1e-12           # noul has no threshold of its own: deferred
    assert abs(got["error"] - (len(want) - sum(want)) / len(want)) < 1e-12


def test_other_option_orders_drop_task_thresholds_too(tmp_path):
    """A file fitted on other option orders keeps its temperatures only: no decision, fallback or task thresholds."""
    md = tmp_path / "m"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(dict(SHIPPED, orders=1)), encoding="utf-8")
    data, f, why = load_calibration(md, "eager", None, None, 2)
    assert why and "task_thresholds" not in data and "thresholds" not in data and data["temperature_per_prim"]
    (md / "CALIBRATION.o2.json").write_text(json.dumps(dict(SHIPPED, orders=2)), encoding="utf-8")
    data, f, why = load_calibration(md, "eager", None, None, 2)
    assert why is None and f.name == "CALIBRATION.o2.json" and data["task_thresholds"]


def test_calibrate_compares_the_file_basal_serve_loads_for_its_orders(tmp_path, engine, capsys):
    (engine / "CALIBRATION.json").write_text(json.dumps(dict(SHIPPED, orders=1)), encoding="utf-8")
    o2 = dict(SHIPPED, orders=2, task_thresholds={"verify": {"0.05": 0.95}})
    (engine / "CALIBRATION.o2.json").write_text(json.dumps(o2), encoding="utf-8")
    write(tmp_path / "t.jsonl", classify_lines(n=200))
    out = tmp_path / "C.json"
    main([str(tmp_path / "t.jsonl"), "--mode", "eager", "--model", "stub", "--orders", "2", "--out", str(out)])
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["baseline"] == "stub/CALIBRATION.o2.json" and cal["task_thresholds"]["verify"] == {"0.05": 0.95}
    assert cal["fitted_on"]["orders"] == 2


def test_a_looser_target_never_keeps_a_stricter_threshold():
    """The fixed-sequence scan can pass early at 5% on a small top level and stop on a few errors below it, while 1%
    passes much lower down (basal-omni mini FP8, 17 Oct): the 5% threshold is floored at the 1% one."""
    rows, i = [], 0
    for conf, n, wrong in ((0.999, 200, 0), (0.998, 40, 12), (0.99, 3000, 0)):
        for k in range(n):
            rows.append(dict(prim="noul", p=[conf, 1 - conf], gold=1 if k < wrong else 0, group=f"g{i}", name="x"))
            i += 1
    cal, report = fit(rows, {"temperature_per_prim": {"noul": 1.0}}, [0.01, 0.05], min_per_type=10 ** 9, min_blend=0,
                      fmt=2)
    one, five = (cal["thresholds"][r]["per_type"]["noul"] for r in ("0.01", "0.05"))
    assert one["confidence"] is not None and abs(one["confidence"] - 0.99) < 1e-6     # float32, as served
    assert five["confidence"] == one["confidence"] and five["floored_from"] == "0.01"
    assert cal["gates"]["automated_at"]["0.05"] >= cal["gates"]["automated_at"]["0.01"] > 0.9
    thr, rep = fit_tasks([dict(task="verify", group=r["group"], c=r["p"][0], ok=r["gold"] == 0) for r in rows],
                         [0.01, 0.05])
    assert thr["verify"]["0.05"] == thr["verify"]["0.01"] and rep["verify"]["targets"]["0.05"]["floored_from"] == "0.01"
