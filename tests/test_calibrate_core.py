"""The calibration core (basal/calibrate.py): one entry builder for the pooled, per-type and task thresholds, one floor,
one automated share as basal-serve applies a file, the probabilities tempered once per row, one row boundary for the
engine readouts and --from-rows files, and the rule of basal-serve for the file compared with (validate_baseline)."""
import json
from types import SimpleNamespace

import httpx
import numpy as np

import basal.server as server_module
from basal import calibrate
from basal.calibrate import compared, fit, fit_tasks, is_cert, main, per_type_thresholds, served

FIT_G = [g for g in (f"g{i}" for i in range(4000)) if not is_cert(g)]
CERT_G = [g for g in (f"g{i}" for i in range(4000)) if is_cert(g)]


def noul_rows(n_fit=600, n_cert=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in FIT_G[:n_fit] + CERT_G[:n_cert]:
        p = float(rng.uniform(0.5, 1.0))
        out.append(dict(prim="noul", p=[p, 1 - p], gold=int(rng.random() > p + 0.04), group=g, family="f",
                        keys=["true", "false"]))
    return out


def test_pooled_per_type_and_task_entries_come_from_one_builder():
    rows = noul_rows()
    fit_rows = [r for r in rows if not is_cert(r["group"])]
    cert_rows = [r for r in rows if is_cert(r["group"])]
    cf, cc = served(fit_rows, {"noul": 1.0}), served(cert_rows, {"noul": 1.0})
    entry, _ = calibrate.certify(cf, cc, 0.05)
    assert per_type_thresholds(fit_rows, cert_rows, {"noul": 1.0}, {}, 0.05)["noul"] == entry
    pooled, _ = calibrate.certify(cf, cc, 0.05, veto=False, method=True)
    cal = fit(rows, {}, [0.05], min_per_type=10 ** 9, min_blend=0)[0]
    assert {k: v for k, v in cal["thresholds"]["0.05"].items() if k not in ("per_type", "pooled")} == pooled
    tasks = [dict(task="classify", group=r["group"], c=c, ok=ok) for r, (c, ok) in zip(rows, served(rows, {}))]
    assert fit_tasks(tasks, [0.05])[1]["classify"]["targets"]["0.05"] == entry


def test_one_floor_for_types_pooled_and_tasks():
    entries = {"0.01": dict(confidence=0.9, fit_error=0.0), "0.05": dict(confidence=None, reason="stopped"),
               "0.1": dict(confidence=0.95)}
    assert calibrate.floor(entries) == {"0.05", "0.1"}
    assert entries["0.05"] == dict(confidence=0.9, fit_error=0.0, floored_from="0.01")   # no stale reason left
    assert entries["0.1"]["floored_from"] == "0.05" and entries["0.1"]["confidence"] == 0.9


def test_a_floored_pooled_entry_keeps_no_stale_reason():
    th = {"0.01": dict(confidence=0.9, per_type={"noul": dict(confidence=0.9)}, pooled="note"),
          "0.05": dict(confidence=None, reason="stopped", per_type={}, pooled="note")}
    assert calibrate.floor_thresholds(th, [0.01, 0.05]) == {"0.05"}
    assert th["0.05"] == dict(confidence=0.9, floored_from="0.01", per_type={"noul": dict(confidence=0.9,
                                                                                         floored_from="0.01")},
                              pooled="note")


def test_the_automated_share_is_the_one_basal_serve_applies():
    rows = noul_rows()
    tempered = calibrate.Tempered({"noul": 1.0})
    entry = {"confidence": 0.5, "per_type": {"noul": {"confidence": 0.8}}}
    fitted = calibrate.automated_gate(rows, tempered, {"thresholds": {"0.05": entry}}, 0.05)
    want = [ok for c, ok in served(rows, {"noul": 1.0}) if c >= 0.8]
    assert fitted["coverage"] == len(want) / len(rows)
    old = calibrate.automated_gate(rows, tempered, {"thresholds": {"0.05": {"confidence": 0.8}}}, 0.05)
    assert old == fitted                                  # a basal-1.5 file (no per_type) keeps its pooled one
    assert calibrate.automated_gate(rows, tempered, {"prompt": 2, "thresholds": {"0.05": {"confidence": 0.8}}},
                                    0.05)["coverage"] == 0.0


def test_each_row_is_tempered_once(monkeypatch):
    calls = []
    real = calibrate.temper
    monkeypatch.setattr(calibrate, "temper", lambda p, T: calls.append(1) or real(p, T))
    rows = noul_rows(n_fit=200, n_cert=100)
    fit(rows, {}, [0.01, 0.05], min_per_type=10, conformal=())
    # FIT and CERT rows with the fitted temperature, CERT with the shipped one, all rows at T = 1 for the accuracy,
    # and the leave-one-family-out refits: never once per target and per report
    assert len(calls) <= 3 * len(rows) + 60 * 2 * len(rows)
    tempered = calibrate.Tempered({"noul": 1.3})
    n = len(calls)
    tempered.cs(rows)
    tempered.cs(rows)
    assert len(calls) - n == len(rows)


def rows_file(path, rows, model="Remek/basal-x", baseline=None):
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model=model, engine="eager",
                data="labels.jsonl", requests=len(rows), baseline=baseline, orders=1, layout="canonical")
    path.write_text("\n".join(json.dumps(x) for x in [head] + rows) + "\n", encoding="utf-8")
    return path


def test_a_bad_row_of_a_rows_file_is_left_out_before_the_sample_size(tmp_path, capsys):
    rows = [dict(r, line=i, name="q") for i, r in enumerate(noul_rows(n_fit=120, n_cert=60))]
    path = rows_file(tmp_path / "rows.jsonl", rows + [{"line": 999, "gold": 0}])     # a row without a type
    main([str(path), "--from-rows", "--report-only", "--targets", "0.05", "--no-calibration", "--min-per-type", "10"])
    assert "1 row(s) left out of the fit: line 999: not a row of a decision type" in capsys.readouterr().out


SHIPPED = {"temperature_per_prim": {"noul": 1.1}, "thresholds": {"0.05": {"confidence": 0.7, "per_type": {
    "noul": {"confidence": 0.7}}}}, "task_thresholds": {"classify": {"0.05": 0.6}}}


def test_a_baseline_of_another_model_is_not_compared_or_kept(tmp_path, capsys, monkeypatch):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "Remek/basal-y"})), encoding="utf-8")
    rows = [dict(r, line=i, name="q") for i, r in enumerate(noul_rows(n_fit=120, n_cert=60))]
    path = rows_file(tmp_path / "rows.jsonl", rows, baseline=str(base))
    main([str(path), "--from-rows", "--report-only", "--targets", "0.05", "--min-per-type", "10"])
    err = capsys.readouterr().err
    assert "fitted on Remek/basal-y, the served model is Remek/basal-x" in err and "not compared or kept" in err
    # the file basal-calibrate compares with in a run: kept for its temperatures only, as basal-serve keeps it
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: tmp_path)
    data, f = compared(tmp_path, "eager", str(base), None, 1, False, identity=("Remek/basal-x", None, None))
    assert f == base and data["temperature_per_prim"] == {"noul": 1.1}
    assert not {"thresholds", "task_thresholds"} & set(data)
    data, _ = compared(tmp_path, "eager", str(base), None, 1, False, identity=("Remek/basal-y", None, None))
    assert data["thresholds"] == SHIPPED["thresholds"]


def test_a_server_name_is_not_checked_against_the_compared_file(tmp_path, capsys, monkeypatch):
    # --url without --model: the server's name (its --name, or the folder's name with "+adapter") stands in for the
    # model, and is no identity to check the --calibration file against
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "Remek/basal-x", "adapter": "ad@sha256:abc",
                                                        "orders": 1})), encoding="utf-8")
    entry = dict(name="basal-x+ad", mode="eager", prompt_format=1, orders=1, adapter="ad@sha256:abc",
                 temperature_per_prim={})
    monkeypatch.setattr(httpx, "get", lambda *_, **__: SimpleNamespace(status_code=200, json=lambda: {"models": [entry]}))
    a = calibrate.parser().parse_args(["in.jsonl", "--url", "http://127.0.0.1:1", "--calibration", str(base)])
    c = calibrate.http_context(a)
    assert c.model == "basal-x+ad" and c.baseline["thresholds"] == SHIPPED["thresholds"]
    assert "not compared or kept" not in capsys.readouterr().err
