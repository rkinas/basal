"""basal-calibrate --keep-shipped (#93): a type the labels do not certify keeps the compared file's temperature and
thresholds together, the other types are fitted, and the file says which types were kept."""
import hashlib
import json

import pytest

from basal.calibrate import KEPT_SOURCE, cert_note, check_keep, fit, main, table, warnings
from basal.server import certified

SHIPPED = {"temperature_per_prim": {"choice": 1.04, "noul": 1.56, "score": 1.33},
           "thresholds": {"0.05": {"confidence": 0.9, "per_type": {"choice": {"confidence": 0.9},
                                                                    "noul": {"confidence": 0.6}}}}}


def rows(n_choice=400, n_noul=60):
    """Many choice decisions (10% wrong, at a lower confidence) and too few noul ones to certify 5% on FIT (~42 < 59),
    all of them right."""
    out = [dict(prim="choice", group=f"c{i}", family=f"f{i % 3}", gold=0,
                p=[0.6, 0.4] if i % 10 == 0 else [0.95, 0.05]) for i in range(n_choice)]
    for r in out[::10]:
        r["gold"] = 1
    out += [dict(prim="noul", group=f"n{i}", family=f"f{i % 3}", gold=0, p=[0.95, 0.05]) for i in range(n_noul)]
    return out


def test_without_the_option_the_uncertified_type_defers_and_the_warning_names_it():
    cal, report = fit(rows(), SHIPPED, [0.05])
    assert cal["thresholds"]["0.05"]["per_type"]["noul"]["confidence"] is None
    assert certified(cal, 0.05, "noul") is None
    hint = [w for w in warnings(report, [0.05]) if w.startswith("max_error 0.05: noul has no certified threshold")]
    assert hint and "the compared file certified 0.6000: --keep-shipped noul keeps" in hint[0]
    assert report["kept"] == []


def test_a_kept_type_takes_the_shipped_temperature_and_threshold_and_the_others_are_fitted():
    cal, report = fit(rows(), SHIPPED, [0.05], keep=("noul",))
    assert cal["temperature_per_prim"]["noul"] == 1.56 and cal["temperature_source"]["noul"] == KEPT_SOURCE
    assert cal["temperature_source"]["choice"].startswith("fitted")
    noul = cal["thresholds"]["0.05"]["per_type"]["noul"]
    assert noul["confidence"] == 0.6 and noul["kept"] == "shipped"
    assert noul["cert_accepted"] > 0 and noul["cert_error"] == 0.0 and noul["fit_coverage"] == 1.0
    assert certified(cal, 0.05, "noul") == 0.6                     # basal-serve automates it as the shipped file did
    assert "kept" not in cal["thresholds"]["0.05"]["per_type"]["choice"]
    assert report["kept"] == ["noul"] and report["gates"]["automated_at"]["0.05"] > 0
    assert not any("noul has no certified threshold" in w for w in warnings(report, [0.05]))
    assert "noul 0.6000" in table(report) and "kept shipped" in table(report)
    note = cert_note("noul", 0.05, dict(noul, cert_accepted=14, cert_error=0.0, cert_upper95=0.19))
    assert note.startswith("noul (kept shipped threshold) at max_error 0.05: certified on the compared file's data;")


def test_a_kept_type_without_a_shipped_threshold_at_a_target_has_none_there():
    cal, _ = fit(rows(), SHIPPED, [0.01, 0.05], keep=("noul",))
    one = cal["thresholds"]["0.01"]["per_type"]["noul"]
    assert one["confidence"] is None and one["kept"] == "shipped" and "compared file has no threshold" in one["reason"]
    assert cal["thresholds"]["0.05"]["per_type"]["noul"]["confidence"] == 0.6


def test_a_kept_choice_keeps_its_bucket_temperatures_in_file_2():
    shipped = dict(SHIPPED, prompt=2, temperature_per_bucket={"choice": {"2-10": 1.2, "11-26": 1.5}})
    cal, _ = fit(rows(), shipped, [0.05], fmt=2, keep=("choice",))
    assert cal["temperature_per_bucket"]["choice"] == {"2-10": 1.2, "11-26": 1.5}
    assert cal["temperature_per_prim"]["choice"] == 1.04
    assert cal["thresholds"]["0.05"]["per_type"]["choice"]["confidence"] == 0.9


def test_keep_needs_a_known_type_and_a_compared_file_with_its_temperature_and_threshold():
    assert check_keep((), {}) == () and check_keep(("noul", "choice"), SHIPPED) == ("choice", "noul")
    with pytest.raises(SystemExit, match="give question types among choice, noul, score"):
        check_keep(("act",), SHIPPED)
    with pytest.raises(SystemExit, match="no noul temperature to keep"):
        check_keep(("noul",), {})
    with pytest.raises(SystemExit, match="no score threshold to keep"):
        check_keep(("score",), SHIPPED)


def test_the_refit_writes_the_kept_types_into_fitted_on(tmp_path):
    baseline, data = tmp_path / "shipped.json", tmp_path / "rows.jsonl"
    baseline.write_text(json.dumps(SHIPPED), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=460, baseline=str(baseline), orders=None)
    data.write_text("\n".join(json.dumps(x) for x in [head] + rows()) + "\n", encoding="utf-8")
    out = tmp_path / "cal.json"
    main([str(data), "--from-rows", "--keep-shipped", "noul", "--targets", "0.05", "--out", str(out)])
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert cal["fitted_on"]["kept_shipped"] == ["noul"] and certified(cal, 0.05, "noul") == 0.6
    main([str(data), "--from-rows", "--targets", "0.05", "--out", str(out)])
    assert "kept_shipped" not in json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    with pytest.raises(SystemExit, match="no noul temperature to keep"):
        main([str(data), "--from-rows", "--no-calibration", "--keep-shipped", "noul", "--out", str(out)])


def noul_errors(per_group=1, n=100, wrong=10):
    """The choice rows of rows() and n noul decisions above the kept 0.6, `wrong` of them wrong; per_group > 1 puts
    them in groups of that size, the wrong ones together in the first group."""
    return rows(n_noul=0) + [dict(prim="noul", group=f"n{i // per_group}", family=f"f{i % 3}", gold=int(i < wrong),
                                  p=[0.95, 0.05]) for i in range(n)]


def test_a_kept_threshold_your_decisions_refute_is_withdrawn():
    cal, report = fit(noul_errors(), SHIPPED, [0.05], keep=("noul",))
    noul = cal["thresholds"]["0.05"]["per_type"]["noul"]
    assert noul["confidence"] is None and noul["refuted"] == 0.6 and noul["kept"] == "shipped"
    assert "your FIT and CERT decisions refute it: 10 errors in 100 accepted" in noul["reason"]
    assert certified(cal, 0.05, "noul") is None and cal["temperature_per_prim"]["noul"] == 1.56
    assert any("the kept shipped threshold 0.6000 is refuted" in w for w in warnings(report, [0.05]))
    # the same 10 errors in one group of 10 go together (ICC 1, design effect 10): about one independent error in
    # ten, which refutes nothing
    cal, _ = fit(noul_errors(per_group=10), SHIPPED, [0.05], keep=("noul",))
    assert cal["thresholds"]["0.05"]["per_type"]["noul"]["confidence"] == 0.6


POOLED = {"temperature_per_prim": {"choice": 1.04, "noul": 1.56}, "thresholds": {"0.05": {"confidence": 0.6}}}


def test_a_pooled_only_threshold_is_not_kept_as_a_type_certificate():
    # a format-1 file without per_type (basal-1.5): its one threshold was certified over all the types together
    assert certified(POOLED, 0.05, "noul") == 0.6                  # basal-serve applies it to every type of that file
    with pytest.raises(SystemExit, match="only one pooled over all the types"):
        check_keep(("noul",), POOLED)
    _, report = fit(rows(), POOLED, [0.05])
    assert not any("--keep-shipped" in w for w in warnings(report, [0.05]))   # no hint at an option that refuses


@pytest.mark.parametrize("name, extra, refused", [
    ("CALIBRATION.json", {}, None),                               # the bf16 engines' file, read with eager: kept
    ("custom.json", {"engine": "fast"}, None),                    # fast and eager share CALIBRATION.json
    ("custom.json", {"engine": "gguf"}, "holds for the gguf engine"),
    ("CALIBRATION.fp8.json", {}, "holds for the fp8 engine"),     # its name says the engine
    ("custom.json", {}, "warning"),                               # says nothing: kept, with a warning
])
def test_keep_refuses_the_thresholds_of_another_engine(tmp_path, capsys, name, extra, refused):
    baseline, data, out = tmp_path / name, tmp_path / "rows.jsonl", tmp_path / "cal.json"
    baseline.write_text(json.dumps(dict(SHIPPED, **extra)), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=460, baseline=str(baseline), orders=None)
    data.write_text("\n".join(json.dumps(x) for x in [head] + rows()) + "\n", encoding="utf-8")
    argv = [str(data), "--from-rows", "--keep-shipped", "noul", "--targets", "0.05", "--out", str(out)]
    if refused and refused != "warning":
        with pytest.raises(SystemExit, match=f"{refused}, this run reads the bf16 one"):
            main(argv)
        return
    main(argv)
    assert json.loads(out.read_text(encoding="utf-8"))["fitted_on"]["kept_shipped"] == ["noul"]
    assert ("does not say which engine" in capsys.readouterr().err) is (refused == "warning")


def test_the_written_file_names_the_file_its_kept_thresholds_come_from(tmp_path):
    baseline, data, out = tmp_path / "CALIBRATION.json", tmp_path / "rows.jsonl", tmp_path / "cal.json"
    baseline.write_text(json.dumps(dict(SHIPPED, certified_on="CERT split of the shipped data")), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=460, baseline=str(baseline), orders=None)
    data.write_text("\n".join(json.dumps(x) for x in [head] + rows()) + "\n", encoding="utf-8")
    main([str(data), "--from-rows", "--keep-shipped", "noul", "--targets", "0.05", "--out", str(out)])
    fitted = json.loads(out.read_text(encoding="utf-8"))["fitted_on"]
    assert fitted["kept_shipped_from"] == dict(file="CALIBRATION.json",
                                               sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                                               certified_on="CERT split of the shipped data")
    main([str(data), "--from-rows", "--targets", "0.05", "--out", str(out)])
    assert "kept_shipped_from" not in json.loads(out.read_text(encoding="utf-8"))["fitted_on"]


def test_under_an_adapter_the_base_models_thresholds_are_not_kept_nor_by_the_refit(tmp_path, monkeypatch, capsys):
    """A run against a server with --adapter compares with the model's file, which is the base model's: kept for its
    temperatures only, as basal-serve keeps it, so --keep-shipped has no threshold to keep. The rows it saves say so,
    and their --from-rows refit keeps none either (it kept the base model's noul threshold for the adapter)."""
    from types import SimpleNamespace

    import httpx

    import basal.calibrate as calibrate_module
    md = tmp_path / "models--owner--stub" / "snapshots" / ("a" * 40)
    md.mkdir(parents=True)
    (md / "basal.json").write_text(json.dumps({"prompt": 1}), encoding="utf-8")
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"state": "s", "questions": {"q": {
        "type": "choice", "criteria": ["a", "b"], "gold": "a"}}}) + "\n", encoding="utf-8")
    entry = dict(name="stub", mode="eager", prompt_format=1, orders=1, adapter="stub-adapter@sha256:abc",
                 temperature_per_prim={})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"models": [entry]}))
    monkeypatch.setattr(calibrate_module, "resolve", lambda *a, **k: md)

    async def readouts(*args):
        return [{"answers": {"q": {"probabilities": {"a": 0.8, "b": 0.2}}}}]
    monkeypatch.setattr(calibrate_module, "over_http", readouts)
    url = [str(labels), "--url", "http://h", "--model", "owner/stub"]
    with pytest.raises(SystemExit, match="no noul threshold to keep"):
        main(url + ["--keep-shipped", "noul", "--out", str(tmp_path / "x.json")])
    rows, out = tmp_path / "rows.jsonl", tmp_path / "cal.json"
    main(url + ["--save-rows", str(rows), "--out", str(out)])
    assert "belongs to the base model" in capsys.readouterr().err
    assert "belongs to the base model" in json.loads(rows.read_text(encoding="utf-8").splitlines()[0])["baseline_why"]
    with pytest.raises(SystemExit, match="no noul threshold to keep"):
        main([str(rows), "--from-rows", "--keep-shipped", "noul", "--targets", "0.05", "--out", str(out)])
    # a file given with --calibration is applied as it is (it may be fitted with the adapter), as basal-serve does
    main([str(rows), "--from-rows", "--calibration", str(md / "CALIBRATION.json"), "--keep-shipped", "noul",
          "--targets", "0.05", "--out", str(out)])
    assert certified(json.loads(out.read_text(encoding="utf-8")), 0.05, "noul") == 0.6
