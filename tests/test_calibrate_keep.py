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
