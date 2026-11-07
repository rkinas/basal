"""The compared calibration file as basal-serve reads it (basal-calibrate): a temperature written as text ("1.2", which
basal-serve reads as 1.2) is a number to the fit, a null thresholds entry is no threshold, and a value basal-serve
refuses at start-up stops the command with the key named. They failed with a traceback after the whole readout (in
the report table or the blend), and the run's results were lost."""
import json

import pytest

from basal.calibrate import compared, main
from tests.test_calibrate_keep import rows


def refit(tmp_path, baseline, n_noul=60, extra=()):
    base, data, out = tmp_path / "shipped.json", tmp_path / "rows.jsonl", tmp_path / "cal.json"
    base.write_text(json.dumps(baseline), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=1, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=460, baseline=str(base), orders=None)
    data.write_text("\n".join(json.dumps(x) for x in [head] + rows(n_noul=n_noul)) + "\n", encoding="utf-8")
    main([str(data), "--from-rows", "--targets", "0.05", "--out", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("n_noul, extra", [(60, ()), (20, ("--min-blend", "10")), (5, ("--min-blend", "10"))])
def test_a_shipped_temperature_written_as_text_is_a_number(tmp_path, n_noul, extra):
    """Fitted (choice), blended (20 noul rows) or kept as shipped (5 noul rows): no failure, numbers written."""
    cal = refit(tmp_path, {"temperature_per_prim": {"choice": "1.04", "noul": "1.56"}}, n_noul, extra)
    assert all(isinstance(T, float) for T in cal["temperature_per_prim"].values())


def test_a_null_thresholds_entry_is_no_threshold(tmp_path):
    cal = refit(tmp_path, {"temperature_per_prim": {"choice": 1.04, "noul": 1.56}, "thresholds": {"0.05": None}})
    assert "0.05" in cal["thresholds"]


@pytest.mark.parametrize("baseline, said", [
    ({"temperature_per_prim": {"noul": "0"}}, "temperature_per_prim['noul'] is '0'"),
    ({"temperature_per_prim": {"noul": "warm"}}, "temperature_per_prim['noul'] is 'warm'"),
    ({"prompt": 2, "temperature_per_bucket": {"choice": {"2-x": 1.0}}}, "bad bucket '2-x'"),
])
def test_a_value_basal_serve_refuses_stops_the_command_with_its_key(tmp_path, baseline, said):
    with pytest.raises(SystemExit, match=said.replace("[", r"\[").replace("]", r"\]")):
        refit(tmp_path, baseline)
    path = tmp_path / "CALIBRATION.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(SystemExit, match="a temperature is|bad bucket"):  # (a run: before any readout)
        compared(tmp_path, "eager", str(path), None, None, False)


def test_a_compared_file_of_another_prompt_format_is_refused_as_basal_serve_refuses_it(tmp_path):
    """A file 1 compared on a format-2 model (in a run, or with the rows of one) stops the command, as basal-serve stops
    at start-up: its temperatures and thresholds were fitted on other prompts (they were blended in, or kept with
    --keep-shipped)."""
    file1 = {"prompt": 1, "temperature_per_prim": {"choice": 1.04, "noul": 1.56}}
    path = tmp_path / "CALIBRATION.json"
    path.write_text(json.dumps(file1), encoding="utf-8")
    with pytest.raises(SystemExit, match="fitted for prompt format 1, the model uses format 2"):
        compared(tmp_path, "eager", str(path), None, None, False, fmt=2)
    assert compared(tmp_path, "eager", str(path), None, None, False, fmt=1)[0]["temperature_per_prim"]["noul"] == 1.56
    base, data = tmp_path / "shipped.json", tmp_path / "rows.jsonl"
    base.write_text(json.dumps(file1), encoding="utf-8")
    head = dict(basal_calibrate_rows=1, fmt=2, kind="eager", quant=None, model="stub", engine="eager",
                data="labels.jsonl", requests=460, baseline=str(base), orders=None)
    data.write_text("\n".join(json.dumps(x) for x in [head] + rows(n_noul=5)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="fitted for prompt format 1, the model uses format 2"):
        main([str(data), "--from-rows", "--targets", "0.05", "--out", str(tmp_path / "cal.json")])


def test_the_run_compares_with_the_numbers_basal_serve_reads(tmp_path):
    path = tmp_path / "CALIBRATION.json"
    path.write_text(json.dumps({"temperature_per_prim": {"noul": "1.2", "custom": "x"}}), encoding="utf-8")
    data, f, _ = compared(tmp_path, "eager", str(path), None, None, False)
    assert data["temperature_per_prim"] == {"noul": 1.2, "custom": "x"} and f == path  # (not a type: never read)
