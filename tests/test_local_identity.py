"""A calibration file fitted on a local model folder is matched to the served model by the folder's files
(fitted_on.model_content, server.model_content), not by its name: a file fitted on /data/run1/final was applied to
/data/run2/final, another checkpoint in a folder of the same name, and refused for the same weights under another
name. A file that records no files (basal-calibrate before 2.8) keeps the name match, with a warning."""
import json
from types import SimpleNamespace

import httpx

import basal.calibrate as calibrate_module
from basal.calibrate import main
from basal.server import calibration_mismatch, matched_by_name, model_content, model_content_id


def folder(path, weights, readme="card", extra=None):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(weights)
    (path / "README.md").write_text(readme, encoding="utf-8")
    for name, data in (extra or {}).items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(data)
    return path


def fitted(path, content=True):
    on = {"model": str(path)}
    if content:
        on["model_content"] = model_content_id(path)
    return {"temperature_per_prim": {"noul": 1.2}, "fitted_on": on}


def test_another_checkpoint_in_a_folder_of_the_same_name_is_not_this_model(tmp_path):
    run1 = folder(tmp_path / "run1" / "final", b"\x01" * 4096)
    run2 = folder(tmp_path / "run2" / "final", b"\x02" * 4096)
    cal = fitted(run1)
    assert "holds other model files" in calibration_mismatch(cal, str(run2), run2)
    assert calibration_mismatch(cal, str(run1), run1) is None


def test_a_copy_of_the_same_weights_matches_whatever_its_name(tmp_path):
    run1 = folder(tmp_path / "run1" / "final", b"\x01" * 4096)
    copy = folder(tmp_path / "elsewhere" / "deployed", b"\x01" * 4096, readme="another card",
                  extra={".cache/huggingface/download/model.safetensors.metadata": b"downloaded at 12:00"})
    assert calibration_mismatch(fitted(run1), str(copy), copy) is None


def test_a_file_that_records_no_files_is_matched_by_name_with_a_warning(tmp_path):
    run1 = folder(tmp_path / "run1" / "final", b"\x01" * 4096)
    run2 = folder(tmp_path / "run2" / "final", b"\x02" * 4096)
    old = fitted(run1, content=False)
    assert calibration_mismatch(old, str(run2), run2) is None                  # (as before: the name decides)
    assert "matched by the folder's name alone" in matched_by_name(old)
    assert matched_by_name(fitted(run1)) is None
    hub = {"fitted_on": {"model": "Org/basal-x", "revision": "a" * 40}}       # (a Hub model: its revision decides)
    assert matched_by_name(hub) is None and matched_by_name({"temperature_per_prim": {}}) is None


def test_the_content_is_sampled_and_follows_the_files_that_make_the_answers(tmp_path):
    big = bytearray(8 * 2**20)
    a = folder(tmp_path / "a", bytes(big))
    big[len(big) // 2] = 1                                                      # (a change in the middle)
    b = folder(tmp_path / "b", bytes(big))
    assert model_content_id(a) != model_content_id(b)
    c = folder(tmp_path / "c", bytes(8 * 2**20), readme="other", extra={"CALIBRATION.json": b"{}", "prepared/x": b"1"})
    assert model_content_id(c) == model_content_id(a)                           # (card, calibration, prepared/: not)
    assert set(model_content(c)) == {"config.json", "model.safetensors"}
    (tmp_path / "empty").mkdir()
    assert model_content(tmp_path / "empty") is None and model_content(tmp_path / "missing") is None
    assert model_content_id(a, "gguf") is None                                  # (weights elsewhere)


def test_basal_calibrate_records_the_files_of_a_local_folder(tmp_path, monkeypatch):
    md = folder(tmp_path / "models" / "final", b"\x03" * 4096)
    (md / "basal.json").write_text(json.dumps({"prompt": 1}), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"state": "s", "questions": {"q": {
        "type": "choice", "criteria": ["a", "b"], "gold": "a"}}}) + "\n", encoding="utf-8")
    entry = dict(name="final", mode="eager", prompt_format=1, orders=1, temperature_per_prim={})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"models": [entry]}))
    monkeypatch.setattr(calibrate_module, "resolve", lambda *a, **k: md)

    async def readouts(*args):
        return [{"answers": {"q": {"probabilities": {"a": 0.8, "b": 0.2}}}}]
    monkeypatch.setattr(calibrate_module, "over_http", readouts)
    rows, first, again = tmp_path / "rows.jsonl", tmp_path / "first.json", tmp_path / "again.json"
    main([str(labels), "--url", "http://h", "--model", str(md), "--no-calibration", "--save-rows", str(rows),
          "--out", str(first)])
    main([str(rows), "--from-rows", "--out", str(again)])
    for out in (first, again):
        assert json.loads(out.read_text(encoding="utf-8"))["fitted_on"]["model_content"] == model_content_id(md)
