"""Benchmarks of a loaded model (the preset, a load test, a determinism check) and an engine-mode comparison of a
downloaded bf16 model in basal-ui's process."""
import pytest

EXTERNAL = ("ollama", "llamacpp")  # need a service of their own


def test_the_benchmark_preset_and_a_load_test(ui, text_model):
    model_id = text_model["id"]
    ui.load(model_id)
    preset = ui.run_job("/api/benchmark", {"id": model_id}, "benchmark")["result"]
    assert preset["errors"] == 0 and preset["items"] == 44 and 0 < preset["p50_ms"] <= preset["p95_ms"]
    assert preset["decisions_per_s"] > 0 and 0 <= preset["accuracy"]["overall"] <= 1
    params = {"n_seq": 20, "n_conc": 40, "concurrency": 4, "questions_per_request": 2}
    load = ui.run_job("/api/benchmark", {"id": model_id, "kind": "loadtest", "params": params}, "benchmark")["result"]
    assert load["kind"] == "loadtest" and load["errors"] == 0 and load["params"]["n_seq"] == 20
    assert 0 < load["p50_ms"] <= load["p95_ms"] and load["questions_per_s"] == pytest.approx(2 * load["decisions_per_s"])
    kinds = [r.get("kind", "benchmark") for r in ui.get("/api/benchmarks")[model_id]]
    assert kinds[-2:] == ["benchmark", "loadtest"]


def test_a_deterministic_model_answers_alone_and_under_load_alike(ui, text_model):
    model_id = text_model["id"]
    ui.load(model_id, {"deterministic": True})
    body = {"id": model_id, "kind": "determinism", "params": {"determinism": 16, "concurrency": 8}}
    d = ui.run_job("/api/benchmark", body, "benchmark")["result"]["determinism"]
    assert d["requests"] == 16 and d["identical"] == 16 and d["max_abs_diff"] == 0.0 and d["decision_flips"] == 0
    ui.load(model_id)


def test_two_engine_modes_agree_and_their_memory_is_released(ui, bf16_model):
    model_id = bf16_model["id"]
    default = next(m["mode"] for m in ui.get("/api/models/options", params={"id": model_id})["modes"] if m["default"])
    others = [m for m in ui.get("/api/modes", params={"id": model_id})["modes"]
              if m not in EXTERNAL and m not in ("eager-fp32", default)]
    if not others:
        pytest.skip(f"{model_id}: no in-process mode besides {default} here")
    modes = [default, others[0]]  # the serving engine is the reference
    committed = ui.status()["memory"]["committed_bytes"]
    body = {"id": model_id, "kind": "modes", "modes": modes, "n": 44, "lat_n": 10}
    result = ui.run_job("/api/benchmark", body, "benchmark")["result"]
    rows = {r["mode"]: r for r in result["modes"]}
    assert set(rows) == set(modes) and not [r for r in rows.values() if "error" in r], rows
    assert rows[modes[0]].get("agree") is None and rows[modes[1]]["agree"] >= 0.9  # the same decisions, mostly
    assert all(r["dec_s"] > 0 and r["lat2_ms"] > 0 for r in rows.values())
    assert ui.status()["memory"]["committed_bytes"] == committed
    assert ui.detail("POST", "/api/benchmark", {"id": model_id, "kind": "modes", "modes": ["warp"]}, 400)
