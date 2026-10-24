import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from basal import bench as offline
from basal import loadtest
from basal.ui import bench, catalog, settings
from basal.ui.app import Fleet, Runner, create_app
from basal.ui.catalog import Variant
from basal.ui.jobs import Cancelled, Job, LogBuffer
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())


def write_set(path, golds):
    path.write_text("\n".join(json.dumps({"state": f"s{i}", "question": "q?", "options": ["a", "b"], "gold": g})
                              for i, g in enumerate(golds)) + "\n")
    return path


class AnswersOptionOne:
    """A fleet whose model always prefers option_1."""

    def __init__(self):
        self.sources = []

    async def adecide(self, model_id, body, path="/v1/systemone", source="api", client=None):
        self.sources.append(source)
        await asyncio.sleep(0.001)
        return httpx.Response(200, json={"answers": {"q0": {"probabilities": {"option_1": 0.7, "option_2": 0.3}}},
                                         "usage": {"input_tokens": 10}}, request=httpx.Request("POST", "http://x"))


def test_run_benchmark_measures_latency_throughput_and_accuracy(tmp_path):
    fleet = AnswersOptionOne()
    paths = [str(write_set(tmp_path / "two.jsonl", [0, 1]))]
    job = Job("benchmark", "m", "two", LogBuffer())
    phases = []
    job.progress = lambda phase, done=0, total=0: phases.append(phase)
    result = asyncio.run(bench.run_benchmark(fleet, "m", paths, "two", job, n_seq=4, n_conc=10, concurrency=3))
    assert result["accuracy"] == {"overall": 0.5, "by_file": {"two": 0.5}}
    assert (result["model"], result["set"], result["items"], result["errors"]) == ("m", "two", 2, 0)
    assert 0 < result["p50_ms"] <= result["p95_ms"] and result["mean_ms"] > 0
    assert result["decisions_per_s"] > 0 and result["input_tokens_per_s"] == pytest.approx(result["decisions_per_s"] * 10)
    assert set(fleet.sources) == {"benchmark"} and len(fleet.sources) == bench.WARMUP + 4 + 10
    assert phases[0] == "warm-up" and "sequential" in phases and phases[-1] == "concurrent"


def test_bundled_set_and_custom_set_validation(tmp_path):
    label, paths = bench.resolve_set("bundled")
    assert label == "bundled" and len(bench.load_items(paths)) == 44
    for spec, reason in ((str(tmp_path / "missing.jsonl"), "does not exist"),
                         (str(write_set(tmp_path / "small.jsonl", [0, 1, 0])), "3 items"),
                         (str(tmp_path), "at least 10")):
        with pytest.raises(ValueError, match=reason):
            bench.resolve_set(spec)
    (tmp_path / "bad.jsonl").write_text('{"state": "x"\n')
    with pytest.raises(ValueError, match="bad.jsonl:1: not JSON"):
        bench.resolve_set(str(tmp_path / "bad.jsonl"))
    (tmp_path / "set").mkdir()
    write_set(tmp_path / "set" / "one.jsonl", [0] * 6)
    write_set(tmp_path / "set" / "two.jsonl", [1] * 6)
    assert bench.resolve_set(str(tmp_path / "set"))[1] == [str(tmp_path / "set" / f) for f in ("one.jsonl",
                                                                                               "two.jsonl")]


def test_benchmark_routes_run_one_at_a_time_and_store_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    release = threading.Event()

    async def fake_run(fleet, model_id, paths, label, job):
        job.progress("sequential", 1, 30)
        while not release.is_set():
            if job.cancel.is_set():
                raise Cancelled()
            await asyncio.sleep(0.01)
        return {"model": model_id, "set": label, "p50_ms": 1.0}

    monkeypatch.setattr(bench, "run_benchmark", fake_run)
    stats = Stats(tmp_path / "stats.json")
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings, hardware_fn=lambda: HW, stats=stats,
        history=History(tmp_path / "h.jsonl"))
    runner = Runner(internal_secret="x")
    runner.state, runner.model = "ready", "mini"
    fleet.runners["mini"] = runner
    app = create_app(catalog_fn=lambda **_: ([], "test"), hardware_fn=lambda: HW, fleet=fleet, stats=stats,
                     history=fleet.history)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/benchmark", json={"id": "nope"}).status_code == 404
        too_small = client.post("/api/benchmark", json={"id": "mini", "set": str(write_set(
            tmp_path / "s.jsonl", [0, 1]))})
        assert too_small.status_code == 400 and "at least 10" in too_small.json()["detail"]
        started = client.post("/api/benchmark", json={"id": "mini"})
        assert started.status_code == 202 and started.json()["running"] is True
        busy = client.post("/api/benchmark", json={"id": "mini"})
        assert busy.status_code == 409 and busy.json()["detail"] == "a benchmark job is already running"

        def job():
            return client.get("/api/status").json()["jobs"]["benchmark"]
        deadline = time.monotonic() + 3
        while job()["phase"] != "sequential" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job()["phase"] == "sequential"
        release.set()
        deadline = time.monotonic() + 3
        while job()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job()["phase"] == "done" and job()["result"]["p50_ms"] == 1.0
        assert client.get("/api/benchmarks").json() == {"mini": [{"model": "mini", "set": "bundled", "p50_ms": 1.0}]}
        assert {"id": "job:benchmark", "state": "done"} in client.get("/api/logs").json()["sources"]

        release.clear()
        client.post("/api/benchmark", json={"id": "mini"})
        assert client.delete("/api/benchmark").status_code == 200
        deadline = time.monotonic() + 3
        while job()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job()["phase"] == "cancelled"


def make_app(tmp_path, monkeypatch, hw=HW, variants=()):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings, hardware_fn=lambda: hw,
                  stats=Stats(tmp_path / "stats.json"), history=History(tmp_path / "h.jsonl"))
    runner = Runner(internal_secret="x")
    runner.state, runner.model, runner.base_url, runner.models_entry = "ready", "mini", "http://child", {"early_exit": []}
    fleet.runners["mini"] = runner
    app = create_app(catalog_fn=lambda **_: (list(variants), "test"), hardware_fn=lambda: hw, fleet=fleet,
                     stats=fleet.stats, history=fleet.history)
    return app, fleet


def finished(client):
    deadline = time.monotonic() + 5
    while client.get("/api/status").json()["jobs"]["benchmark"]["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    return client.get("/api/status").json()["jobs"]["benchmark"]


def test_load_test_and_determinism_check_against_the_model(tmp_path, monkeypatch):
    def handler(request):  # a deterministic server
        assert request.headers["authorization"] == "Bearer x"
        return httpx.Response(200, json={"answers": {"q0": {"probabilities": {"option_1": 0.7, "option_2": 0.3}}}})

    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    app, fleet = make_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        bad = client.post("/api/benchmark", json={"id": "mini", "kind": "loadtest", "params": {"n_seq": 0}})
        assert bad.status_code == 400 and "n_seq" in bad.json()["detail"]
        no_exit = client.post("/api/benchmark", json={"id": "mini", "kind": "loadtest", "params": {"early_exit": "0.99"}})
        assert no_exit.status_code == 400 and "no early exits" in no_exit.json()["detail"]
        params = {"n_seq": 4, "n_conc": 6, "concurrency": 3}
        assert client.post("/api/benchmark", json={"id": "mini", "kind": "loadtest", "params": params}).status_code == 202
        result = finished(client)["result"]
        assert result["kind"] == "loadtest" and result["errors"] == 0 and result["p50_ms"] > 0
        assert result["decisions_per_s"] > 0 and result["params"]["n_seq"] == 4
        check = {"id": "mini", "kind": "determinism", "params": {"determinism": 5, "concurrency": 2}}
        assert client.post("/api/benchmark", json=check).status_code == 202
        d = finished(client)["result"]
        assert d["kind"] == "determinism" and d["determinism"]["identical"] == d["determinism"]["requests"] == 5
        assert [r["kind"] for r in client.get("/api/benchmarks").json()["mini"]] == ["loadtest", "determinism"]


BF16 = Variant("bf16", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 4 * 1024**3, False)


def test_mode_comparison_runs_each_mode_in_process_and_reserves_its_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: tmp_path)
    monkeypatch.setattr(offline, "resolve", lambda path, metadata_only=False: tmp_path)
    monkeypatch.setattr(offline, "model_format", lambda md: 2)
    built, reserved = [], []

    def fake_measure(mode, md, qs, fmt, lat_n, reference_mode, ref_top=None, *rest):
        if mode == "ollama":
            raise RuntimeError("no Ollama server")
        built.append(mode)
        reserved.append(dict(fleet.reserved))
        top = [0] * len(qs) if mode == "eager-fp32" else [0] * (len(qs) - 1) + [1]
        row = {"mode": mode, "reference_mode": reference_mode, "lat1_ms": 1.0, "lat2_ms": 2.0, "dec_s": 10.0}
        if ref_top is not None:
            row["agree"] = sum(x == y for x, y in zip(top, ref_top)) / len(ref_top)
        return row, top

    monkeypatch.setattr(offline, "measure_mode", fake_measure)
    app, fleet = make_app(tmp_path, monkeypatch, variants=[BF16])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        selectable = client.get("/api/modes", params={"id": "bf16"}).json()["modes"]
        assert selectable[:2] == ["eager-fp32", "eager"] and "fast" not in selectable  # a CPU machine
        unknown = client.post("/api/benchmark", json={"id": "bf16", "kind": "modes", "modes": ["fast"]})
        assert unknown.status_code == 400 and "fast cannot run here" in unknown.json()["detail"]
        body = {"id": "bf16", "kind": "modes", "modes": ["eager-fp32", "eager"], "n": 20, "lat_n": 5}
        assert client.post("/api/benchmark", json=body).status_code == 202
        result = finished(client)["result"]
        assert built == ["eager-fp32", "eager"] and result["kind"] == "modes"
        assert [r.get("agree") for r in result["modes"]] == [None, 1 - 1 / result["items"]]
        est = catalog.assess(BF16, HW).est_memory_bytes
        assert reserved[0] == {"modes": 2 * est} and fleet.reserved == {}  # eager-fp32: twice the bf16 estimate
        settings.update_memory({"budget_bytes": 2 * 1024**3 + est, "reserve_bytes": 0})
        over = client.post("/api/benchmark", json=body)
        assert over.status_code == 409 and over.json()["detail"].startswith("over memory budget")
        assert client.post("/api/benchmark", json={**body, "modes": ["eager", "ollama"]}).status_code == 202
        rows = finished(client)["result"]["modes"]
        assert rows[0]["mode"] == "eager" and rows[1] == {"mode": "ollama", "error": "no Ollama server"}
