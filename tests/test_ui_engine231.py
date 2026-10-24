"""2.3.1 integration contracts; simulated hardware/children, no model downloads."""
import asyncio
import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from basal import run
from basal.prompt import RequestError
from basal.server import build_app, parser
from basal.ui import calibration, catalog, options, settings
from basal.ui.app import Fleet, Runner, create_app
from basal.ui.gateway import build_app as gateway_app
from basal.ui.hardware import Hardware
from basal.ui.jobs import Jobs
from tests.test_http_status import AppServer, Engine, BODY
from tests.test_ui_options import BF16, CPU, CUDA, wait_for
from tests.test_ui_jobs import wait_done

APPLE = Hardware("apple", "Apple", 64 * 1024**3, None, None, frozenset({"mlx", "mps"}))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))


def test_cpu_packed_and_omni_defaults_match_capabilities(tmp_path):
    runner = Runner(platform_fn=lambda: "cpu")
    assessed = dataclasses.replace(catalog.assess(BF16, CPU), mode="mps")
    command = runner._build_command(BF16, tmp_path, assessed, False, 1, settings.serve_settings())
    assert command[command.index("--mode") + 1] == "mps"
    assert command[command.index("--device") + 1] == "cpu"
    assert catalog.assess(BF16, CPU).mode == "eager"  # CPU packing is opt-in upstream
    omni = dataclasses.replace(BF16, family="basal-omni-1.0", experimental=True)
    assert catalog.assess(omni, dataclasses.replace(CUDA, cuda_capability=(12, 0))).mode == "fast"
    assert catalog.assess(omni, CUDA).mode == "eager"  # Hopper without FA3
    assert catalog.assess(omni, dataclasses.replace(CUDA, engines=CUDA.engines | {"gemma-fast"})).mode == "fast"
    assert not next(m for m in options.available_modes(BF16, dataclasses.replace(CUDA, cuda_capability=(7, 5)), assessed)
                    if m["mode"] == "fast")["available"]


@pytest.mark.parametrize("variant,hw,mode,media", [
    (BF16, CPU, "mps", False), (BF16, APPLE, "mlx", True), (BF16, APPLE, "eager", False),
    (dataclasses.replace(BF16, family="basal-omni-1.0"), APPLE, "mlx", False),
    (dataclasses.replace(BF16, port="GGUF"), APPLE, "mlx", False),
])
def test_packed_rejects_unsupported_paths(variant, hw, mode, media):
    with pytest.raises(ValueError, match="packed"):
        options.check_combination(variant, hw, options.validate({"mode": mode, "mlx_packed": True}), mode, media)


def test_packed_batches_flags_and_validation():
    assert options.DEFAULTS["mlx_packed"] is False
    for mode in ("mlx", "mlx-q8"):
        chosen = options.validate({"mode": mode, "mlx_packed": True})
        options.check_combination(BF16, APPLE, chosen, "mlx")
        assert options.args(chosen) == ["--mlx-packed"]
    for batches in ("compact", "default", "auto", "fine"):
        chosen = options.validate({"mode": "fast", "batches": batches, "state_cache_gb": 1.5})
        options.check_combination(BF16, CUDA, chosen, "fast")
        parsed = parser().parse_args(options.args(chosen))
        assert parsed.batches == batches and parsed.state_cache_gb == 1.5
    with pytest.raises(ValueError, match="CUDA graph"):
        options.check_combination(BF16, CPU, options.validate({"batches": "fine"}), "eager")
    with pytest.raises(ValueError, match="batches"):
        options.validate({"batches": "unknown"})


@pytest.mark.parametrize("payload,expected", [
    ({"status": "ok", "running": 0, "queued": 0}, {"reachable": True, "running": 0, "queued": 0}),
    ({"status": "ok", "running": 2, "queued": 3}, {"reachable": True, "running": 2, "queued": 3}),
    ({"status": "ok"}, {"reachable": True, "running": None, "queued": None}),
    ({"status": "ok", "running": True, "queued": -1}, {"reachable": True, "running": None, "queued": None}),
    ([], {"reachable": False, "running": None, "queued": None}),
])
def test_health_is_cached_and_missing_counters_are_not_idle(monkeypatch, payload, expected):
    calls = []
    def get(url, **kw):
        calls.append((url, kw))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    runner = Runner(internal_secret="private")
    runner.state, runner.base_url = "ready", "http://child"
    runner.status()
    wait_for(lambda: not runner.health_polling)
    assert runner.status()["health"] == expected
    assert len(calls) == 1 and calls[0][1] == {"headers": {"Authorization": "Bearer private"}, "timeout": .5}


def test_unreachable_and_old_generation_health_are_not_idle(monkeypatch):
    runner = Runner()
    runner.state, runner.base_url = "ready", "http://child"
    def fail(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", fail)
    runner._poll_health(runner.gen, runner.base_url)
    assert runner.health == {"reachable": False, "running": None, "queued": None}
    old = runner.gen
    runner.detach()
    runner._poll_health(old, "http://old")
    assert runner.health["reachable"] is None


@pytest.mark.parametrize("error,status,code", [
    (RequestError("bad input"), 422, "INVALID_REQUEST"),
    (RuntimeError("MPS backend out of memory"), 503, "OUT_OF_MEMORY"),
    (KeyError("layer"), 500, "INTERNAL_ERROR"),
    (ValueError("wrong shape in engine"), 500, "INTERNAL_ERROR"),
])
@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
def test_error_envelopes_ids_and_load_recovery(error, status, code, path):
    srv = AppServer(Engine(error))
    app = build_app(srv, parser().parse_args(["--mode", "fast"]))
    body = BODY if path == "/v1/systemone" else {"input": "s", "questions": [{"type": "predicate", "instructions": "ok?"}]}
    with TestClient(app) as client:
        response = client.post(path, json=body, headers={"x-request-id": "engine231"})
        assert response.status_code == status
        assert response.headers["x-request-id"] == response.headers["x-typesafe-request-id"] == "engine231"
        assert (response.json() if path == "/v1/systemone" else response.json()["error"])["code"] == code
        assert client.get("/health").json() == {"status": "ok", "running": 0, "queued": 0}


@pytest.mark.parametrize("status,code", [(422, "INVALID_REQUEST"), (503, "OUT_OF_MEMORY"), (500, "INTERNAL_ERROR")])
def test_gateway_and_playground_keep_status_and_diagnostics(monkeypatch, tmp_path, status, code):
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings, hardware_fn=lambda: CPU)
    runner = Runner()
    runner.state, runner.model = "ready", BF16.id
    fleet.runners[BF16.id] = runner
    def response(path):
        data = {"error": {"message": "diagnostic", "code": code, "type": "server_error", "param": None}} if path == "/v1/decisions" else {"error": "diagnostic", "code": code}
        return httpx.Response(status, json=data, headers={"x-request-id": "child-id"})
    async def adecide(self, client, body, path="/v1/systemone"):
        return response(path)
    monkeypatch.setattr(Runner, "adecide", adecide)
    monkeypatch.setattr(Runner, "decide", lambda self, body, path="/v1/systemone": response(path))
    with TestClient(gateway_app(fleet, fleet.stats), base_url="http://127.0.0.1") as client:
        for path in ("/v1/systemone", "/v1/decisions"):
            r = client.post(path, json={"model": BF16.id}, headers={"x-request-id": "public-id"})
            assert r.status_code == status and r.json() == response(path).json()
            assert r.headers["x-request-id"] == "public-id"
    app = create_app(fleet=fleet, hardware_fn=lambda: CPU, catalog_fn=lambda **_: ([BF16], "test"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        r = client.post("/api/decide", json={"model": BF16.id})
        assert r.status_code == status and r.json()["code"] == code
        assert r.headers["x-request-id"] == "child-id"


def test_batch_keeps_each_http_failure_without_retry(tmp_path, monkeypatch):
    statuses = [422, 503, 500]
    calls = []
    def handler(request):
        status = statuses[len(calls)]
        calls.append(request)
        return httpx.Response(status, json={"error": "engine diagnostic", "code": str(status)}, headers={"x-request-id": f"id-{status}"})
    real = httpx.AsyncClient
    monkeypatch.setattr(run.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text('\n'.join(json.dumps({"id": i, **BODY}) for i in range(3)))
    summary = asyncio.run(run.run(SimpleNamespace(input=source, output=output, concurrency=1, url="http://child", early_exit=None)))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["errors"] == 3 and len(calls) == 3
    assert [r["status"] for r in rows] == statuses
    assert [r["request_id"] for r in rows] == [f"id-{s}" for s in statuses]


def test_calibration_warnings_are_actual_report_text_and_jobs_recover(tmp_path):
    report = '  warning: choice: insufficient FIT labels\nwarning: CERT does not confirm target\n--strict: choice: 0 families tested; the family check did not run\nwarning: CERT does not confirm target\n'
    path = tmp_path / "cal.json"
    path.write_text('{"temperature_per_prim": {"choice": 1}}')
    path.with_suffix('.report.txt').write_text(report)
    result = calibration.summary(path)
    assert len(result["warnings"]) == 3 and result["thresholds"] == {}
    jobs = Jobs({})
    jobs.start("calibrate", "m", "warning run", lambda job: result)
    state = wait_done(jobs, "calibrate")
    assert state["phase"] == "done_with_warnings" and not state["running"] and state["error"] is None
    jobs.start("calibrate", "m", "next run", lambda job: {"warnings": []})
    assert wait_done(jobs, "calibrate")["phase"] == "done"


@pytest.mark.parametrize("hw,chosen", [
    (CPU, {"mode": "mps", "device": "cpu"}),
    (APPLE, {"mlx_packed": True}),
    (CUDA, {"batches": "fine", "state_cache_gb": 1.5}),
])
def test_new_options_survive_save_restart_and_auto_load(hw, chosen, tmp_path, monkeypatch):
    commands = []
    def start(self, assessment, gen, media=False, cache_dir=None, serve=None):
        commands.append(self._build_command(BF16, tmp_path, assessment, media, 1, settings.serve_settings()))
        with self.lock:
            self.state = "ready"
        settings.remember_loaded(BF16.id, media, self.options)
    monkeypatch.setattr(Runner, "_start", start)
    def app():
        return create_app(catalog_fn=lambda **_: ([BF16], "test"), hardware_fn=lambda: hw)
    with TestClient(app(), base_url="http://127.0.0.1") as client:
        assert client.put('/api/models/options', json={"id": BF16.id, "options": chosen}).status_code == 200
        assert client.post('/api/load', json={"id": BF16.id}).status_code == 202
        wait_for(lambda: len(commands) == 1)
        assert client.post('/api/restart', json={"id": BF16.id}).status_code == 202
        wait_for(lambda: len(commands) == 2)
        assert settings.last_models()[0]['options'] == chosen
        settings.set_auto_load(True)
    with TestClient(app(), base_url="http://127.0.0.1") as client:
        wait_for(lambda: len(commands) == 3)
        assert client.get('/api/status').json()['models'][0]['options'] == chosen
    for command in commands:
        parsed = parser().parse_args(command[3:])
        for key, value in chosen.items():
            assert getattr(parsed, key) == value


def test_mode_comparison_passes_and_records_packed_option(monkeypatch, tmp_path):
    from basal.ui import bench
    from basal.ui.jobs import Job, LogBuffer
    calls = []
    monkeypatch.setattr(bench.offline, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr(bench.offline, "load_questions", lambda *a: [BODY])
    monkeypatch.setattr(bench.offline, "model_format", lambda *a: 2)
    def measure(*args):
        calls.append(args)
        return {"mode": args[0]}, [0]
    monkeypatch.setattr(bench.offline, "measure_mode", measure)
    result = bench.modes_job(Job("benchmark", "m", "comparison", LogBuffer()),
                            SimpleNamespace(release=lambda _: None, stats=None), BF16, tmp_path,
                            ["mlx"], 1, 1, options.validate({"mlx_packed": True}))
    assert calls[0][-1] is True
    assert result["modes"][0]["mlx_packed"] is True


def test_failed_http_job_preserves_status_body_and_id():
    def fail(job):
        httpx.Response(503, text="GPU OOM diagnostic", request=httpx.Request("POST", "http://child"),
                       headers={"x-request-id": "job-id"}).raise_for_status()
    jobs = Jobs({})
    jobs.start("benchmark", "m", "failed request", fail)
    state = wait_done(jobs, "benchmark")
    assert state["phase"] == "error" and not state["running"]
    assert state["error"] == "HTTP 503: GPU OOM diagnostic [request job-id]"


def test_benchmark_retains_distinct_http_failures_without_retry(tmp_path):
    from basal.ui import bench
    from basal.ui.jobs import Job, LogBuffer
    from tests.test_ui_bench import AnswersOptionOne, write_set
    class FailingConcurrent(AnswersOptionOne):
        async def adecide(self, *args, **kwargs):
            response = await super().adecide(*args, **kwargs)
            index = len(self.sources) - bench.WARMUP - 2
            if index >= 0:
                return httpx.Response([422, 503, 500][index], text="retained diagnostic",
                                      request=response.request, headers={"x-request-id": "bench-id"})
            return response
    fleet = FailingConcurrent()
    job = Job('benchmark', 'm', 'errors', LogBuffer())
    result = asyncio.run(bench.run_benchmark(fleet, 'm', [str(write_set(tmp_path / 'q.jsonl', [0]))],
                                            'errors', job, n_seq=1, n_conc=3, concurrency=1))
    assert result['errors'] == 3 and result['error_statuses'] == {'422': 1, '503': 1, '500': 1}
    assert len(fleet.sources) == bench.WARMUP + 4
    assert 'retained diagnostic' in job.log.text() and 'bench-id' in job.log.text()
