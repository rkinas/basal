import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from basal.ui import catalog, inspect, settings
from basal.ui.app import Fleet, create_app
from basal.ui.catalog import Variant
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())
BF16 = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
MLX = Variant("mini-mlx", "Remek/basal-2.0-mini-MLX-8bit", "basal-2.0", "mini", "MLX-8bit", "MLX 8-bit", None, 100,
              False)


class StubExplainer:
    def __init__(self, a):
        self.a = a
        self.info = {"model": "stub", "device": a.device, "layers": 2, "mode": "eager", "orders": 2,
                     "temperatures": {"choice": 1.1}, "temperature_per_bucket": {}}
        workers = self.workers = []

        class Srv:
            async def worker(self):
                workers.append("running")
                await asyncio.sleep(3600)

            async def decide(self, body):
                return {"answers": {}}

        self.srv = Srv()

    def explain(self, body):
        return {"questions": [], "state": body["state"]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    monkeypatch.setattr(inspect, "Explainer", StubExplainer)
    monkeypatch.setattr(catalog, "is_downloaded", lambda variant, cache_dir, revision=None: tmp_path)
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                  hardware_fn=lambda: HW, stats=Stats(tmp_path / "s.json"), history=History(tmp_path / "h.jsonl"))
    app = create_app(catalog_fn=lambda **_: ([BF16, MLX], "test"), hardware_fn=lambda: HW, fleet=fleet,
                     stats=fleet.stats, history=fleet.history)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.fleet = fleet
        yield c


def test_inspector_serves_basal_viz_for_a_bf16_model_and_stops(client):
    assert client.get("/viz/api/info").status_code == 503
    assert client.get("/viz/").status_code == 200  # the page itself
    refused = client.post("/api/inspect", json={"id": "mini-mlx"})
    assert refused.status_code == 400 and "bf16 PyTorch weights" in refused.json()["detail"]
    assert client.post("/api/inspect", json={"id": "mini"}).status_code == 202
    deadline = time.monotonic() + 5
    while client.get("/api/inspect").json()["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.get("/api/inspect").json() == {
        "model": "mini", "state": "ready", "error": None, "mode": "eager", "device": "cpu", "orders": 2,
        "calibration": None, "temperatures": {"choice": 1.1}, "temperature_per_bucket": {}}
    assert client.get("/viz/api/info").json()["model"] == "mini"  # the variant's id
    explained = client.post("/viz/api/explain", json={"state": "s", "questions": {}})
    assert explained.json() == {"questions": [], "state": "s", "served": {}}
    assert client.fleet.reserved["inspect"] == catalog.assess(BF16, HW).est_memory_bytes
    assert client.delete("/api/inspect").json()["state"] == "idle"
    assert client.get("/viz/api/info").status_code == 503 and client.fleet.reserved == {}


def test_stopping_during_explainer_construction_never_publishes_ready(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    created = []

    class SlowExplainer(StubExplainer):
        def __init__(self, args):
            entered.set()
            release.wait(5)
            created.append(self)
            super().__init__(args)

    monkeypatch.setattr(inspect, "Explainer", SlowExplainer)
    assert client.post("/api/inspect", json={"id": "mini"}).status_code == 202
    assert entered.wait(3)
    stopped = client.delete("/api/inspect")
    assert stopped.json()["state"] == "idle"
    release.set()
    deadline = time.monotonic() + 5
    while client.get("/api/status").json()["jobs"]["inspect"]["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.get("/api/inspect").json()["state"] == "idle"
    assert client.fleet.reserved == {}
    assert created
