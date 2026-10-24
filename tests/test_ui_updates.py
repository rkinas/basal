import httpx
import pytest
from fastapi.testclient import TestClient

from basal.ui import catalog, settings, updates
from basal.ui.app import create_app
from basal.ui.hardware import Hardware

HW = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())


def app(tmp_path):
    return create_app(catalog_fn=lambda **_: ([], "test"), hardware_fn=lambda: HW)


def test_update_newer_older_errors_and_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    monkeypatch.setattr(updates, "_cached", (0.0, None, None))
    replies = iter(("9.0.0", "1.0.0"))

    class Reply:
        def __init__(self, version):
            self.version = version

        @staticmethod
        def raise_for_status():
            pass

        def json(self):
            return {"tag_name": self.version}

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Reply(next(replies)))
    with TestClient(app(tmp_path), base_url="http://127.0.0.1") as client:
        assert client.get("/api/updates").json()["newer"] is True
        monkeypatch.setattr(updates, "_cached", (0.0, None, None))
        assert client.get("/api/updates").json()["newer"] is False
        monkeypatch.setattr(updates, "_cached", (0.0, None, None))
        monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.HTTPError("offline")))
        assert client.get("/api/updates").json()["error"] == "offline"
        client.put("/api/settings", json={"check_updates": False})
        monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled update check made a request")))
        disabled = client.get("/api/updates").json()
        assert disabled["enabled"] is False and disabled["latest"] is None and disabled["error"] is None


def test_a_failed_check_is_retried_after_ten_minutes_not_six_hours(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(updates.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(updates, "_cached", (0.0, None, None))
    replies = iter((httpx.HTTPError("offline"), "9.0.0"))

    class Reply:
        def __init__(self, version):
            self.version = version

        @staticmethod
        def raise_for_status():
            pass

        def json(self):
            return {"tag_name": self.version}

    def get(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return Reply(reply)
    monkeypatch.setattr(httpx, "get", get)
    assert updates.latest() == (None, "offline")
    clock[0] += 9 * 60
    assert updates.latest() == (None, "offline")  # cached
    clock[0] += 2 * 60
    assert updates.latest() == ("9.0.0", None)
    clock[0] += 60 * 60
    assert updates.latest() == ("9.0.0", None)  # a success stays cached for hours


@pytest.mark.parametrize("source", ["cache", "hub"])
def test_only_the_hub_catalog_seeds_and_reports_new_models(tmp_path, monkeypatch, source):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    settings.set_check_updates(False)
    variants = [catalog.Variant("a", "r/a", "basal-2.0", "mini", "bf16", "bf16", None, 1, False)]
    catalogs = [(variants[:0], source), (variants, source)]
    app = create_app(catalog_fn=lambda **_: catalogs[0], hardware_fn=lambda: HW)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = client.get("/api/updates").json()
        assert first["new_models"] == [] and first["catalog"] == source
        catalogs.pop(0)
        later = client.get("/api/updates").json()
    if source == "hub":  # the first hub catalog is the starting point; a model added later is new
        assert settings.load()["seen_models"] == [] and later["new_models"] == ["a"]
    else:  # a cache-only catalog is partial: it neither seeds nor reports
        assert "seen_models" not in settings.load() and later["new_models"] == []
