import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from basal.ui import settings


def test_interprocess_lock_context_runs_the_wrapped_function(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    called = []
    locked = settings._locked(lambda: called.append("inside"))
    locked()
    assert called == ["inside"]

def test_set_and_reset_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    target = tmp_path / "models"

    assert settings.set_models_dir(str(target)) == target
    assert target.is_dir()
    assert settings.models_dir() == target
    assert settings.set_models_dir(None) == Path(HF_HUB_CACHE)
    assert settings.models_dir() == Path(HF_HUB_CACHE)


def test_set_models_dir_rejects_unwritable_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    target = tmp_path / "readonly"
    target.mkdir()
    target.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="is not writable"):
            settings.set_models_dir(str(target))
    finally:
        target.chmod(0o700)


def test_invalid_json_shape_uses_default_and_relative_path_is_absolute(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setenv("BASAL_UI_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)
    config.write_text("[]")
    assert settings.models_dir() == Path(HF_HUB_CACHE)
    stored = settings.set_models_dir("relative-models-dir")
    assert stored == (tmp_path / "relative-models-dir").resolve()
    assert settings.load()["models_dir"] == str(stored)


def test_serve_settings_validation_and_api_key_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.serve_settings()["enabled"] is False
    for patch, message in (({"port": 80}, "port"), ({"host": "example.com"}, "host"),
                           ({"max_len": 256}, "max_len"), ({"cors_origins": ["not-an-origin"]}, "cors_origins")):
        with pytest.raises(ValueError, match=message):
            settings.update_serve(patch)
    settings.update_serve({"enabled": True, "cors_origins": ["https://example.com"]})
    record, secret = settings.create_key("laptop")
    assert secret.startswith("bsl_") and record["prefix"] == secret[:8]
    path = settings.keys_file("a" * 64)
    content = __import__("json").loads(path.read_text())
    assert content["keys"] == [
        {"name": "laptop", "sha256": __import__("hashlib").sha256(secret.encode()).hexdigest(), "admin": False},
        {"name": "basal-ui", "sha256": "a" * 64, "admin": True},
    ]
    assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "settings.json").stat().st_mode & 0o777 == 0o600
    settings.delete_key("laptop")
    assert settings.keys() == []


def test_serve_settings_reject_invalid_fields_and_normalize_log_path(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    invalid = (
        ({"enabled": "yes"}, "enabled"),
        ({"require_key": 1}, "require_key"),
        ({"mcp": 0}, "mcp"),
        ({"host": "10.0.0.1"}, "host"),
        ({"port": 80}, "port"),
        ({"port": 70000}, "port"),
        ({"max_len": 100}, "max_len"),
        ({"max_len": 131073}, "max_len"),
        ({"cors_origins": ["not-an-origin"]}, "cors_origins"),
        ({"cors_origins": ["https://example.com/path"]}, "cors_origins"),
    )
    for patch, field in invalid:
        with pytest.raises(ValueError, match=field):
            settings.update_serve(patch)
    with pytest.raises(ValueError, match="decision log path is a directory"):
        settings.update_serve({"log_decisions": str(tmp_path)})
    with pytest.raises(ValueError, match="decision log directory"):
        settings.update_serve({"log_decisions": str(tmp_path / "missing" / "decisions.jsonl")})
    log_path = tmp_path / "decisions.jsonl"
    settings.update_serve({"log_decisions": str(log_path)})
    assert settings.serve_settings()["log_decisions"] == str(log_path.resolve())


def test_mcp_setting_requires_available_extra(tmp_path, monkeypatch):
    from basal import mcp_server

    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    monkeypatch.setattr(mcp_server, "available", lambda: "install basal[mcp]")
    with pytest.raises(ValueError, match="install basal"):
        settings.update_serve({"mcp": True})


def test_api_key_names_are_unique_validated_and_revocable(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    for name in ("", "a" * 65, ".", "..", " x", ".hidden"):
        with pytest.raises(ValueError, match="name must"):
            settings.create_key(name)
    record, _ = settings.create_key("a.b")
    assert record["name"] == "a.b"
    with pytest.raises(ValueError, match="already exists"):
        settings.create_key("a.b")
    settings.delete_key("a.b")
    record, _secret = settings.create_key("laptop")
    with pytest.raises(ValueError, match="already exists"):
        settings.create_key("laptop")
    assert settings.keys() == [{**record, "sha256": settings.load()["api_keys"][0]["sha256"]}]
    own = settings.keys_file("a" * 64)
    assert own.name == f"settings.keys-{os.getpid()}.json"
    first = own.read_text()
    settings.delete_key("laptop")
    second = own.read_text()
    assert "laptop" in first and "laptop" not in second
    with pytest.raises(ValueError, match="unknown API key"):
        settings.delete_key("laptop")


def test_key_changes_reach_every_live_instance_and_drop_dead_ones(tmp_path, monkeypatch):
    """Two basal-ui processes share one settings file: each keeps its own internal key, both see revocations."""
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    other, dead = os.getppid(), 2**22 + 12345  # a live process and a pid that is not running
    other_file, dead_file = (tmp_path / f"settings.keys-{other}.json",
                             tmp_path / f"settings.keys-{dead}.json")
    unrelated_file = tmp_path / "other.keys-999999.json"
    unrelated_file.write_text(json.dumps({"keys": []}))
    other_internal = {"name": "basal-ui", "sha256": "b" * 64, "admin": True}
    other_file.write_text(json.dumps({"keys": [other_internal]}))
    settings.keys_file("a" * 64)
    _record, _secret = settings.create_key("laptop")
    other_keys = json.loads(other_file.read_text())["keys"]
    assert [k["name"] for k in other_keys] == ["laptop", "basal-ui"] and other_internal in other_keys
    assert not dead_file.exists()
    settings.delete_key("laptop")
    assert json.loads(other_file.read_text())["keys"] == [other_internal]
    own = json.loads((tmp_path / f"settings.keys-{os.getpid()}.json").read_text())["keys"]
    assert own == [{"name": "basal-ui", "sha256": "a" * 64, "admin": True}]
    settings.remove_keys_file()
    assert not (tmp_path / f"settings.keys-{os.getpid()}.json").exists() and other_file.exists()
    assert not dead_file.exists() and unrelated_file.exists()


def test_auto_load_setting_requires_boolean(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.set_auto_load(True) is True
    assert settings.load()["auto_load"] is True
    with pytest.raises(ValueError, match="auto_load must be a boolean"):
        settings.set_auto_load("yes")


def test_loaded_models_keep_load_order_update_media_and_migrate_the_single_model(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setenv("BASAL_UI_CONFIG", str(config))
    config.write_text(json.dumps({"last_model": {"id": "old", "media": True}, "auto_load": True}))
    assert settings.last_models() == [{"id": "old", "media": True}]
    settings.remember_loaded("new", False)
    settings.remember_loaded("old", False)
    assert settings.last_models() == [{"id": "old", "media": False}, {"id": "new", "media": False}]
    assert "last_model" not in json.loads(config.read_text())
    settings.forget_loaded("old")
    settings.forget_loaded("never-loaded")
    assert settings.last_models() == [{"id": "new", "media": False}]


def test_memory_settings_validate_budget_and_reserve(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.memory_settings() == {"budget_bytes": None, "reserve_bytes": 2 * 1024**3}
    for patch in ({"budget_bytes": 1024**3 - 1}, {"budget_bytes": "8G"}, {"reserve_bytes": -1},
                  {"reserve_bytes": 1.5}, {"swap": 1}):
        with pytest.raises(ValueError):
            settings.update_memory(patch)
    assert settings.update_memory({"budget_bytes": 1024**3, "reserve_bytes": 0}) == {
        "budget_bytes": 1024**3, "reserve_bytes": 0}
    assert settings.update_memory({"budget_bytes": None})["budget_bytes"] is None
    assert settings.memory_settings() == {"budget_bytes": None, "reserve_bytes": 0}


def test_cross_process_settings_updates_do_not_lose_keys(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setenv("BASAL_UI_CONFIG", str(config))
    env = {**os.environ, "BASAL_UI_CONFIG": str(config)}
    code = ("from basal.ui import settings\n"
            "import sys\n"
            "for i in range(20): settings.create_key(f'{sys.argv[1]}-{i}')")
    workers = [subprocess.Popen([sys.executable, "-c", code, prefix], env=env)
               for prefix in ("first", "second")]
    assert [worker.wait(timeout=20) for worker in workers] == [0, 0]
    assert len(settings.keys()) == 40


def test_cross_process_revoke_is_not_undone_by_creates(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    monkeypatch.setenv("BASAL_UI_CONFIG", str(config))
    env = {**os.environ, "BASAL_UI_CONFIG": str(config)}
    settings.create_key("revoke-me")
    start = tmp_path / "start"
    code = ("from basal.ui import settings\n"
            "import pathlib, sys, time\n"
            "start = pathlib.Path(sys.argv[1])\n"
            "while not start.exists(): time.sleep(.001)\n"
            "for i in range(100): settings.create_key(f'loop-{i}')\n")
    creator = subprocess.Popen([sys.executable, "-c", code, str(start)], env=env)
    revoker = subprocess.Popen(
        [sys.executable, "-c",
         ("from basal.ui import settings\n"
          "import pathlib, sys, time\n"
          "start = pathlib.Path(sys.argv[1])\n"
          "while not start.exists(): time.sleep(.001)\n"
          "settings.delete_key('revoke-me')"), str(start)], env=env)
    start.touch()
    assert creator.wait(timeout=30) == 0
    assert revoker.wait(timeout=30) == 0
    assert all(record["name"] != "revoke-me" for record in settings.keys())  # noqa: SIM118 — keys() returns records


def test_presets_and_update_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.presets() == {"load": {}, "request": {}}
    settings.set_preset("load", "fast", {"mode": "eager"})
    settings.set_preset("request", "demo", {"questions": {"q": {"type": "choice"}}})
    assert settings.presets() == {"load": {"fast": {"mode": "eager"}},
                                  "request": {"demo": {"questions": {"q": {"type": "choice"}}}}}
    settings.set_preset("load", "fast", None)
    assert settings.presets()["load"] == {}
    assert settings.check_updates() is True
    settings.set_check_updates(False)
    assert settings.check_updates() is False


def test_the_first_catalog_check_reports_no_new_models_then_only_new_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))
    assert settings.update_seen_ids(["a", "b"]) == []  # a fresh install: everything is the starting catalog
    assert settings.update_seen_ids(["a", "b", "c"]) == ["c"]
    settings.mark_models_seen(["c"])
    assert settings.update_seen_ids(["a", "b", "c"]) == []
