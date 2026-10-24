"""Persistent basal-ui settings: the models directory, the API server and its keys, the loaded models, the memory budget.

Every change is a read-modify-write of one JSON file under a thread and inter-process lock, written atomically (a private
temp file renamed over the old one), so concurrent requests and processes never lose each other's changes. The file and
the basal-serve key file hold SHA-256 digests of API keys only and are readable by the owner alone (0600)."""
import functools
import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

DEFAULT_SERVE = {"enabled": False, "host": "127.0.0.1", "port": 8000, "require_key": False,
                 "max_len": 4096, "cors_origins": [], "log_decisions": "", "mcp": False}
KEY_NAME = re.compile(r"[\w][\w .-]{0,63}\Z")
_internal_sha256 = None
_LOCK = threading.RLock()
_LOCK_STATE = threading.local()

SETTINGS_PATH = Path(os.environ.get("BASAL_UI_CONFIG", "~/.config/basal-ui/settings.json")).expanduser()


def _settings_path() -> Path:
    return Path(os.environ.get("BASAL_UI_CONFIG", SETTINGS_PATH)).expanduser()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            depth = getattr(_LOCK_STATE, "depth", 0)
            if depth:
                _LOCK_STATE.depth = depth + 1
                try:
                    return fn(*args, **kwargs)
                finally:
                    _LOCK_STATE.depth -= 1
            path = _settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.name == "posix":
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                elif os.name == "nt":
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\0")
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                _LOCK_STATE.depth = 1
                try:
                    return fn(*args, **kwargs)
                finally:
                    _LOCK_STATE.depth = 0
                    if os.name == "posix":
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    elif os.name == "nt":
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
    return wrapper


def check_updates() -> bool:
    return load().get("check_updates", True) is not False


@_locked
def set_check_updates(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("check_updates must be a boolean")
    data = load()
    data["check_updates"] = enabled
    save(data)
    return enabled


def presets() -> dict:
    value = load().get("presets")
    value = value if isinstance(value, dict) else {}
    return {kind: value.get(kind) if isinstance(value.get(kind), dict) else {}
            for kind in ("load", "request")}


@_locked
def set_preset(kind: str, name: str, value) -> None:
    if kind not in ("load", "request"):
        raise ValueError("kind must be load or request")
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or name[0] in ". ":
        raise ValueError("name must be 1..64 characters and not start with dot or space")
    data = load()
    saved = data.get("presets") if isinstance(data.get("presets"), dict) else {}
    entries = saved.get(kind) if isinstance(saved.get(kind), dict) else {}
    if value is None:
        entries.pop(name, None)
    else:
        entries[name] = value
    saved[kind] = entries
    data["presets"] = saved
    save(data)


@_locked
def mark_models_seen(model_ids) -> None:
    data = load()
    seen = data.get("seen_models", [])
    data["seen_models"] = sorted((set(seen) if isinstance(seen, list) else set()) | set(model_ids))
    save(data)


def seen_models() -> set[str]:
    value = load().get("seen_models", [])
    return set(x for x in value if isinstance(x, str)) if isinstance(value, list) else set()


def update_seen_ids(model_ids) -> list[str]:
    """The ids not seen before; the first check (no seen list yet) marks the whole catalog seen and reports none."""
    model_ids = set(model_ids)
    if not isinstance(load().get("seen_models"), list):
        mark_models_seen(model_ids)
        return []
    return sorted(model_ids - seen_models())


def _write_private(path: Path, text: str) -> None:
    """Replace path with text atomically; the file is created 0600, never briefly readable by others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load() -> dict:
    try:
        value = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    legacy = value.pop("last_model", None)  # one model before basal-ui loaded several
    if isinstance(legacy, dict) and "last_models" not in value:
        value["last_models"] = [legacy]
    return value


@_locked
def save(settings: dict) -> None:
    _write_private(_settings_path(), json.dumps(settings, indent=2))


def models_dir() -> Path:
    configured = load().get("models_dir")
    return Path(configured).expanduser() if configured else Path(HF_HUB_CACHE)


@_locked
def set_models_dir(path: str | None) -> Path:
    if path is None:
        value = load()
        value.pop("models_dir", None)
        save(value)
        return Path(HF_HUB_CACHE)
    directory = Path(path).expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"{directory} is not writable") from exc
    if not os.access(directory, os.W_OK):
        raise ValueError(f"{directory} is not writable")
    value = load()
    value["models_dir"] = str(directory)
    save(value)
    return directory


def serve_settings() -> dict:
    value = load().get("serve", {})
    return {**DEFAULT_SERVE, **(value if isinstance(value, dict) else {})}


@_locked
def update_serve(patch: dict) -> dict:
    """Validate a partial update of the API server settings and save it; ValueError names the offending field."""
    allowed = set(DEFAULT_SERVE)
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"unknown serve setting: {min(unknown)}")
    value = {**serve_settings(), **patch}
    for name in ("enabled", "require_key", "mcp"):
        if not isinstance(value[name], bool):
            raise ValueError(f"{name} must be a boolean")  # noqa: TRY004 — setting validation uses ValueError
    if value["host"] not in ("127.0.0.1", "0.0.0.0"):
        raise ValueError("host must be 127.0.0.1 or 0.0.0.0")
    if type(value["port"]) is not int or not 1024 <= value["port"] <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    if type(value["max_len"]) is not int or not 512 <= value["max_len"] <= 131072:
        raise ValueError("max_len must be between 512 and 131072")
    origins = value["cors_origins"]
    if not isinstance(origins, list) or any(not isinstance(origin, str) or
            (origin != "*" and not re.fullmatch(r"https?://[^/\s]+", origin)) for origin in origins):
        raise ValueError("cors_origins must contain * or origins like https://example.com")
    log_path = value["log_decisions"]
    if not isinstance(log_path, str):
        raise ValueError("log_decisions must be a path or empty")  # noqa: TRY004 — setting validation uses ValueError
    if log_path:
        path = Path(log_path).expanduser().resolve()
        if path.is_dir():
            raise ValueError("decision log path is a directory")
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            raise ValueError(f"decision log directory is not writable: {path.parent}")
        value["log_decisions"] = str(path)
    if value["mcp"]:
        from basal.mcp_server import available
        message = available()
        if message:
            raise ValueError(message)
    data = load()
    data["serve"] = value
    save(data)
    return value


@_locked
def set_auto_load(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("auto_load must be a boolean")  # noqa: TRY004 — setting validation uses ValueError
    data = load()
    data["auto_load"] = enabled
    save(data)
    return enabled


@_locked
def remember_loaded(model_id: str, media: bool, options: dict | None = None) -> None:
    """A model that reached ready, for auto-load at the next start (load order kept), with the load options it was
    started with (only the ones that differ from basal-serve's defaults)."""
    data = load()
    entries = last_models(data)
    entry = {"id": model_id, "media": bool(media), **({"options": options} if options else {})}
    if any(e["id"] == model_id for e in entries):
        entries = [entry if e["id"] == model_id else e for e in entries]
    else:
        entries.append(entry)
    data["last_models"] = entries
    save(data)


def load_options(model_id: str) -> dict:
    """The load options remembered for a model ({} = basal-serve's defaults)."""
    value = (load().get("load_options") or {}).get(model_id)
    return value if isinstance(value, dict) else {}


@_locked
def set_load_options(model_id: str, options: dict | None) -> None:
    data = load()
    saved = data.get("load_options") if isinstance(data.get("load_options"), dict) else {}
    if options:
        saved[model_id] = options
    else:
        saved.pop(model_id, None)
    data["load_options"] = saved
    save(data)


@_locked
def forget_loaded(model_id: str) -> None:
    data = load()
    data["last_models"] = [e for e in last_models(data) if e["id"] != model_id]
    save(data)


def last_models(data: dict | None = None) -> list[dict]:
    value = (load() if data is None else data).get("last_models")
    return [e for e in value if isinstance(e, dict) and isinstance(e.get("id"), str)] if isinstance(value, list) else []


DEFAULT_RESERVE = 2 * 1024**3


def memory_settings() -> dict:
    value = load().get("memory")
    value = value if isinstance(value, dict) else {}
    budget, reserve = value.get("budget_bytes"), value.get("reserve_bytes", DEFAULT_RESERVE)
    return {"budget_bytes": budget if type(budget) is int else None,
            "reserve_bytes": reserve if type(reserve) is int and reserve >= 0 else DEFAULT_RESERVE}


@_locked
def update_memory(patch: dict) -> dict:
    """budget_bytes: null (the hardware budget) or at least 1 GiB; reserve_bytes: kept free, >= 0."""
    unknown = set(patch) - {"budget_bytes", "reserve_bytes"}
    if unknown:
        raise ValueError(f"unknown memory setting: {min(unknown)}")
    value = {**memory_settings(), **patch}
    if value["budget_bytes"] is not None and (type(value["budget_bytes"]) is not int
                                              or value["budget_bytes"] < 1024**3):
        raise ValueError("budget_bytes must be null or at least 1 GiB")
    if type(value["reserve_bytes"]) is not int or value["reserve_bytes"] < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    data = load()
    data["memory"] = value
    save(data)
    return value


def _sibling(suffix: str) -> Path:
    path = _settings_path()
    return path.with_name(f"{path.stem}.{suffix}")


def stats_path() -> Path:
    return _sibling("stats.json")


def history_path() -> Path:
    return _sibling("history.jsonl")


def calibrations_dir(model_id: str) -> Path:
    """Where basal-ui writes the calibration files it fits for a model: <settings dir>/calibrations/<org>--<name>/."""
    return _settings_path().parent / "calibrations" / model_id.replace("/", "--")


def calibration_selection() -> dict:
    """{model id: calibration file basal-serve is started with}; a model without an entry uses its shipped file."""
    value = load().get("calibration")
    return {k: v for k, v in value.items() if isinstance(v, str)} if isinstance(value, dict) else {}


@_locked
def select_calibration(model_id: str, path: str | None) -> None:
    """Use `path` for the model from its next start; None returns it to the shipped calibration."""
    data = load()
    selection = calibration_selection() | ({model_id: path} if path else {})
    if not path:
        selection.pop(model_id, None)
    data["calibration"] = selection
    save(data)


def keys() -> list[dict]:
    value = load().get("api_keys", [])
    return value if isinstance(value, list) else []


@_locked
def create_key(name: str) -> tuple[dict, str]:
    """A new API key -> (its public record, the secret). Only the secret's SHA-256 is stored: the secret is returned
    this once."""
    if not isinstance(name, str) or not KEY_NAME.fullmatch(name):
        raise ValueError("name must be 1–64 letters, digits, spaces, dots, underscores or hyphens")
    if name == "basal-ui":
        raise ValueError("basal-ui is reserved for the app's own key")
    data = load()
    records = data.get("api_keys", [])
    if any(record.get("name") == name for record in records):
        raise ValueError(f"API key {name!r} already exists")
    secret = "bsl_" + secrets.token_urlsafe(32)
    record = {"name": name, "sha256": hashlib.sha256(secret.encode()).hexdigest(),
              "prefix": secret[:8], "created": int(time.time())}
    data["api_keys"] = [*records, record]
    save(data)
    _sync_key_files(data)
    return {k: v for k, v in record.items() if k != "sha256"}, secret


@_locked
def delete_key(name: str) -> None:
    """Revoke a key: a running basal-serve re-reads the key file on its next request."""
    data = load()
    records = data.get("api_keys", [])
    remaining = [record for record in records if record.get("name") != name]
    if len(remaining) == len(records):
        raise ValueError(f"unknown API key {name!r}")
    data["api_keys"] = remaining
    save(data)
    _sync_key_files(data)


# Each basal-ui process writes two key files next to the settings file. <stem>.keys-<pid>.json holds the user's keys
# (shared, from the settings) plus that process's internal admin key: the API server reads it. <stem>.internal-<pid>.json
# holds the internal key alone: each basal-serve child reads it, so only basal-ui reaches a child. Two basal-ui
# instances on one settings file therefore never overwrite each other's internal key, and key changes reach each file.
def _keys_path(pid: int | None = None) -> Path:
    settings_path = _settings_path()
    return settings_path.with_name(f"{settings_path.stem}.keys-{os.getpid() if pid is None else pid}.json")


def _internal_keys_path() -> Path:
    settings_path = _settings_path()
    return settings_path.with_name(f"{settings_path.stem}.internal-{os.getpid()}.json")


def _user_records(data: dict) -> list[dict]:
    return [{"name": record["name"], "sha256": record["sha256"], "admin": False}
            for record in data.get("api_keys", [])]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sync_key_files(data: dict) -> None:
    """Rewrite every live instance's key file with the current user keys; drop the files of dead processes."""
    settings_path = _settings_path()
    for path in settings_path.parent.glob(f"{settings_path.stem}.keys-*.json"):
        try:
            pid = int(path.stem.removeprefix(f"{settings_path.stem}.keys-"))
        except ValueError:
            continue
        if not _alive(pid):
            path.unlink(missing_ok=True)
            continue
        if pid == os.getpid():
            internal = [{"name": "basal-ui", "sha256": _internal_sha256, "admin": True}] if _internal_sha256 else []
        else:
            try:
                internal = [r for r in json.loads(path.read_text(encoding="utf-8"))["keys"] if r.get("admin")]
            except (OSError, ValueError, KeyError, TypeError):
                internal = []
        _write_private(path, json.dumps({"keys": _user_records(data) + internal}, indent=2))


@_locked
def keys_file(internal_sha256: str) -> Path:
    """This process's API server key file (format: basal/auth.py): the user's keys and this app's own admin key."""
    global _internal_sha256
    _internal_sha256 = internal_sha256
    path = _keys_path()
    records = _user_records(load()) + [{"name": "basal-ui", "sha256": internal_sha256, "admin": True}]
    _write_private(path, json.dumps({"keys": records}, indent=2))
    return path


def internal_keys_file(internal_sha256: str) -> Path:
    """The key file of a basal-serve child: this app's admin key only, so user keys never reach a child directly."""
    path = _internal_keys_path()
    _write_private(path, json.dumps({"keys": [{"name": "basal-ui", "sha256": internal_sha256, "admin": True}]}))
    return path


@_locked
def remove_keys_file() -> None:
    """Called when basal-ui exits: its API server and children are gone, so are its key files."""
    _keys_path().unlink(missing_ok=True)
    _internal_keys_path().unlink(missing_ok=True)
