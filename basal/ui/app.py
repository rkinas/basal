"""basal-ui: a local web app that detects the hardware, suggests basal models (size and format), downloads them,
starts basal-serve for each and runs decisions in a playground.

  basal-ui --port 7860      then open http://127.0.0.1:7860

Every loaded model runs in its own basal-serve subprocess on a free loopback port that only basal-ui knows (internal
API key); /api/decide and the API server (basal/ui/gateway.py, one port for every model) forward to it, so a crashed
or unloaded model never takes the UI down.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import catalog, hardware, settings
from . import options as load_options
from .gateway import Gateway
from .history import History, make_record
from .jobs import Jobs, LogBuffer
from .stats import Stats

STATIC = Path(__file__).parent / "static"
FAMILY_ORDER = {f"basal-{n}": i for i, n in enumerate(("2.0", "omni-1.1", "omni-1.0", "omni", "1.5", "1.0"))}
START_TIMEOUT_S = 1800  # --mode fast compiles for 4-17 minutes on the first start (README, Serving modes)
MODEL_LOGS_KEPT = 10  # log buffers of unloaded models kept for the Logs view
RSS_INTERVAL_S = 2  # `ps` runs at most this often for the resident memory of the children


def _lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostname()


def _fmt(n):
    return f"{n / 1024**3:.1f} GiB"


def rss_by_pgid() -> dict[int, int]:
    """Resident memory (bytes) per process group: a basal-serve child leads its own group (start_new_session), which
    also holds its engine's workers. Empty where ps is unavailable (Windows)."""
    if os.name != "posix":
        return {}
    try:
        out = subprocess.run(["ps", "-eo", "pgid=,rss="], capture_output=True, text=True, timeout=5,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    totals: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            totals[int(parts[0])] = totals.get(int(parts[0]), 0) + int(parts[1]) * 1024
    return totals


class Downloads:
    """Model downloads by repository, shared by the runners and the Download button. A download outlives the load that
    started it (an unload leaves it finishing into the cache); a failed one stays reported for ERROR_KEEP_S."""
    ERROR_KEEP_S = 60

    def __init__(self, event=None):
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        self.event = event or (lambda level, message: None)

    def start(self, variant, cache_dir, revision=None) -> dict:
        with self.lock:
            record = self.records.get(variant.repo)
            if record and record["thread"].is_alive():
                return record
            record = {"bar": _byte_counter(), "total": variant.weights_bytes or 0, "path": None, "error": None,
                      "ended": None, "revision": revision}
            record["thread"] = threading.Thread(target=self._run, args=(record, variant, cache_dir), daemon=True)
            self.records[variant.repo] = record
            record["thread"].start()
        self.event("info", f"download started: {variant.repo}")
        return record

    def _run(self, record, variant, cache_dir):
        try:
            record["path"] = catalog.download(variant, cache_dir, tqdm_class=record["bar"],
                                              revision=record["revision"])
            self.event("info", f"download finished: {variant.repo}")
        except Exception as exc:  # reported in the UI and re-raised by a waiting load
            record["error"] = exc
            self.event("error", f"download failed: {variant.repo}: {exc}")
        finally:
            record["ended"] = time.monotonic()

    def alive(self, repo) -> bool:
        with self.lock:
            record = self.records.get(repo)
            return bool(record and record["thread"].is_alive())

    def any_alive(self) -> bool:
        with self.lock:
            return any(record["thread"].is_alive() for record in self.records.values())

    def status(self) -> dict:
        now, out = time.monotonic(), {}
        with self.lock:
            for repo, record in list(self.records.items()):
                failed = record["error"] is not None and now - record["ended"] < self.ERROR_KEEP_S
                if record["thread"].is_alive() or failed:
                    done = record["bar"].done
                    out[repo] = {"done_bytes": done, "total_bytes": max(record["total"], done),
                                 "error": str(record["error"]) if record["error"] is not None else None}
                else:
                    del self.records[repo]
        return out


def _child_settings(serve, calibration=None):
    """The settings a basal-serve child is started with (the rest belongs to the API server): the decision log, the
    vLLM / SGLang context and the calibration file selected for the model (None: the shipped one)."""
    return {"log_decisions": serve["log_decisions"], "max_len": serve["max_len"], "calibration": calibration}


class Runner:
    """One model: idle -> downloading -> starting -> ready, or error. Every load or unload starts a new generation; a
    background load that finds its generation outdated stops instead of touching the state."""

    def __init__(self, models_dir_fn=settings.models_dir, platform_fn=lambda: hardware.detect().platform,
                 settings_mod=settings, downloads=None, internal_secret=None, log=None, event=None):
        self.models_dir_fn = models_dir_fn
        self.platform_fn = platform_fn
        self.settings_mod = settings_mod
        self.downloads = downloads if downloads is not None else Downloads()
        self.internal_secret = internal_secret or secrets.token_urlsafe(32)
        self.log = log if log is not None else LogBuffer()
        self.event = event or (lambda level, message: None)
        self.lock = threading.RLock()
        self.gen = 0
        self.process = None
        self.rss_bytes = None
        self._reset()

    def _reset(self):
        self.state, self.model, self.model_repo, self.mode = "idle", None, None, None
        self.command, self.progress, self.error, self.base_url = None, None, None, None
        self.media_requested, self.media = False, []
        self.est_memory_bytes, self.models_entry, self.ready_at = None, None, None
        self.serve_snapshot = None  # the child settings the running basal-serve was started with
        self.extra_args, self.temporary, self.options = (), False, {}
        self.health = {"reachable": None, "running": None, "queued": None}
        self.health_checked = 0.0
        self.health_polling = False

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def _poll_health(self, gen, url):
        health = {"reachable": False, "running": None, "queued": None}
        try:
            response = httpx.get(url + "/health", headers=self._headers(), timeout=0.5)
            response.raise_for_status()
            data = response.json()
            health["reachable"] = isinstance(data, dict) and data.get("status") == "ok"
            if health["reachable"]:
                for key in ("running", "queued"):
                    value = data.get(key)
                    if type(value) is int and value >= 0:
                        health[key] = value
        except (httpx.HTTPError, ValueError):
            pass
        finally:
            with self.lock:
                if gen == self.gen:
                    self.health, self.health_polling = health, False

    def status(self):
        with self.lock:
            if self.state == "ready" and self.process and self.process.poll() is not None:
                self.state = "error"
                self.error = f"basal-serve exited ({self.process.returncode}): {self.log.last() or ''}"
                self.event("error", f"{self.model}: {self.error}")
            if self.state == "ready" and self.base_url and not self.health_polling and time.monotonic() - self.health_checked >= 5:
                self.health_checked, self.health_polling = time.monotonic(), True
                threading.Thread(target=self._poll_health, args=(self.gen, self.base_url), daemon=True).start()
            restart_required = (self.state == "ready" and self.serve_snapshot is not None and not self.temporary
                                and self.serve_snapshot != _child_settings(
                                    self.settings_mod.serve_settings(),
                                    self.settings_mod.calibration_selection().get(self.model)))
            return {"id": self.model, "model": self.model, "repo": self.model_repo, "state": self.state,
                    "mode": self.mode, "command": self.command, "progress": self.progress, "error": self.error,
                    "media_requested": self.media_requested, "media": self.media,
                    "restart_required": restart_required, "est_memory_bytes": self.est_memory_bytes,
                    "rss_bytes": self.rss_bytes, "base_url": self.base_url, "entry": self.models_entry,
                    "options": dict(self.options), "health": dict(self.health) if self.state == "ready" else
                    {"reachable": None, "running": None, "queued": None}}

    def load(self, assessment, media=False, options=None, extra_args=(), temporary=False):
        """options: load options (basal/ui/options.py; their mode replaces the assessment's, except with media);
        temporary: a private child for a job (basal-calibrate): no API key (it is reached on loopback without one),
        no calibration selected in basal-ui, not remembered for auto-load; extra_args go to basal-serve last."""
        options = load_options.non_default(load_options.validate(options, assessment.mode))
        if media:  # --media on: eager, or vLLM where chosen (prepare checks it can run)
            assessment = dataclasses.replace(assessment, mode="vllm" if options.get("mode") == "vllm" else "eager")
        elif options.get("mode"):
            assessment = dataclasses.replace(assessment, mode=options["mode"])
        with self.lock:
            if self.state in ("downloading", "starting"):
                raise RuntimeError("the model is already loading; unload it first")
            old, _ = self.detach()
            self.state, self.model, self.model_repo = "downloading", assessment.variant.id, assessment.variant.repo
            self.mode = assessment.mode
            self.media_requested, self.media = media, []
            self.est_memory_bytes = assessment.est_memory_bytes
            self.extra_args, self.temporary, self.options = tuple(extra_args), temporary, options
            gen = self.gen
            cache_dir = self.models_dir_fn()
            serve = self.settings_mod.serve_settings()
        self._terminate(old)
        threading.Thread(target=self._start, args=(assessment, gen, media, cache_dir, serve), daemon=True).start()
        return self.status()

    def _update(self, gen, **fields):
        """Set fields if gen is still the current load; False when it was unloaded or replaced meanwhile."""
        with self.lock:
            if gen != self.gen:
                return False
            for k, v in fields.items():
                setattr(self, k, v)
            return True

    def _download(self, v, gen, cache_dir, revision=None):
        record = self.downloads.start(v, cache_dir, revision=revision)
        while record["thread"].is_alive():
            done = record["bar"].done
            if not self._update(gen, progress={"done_bytes": done, "total_bytes": max(v.weights_bytes or 0, done)}):
                return None  # unloaded while downloading: the download finishes in the background, into the cache
            record["thread"].join(0.5)
        if record["error"] is not None:
            raise record["error"]
        return Path(record["path"])

    def _calibration(self, v):
        return None if self.temporary else self.settings_mod.calibration_selection().get(v.id)

    def _build_command(self, v, path, assessment, media, port, serve):
        """The child is private: loopback, a free port, the internal key (basal-ui's admin key) required."""
        command = [sys.executable, "-m", "basal.server",
                   *catalog.serve_args(v, path, assessment.mode, self.platform_fn() if media else None,
                                       platform=self.platform_fn()),
                   "--name", v.id, "--host", "127.0.0.1", "--port", str(port)]
        if not self.temporary:
            command += ["--api-keys", str(self.settings_mod.internal_keys_file(hashlib.sha256(
                self.internal_secret.encode()).hexdigest()))]
        if self._calibration(v):
            command += ["--calibration", self._calibration(v)]
        if serve["log_decisions"] and not self.temporary:
            command += ["--log-decisions", serve["log_decisions"]]
        if assessment.mode in ("vllm", "sglang") and self.options.get("max_len") is None:
            command += ["--max-len", str(serve["max_len"])]
        if not media and assessment.mode == "mps" and self.platform_fn() != "apple" and not self.options.get("device"):
            command += ["--device", "cpu"]
        return command + load_options.args(self.options) + list(self.extra_args)

    def _models_entry(self, media):
        """The child's GET /v1/models entry and, with media, its media kinds (required then, best effort else)."""
        attempts, error = (5 if media else 3), None
        for attempt in range(attempts):
            try:
                response = httpx.get(self.base_url + "/v1/models", headers=self._headers(), timeout=1)
                response.raise_for_status()
                models = response.json().get("models", [])
                entry = models[0] if models else None
                return entry, (entry.get("media", []) if media and entry else [])
            except Exception as exc:  # noqa: BLE001
                error = exc
                if attempt < attempts - 1:
                    time.sleep(0.5)
        if media:
            raise RuntimeError(f"could not read media capabilities from basal-serve: {error}")
        return None, []

    def _start(self, assessment, gen, media=False, cache_dir=None, serve=None):
        v = assessment.variant
        serve = serve or self.settings_mod.serve_settings()
        try:
            revision = self.options.get("revision")
            path = catalog.is_downloaded(v, cache_dir, revision=revision) or self._download(v, gen, cache_dir, revision)
            if path is None:
                return
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            command = self._build_command(v, path, assessment, media, port, serve)
            options = ({"start_new_session": True} if os.name == "posix" else
                       {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)})
            with self.lock:
                if gen != self.gen:
                    return
                self.state, self.progress, self.command = "starting", None, command
                self.base_url = f"http://127.0.0.1:{port}"
                self.serve_snapshot = _child_settings(serve, self._calibration(v))
                self.process = proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                                       **options)
            self.event("info", f"{v.id}: basal-serve starting on 127.0.0.1:{port}: {shlex.join(map(str, command))}")
            threading.Thread(target=self._read_logs, args=(proc,), daemon=True).start()
            deadline = time.monotonic() + START_TIMEOUT_S
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    time.sleep(0.2)  # let the log reader drain the last lines
                    raise RuntimeError(f"basal-serve exited ({proc.returncode}): {self.log.last() or 'no output'}")
                try:
                    if httpx.get(self.base_url + "/health", headers=self._headers(), timeout=1).status_code == 200:
                        entry, media_kinds = self._models_entry(media)
                        with self.lock:
                            if gen != self.gen:
                                return
                            self.state, self.media = "ready", media_kinds
                            self.models_entry, self.ready_at = entry, time.time()
                            if not self.temporary:
                                self.settings_mod.remember_loaded(v.id, media, self.options)
                        self.event("info", f"{v.id}: ready")
                        return
                except httpx.HTTPError:
                    pass
                if gen != self.gen:
                    return
                time.sleep(0.5)
            raise RuntimeError(f"basal-serve did not become healthy within {START_TIMEOUT_S} s")
        except Exception as exc:  # any failure of this load is shown in the UI
            proc = None
            with self.lock:
                if gen == self.gen:
                    proc = self.process
                    self.process = None
                    self.gen += 1
                    self.state, self.error = "error", str(exc)
            if proc is not None or self.state == "error":
                self.event("error", f"{v.id}: {exc}")
            self._terminate(proc)

    def _read_logs(self, proc):
        for line in proc.stdout:
            self.log.append(line.rstrip())

    def unload(self):
        proc, _ = self.detach()
        self._terminate(proc)
        return self.status()

    def detach(self):
        """Start a new generation and forget the model; -> (the child to terminate, the memory estimate it held)."""
        with self.lock:
            self.gen += 1
            proc, self.process = self.process, None
            est = self.est_memory_bytes if self.state not in ("idle", "error") else None
            self._reset()
            return proc, est

    @staticmethod
    def _terminate(proc):
        """Stop basal-serve and everything it started (its own process group / session), then reap it."""
        if not proc:
            return

        def signal_group(kill):
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
                else:
                    proc.kill() if kill else proc.terminate()
            except ProcessLookupError:
                pass

        signal_group(kill=False)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_group(kill=True)
            proc.wait()

    def _headers(self):
        return {"Authorization": f"Bearer {self.internal_secret}"}

    def _url(self, path):
        with self.lock:
            if self.state != "ready":
                raise RuntimeError(f"model {self.model} is not ready ({self.state})")
            return self.base_url + path

    def decide(self, body, path="/v1/systemone"):
        return httpx.post(self._url(path), json=body, headers=self._headers(), timeout=120)

    async def adecide(self, client, body, path="/v1/systemone"):
        return await client.post(self._url(path), json=body, headers=self._headers())


class Fleet:
    """The loaded models (one Runner each, in load order), their shared downloads, logs, statistics and history."""

    def __init__(self, models_dir_fn, platform_fn, settings_mod, hardware_fn, stats=None, history=None):
        self.models_dir_fn, self.platform_fn = models_dir_fn, platform_fn
        self.settings_mod, self.hardware_fn = settings_mod, hardware_fn
        self.stats, self.history = stats, history
        self.internal_secret = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.runners: dict[str, Runner] = {}
        self.logs: dict[str, LogBuffer] = {"basal-ui": LogBuffer()}
        self.downloads = Downloads(self.event)
        self.jobs = Jobs(self.logs, self.event)
        self.temporary: dict[str, Runner] = {}  # private children of jobs (basal-calibrate's uncalibrated server)
        self.reserved: dict[str, int] = {}  # memory of models a job loads into basal-ui's own process
        self._lan_ip = None
        self.retiring: dict[Runner, int | None] = {}  # unloaded runners whose child is still being terminated
        self._rss_cache = (0.0, {})

    def event(self, level, message):
        self.logs["basal-ui"].append(f"{level} {message}")

    def internal_sha256(self):
        return hashlib.sha256(self.internal_secret.encode()).hexdigest()

    def lan_ip(self):
        if self._lan_ip is None:
            self._lan_ip = _lan_ip()
        return self._lan_ip

    def _memory(self, rss_total=0):
        config = self.settings_mod.memory_settings()
        hw = self.hardware_fn()
        budget = config["budget_bytes"] or hw.memory_bytes
        committed = (sum(r.est_memory_bytes or 0 for r in self.runners.values() if r.state not in ("idle", "error"))
                     + sum(r.est_memory_bytes or 0 for r in self.temporary.values() if r.state not in ("idle", "error"))
                     + sum(est or 0 for est in self.retiring.values()) + sum(self.reserved.values()))
        return {"budget_bytes": budget, "reserve_bytes": config["reserve_bytes"],
                "hardware_budget_bytes": hw.memory_bytes, "available_bytes": hw.available_bytes,
                "committed_bytes": committed, "rss_bytes": rss_total,
                "over": committed > budget - config["reserve_bytes"]}

    def check_budget(self, needed):
        """RuntimeError when `needed` more bytes would go over the memory budget (call under the fleet lock)."""
        memory = self._memory()  # an errored load being replaced has no process and does not count
        limit = memory["budget_bytes"] - memory["reserve_bytes"]
        if memory["committed_bytes"] + (needed or 0) > limit:
            raise RuntimeError(
                f"over memory budget: {_fmt(memory['committed_bytes'])} loaded + {_fmt(needed or 0)} needed > "
                f"{_fmt(limit)} ({_fmt(memory['budget_bytes'])} budget − {_fmt(memory['reserve_bytes'])} kept "
                "free); unload a model or raise the budget")

    def _runner(self, log):
        return Runner(models_dir_fn=self.models_dir_fn, platform_fn=self.platform_fn, settings_mod=self.settings_mod,
                      downloads=self.downloads, internal_secret=self.internal_secret, log=log, event=self.event)

    def start_temporary(self, kind, assessment, log, options=None, extra_args=()) -> Runner:
        """A private, unremembered child for a job (its memory counted in the budget); stop_temporary ends it."""
        with self.lock:
            if kind in self.temporary:
                raise RuntimeError(f"a {kind} child is already running")
            self.check_budget(assessment.est_memory_bytes)
            runner = self.temporary[kind] = self._runner(log)
            runner.load(assessment, options=options, extra_args=extra_args, temporary=True)
        self.event("info", f"{kind}: temporary basal-serve for {assessment.variant.id}")
        return runner

    def stop_temporary(self, kind):
        with self.lock:
            runner = self.temporary.pop(kind, None)
            if runner is not None:
                self.retiring[runner] = runner.est_memory_bytes
        if runner is not None:
            runner.unload()
            with self.lock:
                self.retiring.pop(runner, None)

    def reserve(self, kind, needed):
        """Count `needed` bytes for a job that loads a model into basal-ui's own process (RuntimeError over the
        budget); release(kind) frees them."""
        with self.lock:
            if kind in self.reserved:
                raise RuntimeError(f"a {kind} job already holds memory")
            self.check_budget(needed)
            self.reserved[kind] = needed or 0

    def release(self, kind):
        with self.lock:
            self.reserved.pop(kind, None)

    def load(self, assessment, media=False, options=None, _reservation=None):
        model_id = assessment.variant.id
        with self.lock:
            existing = self.runners.get(model_id)
            state = existing.status()["state"] if existing is not None else "idle"
            if state not in ("idle", "error"):
                raise RuntimeError(f"model {model_id} is already {'loaded' if state == 'ready' else 'loading'}")
            if _reservation is None:
                self.check_budget(assessment.est_memory_bytes)
            else:
                reserved = self.reserved.get(_reservation)
                if reserved is None:
                    raise RuntimeError("restart memory reservation was lost")
                self.reserved.pop(_reservation)
            self.logs.pop(model_id, None)
            self.logs[model_id] = log = LogBuffer()  # most recent last; a reload starts a fresh log
            runner = self._runner(log)
            self.runners[model_id] = runner
            self._prune_logs()
            runner.load(assessment, media=media, options=options)
        if existing is not None:
            existing.unload()
        self.event("info", f"load requested: {model_id}{' with media' if media else ''}")
        return self.status()

    def _prune_logs(self):
        unloaded = [key for key in self.logs
                    if key != "basal-ui" and not key.startswith("job:") and key not in self.runners]
        for key in unloaded[:max(0, len(self.logs) - 1 - MODEL_LOGS_KEPT)]:
            del self.logs[key]

    def unload(self, model_id=None, forget=True):
        """Unload one model (KeyError when it is not loaded) or every model; forget=False keeps it for auto-load."""
        with self.lock:
            if model_id is not None and model_id not in self.runners:
                raise KeyError(model_id)
            ids = [model_id] if model_id is not None else list(self.runners)
            retiring = []
            for i in ids:
                runner = self.runners.pop(i)
                # detach under the fleet lock: the old load can no longer reach ready and remember itself after
                # forget_loaded; its memory stays committed until the child is gone (a new load must wait for it)
                proc, est = runner.detach()
                retiring.append((runner, proc))
                self.retiring[runner] = est
                if forget:
                    self.settings_mod.forget_loaded(i)
            self._prune_logs()
        for i, (runner, proc) in zip(ids, retiring):
            Runner._terminate(proc)
            with self.lock:
                self.retiring.pop(runner, None)
            self.event("info", f"unloaded: {i}")
        if model_id is None:  # every model: the private children of jobs too (their job then fails)
            with self.lock:
                temporary = list(self.temporary)
            for kind in temporary:
                self.stop_temporary(kind)
        return self.status()

    def restart(self, assessment, media=False, options=None):
        """Unload and load again; options None keeps the running child's load options."""
        model_id = assessment.variant.id
        with self.lock:
            current = self.runners.get(model_id)
            if current is None:
                raise KeyError(model_id)
            if options is None:
                options = current.options
            proc, est = current.detach()
            self.runners.pop(model_id)
            self.retiring[current] = est
            # The existing remembered model remains available for auto-load after the restart.
        self.event("info", f"restart: {model_id}")
        Runner._terminate(proc)
        reservation = f"restart:{model_id}"
        with self.lock:
            self.retiring.pop(current, None)
            self.reserved[reservation] = assessment.est_memory_bytes or 0
        return self.load(assessment, media=media, options=options, _reservation=reservation)

    def download(self, variant) -> bool:
        """Start a runner-less download into the models directory; False when the model is on disk already.
        Under the fleet lock, so the directory cannot change (while_idle) or the repository be deleted
        (while_not_busy) between the check and the start."""
        with self.lock:
            cache_dir = self.models_dir_fn()
            if catalog.is_downloaded(variant, cache_dir):
                return False
            self.downloads.start(variant, cache_dir)
            return True

    def status(self):
        with self.lock:
            runners = list(self.runners.values())
        rss = self._rss() if any(r.process for r in runners) else {}
        models = []
        for runner in runners:
            runner.rss_bytes = rss.get(runner.pid) if runner.pid else None
            models.append(runner.status())
        with self.lock:
            memory = self._memory(sum(m["rss_bytes"] or 0 for m in models))
        return {"models": models, "downloads": self.downloads.status(), "memory": memory, "jobs": self.jobs.states()}

    def _rss(self):
        """rss_by_pgid() at most every RSS_INTERVAL_S: the UI polls every second while a model loads."""
        now = time.monotonic()
        if now - self._rss_cache[0] > RSS_INTERVAL_S:
            self._rss_cache = (now, rss_by_pgid())
        return self._rss_cache[1]

    def ready(self, model_id) -> Runner:
        with self.lock:
            runner = self.runners.get(model_id)
        if runner is None:
            raise KeyError(model_id)
        state = runner.status()["state"]
        if state != "ready":
            raise RuntimeError(f"model {model_id} is not ready ({state})")
        return runner

    def loaded_ids(self):
        with self.lock:
            runners = list(self.runners.items())
        return [model_id for model_id, runner in runners if runner.status()["state"] == "ready"]

    def model_entries(self):
        """The ready models for GET /v1/models: id, the child's own entry, the time it became ready."""
        with self.lock:
            runners = list(self.runners.items())
        return [{"id": model_id, "entry": runner.models_entry, "created": int(runner.ready_at or 0)}
                for model_id, runner in runners if runner.status()["state"] == "ready"]

    def busy(self, repo) -> bool:
        """A download of the repository is running (also one left in the background by an unload), or a loaded or
        loading model comes from it."""
        with self.lock:
            return self.downloads.alive(repo) or any(r.model_repo == repo and r.state != "idle"
                                                     for r in [*self.runners.values(), *self.temporary.values()])

    def while_idle(self, fn):
        """fn() with no model, job child and download running, atomically with load()."""
        with self.lock:
            if self.runners or self.temporary or self.downloads.any_alive():
                raise RuntimeError("unload every model before changing the models directory")
            return fn()

    def while_not_busy(self, repo, fn):
        """fn() while nothing uses the repository, atomically with load()."""
        with self.lock:
            if self.busy(repo):
                raise RuntimeError("unload the model first")
            return fn()

    def _record(self, model_id, body, path, source, status, data, latency_ms):
        if source == "benchmark":
            return
        if self.stats is not None:
            self.stats.record(model_id, data, latency_ms, ok=200 <= status < 300)
        if self.history is not None:
            request = {"model": model_id, **body} if path == "/v1/systemone" else body
            self.history.append(make_record(model_id, source, request, status, data, latency_ms))

    @staticmethod
    def _json(response):
        try:
            return response.json()
        except ValueError:
            return {"error": response.text}

    def decide(self, model_id, body, path="/v1/systemone", source="playground"):
        runner = self.ready(model_id)
        t0 = time.perf_counter()
        try:
            response = runner.decide(body, path)
        except httpx.HTTPError as exc:
            self._record(model_id, body, path, source, 502, {"error": f"basal-serve: {exc}"},
                         (time.perf_counter() - t0) * 1000)
            raise
        self._record(model_id, body, path, source, response.status_code, self._json(response),
                     (time.perf_counter() - t0) * 1000)
        return response

    async def adecide(self, model_id, body, path="/v1/systemone", source="api", client=None):
        runner = self.ready(model_id)
        t0 = time.perf_counter()
        try:
            response = await runner.adecide(client, body, path)
        except httpx.HTTPError as exc:
            self._record(model_id, body, path, source, 502, {"error": f"basal-serve: {exc}"},
                         (time.perf_counter() - t0) * 1000)
            raise
        self._record(model_id, body, path, source, response.status_code, self._json(response),
                     (time.perf_counter() - t0) * 1000)
        return response

    def shutdown(self):
        self.jobs.cancel_all()
        self.unload(forget=False)


class _Sink:
    def write(self, _text):
        pass

    def flush(self):
        pass


def _byte_counter():
    """A tqdm class for snapshot_download that prints nothing and keeps, as the class attribute `done`, the larger of
    its two aggregated byte bars: bytes received ("Downloading bytes", which Xet reports while chunks arrive) and
    bytes written ("Reconstructing", fed by every file, Xet or plain HTTP)."""
    from tqdm.auto import tqdm

    class Counter(tqdm):
        done = 0

        def __init__(self, *args, **kwargs):
            kwargs["file"] = _Sink()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            out = super().update(n)
            if self.unit == "B" and str(self.desc).startswith(("Reconstructing", "Downloading bytes")):
                Counter.done = max(Counter.done, int(self.n))
            return out

    return Counter


def host_middleware(app, allowed_hosts):
    """Apply basal-ui's Host-header allowlist to an ASGI app."""
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    return app


def create_app(catalog_fn=catalog.list_variants, hardware_fn=hardware.detect, fleet=None,
               settings_mod=settings, allowed_hosts=("127.0.0.1", "localhost", "::1"), gateway=None, stats=None,
               history=None) -> FastAPI:
    stats = stats or Stats(settings_mod.stats_path())
    history = history or History(settings_mod.history_path())
    fleet = fleet or Fleet(models_dir_fn=settings_mod.models_dir, platform_fn=lambda: hardware_fn().platform,
                           settings_mod=settings_mod, hardware_fn=hardware_fn, stats=stats, history=history)
    gateway = gateway or Gateway(fleet, settings_mod, stats)
    allowed_hosts = tuple(allowed_hosts)
    from basal.viz import examples as viz_examples
    from basal.viz import make_app as viz_app

    from .inspect import Inspector
    inspector = Inspector()

    def hw():
        return hardware_fn()

    def prepare(variant, media, options):
        """(assessment, the non-default load options) of a load; ValueError says why this machine cannot load it so:
        an unavailable mode, invalid options, or a variant the default mode cannot serve."""
        current = hw()
        assessment = catalog.assess(variant, current)
        merged = load_options.validate(options, assessment.mode)
        if media:
            modes = {m["mode"]: m for m in load_options.available_media_modes(assessment, current)}
            chosen = modes.get(merged["mode"] or "eager")
            if chosen is None:
                raise ValueError("--media on needs --mode eager or vllm (the media paths of basal-omni)")
            if not chosen["available"]:
                raise ValueError(chosen["reason"])
            assessment = dataclasses.replace(assessment, mode=merged["mode"] or "eager")
        elif merged["mode"]:
            mode = next(m for m in load_options.available_modes(variant, current, assessment)
                        if m["mode"] == merged["mode"])
            if not mode["available"]:
                raise ValueError(f"--mode {merged['mode']}: {mode['reason']}")
            assessment = dataclasses.replace(assessment, mode=merged["mode"])
        elif not assessment.runnable:
            raise ValueError(assessment.reason)
        load_options.check_combination(variant, current, merged, assessment.mode, media)
        return assessment, load_options.non_default(merged)

    auto_load_stop = threading.Event()
    auto_load_lock = threading.Lock()

    def _auto_load():
        try:
            variants, _ = catalog_fn(cache_dir=settings_mod.models_dir())
        except Exception as exc:  # the catalog is unavailable: nothing to load
            print(f"basal-ui: auto-load skipped: {exc}", file=sys.stderr)
            fleet.event("warn", f"auto-load skipped: {exc}")
            return
        for last in settings_mod.last_models():
            if auto_load_stop.is_set():
                return
            try:
                variant = next((v for v in variants if v.id == last["id"]), None)
                if variant is None:
                    raise ValueError("model is unknown")
                assessment, options = prepare(variant, last.get("media", False), last.get("options"))
                with auto_load_lock:
                    if not auto_load_stop.is_set():
                        fleet.load(assessment, media=last.get("media", False), options=options)
                        fleet.event("info", f"auto-load: {last['id']}")
            except Exception as exc:
                print(f"basal-ui: auto-load of {last['id']} skipped: {exc}", file=sys.stderr)
                fleet.event("warn", f"auto-load of {last['id']} skipped: {exc}")

    @asynccontextmanager
    async def lifespan(_app):
        auto_load_thread = None
        if settings_mod.load().get("auto_load") and settings_mod.last_models():
            auto_load_thread = threading.Thread(target=_auto_load, daemon=True)
            auto_load_thread.start()
        inspector.bind(asyncio.get_running_loop())  # the explainer's batching loop runs on this one
        await asyncio.to_thread(gateway.start)
        yield
        with auto_load_lock:
            auto_load_stop.set()
        if auto_load_thread is not None:
            while auto_load_thread.is_alive():
                auto_load_thread.join(timeout=5)
        while not gateway.stop():
            fleet.event("error", "API server shutdown is still draining; waiting before tearing down shared state")
            await asyncio.sleep(1)
        inspector.stop()
        fleet.shutdown()
        stats.close()
        settings_mod.remove_keys_file()

    app = host_middleware(FastAPI(title="basal-ui", lifespan=lifespan), allowed_hosts)
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/viz", viz_app(inspector.explainer, viz_examples))  # basal-viz for the Inspect view


    @app.middleware("http")
    async def refuse_cross_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and origin and "*" not in allowed_hosts:
            try:
                parsed = urlsplit(origin)
                origin_host = parsed.hostname
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
            except ValueError:
                parsed, origin_host = None, None
                origin_port = request_port = None
            if (origin_host is None or origin_host.lower() not in {host.lower() for host in allowed_hosts}
                    or parsed.scheme != request.url.scheme or parsed.path or parsed.query or parsed.fragment
                    or parsed.username is not None or parsed.password is not None or origin_port != request_port):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        # revalidate on every load: a cached copy from an older basal-ui would talk to the new API
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    def find_variant(model_id):
        variants, _ = catalog_fn(cache_dir=settings_mod.models_dir())
        variant = next((v for v in variants if v.id == model_id), None)
        if variant is None:
            raise HTTPException(400, f"unknown model {model_id!r}")
        return variant

    def full_status():
        return {**fleet.status(), "serving": gateway.status(), "inspect": inspector.state()}

    def serve_response():
        config = settings_mod.serve_settings()
        available = None
        hint = None
        try:
            from basal.mcp_server import available as mcp_available
            hint = mcp_available()
            available = hint is None
        except ImportError as exc:
            hint = str(exc)
        usage = gateway.key_usage()
        lan = fleet.lan_ip() if config["host"] == "0.0.0.0" else None
        return {"settings": config, "auto_load": settings_mod.load().get("auto_load", False),
                "keys": [{**{k: v for k, v in key.items() if k != "sha256"},
                          "last_used": usage.get(key["name"])} for key in settings_mod.keys()],
                "mcp_available": available, "mcp_hint": hint, "lan_ip": lan}

    @app.get("/api/serve")
    def get_serve():
        return serve_response()

    @app.put("/api/serve")
    def put_serve(body: dict = Body(...)):
        try:
            if "auto_load" in body and not isinstance(body["auto_load"], bool):
                raise ValueError("auto_load must be a boolean")
            patch = {k: v for k, v in body.items() if k != "auto_load"}
            if patch:
                settings_mod.update_serve(patch)
            if "auto_load" in body:
                settings_mod.set_auto_load(body["auto_load"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return serve_response()

    @app.post("/api/serve/restart")
    def restart_serve():
        fleet.event("info", "API server restart requested")
        return gateway.restart()

    @app.post("/api/serve/keys", status_code=201)
    def post_key(body: dict = Body(...)):
        try:
            record, secret = settings_mod.create_key(body.get("name"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**record, "key": secret}

    @app.delete("/api/serve/keys/{name}")
    def remove_key(name: str):
        try:
            settings_mod.delete_key(name)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"deleted": name}

    @app.get("/api/memory")
    def get_memory():
        current = hw()
        return {"settings": settings_mod.memory_settings(), "hardware_budget_bytes": current.memory_bytes,
                "available_bytes": current.available_bytes}

    @app.put("/api/memory")
    def put_memory(body: dict = Body(...)):
        try:
            settings_mod.update_memory(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return get_memory()

    @app.post("/api/restart", status_code=202)
    def restart_model(body: dict = Body(...)):
        model_id = body.get("id")
        current = next((m for m in fleet.status()["models"] if m["id"] == model_id), None)
        if current is None:
            raise HTTPException(404, f"model {model_id!r} is not loaded")
        if current["state"] in ("downloading", "starting"):
            raise HTTPException(409, "model is still loading")
        variants, _ = catalog_fn(cache_dir=settings_mod.models_dir())
        variant = next((v for v in variants if v.id == model_id), None)
        if variant is None:
            raise HTTPException(400, "loaded model is no longer available")
        try:
            options = body.get("options")
            assessment, options = prepare(variant, current.get("media_requested", False),
                                          current.get("options") if options is None else options)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            fleet.restart(assessment, media=current.get("media_requested", False), options=options)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return full_status()

    @app.get("/api/hardware")
    def get_hardware():
        return hw().to_dict()

    def settings_response():
        from huggingface_hub.constants import HF_HUB_CACHE
        models = Path(settings_mod.models_dir()).resolve()
        default = Path(HF_HUB_CACHE).resolve()
        return {"models_dir": str(models), "default_models_dir": str(default), "is_default": models == default,
                "check_updates": settings_mod.check_updates()}

    @app.get("/api/settings")
    def get_settings():
        return settings_response()

    @app.put("/api/settings")
    def put_settings(body: dict = Body(...)):
        models_dir = body.get("models_dir")
        check_updates = body.get("check_updates")
        if "check_updates" in body and not isinstance(check_updates, bool):
            raise HTTPException(400, "check_updates must be a boolean")
        if models_dir is not None and not isinstance(models_dir, str):
            raise HTTPException(400, "models_dir must be a string or null")
        try:
            if "models_dir" in body:
                fleet.while_idle(lambda: settings_mod.set_models_dir(models_dir))
            if "check_updates" in body:
                settings_mod.set_check_updates(check_updates)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return settings_response()

    @app.get("/api/models")
    def get_models(refresh: bool = False):
        cache_dir = settings_mod.models_dir()
        current_hw = hw()
        variants, source = catalog_fn(refresh=refresh, cache_dir=cache_dir)
        assessed = [catalog.assess(v, current_hw) for v in variants]
        assessed.sort(key=lambda a: (FAMILY_ORDER.get(a.variant.family, 99),
                                     catalog.TIER_ORDER.get(a.variant.tier, 99), a.variant.port, a.variant.label))
        fields = ("id", "repo", "author", "community", "base_repo", "family", "tier", "port", "label", "gguf_file",
                  "weights_bytes", "experimental")
        on_disk = {a.variant.id: catalog.downloaded_bytes(a.variant, cache_dir) for a in assessed}
        saved = settings_mod.load().get("load_options")  # once, not load_options() per row
        revisions = {k: v.get("revision") for k, v in saved.items() if isinstance(v, dict)} if isinstance(
            saved, dict) else {}
        return {"source": source, "suggestions": catalog.suggest(assessed, current_hw),
                "models_dir": str(Path(cache_dir).resolve()), "models": [
            {**{k: getattr(a.variant, k) for k in fields}, "runnable": a.runnable, "reason": a.reason,
             "mode": a.mode, "est_memory_bytes": a.est_memory_bytes, "fit": a.fit, "fit_now": a.fit_now,
             "media": list(a.variant.media), "media_mode": a.media_mode, "media_reason": a.media_reason,
             "downloaded": on_disk[a.variant.id] is not None, "downloaded_bytes": on_disk[a.variant.id],
             "revision": revisions.get(a.variant.id)}
            for a in assessed]}

    @app.get("/api/presets")
    def get_presets():
        return {"load": settings_mod.presets().get("load", {}),
                "request": settings_mod.presets().get("request", {})}

    @app.put("/api/presets")
    def put_preset(body: dict = Body(...)):
        kind, name, value = body.get("kind"), body.get("name"), body.get("value")
        try:
            if kind == "load" and value is not None:
                value = load_options.non_default(load_options.validate(value))
            elif kind == "request" and value is not None:
                if not isinstance(value, dict) or not isinstance(value.get("questions"), dict) or not value["questions"]:
                    raise ValueError("request preset must contain a non-empty questions object")
                value = {k: v for k, v in value.items() if k != "model"}
            settings_mod.set_preset(kind, name, value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return get_presets()

    @app.get("/api/models/card")
    def get_model_card(id: str):
        variant = find_variant(id)
        cache_dir = settings_mod.models_dir()
        local_dir = catalog.is_downloaded(variant, cache_dir, revision=served_revision(variant))
        assessment = catalog.assess(variant, hw())
        loaded = next((m.get("entry") for m in fleet.status()["models"] if m["id"] == variant.id), None)
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound
        try:
            info = scan_cache_dir(cache_dir)
            revisions = sorted({s.commit_hash for repo in info.repos if repo.repo_id == variant.repo
                                for s in repo.revisions})
        except (CacheNotFound, OSError, ValueError):
            revisions = []
        revision = served_revision(variant) or "main"
        readme = None
        if local_dir and (Path(local_dir) / "README.md").is_file():
            readme = (Path(local_dir) / "README.md").read_text(encoding="utf-8", errors="replace")[:20000]
        else:
            try:
                response = httpx.get(f"https://huggingface.co/{variant.repo}/raw/{revision}/README.md", timeout=5)
                if response.is_success:
                    readme = response.text[:20000]
            except httpx.HTTPError:
                pass

        def metadata(name):
            """A JSON object of the repository (basal.json, config.json): from the snapshot, else from the Hub."""
            try:
                if local_dir:
                    data = json.loads((Path(local_dir) / name).read_text(encoding="utf-8"))
                else:
                    response = httpx.get(f"https://huggingface.co/{variant.repo}/raw/{revision}/{name}", timeout=5)
                    data = response.json() if response.is_success else None
            except (OSError, ValueError, httpx.HTTPError):
                return None
            return data if isinstance(data, dict) else None

        basal_json, config_json = metadata("basal.json"), metadata("config.json")
        shipped_calibration = None
        if local_dir:
            from . import calibration
            try:
                path = calibration.shipped_file(local_dir, served_mode(variant, assessment))
                shipped_calibration = calibration.summary(path) if path.is_file() else None
            except (OSError, ValueError, KeyError) as exc:
                shipped_calibration = {"error": str(exc)}
        return {"id": variant.id, "repo": variant.repo, "url": f"https://huggingface.co/{variant.repo}",
                "author": variant.author, "community": variant.community, "base_repo": variant.base_repo,
                "base_revision": variant.base_revision,
                "family": variant.family, "tier": variant.tier, "port": variant.port, "label": variant.label,
                "size_bytes": variant.weights_bytes, "media": list(variant.media), "downloaded": local_dir is not None,
                "local_dir": str(local_dir) if local_dir else None, "revisions": revisions, "basal_json": basal_json,
                "quantisation": catalog.quantisation(variant, basal_json, config_json),
                "shipped_calibration": shipped_calibration, "readme": readme,
                "assessment": {"mode": assessment.mode, "runnable": assessment.runnable,
                               "reason": assessment.reason, "est_memory_bytes": assessment.est_memory_bytes,
                               "fit": assessment.fit, "fit_now": assessment.fit_now},
                "modes": load_options.available_modes(variant, hw(), assessment), "loaded": loaded}

    @app.get("/api/updates")
    def get_updates():
        from . import updates
        import basal
        enabled = settings_mod.check_updates()
        latest, error = updates.latest() if enabled else (None, None)
        installed = basal.__version__
        try:
            newer = latest is not None and tuple(map(int, latest.split("."))) > tuple(map(int, installed.split(".")))
        except ValueError:
            newer = False
        variants, source = catalog_fn(cache_dir=settings_mod.models_dir())
        # only the hub catalog is the whole catalog: the cache fallback would seed (or report) a partial list
        new_models = settings_mod.update_seen_ids(v.id for v in variants) if source == "hub" else []
        return {"enabled": enabled, "installed": installed, "latest": latest,
                "url": f"https://github.com/rkinas/basal/releases/tag/v{latest}" if latest else None,
                "newer": newer, "error": error, "new_models": new_models, "catalog": source}

    @app.post("/api/updates/seen")
    def mark_updates_seen():
        variants, source = catalog_fn(cache_dir=settings_mod.models_dir())
        if source == "hub":
            settings_mod.mark_models_seen(v.id for v in variants)
        return get_updates()

    @app.get("/api/models/options")
    def get_model_options(id: str):
        variant = find_variant(id)
        current = hw()
        assessment = catalog.assess(variant, current)
        saved = settings_mod.load_options(variant.id)
        try:
            merged = load_options.validate(saved, assessment.mode)
        except ValueError:  # options saved by another basal-ui version
            merged = dict(load_options.DEFAULTS)
        family_ok = variant.family == "basal-2.0" and variant.tier == "4.5B"
        return {"modes": load_options.available_modes(variant, current, assessment), "options": merged,
                "saved": bool(saved), "defaults": load_options.DEFAULTS,
                "media_modes": load_options.available_media_modes(assessment, current),
                "mlx_packed_reason": load_options.mlx_packed_unavailable(variant, current),
                "think": {"model": family_ok, "vllm": "vllm" in current.engines,
                          "reason": None if family_ok else load_options.THINK_REASON}}

    @app.put("/api/models/options")
    def put_model_options(body: dict = Body(...)):
        variant = find_variant(body.get("id"))
        try:
            current = hw()
            default_mode = catalog.assess(variant, current).mode
            merged = load_options.validate(body.get("options"), default_mode)
            load_options.check_combination(variant, current, merged, default_mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        settings_mod.set_load_options(variant.id, load_options.non_default(merged))
        return get_model_options(variant.id)

    @app.post("/api/models/delete")
    def delete_model(body: dict = Body(...)):
        variant = find_variant(body.get("id"))
        try:
            freed = fleet.while_not_busy(variant.repo, lambda: catalog.delete(variant, settings_mod.models_dir()))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"freed_bytes": freed}

    @app.post("/api/download", status_code=202)
    def download(body: dict = Body(...)):
        variant = find_variant(body.get("id"))
        if not fleet.download(variant):
            raise HTTPException(409, "already on disk")
        return full_status()

    @app.post("/api/load", status_code=202)
    def load(body: dict = Body(...)):
        variant = find_variant(body.get("id"))
        media = body.get("media", False)
        if not isinstance(media, bool):
            raise HTTPException(400, "media must be a boolean")
        try:
            if "options" in body:
                options = body["options"]
            else:  # remembered; their engine mode applies with media only when it is a media mode
                options = settings_mod.load_options(variant.id)
                if media and options.get("mode") not in (None, "eager", "vllm"):
                    options = {k: v for k, v in options.items() if k != "mode"}
            assessment, options = prepare(variant, media, options)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            fleet.load(assessment, media=media, options=options)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return full_status()

    @app.post("/api/unload")
    def unload(body: dict | None = Body(None)):
        model_id = (body or {}).get("id")
        try:
            fleet.unload(model_id)
        except KeyError as exc:
            raise HTTPException(404, f"model {model_id!r} is not loaded") from exc
        return full_status()

    @app.get("/api/status")
    def status():
        return full_status()

    @app.get("/api/examples")
    def get_examples():
        from basal.viz import examples
        return examples()

    @app.post("/api/decide")
    def decide(body: dict = Body(...)):  # sync: runs in the threadpool, a slow decision does not block the UI
        body = dict(body)
        model_id = body.pop("model", None)
        if not isinstance(model_id, str) or not model_id:
            raise HTTPException(400, f"model is required; loaded: {fleet.loaded_ids()}")
        try:
            r = fleet.decide(model_id, body, source="playground")
        except KeyError as exc:
            raise HTTPException(404, f"model {model_id!r} is not loaded") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"basal-serve: {exc}") from exc
        return Response(r.content, status_code=r.status_code,
                        headers={k: r.headers[k] for k in ("x-request-id", "x-typesafe-request-id") if k in r.headers},
                        media_type=r.headers.get("content-type", "application/json"))

    @app.get("/api/stats")
    def get_stats():
        return stats.snapshot()

    @app.delete("/api/stats")
    def reset_stats():
        stats.reset()
        return {"reset": "all"}

    @app.delete("/api/stats/{model_id:path}")
    def reset_model_stats(model_id: str):
        stats.reset(model_id)
        return {"reset": model_id}

    @app.post("/api/benchmark", status_code=202)
    def post_benchmark(body: dict = Body(...)):
        """kind benchmark (the preset), loadtest, determinism (a loaded model) or modes (a downloaded model, loaded
        in basal-ui's process mode by mode); one measurement at a time."""
        from . import bench

        model_id, kind = body.get("id"), body.get("kind", "benchmark")
        if kind not in ("benchmark", "loadtest", "determinism", "modes"):
            raise HTTPException(400, "kind must be benchmark, loadtest, determinism or modes")
        if kind == "modes":
            return start_modes(model_id, body)
        try:
            label, paths = bench.resolve_set(body.get("set", "bundled"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            runner = fleet.ready(model_id)
            if kind == "benchmark":
                return fleet.jobs.start("benchmark", model_id, label, bench.benchmark_job, fleet, model_id, paths,
                                        label)
            try:
                params = bench.loadtest_params(body.get("params") or {}, kind,
                                               (runner.models_entry or {}).get("early_exit") or [])
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if params["synthetic"]:
                label = f"synthetic {params['synthetic']} tokens"
            return fleet.jobs.start("benchmark", model_id, f"{kind}: {label}", bench.loadtest_job, fleet, model_id,
                                    label, paths, params)
        except KeyError as exc:
            raise HTTPException(404, f"model {model_id!r} is not loaded") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    def modes_context(variant):
        """(local dir, assessment, merged load options, the modes a comparison may run) of a downloaded variant."""
        from . import bench

        local_dir = downloaded(variant)
        current = hw()
        assessment = catalog.assess(variant, current)
        options = load_options.validate(settings_mod.load_options(variant.id), assessment.mode)
        return local_dir, assessment, options, bench.comparison_modes(
            load_options.available_modes(variant, current, assessment))

    def start_modes(model_id, body):
        from . import bench

        variant = find_variant(model_id)
        local_dir, assessment, options, selectable = modes_context(variant)
        modes, n, lat_n = body.get("modes"), body.get("n", 100), body.get("lat_n", 30)
        if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes):
            raise HTTPException(400, "modes must be a list of distinct modes")
        unknown = [m for m in modes if m not in selectable]
        if unknown:
            raise HTTPException(400, f"{unknown[0]} cannot run here; selectable: {selectable}")
        for key, value, lo in (("n", n, 10), ("lat_n", lat_n, 5)):
            if isinstance(value, bool) or not isinstance(value, int) or value < lo:
                raise HTTPException(400, f"{key} must be an integer >= {lo}")
        if fleet.jobs.get("benchmark") is not None and fleet.jobs.get("benchmark").running:
            raise HTTPException(409, "a benchmark job is already running")
        try:
            fleet.reserve("modes", bench.comparison_memory(modes, assessment.est_memory_bytes))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        try:
            return fleet.jobs.start("benchmark", variant.id, f"modes: {', '.join(modes)}", bench.modes_job, fleet,
                                    variant, local_dir, modes, n, lat_n, options)
        except RuntimeError as exc:
            fleet.release("modes")
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/modes")
    def get_comparison_modes(id: str):
        variant = find_variant(id)
        return {"modes": modes_context(variant)[3]}

    @app.delete("/api/benchmark")
    def cancel_benchmark():
        return fleet.jobs.cancel("benchmark")

    def served_revision(variant):
        """The revision the model is loaded at (load option), None for the default branch."""
        return settings_mod.load_options(variant.id).get("revision")

    def served_mode(variant, assessment):
        """The mode the model is served in; eager names the default CALIBRATION.json."""
        return settings_mod.load_options(variant.id).get("mode") or assessment.mode or "eager"

    def downloaded(variant):
        local_dir = catalog.is_downloaded(variant, settings_mod.models_dir(), revision=served_revision(variant))
        if not local_dir:
            raise HTTPException(400, f"download {variant.id} first")
        return Path(local_dir)

    @app.post("/api/calibrate", status_code=202)
    def post_calibrate(body: dict = Body(...)):
        from . import calibration

        variant = find_variant(body.get("id"))
        local_dir = downloaded(variant)
        source = body.get("input")
        if not isinstance(source, str) or not source.strip():
            raise HTTPException(400, "input must be the path of a labelled JSONL file (docs/CALIBRATION.md)")
        source = Path(source).expanduser()
        try:
            with source.open("rb") as f:
                f.read(1)
        except OSError as exc:
            raise HTTPException(400, f"cannot read {source}: {exc.strerror or exc}") from exc
        advanced = body.get("advanced") or {}
        if not isinstance(advanced, dict):
            raise HTTPException(400, "advanced must be an object")
        allowed = {"strict", "reference", "min_per_type", "min_blend", "min_family", "min_per_set", "concurrency",
                   "compare", "save_rows", "report_only"}
        if set(advanced) - allowed:
            raise HTTPException(400, f"unknown advanced calibration option: {min(set(advanced) - allowed)}")
        for key in ("strict", "save_rows", "report_only"):
            if key in advanced and not isinstance(advanced[key], bool):
                raise HTTPException(400, f"{key} must be a boolean")
        for key, minimum in (("min_per_type", 1), ("min_blend", 0), ("min_family", 1), ("min_per_set", 1),
                             ("concurrency", 1)):
            if key in advanced and (type(advanced[key]) is not int or advanced[key] < minimum or
                                    (key == "concurrency" and advanced[key] > 256)):
                raise HTTPException(400, f"{key} must be an integer >= {minimum}")
        reference = advanced.get("reference")
        if reference is not None and not isinstance(reference, str):
            raise HTTPException(400, "reference must be basal-1.5, a path, or null")
        compare = advanced.get("compare", "shipped")
        if not isinstance(compare, str) or compare not in ("shipped", "none") and not compare.strip():
            raise HTTPException(400, "compare must be shipped, none, or a calibration path")
        for key, value, names in (("reference", reference, ("basal-1.5",)), ("compare", compare, ("shipped", "none"))):
            if value and value not in names:
                path = Path(value).expanduser()
                if not path.is_file():
                    raise HTTPException(400, f"{key}: {value} is not a readable file")
                advanced[key] = str(path)
        from_rows = body.get("from_rows", False)
        if not isinstance(from_rows, bool):
            raise HTTPException(400, "from_rows must be a boolean")
        advanced["from_rows"] = from_rows
        units = []
        if not from_rows:
            try:
                _, units = calibration.labelled(source)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        in_process = bool(units)
        targets = body.get("targets") or list(calibration.TARGETS)
        try:
            if not isinstance(targets, list) or not targets or not all(0 < float(t) < 1 for t in targets):
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(400, "targets must be a list of max_error values between 0 and 1") from None
        if from_rows:  # a refit of saved rows runs no engine: no load options, no memory check
            assessment, options = catalog.assess(variant, hw()), {}
        else:
            try:
                assessment, options = prepare(variant, False, settings_mod.load_options(variant.id))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        out = settings_mod.calibrations_dir(variant.id) / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        label = f"{variant.id} on {source.name}" + (" (engine in process: task questions)" if in_process else "")
        reserve = in_process and not from_rows
        try:
            if reserve:
                fleet.reserve("calibrate", assessment.est_memory_bytes)
            elif not from_rows:
                with fleet.lock:
                    fleet.check_budget(assessment.est_memory_bytes)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        engine_args = None
        if in_process and not from_rows:
            engine_args = calibration.engine_flags(catalog.serve_args(
                variant, local_dir, assessment.mode, platform=hw().platform) + load_options.args(options))
        advanced["reserve"] = reserve
        try:
            return fleet.jobs.start("calibrate", variant.id, label, calibration.run_calibration, fleet, assessment,
                                    local_dir, source, out, [str(float(t)) for t in targets], options, engine_args,
                                    advanced)
        except RuntimeError as exc:
            if reserve:
                fleet.release("calibrate")
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/calibrate")
    def cancel_calibrate():
        state = fleet.jobs.cancel("calibrate")
        fleet.stop_temporary("calibrate")  # basal-calibrate's requests then fail and the job ends
        return state

    def calibrations_response(variant):
        from . import calibration

        local_dir = catalog.is_downloaded(variant, settings_mod.models_dir(), revision=served_revision(variant))
        shipped = None
        if local_dir:
            mode = served_mode(variant, catalog.assess(variant, hw()))
            path = calibration.shipped_file(local_dir, mode)
            shipped = {"file": str(path)}
            try:
                shipped.update(calibration.summary(path))
            except (OSError, ValueError) as exc:
                shipped["error"] = str(exc)
        return {"selected": settings_mod.calibration_selection().get(variant.id),
                "files": calibration.listing(variant.id, settings_mod, local_dir), "shipped": shipped}

    @app.post("/api/calibrations/import")
    def import_calibration(body: dict = Body(...)):
        """A calibration file fitted elsewhere (basal-calibrate on another machine, a colleague's file): copied into the
        model's calibrations, then selectable; a file of another model is listed with basal-serve's warning."""
        from . import calibration

        variant = find_variant(body.get("id"))
        downloaded(variant)  # the listing checks the file against the model's directory
        source = body.get("path")
        if not isinstance(source, str) or not source.strip():
            raise HTTPException(400, "path must name a calibration file (JSON)")
        try:
            target = calibration.import_file(variant.id, settings_mod, source)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        fleet.event("info", f"calibration imported for {variant.id}: {source} -> {target}")
        return {"file": str(target), **calibrations_response(variant)}

    @app.get("/api/calibrations")
    def get_calibrations(id: str):
        return calibrations_response(find_variant(id))

    @app.post("/api/calibrations/select")
    def select_calibration(body: dict = Body(...)):
        from . import calibration

        variant = find_variant(body.get("id"))
        path = body.get("file")
        if path is not None and path not in {f["file"] for f in calibration.listing(variant.id, settings_mod)}:
            raise HTTPException(400, f"{path!r} is not one of the calibrations of {variant.id} in basal-ui")
        settings_mod.select_calibration(variant.id, path)
        fleet.event("info", f"calibration for {variant.id}: {path or 'shipped'} (applies at the next start)")
        return calibrations_response(variant)

    @app.get("/api/calibrations/report")
    def get_calibration_report(id: str, file: str):
        from . import calibration

        variant = find_variant(id)
        if file not in {f["file"] for f in calibration.listing(variant.id, settings_mod)}:
            raise HTTPException(404, f"{file!r} is not one of the calibrations of {variant.id} in basal-ui")
        try:
            return {"report": Path(file).with_suffix(".report.txt").read_text(encoding="utf-8")}
        except OSError as exc:
            raise HTTPException(404, f"no report for {file}") from exc

    @app.post("/api/inspect", status_code=202)
    def post_inspect(body: dict = Body(...)):
        from . import calibration
        from .inspect import inspectable

        variant = find_variant(body.get("id"))
        reason = inspectable(variant)
        if reason:
            raise HTTPException(400, reason)
        local_dir = downloaded(variant)
        cal = body.get("calibration")
        if cal is not None and cal not in {f["file"] for f in calibration.listing(variant.id, settings_mod)}:
            raise HTTPException(400, f"{cal!r} is not one of the calibrations of {variant.id} in basal-ui")
        orders = body.get("orders")
        if orders not in (None, 1, 2):
            raise HTTPException(400, "orders must be 1, 2 or null (the model's own)")
        job = fleet.jobs.get("inspect")
        if job is not None and job.running:
            raise HTTPException(409, "the inspector is loading a model")
        inspector.stop()
        current = hw()
        try:
            fleet.reserve("inspect", catalog.assess(variant, current).est_memory_bytes)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

        def start(job):
            try:
                return inspector.start(job, variant, local_dir, current, cal, orders,
                                       on_exit=lambda: fleet.release("inspect"))
            except BaseException:
                fleet.release("inspect")
                raise

        try:
            return fleet.jobs.start("inspect", variant.id, variant.id, start)
        except RuntimeError as exc:
            fleet.release("inspect")
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/inspect")
    def stop_inspect():
        fleet.jobs.cancel("inspect")
        inspector.stop()
        fleet.event("info", "inspector stopped")
        return inspector.state()

    @app.get("/api/inspect")
    def get_inspect():
        return inspector.state()

    @app.post("/api/batch", status_code=202)
    def post_batch(body: dict = Body(...)):
        from . import batch

        model_id = body.get("id")
        source = body.get("input")
        if not isinstance(source, str) or not source.strip():
            raise HTTPException(400, "input must be the path of a JSONL file (docs/SERVING.md#inference-on-a-jsonl-file)")
        source = Path(source).expanduser()
        try:
            lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeDecodeError) as exc:
            raise HTTPException(400, f"cannot read {source}: {getattr(exc, 'strerror', None) or exc}") from exc
        for n, line in enumerate(lines, 1):
            try:
                item = json.loads(line)
            except ValueError:
                item = None
            if not isinstance(item, dict) or not ("questions" in item or "options" in item):
                raise HTTPException(400, f"{source.name}: line {n} is neither a simple item (state, question, "
                                         "options) nor a /v1/systemone request")
        if not lines:
            raise HTTPException(400, f"{source} has no lines")
        out = Path(body["output"]).expanduser() if body.get("output") else batch.default_output(source)
        if out.resolve() == source.resolve():
            raise HTTPException(400, "output must not be the input file")
        if not out.parent.is_dir() or not os.access(out.parent, os.W_OK):
            raise HTTPException(400, f"cannot write to {out.parent}")
        concurrency = body.get("concurrency", 16)
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or not 1 <= concurrency <= 256:
            raise HTTPException(400, "concurrency must be an integer between 1 and 256")
        try:
            runner = fleet.ready(model_id)
        except KeyError as exc:
            raise HTTPException(404, f"model {model_id!r} is not loaded") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        early_exit = body.get("early_exit")
        levels = (runner.models_entry or {}).get("early_exit") or []
        if early_exit is not None and early_exit not in levels:
            raise HTTPException(400, f"early_exit must be one of {levels} for {model_id}" if levels else
                                f"{model_id} has no early exits (basal-serve --mode fast-exit)")
        try:
            return fleet.jobs.start("batch", model_id, f"{model_id} on {source.name}", batch.run_batch, fleet,
                                    model_id, source, out, concurrency, early_exit)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/batch")
    def cancel_batch():
        return fleet.jobs.cancel("batch")

    def batch_output():
        job = fleet.jobs.get("batch")
        state = job.state() if job is not None else None
        if not state or state["phase"] != "done" or not Path(state["result"]["output"]).is_file():
            raise HTTPException(404, "no finished batch")
        return Path(state["result"]["output"])

    @app.get("/api/batch/rows")
    def get_batch_rows(offset: int = 0, limit: int = 100):
        from . import batch

        return batch.rows(batch_output(), max(0, offset), max(1, min(limit, 1000)))

    @app.get("/api/batch/download")
    def download_batch():
        path = batch_output()
        return FileResponse(path, filename=path.name, media_type="application/x-ndjson")

    @app.get("/api/benchmarks")
    def get_benchmarks():
        return stats.snapshot()["benchmarks"]

    @app.get("/api/history")
    def get_history(model: str | None = None, source: str | None = None, q: str | None = None, limit: int = 100,
                    before: str | None = None, after: str | None = None):
        items = history.list(model=model or None, source=source or None, q=q or None,
                             limit=max(1, min(limit, 500)), before=before, after=after)
        return {"items": items, "models": history.models()}

    @app.get("/api/history/{record_id}")
    def get_history_record(record_id: str):
        record = history.get(record_id)
        if record is None:
            raise HTTPException(404, f"unknown history record {record_id!r}")
        return record

    @app.delete("/api/history")
    def clear_history():
        return {"cleared": history.clear()}

    @app.get("/api/logs")
    def get_logs(source: str = "basal-ui", after: int = 0):
        with fleet.lock:
            buffers = dict(fleet.logs)
            runners = dict(fleet.runners)
        if source not in buffers:
            raise HTTPException(404, f"unknown log source {source!r}")
        sources = [{"id": "basal-ui", "state": "ui"}]
        loaded = [key for key in runners if key in buffers]
        jobs = fleet.jobs.states()
        job_logs = [key for key in buffers if key.startswith("job:")]
        gone = [key for key in buffers if key != "basal-ui" and key not in runners and key not in job_logs]
        sources += [{"id": key, "state": runners[key].state} for key in loaded]
        sources += [{"id": key, "state": "running" if jobs.get(key[4:], {}).get("running") else "done"}
                    for key in job_logs]
        sources += [{"id": key, "state": "gone"} for key in gone]
        return {"sources": sources, "lines": buffers[source].since(after)}

    return app


def main():
    ap = argparse.ArgumentParser(description="basal browser UI: hardware check, model choice, playground")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    wildcard = a.host in ("0.0.0.0", "::")
    allowed = ["*"] if wildcard else ["127.0.0.1", "localhost", "::1", a.host]
    if wildcard:
        print("WARNING: basal-ui has no authentication; it is exposed to all network hosts.", flush=True)
    import uvicorn
    print(f"basal-ui -> http://{a.host}:{a.port}", flush=True)
    uvicorn.run(create_app(allowed_hosts=allowed), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
