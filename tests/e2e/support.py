"""Helpers of the end-to-end tests (tests/e2e/conftest.py): a basal-ui process and its HTTP API, model picking."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "basal" / "examples"
LOAD_TIMEOUT = float(os.environ.get("BASAL_E2E_LOAD_TIMEOUT", "1800"))  # --mode fast compiles for minutes
DOWNLOAD = os.environ.get("BASAL_E2E_DOWNLOAD") == "1"
FALLBACK_MODEL = "Remek/basal-2.0-mini"  # bf16: runnable on CUDA (fast), Apple Silicon (mlx) and the CPU (eager)
GiB = 1024**3


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class UI:
    """A basal-ui process: its URL, settings directory and log, and JSON helpers that fail with the server's answer."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = self.root / "settings.json"
        self.port, self.proc = None, None
        self.http = None

    def start(self):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.http = httpx.Client(base_url=self.url, timeout=600)
        log = open(self.root / "basal-ui.log", "a")  # noqa: SIM115 — kept open for the process
        env = {**os.environ, "BASAL_UI_CONFIG": str(self.settings)}
        self.proc = subprocess.Popen([sys.executable, "-m", "basal.ui.app", "--port", str(self.port)], cwd=ROOT,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"basal-ui exited ({self.proc.returncode}): {self.root / 'basal-ui.log'}")
            try:
                if self.http.get("/api/status").status_code == 200:
                    if os.environ.get("BASAL_E2E_MODELS_DIR"):
                        self.put("/api/settings", {"models_dir": os.environ["BASAL_E2E_MODELS_DIR"]})
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        raise RuntimeError("basal-ui did not answer within 180 s")

    def child_pids(self):
        """The basal-serve children of this basal-ui (the private one of a calibration too)."""
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,command="], capture_output=True, text=True, check=False).stdout
        pids = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[1] == str(self.proc.pid) and "basal.server" in parts[2]:
                pids.append(int(parts[0]))
        return pids

    def stop(self):
        """Ctrl+C (SIGINT): basal-ui unloads every model and exits; -> the children still alive afterwards."""
        if self.proc is None or self.proc.poll() is not None:
            return []
        pids = self.child_pids()
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise AssertionError("basal-ui did not exit within 120 s of SIGINT") from None
        time.sleep(1)
        self.http.close()
        return [pid for pid in pids if alive(pid)]

    # --- HTTP --------------------------------------------------------------------------------------------------
    def call(self, method, path, expect=(200, 201, 202), **kw):
        r = self.http.request(method, path, **kw)
        if r.status_code not in expect:
            raise AssertionError(f"{method} {path} -> {r.status_code} (expected {expect}): {r.text[:800]}")
        return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else r

    def get(self, path, **kw):
        return self.call("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.call("POST", path, json=body, **kw)

    def put(self, path, body=None, **kw):
        return self.call("PUT", path, json=body, **kw)

    def delete(self, path, **kw):
        return self.call("DELETE", path, **kw)

    def detail(self, method, path, body, status):
        """The {"detail"} message of a refused request (asserting its status)."""
        return self.call(method, path, expect=(status,), json=body)["detail"]

    # --- models and jobs ----------------------------------------------------------------------------------------
    def status(self):
        return self.get("/api/status")

    def model(self, model_id):
        return next((m for m in self.status()["models"] if m["id"] == model_id), None)

    def wait_model(self, model_id, timeout=LOAD_TIMEOUT, appear=False):
        """The model's row once it is ready or failed (None when it is not loaded, unless `appear`: then it is waited
        for, as an auto-load that has not started yet)."""
        deadline, m = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            m = self.model(model_id)
            if (m is None and not appear) or (m is not None and m["state"] in ("ready", "error")):
                return m
            time.sleep(1)
        raise AssertionError(f"{model_id} did not finish loading within {timeout:.0f} s: {m}")

    def load(self, model_id, options=None):
        """The model ready with exactly these load options (a model loaded with others is restarted)."""
        options = options or {}
        m = self.model(model_id)
        if m and m["state"] == "ready" and m["options"] == options:
            return m
        if m and m["state"] in ("ready", "error"):
            self.post("/api/restart", {"id": model_id, "options": options})
        else:
            self.post("/api/load", {"id": model_id, "options": options})
        m = self.wait_model(model_id)
        assert m and m["state"] == "ready", f"{model_id} did not load: {m and m['error']}"
        return m

    def unload(self, model_id):
        if self.model(model_id):
            self.post("/api/unload", {"id": model_id})

    def decide(self, model_id, body, expect=(200,)):
        return self.call("POST", "/api/decide", expect=expect, json={"model": model_id, **body})

    def wait_job(self, kind, timeout=LOAD_TIMEOUT):
        deadline, job = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            job = self.status()["jobs"].get(kind)
            if job and not job["running"]:
                return job
            time.sleep(0.5)
        raise AssertionError(f"the {kind} job did not finish within {timeout:.0f} s: {job}")

    def run_job(self, path, body, kind, timeout=LOAD_TIMEOUT):
        """Start a job, wait for it; -> its final state, which must be done."""
        self.post(path, body)
        job = self.wait_job(kind, timeout)
        assert job["phase"] in ("done", "done_with_warnings"), f"{kind} ended {job['phase']}: {job['error']}"
        return job


def catalog(ui):
    return ui.get("/api/models")["models"]


def ensure_downloaded(ui, model_id):
    """Download a model through basal-ui's Download button (BASAL_E2E_DOWNLOAD=1) and wait for it."""
    ui.post("/api/download", {"id": model_id}, expect=(202, 409))
    deadline = time.monotonic() + LOAD_TIMEOUT
    while time.monotonic() < deadline:
        record = ui.status()["downloads"].get(model_id.split(":")[0])
        if record is None:
            break
        assert record["error"] is None, record["error"]
        time.sleep(2)
    return next(m for m in catalog(ui) if m["id"] == model_id)


def pick(ui, env, predicate, why, prefer=lambda m: 0):
    """The model named by `env`, else the smallest downloaded one that satisfies `predicate` (with `prefer` first);
    BASAL_E2E_DOWNLOAD=1 downloads FALLBACK_MODEL when none is on disk; else the dependent tests are skipped."""
    models = catalog(ui)
    named = os.environ.get(env)
    if named:
        m = next((m for m in models if m["id"] == named), None)
        if m is None or not m["downloaded"]:
            pytest.fail(f"{env}={named} is not a downloaded model of the catalog")
        return m
    found = sorted((m for m in models if m["downloaded"] and predicate(m)),
                   key=lambda m: (prefer(m), m["weights_bytes"] or 0))
    if found:
        return found[0]
    fallback = next((m for m in models if m["id"] == FALLBACK_MODEL), None)
    if DOWNLOAD and fallback is not None and predicate({**fallback, "downloaded": True}):
        return ensure_downloaded(ui, FALLBACK_MODEL)
    pytest.skip(f"{why}: download one or set {env} (or BASAL_E2E_DOWNLOAD=1)")


def text_ok(m):
    return m["runnable"] and not m["experimental"] and m["family"] in ("basal-2.0", "basal-1.5", "basal-1.0")


def advanced_requests():
    return [json.loads(line) for line in (EXAMPLES / "advanced.jsonl").read_text().splitlines() if line.strip()]


CHOICE = {"state": "Klient pisze, że karta została zablokowana po trzech błędnych PIN-ach i prosi o odblokowanie.",
          "questions": {"dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
                                 "criteria": {"cards": "karty płatnicze", "loans": "kredyty",
                                              "online": "bankowość internetowa"}}}}


def long_state(tokens):
    """A Polish state longer than `tokens` tokens (about 5 characters per token, 6 taken)."""
    sentence = "Umowa najmu lokalu mieszkalnego przy ulicy Polnej w Krakowie została zawarta na czas określony. "
    return (sentence * (tokens * 6 // len(sentence) + 1))[: tokens * 6]
