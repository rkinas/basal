"""Background jobs of basal-ui (benchmark, calibration, batch, ...): one running job per kind, each in its own thread,
its progress in /api/status and its output in the Logs view (source job:<kind>)."""
import threading
import time
from collections import deque

import httpx


class LogBuffer:
    """The last lines of one source (a basal-serve child, a job or basal-ui's own events), numbered for incremental
    reads."""

    def __init__(self, maxlen=1000):
        self.lines = deque(maxlen=maxlen)
        self.seq = 0
        self.lock = threading.Lock()

    def append(self, text):
        with self.lock:
            self.seq += 1
            self.lines.append({"seq": self.seq, "ts": time.time(), "text": text})

    def since(self, seq=0):
        with self.lock:
            return [line for line in self.lines if line["seq"] > seq]

    def last(self):
        with self.lock:
            return self.lines[-1]["text"] if self.lines else None

    def text(self):
        with self.lock:
            return "\n".join(line["text"] for line in self.lines)


class Cancelled(Exception):
    """A job's target stops early: Job.cancel was set."""


class Job:
    """What /api/status shows of one job; the target reports through progress() and polls cancel / check()."""

    def __init__(self, kind, model, label, log):
        self.kind, self.model, self.label, self.log = kind, model, label, log
        self.phase, self.done, self.total, self.error = "starting", 0, 0, None
        self.running, self.started, self.result = True, time.time(), None
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def progress(self, phase, done=0, total=0):
        with self.lock:
            self.phase, self.done, self.total = phase, done, total

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def finish(self, phase, result=None, error=None):
        with self.lock:
            self.phase, self.result, self.error, self.running = phase, result, error, False

    def state(self) -> dict:
        with self.lock:
            return {"kind": self.kind, "model": self.model, "label": self.label, "phase": self.phase,
                    "done": self.done, "total": self.total, "error": self.error, "running": self.running,
                    "started": self.started, "result": self.result}


class Jobs:
    """The last job of each kind; `logs` is the fleet's log buffers (a job's output goes to logs["job:<kind>"])."""

    def __init__(self, logs: dict, event=None):
        self.logs = logs
        self.event = event or (lambda level, message: None)
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}

    def start(self, kind, model, label, target, *args) -> dict:
        """Run target(job, *args) in a daemon thread; its return value is the job's result. RuntimeError when a job of
        the kind is running."""
        with self.lock:
            current = self.jobs.get(kind)
            if current is not None and current.running:
                raise RuntimeError(f"a {kind} job is already running")
            log = LogBuffer()
            self.logs[f"job:{kind}"] = log
            job = self.jobs[kind] = Job(kind, model, label, log)

        def run():
            try:
                result = target(job, *args)
                job.finish("done_with_warnings" if isinstance(result, dict) and result.get("warnings") else "done",
                           result=result)
                self.event("info", f"{kind} finished: {label}")
            except Cancelled:
                job.finish("cancelled")
                self.event("info", f"{kind} cancelled: {label}")
            except (Exception, SystemExit) as exc:  # noqa: BLE001 — shown in the UI and the job's log (basal's
                # engines and CLIs refuse with SystemExit)
                message = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    response = exc.response
                    message = f"HTTP {response.status_code}: {response.text}"
                    if response.headers.get("x-request-id"):
                        message += f" [request {response.headers['x-request-id']}]"
                job.log.append(f"error: {message}")
                job.finish("error", error=message)
                self.event("error", f"{kind} failed: {label}: {exc}")

        self.event("info", f"{kind} started: {label}")
        threading.Thread(target=run, name=f"basal-ui-{kind}", daemon=True).start()
        return job.state()

    def get(self, kind) -> Job | None:
        with self.lock:
            return self.jobs.get(kind)

    def cancel(self, kind) -> dict | None:
        job = self.get(kind)
        if job is None:
            return None
        job.cancel.set()
        return job.state()

    def cancel_all(self):
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            job.cancel.set()

    def states(self) -> dict[str, dict]:
        with self.lock:
            jobs = dict(self.jobs)
        return {kind: job.state() for kind, job in jobs.items()}

    def running(self) -> bool:
        with self.lock:
            return any(job.running for job in self.jobs.values())
