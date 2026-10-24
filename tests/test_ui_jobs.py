import threading
import time

import pytest

from basal.ui.jobs import Jobs


def wait_done(jobs, kind, timeout=3):
    deadline = time.monotonic() + timeout
    while jobs.states()[kind]["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    return jobs.states()[kind]


def test_one_job_per_kind_cancel_error_and_result():
    logs, events = {}, []
    jobs = Jobs(logs, lambda level, message: events.append((level, message)))
    release = threading.Event()

    def waits(job):
        job.progress("waiting", 1, 2)
        while not release.is_set():
            job.check()
            time.sleep(0.01)
        job.log.append("finished")
        return {"value": 1}

    assert jobs.start("bench", "m", "label", waits)["running"] is True
    with pytest.raises(RuntimeError, match="a bench job is already running"):
        jobs.start("bench", "m", "label", waits)
    jobs.start("other", None, "other", lambda job: {"ok": True})  # another kind runs alongside
    assert wait_done(jobs, "other")["result"] == {"ok": True}
    release.set()
    state = wait_done(jobs, "bench")
    assert (state["phase"], state["result"], state["error"]) == ("done", {"value": 1}, None)
    assert logs["job:bench"].last() == "finished"

    release.clear()
    jobs.start("bench", "m", "again", waits)
    assert jobs.cancel("bench")["running"] is True
    assert wait_done(jobs, "bench")["phase"] == "cancelled"
    assert jobs.cancel("missing") is None

    def fails(job):
        raise ValueError("bad input")

    jobs.start("bench", "m", "fails", fails)
    state = wait_done(jobs, "bench")
    assert (state["phase"], state["error"], state["running"]) == ("error", "bad input", False)
    assert logs["job:bench"].last() == "error: bad input" and ("error", "bench failed: fails: bad input") in events
