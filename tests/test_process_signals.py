"""SIGTERM (kill, docker stop, systemctl stop) of basal-serve runs the exit handlers, so the vLLM sidecar of think mode
and the merged adapter copy in /tmp do not outlive it, and the process still ends with status -15."""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from basal import adapter
from basal import server
from basal import think

REPO = str(Path(__file__).resolve().parent.parent)
posix = pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="SIGTERM status of a child")


def run_child(code, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (REPO, os.environ.get("PYTHONPATH")) if x))
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, timeout=60)


# basal-serve's main() with its argument parser replaced by a function that sends SIGTERM to the process: what happens
# on the way out is what `kill` does to a running basal-serve (the stubs of the test run stand in for torch).
SERVE_THEN_SIGTERM = """
import atexit, os, signal, sys
from basal import server
BODY
server.parser = lambda: os.kill(os.getpid(), signal.SIGTERM)
server.main()
sys.exit(7)  # not reached
"""


@posix
def test_sigterm_runs_the_exit_handlers_and_the_status_stays_minus_15(tmp_path):
    mark = tmp_path / "mark"
    r = run_child(SERVE_THEN_SIGTERM.replace("BODY", f"atexit.register(lambda: open({str(mark)!r}, 'w').write('ran'))"))
    assert mark.exists() and mark.read_text() == "ran", r.stderr[-300:]  # the exit handler ran
    assert r.returncode == -signal.SIGTERM  # killed by SIGTERM, not exit code 143


@posix
def test_a_handler_somebody_installed_stays(tmp_path):
    code = ("import signal\nfrom basal import signals\nmine = lambda s, f: None\nsignal.signal(signal.SIGTERM, mine)\n"
            "signals.exit_on_sigterm()\nassert signal.getsignal(signal.SIGTERM) is mine\n")
    r = run_child(code)
    assert r.returncode == 0, r.stderr[-300:]


def test_the_entry_point_of_basal_serve_arms_the_handler(monkeypatch):
    class Armed(Exception):
        pass

    def arm():
        raise Armed

    monkeypatch.setattr(server, "exit_on_sigterm", arm, raising=False)
    with pytest.raises(Armed):
        server.main()


def test_a_function_of_the_library_does_not_touch_the_signal_handler(monkeypatch, tmp_path):
    old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        monkeypatch.setattr(think.subprocess, "Popen", lambda *a, **k: type("P", (), {"poll": lambda s: None})())
        monkeypatch.setattr(think.atexit, "register", lambda *a, **k: None)
        monkeypatch.setattr(adapter.atexit, "register", lambda *a, **k: None)
        think.Sidecar("model", log=str(tmp_path / "vllm.log"))
        out = adapter._temp_dir()
        out.rmdir()
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, old)


@posix
def test_the_temp_dir_of_the_merged_adapter_is_removed_on_sigterm(tmp_path):
    note = tmp_path / "dir"
    r = run_child(SERVE_THEN_SIGTERM.replace("BODY", f"from basal import adapter\nopen({str(note)!r}, 'w').write(str(adapter._temp_dir()))"))
    assert note.exists(), r.stderr[-300:]
    assert not Path(note.read_text()).exists()  # the merged copy of the checkpoint is gone
    assert r.returncode == -signal.SIGTERM
