"""Regression tests for fix/process-signals (Z-1089)."""
import shutil
import signal
import types

import pytest

from basal import adapter
from basal import think


@pytest.fixture
def default_sigterm():
    old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGTERM, old)


def test_e1_a_sidecar_turns_sigterm_into_a_system_exit(monkeypatch, tmp_path, default_sigterm):
    monkeypatch.setattr(think.subprocess, "Popen", lambda *a, **k: types.SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(think.atexit, "register", lambda *a, **k: None)
    think.Sidecar("model", log=str(tmp_path / "vllm.log"))
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)  # not SIG_DFL, which ends the process without atexit
    with pytest.raises(SystemExit) as e:
        handler(signal.SIGTERM, None)
    assert e.value.code == 128 + signal.SIGTERM


def test_e1_a_handler_somebody_installed_stays(default_sigterm):
    from basal import signals
    mine = lambda signum, frame: None  # noqa: E731
    signal.signal(signal.SIGTERM, mine)
    signals.exit_on_sigterm()
    assert signal.getsignal(signal.SIGTERM) is mine


def test_e2_the_temp_dir_of_the_merged_adapter_survives_sigterm_only_as_a_system_exit(monkeypatch):
    old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    monkeypatch.setattr(adapter.atexit, "register", lambda *a, **k: None)
    try:
        out = adapter._temp_dir()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)  # not SIG_DFL: atexit (shutil.rmtree) runs on the way out
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, old)
        shutil.rmtree(out, True)
