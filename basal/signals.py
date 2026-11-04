"""SIGTERM that runs the exit handlers and still ends the process as a SIGTERM: for the entry point of a command."""
import atexit
import os
import signal
import threading


def _terminate(signum, frame):
    """The exit handlers (a child process started with atexit.register(close), a temporary directory removed by atexit)
    get their turn; then the default action returns and the signal is sent again, so the process ends the way an
    unhandled SIGTERM ends it: status -15 for its parent, which is what systemd, docker and a shell wait for."""
    try:
        atexit._run_exitfuncs()
    finally:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def exit_on_sigterm():
    """Python's default SIGTERM ends the process at once: no atexit handler runs, so a child process started with
    atexit.register(close) or a temporary directory removed by atexit outlives a `kill` / `docker stop` / `systemctl
    stop`. Called from the entry point of a command (basal-serve), never from a function of the library: a program
    that uses basal as a library keeps its own signal handling. A handler somebody installed already stays; only the
    main thread can install one."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        return
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _terminate)
