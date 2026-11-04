"""SIGTERM that ends the process like SystemExit does (atexit handlers, finally blocks), not the way the OS default does."""
import signal
import sys
import threading


def exit_on_sigterm():
    """Python's default SIGTERM ends the process at once: no atexit handler runs, so a child process started with
    atexit.register(close) or a temporary directory removed by atexit outlives a `kill` / `docker stop` / `systemctl
    stop`. A handler that raises SystemExit gives them their turn. A handler somebody installed already stays; only the
    main thread can install one."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        return
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
