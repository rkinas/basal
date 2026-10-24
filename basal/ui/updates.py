"""Check the latest basal release with a small process-local cache."""
import threading
import time

import httpx

_URL = "https://api.github.com/repos/rkinas/basal/releases/latest"
_TTL = 6 * 60 * 60
_ERROR_TTL = 10 * 60  # a failed check is retried sooner than a successful one
_lock = threading.Lock()
_cached = (0.0, None, None)


def latest():
    global _cached
    with _lock:
        now, version, error = _cached
        if now and time.monotonic() - now < (_TTL if error is None else _ERROR_TTL):
            return version, error
        try:
            response = httpx.get(_URL, timeout=5)
            response.raise_for_status()
            version = response.json().get("tag_name", "").removeprefix("v") or None
            error = None if version else "release response has no tag_name"
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            version, error = None, str(exc)
        _cached = (time.monotonic(), version, error)
        return version, error
