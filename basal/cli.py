"""What the command-line tools share: counts that must be positive, and the API key of a server started with
--api-keys (basal-run, basal-loadtest and basal-calibrate --url send it as "Authorization: Bearer <key>")."""
import argparse
import os

API_KEY_ENV = "BASAL_API_KEY"


def positive(s):
    """An argparse type: an integer >= 1 (a count of 0 would wait for ever or divide by zero)."""
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"{s}: must be at least 1")
    return v


def add_api_key(ap):
    ap.add_argument("--api-key", dest="api_key", default=os.environ.get(API_KEY_ENV),
                    help="the API key of a basal-serve started with --api-keys, sent as \"Authorization: Bearer <key>\" "
                         f"(default: ${API_KEY_ENV}, which keeps the key out of the process list)")
    return ap


def headers(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def refused(url, status, api_key):
    """The message for a request the server refused for its key (HTTP 401 / 403), else None."""
    if status not in (401, 403):
        return None
    why = ("the API key was refused" if api_key else
           f"the server needs an API key: give --api-key KEY or set {API_KEY_ENV}")
    return f"{url}: HTTP {status}: {why}"
