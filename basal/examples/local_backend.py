"""One decision from a running basal server, for code that used a hosted LLM as its decision backend.

  python basal/examples/local_backend.py --url http://127.0.0.1:8000 --state "Order arrived damaged, money back please"
"""
import argparse

import httpx


def ask_choice(url, state, instructions, options, max_error=None, timeout=60):
    """-> (option key, confidence); the key is None when the server marks the answer abstain (needs max_error)."""
    q = {"type": "choice", "instructions": instructions, "criteria": options}
    if max_error is not None:
        q["max_error"] = max_error
    r = httpx.post(url.rstrip("/") + "/v1/systemone", json={"state": state, "questions": {"q": q}}, timeout=timeout)
    r.raise_for_status()
    ans = r.json()["answers"]["q"]
    if ans.get("abstain"):
        return None, ans["confidence"]
    return ans["choice"], ans["confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--state", required=True)
    ap.add_argument("--max-error", type=float)
    a = ap.parse_args()
    print(ask_choice(a.url, a.state, "What does the customer want?",
                     {"refund": "money back", "replace": "a new item", "info": "information only"}, a.max_error))


if __name__ == "__main__":
    main()
