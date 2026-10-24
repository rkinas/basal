"""Batch inference: send every line of a JSONL file to a running basal server and write one answer per line.

  basal-run --input basal/examples/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone

Two input formats are accepted (they can be mixed):
  simple   {"id": ..., "state": "...", "question": "...", "options": ["...", "..."], "type": "choice"|"noul"|"score",
            "gold": <index, optional>}      type defaults to "choice"; for "noul" the options are [yes-text, no-text]
  request  {"id": ..., "state": ..., "questions": {...}, "early_exit": ...}   a full /v1/systemone request body

Output (one line per input line, same order): {"id", "answers", "latency_ms"} plus, for simple lines, "prediction"
(index of the chosen option), "confidence" and "correct" when "gold" is given. A summary (accuracy if gold is present,
latency) is printed at the end.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def to_request(item, early_exit=None):
    if "questions" in item:
        body = dict(item)
    else:
        t, opts = item.get("type", "choice"), item["options"]
        if t == "noul":
            if len(opts) != 2:
                raise ValueError("a noul item needs exactly two options: [yes-text, no-text]")
            crit = {"true": opts[0], "false": opts[1]}
        else:
            crit = {str(k): o for k, o in enumerate(opts)}
        q = {"type": t, "instructions": item["question"], "criteria": crit}
        if t != "noul":
            q["option_keys"] = "hide"  # keys are the placeholders "0", "1", ...; the option texts define the options
        body = {"state": item["state"], "questions": {"q": q}}
    if "questions" in item:
        if early_exit is not None:
            body["early_exit"] = early_exit
    elif early_exit:
        body["early_exit"] = early_exit
    return body


def summarise(item, resp):
    out = {"id": item.get("id"), "answers": resp.get("answers"), "latency_ms": resp.get("usage", {}).get("latency_ms")}
    if "questions" not in item and resp.get("answers"):
        a = resp["answers"]["q"]
        p = a["probabilities"]
        keys = ["true", "false"] if a["type"] == "noul" else [str(k) for k in range(len(item["options"]))]
        pred = max(range(len(keys)), key=lambda k: p[keys[k]])
        out.update(prediction=pred, option=item["options"][pred], confidence=a["confidence"])
        if "gold" in item:
            out["correct"] = pred == item["gold"]
    if "error" in resp:
        out["error"] = resp["error"]
    for key in ("status", "code", "request_id"):
        if key in resp:
            out[key] = resp[key]
    return out


async def run(a) -> dict:
    """Answer every line of a.input into a.output; -> the summary. Optional: a.headers (sent with every request) and
    a.progress(done, total), called after each answer (an exception it raises stops the run)."""
    items = [json.loads(line) for line in Path(a.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, it in enumerate(items):
        it.setdefault("id", i)
    results: list[dict] = [{} for _ in items]
    sem = asyncio.Semaphore(a.concurrency)
    progress, done = getattr(a, "progress", None), 0
    async with httpx.AsyncClient(timeout=300, headers=getattr(a, "headers", None)) as c:
        async def one(i, it):
            nonlocal done
            async with sem:
                try:
                    r = await c.post(a.url, json=to_request(it, a.early_exit))
                    try:
                        resp = r.json()
                    except ValueError:
                        resp = {"error": r.text or r.reason_phrase}
                    if not isinstance(resp, dict):
                        resp = {"error": "invalid response body"}
                    if not r.is_success:
                        resp.setdefault("error", r.text or r.reason_phrase)
                        resp["status"] = r.status_code
                        resp["request_id"] = r.headers.get("x-request-id")
                except (httpx.HTTPError, ValueError) as e:
                    resp = {"error": str(e)}
                results[i] = summarise(it, resp)
                done += 1
                if progress:
                    progress(done, len(items))
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
        wall = time.perf_counter() - t0
    with open(a.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in results if "error" not in r]
    lat = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
    s = {"items": len(items), "errors": len(items) - len(ok), "seconds": round(wall, 2),
         "items_per_s": round(len(items) / wall, 1) if wall else None,
         "server_latency_p50_ms": round(statistics.median(lat), 2) if lat else None}
    graded = [r for r in ok if "correct" in r]
    if graded:
        s["accuracy"] = round(sum(r["correct"] for r in graded) / len(graded), 4)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL file")
    ap.add_argument("--output", required=True, help="JSONL file with one answer per input line")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    ap.add_argument("--concurrency", type=int, default=16, help="requests in flight (the server batches them)")
    ap.add_argument("--early-exit", dest="early_exit", default=None, help="e.g. 0.99 (server mode fast-exit)")
    print(json.dumps(asyncio.run(run(ap.parse_args()))))


if __name__ == "__main__":
    main()
