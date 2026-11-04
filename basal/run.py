"""Batch inference: send every line of a JSONL file to a running basal server and write one answer per line.

  basal-run --input basal/examples/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone

Two input formats are accepted (they can be mixed):
  simple   {"id": ..., "state": "...", "question": "...", "options": ["...", "..."], "type": "choice"|"noul"|"score",
            "gold": <index, optional>}      type defaults to "choice"; for "noul" the options are [yes-text, no-text]
  request  {"id": ..., "state": ..., "questions": {...}, "early_exit": ...}   a full /v1/systemone request body

Output (one line per input line, same order): {"id", "answers", "latency_ms"} plus, for simple lines, "prediction"
(index of the chosen option), "confidence" and "correct" when "gold" is given; a line that cannot be sent or read
back gets "error" instead, and the other lines go on. Lines are read and answers written as they go (at most a few
times --concurrency lines are held), into <output>.part, renamed to <output> at the end (an existing <output> is
replaced then): an interrupted run keeps the answers so far, in order, in <output>.part. A summary (accuracy if gold is present, latency) is printed at the end.
"""
import argparse
import asyncio
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from .cli import positive


def to_request(item, early_exit=None, preset=None, lang=None):
    if preset and "questions" not in item and "question" not in item:  # --preset: its questions over the state
        from .presets import questions
        item = dict(item, questions=questions(preset, lang, item["state"]))
    if "questions" in item:
        body = {k: item[k] for k in ("state", "questions", "early_exit") if k in item}
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
    if early_exit and "early_exit" not in body:
        body["early_exit"] = early_exit
    return body


def summarise(item, resp):
    out = {"id": item.get("id"), "answers": resp.get("answers"), "latency_ms": resp.get("usage", {}).get("latency_ms")}
    if "question" in item and "questions" not in item and resp.get("answers"):  # a simple line (a --preset line has neither)
        a = resp["answers"]["q"]
        p = a["probabilities"]
        keys = ["true", "false"] if a["type"] == "noul" else [str(k) for k in range(len(item["options"]))]
        pred = max(range(len(keys)), key=lambda k: p[keys[k]])
        out.update(prediction=pred, option=item["options"][pred], confidence=a["confidence"])
        if "gold" in item:
            out["correct"] = pred == item["gold"]
    if "error" in resp:
        out["error"] = resp["error"]
    return out


def failed(i, error, answers=None):
    return {"id": i, "answers": answers, "latency_ms": None, "error": str(error)}


def same_file(a, b):
    a, b = Path(a), Path(b)
    try:
        return a.resolve() == b.resolve() or b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def lines(path):
    """The item of every non-blank line (or the error of a line that is not a JSON object), read as they go and split
    at "\n" only (U+2028 inside a string stays in its line)."""
    with open(path, encoding="utf-8", newline="\n") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as e:
                yield ValueError(f"line {n}: not JSON ({e})")
                continue
            yield item if isinstance(item, dict) else ValueError(f"line {n}: not a JSON object")


async def run(a, transport=None):
    if same_file(a.input, a.output):
        raise SystemExit(f"--output {a.output} is the --input file: give another file (the answers would replace "
                         "the requests)")
    part = Path(str(a.output) + ".part")
    sem = asyncio.Semaphore(a.concurrency)
    window = collections.deque()  # the lines in flight, in input order (a bounded reorder buffer)
    n_items = n_err = 0
    lat, graded = [], []
    async with httpx.AsyncClient(timeout=300, transport=transport) as c:
        async def one(i, it):
            if isinstance(it, Exception):
                return failed(i, it)
            it.setdefault("id", i)
            async with sem:
                try:
                    body = to_request(it, a.early_exit, getattr(a, "preset", None), getattr(a, "lang", None))
                except (KeyError, TypeError, ValueError, AttributeError) as e:  # a malformed line: its own error
                    return failed(it["id"], f"not a valid line: {type(e).__name__}: {e}")
                try:
                    r = await c.post(a.url, json=body)
                    resp = r.json()
                except (httpx.HTTPError, ValueError) as e:
                    resp = {"error": str(e) or type(e).__name__}
            try:
                return summarise(it, resp)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
                return failed(it["id"], f"unexpected answer: {type(e).__name__}: {e}",
                              resp.get("answers") if isinstance(resp, dict) else None)

        def put(f, r):
            nonlocal n_items, n_err
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_items += 1
            if "error" in r:
                n_err += 1
                return
            if r.get("latency_ms") is not None:
                lat.append(r["latency_ms"])
            if "correct" in r:
                graded.append(r["correct"])

        done = False
        t0 = time.perf_counter()
        with open(part, "w", encoding="utf-8") as f:
            try:
                for i, it in enumerate(lines(a.input)):
                    window.append(asyncio.ensure_future(one(i, it)))
                    while len(window) >= 4 * a.concurrency or window and window[0].done():
                        put(f, await window.popleft())
                while window:
                    put(f, await window.popleft())
                done = True
            finally:
                for t in window:
                    t.cancel()
                if not done:
                    f.flush()
                    print(f"basal-run: stopped; the {n_items} answers so far are in {part}", file=sys.stderr)
        wall = time.perf_counter() - t0
    os.replace(part, a.output)
    s = {"items": n_items, "errors": n_err, "seconds": round(wall, 2),
         "items_per_s": round(n_items / wall, 1) if wall else None,
         "server_latency_p50_ms": round(statistics.median(lat), 2) if lat else None}
    if graded:
        s["accuracy"] = round(sum(graded) / len(graded), 4)
    print(json.dumps(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="JSONL file")
    ap.add_argument("--output", help="JSONL file with one answer per input line (an existing file is replaced; not the "
                                     "--input file)")
    ap.add_argument("--preset", default=None, help="ready questions for lines with a state only (--list-presets)")
    ap.add_argument("--lang", choices=["pl", "en"], default=None, help="--preset wording (default: the state's)")
    ap.add_argument("--list-presets", dest="list_presets", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    ap.add_argument("--concurrency", type=positive, default=16, help="requests in flight (the server batches them)")
    ap.add_argument("--early-exit", dest="early_exit", default=None, help="e.g. 0.99 (server mode fast-exit)")
    a = ap.parse_args()
    if a.list_presets:
        from .presets import names, questions
        for n in names():
            print(n + ": " + ", ".join(questions(n, "en")))
        return
    if not a.input or not a.output:
        ap.error("--input and --output are required")
    if a.preset:
        from .presets import names
        if a.preset not in names():
            ap.error(f"--preset {a.preset}: one of {', '.join(names())}")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
