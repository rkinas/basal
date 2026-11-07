"""Batch inference: send every line of a JSONL file to a running basal server and write one answer per line.

  basal-run --input basal/examples/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone

Two input formats are accepted (they can be mixed):
  simple   {"id": ..., "state": "...", "question": "...", "options": ["...", "..."], "type": "choice"|"noul"|"score",
            "gold": <index, optional>}      type defaults to "choice"; for "noul" the options are [yes-text, no-text]
  request  {"id": ..., "state": ..., "questions": {...}, "early_exit": ..., ...}   a full /v1/systemone request body:
           every field is sent as it is (media, facts, lang, think, ...) except the line's own "id", "gold", "group",
           "family" and fields whose name starts with "_"

Output (one line per input line, same order): {"id", "answers", "latency_ms"} plus, for simple lines, "prediction"
(index of the chosen option), "confidence" and "correct" when "gold" is given; a line that cannot be sent or read
back gets "error" instead, and the other lines go on. Lines are read and answers written as they go (at most a few
times --concurrency lines are held), into <output>.part, renamed to <output> at the end (an existing <output> is
replaced then): an interrupted run keeps the answers so far, in order, in <output>.part. A summary (accuracy if gold is present, latency) is printed at the end.

Resuming (2.8): the same command, run again after an interruption, goes on from <output>.part instead of starting
again: its answers are kept and only the lines after them are sent (a line cut short by the interruption, and errors
after the last answer, are sent again). <output>.part.json records the run: the --input file's size and sha256, the
options that make the requests (--preset, --lang, --early-exit) and the server's GET /v1/models. A .part of another
input, other options or another server is refused, as is one without that record; --restart replaces it.
--concurrency may differ.

A server that stops answering (2.8): after --stop-after (16) lines in a row that it did not answer (a connection
error, or HTTP 502, 503 or 504), the run stops instead of failing every line after them. Those lines are not written,
so <output>.part ends with the last answer, and the same command, run again once the server is back, goes on from
there. Fewer such lines between answers are each an "error" line, as any other failed line.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from .cli import add_api_key, headers, positive, refused
from .textio import open_text

LOCAL = ("id", "gold", "group", "family")  # a line's own fields (as basal-calibrate reads them), never sent
GONE = (502, 503, 504)  # no answer from the server now: a proxy without its server, starting / overloaded, timed out
STOP_AFTER = 16  # --stop-after: lines in a row the server did not answer, after which the run stops


def to_request(item, early_exit=None, preset=None, lang=None):
    if preset and "questions" not in item and "question" not in item:  # --preset: its questions over the state
        from .presets import questions
        item = dict(item, questions=questions(preset, lang, item["state"]))
    if "questions" in item:  # a full request: every field but the line's own
        body = {k: v for k, v in item.items() if k not in LOCAL and not k.startswith("_")}
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


class KeyRefused(Exception):
    """The server refused the API key (HTTP 401 / 403): every other line would fail the same way."""


class ServerGone(Exception):
    """The server answered none of the last --stop-after lines (a connection error, or HTTP 502 / 503 / 504): it stopped
    or cannot be reached, and every line after them would fail the same way."""


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
    at "\n" only (U+2028 inside a string stays in its line); UTF-8 with or without a BOM, or UTF-16 with its BOM."""
    with open_text(path, newline="\n") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as e:
                yield ValueError(f"line {n}: not JSON ({e})")
                continue
            yield item if isinstance(item, dict) else ValueError(f"line {n}: not a JSON object")


def fingerprint(path):
    """The --input file's size and sha256, or None when it is not a file (a pipe cannot be read again to check it)."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"bytes": p.stat().st_size, "sha256": h.hexdigest()}


def run_options(a):
    """The options that make the requests: a resumed run must send the same."""
    return {"preset": getattr(a, "preset", None), "lang": getattr(a, "lang", None), "early_exit": a.early_exit}


async def served_models(c, url):
    """GET /v1/models of the server behind --url (the served model, mode, calibration), or None when it does not
    answer it."""
    if "/v1/" not in url:
        return None
    try:
        r = await c.get(url.split("/v1/")[0] + "/v1/models")
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001  (not served, not JSON, down)
        return None


def resume_refused(record, want, part):
    """Why <output>.part cannot be continued by this run (its record against this run's), or None."""
    if record is None:
        return f"it has no record of its run ({part}.json, written by basal-run 2.8 and later)"
    if want["input"] is None:
        return "the --input is not a file, so it cannot be checked to be the same"
    if record.get("input") != want["input"]:
        return "another --input (its size or contents differ)"
    other = [k for k in want["options"] if record.get("options", {}).get(k) != want["options"][k]]
    if other:
        return "other options: " + ", ".join(f"--{k.replace('_', '-')}" for k in other)
    if record.get("models") is not None and want["models"] is not None and record["models"] != want["models"]:
        return "another server (its GET /v1/models differs: another model, mode or calibration)"
    return None


def kept(part, items):
    """The answers of <output>.part a resumed run keeps: its complete lines, each the answer of the input line of its
    place (its id), up to the last one without an error -> (bytes kept, [what the summary counts of each]). A line
    cut short by the interruption and the errors after the last answer are sent again."""
    good, end, upto = [], 0, (0, 0)  # (bytes, lines) up to the last answer
    with open(part, "rb") as f:
        for i, it in enumerate(items):
            line = f.readline()
            if not line.endswith(b"\n"):  # (the end, or a line cut short)
                break
            try:
                r = json.loads(line)
            except ValueError:
                break
            want = it.get("id", i) if isinstance(it, dict) else i
            if not isinstance(r, dict) or r.get("id") != want:
                break
            good.append({k: r[k] for k in ("error", "latency_ms", "correct") if k in r})
            end += len(line)
            if "error" not in r:
                upto = (end, len(good))
    return upto[0], good[: upto[1]]


async def run(a, transport=None):
    if same_file(a.input, a.output):
        raise SystemExit(f"--output {a.output} is the --input file: give another file (the answers would replace "
                         "the requests)")
    part = Path(str(a.output) + ".part")
    record_path = Path(str(part) + ".json")
    sem = asyncio.Semaphore(a.concurrency)
    window = collections.deque()  # the lines in flight, in input order (a bounded reorder buffer)
    n_items = n_err = 0
    lat, graded = [], []
    key = getattr(a, "api_key", None)
    async with httpx.AsyncClient(timeout=300, transport=transport, headers=headers(key)) as c:
        async def one(i, it):
            if isinstance(it, Exception):
                return failed(i, it)
            it.setdefault("id", i)
            async with sem:
                try:
                    body = to_request(it, a.early_exit, getattr(a, "preset", None), getattr(a, "lang", None))
                except (KeyError, TypeError, ValueError, AttributeError) as e:  # a malformed line: its own error
                    return failed(it["id"], f"not a valid line: {type(e).__name__}: {e}")
                gone = False  # the server did not answer this line (ServerGone after a run of them)
                try:
                    r = await c.post(a.url, json=body)
                    if refused(a.url, r.status_code, key):
                        raise KeyRefused(refused(a.url, r.status_code, key))
                    gone = r.status_code in GONE
                    try:
                        resp = r.json()
                    except ValueError as e:
                        resp = {"error": f"HTTP {r.status_code}" if gone else str(e) or type(e).__name__}
                except httpx.TransportError as e:  # (connection refused or reset, a timeout)
                    gone, resp = True, {"error": str(e) or type(e).__name__}
                except httpx.HTTPError as e:
                    resp = {"error": str(e) or type(e).__name__}
            try:
                out = summarise(it, resp)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
                out = failed(it["id"], f"unexpected answer: {type(e).__name__}: {e}",
                             resp.get("answers") if isinstance(resp, dict) else None)
            if gone and "error" in out:
                out["_gone"] = True
            return out

        def count(r):
            nonlocal n_items, n_err
            n_items += 1
            if "error" in r:
                n_err += 1
                return
            if r.get("latency_ms") is not None:
                lat.append(r["latency_ms"])
            if "correct" in r:
                graded.append(r["correct"])

        def write(f, r):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count(r)

        held = []  # the last lines the server did not answer, not written yet (written when an answer follows)
        stop_after = getattr(a, "stop_after", STOP_AFTER)

        def put(f, r):
            if r.pop("_gone", False) and stop_after:
                held.append(r)
                if len(held) >= stop_after:
                    raise ServerGone(f"the server answered none of the last {len(held)} lines (the last: "
                                     f"{r['error'][:200]}): it stopped or cannot be reached")
                return
            for h in held:
                write(f, h)
            held.clear()
            write(f, r)

        want = {"basal_run": 1, "input": fingerprint(a.input), "options": run_options(a),
                "models": await served_models(c, a.url)}
        skip = 0
        if part.exists() and not getattr(a, "restart", False):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
            except ValueError:
                record = None
            why = resume_refused(record, want, part)
            if why:
                raise SystemExit(f"basal-run: {part} is from another run: {why}. --restart replaces it (its answers "
                                 "are lost), or give another --output")
            size, old = kept(part, lines(a.input))
            with open(part, "r+b") as f:
                f.truncate(size)
            for r in old:
                count(r)
            skip = len(old)
            unchecked = " (the server not compared: no GET /v1/models)" if want["models"] is None else ""
            print(f"basal-run: resuming {part}: {skip} answers kept, going on from line {skip + 1}{unchecked}",
                  file=sys.stderr)
        else:
            tmp = Path(str(record_path) + ".tmp")
            tmp.write_text(json.dumps(want, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, record_path)

        done = False
        t0 = time.perf_counter()
        with open(part, "a" if skip else "w", encoding="utf-8") as f:
            try:  # (KeyRefused ends the run below, after the answers so far are kept)
                for i, it in enumerate(lines(a.input)):
                    if i < skip:  # (answered by the run this one resumes)
                        continue
                    window.append(asyncio.ensure_future(one(i, it)))
                    while len(window) >= 4 * a.concurrency or window and window[0].done():
                        put(f, await window.popleft())
                while window:
                    put(f, await window.popleft())
                for h in held:  # (fewer than --stop-after at the end: error lines, as any other)
                    write(f, h)
                done = True
            finally:
                for t in window:  # the lines in flight are dropped (an exception of a finished one is read)
                    if not t.done():
                        t.cancel()
                    elif not t.cancelled():
                        t.exception()
                if not done:
                    f.flush()
                    print(f"basal-run: stopped; the {n_items} answers so far are in {part}: the same command goes on "
                          "from there (--restart starts again)", file=sys.stderr)
        wall = time.perf_counter() - t0
    os.replace(part, a.output)
    record_path.unlink(missing_ok=True)
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
    ap.add_argument("--stop-after", dest="stop_after", type=int, default=STOP_AFTER,
                    help="stop the run after this many lines in a row that the server did not answer (a connection "
                         "error, HTTP 502 / 503 / 504), keeping <output>.part for the same command to go on from; 0: "
                         "never stop (every such line is an error line)")
    ap.add_argument("--restart", action="store_true",
                    help="start again: replace an <output>.part of an interrupted run (by default the same command "
                         "goes on from it, and a .part of another run is refused)")
    add_api_key(ap)
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
    if a.stop_after < 0:
        ap.error("--stop-after: 0 (never) or more lines")
    try:
        asyncio.run(run(a))
    except (KeyRefused, ServerGone) as e:
        raise SystemExit(f"basal-run: {e}") from None


if __name__ == "__main__":
    main()
