"""HTTP load test of a running basal server: sequential latency and throughput with concurrent clients.

  basal-loadtest --url http://127.0.0.1:8000/v1/systemone --questions basal/examples/questions.jsonl
  basal-loadtest --url http://127.0.0.1:8000/v1/systemone --synthetic 1500 --concurrency 16   # ~1.5k-token states
  basal-loadtest --url http://127.0.0.1:8000/v1/systemone --synthetic 1500 --determinism 100

--determinism N asks N requests one at a time, then the same N again mixed with N other requests at --concurrency, and
compares the returned probabilities exactly (a server started with --deterministic must return identical answers).
"""
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx

from . import synth
from .bench import DEFAULT_QUESTIONS, load_questions


def body(q, early_exit=None, questions=1):
    """One request: the item's question, or (questions > 1) that many fixed questions over the item's state (SOAM)."""
    if questions > 1:
        from .identity import QUESTIONS
        from .prompt import lang_of
        qs = QUESTIONS[lang_of(q["state"])][:questions]
    else:
        qs = [(q["question"], q["options"])]
    b = {"state": q["state"], "questions": {
        f"q{i}": {"type": "choice", "instructions": t, "option_keys": "hide",
                  "criteria": {f"option_{k + 1}": o for k, o in enumerate(opts)}} for i, (t, opts) in enumerate(qs)}}
    if early_exit:
        b["early_exit"] = early_exit
    return b


def request_bodies(a, n):
    """n request bodies: synthetic items, or the simple items of --questions (a small file reused); SystemExit when
    there is none (a file of full requests only, or an empty one), which would loop for ever doubling no bodies."""
    qs = synth.items(n, a.synthetic, a.seed, long_frac=a.long_frac) if a.synthetic else load_questions(a.questions, n)
    bodies = [body(q, a.early_exit, a.questions_per_request) for q in qs]
    if not bodies and n > 0:
        raise SystemExit(f"basal-loadtest: no simple items in {' '.join(map(str, a.questions))} (lines with \"state\", "
                         "\"question\" and \"options\"; full requests with \"questions\" are not used)")
    while len(bodies) < n:  # small question files are reused
        bodies += bodies
    return bodies


def nonnegative(s):
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f"{s}: must be 0 or more")
    return v


async def run(a):
    if a.n_seq < 1:
        raise SystemExit("basal-loadtest: --n-seq must be at least 1")
    # warm-up, sequential and concurrent phases ask different requests (as far as the questions go), so an engine with
    # a prefix cache across requests (vLLM, SGLang, llama.cpp) cannot answer a timed request from an earlier phase
    n = 10 + a.n_seq + a.n_conc
    bodies = request_bodies(a, n)
    async with httpx.AsyncClient(timeout=300) as c:
        for b in bodies[:10]:  # warm-up
            (await c.post(a.url, json=b)).raise_for_status()
        lat = []
        for b in bodies[10: 10 + a.n_seq]:
            t0 = time.perf_counter()
            r = await c.post(a.url, json=b)
            lat.append(time.perf_counter() - t0)
            r.raise_for_status()
        lat.sort()
        sem = asyncio.Semaphore(a.concurrency)

        async def one(b):
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await c.post(a.url, json=b)
                except httpx.HTTPError:  # a timeout or a refused connection is an error, it does not end the run
                    return time.perf_counter() - t0, 0
                return time.perf_counter() - t0, r.status_code

        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(b) for b in bodies[10 + a.n_seq: n]])
        wall = time.perf_counter() - t0
    ok = sorted(x for x, s in res if s == 200)  # p95: the nearest rank, ceil(0.95 n)-th smallest (as basal-bench)
    out = dict(url=a.url, p50_ms=statistics.median(lat) * 1000, p95_ms=lat[math.ceil(0.95 * len(lat)) - 1] * 1000,
               concurrency=a.concurrency, decisions_per_s=len(ok) / wall, errors=len(res) - len(ok))
    if a.questions_per_request > 1:  # a request is several decisions; the rates above count requests
        out.update(questions_per_request=a.questions_per_request,
                   questions_per_s=len(ok) * a.questions_per_request / wall)
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


async def determinism(a, bodies):
    """The same requests alone and under load -> identical answers, max |dp|."""
    n = a.determinism
    probe, other = bodies[:n], bodies[n: 2 * n]

    def probs(r):
        r.raise_for_status()
        return [list(ans["probabilities"].values()) for ans in r.json()["answers"].values()]
    async with httpx.AsyncClient(timeout=300) as c:
        alone = [probs(await c.post(a.url, json=b)) for b in probe]
        sem = asyncio.Semaphore(a.concurrency)

        async def one(b):
            async with sem:
                return await c.post(a.url, json=b)
        mixed = [x for pair in zip(probe, other) for x in pair]
        res = await asyncio.gather(*[one(b) for b in mixed])
    loaded = [probs(r) for r in res[::2]]
    diffs = [max(abs(x - y) for pa, qa in zip(p, q) for x, y in zip(pa, qa)) for p, q in zip(alone, loaded)]
    top = lambda ps: max(range(len(ps)), key=ps.__getitem__)  # noqa: E731
    flips = sum(top(pa) != top(qa) for p, q in zip(alone, loaded) for pa, qa in zip(p, q))
    out = dict(url=a.url, determinism=dict(requests=n, identical=sum(d == 0.0 for d in diffs), max_abs_diff=max(diffs),
                                           decision_flips=flips, concurrency=a.concurrency))
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    ap.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS, help="JSONL file(s) with simple items")
    ap.add_argument("--synthetic", type=int, default=None,
                    help="synthetic items with states of about this many tokens instead of --questions (basal.synth)")
    ap.add_argument("--long-frac", dest="long_frac", type=float, default=0.0,
                    help="--synthetic: share of long states (3k-10k tokens)")
    ap.add_argument("--seed", type=int, default=0,
                    help="--synthetic: the items' seed (a repeated run with another seed asks states that a server "
                         "with a prefix cache has not seen)")
    ap.add_argument("--questions-per-request", dest="questions_per_request", type=int, default=1, choices=range(1, 6),
                    help="2-5: that many fixed questions over each item's state in one request (SOAM)")
    ap.add_argument("--n-seq", dest="n_seq", type=int, default=200)
    ap.add_argument("--n-conc", dest="n_conc", type=nonnegative, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--early-exit", dest="early_exit", default=None)
    ap.add_argument("--determinism", type=nonnegative, default=0,
                    help="N: compare N answers given alone and under load (instead of the load test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.concurrency < 1:
        raise SystemExit("basal-loadtest: --concurrency must be at least 1")
    if a.determinism:
        return asyncio.run(determinism(a, request_bodies(a, 2 * a.determinism)))
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
