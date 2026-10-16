"""Interleaved A/B of two running basal-serve engines: every round sends each request shape of the matrix to both,
alternating which goes first, with unique state markers; then bursts of single questions from 32 clients, alternating.
Thermal drift and the power limit hit both engines alike, which separate matrix runs do not guarantee.

  python docs/benchmarks/bench_ab.py --a gguf=http://127.0.0.1:8031/v1/systemone \
      --b strix=http://127.0.0.1:8030/v1/systemone --out results/2.0/ab.json
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

import bench_matrix as bm

SHAPES = ["S-q1", "S-q5", "M-q1", "M-q10", "L-q1", "L-q5", "XL-q1", "XL-q5", "XL-q20"]


async def main(a):
    from transformers import AutoTokenizer
    url = dict(x.split("=", 1) for x in (a.a, a.b))
    eng = list(url)
    cs = bm.cases(AutoTokenizer.from_pretrained(a.tokenizer))
    lat = {e: {s: [] for s in SHAPES} for e in eng}
    conc = {e: [] for e in eng}
    async with httpx.AsyncClient(timeout=3600) as c:
        async def post(e, b):
            t0 = time.perf_counter()
            r = await c.post(url[e], json=b)
            r.raise_for_status()
            return time.perf_counter() - t0
        for e in eng:
            for s in ("S-q1", "XL-q1"):
                await post(e, bm.mark(cs[s], f"warm-{e}"))
        for r in range(a.rounds):
            for i, s in enumerate(SHAPES):
                for e in (eng if (r + i) % 2 == 0 else eng[::-1]):
                    lat[e][s].append(await post(e, bm.mark(cs[s], f"ab{r}:{s}")))
            print(f"round {r} done", flush=True)
        sing, qs = bm.singles(), bm.pool()
        first = len(qs) - len(sing)  # the single questions close the pool, in the order of singles()
        for r in range(a.conc_rounds):
            for e in (eng if r % 2 == 0 else eng[::-1]):
                reqs = [bm.mark({"state": sing[i % len(sing)]["state"], "questions": {"q": qs[first + i % len(sing)][1]}},
                                f"ab-c{r}:{i}") for i in range(a.conc_requests)]
                sem = asyncio.Semaphore(a.clients)

                async def one(b):
                    async with sem:
                        await post(e, b)
                t0 = time.perf_counter()
                await asyncio.gather(*[one(b) for b in reqs])
                conc[e].append(a.conc_requests / (time.perf_counter() - t0))
    out = {"engines": url, "rounds": a.rounds, "clients": a.clients, "shapes": {},
           "conc_dec_s": {e: statistics.median(v) for e, v in conc.items()}, "raw": {"lat": lat, "conc": conc}}
    ea, eb = eng
    for s in SHAPES:
        m = {e: statistics.median(lat[e][s]) * 1000 for e in eng}
        out["shapes"][s] = {**{f"{e}_ms": m[e] for e in eng}, "speedup": m[ea] / m[eb]}
        print(f"{s:7s} {ea} {m[ea]:8.1f} ms  {eb} {m[eb]:8.1f} ms  {ea}/{eb} {m[ea] / m[eb]:.2f}x")
    print(f"{a.clients} clients dec/s:", {e: round(v, 2) for e, v in out["conc_dec_s"].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="label=url of the baseline engine")
    ap.add_argument("--b", required=True, help="label=url of the engine compared with it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Remek/basal-2.0-4.5B", help="tokenizer of the served model (basal-1.5 and "
                    "basal-2.0 share it)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--conc-rounds", type=int, default=2)
    ap.add_argument("--conc-requests", type=int, default=128)
    ap.add_argument("--clients", type=int, default=32)
    asyncio.run(main(ap.parse_args()))
