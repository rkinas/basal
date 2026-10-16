"""Request-shape benchmark of a running basal-serve: latency by state length x questions per request, a concurrency
sweep, and every answer (for agreement with a reference run).

  python docs/benchmarks/bench_matrix.py --url http://127.0.0.1:8000/v1/systemone --label gguf

States S / M / L / XL (~100 / ~500 / ~2,000 / ~6,000 tokens) are built from the bundled examples; questions come from a
fixed mixed pool (choice, noul, score, multi from complex.jsonl, then the bundled single questions as choice). Pass 0
sends the requests as they are and keeps the answers; timing passes put a marker in front of the state so that no
engine can serve a repeat from a prompt cache (the fixed system-prompt start is still shared, as in real traffic).
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

EX = Path(__file__).resolve().parents[2] / "basal" / "examples"
KS = [1, 2, 5, 10, 20]
TARGET = {"S": 100, "M": 500, "L": 2000, "XL": 6000}


def jl(name):
    return [json.loads(x) for x in (EX / name).read_text().splitlines() if x.strip()]


def singles():
    return [q for f in ("questions", "choice", "noul", "score") for q in jl(f"{f}.jsonl")]


def pool():
    """Questions in a fixed order that mixes the types: the complex questions first (choice / noul / score / multi /
    act), then the single questions as choice questions."""
    qs = []
    for c in jl("complex.jsonl"):
        for n, q in c["questions"].items():
            qs.append((f"{c['id'][:5]}-{n}", q))
    qs[2:2] = [  # a multi-label question (one yes/no branch per label) and a cost-based action, early in the pool
        ("multi-topics", {"type": "multi", "instructions": "Czego dotyczy ten tekst?",
                          "criteria": {"dates": "Terminy i daty", "money": "Kwoty i płatności", "people": "Konkretne osoby",
                                       "law": "Przepisy prawa"}}),
        ("act-close", {"type": "act", "instructions": "Czy sprawę można zamknąć bez udziału człowieka?",
                       "criteria": {"true": "Tak", "false": "Nie"}, "costs": {"wrong": 10, "defer": 1}})]
    for k, s in enumerate(singles()):
        qs.append((f"s{k:02d}", {"type": "choice", "instructions": s["question"], "option_keys": "hide",
                                 "criteria": {f"o{j + 1}": o for j, o in enumerate(s["options"])}}))
    return qs


def states(tok):
    """State texts of about TARGET tokens: the single-question states joined (repeated for XL), cut at a paragraph."""
    paras = [s["state"] for s in singles()]
    out = {}
    for name, n in TARGET.items():
        text, k = "", 0
        while len(tok(text, add_special_tokens=False).input_ids) < n:
            text += ("\n\n" if text else "") + paras[k % len(paras)]
            k += 1
        out[name] = text
    return out


def cases(tok):
    st, qs = states(tok), pool()
    return {f"{s}-q{k}": {"state": st[s], "questions": dict(qs[:k])} for s in TARGET for k in KS}


def mark(body, m):
    return {**body, "state": f"[{m}] " + body["state"]}


async def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cs = cases(tok)
    if a.only:
        cs = {k: v for k, v in cs.items() if any(k == p or (p.endswith("-") and k.startswith(p)) for p in a.only)}
    out = {"label": a.label, "url": a.url, "cases": {}, "conc": {}}
    async with httpx.AsyncClient(timeout=3600) as c:
        async def post(b):
            t0 = time.perf_counter()
            r = await c.post(a.url, json=b)
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                raise SystemExit(f"{r.status_code}: {r.text[:500]}")
            return dt, r.json()
        for name in list(cs)[:2]:
            await post(mark(cs[name], "warm"))
        for name, b in cs.items():
            dt, r = await post(b)
            rec = {"answers": {k: v.get("probabilities") for k, v in r["answers"].items()},
                   "input_tokens": r["usage"]["input_tokens"], "branches": r["usage"]["branches"],
                   "questions": r["usage"]["questions"], "lat": [dt]}
            for p in range(1, a.passes + 1):
                rec["lat"].append((await post(mark(b, f"{p}:{name}")))[0])
            rec["p50_ms"] = statistics.median(rec["lat"][1:] if a.passes else rec["lat"]) * 1000
            out["cases"][name] = rec
            print(f"{a.label} {name:8s} branches {rec['branches']:3d} tokens {rec['input_tokens']:6d} "
                  f"p50 {rec['p50_ms']:8.1f} ms", flush=True)
        # concurrency: single questions over the short states of the examples (unique per request)
        sing, qs = singles(), pool()
        first = len(qs) - len(sing)  # the single questions close the pool, in the order of singles()
        for conc in a.conc:
            n = max(48, 4 * conc)
            reqs = [mark({"state": sing[i % len(sing)]["state"], "questions": {"q": qs[first + i % len(sing)][1]}},
                         f"c{conc}:{i}") for i in range(n)]
            sem = asyncio.Semaphore(conc)
            lat = []

            async def one(b):
                async with sem:
                    lat.append((await post(b))[0])
            t0 = time.perf_counter()
            await asyncio.gather(*[one(b) for b in reqs])
            wall = time.perf_counter() - t0
            lat.sort()
            out["conc"][conc] = {"dec_s": n / wall, "p50_ms": statistics.median(lat) * 1000,
                                 "p90_ms": lat[int(0.9 * (n - 1))] * 1000}
            print(f"{a.label} conc {conc:3d}: {n / wall:6.2f} dec/s  p50 {out['conc'][conc]['p50_ms']:7.1f} ms  "
                  f"p90 {out['conc'][conc]['p90_ms']:7.1f} ms", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.label}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results/matrix")
    ap.add_argument("--tokenizer", default="Remek/basal-1.5-4.5B", help="tokenizer of the served model (any basal-1.5 or basal-2.0 folder or repo: they share it)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--only", nargs="*", default=None, help="case names (XL-q2) or prefixes ending in - (S-)")
    asyncio.run(main(ap.parse_args()))
