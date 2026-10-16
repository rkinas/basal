"""Markdown report of the matrix and edge-case runs:

  python docs/benchmarks/report.py results/matrix results/edge > report.md

Engines: `--mode gguf` (llama.cpp in process, the baseline), `--mode fast-nocompile` (main's PyTorch graph mode, upstream
defaults), `--mode strix`, and the fp32 reference.
"""
import json
import sys
from pathlib import Path

D, E = Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None
ORDER = ["gguf", "pre-graph-default", "strix", "ref-fp32"]
NAMES = {"gguf": "gguf", "pre-graph-default": "fast-nocompile", "strix": "**strix**", "ref-fp32": "eager fp32 (reference)"}
BASE, NEW = "gguf", "strix"


def load(d):
    r = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if not p.stem.endswith(".startup")}
    s = {p.name.removesuffix(".startup.json"): json.loads(p.read_text()) for p in d.glob("*.startup.json")}
    return r, s


R, S = load(D)
eng = [e for e in ORDER if e in R]
timed = [e for e in eng if e != "ref-fp32"]
cases = list(R[NEW]["cases"]) if NEW in R else list(R[timed[0]]["cases"])


def decisions(ans):
    """question/label -> (decision, probabilities): argmax for single-answer types, p >= 0.5 per label for multi."""
    out = {}
    for q, p in ans.items():
        if not p:
            continue
        if q.startswith("multi") or q == "tags":
            for lab, v in p.items():
                out[f"{q}/{lab}"] = (v >= 0.5, {"yes": v})
        else:
            out[q] = (max(p, key=p.get), p)
    return out


def agree(pairs):
    same = n = 0
    dps = []
    for a, b in pairs:
        da, db = decisions(a), decisions(b)
        for k in da:
            if k in db:
                n += 1
                same += da[k][0] == db[k][0]
                dps += [abs(da[k][1][x] - db[k][1][x]) for x in da[k][1]]
    return same, n, (sum(dps) / len(dps) if dps else float("nan")), max(dps, default=float("nan"))


def case_pairs(a, b):
    return [(a["cases"][c]["answers"], b["cases"][c]["answers"]) for c in a["cases"] if c in b["cases"]]


def fmt(x, spec):
    """A number in the given format; – for a check that could not run (its request failed)."""
    return "–" if x is None else format(x, spec)


def ms(x):
    return f"{x / 1000:.2f} s" if x >= 1000 else f"{x:.0f} ms"


def lat(e, c):
    x = R.get(e, {}).get("cases", {}).get(c)
    return x["p50_ms"] if x else None


if BASE in R and NEW in R:
    print("## gguf → strix per request shape\n")
    print("`--mode gguf` (llama.cpp in process, Vulkan, Q8_0) against `--mode strix` (PyTorch + Triton kernels for "
          "gfx1151, bf16). p50 of 2 timed passes, unique states.\n")
    print("| state | questions | prompt tokens | gguf | strix | speed-up |\n|---|---|---|---|---|---|")
    for c in cases:
        st, q = c.split("-q")
        a, b = lat(BASE, c), lat(NEW, c)
        if a and b:
            print(f"| {st} | {q} | {R[NEW]['cases'][c]['input_tokens']:,} | {ms(a)} | **{ms(b)}** | {a / b:.2f}× |")
    print("\n| clients (single questions) | gguf dec/s · p50 | strix dec/s · p50 | throughput |\n|---|---|---|---|")
    for c in sorted({int(k) for k in R[NEW]["conc"]}):
        a, b = R[BASE]["conc"].get(str(c)), R[NEW]["conc"].get(str(c))
        if a and b:
            print(f"| {c} | {a['dec_s']:.1f}/s · {ms(a['p50_ms'])} | **{b['dec_s']:.1f}/s** · {ms(b['p50_ms'])} | "
                  f"{b['dec_s'] / a['dec_s']:.2f}× |")

print("\n## Start-up and memory\n")
print("| engine | start-up | GTT after start |\n|---|---|---|")
for e in eng:
    if e in S:
        print(f"| {NAMES[e]} | {S[e]['startup_s']:.1f} s | {S[e]['gtt_gib']} GiB |")

print("\n## Latency per request, all engines (p50; bold: fastest)\n")
print("| state | q | branches | " + " | ".join(NAMES[e] for e in timed) + " |")
print("|---|---|---|" + "---|" * len(timed))
for c in cases:
    st, q = c.split("-q")
    vals = {e: lat(e, c) for e in timed if lat(e, c)}
    if not vals:
        continue
    best = min(vals.values())
    cells = [(f"**{ms(vals[e])}**" if vals[e] == best else ms(vals[e])) if e in vals else "–" for e in timed]
    print(f"| {st} | {q} | {R[timed[0]]['cases'][c]['branches']} | " + " | ".join(cells) + " |")

print("\n## Cost of one more question (1 → 20 questions, per added question)\n")
print("| state | " + " | ".join(NAMES[e] for e in timed) + " |\n|---|" + "---|" * len(timed))
for st in ("S", "M", "L", "XL"):
    cells = []
    for e in timed:
        a, b = lat(e, f"{st}-q1"), lat(e, f"{st}-q20")
        cells.append(ms((b - a) / 19) if a and b else "–")
    print(f"| {st} | " + " | ".join(cells) + " |")

print("\n## Concurrency (single questions over short states, unique per request)\n")
concs = sorted({int(c) for e in timed for c in R[e]["conc"]})
print("| clients | " + " | ".join(NAMES[e] for e in timed) + " |\n|---|" + "---|" * len(timed))
for c in concs:
    cells = []
    for e in timed:
        x = R[e]["conc"].get(str(c))
        cells.append(f"{x['dec_s']:.1f}/s · {ms(x['p50_ms'])}" if x else "–")
    print(f"| {c} | " + " | ".join(cells) + " |")

if "ref-fp32" in R:
    ref = R["ref-fp32"]
    print("\n## Agreement with the fp32 reference\n")
    print("Reference: `--mode eager --dtype float32 --soam off --max-batch 1` (every prompt computed separately in fp32), "
          f"on {len(ref['cases'])} requests ({', '.join(ref['cases'])}). A decision is the top option of a question or "
          "the yes/no of one label of a multi-label question.\n")
    print("| engine | same decision | mean \\|Δp\\| | max \\|Δp\\| |\n|---|---|---|---|")
    for e in timed:
        s, n, m, x = agree(case_pairs(ref, R[e]))
        print(f"| {NAMES[e]} | {s}/{n} | {m:.4f} | {x:.3f} |")

if BASE in R and NEW in R:
    print("\n## strix against gguf (whole matrix)\n")
    print("| comparison | same decision | mean \\|Δp\\| | max \\|Δp\\| |\n|---|---|---|---|")
    s, n, m, x = agree(case_pairs(R[BASE], R[NEW]))
    print(f"| gguf (Q8_0) vs strix (bf16) | {s}/{n} | {m:.4f} | {x:.3f} |")

if E and E.exists():
    ER, ES = load(E)
    print("\n## Edge cases\n")
    print("12 requests: mixed PL / EN questions over one state, 10 options, the same question three times, a JSON "
          "state, `facts: auto`, long option texts, a 5-level score, an 8-label multi with min / max, an `act` with "
          "a cost matrix, an English-only request, a 2-character state, `option_keys` show vs hide. Sent one after "
          "another, then all at once (the server batches them).\n")
    print("| engine | errors | concurrent = sequential decisions | concurrent vs sequential, max \\|Δp\\| | "
          "duplicate questions, max \\|Δp\\| | multi within min/max | act action | vs fp32: same decision | "
          "vs fp32: max \\|Δp\\| |")
    print("|---|---|---|---|---|---|---|---|---|")
    eref = ER.get("ref-fp32")
    for e in [x for x in ORDER if x in ER]:
        c = ER[e]["checks"]
        if eref and e != "ref-fp32":
            pairs = [({k: v.get("probabilities") for k, v in eref["sequential"][n]["answers"].items()},
                      {k: v.get("probabilities") for k, v in ER[e]["sequential"][n]["answers"].items()})
                     for n in eref["sequential"] if "answers" in eref["sequential"][n] and "answers" in ER[e]["sequential"][n]]
            s, n, m, x = agree(pairs)
            vs = (f"{s}/{n}", f"{x:.3f}")
        else:
            vs = ("–", "–")
        cs = agree([({k: v.get("probabilities") for k, v in ER[e]["sequential"][n]["answers"].items()},
                     {k: v.get("probabilities") for k, v in ER[e]["concurrent"][n]["answers"].items()})
                    for n in ER[e]["sequential"] if "answers" in ER[e]["sequential"][n]
                    and "answers" in ER[e]["concurrent"][n]])
        print(f"| {NAMES[e]} | {len(c['errors']) or 'none'} | {cs[0]}/{cs[1]} | {c['concurrent_vs_sequential_max_dp']:.4f} | "
              f"{fmt(c['duplicate_questions_max_dp'], '.1e')} | {c['multi_selected_within_min_max']} | {c['act_action']} | "
              f"{vs[0]} | {vs[1]} |")
