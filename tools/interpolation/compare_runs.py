"""Compare two --save-rows files of the same candidate (a run and its repeat): decision agreement and probability drift.

    python tools/interpolation/compare_runs.py rows/x.jsonl rows_rep/x.jsonl
First output line is a one-line verdict. Decisions present in only one file, or answered over a different number of
options, make the runs DIFFER whatever the probabilities of the shared ones.
"""
import json
import sys


def load(path):
    with open(path) as f:
        header = json.loads(f.readline())
        rows = {(r["line"], r["name"]): r for r in map(json.loads, f)}
    return header, rows


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def main(a, b):
    _, ra = load(a)
    _, rb = load(b)
    keys = ra.keys() & rb.keys()
    unmatched = len(ra.keys() ^ rb.keys())
    agree, max_dp, sum_dp, shape = 0, 0.0, 0.0, 0
    for k in keys:
        pa, pb = ra[k]["p"], rb[k]["p"]
        if len(pa) != len(pb):
            shape += 1
            continue
        agree += argmax(pa) == argmax(pb)
        d = max(abs(x - y) for x, y in zip(pa, pb))
        max_dp, sum_dp = max(max_dp, d), sum_dp + d
    n = len(keys)
    acc = lambda r: sum(argmax(x["p"]) == x["gold"] for x in r.values()) / max(len(r), 1)
    if unmatched or shape:
        verdict = "DIFFERS"
    else:
        verdict = "IDENTICAL" if max_dp == 0 else ("OK" if agree == n and max_dp < 1e-2 else "DIFFERS")
    print(f"{verdict}: {n} decisions in both ({len(ra)} / {len(rb)}, {unmatched} in one file only, {shape} with a "
          f"different option count), argmax agreement {agree}/{n - shape}, max |dp| {max_dp:.2e}, "
          f"mean |dp| {sum_dp / max(n - shape, 1):.2e}, accuracy {acc(ra):.4f} / {acc(rb):.4f}")
    for k in sorted(keys - {k for k in keys if ra[k]["p"] == rb[k]["p"]})[:20]:
        print("  differs:", k, ra[k]["p"][:4], rb[k]["p"][:4])
    for k in sorted(ra.keys() ^ rb.keys())[:20]:
        print("  only in", a if k in ra else b, k)


if __name__ == "__main__":
    main(*sys.argv[1:3])
