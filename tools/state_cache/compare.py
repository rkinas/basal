"""Compare two basal-run --output files (a reference run and a candidate: a restart over the disk tier, another engine's
states, ...): how close the candidate's answers are, optionally its speed and the state cache counters of the server.

    python tools/state_cache/compare.py --reference REF.jsonl --candidate CAND.jsonl [--metrics M.txt ...]
        [--speed-off S0.jsonl --speed-on S1.jsonl] [--require EXPR ...] --label TEXT [--out ROW.json]

A readout is one question of one line (matched by id and question name), its probabilities in the order of the sorted
option keys. Printed: one JSON object, then (with --out) the Markdown row of docs/HARDWARE.md.
  identical               every probability equal (and no readout missing)
  agreement               share of readouts with the same top option
  agreement_at_margin_0_2 the same over readouts whose reference top two are >= 0.2 apart (null: none)
  max_dp / p99_dp / median_dp   of each readout's largest |difference of a probability|
  near_ties               disagreements whose reference top two are < 0.02 apart (as basal/identity.py)
  missing                 readouts on one side only (not compared; exit status 2, as with no readout compared at all)
--metrics: the basal_state_cache_* values of GET /metrics dumps. --speed-off / --speed-on: p50 / p95 of the per-line
latency_ms of two runs, speedup = p50 off / p50 on. --require NAME OP VALUE (OP >=, <= or =; a name of the object, or
of the last --metrics dump): exit status 1 when one fails, each printed as FAIL <expr> (<actual>).
"""
import argparse
import json
import re
import statistics
import sys

METRICS = ("hits", "misses", "disk_loads", "disk_prefix_loads", "disk_writes", "disk_write_drops", "disk_errors",
           "disk_evictions", "disk_bytes", "disk_entries")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def readouts(path):
    """(id, question) -> probabilities in the order of the sorted option keys."""
    out = {}
    for r in lines(path):
        for q, a in (r.get("answers") or {}).items():
            p = a.get("probabilities") or {}
            out[(json.dumps(r.get("id")), q)] = [p[k] for k in sorted(p)]
    return out


def gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def percentile(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))] if s else None


def compare(ref, cand):
    keys = sorted(ref.keys() & cand.keys())
    missing = len(ref.keys() ^ cand.keys())
    dps, agree, margin, margin_agree, near, identical = [], 0, 0, 0, 0, missing == 0 and bool(keys)
    for k in keys:
        a, b = ref[k], cand[k]
        if len(a) != len(b):
            identical = False
            dps.append(1.0)
            continue
        identical &= a == b
        dps.append(max(abs(x - y) for x, y in zip(a, b)))
        same = argmax(a) == argmax(b)
        agree += same
        near += not same and gap(a) < 0.02
        if gap(a) >= 0.2:
            margin += 1
            margin_agree += same
    n = len(keys)
    return {"readouts": n, "identical": identical, "agreement": agree / n if n else None,
            "agreement_at_margin_0_2": margin_agree / margin if margin else None,
            "max_dp": max(dps) if dps else None, "p99_dp": percentile(dps, 99),
            "median_dp": statistics.median(dps) if dps else None, "near_ties": near, "disagreements": n - agree,
            "missing": missing, "agree_n": agree}


def metrics(path):
    out = {"file": path}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = re.match(r"basal_state_cache_(\w+?)(?:_total)?\s+(\S+)$", ln.strip())
            if m and m.group(1) in METRICS:
                v = float(m.group(2))
                out[m.group(1)] = int(v) if v.is_integer() else v
    return out


def speed(off, on):
    lat = lambda path: [r["latency_ms"] for r in lines(path) if r.get("latency_ms") is not None]
    a, b = lat(off), lat(on)
    p50_off, p50_on = statistics.median(a), statistics.median(b)
    return {"p50_off_ms": p50_off, "p50_on_ms": p50_on, "p95_off_ms": percentile(a, 95),
            "p95_on_ms": percentile(b, 95), "speedup": p50_off / p50_on if p50_on else None}


def parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return float(v)


def check(expr, res):
    m = re.fullmatch(r"(\w+)\s*(>=|<=|=)\s*(\S+)", expr)
    if not m:
        raise SystemExit(f"--require {expr!r}: NAME OP VALUE, OP one of >=, <=, =")
    name, op, want = m.group(1), m.group(2), parse_value(m.group(3))
    got = res.get(name, (res.get("metrics") or [{}])[-1].get(name))
    if got is None:
        return False, got
    if op == "=":
        return (got == want if isinstance(want, bool) else abs(float(got) - want) < 1e-12), got
    return (got >= want if op == ">=" else got <= want), got


def fmt(x, spec=".3g"):
    return "–" if x is None else format(x, spec)


def row(label, r):
    dp = f"{fmt(r['median_dp'], '.4f')} / {fmt(r['p99_dp'], '.4f')} / {fmt(r['max_dp'], '.4f')}"
    speedup = f"{r['speedup']:.2f}x" if r.get("speedup") is not None else "–"
    return (f"| {label} | {fmt(r.get('p50_off_ms'), '.0f')} ms | {fmt(r.get('p50_on_ms'), '.0f')} ms | {speedup} "
            f"| {r['agree_n']}/{r['readouts']} | {fmt(r['agreement_at_margin_0_2'], '.4f')} | {dp} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metrics", nargs="+", default=[])
    ap.add_argument("--speed-off", dest="speed_off")
    ap.add_argument("--speed-on", dest="speed_on")
    ap.add_argument("--require", nargs="+", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = compare(readouts(a.reference), readouts(a.candidate))
    if a.metrics:
        res["metrics"] = [metrics(p) for p in a.metrics]
    if bool(a.speed_off) != bool(a.speed_on):
        ap.error("--speed-off and --speed-on go together")
    if a.speed_off:
        res.update(speed(a.speed_off, a.speed_on))
    res["label"] = a.label
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(row(a.label, res))
    if res["missing"] or not res["readouts"]:  # (no readout compared: a failed run never passes a gate)
        print(f"FAIL missing=0 ({res['missing']})" if res["missing"] else "FAIL readouts>0 (0)")
        return 2
    failed = 0
    for expr in a.require:
        ok, got = check(expr, res)
        if not ok:
            print(f"FAIL {expr} ({got})")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
