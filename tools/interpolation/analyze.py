"""Compare basal-calibrate --save-rows files of several candidates on one panel (panel_en.py / panel_pl.py).

  python tools/interpolation/analyze.py rows/a0.jsonl rows/a0.5.jsonl rows/a1.jsonl --panel panel_en.jsonl --ref a0
  python tools/interpolation/analyze.py base=rows/x.jsonl merged=rows/y.jsonl --ref base

Row file (basal/calibrate.py, main(): --save-rows): line 1 is a header
  {"basal_calibrate_rows": 1, "fmt", "kind", "quant", "vllm_preset", "model", "engine", "data", "requests",
   "baseline", "orders"}
then one row per decision (raw_rows()): {"line", "req", "name", "prim", "gold", "keys", "group", "family", "p"}
with p the uncalibrated probabilities over keys and gold an index into keys. There is no "id": "group" holds the
item id (read() falls back to the id when an item has no group; the panel has none) and "line" is its 1-based line in
the panel; rows are matched to the panel by group == id and checked against the line. Failed requests have no row.

Accuracy: argmax(p) == gold (a temperature does not move the argmax). ECE as calibrate.ece (top answer, 10 equal
bins). Calibrated ECE applies, as basal-serve does, temperature_per_bucket[prim]["lo-hi"] for the option count, else
temperature_per_prim[prim], else 1, from calib/<name>.json; without that file the temperatures are refitted on the
rows' FIT split (calibrate.fit_temperature on the same grid, buckets for fmt 2 with >= 50 decisions). "ECE cal CERT"
is measured on the CERT split (30% by sha256 of the group) only, the decisions those temperatures were not fitted on.

Bootstrap: paired over the items both candidates answered, items resampled with replacement within each task (the
panels sample items per task), the difference of the macro accuracy (mean of the per-task accuracies) candidate -
reference, percentile 95% CI.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BUCKETS = (("2-10", 2, 10), ("11-26", 11, 26), ("27-255", 27, 255))
T_GRID = (0.25, 8.0)
MIN_PER_TYPE = 50


def is_cert(group):
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10 < 3


def load_panel(path):
    items, line_of = {}, {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            r = json.loads(line)
            items[str(r["id"])] = r
            line_of[str(r["id"])] = n
    return items, line_of


def load_rows(path, panel, line_of):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    if head.get("basal_calibrate_rows") != 1:
        raise SystemExit(f"{path}: not a file written by basal-calibrate --save-rows")
    rows, bad = {}, []
    for x in lines[1:]:
        if not x.strip():
            continue
        r = json.loads(x)
        iid = r["group"]
        if iid not in panel:
            bad.append(f"group {iid!r} is not a panel id")
            continue
        if r.get("line") != line_of[iid]:
            bad.append(f"{iid}: row line {r.get('line')} != panel line {line_of[iid]}")
        if iid in rows:
            bad.append(f"{iid}: duplicate row")
        r.setdefault("family", panel[iid]["family"])
        if r["family"] != panel[iid]["family"]:
            bad.append(f"{iid}: row family {r['family']!r} != panel {panel[iid]['family']!r}")
        rows[iid] = r
    if bad:
        print(f"warning: {path}: {len(bad)} problem(s): " + "; ".join(bad[:5]), file=sys.stderr)
    return head, rows


def temperature(cal, prim, k):
    for b, T in ((cal.get("temperature_per_bucket") or {}).get(prim) or {}).items():
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= k <= hi:
            return float(T)
    return float((cal.get("temperature_per_prim") or {}).get(prim, 1.0))


def temper(p, T):
    p = np.asarray(p, np.float32)
    if T == 1.0:
        return p
    z = np.log(np.clip(p, np.float32(1e-12), None)) / np.float32(T)
    z = np.exp(z - z.max())
    return z / z.sum()


def fit_temperature(rows):
    K = max(len(r["p"]) for r in rows)
    L = np.full((len(rows), K), -np.inf)
    for i, r in enumerate(rows):
        L[i, : len(r["p"])] = np.log(np.clip(r["p"], 1e-12, 1))
    gold = np.array([r["gold"] for r in rows])
    grid = np.exp(np.linspace(np.log(T_GRID[0]), np.log(T_GRID[1]), 241))
    nll = []
    for t in grid:
        z = L / t
        m = z.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(1))
        nll.append(float(np.mean(lse - z[np.arange(len(rows)), gold])))
    return float(grid[int(np.argmin(nll))])


def refit(rows, fmt):
    """Temperatures fitted on the FIT split of the rows (no shipped baseline: small types/buckets keep 1.0 / type T)."""
    fit_rows = [r for r in rows if not is_cert(r["group"])]
    temps, buckets = {}, {}
    for t in sorted({r["prim"] for r in fit_rows}):
        rs = [r for r in fit_rows if r["prim"] == t]
        if len(rs) >= MIN_PER_TYPE:
            temps[t] = fit_temperature(rs)
    if fmt == 2:
        for name, lo, hi in BUCKETS:
            rs = [r for r in fit_rows if r["prim"] == "choice" and lo <= len(r["p"]) <= hi]
            if len(rs) >= MIN_PER_TYPE:
                buckets.setdefault("choice", {})[name] = fit_temperature(rs)
    return dict(temperature_per_prim=temps, temperature_per_bucket=buckets)


def ece(conf, ok, bins=10):
    if len(conf) == 0:
        return None
    c, ok = np.asarray(conf), np.asarray(ok, float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            e += m.mean() * abs(c[m].mean() - ok[m].mean())
    return float(e)


def macro(correct, fam, n_fam):
    return float(np.mean([correct[fam == f].mean() for f in range(n_fam) if (fam == f).any()]))


def boot_diff(ids, cand, ref, fam_index, n_fam, reps, seed):
    """Paired bootstrap of macro(cand) - macro(ref) over the items both answered -> (diff, lo, hi, n). Items are
    resampled within each task, as the panels sample them per task: every replicate keeps each task's item count, so
    the task mix (and with it the macro weights) stays that of the panel."""
    common = [i for i in ids if i in cand and i in ref]
    a = np.array([cand[i] for i in common], float)
    b = np.array([ref[i] for i in common], float)
    fam = np.array([fam_index[i] for i in common])
    diff = macro(a, fam, n_fam) - macro(b, fam, n_fam)
    rng = np.random.default_rng(seed)
    d = a - b
    per_task = []
    for f in range(n_fam):
        df = d[fam == f]
        if len(df):
            per_task.append(df[rng.integers(0, len(df), (reps, len(df)))].mean(1))
    out = np.mean(per_task, axis=0)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return diff, float(lo), float(hi), len(common)


def parse_cand(spec):
    name, sep, path = spec.partition("=")
    return (name, path) if sep else (Path(spec).stem, spec)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def num(x):
    return "-" if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="--save-rows files, PATH or NAME=PATH (name: the file stem)")
    ap.add_argument("--panel", required=True, help="the panel the rows were run on (panel_en.py / panel_pl.py output)")
    ap.add_argument("--calib-dir", default=None, help="directory with <name>.json calibration files (basal-calibrate --out)")
    ap.add_argument("--calib", action="append", default=[], help="NAME=FILE, overrides --calib-dir for one candidate")
    ap.add_argument("--ref", default=None, help="reference candidate of the bootstrap (default: the first)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default=None, help="also write the markdown here")
    a = ap.parse_args(argv)

    panel, line_of = load_panel(a.panel)
    ids = list(panel)
    fams = sorted({it["family"] for it in panel.values()})
    fam_index = {i: fams.index(panel[i]["family"]) for i in ids}
    calib_over = dict(parse_cand(c) for c in a.calib)

    cands = []
    for spec in a.rows:
        name, path = parse_cand(spec)
        head, rows = load_rows(path, panel, line_of)
        cal_path = calib_over.get(name) or (Path(a.calib_dir) / f"{name}.json" if a.calib_dir else None)
        if cal_path and Path(cal_path).exists():
            cal = json.loads(Path(cal_path).read_text(encoding="utf-8"))
            src = Path(cal_path).name
        else:
            cal, src = refit(list(rows.values()), head.get("fmt", 1)), "refit on rows (FIT split)"
        correct, conf_raw, conf_cal, cert = {}, [], [], []
        for i, r in rows.items():
            p = np.asarray(r["p"], np.float64)
            pred = int(np.argmax(p))
            correct[i] = pred == r["gold"]
            conf_raw.append(float(p.max()))
            q = temper(r["p"], temperature(cal, r["prim"], len(r["p"])))
            conf_cal.append(float(q.max()))
            cert.append(is_cert(r["group"]))
        ok = np.array([correct[i] for i in rows], float)
        cert = np.array(cert, bool)
        conf_cal = np.array(conf_cal)
        fam = np.array([fam_index[i] for i in rows])
        per_task = {f: (float(ok[fam == k].mean()) if (fam == k).any() else None) for k, f in enumerate(fams)}
        cands.append(dict(
            name=name, path=path, head=head, n=len(rows), missing=len(ids) - len(rows), correct=correct,
            acc=float(ok.mean()), macro=macro(ok, fam, len(fams)), per_task=per_task,
            ece_raw=ece(conf_raw, ok, a.bins), ece_cal=ece(conf_cal, ok, a.bins),
            ece_cal_cert=ece(conf_cal[cert], ok[cert], a.bins), ece_raw_cert=ece(np.array(conf_raw)[cert], ok[cert], a.bins),
            cal_src=src, temps=cal.get("temperature_per_prim") or {}, buckets=cal.get("temperature_per_bucket") or {},
            lofo=((cal.get("gates") or {}).get("ece_lofo")) if "gates" in cal else None))

    names = [c["name"] for c in cands]
    if len(set(names)) != len(names):
        raise SystemExit(f"candidate names repeat: {names}; give NAME=PATH")
    ref_name = a.ref or names[0]
    if ref_name not in names:
        raise SystemExit(f"--ref {ref_name!r}: not one of {names}")
    ref = next(c for c in cands if c["name"] == ref_name)

    md = [f"panel: {Path(a.panel).name} ({len(ids)} items, {len(fams)} tasks); reference: {ref_name}; "
          f"bootstrap: {a.reps} paired item resamples, seed {a.seed}", ""]
    md += ["| candidate | n | missing | accuracy | macro acc | Δ macro vs ref [95% CI] | ECE raw | ECE cal | "
           "ECE raw CERT | ECE cal CERT | temperatures |",
           "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for c in cands:
        if c is ref:
            delta = "(ref)"
        else:
            d, lo, hi, n = boot_diff(ids, c["correct"], ref["correct"], fam_index, len(fams), a.reps, a.seed)
            delta = f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" + (f" (n={n})" if n != len(ids) else "")
        T = ", ".join(f"{t} {v:.2f}" for t, v in sorted(c["temps"].items()))
        B = "; ".join(f"{t} {b} {v:.2f}" for t, bs in sorted(c["buckets"].items()) for b, v in sorted(bs.items()))
        md.append(f"| {c['name']} | {c['n']} | {c['missing']} | {pct(c['acc'])} | {pct(c['macro'])} | {delta} | "
                  f"{num(c['ece_raw'])} | {num(c['ece_cal'])} | {num(c['ece_raw_cert'])} | {num(c['ece_cal_cert'])} | "
                  f"{T}{(' / ' + B) if B else ''} ({c['cal_src']}) |")
    md += ["", "Per-task accuracy (%)", "",
           "| task | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for f in fams:
        md.append(f"| {f} | " + " | ".join(pct(c["per_task"][f]) for c in cands) + " |")
    lofo = [(c["name"], c["lofo"]) for c in cands if c["lofo"]]
    if lofo:
        md += ["", "ece_lofo gate of the calibration file (mean / max over families)", "",
               "| candidate | type | mean | max | reference |", "|---|---|---:|---:|---:|"]
        for name, g in lofo:
            for t, e in sorted(g.items()):
                md.append(f"| {name} | {t} | {num(e.get('mean'))} | {num(e.get('max'))} | {num(e.get('reference'))} |")
    md += ["", "accuracy = argmax of the raw probabilities; ECE raw = uncalibrated, ECE cal = with the candidate's "
           "temperatures (all items; those were fitted on the FIT 70%, so CERT columns are the held-out 30%)."]
    text = "\n".join(md) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
