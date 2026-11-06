"""Option-order check: how often the top pick changes when the options are shown in another order, and what averaging
over K orders (basal-serve --order-average K) gains and costs.

  basal-bench --order-check --model Remek/basal-omni-1.0-mini                  # bundled examples, serving default
  basal-bench --order-check --model Remek/basal-omni-1.0-mini --mode eager --questions dev.jsonl --average 2 4 6

Questions file (JSONL), as basal-bench: {"state": ..., "question": ..., "options": [...], "gold": <index, optional>}.
Every question is read in its first M option orders (basal.server.option_orders: the given order, the reversed one,
then each rotation and its reverse; M = the largest of 3 and --average), all orders of a question in one call over
one state. Reported per mode:
  flip_reversed  share of questions whose top option (single order) differs between the given and the reversed order
  flip_rotated   the same for the order rotated by one (the first option moved last; with 2 options = reversed)
  flip_any       the top option differs in at least one of the M orders
  first_share    share of single-order top picks at the first position, over the M orders (chance: mean 1 / k)
  acc_given, acc_reversed, acc_rotated     accuracy of each single order (with gold)
  acc_mean<K>    accuracy of the mean over the first K orders, as --order-average K serves it (K = 1: the given order)
  agree_mean<K>  share of questions whose mean-over-K top option equals the given order's
  lat_mean<K>_ms median latency of one question with its first K orders at batch size 1 (--lat-n questions)
Flip rates and accuracies are also split by the number of options (by_k).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .engine import resolve
from .prompt import letter_ids, model_format, render


def top(p):
    return max(range(len(p)), key=p.__getitem__)


def canonical(perms, probs):
    """Position probabilities of each order -> the probabilities of each order in the canonical option order."""
    out = []
    for perm, p in zip(perms, probs):
        c = [0.0] * len(perm)
        for pos, j in enumerate(perm):
            c[j] = float(p[pos])
        out.append(c)
    return out


def mean(rows):
    return [sum(x) / len(rows) for x in zip(*rows)]


def groups_for(tok, qs, m, fmt=1):
    """Each question in its first m option orders -> [(question, perms, prompts, code ids per prompt)]."""
    from .server import option_orders
    out = []
    for q in qs:
        k = len(q["options"])
        perms = option_orders(k, m)
        prompts = [render(tok, q["state"], q["question"], [q["options"][c] for c in p], fmt=fmt) for p in perms]
        ids = letter_ids(tok, prompts[0], k, fmt)
        if ids and k >= 2:
            out.append((q, perms, prompts, [ids] * len(perms)))
    return out


def summarise(groups, canon, averages):
    """Per-question canonical probabilities of every order -> the report (without latencies)."""
    rot_of = lambda k: list(range(1, k)) + [0]  # the order rotated by one
    rows = []
    for (q, perms, _, _), c in zip(groups, canon):
        k = len(perms[0])
        tops = [top(x) for x in c]
        rot = perms.index(rot_of(k))
        pos = [p.index(t) for p, t in zip(perms, tops)]  # position of each order's top pick
        rows.append(dict(k=k, gold=q.get("gold"), tops=tops, rev=tops[1] != tops[0], rot=tops[rot] != tops[0],
                         any=len(set(tops)) > 1, first=sum(x == 0 for x in pos) / len(pos), chance=1 / k,
                         rot_i=rot, means={K: top(mean(c[:K])) for K in averages}))
    gold = all(r["gold"] is not None for r in rows)

    def block(rs):
        n = len(rs)
        if not n:
            return {}
        b = dict(n=n, flip_reversed=sum(r["rev"] for r in rs) / n, flip_rotated=sum(r["rot"] for r in rs) / n,
                 flip_any=sum(r["any"] for r in rs) / n, first_share=sum(r["first"] for r in rs) / n,
                 first_chance=sum(r["chance"] for r in rs) / n)
        if gold:
            b.update(acc_given=sum(r["tops"][0] == r["gold"] for r in rs) / n,
                     acc_reversed=sum(r["tops"][1] == r["gold"] for r in rs) / n,
                     acc_rotated=sum(r["tops"][r["rot_i"]] == r["gold"] for r in rs) / n)
        for K in averages:
            if gold:
                b[f"acc_mean{K}"] = sum(r["means"][K] == r["gold"] for r in rs) / n
            b[f"agree_mean{K}"] = sum(r["means"][K] == r["tops"][0] for r in rs) / n
        return b

    out = block(rows)
    out["by_k"] = {label: block([r for r in rows if lo <= r["k"] <= hi])
                   for label, lo, hi in (("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10 ** 6))
                   if any(lo <= r["k"] <= hi for r in rows)}
    return out


def latencies(be, groups, averages, lat_n):
    """Median latency of one question with its first K orders at batch size 1, per K."""
    from .bench import timed
    out = {}
    for K in averages:
        lat = []
        for g in groups[: lat_n + 3]:
            unit = [(g[2][:K], g[3][:K])]
            be.run_shared(unit)  # warm-up of the shapes
            lat += timed(lambda: be.run_shared(unit), 1, be)
        out[f"lat_mean{K}_ms"] = statistics.median(lat[3:] or lat) * 1000
    return out


def run(be, qs, fmt, averages, lat_n):
    m = max([3] + list(averages))
    groups = groups_for(be.tok, qs, m, fmt)
    probs = []
    for i in range(0, len(groups), 16):
        probs += be.run_shared([(g[2], g[3]) for g in groups[i: i + 16]])
    canon = [canonical(g[1], p) for g, p in zip(groups, probs)]
    r = summarise(groups, canon, averages)
    if lat_n > 0:
        r.update(latencies(be, groups, averages, min(lat_n, len(groups))))
    return r


def main(argv=None):
    from .bench import DEFAULT_QUESTIONS, build, device_name, questions_or_exit
    from .server import default_mode
    ap = argparse.ArgumentParser(prog="basal-bench --order-check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Remek/basal-1.5-4.5B")
    ap.add_argument("--mode", default=None, help="a basal-bench mode (default: the serving default)")
    ap.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS,
                    help="JSONL file(s) with simple items (default: the bundled examples)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--average", nargs="+", type=int, default=[1, 2, 4],
                    help="K values of the mean over the first K option orders (as basal-serve --order-average K)")
    ap.add_argument("--lat-n", dest="lat_n", type=int, default=50, help="questions timed per K (0: no timing)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args(argv)
    if any(K < 1 for K in a.average):
        raise SystemExit("--average: K >= 1 option orders")
    mode = a.mode or default_mode(a.model)
    md = resolve(a.model)
    fmt = model_format(md)
    qs = questions_or_exit(a.questions, a.n)
    t0 = time.time()
    be = build(mode, md, fmt=fmt)
    r = dict(mode=mode, model=a.model, gpu=device_name(be), load_s=round(time.time() - t0, 1),
             **run(be, qs, fmt, sorted(set(a.average)), a.lat_n))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k != "by_k"}), flush=True)
    for label, b in r["by_k"].items():
        print(f"  {label} options: n {b['n']}, flip reversed {b['flip_reversed']:.3f}, rotated "
              f"{b['flip_rotated']:.3f}, any {b['flip_any']:.3f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
