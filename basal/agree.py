"""basal-agree: every backend against the native engine, on the same prompts.

  basal-agree --model Remek/basal-2.0-mini --backends mps mlx mlx=Remek/basal-2.0-mini-MLX-8bit
  basal-agree --model Remek/basal-omni-1.0-mini --reference eager --backends fast vllm --questions items.jsonl
  basal-agree --identity --model Remek/basal-1.5-4.5B    # the identity gate of the CUDA fast path (basal/identity.py)

Items are synthetic (basal.synth, no real data): single questions over ~1.5k-token states, five-question requests over
one state and long states, every question in both option orders; --questions adds your own simple items (a "gold"
index adds accuracy). The reference is the native engine (--reference, default eager-fp32: basal's PyTorch forward in
fp32 on this machine's device) with every prompt alone, one per call. Its rows are cached (--cache) under a key of the
model revision, the basal version, the reference and the prompts, so further runs against other backends reuse them.

A backend is a basal-bench mode (eager, mps, mlx, mlx-q8, fast, vllm, gguf@FILE, ollama, ...), or MODE=WEIGHTS for the
same model in other weights with the same tokenizer (mlx=Remek/basal-2.0-mini-MLX-8bit). Backends are built, run and
freed one at a time; each answers the requests as served: both orders of a question in one call, the five questions of
a request in one call. Per backend:

  readouts     top-1 agreement with the reference, with a 95% bootstrap interval over requests; near ties among the
               disagreements (top two within 0.02 in the reference); mean / p99 / max |dp| (per readout: the largest
               change of an option's probability)
  decisions    the single questions as the server answers them by default: both orders averaged for prompt-format-1
               models, the given order for format-2 models (basal-serve's default orders): top-1 agreement
  accuracy     items with gold: accuracy and the paired bootstrap difference to the reference (95% interval)
  independent  question independence: every prompt of the five-question requests alone through the same backend
               against its readout in the request (top-1 agreement, max |dp|); a backend whose answer moves with the
               other questions of the request shows it here, whatever the reference says

Gate: the readout agreement outside near ties (a disagreement whose reference top two are within 0.02 counts as
agreeing: any change of rounding can flip those) >= --gate (default 0.99), and the same for question independence.
The exit status is 0 when every backend passes, 1 otherwise.
"""
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

from . import synth
from .jsonl import read_lines
from .prompt import letter_ids, model_format, render

GATE = 0.99
OLLAMA_URL, LLAMACPP_URL = "http://127.0.0.1:11434", "http://127.0.0.1:8080"
BOOT = 2000

QUESTIONS = {
    "pl": [("Który dział powinien obsłużyć ostatnie zgłoszenie?", ["Reklamacje", "Płatności", "Logistyka", "Sprzedaż"]),
           ("Czy klient zgłosił uszkodzenie przesyłki?", ["Tak", "Nie"]),
           ("Jak pilne jest ostatnie zgłoszenie?", ["niska pilność", "średnia pilność", "wysoka pilność"]),
           ("Jaki status ma ostatnie zgłoszenie?", ["przyjęte", "w toku", "czeka na klienta", "przekazane", "zamknięte",
                                                   "nie wiadomo"]),
           ("Czy w historii są co najmniej trzy zgłoszenia tej samej osoby?", ["Tak", "Nie"])],
    "en": [("Which department should handle the latest ticket?", ["Complaints", "Payments", "Logistics", "Sales"]),
           ("Did the customer report a damaged parcel?", ["Yes", "No"]),
           ("How urgent is the latest ticket?", ["low urgency", "medium urgency", "high urgency"]),
           ("What is the status of the latest ticket?", ["received", "in progress", "waiting for customer", "forwarded",
                                                         "closed", "unknown"]),
           ("Are there at least three tickets from the same person in the history?", ["Yes", "No"])],
}


def orders(tok, fmt, state, question, options):
    """Both option orders of one question -> (token lists, code ids)."""
    k = len(options)
    prompts = [render(tok, state, question, [options[c] for c in perm], fmt=fmt)
               for perm in (list(range(k)), list(range(k))[::-1])]
    ids = letter_ids(tok, prompts[0], k, fmt)
    if ids is None:
        raise ValueError("option codes are not single tokens for this tokenizer")
    return tok(prompts, add_special_tokens=False).input_ids, [ids, ids]


def build(tok, fmt, n_single, n_multi, n_long, seed=0, questions=None):
    """Synthetic single questions (plus the simple items of `questions`, a list of dicts with state / question /
    options), five-question requests over one state, long states."""
    items = synth.items(n_single, 1500, seed=seed) + list(questions or [])
    single = [orders(tok, fmt, it["state"], it["question"], it["options"]) for it in items]
    multi = []
    for i, it in enumerate(synth.items(n_multi, 1500, seed=seed + 1)):
        lang = "pl" if i % 2 == 0 else "en"
        req = [orders(tok, fmt, it["state"], q, o) for q, o in QUESTIONS[lang]]
        multi.append(([t for r in req for t in r[0]], [x for r in req for x in r[1]]))
    long = [orders(tok, fmt, it["state"], it["question"], it["options"])
            for it in synth.items(n_long, 5600, seed=seed + 2, spread=0.4)]
    return single, multi, long


def top(p):
    return max(range(len(p)), key=p.__getitem__)


def compare(ref, got):
    """Lists of probability vectors -> agreement, |dp| statistics (per readout: the largest change of an option's
    probability), near-tie disagreements."""
    agree, dps, ties = 0, [], 0
    for r, g in zip(ref, got):
        a, b = top(r), top(g)
        agree += a == b
        if a != b:
            hi = sorted(r, reverse=True)
            ties += hi[0] - hi[1] < 0.02
        dps.append(max(abs(x - y) for x, y in zip(r, g)))
    mean = sum(dps) / len(dps)
    dps.sort()
    return dict(readouts=len(ref), agreement=agree / len(ref), mean_dp=mean, max_dp=dps[-1],
                p99_dp=dps[int(0.99 * (len(dps) - 1))], disagreements=len(ref) - agree, near_ties=ties)


def decisions(readouts, orders=2):
    """Readouts of the single questions (order 1, reversed order, per item) -> the canonical probabilities as the
    server answers with `orders` (server.default_orders: 2 for prompt-format-1 models, the two averaged; 1 for
    format-2 models, the given order alone)."""
    if orders == 1:
        return [list(o1) for o1 in readouts[0::2]]
    out = []
    for o1, o2 in zip(readouts[0::2], readouts[1::2]):
        out.append([(x + y) / 2 for x, y in zip(o1, o2[::-1])])
    return out


def bootstrap(units, stat, n=BOOT, seed=0):
    """95% percentile interval of stat(sample) over n resamples (with replacement) of the units."""
    rnd = random.Random(seed)
    vals = sorted(stat([units[rnd.randrange(len(units))] for _ in units]) for _ in range(n))
    return [vals[int(0.025 * (n - 1))], vals[int(0.975 * (n - 1))]]


def flat(groups):
    return [p for g in groups for p in g]


def run_groups(be, groups, per_call):
    out = []
    for i in range(0, len(groups), per_call):
        out += be.run_shared(groups[i: i + per_call])
    return out


def alone(groups):
    """Every prompt of the groups as its own group."""
    return [([t], [x]) for toks, ids in groups for t, x in zip(toks, ids)]


def regroup(rows, groups):
    """Flat readouts -> one list per group."""
    out, k = [], 0
    for toks, _ in groups:
        out.append(rows[k: k + len(toks)])
        k += len(toks)
    return out


def score(sets, ref, got, ind, gold, seed=0, orders=2):
    """One backend's row: readouts per set and overall (with the bootstrap interval over requests), decisions,
    accuracy against gold, question independence. ref / got: per set, per group, the readouts."""
    row = {"sets": {k: compare(flat(ref[k]), flat(got[k])) for k in sets if ref[k]}}
    units = [(r, g) for k in sets for r, g in zip(ref[k], got[k])]
    row.update(compare([p for r, _ in units for p in r], [p for _, g in units for p in g]))
    row["agreement_ci"] = bootstrap(units, lambda s: sum(top(a) == top(b) for r, g in s for a, b in zip(r, g))
                                    / sum(len(r) for r, _ in s), seed=seed)
    rd, gd = decisions(flat(ref["single"]), orders), decisions(flat(got["single"]), orders)
    row["decision_agreement"] = compare(rd, gd)["agreement"]
    if gold:  # (decision index, gold index) of the items with gold
        pairs = [(top(rd[i]) == y, top(gd[i]) == y) for i, y in gold]
        row["accuracy"] = sum(b for _, b in pairs) / len(pairs)
        row["reference_accuracy"] = sum(a for a, _ in pairs) / len(pairs)
        row["accuracy_diff_ci"] = bootstrap(pairs, lambda s: (sum(b for _, b in s) - sum(a for a, _ in s)) / len(s),
                                            seed=seed)
    if ind is not None:
        c = compare(flat(got["multi"]), ind)
        row["independence"] = c["agreement"]
        row["independence_ties"] = c["near_ties"]
        row["independence_max_dp"] = c["max_dp"]
    return row


def gate(row, g=GATE):
    """Agreement outside near ties >= g, for the readouts and for question independence."""
    ok = row["agreement"] + row["near_ties"] / row["readouts"] >= g
    if "independence" in row:
        ok = ok and row["independence"] + row["independence_ties"] / (row["readouts_multi"]) >= g
    return "pass" if ok else "FAIL"


def cache_key(model, revision, version, reference, sets, weights=None):
    """`weights` holds only what the reference reads and what differs from the default; with none, the key is the
    one of earlier versions, so the cached rows of the default reference stay valid."""
    parts = dict(model=model, revision=revision, version=version, reference=reference, sets=sets)
    if weights:
        parts["weights"] = weights
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:24]


def reference_weights(a):
    """The options that choose the weights of the reference engine (gguf, ollama, llamacpp), if not at the default."""
    mode = a.reference.partition("=")[0]
    if mode == "gguf":
        fields = dict(gguf=(a.gguf, None))
    elif mode == "ollama":
        fields = dict(ollama_model=(a.ollama_model, None), ollama_url=(a.ollama_url, OLLAMA_URL))
    elif mode == "llamacpp":
        fields = dict(llamacpp_url=(a.llamacpp_url, LLAMACPP_URL))
    else:
        fields = {}
    return {k: v for k, (v, default) in fields.items() if v != default}


def cached_reference(path, make, sets):
    """The reference rows: from the cache file if it is there, else computed (every prompt alone) and stored."""
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    be = make()
    ref = {k: regroup(flat(run_groups(be, alone(v), 1)), v) for k, v in sets.items()}
    free(be)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ref), encoding="utf-8")
    return ref, False


def free(be):
    from .bench import reset_memory
    if hasattr(be, "llm"):
        del be.llm
    del be
    gc.collect()
    reset_memory()


def backend_factory(spec, md, a, fmt):
    """basal-bench mode, or MODE=WEIGHTS (the same model in other weights; the tokenizer must match)."""
    from .bench import build as build_mode
    from .engine import resolve
    mode, _, weights = spec.partition("=")
    wd = resolve(weights) if weights else md
    return lambda: build_mode(mode, wd, gguf=a.gguf, ollama_model=a.ollama_model, ollama_url=a.ollama_url,
                              llamacpp_url=a.llamacpp_url, fmt=fmt)


PROBE = "Stan: zażalenie złożono 3 marca 2026 r. State: 12.50 EUR"


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if "--identity" in argv:  # the identity gate of the CUDA fast path
        from .identity import main as identity
        rows = identity([x for x in argv if x != "--identity"])
        return 0 if all(r["gate"] == "pass" for r in rows) else 1
    ap = argparse.ArgumentParser(prog="basal-agree", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backends", nargs="+", required=True, help="basal-bench modes, or MODE=WEIGHTS")
    ap.add_argument("--reference", default="eager-fp32", help="the native engine's mode (default eager-fp32)")
    ap.add_argument("--n-single", dest="n_single", type=int, default=100)
    ap.add_argument("--n-multi", dest="n_multi", type=int, default=20)
    ap.add_argument("--n-long", dest="n_long", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", default=None, help="JSONL of simple items {state, question, options, gold?}")
    ap.add_argument("--n-questions", dest="n_questions", type=int, default=500)
    ap.add_argument("--per-call", dest="per_call", type=int, default=8, help="requests per call of a backend")
    ap.add_argument("--independence", choices=["on", "off"], default="on")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--cache", default=str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "basal"
                                           / "agree"), help="directory of the cached reference rows ('' = none)")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--ollama-model", dest="ollama_model", default=None)
    ap.add_argument("--ollama-url", dest="ollama_url", default=OLLAMA_URL)
    ap.add_argument("--llamacpp-url", dest="llamacpp_url", default=LLAMACPP_URL)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from . import __version__
    from .engine import resolve
    from .server import default_orders, model_identity
    md = resolve(a.model)
    fmt = model_format(md)
    tok = AutoTokenizer.from_pretrained(md)
    for spec in a.backends:
        w = spec.partition("=")[2]
        if w and AutoTokenizer.from_pretrained(resolve(w))(PROBE).input_ids != tok(PROBE).input_ids:
            raise SystemExit(f"{spec}: the weights {w} have a different tokenizer from {a.model}")
    qs = []
    if a.questions:
        qs = [json.loads(x) for x in read_lines(a.questions) if x.strip()]
        qs = [q for q in qs if "options" in q and 2 <= len(q["options"]) <= (255 if fmt == 2 else 10)][: a.n_questions]
    single, multi, long = build(tok, fmt, a.n_single, a.n_multi, a.n_long, a.seed, qs)
    sets = {"single": single, "multi": multi, "long": long}
    gold = [(a.n_single + i, q["gold"]) for i, q in enumerate(qs) if isinstance(q.get("gold"), int)]
    model, revision = model_identity(a.model, md)
    key = cache_key(model, revision, __version__, a.reference, sets, reference_weights(a))
    path = Path(a.cache) / f"{key}.json" if a.cache else None
    t0 = time.time()
    ref, hit = cached_reference(path, backend_factory(a.reference, md, a, fmt), sets)
    print(json.dumps(dict(model=model, revision=revision, reference=a.reference, cache=str(path) if path else None,
                          cached=hit, reference_s=round(time.time() - t0, 1),
                          readouts={k: sum(len(g[0]) for g in v) for k, v in sets.items()})), flush=True)
    rows = []
    for spec in a.backends:
        t0 = time.time()
        be = backend_factory(spec, md, a, fmt)()
        load_s = time.time() - t0
        t0 = time.time()
        got = {k: run_groups(be, v, a.per_call) for k, v in sets.items()}  # per set, per group, readouts
        run_s = time.time() - t0
        ind = flat(run_groups(be, alone(multi), 1)) if a.independence == "on" else None
        free(be)
        row = dict(backend=spec, load_s=round(load_s, 1), run_s=round(run_s, 1), **score(sets, ref, got, ind, gold,
                                                                                    orders=default_orders(None, fmt)))
        row["readouts_multi"] = len(flat(got["multi"]))
        row["gate"] = gate(row, a.gate)
        rows.append(row)
        print(json.dumps({k: round(v, 7) if isinstance(v, float) else v for k, v in row.items() if k != "sets"}),
              flush=True)
    w = max([28] + [len(r["backend"]) + 2 for r in rows])
    print(f"\n{'backend':<{w}}{'readouts':>9}{'top-1':>8}{'95% CI':>16}{'ties':>7}{'mean|dp|':>10}{'max|dp|':>10}"
          f"{'decisions':>10}{'indep.':>8}  gate (reference {a.reference})")
    for r in rows:
        ci = f"[{r['agreement_ci'][0]:.3f}, {r['agreement_ci'][1]:.3f}]"
        ind = f"{r['independence']:.3f}" if "independence" in r else "-"
        print(f"{r['backend']:<{w}}{r['readouts']:>9}{r['agreement']:>8.4f}{ci:>16}"
              f"{str(r['near_ties']) + '/' + str(r['disagreements']):>7}{r['mean_dp']:>10.2e}{r['max_dp']:>10.2e}"
              f"{r['decision_agreement']:>10.4f}{ind:>8}  {r['gate']}")
        if "accuracy" in r:
            lo, hi = r["accuracy_diff_ci"]
            print(f"{'':<{w}}accuracy {r['accuracy']:.4f} vs {r['reference_accuracy']:.4f} "
                  f"(paired 95% CI of the difference [{lo:+.4f}, {hi:+.4f}], {len(gold)} items)")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(model=model, revision=revision, reference=a.reference, version=__version__,
                                               questions=len(qs), rows=rows), indent=1), encoding="utf-8")
    return 0 if all(r["gate"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
