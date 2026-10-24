"""Fit the calibration on your own labelled requests: one temperature per question type (for prompt-format-2 models
also one per option-count bucket of choice questions), the certified confidence thresholds of act's max_error (1%
and 5% by default), the calibration gates and, from labelled multi questions, the fallback_threshold of each label
set, written as a calibration file for basal-serve.

  basal-calibrate labelled.jsonl --model Remek/basal-1.5-4.5B --mode mlx        # the engine in process
  basal-calibrate labelled.jsonl --url http://127.0.0.1:8000 --model Remek/basal-1.5-4.5B
      # a running `basal-serve ... --no-calibration`; --model (or --calibration) gives the shipped file to compare with
  basal-serve --model Remek/basal-1.5-4.5B --mode mlx --calibration CALIBRATION.custom.json

Input (JSONL): one /v1/systemone request per line, every question with its correct answer in "gold":
  {"id": "t-17", "state": "...", "group": "customer-17",
   "questions": {"dept": {"type": "choice", "instructions": "...", "criteria": {"returns": "...", "it": "..."},
                          "gold": "returns"},
                 "late": {"type": "noul", "instructions": "...", "gold": "yes"},
                 "urg":  {"type": "score", "instructions": "...", "criteria": ["low", "mid", "high"], "gold": 2}}}
  gold    choice: an option key or its description; noul: "yes" / "no" (true / false); score: the level index
          (0 = first level), key or text; act: an outcome, as for choice (yes / no for true / false outcomes);
          multi: the list of labels that apply (keys or descriptions; [] = none)
  group   optional, on the line or on a question: decisions of one group (twins, duplicates, one customer) stay on
          the same side of the FIT / CERT split; default: one group per line (its questions share the state)
  family  optional, on the line or on a question: the task family of the leave-one-family-out ECE gate (default: the
          question name)
  id      optional, used in messages.
  multi   questions with a "fallback" label only (and the objective, threshold, min, max and exclusive they are served
          with): they fit the fallback_threshold of their label set; they do not enter the temperatures (to calibrate
          the labels, ask each one as a noul question with its gold).
  The simple items of basal-run and basal-bench ({"state", "question", "options", "type", "gold": <index>}) are
  accepted too, e.g. basal/examples/questions.jsonl.
  tasks   classify (single label; gold = a label key, "none" with "none": true), verify (gold = {claim key: verdict},
          or a list in the order of the claims; supported / contradicted / not_stated, binary: supported /
          not_supported; null = not labelled) and extract (gold = the field values nested as in the schema; null = not
          stated; a field left out is not labelled; arrays of enum are not certified) certify the task_thresholds of
          their `automated` flag. They need the engine in process (not --url): the thresholds are certified on the
          answers as served with the fitted temperatures. A file with task questions only keeps the decision
          calibration it compares with and adds the task thresholds.

Method (the one the shipped files were fitted with):
  * the decisions are split by a hash of their group into FIT (70%) and CERT (30%);
  * one temperature per type (choice / noul / score; act counts as its base type) minimises the NLL of the served
    prediction on FIT; a type with fewer than --min-per-type FIT decisions but at least --min-blend (10) gets it
    blended in log space with the shipped one in proportion n / --min-per-type; with fewer it keeps the shipped one;
  * prompt-format-2 models (basal.json "prompt": 2): the file is calibration file 2 ("prompt": 2), and choice questions
    also get a temperature per option-count bucket (2-10, 11-26, 27-255) fitted the same way where FIT has at least
    --min-per-type decisions in the bucket (otherwise the shipped bucket temperature, else the type temperature);
  * for each target error r, the threshold is the lowest confidence whose accepted FIT decisions (at least 50; equal
    confidences accepted together) have a one-sided 95% Clopper-Pearson upper bound <= r; CERT reports its coverage,
    error and bound. The threshold is fitted per type (thresholds[r].per_type[type], on the type's own decisions), and
    once pooled over all types (thresholds[r].confidence, informational: basal-serve never applies it). A type
    without a certified threshold of its own ("confidence": null, or no entry) has every automatic action of act and
    tool with that max_error deferred ("no certified threshold for this type at this max_error"). A few hundred
    decisions usually certify 5%; 1% needs at least 299 accepted FIT decisions of the type without an error (about
    1,000 items of the type or more);
  * gates: ece_lofo per type, for every family the temperatures fitted on the FIT decisions of the other families and
    the ECE measured on its CERT decisions (mean and max over the families with at least --min-family CERT decisions
    of the type, and pooled over all of them), next to --reference (basal-1.5: the same statistic on our calibration
    split); automated_at, the share of CERT decisions at or above their type's certified threshold. --strict exits with
    status 1 when a type's mean LOFO ECE is above the reference (a type without a family of --min-family CERT
    decisions is not tested: a warning on stderr and in the report, status unchanged);
  * multi: per label set (its labels and fallback; id = basal.sets.label_set_id) the fallback_threshold with the best
    mean IoU on FIT over a log grid (null = the default rule wins), with IoU, empty-set accuracy and false-empty rate
    on CERT next to the default. basal-serve uses it for a request with the same label set and fallback and no
    fallback_threshold of its own;
  * tasks: per task (classify, verify, extract; every claim and field is one decision) the certified threshold of each
    target error, by the same bound on FIT, with its coverage, error and bound on CERT ("task_thresholds" and
    "task_gates"). The task thresholds a shipped file holds are kept only when the temperatures are (no decision
    questions in the input); otherwise a task without labelled questions has none and its answers report
    automated: null.
  * fitted_on names the model (and its Hugging Face revision): basal-serve warns when a --calibration file was fitted
    on another model or revision, and refuses it with --strict-calibration.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from . import tasks as task_types
from .engine import resolve
from .prompt import max_options, model_format, model_layout
from .server import (MODES, Server, _text, add_engine_args, calibration_file, calibration_name, calibration_orders,
                     certified, default_mode, load_calibration, model_identity, refuse_lean_flex, temper, temperature,
                     to_items)
from .sets import label_set_id
from .sets import select as select_set

PRIMS = ("choice", "noul", "score")
BUCKETS = (("2-10", 2, 10), ("11-26", 11, 26), ("27-255", 27, 255))  # option-count buckets of calibration file 2
ALPHA = 0.05        # one-sided 95% bounds
MIN_ACCEPTED = 50   # a threshold must accept at least this many FIT decisions
MIN_BLEND = 10      # from this many FIT decisions (below --min-per-type) a temperature is blended, not kept
RECOMMENDED = 200   # fewer labelled decisions: loose bounds
T_GRID = (0.25, 8.0)  # temperatures searched (as for the shipped files)
METHOD = "bound-controlled (one-sided 95% Clopper-Pearson on FIT)"
POOLED_NOTE = ("informational: \"confidence\" here is one threshold pooled over all types; basal-serve applies only "
               "per_type[type].confidence and defers a type without one")
YES, NO = {"yes", "true", "tak"}, {"no", "false", "nie"}
TASKS = ("classify", "verify", "extract")  # the tasks with an `automated` flag (compress has none)
VERDICTS = ("supported", "contradicted", "not_stated")
# what a file with task questions only does not take over from the file it keeps (written anew, or merged)
KEPT_OUT = ("fitted_on", "certified_on", "baseline", "engine", "task_thresholds", "task_gates", "decision_calibration")
# mean leave-one-family-out ECE of basal-1.5 (4.5B) per type on our calibration split (--reference basal-1.5)
REFERENCE = {"basal-1.5": {"choice": 0.082, "noul": 0.062, "score": 0.109}}
# fallback_threshold candidates of the multi fit (None: the default rule), 0.001-0.81 log-spaced and 0.05-0.95
FALLBACK_GRID = (None,) + tuple(sorted({round(0.001 * 1.25 ** k, 4) for k in range(31)} |
                                       {x / 100 for x in range(5, 96, 5)}))


class InputError(ValueError):
    pass


# -- input ---------------------------------------------------------------------------------------------------------
def gold_index(q, item):
    """The gold answer of a question -> index of the correct option among the item's keys."""
    g, keys = q["gold"], [str(k) for k in item["keys"]]
    if keys == ["true", "false"]:  # noul, or act with yes/no outcomes
        s = str(g).strip().lower() if isinstance(g, (str, bool)) else None
        if s in YES or s in NO:
            return 0 if s in YES else 1
        raise ValueError(f'gold must be "yes" or "no" (or true / false), got {g!r}')
    crit = q.get("criteria") or q.get("levels") or []
    if isinstance(g, int) and not isinstance(g, bool) and item["type"] == "score" and isinstance(crit, list):
        if 0 <= g < len(keys):
            return g
        raise ValueError(f"gold level {g} is out of range 0..{len(keys) - 1}")
    if not isinstance(g, (dict, list)) and str(g) in keys:
        return keys.index(str(g))
    texts = list(crit.values()) if isinstance(crit, dict) else list(crit)
    hits = {i for i, t in enumerate(texts) if t is not None and _text(t) == _text(g)}
    hits |= {i for i, t in enumerate(item["options"]) if t == _text(g)}
    if len(hits) == 1:
        return hits.pop()
    if hits:
        raise ValueError(f"gold {g!r} matches several options; give the key")
    what = "a level index, key or text" if item["type"] == "score" else "an option key or description"
    raise ValueError(f"gold {g!r} is not an option (give {what}; keys: {', '.join(keys[:10])})")


def gold_set(q, item):
    """The gold of a multi question (the labels that apply) -> sorted label indices."""
    g, keys = q["gold"], [str(k) for k in item["keys"]]
    if not isinstance(g, list):
        raise ValueError("gold of a multi question must be the list of labels that apply ([] = none)")
    out = set()
    for x in g:
        if not isinstance(x, (dict, list)) and str(x) in keys:
            out.add(keys.index(str(x)))
            continue
        hits = [i for i, t in enumerate(item["labels"]) if t == _text(x)]
        if len(hits) != 1:
            raise ValueError(f"gold label {x!r} " + ("matches several labels; give the key" if hits else
                                                      f"is not a label (keys: {', '.join(keys[:10])})"))
        out.add(hits[0])
    return sorted(out)


def simple(rec):
    """A simple item of basal-run / basal-bench (state, question, options, type, gold index) -> a labelled request."""
    from .run import to_request
    body = to_request(rec)
    q = body["questions"]["q"]
    if "gold" in rec:
        g = rec["gold"]
        q["gold"] = {0: "yes", 1: "no"}.get(g, g) if q["type"] == "noul" and not isinstance(g, bool) else g
    return dict(body, **{k: rec[k] for k in ("id", "group", "family") if k in rec})


def _at(obj, path):
    """(found, value) at a dotted field path of a nested gold object."""
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return False, None
        obj = obj[k]
    return True, obj


def field_gold(path, kind, spec, v):
    """The gold of an extract field -> the value basal-serve would return for it (None = not stated)."""
    if v is None:
        return None
    if kind == "enum":
        vals = [c for c, _ in task_types._consts(spec)]
        hit = [c for c in vals if c == v or str(c) == str(v)]
        if not hit:
            raise ValueError(f"field {path!r}: gold {v!r} is not one of its values {vals[:10]}")
        return hit[0]
    if kind == "boolean":
        s = str(v).strip().lower()
        if s in YES or s in NO:
            return s in YES
        raise ValueError(f"field {path!r}: gold {v!r} is not true / false")
    if kind in ("number", "integer"):
        x = v if isinstance(v, (int, float)) and not isinstance(v, bool) else task_types.parse_number(str(v))
        if x is None:
            raise ValueError(f"field {path!r}: gold {v!r} is not a number")
        return x
    if isinstance(v, (dict, list)):
        raise ValueError(f"field {path!r}: gold must be the text of the value")
    return str(v)


def field_correct(kind, value, gold):
    """Whether an extract field's value is its gold: numbers by value, text up to spacing and case."""
    if value is None or gold is None:
        return value is None and gold is None
    if kind in ("number", "integer"):
        return abs(float(value) - float(gold)) <= 1e-9 * max(1.0, abs(float(gold)))
    if kind == "string":
        norm = lambda s: " ".join(str(s).split()).casefold()
        return norm(value) == norm(gold)
    return value == gold


def task_units(name, q, gold, state):
    """A labelled task question -> its labelled decisions [(part, gold)]: classify one (part None), verify one per
    labelled claim, extract one per labelled field. The question must expand as basal-serve expands it."""
    t = q["type"]
    if t == "compress":
        raise ValueError("compress has no automated flag, so there is nothing to certify: leave it out")
    task_types.expand({"state": state, "questions": {name: q}})  # the errors basal-serve would give
    if t == "classify":
        if q.get("multi_label"):
            raise ValueError("multi-label classify has no automated flag (it reports set_confidence): ask it "
                             "single-label, or each label as a noul question")
        keys = list(task_types._labels(q.get("labels") or q.get("criteria"), name)) + (["none"] if q.get("none") else [])
        if isinstance(gold, (dict, list)) or str(gold) not in keys:
            raise ValueError(f"gold {gold!r} is not a label (keys: {', '.join(keys[:10])})")
        return [(None, str(gold))]
    if t == "verify":
        claims = q["claims"]
        keys = [str(i) for i in range(len(claims))] if isinstance(claims, list) else [str(k) for k in claims]
        if isinstance(gold, list):
            if len(gold) != len(keys):
                raise ValueError(f"gold has {len(gold)} verdicts for {len(keys)} claims")
            gold = dict(zip(keys, gold))
        if not isinstance(gold, dict):
            raise ValueError("gold of verify: {claim key: verdict}, or a list in the order of the claims "
                             "(null = not labelled)")
        allowed = ("supported", "not_supported") if q.get("binary") else VERDICTS
        out = []
        for k, v in gold.items():
            if str(k) not in keys:
                raise ValueError(f"gold names claim {k!r}, which is not one of {keys[:10]}")
            if v is None:
                continue
            if isinstance(v, bool) and q.get("binary"):
                v = "supported" if v else "not_supported"
            v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
            if v not in allowed:
                raise ValueError(f"claim {k!r}: verdict {v!r} is not one of {', '.join(allowed)}")
            out.append((str(k), v))
        if not out:
            raise ValueError("no claim has a verdict")
        return out
    if not isinstance(gold, dict):
        raise ValueError("gold of extract: the field values nested as in the schema (null = not stated; a field left "
                         "out is not labelled)")
    out = []
    for path, kind, spec in task_types.leaves(q.get("schema") or {}):
        found, v = _at(gold, path)
        if found and kind != "enum_list":  # a label set has no certified threshold
            out.append((path, (kind, field_gold(path, kind, spec, v))))
    if not out:
        raise ValueError("gold labels no field (enum, boolean, string, number or integer) of the schema")
    return out


def load(path):
    """Labelled JSONL -> (requests to send, decisions): the decision questions only (read() also returns the task
    questions). Raises InputError listing every problem with its line."""
    bodies, decisions, _, _ = read(path)
    return bodies, decisions


def read(path):
    """Labelled JSONL -> (decision requests, decisions, task requests, labelled task decisions). Raises InputError
    listing every problem with its line."""
    errors, bodies, decisions, task_bodies, units = [], [], [], [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not JSON ({e.msg})")
            continue
        if isinstance(rec, dict) and "questions" not in rec and "question" in rec:
            try:
                rec = simple(rec)
            except (ValueError, KeyError, TypeError) as e:
                errors.append(f"line {n}: {e}")
                continue
        if not isinstance(rec, dict) or "state" not in rec or not isinstance(rec.get("questions"), dict) \
                or not rec["questions"]:
            errors.append(f'line {n}: a request needs "state" and a non-empty "questions" object')
            continue
        where = f"line {n}" + (f" (id {rec['id']})" if "id" in rec else "")
        if rec.get("facts", "off") not in ("off", "auto"):
            errors.append(f'{where}: facts must be "off" or "auto"')
            continue
        body = {k: v for k, v in rec.items() if k not in ("id", "group", "family")}
        body["questions"], task_body = {}, dict(body, questions={})
        for name, q in rec["questions"].items():
            at = f"{where}, question {name!r}"
            if not isinstance(q, dict):
                errors.append(f"{at}: not an object")
                continue
            t = q.get("type", "choice")
            if t in task_types.TASKS:  # classify / verify / extract: their task thresholds (in process, served)
                if "gold" not in q:
                    errors.append(f'{at}: no "gold" answer')
                    continue
                clean = {k: v for k, v in q.items() if k not in ("gold", "group", "family")}
                try:
                    labelled = task_units(name, clean, q["gold"], rec["state"])
                except (ValueError, KeyError, TypeError) as e:
                    errors.append(f"{at}: {str(e).removeprefix(f'question {name!r}: ')}")
                    continue
                task_body["questions"][name] = clean
                group = str(q.get("group", rec.get("group", rec.get("id", n))))
                units += [dict(line=n, req=len(task_bodies), name=name, task=t, part=part, gold=g, group=group)
                          for part, g in labelled]
                continue
            if t == "multi" and q.get("fallback") is None:
                errors.append(f'{at}: multi needs a "fallback" label (its fallback_threshold is fitted); to calibrate '
                              "the labels, ask each one as a noul question with its own gold")
                continue
            if t not in ("choice", "noul", "score", "act", "multi"):
                errors.append(f"{at}: unknown type {t!r} (choice, noul, score, act, multi, classify, verify or "
                              "extract)")
                continue
            if "gold" not in q:
                errors.append(f'{at}: no "gold" answer')
                continue
            clean = {k: v for k, v in q.items()
                     if k not in ("gold", "group", "family", "evidence", "fallback_threshold")}
            try:
                item = to_items(rec["state"], {name: clean}, max_options(2))[0]  # the model's limit: when served
                gold = gold_set(q, item) if t == "multi" else gold_index(q, item)
            except (ValueError, KeyError, TypeError) as e:
                errors.append(f"{at}: {str(e).removeprefix(f'question {name!r}: ')}")
                continue
            body["questions"][name] = clean
            group = q.get("group", rec.get("group", rec.get("id", n)))
            d = dict(line=n, req=len(bodies), name=name, prim=item.get("base", t), gold=gold,
                     keys=[str(k) for k in item["keys"]], group=str(group),
                     family=str(q.get("family", rec.get("family", name))))
            if t == "multi":
                d.update(prim="multi", set=dict(
                    id=label_set_id(item["keys"], item["labels"], item["fallback"]), objective=item["objective"],
                    threshold=item["threshold"], min=item["min"], max=item["max"], exclusive=item["exclusive"],
                    fallback=item["fallback"]))
            decisions.append(d)
        if body["questions"]:
            bodies.append(body)
        if task_body["questions"]:
            task_bodies.append(task_body)
    if errors:
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise InputError(f"{path}: {len(errors)} problem(s)\n  " + "\n  ".join(errors[:20]) + more)
    if not decisions and not units:
        raise InputError(f"{path}: no labelled questions")
    return bodies, decisions, task_bodies, units


# -- probabilities ---------------------------------------------------------------------------------------------------
async def collect(decide, bodies, concurrency=32):
    """Every request through `decide` (Server.decide in process, or HTTP) -> responses (an Exception on failure)."""
    sem, done, step = asyncio.Semaphore(concurrency), 0, max(1, len(bodies) // 10)

    async def one(b):
        nonlocal done
        async with sem:
            try:
                r = await decide(b)
            except Exception as e:  # noqa: BLE001
                r = e
        done += 1
        if done % step == 0 or done == len(bodies):
            print(f"  {done}/{len(bodies)} requests", file=sys.stderr, flush=True)
        return r
    return await asyncio.gather(*(one(b) for b in bodies))


async def in_process(srv, bodies, concurrency):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        return await collect(srv.decide, bodies, concurrency)
    finally:
        worker.cancel()


async def two_passes(srv, bodies, task_bodies, concurrency, settle):
    """In process: the decision requests without calibration, then settle(responses) -> the calibration to serve, and
    the task requests served with it (their thresholds are certified on the answers as served). One event loop for
    both: the server's queue belongs to it."""
    worker = asyncio.get_running_loop().create_task(srv.worker())
    try:
        responses = await collect(srv.decide, bodies, concurrency) if bodies else []
        cal = settle(responses)
        if not task_bodies:
            return responses, []
        print(f"  task requests, served with the {'fitted' if bodies else 'kept'} calibration", file=sys.stderr)
        srv.cal, srv.temps = cal, cal.get("temperature_per_prim", {})
        return responses, await collect(srv.decide, task_bodies, concurrency)
    finally:
        worker.cancel()


async def over_http(endpoint, bodies, concurrency):
    import httpx
    async with httpx.AsyncClient(timeout=600) as c:
        async def decide(b):
            r = await c.post(endpoint, json=b)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}" +
                                   (f" [request {r.headers['x-request-id']}]" if "x-request-id" in r.headers else ""))
            return r.json()
        return await collect(decide, bodies, concurrency)


def uncalibrated(entry, base):
    """The GET /v1/models entry of a running server -> the entry; refuses a server that applies temperatures."""
    temps = entry.get("temperature_per_prim")
    if temps is None:
        print(f"warning: {base} does not report its temperatures (basal-serve before 2.0.0); make sure it runs with "
              "--no-calibration", file=sys.stderr)
    elif any(float(t) != 1.0 for t in temps.values()):
        raise SystemExit(f"{base} serves calibrated probabilities (temperatures {temps}); start basal-serve with "
                         "--no-calibration for basal-calibrate")
    return entry


def raw_rows(decisions, responses):
    """Decisions + responses -> (rows with the uncalibrated probabilities `p` over their keys, failed requests)."""
    rows, failed = [], {}
    for d in decisions:
        r = responses[d["req"]]
        if isinstance(r, Exception):
            failed[d["req"]] = f"line {d['line']}: {r}"
            continue
        p = r["answers"][d["name"]]["probabilities"]
        rows.append(dict(d, p=[float(p[k]) for k in d["keys"]]))
    return rows, list(failed.values())


# -- method ----------------------------------------------------------------------------------------------------------
def is_cert(group):
    """FIT / CERT split by group: 30% of the groups certify (the hash the shipped files used)."""
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10 < 3


def fit_temperature(rows):
    """Temperature minimising the NLL of the gold answers on a log grid over T_GRID."""
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


def served(rows, temps, buckets=None):
    """(confidence, correct) of every row as basal-serve returns it with these temperatures (same float32 maths):
    temps per type, buckets {type: {"lo-hi": T}} (calibration file 2) before them."""
    cal = dict(temperature_per_prim=temps, temperature_per_bucket=buckets or {})
    out = []
    for r in rows:
        p = temper(torch.tensor(r["p"]), temperature(cal, r["prim"], len(r["p"]))).tolist()
        out.append((float(max(p)), int(np.argmax(p)) == r["gold"]))
    return out


def ece(cs, bins=10):
    """Expected calibration error of the top answer (equal-width bins)."""
    if not cs:
        return None
    c, ok = np.array([x for x, _ in cs]), np.array([o for _, o in cs], float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            e += m.mean() * abs(c[m].mean() - ok[m].mean())
    return float(e)


_LGAMMA = [0.0]


def cp_upper(e, n, alpha=ALPHA):
    """One-sided Clopper-Pearson upper bound of a binomial error rate (bisection on the binomial CDF, in log space)."""
    if n == 0:
        return 1.0
    while len(_LGAMMA) <= n + 1:
        _LGAMMA.append(math.lgamma(len(_LGAMMA) + 1))  # _LGAMMA[i] = log(i!)
    lf = np.array(_LGAMMA[: n + 1])
    i = np.arange(e + 1)
    comb = lf[n] - lf[i] - lf[n - i]

    def cdf(p):
        if p >= 1.0:
            return 1.0 if e >= n else 0.0
        t = comb + i * math.log(p) + (n - i) * math.log1p(-p)
        m = t.max()
        return math.exp(m) * float(np.exp(t - m).sum())
    lo, hi = e / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > alpha else (lo, mid)
    return hi


def levels(cs):
    """(confidence, correct) -> [(confidence, accepted, errors)] from the most confident level down, ties together."""
    by = defaultdict(lambda: [0, 0])
    for c, ok in cs:
        by[c][0] += 1
        by[c][1] += not ok
    out, n, e = [], 0, 0
    for c in sorted(by, reverse=True):
        n, e = n + by[c][0], e + by[c][1]
        out.append((c, n, e))
    return out


def threshold(cs, risk, alpha=ALPHA, min_n=MIN_ACCEPTED):
    """Most permissive confidence whose accepted decisions have an upper bound <= risk, scanned from the most confident
    level down and stopped at the first level that fails after one passed (fixed sequence, as for the shipped files)."""
    thr = None
    for c, n, e in levels(cs):
        if n < min_n:
            continue
        if e / n <= risk and cp_upper(e, n, alpha) <= risk:
            thr = c
        elif thr is not None:
            break
    return thr


def accepted_needed(risk, errors=0, alpha=ALPHA, min_n=MIN_ACCEPTED):
    """The fewest accepted decisions with `errors` errors whose one-sided upper bound is at most `risk` (0 / 1 / 2
    errors: 299 / 473 / 628 at 1%, 59 / 93 / 124 at 5%)."""
    if errors == 0:
        return max(min_n, math.ceil(math.log(alpha) / math.log(1 - risk)))
    lo = hi = accepted_needed(risk, 0, alpha, min_n)
    while cp_upper(errors, hi, alpha) > risk:
        lo, hi = hi, hi * 2
    while lo < hi:  # lo fails (or equals hi), hi passes
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if cp_upper(errors, mid, alpha) > risk else (lo, mid)
    return hi


def sample_size(decisions, units, targets):
    """Before any readout: per question type (decisions) and task (units) and per target, whether its FIT decisions
    (the groups outside CERT, about 70%) could certify the target at all, i.e. with every one of them accepted and
    correct -> warnings for those that cannot."""
    count = defaultdict(lambda: [0, 0])  # name -> [labelled decisions, FIT decisions]
    for name, group in [(d["prim"], d["group"]) for d in decisions if d["prim"] != "multi"] + \
            [(u["task"], u["group"]) for u in units]:
        count[name][0] += 1
        count[name][1] += not is_cert(group)
    out = []
    for name, (n, n_fit) in count.items():
        what = {"verify": "claims", "extract": "fields"}.get(name, "questions")
        for risk in sorted(targets):
            need = accepted_needed(risk)
            if n_fit >= need:
                continue
            falls = (f"its answers report automated: null at this max_error" if name in TASKS else
                     f"basal-serve defers the automatic actions of act and tool on {name} at this max_error")
            out.append(f"{name}: max_error {risk:g} needs >= {need} accepted FIT decisions without an error "
                       f"({accepted_needed(risk, 1)} with 1 error, {accepted_needed(risk, 2)} with 2); this file gives "
                       f"{n_fit} FIT decisions of {n} (FIT: about 70% of the groups); label about "
                       f"{max(1, math.ceil(need / 0.7) - n)} more {name} {what}. {name} will not be certified at "
                       f"max_error {risk:g}: {falls}")
    return out


def cert_note(name, risk, x):
    """A written threshold (entry x: confidence, cert_accepted, cert_error, cert_upper95) that CERT does not confirm:
    its observed error is above the target, or too few accepted CERT decisions bring the upper bound below it. The
    threshold is certified on FIT and stays; None when CERT confirms it."""
    if x.get("confidence") is None:
        return None
    n = x.get("cert_accepted") or 0
    e = round((x.get("cert_error") or 0.0) * n)
    errs, need = f"{e} error{'' if e == 1 else 's'}", accepted_needed(risk)
    if n == 0:
        return (f"{name} at max_error {risk:g}: certified on FIT; CERT accepts no decision at this threshold, so it "
                f"cannot confirm it (0 errors would need >= {need} accepted CERT decisions)")
    if e / n > risk:
        return (f"{name} at max_error {risk:g}: certified on FIT, but CERT has {n} accepted, {errs} "
                f"({pct(e / n)}), above the target: the threshold does not hold on CERT; label more data before "
                "relying on it")
    if x.get("cert_upper95") is not None and x["cert_upper95"] > risk:
        return (f"{name} at max_error {risk:g}: certified on FIT; CERT has {n} accepted, {errs}, upper bound "
                f"{100 * x['cert_upper95']:.2f}% > {100 * risk:g}%, too few to confirm on CERT (0 errors would need "
                f">= {need} accepted CERT decisions)")
    return None


def cert_notes(report, task_report, targets):
    """cert_note for every written threshold of the decision types and the tasks."""
    out = []
    for risk in targets if report else ():
        for t, x in ((report["targets"][risk]["fitted"].get("per_type")) or {}).items():
            out.append(cert_note(t, risk, x))
    for t, e in (task_report or {}).items():
        for risk, x in e["targets"].items():
            out.append(cert_note(t, float(risk), x))
    return [x for x in out if x]


def no_threshold_reason(cs, risk, alpha=ALPHA, min_n=MIN_ACCEPTED):
    need = accepted_needed(risk, 0, alpha, min_n)  # accepted decisions without an error
    if len(cs) < need:
        return f"a {risk:g} bound needs at least {need} accepted FIT decisions without an error; FIT has {len(cs)}"
    best = min(((cp_upper(e, n, alpha), n) for _, n, e in levels(cs) if n >= min_n), default=None)
    return (f"no confidence level on FIT meets the bound (lowest upper bound {best[0]:.3f}, "
            f"at {best[1] / len(cs):.0%} coverage)") if best else "no confidence level on FIT accepts enough decisions"


def gate(cs, thr):
    acc = [ok for c, ok in cs if thr is not None and c >= thr]
    e = len(acc) - sum(acc)
    return dict(coverage=len(acc) / len(cs) if cs else None, accepted=len(acc), errors=e,
                error=e / len(acc) if acc else None, upper95=cp_upper(e, len(acc)) if acc else None)


def blend(T_fit, T_old, n, min_per_type):
    """A temperature fitted on n < min_per_type decisions, blended in log space with the one it replaces in proportion
    n / min_per_type (so the full fit takes over continuously at min_per_type) -> (temperature, weight of the fit)."""
    w = n / min_per_type
    return float(math.exp(w * math.log(T_fit) + (1 - w) * math.log(T_old))), w


def fit_temps(fit_rows, shipped_T, shipped_B, min_per_type, fmt, min_blend=MIN_BLEND):
    """Temperatures per type (and per choice bucket for format 2) fitted on FIT rows where there are enough decisions,
    blended with the shipped ones from min_blend decisions (0: never), else the shipped ones -> (temps, source,
    buckets, bucket source)."""
    temps, source = {}, {}
    for t in PRIMS:
        rs = [r for r in fit_rows if r["prim"] == t]
        if len(rs) >= min_per_type:
            temps[t], source[t] = fit_temperature(rs), f"fitted on {len(rs)} FIT decisions"
        elif min_blend and len(rs) >= min_blend:
            temps[t], w = blend(fit_temperature(rs), shipped_T.get(t, 1.0), len(rs), min_per_type)
            source[t] = (f"blended: {w:.0%} fitted on {len(rs)} FIT decisions, {1 - w:.0%} "
                         + ("shipped" if t in shipped_T else "default 1.0"))
        elif t in shipped_T:
            temps[t] = shipped_T[t]
            source[t] = f"shipped ({len(rs)} FIT decisions < {min_per_type})" if rs else "shipped (no decisions)"
    buckets, bsource = {}, {}
    if fmt == 2:
        for name, lo, hi in BUCKETS:
            rs = [r for r in fit_rows if r["prim"] == "choice" and lo <= len(r["p"]) <= hi]
            old = (shipped_B.get("choice") or {}).get(name)
            if len(rs) >= min_per_type:
                buckets.setdefault("choice", {})[name] = fit_temperature(rs)
                bsource[name] = f"fitted on {len(rs)} FIT decisions"
            elif min_blend and len(rs) >= min_blend:
                buckets.setdefault("choice", {})[name], w = blend(
                    fit_temperature(rs), old if old is not None else temps.get("choice", 1.0), len(rs), min_per_type)
                bsource[name] = (f"blended: {w:.0%} fitted on {len(rs)} FIT decisions, {1 - w:.0%} "
                                 + ("shipped bucket" if old is not None else "type temperature"))
            elif old is not None:
                buckets.setdefault("choice", {})[name] = old
                bsource[name] = f"shipped ({len(rs)} FIT decisions < {min_per_type})"
            else:
                bsource[name] = f"type temperature ({len(rs)} FIT decisions < {min_per_type})"
    return temps, source, buckets, bsource


def lofo(fit_rows, cert_rows, shipped_T, shipped_B, min_per_type, fmt, min_family=30, reference=None, draws=50, seed=0,
         min_blend=MIN_BLEND):
    """Leave-one-family-out ECE per type: for every family the temperatures are fitted on the FIT decisions of the
    other families and applied to its CERT decisions. mean / max over the families with >= min_family CERT decisions of
    the type, mean_floor (the mean a perfectly calibrated predictor with the same confidences would show, `draws`
    simulations: small families make the per-family ECE noisy), pooled over all held-out CERT decisions."""
    fam = lambda r: r.get("family", "")
    fams = sorted({fam(r) for r in fit_rows + cert_rows})
    types = [t for t in PRIMS if any(r["prim"] == t for r in cert_rows)]
    if len(fams) < 2:
        return {t: dict(n=0, n_families=0, mean=None, pooled=None, reference=(reference or {}).get(t), regression=None,
                        reason='one family: give "family" on the lines or questions') for t in types}
    per, pooled, rng = defaultdict(list), defaultdict(list), np.random.default_rng(seed)
    for f in fams:
        held = [r for r in cert_rows if fam(r) == f]
        if not held:
            continue
        temps, _, buckets, _ = fit_temps([r for r in fit_rows if fam(r) != f], shipped_T, shipped_B,
                                         min_per_type, fmt, min_blend)
        for t in types:
            cs = served([r for r in held if r["prim"] == t], temps, buckets)
            pooled[t] += cs
            if len(cs) >= min_family:
                floor = float(np.mean([ece([(c, rng.random() < c) for c, _ in cs]) for _ in range(draws)]))
                per[t].append((f, len(cs), ece(cs), floor))
    out = {}
    for t in types:
        e, ref = dict(n=len(pooled[t]), n_families=len(per[t]), pooled=ece(pooled[t])), (reference or {}).get(t)
        if per[t]:
            worst = max(per[t], key=lambda x: x[2])
            e.update(mean=float(np.mean([x[2] for x in per[t]])), max=worst[2], worst_family=worst[0],
                     mean_floor=float(np.mean([x[3] for x in per[t]])))
        else:
            e.update(mean=None, reason=f"no family has {min_family} CERT decisions of this type")
        e.update(reference=ref, regression=None if ref is None or e["mean"] is None else e["mean"] > ref)
        out[t] = e
    return out


def set_probs(r, T):
    """P(label applies) of a multi row as served with the noul temperature T."""
    p = torch.tensor([[x, 1 - x] for x in r["p"]], dtype=torch.float32)
    return dict(zip(r["keys"], temper(p, T)[:, 0].tolist()))


def set_metrics(prepared, fthr):
    """[(row, probs)] -> mean IoU, empty-set accuracy (gold empty, nothing selected) and false-empty rate (gold not
    empty, nothing selected) with this fallback_threshold (None: the default rule)."""
    iou, empty, false_empty = [], [], []
    for r, probs in prepared:
        s = r["set"]
        sel, _, _ = select_set(probs, s["objective"], s["threshold"], s["min"], s["max"], s["exclusive"], s["fallback"],
                               fthr)
        S, G = set(sel), {r["keys"][i] for i in r["gold"]}
        iou.append(1.0 if not S and not G else len(S & G) / len(S | G))
        (false_empty if G else empty).append(not S)
    mean = lambda x: float(np.mean(x)) if x else None
    return dict(iou=mean(iou), empty_accuracy=mean(empty), false_empty=mean(false_empty))


def fit_fallback(sets, T, min_per_set=30):
    """Multi rows -> {label set id: entry}: per label set the fallback_threshold of FALLBACK_GRID with the best mean
    IoU on FIT (None, the default rule, wins ties), and its CERT metrics next to the default's."""
    by = defaultdict(list)
    for r in sets:
        by[r["set"]["id"]].append(r)
    out = {}
    for sid, rs in by.items():
        fit_p = [(r, set_probs(r, T)) for r in rs if not is_cert(r["group"])]
        cert_p = [(r, set_probs(r, T)) for r in rs if is_cert(r["group"])]
        e = dict(fallback=rs[0]["set"]["fallback"], labels=rs[0]["keys"], n=len(rs), n_fit=len(fit_p),
                 n_cert=len(cert_p))
        if len(fit_p) < min_per_set:
            out[sid] = dict(e, fallback_threshold=None, reason=f"{len(fit_p)} FIT questions < {min_per_set}")
            continue
        best = max(FALLBACK_GRID, key=lambda t: set_metrics(fit_p, t)["iou"])
        fitted, default = set_metrics(cert_p, best), set_metrics(cert_p, None)
        out[sid] = dict(e, fallback_threshold=best, method="mean IoU on FIT over a log grid (null: the default rule)",
                        cert={**fitted, **{f"{k}_default": v for k, v in default.items()}})
    return out


def per_type_thresholds(fit_rows, cert_rows, temps, buckets, risk):
    """The certified threshold of every type on its own decisions: a temperature changes neither the order of a type's
    yes/no confidences nor (up to ties) of its other confidences, so the accepted decisions do not depend on it, while
    one threshold pooled over the types moves with the temperatures of the others."""
    out = {}
    for t in PRIMS:
        cf = served([r for r in fit_rows if r["prim"] == t], temps, buckets)
        if not cf:
            continue
        cc = served([r for r in cert_rows if r["prim"] == t], temps, buckets)
        thr = threshold(cf, risk)
        if thr is None:
            out[t] = dict(confidence=None, reason=no_threshold_reason(cf, risk))
            continue
        g_fit, g_cert = gate(cf, thr), gate(cc, thr)
        out[t] = dict(confidence=thr, fit_coverage=g_fit["coverage"], fit_error=g_fit["error"],
                      fit_upper95=g_fit["upper95"], cert_coverage=g_cert["coverage"], cert_error=g_cert["error"],
                      cert_upper95=g_cert["upper95"], cert_accepted=g_cert["accepted"])
    return out


def effective_gate(rows, temps, buckets, entry):
    """Coverage and error of the decisions basal-serve automates with this threshold entry: each type at its own
    certified threshold; a type without one is not automated (deferred)."""
    if not rows:
        return dict(coverage=None, error=None)
    acc = []
    for r, (c, ok) in zip(rows, served(rows, temps, buckets)):
        thr = type_threshold(entry, r["prim"])
        if thr is not None and c >= thr:
            acc.append(ok)
    return dict(coverage=len(acc) / len(rows), error=(len(acc) - sum(acc)) / len(acc) if acc else None)


def served_gate(rows, temps, buckets, cal, risk):
    """Share and error of the decisions basal-serve automates at max_error `risk` with the calibration `cal` (its
    temperatures given as temps / buckets): each type at the threshold the server applies to it (server.certified)."""
    if not rows:
        return dict(coverage=None, error=None)
    acc = []
    for r, (c, ok) in zip(rows, served(rows, temps, buckets)):
        thr = certified(cal, risk, r["prim"])
        if thr is not None and c >= thr:
            acc.append(ok)
    return dict(coverage=len(acc) / len(rows), error=(len(acc) - sum(acc)) / len(acc) if acc else None)


def floor_thresholds(thresholds, targets):
    """A confidence certified at a target error is certified at every looser target too (its bound is below both), so a
    looser target never keeps a stricter threshold than a tighter one: the scan stops at the first failing level after
    a pass, and at a loose target it can pass early on a small top level and stop on a few errors below it (basal-omni
    mini FP8, 17 Oct 2026: 5% at 0.9999 against 1% at 0.9587). Floors the pooled and per-type entries in place, marking
    them "floored_from"; returns the targets whose entries changed."""
    changed = set()
    rs = sorted(float(r) for r in targets)
    for lo, hi in zip(rs, rs[1:]):
        a, b = thresholds[str(lo)], thresholds[str(hi)]
        for t, e in (a.get("per_type") or {}).items():
            eh = (b.get("per_type") or {}).get(t) or {}
            if e.get("confidence") is not None and (eh.get("confidence") is None or eh["confidence"] > e["confidence"]):
                b.setdefault("per_type", {})[t] = dict(e, floored_from=str(lo))
                changed.add(str(hi))
        if a.get("confidence") is not None and (b.get("confidence") is None or b["confidence"] > a["confidence"]):
            b.update({k: v for k, v in a.items() if k not in ("per_type", "pooled")}, floored_from=str(lo))
            changed.add(str(hi))
    return changed


def type_threshold(entry, q_type):
    """The threshold basal-serve applies to a question type: its own certified one, else None (the pooled one is
    never applied)."""
    return ((((entry or {}).get("per_type") or {}).get(q_type)) or {}).get("confidence")


def fit(rows, baseline, targets, min_per_type=50, fmt=1, min_family=30, reference=None, min_per_set=30,
        min_blend=MIN_BLEND):
    """Rows with uncalibrated probabilities -> (calibration dict, report for the table). fmt 2: calibration file 2,
    with the choice temperatures per option-count bucket. Multi rows fit the fallback_threshold of their label set."""
    sets = [r for r in rows if r["prim"] == "multi"]
    rows = [r for r in rows if r["prim"] != "multi"]
    fit_rows = [r for r in rows if not is_cert(r["group"])]
    cert_rows = [r for r in rows if is_cert(r["group"])]
    shipped_T = baseline.get("temperature_per_prim", {})
    shipped_B = baseline.get("temperature_per_bucket", {}) if fmt == 2 else {}
    temps, source, buckets, bsource = fit_temps(fit_rows, shipped_T, shipped_B, min_per_type, fmt, min_blend)
    cs_fit, cs_cert = served(fit_rows, temps, buckets), served(cert_rows, temps, buckets)
    shipped_cert = served(cert_rows, shipped_T, shipped_B)
    thresholds, report = {}, dict(types={}, targets={}, n_fit=len(fit_rows), n_cert=len(cert_rows), compared={})
    automated, automated_pooled, automated_error = {}, {}, {}
    for risk in targets:
        thr = threshold(cs_fit, risk)
        g_fit, g_cert = gate(cs_fit, thr), gate(cs_cert, thr)
        if thr is None:
            entry = dict(confidence=None, method=METHOD, reason=no_threshold_reason(cs_fit, risk))
        else:
            entry = dict(confidence=thr, method=METHOD, fit_coverage=g_fit["coverage"], fit_error=g_fit["error"],
                         fit_upper95=g_fit["upper95"], cert_coverage=g_cert["coverage"], cert_error=g_cert["error"],
                         cert_upper95=g_cert["upper95"], cert_accepted=g_cert["accepted"])
        entry["per_type"] = per_type_thresholds(fit_rows, cert_rows, temps, buckets, risk)
        entry["pooled"] = POOLED_NOTE
        thresholds[str(float(risk))] = entry
        eff = effective_gate(cert_rows, temps, buckets, entry)
        automated[str(float(risk))] = eff["coverage"]
        automated_error[str(float(risk))] = eff["error"]
        automated_pooled[str(float(risk))] = (g_cert["coverage"] if thr is not None else 0.0) if cs_cert else None
        if (baseline.get("thresholds") or {}).get(str(float(risk))):  # the compared file as basal-serve applies it
            report["compared"][str(float(risk))] = served_gate(cert_rows, shipped_T, shipped_B, baseline, risk)
        old = (baseline.get("thresholds") or {}).get(str(float(risk)), {}).get("confidence")
        report["targets"][risk] = dict(shipped=old, shipped_cert=gate(shipped_cert, old) if old is not None else None,
                                       fitted=entry, fitted_cert=g_cert)
    for k in floor_thresholds(thresholds, targets):  # the automated share of a floored target, as served
        eff = effective_gate(cert_rows, temps, buckets, thresholds[k])
        automated[k], automated_error[k] = eff["coverage"], eff["error"]
        thr = thresholds[k]["confidence"]
        if thr is not None and cs_cert:
            automated_pooled[k] = gate(cs_cert, thr)["coverage"]
        report["targets"][float(k)]["fitted_cert"] = gate(cs_cert, thr)
    for t in PRIMS:
        rs = [r for r in rows if r["prim"] == t]
        if not rs:
            continue
        cert = [r for r in cert_rows if r["prim"] == t]
        # as served: a choice question of a format-2 file gets its option-count bucket temperature first
        report["types"][t] = dict(n=len(rs), n_fit=len(rs) - len(cert), n_cert=len(cert),
                                  accuracy=float(np.mean([ok for _, ok in served(rs, {})])),
                                  T_shipped=shipped_T.get(t), T=temps.get(t), source=source.get(t, "default 1.0"),
                                  ece_shipped=ece(served(cert, shipped_T, shipped_B)),
                                  ece=ece(served(cert, temps, buckets)))
    gates = dict(ece_lofo=lofo(fit_rows, cert_rows, shipped_T, shipped_B, min_per_type, fmt, min_family, reference,
                               min_blend=min_blend),
                 automated_at=automated, automated_error=automated_error, automated_at_pooled=automated_pooled,
                 min_family=min_family)
    report["gates"] = gates
    multi = fit_fallback(sets, temps.get("noul", 1.0), min_per_set) if sets else {}
    report["multi"] = multi
    extra = dict(gates=gates)
    if multi:
        extra["multi"] = {k: v for k, v in multi.items() if "reason" not in v}
    if fmt == 2:
        report["buckets"] = {}
        for name, lo, hi in BUCKETS:
            rs = [r for r in rows if r["prim"] == "choice" and lo <= len(r["p"]) <= hi]
            if rs:
                cert = [r for r in rs if is_cert(r["group"])]
                report["buckets"][name] = dict(n=len(rs), n_cert=len(cert),
                                               T=(buckets.get("choice") or {}).get(name), source=bsource[name],
                                               ece_shipped=ece(served(cert, shipped_T, shipped_B)),
                                               ece=ece(served(cert, temps, buckets)))
        cal = {"prompt": 2, "temperature_per_prim": temps, "temperature_per_bucket": buckets,
               "temperature_source": dict(source, **{f"choice {k}": v for k, v in bsource.items()}),
               "thresholds": thresholds, **extra}
        return cal, report
    return dict(temperature_per_prim=temps, temperature_source=source, thresholds=thresholds, **extra), report


def compared(md, kind, path, quant, orders, off, preset=None, layout="canonical"):
    """The calibration to compare with and keep, as basal-serve would apply it with these option orders
    (server.load_calibration: the model's file for those orders when it ships one; a file fitted on other orders
    without its thresholds) -> (calibration, its file or None)."""
    if off or (md is None and not path):
        return {}, None
    if orders is None:  # a server that does not report its orders: the file as it is
        f = calibration_file(md, kind, path, quant, preset, layout)
        return (json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}), f
    data, f, why = load_calibration(md, kind, path, quant, orders, preset=preset, layout=layout)
    if why:
        print(f"warning: {why}: its thresholds are not compared or kept (as in basal-serve)", file=sys.stderr)
    return data, f if f.exists() else None


def task_rows(units, responses):
    """Labelled task decisions + the served task answers -> (rows {task, group, c: confidence, ok}, failed requests).
    An extract number that does not parse is left out: basal-serve never automates it."""
    rows, failed = [], {}
    for u in units:
        r = responses[u["req"]]
        if isinstance(r, Exception):
            failed[u["req"]] = f"line {u['line']}: {r}"
            continue
        a = r["answers"][u["name"]]
        if u["task"] == "classify":
            conf, ok = a["confidence"], a["label"] == u["gold"]
        elif u["task"] == "verify":
            c = a["claims"][u["part"]]
            conf, ok = c["confidence"], c["verdict"] == u["gold"]
        else:
            f = a["fields"][u["part"]]
            if "error" in f:
                continue
            kind, gold = u["gold"]
            conf, ok = f["confidence"], field_correct(kind, f["value"], gold)
        rows.append(dict(task=u["task"], group=u["group"], c=float(conf), ok=bool(ok)))
    return rows, list(failed.values())


def fit_tasks(rows, targets, shipped=None):
    """Task rows -> ({task: {target: threshold}} for the certified ones, report {task: ...}): the bound of the decision
    types on each task's FIT decisions, with coverage, error and bound on its CERT decisions; `shipped` (the task
    thresholds of the compared file, when the answers were served with its temperatures) is measured on CERT too."""
    thresholds, report = {}, {}
    for t in TASKS:
        rs = [r for r in rows if r["task"] == t]
        if not rs:
            continue
        cf = [(r["c"], r["ok"]) for r in rs if not is_cert(r["group"])]
        cc = [(r["c"], r["ok"]) for r in rs if is_cert(r["group"])]
        e = dict(n=len(rs), n_fit=len(cf), n_cert=len(cc), accuracy=float(np.mean([r["ok"] for r in rs])),
                 method=METHOD, targets={})
        for risk in targets:
            old = ((shipped or {}).get(t) or {}).get(str(float(risk)))
            if old is not None:
                e.setdefault("shipped", {})[str(float(risk))] = dict(confidence=old, cert=gate(cc, old))
            thr = threshold(cf, risk) if cf else None
            if thr is None:
                e["targets"][str(float(risk))] = dict(
                    confidence=None, reason=no_threshold_reason(cf, risk) if cf else "no FIT decisions")
                continue
            g_fit, g_cert = gate(cf, thr), gate(cc, thr)
            e["targets"][str(float(risk))] = dict(
                confidence=thr, fit_coverage=g_fit["coverage"], fit_error=g_fit["error"], fit_upper95=g_fit["upper95"],
                cert_coverage=g_cert["coverage"], cert_error=g_cert["error"], cert_upper95=g_cert["upper95"],
                cert_accepted=g_cert["accepted"])
            thresholds.setdefault(t, {})[str(float(risk))] = thr
        rs_ = sorted(float(r) for r in targets)  # a looser target never keeps a stricter threshold (floor_thresholds)
        for lo, hi in zip(rs_, rs_[1:]):
            a, b = e["targets"][str(lo)], e["targets"][str(hi)]
            if a["confidence"] is not None and (b["confidence"] is None or b["confidence"] > a["confidence"]):
                e["targets"][str(hi)] = dict(a, floored_from=str(lo))
                thresholds.setdefault(t, {})[str(hi)] = a["confidence"]
        report[t] = e
    return thresholds, report


def reference_values(ref):
    """--reference: a built-in name (basal-1.5) or a calibration file with gates.ece_lofo -> ({type: mean}, label)."""
    if ref is None:
        return None, None
    if ref in REFERENCE:
        return REFERENCE[ref], f"{ref} (4.5B): mean leave-one-family-out ECE on its calibration split"
    p = Path(ref)
    if not p.exists():
        raise SystemExit(f"--reference {ref}: not {' or '.join(REFERENCE)} and no such file")
    lf = (json.loads(p.read_text(encoding="utf-8")).get("gates") or {}).get("ece_lofo") or {}
    vals = {t: v["mean"] for t, v in lf.items() if isinstance(v, dict) and v.get("mean") is not None}
    if not vals:
        raise SystemExit(f"--reference {ref}: the file has no gates.ece_lofo means")
    return vals, p.name


# -- output ----------------------------------------------------------------------------------------------------------
def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def num(x, f="{:.3f}"):
    return "-" if x is None else f.format(x)


def table(report):
    lines = ["", f"{'type':<7}{'n':>6}{'FIT':>6}{'CERT':>6}{'accuracy':>10}   {'temperature shipped -> fitted':<32}"
                 f"ECE on CERT shipped -> fitted"]
    for t, r in report["types"].items():
        temp = f"{num(r['T_shipped'])} -> {num(r['T'])}" + (
            "" if r["source"].startswith("fitted") else " (blended)" if r["source"].startswith("blended") else " (kept)")
        lines.append(f"{t:<7}{r['n']:>6}{r['n_fit']:>6}{r['n_cert']:>6}{r['accuracy']:>10.3f}   {temp:<32}"
                     f"{num(r['ece_shipped'])} -> {num(r['ece'])}")
    for name, r in (report.get("buckets") or {}).items():  # calibration file 2: choice per option-count bucket
        lines.append(f"  {name:<9}{r['n']:>6}{'':>6}{r['n_cert']:>6}{'':>10}   "
                     f"{('-> ' + num(r['T'])) if r['T'] is not None else '(type temperature)':<32}"
                     f"{num(r['ece_shipped'])} -> {num(r['ece'])}")
    lines += ["", f"max_error on CERT   {'shipped: threshold':>18}{'coverage':>10}{'error':>8}   "
                  f"{'fitted: threshold':>17}{'coverage':>10}{'error':>8}{'bound':>8}"]
    for risk, r in report["targets"].items():
        s = r["shipped_cert"] or {}
        left = f"{risk:<20g}{num(r['shipped'], '{:.4f}'):>18}{pct(s.get('coverage')):>10}{pct(s.get('error')):>8}   "
        f = r["fitted"]
        if f["confidence"] is None:
            lines.append(left + f"none: {f['reason']}")
        else:
            g = r["fitted_cert"]
            lines.append(left + f"{f['confidence']:>17.4f}{pct(g['coverage']):>10}{pct(g['error']):>8}"
                                f"{pct(g['upper95']):>8}")
    lines.append("(one threshold pooled over the types, informational: basal-serve applies each type's own, as in "
                 "\"automated on CERT\" below)")
    gates = report.get("gates") or {}
    if gates.get("ece_lofo"):
        lines += ["", f"LOFO ECE on CERT   {'families':>9}{'mean':>8}{'floor':>8}   {'max (family)':<26}{'pooled':>8}"
                      f"{'reference':>11}"]
        for t, g in gates["ece_lofo"].items():
            if g.get("mean") is None:
                lines.append(f"{t:<18}{g['n_families']:>9}   none: {g['reason']}")
                continue
            worst = f"{g['max']:.3f} ({str(g['worst_family'])[:16]})"
            mark = "  REGRESSION" if g["regression"] else ""
            lines.append(f"{t:<18}{g['n_families']:>9}{g['mean']:>8.3f}{g['mean_floor']:>8.3f}   {worst:<26}"
                         f"{num(g['pooled']):>8}{num(g['reference']):>11}{mark}")
    if gates.get("automated_at"):
        compared = report.get("compared") or {}
        lines += ["", f"automated on CERT   {'per-type thresholds':>20}{'error':>8}   {'pooled (not applied)':>20}"
                      + ("   compared file, as served" if compared else "")]
        for k, v in gates["automated_at"].items():
            c = compared.get(k)
            lines.append(f"max_error {float(k):<9g} {pct(v):>20}{pct(gates['automated_error'].get(k)):>8}   "
                         f"{pct(gates['automated_at_pooled'].get(k)):>20}"
                         + (f"   {pct(c['coverage'])} at {pct(c['error'])} error" if c else ""))
        for risk, r in report["targets"].items():
            per = r["fitted"].get("per_type") or {}
            lines.append(f"  {risk:g} per type: " + ", ".join(
                f"{t} " + (f"{x['confidence']:.4f} ({pct(x['cert_coverage'])})" if x.get("confidence") is not None
                           else "none (defers)") for t, x in per.items()))
    if report.get("multi"):
        lines += ["", f"multi fallback     {'label set':<18}{'n':>5}{'threshold':>11}   {'IoU default -> fitted':<22}"
                      f"{'empty acc.':<17}false-empty"]
        for sid, m in report["multi"].items():
            head = f"{str(m['fallback'])[:18]:<19}{sid:<18}{m['n']:>5}"
            if "reason" in m:
                lines.append(head + f"   none: {m['reason']}")
                continue
            c = m["cert"]
            thr = "default" if m["fallback_threshold"] is None else f"{m['fallback_threshold']:.4g}"
            lines.append(head + f"{thr:>11}   {num(c['iou_default']) + ' -> ' + num(c['iou']):<22}"
                                f"{pct(c['empty_accuracy_default']) + ' -> ' + pct(c['empty_accuracy']):<17}"
                                f"{pct(c['false_empty_default'])} -> {pct(c['false_empty'])}")
    return "\n".join(lines)


def task_table(report):
    lines = ["", f"{'task':<9}{'n':>6}{'FIT':>6}{'CERT':>6}{'accuracy':>10}   {'max_error':<10}{'threshold':>10}"
                 f"{'automated':>11}{'error':>8}{'bound':>8}   (automated share, error and bound on CERT)"]
    for t, e in report.items():
        head = f"{t:<9}{e['n']:>6}{e['n_fit']:>6}{e['n_cert']:>6}{e['accuracy']:>10.3f}   "
        for i, (risk, x) in enumerate(e["targets"].items()):
            left = head if i == 0 else " " * len(head)
            if x["confidence"] is None:
                lines.append(left + f"{float(risk):<10g}none: {x['reason']}")
            else:
                lines.append(left + f"{float(risk):<10g}{x['confidence']:>10.4f}{pct(x['cert_coverage']):>11}"
                                    f"{pct(x['cert_error']):>8}{pct(x['cert_upper95']):>8}")
        for risk, x in (e.get("shipped") or {}).items():
            g = x["cert"]
            lines.append(f"{'':<9}compared file's threshold {float(risk):g}: {x['confidence']:.4f}, automated "
                         f"{pct(g['coverage'])}, error {pct(g['error'])}, bound {pct(g['upper95'])} on CERT")
    return "\n".join(lines)


def task_warnings(report, dropped=()):
    out = []
    for t, e in report.items():
        if e["n"] < RECOMMENDED:
            out.append(f"{t}: only {e['n']} labelled decisions: the bounds are loose")
        for risk, x in e["targets"].items():
            if x["confidence"] is None:
                out.append(f"max_error {float(risk):g}: {t} has no certified threshold; its answers report "
                           "automated: null at this max_error")
            elif cert_note(t, float(risk), x):
                out.append(cert_note(t, float(risk), x))
    for t in dropped:
        out.append(f"{t}: the shipped task threshold was certified with the shipped temperatures, which this file "
                   f"changes; it is left out (add labelled {t} questions to certify one)")
    return out


def warnings(report, targets):
    out = []
    n = report["n_fit"] + report["n_cert"]
    if n < RECOMMENDED:
        out.append(f"only {n} labelled decisions: the bounds are loose (a few hundred usually certify 5%)")
    if not report["n_fit"] or not report["n_cert"]:
        out.append("one side of the FIT / CERT split is empty: use more groups")
    for t, r in report["types"].items():
        if r["source"].startswith("fitted") and any(math.isclose(r["T"], x) for x in T_GRID):
            out.append(f"{t}: the fitted temperature is at the edge of the search ({r['T']:.2f}): too few errors "
                       "or too few decisions to fit it; label more or raise --min-per-type")
    for risk in targets:
        r = report["targets"][risk]
        for t, x in (r["fitted"].get("per_type") or {}).items():
            if x.get("confidence") is None:
                out.append(f"max_error {risk:g}: {t} has no certified threshold; basal-serve defers every automatic "
                           f"action of act and tool on {t} questions with this max_error")
            elif cert_note(t, risk, x):
                out.append(cert_note(t, risk, x))
    for t, g in ((report.get("gates") or {}).get("ece_lofo") or {}).items():
        if g.get("regression"):
            out.append(f"{t}: the leave-one-family-out ECE {g['mean']:.3f} is above the reference {g['reference']:.3f}")
        elif g.get("mean") is None:
            out.append(f"{t}: no leave-one-family-out ECE, 0 families tested ({g['reason']}); --strict does not check "
                       f"{t}")
    for sid, m in (report.get("multi") or {}).items():
        if "reason" in m:
            out.append(f"multi label set {sid} (fallback {m['fallback']}): no fallback_threshold fitted "
                       f"({m['reason']})")
    return out


def regressions(report):
    """Types whose mean LOFO ECE is above the reference (--strict)."""
    return [t for t, g in ((report.get("gates") or {}).get("ece_lofo") or {}).items() if g.get("regression")]


def summary(lines):
    """End of a run, on stderr: the targets the labels could not certify (sample_size) and the written thresholds that
    CERT does not confirm (cert_notes), also in the report above."""
    if lines:
        print("basal-calibrate: summary\n" + "\n".join(f"  warning: {x}" for x in lines), file=sys.stderr)


def strict_check(report, min_family):
    """--strict: on stderr, every type the family check did not test (no family with min_family CERT decisions of the
    type, or one family: it passes them), then status 1 when a type's mean LOFO ECE is above the reference."""
    for t, g in ((report.get("gates") or {}).get("ece_lofo") or {}).items():
        if g.get("mean") is None:
            why = (f"each needs >= {min_family} CERT decisions of the type" if g["reason"].startswith("no family")
                   else g["reason"])
            print(f"--strict: {t}: 0 families tested ({why}); the family check did not run", file=sys.stderr)
    bad = regressions(report)
    if bad:
        print(f"--strict: leave-one-family-out ECE above the reference for {', '.join(bad)}", file=sys.stderr)
        sys.exit(1)


def engine_name(kind, quant, preset=None):
    """The engine a calibration file is for (its "engine" field): the name of its CALIBRATION.<engine>.json, else
    fast for the basal CUDA engine and the backend's own name (eager, mps, gguf)."""
    return calibration_name(kind, quant, preset) or {"graph": "fast", "exit": "fast"}.get(kind, kind or "unknown")


def refit(a, targets, reference, reference_name):
    """--from-rows: the fit on the probabilities a --save-rows run recorded (no engine). The baseline is --calibration,
    else the file that run compared with."""
    lines = Path(a.input).read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    if head.get("basal_calibrate_rows") != 1:
        raise SystemExit(f"{a.input}: not a file written by basal-calibrate --save-rows")
    rows = [json.loads(x) for x in lines[1:] if x.strip()]
    small = sample_size(rows, [], targets)
    for w in small:
        print(f"warning: {w}", file=sys.stderr)
    base_file = Path(a.calibration) if a.calibration else (Path(head["baseline"]) if head.get("baseline") else None)
    baseline = json.loads(base_file.read_text(encoding="utf-8")) if base_file is not None and base_file.exists() else {}
    fo = calibration_orders(baseline)
    if baseline and fo is not None and head.get("orders") is not None and fo != head["orders"]:  # as basal-serve
        print(f"warning: {base_file.name} was fitted on {fo} option order(s), the rows on {head['orders']}: its "
              "thresholds are not compared or kept (as in basal-serve)", file=sys.stderr)
        baseline = {k: v for k, v in baseline.items() if k not in ("thresholds", "multi", "task_thresholds")}
    cal, report = fit(rows, baseline, targets, a.min_per_type, head["fmt"], a.min_family, reference, a.min_per_set,
                      a.min_blend)
    cal["gates"]["ece_reference"] = reference_name
    cal["engine"] = engine_name(head["kind"], head["quant"], head.get("vllm_preset"))
    cal["fitted_on"] = dict(data=head["data"], decisions=sum(r["prim"] != "multi" for r in rows),
                            requests=head["requests"], fit=report["n_fit"], cert=report["n_cert"],
                            per_type={t: r["n"] for t, r in report["types"].items()},
                            multi=sum(r["prim"] == "multi" for r in rows), date=datetime.date.today().isoformat(),
                            model=head["model"], engine=head["engine"], orders=head["orders"],
                            split="FIT 70% / CERT 30% by group", tool=f"basal-calibrate {__version__} (--from-rows)")
    cal["certified_on"] = "CERT split of fitted_on.data (30% by group)"
    cal["baseline"] = str(base_file) if baseline else "none"
    print(table(report))
    notes = [f"{t}: task thresholds are not refitted from rows (they are certified on the answers as served); the "
             "refitted file leaves them out: fit them with the engine" for t in sorted(baseline.get("task_thresholds") or {})]
    for w in warnings(report, targets) + notes:
        print(f"warning: {w}")
    if a.report_only:
        print("\n--report-only: no file written")
    else:
        Path(a.out).write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out} (calibration file {head['fmt']}, engine {cal['engine']}, refitted from {a.input})")
    summary(small + cert_notes(report, None, targets))
    if a.strict:
        strict_check(report, a.min_family)


# -- command line ----------------------------------------------------------------------------------------------------
def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="labelled requests (JSONL, format above)")
    ap.add_argument("--out", default="CALIBRATION.custom.json", help="calibration file to write")
    ap.add_argument("--url", default=None,
                    help="a running basal-serve started with --no-calibration (instead of the engine in process)")
    ap.add_argument("--calibration", default=None,
                    help="the calibration to compare with and to keep for small types (default: the file basal-serve "
                         "loads for --model and the engine)")
    ap.add_argument("--no-calibration", dest="no_calibration", action="store_true",
                    help="no shipped calibration to compare with (types below --min-per-type keep temperature 1.0)")
    ap.add_argument("--targets", default="0.01,0.05", help="max_error targets (comma-separated)")
    ap.add_argument("--min-per-type", dest="min_per_type", type=int, default=50,
                    help="FIT decisions needed to fit a type's temperature")
    ap.add_argument("--min-blend", dest="min_blend", type=int, default=MIN_BLEND,
                    help="a type (or bucket) with fewer FIT decisions than --min-per-type but at least this many gets "
                         "the fitted temperature blended with the shipped one in proportion n / --min-per-type; 0 "
                         "keeps the shipped one")
    ap.add_argument("--min-family", dest="min_family", type=int, default=30,
                    help="CERT decisions of a type a family needs for its leave-one-family-out ECE")
    ap.add_argument("--min-per-set", dest="min_per_set", type=int, default=30,
                    help="FIT multi questions a label set needs to fit its fallback_threshold")
    ap.add_argument("--reference", nargs="?", const="basal-1.5", default=None,
                    help="LOFO ECE reference: basal-1.5 (the default value of the flag) or a calibration file with "
                         "gates.ece_lofo")
    ap.add_argument("--save-rows", dest="save_rows", default=None,
                    help="also write the uncalibrated probabilities of every decision (JSONL), to refit without a model")
    ap.add_argument("--from-rows", dest="from_rows", action="store_true",
                    help="the input is a file written by --save-rows: refit on it without running any engine")
    ap.add_argument("--report-only", dest="report_only", action="store_true",
                    help="print the report on your labelled requests (accuracy, ECE, certified thresholds, automated "
                         "share) without writing a calibration file")
    ap.add_argument("--strict", action="store_true",
                    help="exit with status 1 when a type's leave-one-family-out ECE is above --reference")
    ap.add_argument("--concurrency", type=int, default=32, help="requests in flight (the engine batches them)")
    add_engine_args(ap.add_argument_group("engine (in process, as basal-serve)"))
    ap.set_defaults(model=None)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        targets = sorted({float(x) for x in a.targets.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--targets {a.targets!r}: give numbers such as 0.01,0.05") from None
    if not targets or not all(0 < r < 1 for r in targets):
        raise SystemExit("--targets must lie between 0 and 1")
    if a.calibration and not Path(a.calibration).exists():
        raise SystemExit(f"--calibration {a.calibration}: no such file")
    if a.strict and a.reference is None:
        raise SystemExit("--strict needs --reference (basal-1.5 or a calibration file with gates.ece_lofo)")
    reference, reference_name = reference_values(a.reference)
    if a.from_rows:
        return refit(a, targets, reference, reference_name)
    try:
        bodies, decisions, task_bodies, units = read(a.input)
    except InputError as e:
        raise SystemExit(str(e)) from None
    print(f"basal-calibrate: {len(decisions)} labelled decisions in {len(bodies)} requests"
          + (f", {len(units)} labelled task decisions in {len(task_bodies)} requests" if units else ""),
          file=sys.stderr)
    small = sample_size(decisions, units, targets)
    for w in small:  # before the readouts: a target these labels cannot certify (feedback from BRAUN SI)
        print(f"warning: {w}", file=sys.stderr)
    if units and a.url:
        raise SystemExit("task questions (classify / verify / extract) need the engine in process: their thresholds "
                         "are certified on the answers as served with the fitted temperatures; give --model and --mode "
                         "instead of --url, or leave the task questions out")
    fitted = {}  # settle(): the decision rows, the file to serve and the report

    def settle(responses):
        """The decision part: the fit on the decision questions, else the compared calibration kept as it is."""
        rows, failed = raw_rows(decisions, responses)
        if failed:
            print(f"warning: {len(failed)} request(s) failed and are left out:\n  " + "\n  ".join(failed[:5]),
                  file=sys.stderr)
        if decisions and not rows:
            raise SystemExit("no decisions to fit")
        if rows:
            cal, report = fit(rows, baseline, targets, a.min_per_type, fmt, a.min_family, reference, a.min_per_set,
                              a.min_blend)
            cal["gates"]["ece_reference"] = reference_name
        else:  # task questions only: the temperatures and thresholds of the compared file stay as they are
            cal, report = {k: v for k, v in baseline.items() if k not in KEPT_OUT}, None
        fitted.update(rows=rows, failed=failed, cal=cal, report=report)
        return cal

    if a.url:
        import httpx
        base = a.url.rstrip("/").removesuffix("/v1/systemone").removesuffix("/v1/basal")
        if not (a.model or a.calibration or a.no_calibration):
            raise SystemExit("with --url, give --model (its shipped calibration is compared with and kept for small "
                             "types), --calibration FILE or --no-calibration")
        try:
            entry = uncalibrated(httpx.get(base + "/v1/models", timeout=30).json()["models"][0], base)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            raise SystemExit(f"--url {a.url}: no basal server answers ({e})") from None
        mode = a.mode or entry.get("mode")
        engine = f"{mode} at {base}"
        orders = entry.get("orders")  # the server's option orders (servers before this field: unknown)
        kind, quant = MODES.get(mode, ("", None, False))[:2]
        preset = entry.get("vllm_preset")  # --mode vllm: the server's preset (servers before 2.2.1: none)
        md = resolve(a.model, a.revision, metadata_only=True) if a.model else None
        # a Hugging Face cache snapshot is recorded as its repo id and commit, as basal-serve names it
        model, revision = model_identity(a.model, md) if md is not None else (entry.get("name"), None)
        fmt = entry.get("prompt_format") or (model_format(md) if md is not None else 1)
        # the layout the server reads prompts in (servers before 2.4.1 and canonical ones do not say)
        a.layout = getattr(a, "layout", None) or entry.get("layout") or (model_layout(md) if md is not None
                                                                          else "canonical")
        baseline, base_file = compared(md, kind, a.calibration, quant, orders, a.no_calibration, preset, a.layout)
        settle(asyncio.run(over_http(base + "/v1/systemone", bodies, a.concurrency)) if bodies else [])
        task_responses = []
    else:
        a.model = a.model or add_engine_args(argparse.ArgumentParser()).get_default("model")  # basal-serve's
        refuse_lean_flex(a.lean, a.tree)  # before default_mode, which resolves the model on CUDA GPUs without FA3
        mode = a.mode or default_mode(a.model, a.revision)
        srv = Server(argparse.Namespace(**{**vars(a), "mode": mode, "name": None, "calibration": None,
                                           "no_calibration": True, "log_decisions": None}))
        kind, quant, fmt, orders = srv.kind, a.quant or MODES[mode][1], srv.fmt, srv.orders
        preset = getattr(srv.backend, "preset", None)
        a.layout = srv.layout  # --layout, else the model's basal.json "layout"
        baseline, base_file = compared(srv.md, srv.kind, a.calibration, quant, orders, a.no_calibration, preset,
                                       a.layout)
        model, revision = model_identity(a.model, srv.md)
        engine = mode + (f" --quant {a.quant}" if a.quant else "") + (f" --gguf {Path(a.gguf).name}" if a.gguf else "") \
            + (f" --vllm-preset {preset}" if preset not in (None, "default") else "")
        _, task_responses = asyncio.run(two_passes(srv, bodies, task_bodies, a.concurrency, settle))
    rows, failed, cal, report = fitted["rows"], fitted["failed"], fitted["cal"], fitted["report"]
    trows, tfailed = task_rows(units, task_responses) if units else ([], [])
    if tfailed:
        print(f"warning: {len(tfailed)} task request(s) failed and are left out:\n  " + "\n  ".join(tfailed[:5]),
              file=sys.stderr)
    if units and not trows:
        raise SystemExit("no task decisions to fit")
    shipped_tasks = baseline.get("task_thresholds") or {}
    # the compared file's task thresholds are measured on your data when the answers were served with its temperatures
    task_thr, task_report = fit_tasks(trows, targets, None if rows else shipped_tasks) if trows else ({}, {})
    # a shipped task threshold holds for the shipped temperatures: kept only when this file keeps them too
    dropped = sorted(t for t in shipped_tasks if t not in task_report) if rows else []
    kept = {} if rows else {t: v for t, v in shipped_tasks.items() if t not in task_report}
    if kept or task_thr:
        cal["task_thresholds"] = {**kept, **task_thr}
    if task_report:
        cal["task_gates"] = task_report
    if a.save_rows and rows:
        head = dict(basal_calibrate_rows=1, fmt=fmt, kind=kind, quant=quant, vllm_preset=preset, model=str(model),
                    engine=engine, data=Path(a.input).name, requests=len(bodies) - len(failed),
                    baseline=str(base_file) if baseline else None, orders=orders)
        with open(a.save_rows, "w", encoding="utf-8") as f:
            for x in [head] + rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    cal["engine"] = engine_name(kind, quant, preset)
    cal["fitted_on"] = dict(data=Path(a.input).name, decisions=sum(r["prim"] != "multi" for r in rows),
                            requests=len(bodies) - len(failed),
                            fit=report["n_fit"] if report else 0, cert=report["n_cert"] if report else 0,
                            per_type={t: r["n"] for t, r in report["types"].items()} if report else {},
                            multi=sum(r["prim"] == "multi" for r in rows),
                            date=datetime.date.today().isoformat(), model=model, engine=engine,
                            split="FIT 70% / CERT 30% by group", tool=f"basal-calibrate {__version__}")
    if revision:
        cal["fitted_on"]["revision"] = revision
    if task_report:
        cal["fitted_on"]["tasks"] = {t: e["n"] for t, e in task_report.items()}
        cal["fitted_on"]["task_requests"] = len(task_bodies) - len(tfailed)
    if orders is not None:  # basal-serve applies the file only to these orders (or loads CALIBRATION.o<N>.json)
        cal["fitted_on"]["orders"] = orders
    if getattr(a, "layout", None) not in (None, "canonical"):  # basal-serve applies it only in this layout (2.4.1)
        cal["fitted_on"]["layout"] = a.layout
    cal["certified_on"] = "CERT split of fitted_on.data (30% by group)"
    cal["baseline"] = (str(a.calibration or f"{a.model}/{base_file.name}") if baseline else "none") + \
        (f" ({baseline['source_tag']})" if baseline.get("source_tag") else "")
    if not rows:
        cal["decision_calibration"] = (f"kept from {cal['baseline']} (no decision questions in the input)" if baseline
                                       else "none (no decision questions in the input, no calibration to keep)")
    if report:
        print(table(report))
    else:
        print(f"\ndecision questions: none; temperatures and thresholds kept from {cal['baseline']}")
    if task_report:
        print(task_table(task_report))
    for w in (warnings(report, targets) if report else []) + task_warnings(task_report, dropped):
        print(f"warning: {w}")
    if a.report_only:
        print("\n--report-only: no file written")
    else:
        Path(a.out).write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out} (calibration file {fmt}, engine {cal['engine']})\n"
              f"  basal-serve --model {a.model or '<model>'} --mode {mode} --calibration {a.out}")
    summary(small + cert_notes(report, task_report, targets))
    if a.strict and report:
        strict_check(report, a.min_family)

if __name__ == "__main__":
    main()
