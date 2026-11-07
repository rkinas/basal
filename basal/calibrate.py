"""Fit the calibration on your own labelled requests: one temperature per question type (for prompt-format-2 models
also one per option-count bucket of choice questions), the certified confidence thresholds of act's max_error (1%
and 5% by default), the calibration gates, from labelled multi questions the fallback_threshold of each label set
and, with --conformal, the answer-set thresholds of each coverage level, written as a calibration file for basal-serve.

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
          (0 = first level; an integer is always the index), key or text; act: an outcome, as for choice (yes / no for true / false outcomes);
          multi: the list of labels that apply (keys or descriptions; [] = none)
  group   optional, on the line or on a question: decisions of one group (twins, duplicates, one customer) stay on
          the same side of the FIT / CERT split; default: one group per line (its questions share the state). When the
          errors of a group's decisions go together (claims or fields of one document), the bound counts them as
          fewer independent decisions (design effect 1 + (m - 1) ICC)
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
  * for each target error r, the threshold comes from a fixed-sequence test on FIT: the confidence levels (equal
    confidences accepted together) are tested from the most confident down, each with the one-sided 95%
    Clopper-Pearson upper bound of its accepted decisions <= r, and the scan stops at the first level that fails; the
    threshold is the last level that passed (none when the first fails). The first level tested is set by the counts
    alone: the first that accepts enough decisions to meet the bound with one error (93 at 5%, 473 at 1%) when the
    type has at least twice that many FIT decisions, else without an error (59 / 299; at least 50). CERT reports its coverage, error and bound, and a threshold CERT refutes (its
    one-sided 95% lower bound of the error above r) is withdrawn. The threshold is fitted per type (thresholds[r].per_type[type], on the type's own decisions), and
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
  * --conformal 0.8,0.9,0.95 (off by default): split-conformal answer sets per type and coverage level L. The score
    of a CERT decision is 1 - p(gold) under the fitted temperatures (CERT took no part in fitting them); p_min is the
    k-th largest p(gold) of the type's CERT decisions, k = ceil((n + 1) L), and the set of an answer is every option
    with p >= p_min (the top one always). The set then holds the correct answer with probability >= L, on average
    over requests like the labelled ones. With n < L / (1 - L) CERT decisions of a type there is no p_min and the set
    is every option. The FIT decisions report the coverage and mean size of the sets. basal-serve returns the set for
    a choice / noul / score question with "coverage": L;
  * tasks: per task (classify, verify, extract; every claim and field is one decision) the certified threshold of each
    target error, by the same bound on FIT, with its coverage, error and bound on CERT ("task_thresholds" and
    "task_gates"). The task thresholds a shipped file holds are kept only when the temperatures are (no decision
    questions in the input); otherwise a task without labelled questions has none and its answers report
    automated: null.
  * --keep-shipped noul (a list of choice / noul / score; an act question counts as its base type): those types keep the compared
    file's temperature (and for choice its option-count bucket temperatures) together with its per-type threshold of
    every target, since a shipped threshold is certified only with the shipped temperature. For a type the labels do
    not certify at a max_error the compared file did (the warning names it), this keeps automating it as the shipped
    file did instead of deferring it. The report shows how the kept thresholds do on this file's FIT and CERT split
    ("kept": "shipped" in the file, fitted_on.kept_shipped lists the types and kept_shipped_from names the file).
    A kept threshold your FIT and CERT decisions refute is withdrawn; a pooled-only threshold (a file without per-type
    ones) and a file of another engine are refused; the other types are fitted as usual;
  * fitted_on names the model (and its Hugging Face revision, and model_files: the identity of the snapshot's files
    but its card; model_content: the identity of the folder's files by their content, which matches a local folder):
    basal-serve keeps a --calibration file fitted on another model, or on a revision with other model files, for its
    temperatures only (its thresholds are not applied, the answers are marked unvalidated) with a warning, and refuses
    it with --strict-calibration. A revision that changes only the card keeps the file.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from . import tasks as task_types
from .cli import add_api_key, headers, positive, refused
from .engine import resolve
from .jsonl import read_lines
from .prompt import max_options, model_format, model_layout
from .server import (MODES, QUESTION_TYPES, UNVALIDATED_OUT, Server, _text, add_engine_args, answer_set,
                     calibration_file, calibration_name, certified, check_calibration, default_mode, identity_mismatch,
                     load_calibration, model_content_id, model_files_id, model_identity, refuse_lean_flex, temper, temperature, to_items,
                     validate_baseline)
from .sets import label_set_id
from .sets import select as select_set
from .textio import read_text

PRIMS = ("choice", "noul", "score")
BUCKETS = (("2-10", 2, 10), ("11-26", 11, 26), ("27-255", 27, 255))  # option-count buckets of calibration file 2
ALPHA = 0.05        # one-sided 95% bounds
MIN_ACCEPTED = 50   # a threshold must accept at least this many FIT decisions
MIN_BLEND = 10      # from this many FIT decisions (below --min-per-type) a temperature is blended, not kept
RECOMMENDED = 200   # fewer labelled decisions: loose bounds
T_GRID = (0.25, 8.0)  # temperatures searched (as for the shipped files)
METHOD = "bound-controlled (one-sided 95% Clopper-Pearson on FIT)"
CONFORMAL_METHOD = "split conformal on CERT (score 1 - p(gold), p_min = k-th largest p(gold), k = ceil((n + 1) L))"
POOLED_NOTE = ("informational: \"confidence\" here is one threshold pooled over all types; basal-serve applies only "
               "per_type[type].confidence and defers a type without one")
YES, NO = {"yes", "true", "tak"}, {"no", "false", "nie"}
TASKS = ("classify", "verify", "extract")  # the tasks with an `automated` flag (compress has none)
VERDICTS = ("supported", "contradicted", "not_stated")
KEPT_SOURCE = "shipped (--keep-shipped, with its thresholds)"
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
    if isinstance(g, int) and not isinstance(g, bool) and item["type"] == "score":  # an integer is the level index
        if not 0 <= g < len(keys):
            raise ValueError(f"gold level {g} is out of range 0..{len(keys) - 1}")
        if str(g) in keys and keys.index(str(g)) != g:
            raise ValueError(f"gold {g} is ambiguous: level index {g} is key {keys[g]!r}, while key {str(g)!r} is "
                             f"level {keys.index(str(g))}; give the key as a string")
        return g
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


def _at(obj, keys):
    """(found, value) at a field of a nested gold object: its property names from the root (tasks.field_keys)."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return False, None
        obj = obj[k]
    return True, obj


def field_gold(path, kind, spec, v, lang=None):
    """The gold of an extract field -> the value basal-serve would return for it (None = not stated); a number given as
    text is read as basal-serve reads it in the question's language."""
    if v is None:
        return None
    if kind == "enum":
        vals = [c for c, _ in task_types._consts(spec)]
        hit = [c for c in vals if type(c) is type(v) and c == v] or [c for c in vals if str(c) == str(v)]
        if not hit:
            raise ValueError(f"field {path!r}: gold {v!r} is not one of its values {vals[:10]}")
        if len(hit) > 1:
            raise ValueError(f"field {path!r}: gold {v!r} matches several values {hit[:10]}; give it with its JSON type")
        return hit[0]
    if kind == "boolean":
        s = str(v).strip().lower()
        if s in YES or s in NO:
            return s in YES
        raise ValueError(f"field {path!r}: gold {v!r} is not true / false")
    if kind in ("number", "integer"):
        x = v if isinstance(v, (int, float)) and not isinstance(v, bool) else task_types.parse_number(str(v), lang=lang)
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
        if isinstance(value, int) and isinstance(gold, int):  # exact, also beyond a float's 53 bits
            return value == gold
        try:
            return abs(float(value) - float(gold)) <= 1e-9 * max(1.0, abs(float(gold)))
        except OverflowError:  # an integer beyond a float's range against a fraction
            return False
    if kind == "string":
        norm = lambda s: " ".join(str(s).split()).casefold()
        return norm(value) == norm(gold)
    return type(value) is type(gold) and value == gold  # enum 1 is not true, nor "1"


def task_units(name, q, gold, state, lang=None):
    """A labelled task question -> its labelled decisions [(part, gold)]: classify one (part None), verify one per
    labelled claim, extract one per labelled field. The question must expand as basal-serve expands it."""
    t = q["type"]
    if t == "compress":
        raise ValueError("compress has no automated flag, so there is nothing to certify: leave it out")
    _, plans = task_types.expand({"state": state, "questions": {name: q}, **({"lang": lang} if lang else {})})
    lang = plans[name].get("lang")  # the errors basal-serve would give, and the language it reads numbers in
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
    keys = task_types.field_keys(q.get("schema") or {})
    for path, kind, spec in task_types.leaves(q.get("schema") or {}):
        found, v = _at(gold, keys[path])
        if found and kind != "enum_list":  # a label set has no certified threshold
            out.append((path, (kind, field_gold(path, kind, spec, v, lang))))
    if not out:
        raise ValueError("gold labels no field (enum, boolean, string, number or integer) of the schema")
    return out


# a labelled question is asked whatever the answers of the others: its gold is its answer (and a condition can name a
# question of the other request, decisions or tasks)
CONDITIONAL = ("ask_if", "depends_on")


def group_of(q, rec, n):
    """The split group of a labelled question: its own "group", else the line's "group", else the line's "id", else the
    line number; null counts as not given (not one group "None" for every such line)."""
    return str(next((x for x in (q.get("group"), rec.get("group"), rec.get("id")) if x is not None), n))


def load(path):
    """Labelled JSONL -> (requests to send, decisions): the decision questions only (read() also returns the task
    questions). Raises InputError listing every problem with its line."""
    bodies, decisions, _, _ = read(path)
    return bodies, decisions


def read(path):
    """Labelled JSONL -> (decision requests, decisions, task requests, labelled task decisions). Raises InputError
    listing every problem with its line."""
    errors, bodies, decisions, task_bodies, units = [], [], [], [], []
    for n, line in enumerate(read_lines(path), 1):
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
                clean = {k: v for k, v in q.items() if k not in ("gold", "group", "family") + CONDITIONAL}
                try:
                    labelled = task_units(name, clean, q["gold"], rec["state"], rec.get("lang"))
                except (ValueError, KeyError, TypeError) as e:
                    errors.append(f"{at}: {str(e).removeprefix(f'question {name!r}: ')}")
                    continue
                task_body["questions"][name] = clean
                group = group_of(q, rec, n)
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
                     if k not in ("gold", "group", "family", "evidence", "fallback_threshold") + CONDITIONAL}
            try:
                item = to_items(rec["state"], {name: clean}, max_options(2))[0]  # the model's limit: when served
                gold = gold_set(q, item) if t == "multi" else gold_index(q, item)
            except (ValueError, KeyError, TypeError) as e:
                errors.append(f"{at}: {str(e).removeprefix(f'question {name!r}: ')}")
                continue
            body["questions"][name] = clean
            d = dict(line=n, req=len(bodies), name=name, prim=item.get("base", t), gold=gold,
                     keys=[str(k) for k in item["keys"]], group=group_of(q, rec, n),
                     family=str(next((x for x in (q.get("family"), rec.get("family")) if x is not None), name)))
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


async def over_http(endpoint, bodies, concurrency, api_key=None):
    import httpx
    async with httpx.AsyncClient(timeout=600, headers=headers(api_key)) as c:
        async def decide(b):
            r = await c.post(endpoint, json=b)
            if r.status_code != 200:
                raise RuntimeError(r.text)
            return r.json()
        return await collect(decide, bodies, concurrency)


def uncalibrated(entry, base):
    """The GET /v1/models entry of a running server -> the entry; refuses a server that applies temperatures."""
    temps = entry.get("temperature_per_prim")
    buckets = entry.get("temperature_per_bucket") or {}  # {type: {bucket: T}}, a server with a file-2 calibration
    if buckets:
        temps = dict(temps or {})
        for prim, per_bucket in buckets.items():
            for bucket, t in per_bucket.items():
                temps[f"{prim}/{bucket}"] = t
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
        a = (r.get("answers") or {}).get(d["name"])
        if not isinstance(a, dict) or not isinstance(a.get("probabilities"), dict):
            failed[(d["req"], d["name"])] = f"line {d['line']}, question {d['name']!r}: no answer"
            continue
        p = a["probabilities"]
        rows.append(dict(d, p=[float(p[k]) if isinstance(p.get(k), (int, float)) else math.nan for k in d["keys"]]))
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


def buckets_cover(buckets, prim="choice", lo=2, hi=255):
    """Whether the bucket temperatures of `prim` cover every option count lo..hi, so that basal-serve never applies its
    type temperature (server.temperature: the bucket first)."""
    seen = set()
    for b in (buckets or {}).get(prim) or {}:
        a, z = (int(x) for x in b.split("-"))
        seen.update(range(a, z + 1))
    return seen >= set(range(lo, hi + 1))


class Tempered:
    """The probabilities of rows as basal-serve returns them with one set of temperatures (same float32 maths, one row
    at a time): temps per type, buckets {type: {"lo-hi": T}} (calibration file 2) before them. Each row is tempered
    once: fit() asks for the same rows at every target, per type and in every report."""

    def __init__(self, temps, buckets=None):
        self.cal, self.memo = dict(temperature_per_prim=temps or {}, temperature_per_bucket=buckets or {}), {}

    def p(self, rows):
        out = []
        for r in rows:
            hit = self.memo.get(id(r))
            if hit is None:  # the row is kept with its result, so its id is not reused while the cache lives
                hit = self.memo[id(r)] = (r, temper(torch.tensor(r["p"]),
                                                    temperature(self.cal, r["prim"], len(r["p"]))).tolist())
            out.append(hit[1])
        return out

    def cs(self, rows):
        """(confidence, correct) of every row as served."""
        return [(float(max(p)), int(np.argmax(p)) == r["gold"]) for r, p in zip(rows, self.p(rows))]


def served_p(rows, temps, buckets=None):
    """The probabilities of every row as basal-serve returns them with these temperatures (Tempered)."""
    return Tempered(temps, buckets).p(rows)


def served(rows, temps, buckets=None):
    """(confidence, correct) of every row as basal-serve returns it with these temperatures."""
    return Tempered(temps, buckets).cs(rows)


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


_LF = np.zeros(1)  # _LF[i] = log(i!), grown on demand


def log_factorials(n):
    """log(i!) for i = 0..n at least: one table grown by doubling (a slice of it costs nothing, where a list turned
    into an array on every bound cost O(n) and made a threshold scan over n decisions O(n^2))."""
    global _LF
    if len(_LF) <= n:
        _LF = np.array([math.lgamma(i + 1) for i in range(max(n + 1, 2 * len(_LF)))])
    return _LF


def cp_upper(e, n, alpha=ALPHA):
    """One-sided Clopper-Pearson upper bound of a binomial error rate (bisection on the binomial CDF, in log space)."""
    if n == 0:
        return 1.0
    lf = log_factorials(n)
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


def icc(cs, groups):
    """The intraclass correlation of the errors of decisions (confidence, correct) within their groups (one-way ANOVA
    estimator, clipped to [0, 1]): 0 when no group holds two decisions or there is no error to share. When one claim
    or field of a document is wrong the others are likelier wrong too; the bound then counts fewer independent
    decisions (design_effect)."""
    by = defaultdict(list)
    for (_, ok), g in zip(cs, groups):
        by[g].append(0.0 if ok else 1.0)
    N, k = len(cs), len(by)
    if N == k or k < 2:
        return 0.0
    y = sum(sum(v) for v in by.values()) / N
    ssb = sum(len(v) * (sum(v) / len(v) - y) ** 2 for v in by.values())
    ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in by.values())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(len(v) ** 2 for v in by.values()) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return min(1.0, max(0.0, (msb - msw) / den)) if den > 0 else 0.0


def design_effect(cs, groups):
    """(rho, deff) of decisions in groups: rho = icc, deff = 1 + (m - 1) rho with m = the sum of the squared group
    sizes over the decisions (Kish; the mean group size a decision is in). Every accepted set is discounted by the
    deff of all the decisions: as if a group's decisions were always accepted together, which overstates it for the
    most confident levels (where a group has few accepted decisions) and keeps the bound on the safe side when rho is
    underestimated (rare errors). One decision per group, or rho 0: deff 1."""
    rho = icc(cs, groups)
    return rho, design_effect_of(groups, rho)


def design_effect_of(groups, rho):
    """deff of decisions in these groups at a given rho (CERT's group sizes with FIT's rho)."""
    if not rho or not groups:
        return 1.0
    size = defaultdict(int)
    for g in groups:
        size[g] += 1
    return 1 + (sum(v * v for v in size.values()) / len(groups) - 1) * rho


def effective(n, e, deff=1.0):
    """n decisions with e errors -> (n, e) as independent decisions: both divided by deff, n rounded down, e up."""
    if deff <= 1:
        return n, e
    return math.floor(n / deff + 1e-9), math.ceil(e / deff - 1e-9)


def within(e, n, risk, alpha=ALPHA):
    """Whether the one-sided Clopper-Pearson upper bound of e errors in n is at most risk: P(X <= e) <= alpha for
    X ~ Binomial(n, risk) (the bound is the p where that probability is alpha, and it falls as p grows), one binomial
    tail instead of a bisection."""
    if n == 0 or e / n > risk:
        return False
    lf = log_factorials(n)
    i = np.arange(e + 1)
    t = lf[n] - lf[i] - lf[n - i] + i * math.log(risk) + (n - i) * math.log1p(-risk)
    m = t.max()
    return math.exp(m) * float(np.exp(t - m).sum()) <= alpha


def refutes(e, n, risk, alpha=ALPHA):
    """Whether e errors among n accepted CERT decisions refute a threshold certified at `risk`: P(X >= e) <= alpha for
    X ~ Binomial(n, risk), i.e. the one-sided 95% lower bound of their error is above risk. CERT took no part in the
    fit, so a threshold whose error is at most risk is refuted at most alpha of the time; too few CERT decisions to
    confirm it (cert_note) is not a refutation."""
    if n == 0 or e <= risk * n:
        return False
    lf = log_factorials(n)
    i = np.arange(e)
    t = lf[n] - lf[i] - lf[n - i] + i * math.log(risk) + (n - i) * math.log1p(-risk)
    m = t.max()
    return 1.0 - math.exp(m) * float(np.exp(t - m).sum()) <= alpha


def withdrawn(entry, risk, deff=1.0):
    """A per-type or task entry (certified on FIT, measured on CERT) -> the entry, or with its confidence withdrawn
    (null: basal-serve defers) when CERT refutes it (its decisions counted as the independent ones they are worth, the
    errors rounded down)."""
    n = entry.get("cert_accepted") or 0
    e = round((entry.get("cert_error") or 0.0) * n)
    n_i, e_i = (n, e) if deff <= 1 else (math.ceil(n / deff - 1e-9), math.floor(e / deff + 1e-9))
    if entry.get("confidence") is None or not refutes(e_i, n_i, risk):
        return entry
    return dict(entry, confidence=None, refuted=entry["confidence"],
                reason=f"certified on FIT at {entry['confidence']:.4f}, but CERT refutes it: {e} error"
                       f"{'' if e == 1 else 's'} in {n} accepted, a one-sided 95% lower bound above {risk:g}")


def start_level(lv, risk, alpha=ALPHA, min_n=MIN_ACCEPTED):
    """The first level the fixed sequence tests, of levels lv (levels()): the most confident one that accepts enough
    decisions to meet the bound with one error (93 at 5%, 473 at 1%) when there are at least twice that many decisions,
    else enough to meet it without an error (59 / 299; fewer cannot meet it at all). One error allowed at the start
    keeps a single error among the very top decisions from ending the test, where the data can afford it. The start
    depends on the counts only, never on which decisions are correct -> its index, None when no level accepts that
    many."""
    need = accepted_needed(risk, 1, alpha, min_n)
    if not lv or lv[-1][1] < 2 * need:
        need = accepted_needed(risk, 0, alpha, min_n)
    return next((k for k, (_, n, _) in enumerate(lv) if n >= need), None)


def threshold(cs, risk, alpha=ALPHA, min_n=MIN_ACCEPTED, deff=1.0):
    """The certified threshold of a fixed-sequence test: the levels are tested from the most confident down, starting
    at start_level (set by the counts alone, before any outcome is read), each at the one-sided bound <= risk, and the
    scan stops at the first level that fails; the threshold is the last level that passed, None when the first one
    fails. Testing in a fixed order and stopping at the first failure keeps the chance of certifying a threshold whose
    accepted decisions are above `risk` at <= alpha; scanning on past failures until one level passes does not. A
    level that adds no error to a passing one passes too (more decisions, the same errors: a lower bound), so the
    bound is computed only where errors are added. deff (design_effect, > 1 when errors go together within groups)
    counts every level's decisions and errors as fewer independent ones."""
    lv = [(c, *effective(n, e, deff)) for c, n, e in levels(cs)]
    k = start_level(lv, risk, alpha, min_n)
    thr, passed = None, None  # passed: the errors of the last level that passed
    for c, n, e in (lv[k:] if k is not None else ()):
        if e != passed and not within(e, n, risk, alpha):
            break
        thr, passed = c, e
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
            falls = ("its answers report automated: null at this max_error" if name in TASKS else
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
    threshold is certified on FIT (a --keep-shipped one on the compared file's data) and stays; None when CERT confirms
    it."""
    if x.get("confidence") is None:
        return None
    where = "certified on FIT"
    if x.get("kept"):  # --keep-shipped: certified where the compared file was fitted, measured here
        name, where = f"{name} (kept shipped threshold)", "certified on the compared file's data"
    n = x.get("cert_accepted") or 0
    e = round((x.get("cert_error") or 0.0) * n)
    errs, need = f"{e} error{'' if e == 1 else 's'}", accepted_needed(risk)
    if n == 0:
        return (f"{name} at max_error {risk:g}: {where}; CERT accepts no decision at this threshold, so it "
                f"cannot confirm it (0 errors would need >= {need} accepted CERT decisions)")
    if e / n > risk:
        return (f"{name} at max_error {risk:g}: {where}, but CERT has {n} accepted, {errs} "
                f"({pct(e / n)}), above the target: the threshold does not hold on CERT; label more data before "
                "relying on it")
    if x.get("cert_upper95") is not None and x["cert_upper95"] > risk:
        return (f"{name} at max_error {risk:g}: {where}; CERT has {n} accepted, {errs}, upper bound "
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


def no_threshold_reason(cs, risk, alpha=ALPHA, min_n=MIN_ACCEPTED, deff=1.0):
    need = accepted_needed(risk, 0, alpha, min_n)  # accepted decisions without an error
    worth = "" if deff <= 1 else f" (worth {deff:.2f} times fewer independent ones: errors go together within groups)"
    if effective(len(cs), 0, deff)[0] < need:
        return (f"a {risk:g} bound needs at least {need} accepted FIT decisions without an error; FIT has {len(cs)}"
                + worth)
    lv = [(c, *effective(n, e, deff)) for c, n, e in levels(cs)]
    c, n, e = lv[start_level(lv, risk, alpha, min_n)]
    what = f"the top {n} FIT decisions" if deff <= 1 else f"FIT decisions worth {n} independent ones"
    return (f"the first level tested (confidence {c:.4f}, {what}, {e} error{'' if e == 1 else 's'}) does not meet the "
            f"bound (upper bound {cp_upper(e, n, alpha):.3f}), and the fixed sequence stops there")


def gate(cs, thr, deff=1.0):
    """Coverage, error and upper bound of the decisions a threshold accepts (the bound on as many independent ones as
    they are worth: effective)."""
    acc = [ok for c, ok in cs if thr is not None and c >= thr]
    e = len(acc) - sum(acc)
    n_i, e_i = effective(len(acc), e, deff)
    return dict(coverage=len(acc) / len(cs) if cs else None, accepted=len(acc), errors=e,
                error=e / len(acc) if acc else None, upper95=cp_upper(e_i, n_i) if acc else None)


def certify(cf, cc, risk, deff=1.0, deff_c=1.0, veto=True, method=False):
    """The one entry builder of a certified threshold (pooled, per type, per task): the fixed-sequence threshold of
    the FIT decisions cf at target `risk`, with its FIT and CERT (cc) coverage, error and bound; withdrawn when CERT
    refutes it (`veto`; the informational pooled entry is never withdrawn); deff / deff_c: the design effects of FIT
    and CERT -> (entry, CERT gate at the threshold)."""
    head = dict(method=METHOD) if method else {}
    thr = threshold(cf, risk, deff=deff) if cf else None
    g_cert = gate(cc, thr, deff_c)
    if thr is None:
        return dict(confidence=None, **head, reason=no_threshold_reason(cf, risk, deff=deff) if cf else
                    "no FIT decisions"), g_cert
    g_fit = gate(cf, thr, deff)
    entry = dict(confidence=thr, **head, fit_coverage=g_fit["coverage"], fit_error=g_fit["error"],
                 fit_upper95=g_fit["upper95"], cert_coverage=g_cert["coverage"], cert_error=g_cert["error"],
                 cert_upper95=g_cert["upper95"], cert_accepted=g_cert["accepted"])
    return (withdrawn(entry, risk, deff_c) if veto else entry), g_cert


def floor(entries):
    """The one floor of certified entries {target: entry} (in place): a confidence certified at a target error is
    certified at every looser target too (its bound is below both), so a looser target never keeps a stricter
    threshold than a tighter one: the fixed sequence stops at its first failing level, and at a loose target it starts
    on a small top level and can stop on a few errors below it (5% at 0.9999 against 1% at 0.9587). The looser entry
    becomes the tighter one, marked "floored_from" -> the targets whose entries changed."""
    changed, ks = set(), sorted(entries, key=float)
    for lo, hi in zip(ks, ks[1:]):
        a, b = entries[lo], entries[hi]
        if a.get("confidence") is not None and (b.get("confidence") is None or b["confidence"] > a["confidence"]):
            entries[hi] = dict(a, floored_from=lo)
            changed.add(hi)
    return changed


def blend(T_fit, T_old, n, min_per_type):
    """A temperature fitted on n < min_per_type decisions, blended in log space with the one it replaces in proportion
    n / min_per_type (so the full fit takes over continuously at min_per_type) -> (temperature, weight of the fit)."""
    w = n / min_per_type
    return float(math.exp(w * math.log(T_fit) + (1 - w) * math.log(T_old))), w


def fit_temps(fit_rows, shipped_T, shipped_B, min_per_type, fmt, min_blend=MIN_BLEND, keep=()):
    """Temperatures per type (and per choice bucket for format 2) fitted on FIT rows where there are enough decisions,
    blended with the shipped ones from min_blend decisions (0: never), else the shipped ones; the types in `keep`
    (--keep-shipped) take the shipped ones as they are -> (temps, source, buckets, bucket source)."""
    temps, source = {}, {}
    min_per_type = max(1, min_per_type)  # 0 would fit a type that has no decision at all (max() of nothing)
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
    for t in keep:  # with its thresholds: a shipped threshold is certified with the shipped temperature only
        temps[t], source[t] = shipped_T[t], KEPT_SOURCE
        if fmt == 2 and t == "choice":
            buckets.pop("choice", None)
            if shipped_B.get("choice"):
                buckets["choice"] = dict(shipped_B["choice"])
            for name, _, _ in BUCKETS:
                bsource[name] = KEPT_SOURCE if name in (shipped_B.get("choice") or {}) else "type temperature (kept)"
    return temps, source, buckets, bsource


def lofo(fit_rows, cert_rows, shipped_T, shipped_B, min_per_type, fmt, min_family=30, reference=None, draws=50, seed=0,
         min_blend=MIN_BLEND, keep=()):
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
                                         min_per_type, fmt, min_blend, keep)
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
        if len(fit_p) < max(1, min_per_set):
            out[sid] = dict(e, fallback_threshold=None, reason=f"{len(fit_p)} FIT questions < {max(1, min_per_set)}")
            continue
        best = max(FALLBACK_GRID, key=lambda t: set_metrics(fit_p, t)["iou"])
        fitted, default = set_metrics(cert_p, best), set_metrics(cert_p, None)
        out[sid] = dict(e, fallback_threshold=best, method="mean IoU on FIT over a log grid (null: the default rule)",
                        cert={**fitted, **{f"{k}_default": v for k, v in default.items()}})
    return out


def per_type_thresholds(fit_rows, cert_rows, temps, buckets, risk, tempered=None, baseline=None, keep=()):
    """The certified threshold of every type on its own decisions, while one threshold pooled over the types moves
    with the temperatures of the others. A temperature keeps the order of a type's yes/no confidences (the top
    probability is monotone in T for two options), not of choice and score ones: with three or more options it can
    swap two decisions (logits [0, -3, -3] and [0, -2, -20]: 0.909 > 0.881 at T = 1, 0.403 < 0.512 at T = 10), and
    format-2 choice buckets have temperatures of their own. The levels are therefore ordered as served, with the
    temperatures fitted on the same FIT decisions. Decisions that share a group are counted as fewer independent ones
    when their errors correlate (icc on FIT). `tempered`: a Tempered of temps / buckets to reuse. A type in `keep`
    takes the threshold `baseline` certified for it (with the shipped temperature `temps` holds for it), measured on
    FIT and CERT."""
    tempered = tempered or Tempered(temps, buckets)
    out = {}
    for t in PRIMS:
        rf = [r for r in fit_rows if r["prim"] == t]
        if not rf and t not in keep:
            continue
        rc = [r for r in cert_rows if r["prim"] == t]
        cf, cc = tempered.cs(rf), tempered.cs(rc)
        rho, deff = design_effect(cf, [r["group"] for r in rf])  # FIT's rho, each side's own group sizes
        deff_c = design_effect_of([r["group"] for r in rc], rho)
        if t in keep:  # FIT and CERT together test it (neither certified it), with their own design effect
            _, deff_all = design_effect(cf + cc, [r["group"] for r in rf + rc])
            out[t] = kept_entry(cf, cc, shipped_threshold(baseline, risk, t), risk, deff, deff_c, deff_all)
        else:
            out[t], _ = certify(cf, cc, risk, deff, deff_c)
        if rho:
            out[t].update(icc=rho, design_effect=deff)
    return out


def kept_entry(cf, cc, thr, risk, deff=1.0, deff_c=1.0, deff_all=1.0):
    """--keep-shipped: the entry of a type's shipped threshold thr (certified on the compared file's data), with its
    coverage, error and bound on this file's FIT (cf) and CERT (cc) decisions; withdrawn (null: basal-serve defers)
    when they refute it at `risk`, as a fitted threshold CERT refutes (withdrawn). Neither split took part in
    certifying it, so both test it together, counted as the independent decisions they are worth (deff_all)."""
    if thr is None:
        return dict(confidence=None, kept="shipped", reason="the compared file has no threshold for this type at this "
                                                            "max_error")
    g_fit, g_cert, g_all = gate(cf, thr, deff), gate(cc, thr, deff_c), gate(cf + cc, thr)
    entry = dict(confidence=thr, fit_coverage=g_fit["coverage"], fit_error=g_fit["error"], fit_upper95=g_fit["upper95"],
                 cert_coverage=g_cert["coverage"], cert_error=g_cert["error"], cert_upper95=g_cert["upper95"],
                 cert_accepted=g_cert["accepted"], kept="shipped")
    n, e = g_all["accepted"], g_all["errors"]
    if withdrawn(dict(confidence=thr, cert_accepted=n, cert_error=g_all["error"]), risk, deff_all)["confidence"] is None:
        entry = dict(entry, confidence=None, refuted=thr,
                     reason=f"the compared file's threshold {thr:.4f}, but your FIT and CERT decisions refute it: {e} "
                            f"error{'' if e == 1 else 's'} in {n} accepted, a one-sided 95% lower bound above {risk:g}")
    return entry


def automated_gate(rows, tempered, cal, risk):
    """Share and error of the decisions basal-serve automates at max_error `risk` with calibration `cal` (rows tempered
    with its temperatures): each type at the threshold the server applies to it (server.certified: its own; a type
    without one is not automated)."""
    if not rows:
        return dict(coverage=None, error=None)
    acc = [ok for r, (c, ok) in zip(rows, tempered.cs(rows))
           if (thr := certified(cal, risk, r["prim"])) is not None and c >= thr]
    return dict(coverage=len(acc) / len(rows), error=(len(acc) - sum(acc)) / len(acc) if acc else None)


def effective_gate(rows, temps, buckets, entry):
    """automated_gate of a fitted threshold entry (every type at its own certified threshold)."""
    return automated_gate(rows, Tempered(temps, buckets), {"thresholds": {"1.0": entry}}, 1.0)


def served_gate(rows, temps, buckets, cal, risk):
    """automated_gate of a calibration as basal-serve applies it (its temperatures given as temps / buckets)."""
    return automated_gate(rows, Tempered(temps, buckets), cal, risk)


def floor_thresholds(thresholds, targets):
    """floor() of the per-type entries and of the pooled ones of thresholds {target: entry} (in place) -> the targets
    whose entries changed."""
    keys, changed = [str(float(r)) for r in targets], set()
    for t in sorted({t for k in keys for t in (thresholds[k].get("per_type") or {})}):
        per = {k: ((thresholds[k].get("per_type") or {}).get(t) or {"confidence": None}) for k in keys}
        for k in floor(per):
            thresholds[k].setdefault("per_type", {})[t] = per[k]
            changed.add(k)
    pooled = {k: {x: v for x, v in thresholds[k].items() if x not in ("per_type", "pooled")} for k in keys}
    for k in floor(pooled):
        thresholds[k] = dict(pooled[k], **{x: thresholds[k][x] for x in ("per_type", "pooled") if x in thresholds[k]})
        changed.add(k)
    return changed


def type_threshold(entry, q_type):
    """The threshold basal-serve applies to a question type: its own certified one, else None (the pooled one is
    never applied)."""
    return ((((entry or {}).get("per_type") or {}).get(q_type)) or {}).get("confidence")


def conformal_p_min(p_gold, level):
    """Split conformal with the score 1 - p(gold): the k-th largest p(gold), k = ceil((n + 1) level); None when k > n
    (too few decisions for the level: the set is every option)."""
    k = math.ceil(round((len(p_gold) + 1) * level, 9))
    return sorted(p_gold, reverse=True)[k - 1] if k <= len(p_gold) else None


def conformal_needed(level):
    """The fewest decisions of a type that give a coverage level a p_min (ceil((n + 1) level) <= n)."""
    n = 1
    while math.ceil(round((n + 1) * level, 9)) > n:
        n += 1
    return n


def set_gate(rows, probs, p_min):
    """Coverage (the gold answer in the set), mean size and singleton share of the answer sets at p_min."""
    if not rows:
        return dict(coverage=None, mean_size=None, singleton=None)
    sets = [answer_set(p, p_min) for p in probs]
    return dict(coverage=float(np.mean([r["gold"] in s for r, s in zip(rows, sets)])),
                mean_size=float(np.mean([len(s) for s in sets])), singleton=float(np.mean([len(s) == 1 for s in sets])))


def fit_conformal(fit_rows, cert_rows, temps, buckets, levels):
    """--conformal: per coverage level and type, p_min on the type's CERT decisions (as served with the fitted
    temperatures), with the coverage and size of the sets on CERT and, held out from p_min, on FIT."""
    by_type = {}
    for t in PRIMS:
        rf, rc = [r for r in fit_rows if r["prim"] == t], [r for r in cert_rows if r["prim"] == t]
        if rf or rc:
            by_type[t] = (rf, served_p(rf, temps, buckets), rc, served_p(rc, temps, buckets))
    out = {}
    for level in levels:
        per = {}
        for t, (rf, pf, rc, pc) in by_type.items():
            p_min = conformal_p_min([p[r["gold"]] for r, p in zip(rc, pc)], level)
            e = dict(p_min=p_min, n=len(rc))
            if p_min is None:
                e["reason"] = (f"{len(rc)} CERT decisions of this type, coverage {level:g} needs >= "
                               f"{conformal_needed(level)}: the set is every option")
            g_c, g_f = set_gate(rc, pc, p_min), set_gate(rf, pf, p_min)
            e.update(cert_coverage=g_c["coverage"], cert_mean_size=g_c["mean_size"], fit_coverage=g_f["coverage"],
                     fit_mean_size=g_f["mean_size"], fit_singleton=g_f["singleton"], fit_n=len(rf))
            per[t] = e
        out[str(float(level))] = dict(per_type=per)
    return dict(method=CONFORMAL_METHOD, levels=out)


def row_problem(r):
    """Why a row (an engine readout, or a line of a --save-rows file) cannot enter the fit, None when it can: its
    probabilities must be finite numbers in [0, 1], one per option (at least two; multi: one per label), and its gold
    an option index (multi: a list of label indices)."""
    if not isinstance(r, dict) or r.get("prim") not in PRIMS + ("multi",):
        return "not a row of a decision type"
    p, g = r.get("p"), r.get("gold")
    if not isinstance(p, list) or len(p) < (1 if r["prim"] == "multi" else 2) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                    or not 0 <= x <= 1 for x in p):  # NaN fails 0 <= x
        return "probabilities that are not finite numbers in [0, 1]"
    if r.get("keys") is not None and len(r["keys"]) != len(p):
        return f"{len(p)} probabilities for {len(r['keys'])} options"
    golds = g if r["prim"] == "multi" and isinstance(g, list) else [g]
    if r["prim"] != "multi" and isinstance(g, list) or any(isinstance(x, bool) or not isinstance(x, int)
                                                           or not 0 <= x < len(p) for x in golds):
        return f"gold {g!r} is not an option index"
    return None


def clean_rows(rows):
    """The one boundary of the rows (engine readouts through raw_rows, or the lines of a --save-rows file) -> (the rows
    that can enter the fit, "line N: why" of the others): row_problem."""
    kept, dropped = [], []
    for r in rows:
        why = row_problem(r)
        if why:
            dropped.append(f"line {r.get('line', '?') if isinstance(r, dict) else '?'}: {why}")
        else:
            kept.append(r)
    return kept, dropped


def shipped_threshold(baseline, risk, t):
    """The compared file's own threshold for type t at max_error risk (thresholds[risk].per_type[t]), None when it has
    none. Unlike server.certified, never the pooled threshold of a format-1 file without per_type (the basal-1.5
    files): it was certified over all the types together, so kept per type it would become a certificate of each."""
    entry = ((baseline or {}).get("thresholds") or {}).get(str(float(risk))) or {}
    return ((entry.get("per_type") or {}).get(t) or {}).get("confidence")


def check_keep(keep, baseline):
    """--keep-shipped: the types to keep, in PRIMS order; SystemExit when a type is not choice / noul / score, or the
    compared file has no temperature or no per-type threshold for it (a shipped threshold holds with its temperature
    only; shipped_threshold)."""
    keep = set(keep)
    if not keep:
        return ()
    if keep - set(PRIMS):
        raise SystemExit(f"--keep-shipped {','.join(sorted(keep - set(PRIMS)))}: give question types among "
                         f"{', '.join(PRIMS)} (an act question counts as its base type)")
    shipped_T = (baseline or {}).get("temperature_per_prim") or {}
    for t in sorted(keep):
        if t not in shipped_T:
            raise SystemExit(f"--keep-shipped {t}: the compared calibration has no {t} temperature to keep "
                             "(give --calibration FILE or --model; not with --no-calibration)")
        risks = [float(r) for r in (baseline.get("thresholds") or {})]
        if any(shipped_threshold(baseline, r, t) is not None for r in risks):
            continue
        if any(certified(baseline, r, t) is not None for r in risks):
            raise SystemExit(f"--keep-shipped {t}: the compared calibration has no {t} threshold of its own, only one "
                             "pooled over all the types (a file written before per-type thresholds): it was not "
                             f"certified on {t} decisions, so it is not kept as their certificate")
        raise SystemExit(f"--keep-shipped {t}: the compared calibration has no {t} threshold to keep (a file fitted "
                         "on other option orders, another layout or another model loses its thresholds, as in "
                         "basal-serve)")
    return tuple(t for t in PRIMS if t in keep)


# the engines with a calibration file of their own, CALIBRATION.<engine>.json (server.calibration_name); the others
# (fast, eager, mps, strix: the bf16 path) share CALIBRATION.json
ENGINE_FILES = ("fp8", "nvfp4", "int8", "vllm", "sglang", "mlx", "gguf", "ollama", "llamacpp", "tpu")


def engine_family(engine):
    """The calibration file an engine name (calibration_name, or a file's "engine" field) stands for: its
    CALIBRATION.<engine>.json engine (a vLLM preset: vllm), "bf16" for the engines of CALIBRATION.json, None when
    unknown."""
    if not engine or engine == "unknown":
        return None
    head = str(engine).split(".")[0]
    return head if head in ENGINE_FILES else "bf16"


def file_engine(data, path):
    """The engine a compared calibration holds for (engine_family): its "engine" field (basal-calibrate writes it),
    else its file name (CALIBRATION.json and its .o<N> / .<layout> variants: bf16; CALIBRATION.<engine>...json: that
    engine); None when neither says (a file of another name)."""
    if data.get("engine"):
        return engine_family(data["engine"])
    m = re.fullmatch(r"CALIBRATION((?:\.[^.]+)*)\.json", Path(path).name if path else "")
    if not m:
        return None
    parts = m.group(1).split(".")[1:]
    return parts[0] if parts and parts[0] in ENGINE_FILES else "bf16"


def check_keep_engine(keep, baseline, path, kind, quant, preset=None):
    """--keep-shipped: SystemExit when the compared file holds for another engine than this run's (its thresholds were
    certified on that engine's probabilities, which a bf16 file's are not for GGUF or MLX); a warning when it does not
    say which engine."""
    if not keep or not baseline:
        return
    name = Path(path).name if path else "the compared calibration"
    have, want = file_engine(baseline, path), engine_family(calibration_name(kind, quant, preset) or "fast")
    if have is None:
        print(f"warning: --keep-shipped: {name} does not say which engine it was fitted on (no \"engine\" field); its "
              "kept thresholds hold only on that engine's probabilities", file=sys.stderr)
    elif have != want:
        raise SystemExit(f"--keep-shipped: {name} holds for the {have} engine, this run reads the {want} one: its "
                         "thresholds were certified on that engine's probabilities (compare with the file of this "
                         "engine, or leave out --keep-shipped)")


def kept_from(path, data):
    """fitted_on.kept_shipped_from: the compared file the kept types come from (its name, the sha256 of its bytes and
    its certified_on when it says), so that the written file names where their thresholds were certified."""
    out = dict(file=Path(path).name, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if isinstance(data.get("certified_on"), str):
        out["certified_on"] = data["certified_on"]
    return out


def fit(rows, baseline, targets, min_per_type=50, fmt=1, min_family=30, reference=None, min_per_set=30,
        min_blend=MIN_BLEND, conformal=(), keep=()):
    """Rows with uncalibrated probabilities -> (calibration dict, report for the table). fmt 2: calibration file 2,
    with the choice temperatures per option-count bucket. Multi rows fit the fallback_threshold of their label set.
    Rows that cannot enter the fit (clean_rows: NaN probabilities, a gold out of range) are left out and reported.
    The types in `keep` (--keep-shipped, see check_keep) take the temperatures and thresholds of `baseline`."""
    rows, dropped = clean_rows(rows)
    if not rows:
        raise SystemExit("no decisions to fit: " + "; ".join(dropped[:3]))
    sets = [r for r in rows if r["prim"] == "multi"]
    rows = [r for r in rows if r["prim"] != "multi"]
    fit_rows = [r for r in rows if not is_cert(r["group"])]
    cert_rows = [r for r in rows if is_cert(r["group"])]
    shipped_T = baseline.get("temperature_per_prim", {})
    shipped_B = baseline.get("temperature_per_bucket", {}) if fmt == 2 else {}
    keep = check_keep(keep, baseline)
    temps, source, buckets, bsource = fit_temps(fit_rows, shipped_T, shipped_B, min_per_type, fmt, min_blend, keep)
    fitted, shipped = Tempered(temps, buckets), Tempered(shipped_T, shipped_B)  # each row tempered once
    cs_fit, cs_cert = fitted.cs(fit_rows), fitted.cs(cert_rows)
    shipped_cert = shipped.cs(cert_rows)
    thresholds, report = {}, dict(types={}, targets={}, n_fit=len(fit_rows), n_cert=len(cert_rows), compared={},
                                  dropped=dropped)
    _, deff_pool = design_effect(cs_fit, [r["group"] for r in fit_rows])
    for risk in targets:
        entry, g_cert = certify(cs_fit, cs_cert, risk, deff_pool, veto=False, method=True)  # informational
        entry["per_type"] = per_type_thresholds(fit_rows, cert_rows, temps, buckets, risk, fitted, baseline, keep)
        entry["pooled"] = POOLED_NOTE
        thresholds[str(float(risk))] = entry
        if (baseline.get("thresholds") or {}).get(str(float(risk))):  # the compared file as basal-serve applies it
            report["compared"][str(float(risk))] = automated_gate(cert_rows, shipped, baseline, risk)
        old = ((baseline.get("thresholds") or {}).get(str(float(risk))) or {}).get("confidence")
        report["targets"][risk] = dict(shipped=old, shipped_cert=gate(shipped_cert, old) if old is not None else None,
                                       fitted=entry, fitted_cert=g_cert,
                                       shipped_per_type={t: c for t in PRIMS
                                                         if (c := shipped_threshold(baseline, risk, t)) is not None})
    for k in floor_thresholds(thresholds, targets):  # a floored target's figures, as served
        report["targets"][float(k)].update(fitted=thresholds[k], fitted_cert=gate(cs_cert, thresholds[k]["confidence"]))
    automated, automated_pooled, automated_error = {}, {}, {}
    for k, entry in thresholds.items():
        eff = automated_gate(cert_rows, fitted, {"thresholds": thresholds}, float(k))
        automated[k], automated_error[k] = eff["coverage"], eff["error"]
        automated_pooled[k] = gate(cs_cert, entry["confidence"])["coverage"] if cs_cert else None
    for t in PRIMS:
        rs = [r for r in rows if r["prim"] == t]
        if not rs:
            continue
        cert = [r for r in cert_rows if r["prim"] == t]
        # as served: a choice question of a format-2 file gets its option-count bucket temperature first
        fit_t = [r for r in rs if not is_cert(r["group"])]
        report["types"][t] = dict(n=len(rs), n_fit=len(rs) - len(cert), n_cert=len(cert),
                                  fit_groups=len({r["group"] for r in fit_t}),
                                  **dict(zip(("icc", "design_effect"),
                                             design_effect(fitted.cs(fit_t), [r["group"] for r in fit_t]))),
                                  accuracy=float(np.mean([ok for _, ok in served(rs, {})])),
                                  T_shipped=shipped_T.get(t), T=temps.get(t), source=source.get(t, "default 1.0"),
                                  ece_shipped=ece(shipped.cs(cert)), ece=ece(fitted.cs(cert)))
        if fmt == 2 and t == "choice":  # format 2: a choice question takes its bucket's temperature first
            report["types"][t]["not_served"] = dict(shipped=buckets_cover(shipped_B), fitted=buckets_cover(buckets))
    gates = dict(ece_lofo=lofo(fit_rows, cert_rows, shipped_T, shipped_B, min_per_type, fmt, min_family, reference,
                               min_blend=min_blend, keep=keep),
                 automated_at=automated, automated_error=automated_error, automated_at_pooled=automated_pooled,
                 min_family=min_family)
    report["gates"] = gates
    report["kept"] = list(keep)
    multi = fit_fallback(sets, temps.get("noul", 1.0), min_per_set) if sets else {}
    report["multi"] = multi
    extra = dict(gates=gates)
    if multi:
        extra["multi"] = {k: v for k, v in multi.items() if "reason" not in v}
    if conformal:
        extra["conformal"] = report["conformal"] = fit_conformal(fit_rows, cert_rows, temps, buckets, conformal)
    if fmt == 2:
        report["buckets"] = {}
        for name, lo, hi in BUCKETS:
            rs = [r for r in rows if r["prim"] == "choice" and lo <= len(r["p"]) <= hi]
            if rs:
                cert = [r for r in rs if is_cert(r["group"])]
                old = (shipped_B.get("choice") or {}).get(name)
                new = (buckets.get("choice") or {}).get(name)
                # as served: the bucket's temperature, else the type's (T_*_type: the type's stands in)
                report["buckets"][name] = dict(n=len(rs), n_cert=len(cert),
                                               T_shipped=old if old is not None else shipped_T.get("choice"),
                                               T_shipped_type=old is None,
                                               T=new if new is not None else temps.get("choice"), T_type=new is None,
                                               source=bsource[name],
                                               ece_shipped=ece(shipped.cs(cert)), ece=ece(fitted.cs(cert)))
        cal = {"prompt": 2, "temperature_per_prim": temps, "temperature_per_bucket": buckets,
               "temperature_source": dict(source, **{f"choice {k}": v for k, v in bsource.items()}),
               "thresholds": thresholds, **extra}
        return cal, report
    # (prompt 1 said, as a file 2 says 2: basal-serve refuses the file for a model of the other format at start-up)
    return dict(prompt=1, temperature_per_prim=temps, temperature_source=source, thresholds=thresholds, **extra), report


def as_served(data, path, fmt=None):
    """The compared file as basal-serve reads it, checked before any readout: SystemExit where basal-serve refuses it
    at start-up (a temperature that is no finite number above 0, a bad bucket name, a file of another prompt format
    than fmt, the model's: check_calibration), and its temperatures as numbers (basal-serve reads "1.2" as 1.2; the
    fit blended and printed the text and failed after the whole readout). fmt None: the file's own format."""
    if not data:
        return data
    check_calibration(data, fmt if fmt is not None else data.get("prompt", 1), path)
    out = dict(data)
    prim, buckets = data.get("temperature_per_prim"), data.get("temperature_per_bucket")
    if isinstance(prim, dict):
        out["temperature_per_prim"] = {t: float(T) if t in QUESTION_TYPES else T for t, T in prim.items()}
    if isinstance(buckets, dict):
        out["temperature_per_bucket"] = {t: {b: float(T) for b, T in bs.items()} if t in QUESTION_TYPES else bs
                                         for t, bs in buckets.items()}
    return out


def compared(md, kind, path, quant, orders, off, preset=None, layout="canonical", adapter=None, identity=None,
             fmt=None, served=None):
    """The calibration to compare with and keep, as basal-serve would apply it with these option orders
    (server.load_calibration: the model's file for those orders when it ships one; a file fitted on other orders
    without its thresholds; with --adapter the adapter's file, else the model's without its thresholds) and, with
    `identity` = (model, revision, adapter identity) of the decisions, without the thresholds of a file fitted on
    another model, revision or adapter (as basal-serve keeps an own file: validate_baseline) -> (calibration, its file
    or None, why its thresholds were left out or None). --url to a server with --adapter (adapter None, the adapter's
    identity in `identity`): the model's file is the base model's, without its thresholds, as load_calibration keeps
    it in process. served: (snapshot, model_files digest) the identity's revision is checked on instead of md, when
    md is not that revision's snapshot (--url to a server that runs another revision: (None, its digest))."""
    if off or (md is None and not path):
        return {}, None, None
    if orders is None:  # a server that does not report its orders: the file as it is
        f = calibration_file(md, kind, path, quant, preset, layout)
        data, why = (json.loads(read_text(f)) if f.exists() else {}), None
    else:
        data, f, why = load_calibration(md, kind, path, quant, orders, preset=preset, layout=layout, adapter=adapter)
    if why is None and identity is not None and data:
        imd, files = served if served is not None else (md, None)
        mismatch = identity_mismatch(data, *identity, md=imd, kind=kind, files=files)
        if mismatch:
            why, data = f"{f.name}: {mismatch}", {k: v for k, v in data.items() if k not in UNVALIDATED_OUT}
    if why is None and adapter is None and identity is not None and identity[2] and not path and data:
        why = f"{f.name} belongs to the base model, read with --adapter {identity[2]}"
        data = {k: v for k, v in data.items() if k not in UNVALIDATED_OUT}
    if why:
        print(f"warning: {why}: its thresholds are not compared or kept (as in basal-serve)", file=sys.stderr)
    return as_served(data, f, fmt), (f if f.exists() else None), why


def task_rows(units, responses):
    """Labelled task decisions + the served task answers -> (rows {task, group, c: confidence, ok}, failed requests).
    An extract number that does not parse is left out: basal-serve never automates it."""
    rows, failed = [], {}
    for u in units:
        r = responses[u["req"]]
        if isinstance(r, Exception):
            failed[u["req"]] = f"line {u['line']}: {r}"
            continue
        a = (r.get("answers") or {}).get(u["name"])
        if not isinstance(a, dict):
            failed[(u["req"], u["name"])] = f"line {u['line']}, question {u['name']!r}: no answer"
            continue
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
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:  # NaN is not in [0, 1]
            failed[(u["req"], u["name"], u["part"])] = f"line {u['line']}, question {u['name']!r}: confidence {conf!r}"
            continue
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
        rho, deff = design_effect(cf, [r["group"] for r in rs if not is_cert(r["group"])])
        deff_c = design_effect_of([r["group"] for r in rs if is_cert(r["group"])], rho)
        e = dict(n=len(rs), n_fit=len(cf), n_cert=len(cc), accuracy=float(np.mean([r["ok"] for r in rs])),
                 fit_groups=len({r["group"] for r in rs if not is_cert(r["group"])}), icc=rho, design_effect=deff,
                 method=METHOD, targets={})
        for risk in targets:
            old = ((shipped or {}).get(t) or {}).get(str(float(risk)))
            if old is not None:
                e.setdefault("shipped", {})[str(float(risk))] = dict(confidence=old, cert=gate(cc, old))
            e["targets"][str(float(risk))], _ = certify(cf, cc, risk, deff, deff_c)
        floor(e["targets"])  # a looser target never keeps a stricter threshold
        thresholds.update({t: {k: x["confidence"] for k, x in e["targets"].items() if x["confidence"] is not None}}
                          if any(x["confidence"] is not None for x in e["targets"].values()) else {})
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
    lf = (json.loads(read_text(p)).get("gates") or {}).get("ece_lofo") or {}
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
    w = 52
    lines = ["", f"{'type':<7}{'n':>6}{'FIT':>6}{'CERT':>6}{'accuracy':>10}   {'temperature shipped -> fitted':<{w}}"
                 f"ECE on CERT shipped -> fitted"]
    for t, r in report["types"].items():
        ns = r.get("not_served") or {}  # format 2, choice: buckets cover every option count
        temp = (f"{num(r['T_shipped'])}{' (not served: buckets)' if ns.get('shipped') else ''} -> {num(r['T'])}"
                f"{' (not served)' if ns.get('fitted') else ''}") + (
            "" if r["source"].startswith("fitted") else " (blended)" if r["source"].startswith("blended") else " (kept)")
        lines.append(f"{t:<7}{r['n']:>6}{r['n_fit']:>6}{r['n_cert']:>6}{r['accuracy']:>10.3f}   {temp:<{w}}"
                     f"{num(r['ece_shipped'])} -> {num(r['ece'])}")
    for name, r in (report.get("buckets") or {}).items():  # calibration file 2: choice per option-count bucket
        temp = (f"{num(r.get('T_shipped'))}{' (type)' if r.get('T_shipped_type') else ''} -> "
                f"{num(r['T'])}{' (type)' if r.get('T_type') else ''}")
        lines.append(f"  {name:<9}{r['n']:>6}{'':>6}{r['n_cert']:>6}{'':>10}   {temp:<{w}}"
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
                           else "none (defers)") + (" kept shipped" if x.get("kept") else "")
                for t, x in per.items()))
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
    if report.get("conformal"):
        lines += ["", f"answer sets        {'type':<7}{'CERT':>6}{'p_min':>9}{'CERT cover':>12}   {'FIT cover':>9}"
                      f"{'mean size':>11}{'singleton':>11}   (p_min on CERT, checked on FIT)"]
        for level, e in report["conformal"]["levels"].items():
            for t, x in e["per_type"].items():
                head = f"coverage {float(level):<9g} {t:<7}{x['n']:>6}"
                if x["p_min"] is None:
                    lines.append(head + f"   none: {x['reason']}")
                    continue
                lines.append(head + f"{x['p_min']:>9.4f}{pct(x['cert_coverage']):>12}   {pct(x['fit_coverage']):>9}"
                                    f"{num(x['fit_mean_size'], '{:.2f}'):>11}{pct(x['fit_singleton']):>11}")
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


def clustered(name, n_fit, groups, rho=None, deff=None):
    """The note of a type or task with more FIT decisions than groups (None when each group has one): how strongly
    their errors go together within a group, and so how much the bound discounts them."""
    if not groups or n_fit <= groups:
        return None
    head = f"{name}: {n_fit} FIT decisions in {groups} groups ({n_fit / groups:.1f} per group)"
    if not rho:
        return head + ": their errors do not go together within a group (ICC 0), so the bound counts them as they are"
    return (head + f": their errors go together within a group (ICC {rho:.2f}), so the bound counts them as "
            f"{deff:.2f} times fewer independent decisions; certifying a target takes more labelled groups, not more "
            "decisions per group")


def task_warnings(report, dropped=()):
    out = []
    for t, e in report.items():
        if e["n"] < RECOMMENDED:
            out.append(f"{t}: only {e['n']} labelled decisions: the bounds are loose")
        if clustered(t, e["n_fit"], e.get("fit_groups"), e.get("icc"), e.get("design_effect")):
            out.append(clustered(t, e["n_fit"], e.get("fit_groups"), e.get("icc"), e.get("design_effect")))
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
    if report.get("dropped"):
        out.append(f"{len(report['dropped'])} row(s) left out of the fit: " + "; ".join(report["dropped"][:3]))
    for t, r in report["types"].items():
        if clustered(t, r["n_fit"], r.get("fit_groups"), r.get("icc"), r.get("design_effect")):
            out.append(clustered(t, r["n_fit"], r.get("fit_groups"), r.get("icc"), r.get("design_effect")))
        if r["source"].startswith("fitted") and any(math.isclose(r["T"], x) for x in T_GRID):
            out.append(f"{t}: the fitted temperature is at the edge of the search ({r['T']:.2f}): too few errors "
                       "or too few decisions to fit it; label more or raise --min-per-type")
    for b, r in (report.get("buckets") or {}).items():
        if r.get("T") is not None and not r.get("T_type") and str(r.get("source", "")).startswith("fitted") \
                and any(math.isclose(r["T"], x) for x in T_GRID):
            out.append(f"choice {b} options: the fitted temperature is at the edge of the search ({r['T']:.2f}): too "
                       "few errors or too few decisions to fit it")
    for risk in targets:
        r = report["targets"][risk]
        for t, x in (r["fitted"].get("per_type") or {}).items():
            if x.get("confidence") is None:
                shipped = (r.get("shipped_per_type") or {}).get(t)
                out.append(f"max_error {risk:g}: {t} has no certified threshold; basal-serve defers every automatic "
                           f"action of act and tool on {t} questions with this max_error"
                           + (f" (the compared file certified {shipped:.4f}: --keep-shipped {t} keeps its {t} "
                              "temperature and thresholds together)" if shipped is not None and not x.get("kept")
                              else "")
                           + (f" (the kept shipped threshold {x['refuted']:.4f} is refuted by your FIT and CERT "
                              "decisions)" if x.get("kept") and x.get("refuted") is not None else ""))
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
    try:
        lines = read_lines(a.input)
    except (OSError, UnicodeDecodeError) as e:  # a binary file, a directory
        raise SystemExit(f"{a.input}: not a file written by basal-calibrate --save-rows ({type(e).__name__})") from None
    try:
        head = json.loads(lines[0])
    except (IndexError, ValueError):
        head = {}
    if not isinstance(head, dict) or head.get("basal_calibrate_rows") != 1:
        raise SystemExit(f"{a.input}: not a file written by basal-calibrate --save-rows")
    rows = []
    for i, x in enumerate(lines[1:], 2):
        if not x.strip():
            continue
        try:
            row = json.loads(x)
        except ValueError:
            row = None
        if not isinstance(row, dict):
            raise SystemExit(f"{a.input}: line {i} is not a row written by basal-calibrate --save-rows")
        rows.append(row)
    good, _ = clean_rows(rows)  # what fit() keeps (it reports the others): the counts below
    small = sample_size(good, [], targets)
    for w in small:
        print(f"warning: {w}", file=sys.stderr)
    base_file = None if a.no_calibration else (
        Path(a.calibration) if a.calibration else (Path(head["baseline"]) if head.get("baseline") else None))
    baseline = as_served(json.loads(read_text(base_file)), base_file, head.get("fmt")) \
        if base_file is not None and base_file.exists() else {}
    why = validate_baseline(baseline, base_file.name, head.get("orders"), head.get("layout"),
                            (head.get("model") if head.get("model") not in (None, "None") else None,
                             head.get("revision"), head.get("adapter"), None, head.get("kind"),
                             head.get("model_files"), head.get("model_content"))) if baseline else None
    if baseline and not why and not a.calibration and head.get("baseline_why"):
        why = head["baseline_why"]  # the run left out this file's thresholds (the base model's under --adapter)
    if why:  # basal-serve's rule, for the model, orders and layout the rows were read with
        print(f"warning: {why} (the rows): its thresholds are not compared or kept (as in basal-serve)",
              file=sys.stderr)
        baseline = {k: v for k, v in baseline.items() if k not in UNVALIDATED_OUT}
    check_keep_engine(check_keep(a.keep_shipped, baseline), baseline, base_file, head["kind"], head["quant"],
                      head.get("vllm_preset"))
    cal, report = fit(rows, baseline, targets, a.min_per_type, head["fmt"], a.min_family, reference, a.min_per_set,
                      a.min_blend, a.conformal, a.keep_shipped)
    cal["gates"]["ece_reference"] = reference_name
    cal["engine"] = engine_name(head["kind"], head["quant"], head.get("vllm_preset"))
    cal["fitted_on"] = dict(data=head["data"], decisions=sum(r["prim"] != "multi" for r in good),
                            requests=head["requests"], fit=report["n_fit"], cert=report["n_cert"],
                            per_type={t: r["n"] for t, r in report["types"].items()},
                            multi=sum(r["prim"] == "multi" for r in good), date=datetime.date.today().isoformat(),
                            model=head["model"], engine=head["engine"], orders=head["orders"],
                            split="FIT 70% / CERT 30% by group", tool=f"basal-calibrate {__version__} (--from-rows)")
    cal["fitted_on"].update({k: head[k] for k in ("layout", "adapter", "revision", "model_files", "model_content")
                             if head.get(k) is not None})
    if report["kept"]:
        cal["fitted_on"].update(kept_shipped=report["kept"], kept_shipped_from=kept_from(base_file, baseline))
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
    add_api_key(ap)
    ap.add_argument("--calibration", default=None,
                    help="the calibration to compare with and to keep for small types (default: the file basal-serve "
                         "loads for --model and the engine)")
    ap.add_argument("--no-calibration", dest="no_calibration", action="store_true",
                    help="no shipped calibration to compare with (types below --min-per-type keep temperature 1.0)")
    ap.add_argument("--targets", default="0.01,0.05", help="max_error targets (comma-separated)")
    ap.add_argument("--conformal", default=None,
                    help="coverage levels of the conformal answer sets (comma-separated, e.g. 0.8,0.9,0.95; default: "
                         "none), fitted per type on CERT; basal-serve answers \"coverage\": L with \"answer_set\"")
    ap.add_argument("--min-per-type", dest="min_per_type", type=int, default=50,
                    help="FIT decisions needed to fit a type's temperature")
    ap.add_argument("--min-blend", dest="min_blend", type=int, default=MIN_BLEND,
                    help="a type (or bucket) with fewer FIT decisions than --min-per-type but at least this many gets "
                         "the fitted temperature blended with the shipped one in proportion n / --min-per-type; 0 "
                         "keeps the shipped one")
    ap.add_argument("--keep-shipped", dest="keep_shipped", default=None,
                    help="question types (choice,noul,score) that keep the compared file's temperature and thresholds "
                         "together instead of being fitted, e.g. a type these labels do not certify (with task "
                         "questions only, the compared file's decision calibration is kept whole anyway)")
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
    ap.add_argument("--concurrency", type=positive, default=32, help="requests in flight (the engine batches them)")
    add_engine_args(ap.add_argument_group("engine (in process, as basal-serve)"))
    ap.set_defaults(model=None)
    return ap


def options(a):
    """The checked targets, conformal levels (into a.conformal) and LOFO reference of the command line -> (targets,
    reference values, reference name)."""
    try:
        targets = sorted({float(x) for x in a.targets.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--targets {a.targets!r}: give numbers such as 0.01,0.05") from None
    if not targets or not all(1e-6 <= r < 1 for r in targets):
        raise SystemExit("--targets must lie between 1e-06 and 1 (a smaller max_error needs millions of decisions)")
    try:
        a.conformal = sorted({float(x) for x in (a.conformal or "").split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--conformal {a.conformal!r}: give coverage levels such as 0.8,0.9,0.95") from None
    if not all(0 < x < 1 for x in a.conformal):
        raise SystemExit("--conformal levels must lie between 0 and 1")
    # (checked before any readout: --min-family 0 and --min-blend -1 failed with a traceback after the whole run)
    for flag, value, low in (("--min-family", a.min_family, 1), ("--min-blend", a.min_blend, 0),
                             ("--min-per-type", a.min_per_type, 0), ("--min-per-set", a.min_per_set, 0)):
        if value < low:
            raise SystemExit(f"{flag} {value}: give {low} or more")
    a.keep_shipped = tuple(x.strip() for x in (a.keep_shipped or "").split(",") if x.strip())
    if a.calibration and not Path(a.calibration).exists():
        raise SystemExit(f"--calibration {a.calibration}: no such file")
    if a.strict and a.reference is None:
        raise SystemExit("--strict needs --reference (basal-1.5 or a calibration file with gates.ece_lofo)")
    return (targets, *reference_values(a.reference))


def labelled(a, targets):
    """The labelled input -> (decision requests, decisions, task requests, task decisions, sample-size warnings), the
    warnings printed before any readout (a target these labels cannot certify)."""
    try:
        bodies, decisions, task_bodies, units = read(a.input)
    except InputError as e:
        raise SystemExit(str(e)) from None
    except OSError as e:  # (no such file, a directory, no permission: a message, not a traceback)
        raise SystemExit(f"{a.input}: {e.strerror or type(e).__name__}") from None
    except UnicodeDecodeError:
        raise SystemExit(f"{a.input}: not a text file (UTF-8, or UTF-16 with its BOM)") from None
    print(f"basal-calibrate: {len(decisions)} labelled decisions in {len(bodies)} requests"
          + (f", {len(units)} labelled task decisions in {len(task_bodies)} requests" if units else ""),
          file=sys.stderr)
    small = sample_size(decisions, units, targets)
    for w in small:
        print(f"warning: {w}", file=sys.stderr)
    if units and a.url:
        raise SystemExit("task questions (classify / verify / extract) need the engine in process: their thresholds "
                         "are certified on the answers as served with the fitted temperatures; give --model and --mode "
                         "instead of --url, or leave the task questions out")
    return bodies, decisions, task_bodies, units, small


def server_identity(c, entry, a):
    """--url --model <repo id>: the revision and model files fitted_on records are the ones the server serves
    (GET /v1/models, 2.8), not those of the revision the local cache resolves --model to (c.revision, c.files),
    which may be another one. c.served: what the compared file's identity is checked against when they differ (no
    snapshot of the server's revision here, its model_files digest). As before: --model that is no repo id (no
    revision), and a server that serves no cache snapshot (revision null). A server before 2.8 does not report them:
    the cache's are kept, with a warning."""
    local = c.revision
    if not local:
        return
    if "revision" not in entry:
        print(f"warning: the server does not report the revision it serves (basal-serve before 2.8): fitted_on "
              f"records {local[:12]}, the revision the local cache resolves {a.model} to; if the server runs another "
              "one, pass it with --revision", file=sys.stderr)
        return
    if not entry["revision"]:
        return
    c.revision = entry["revision"]
    c.files = entry.get("model_files") or (c.files if local == c.revision else None)
    if local != c.revision:
        c.served = (None, c.files)
        print(f"warning: the server serves revision {c.revision[:12]} of {c.model}, the local cache resolves "
              f"{a.model} to {local[:12]}: fitted_on records the server's, and the file compared with is the "
              f"cache's (--revision {c.revision} compares with the server's own)", file=sys.stderr)


def http_context(a):
    """--url: the running server (it must serve raw probabilities) and the calibration to compare with -> the engine
    context (base URL, mode, engine text, adapter identity, orders, kind, quant, preset, model, revision, model files,
    prompt format, baseline and its file)."""
    import httpx
    base = a.url.rstrip("/").removesuffix("/v1/systemone").removesuffix("/v1/basal")
    if not (a.model or a.calibration or a.no_calibration):
        raise SystemExit("with --url, give --model (its shipped calibration is compared with and kept for small "
                         "types), --calibration FILE or --no-calibration")
    try:
        r = httpx.get(base + "/v1/models", timeout=30, headers=headers(a.api_key))
        if refused(base, r.status_code, a.api_key):
            raise SystemExit(f"--url {a.url}: {refused(base, r.status_code, a.api_key)}")
        entry = uncalibrated(r.json()["models"][0], base)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise SystemExit(f"--url {a.url}: no basal server answers ({e})") from None
    c = argparse.Namespace(base=base, mode=a.mode or entry.get("mode"))
    c.adapter = entry.get("adapter")  # the server's --adapter identity (name@sha256)
    c.engine = f"{c.mode} at {base}" + (f" --adapter {c.adapter.split('@')[0]}" if c.adapter else "")
    c.orders = entry.get("orders")  # the server's option orders (servers before this field: unknown)
    c.kind, c.quant = MODES.get(c.mode, ("", None, False))[:2]
    c.preset = entry.get("vllm_preset")  # --mode vllm: the server's preset (servers before 2.2.1: none)
    md = resolve(a.model, a.revision, metadata_only=True) if a.model else None
    c.model, c.revision = model_identity(a.model, md) if md is not None else (a.model or entry.get("name"), None)
    c.files = model_files_id(md, c.kind) if md is not None else None  # (when the cache holds the weights too)
    c.content = model_content_id(md, c.kind) if md is not None else None  # (a local --model folder: its files)
    c.served = None
    if md is not None:  # the revision and model files the server serves (2.8)
        server_identity(c, entry, a)
    c.fmt = entry.get("prompt_format") or (model_format(md) if md is not None else 1)
    # the layout the server reads prompts in (servers before 2.4.1 and canonical ones do not say)
    a.layout = getattr(a, "layout", None) or entry.get("layout") or (model_layout(md) if md is not None
                                                                      else "canonical")
    # without --model the server's name stands in for the model (its --name, or the folder's name with "+adapter"): no
    # identity to check the compared file against
    identity = (c.model, c.revision, c.adapter) if md is not None else None
    c.baseline, c.base_file, c.base_why = compared(md, c.kind, a.calibration, c.quant, c.orders, a.no_calibration,
                                                   c.preset, a.layout, fmt=c.fmt, identity=identity, served=c.served)
    return c


def local_context(a):
    """The engine in process, as basal-serve starts it without calibration -> (server, engine context as
    http_context's)."""
    a.model = a.model or add_engine_args(argparse.ArgumentParser()).get_default("model")  # basal-serve's
    refuse_lean_flex(a.lean, a.tree)  # before default_mode, which resolves the model on CUDA GPUs without FA3
    mode = a.mode or default_mode(a.model, a.revision)
    srv = Server(argparse.Namespace(**{**vars(a), "mode": mode, "name": None, "calibration": None,
                                       "no_calibration": True, "log_decisions": None}))
    c = argparse.Namespace(mode=mode, kind=srv.kind, quant=a.quant or MODES[mode][1], fmt=srv.fmt, orders=srv.orders,
                           preset=getattr(srv.backend, "preset", None))
    a.layout = srv.layout  # --layout, else the model's basal.json "layout"
    c.model, c.revision = model_identity(a.model, srv.md)
    c.files = model_files_id(srv.md, srv.kind)
    c.content = model_content_id(srv.md, srv.kind)
    c.adapter = srv.adapter.identity if srv.adapter else None
    c.baseline, c.base_file, c.base_why = compared(srv.md, srv.kind, a.calibration, c.quant, c.orders,
                                                   a.no_calibration, c.preset, a.layout, srv.adapter,
                                                   identity=(c.model, c.revision, c.adapter), fmt=c.fmt)
    c.engine = mode + (f" --quant {a.quant}" if a.quant else "") + (f" --gguf {Path(a.gguf).name}" if a.gguf else "") \
        + (f" --vllm-preset {c.preset}" if c.preset not in (None, "default") else "") \
        + (f" --adapter {srv.adapter.name}" if srv.adapter else "")
    return srv, c


def main(argv=None):
    a = parser().parse_args(argv)
    targets, reference, reference_name = options(a)
    if a.from_rows:
        return refit(a, targets, reference, reference_name)
    bodies, decisions, task_bodies, units, small = labelled(a, targets)
    fitted, c = {}, None  # settle(): the decision rows, the file to serve and the report; c: the engine context

    def settle(responses):
        """The decision part: the fit on the decision questions, else the compared calibration kept as it is."""
        rows, failed = raw_rows(decisions, responses)
        if failed:
            print(f"warning: {len(failed)} request(s) failed and are left out:\n  " + "\n  ".join(failed[:5]),
                  file=sys.stderr)
        if decisions and not rows:
            raise SystemExit("no decisions to fit")
        if rows:
            check_keep_engine(check_keep(a.keep_shipped, c.baseline), c.baseline, c.base_file, c.kind, c.quant,
                              c.preset)
            cal, report = fit(rows, c.baseline, targets, a.min_per_type, c.fmt, a.min_family, reference, a.min_per_set,
                              a.min_blend, a.conformal, a.keep_shipped)
            cal["gates"]["ece_reference"] = reference_name
        else:  # task questions only: the temperatures and thresholds of the compared file stay as they are
            cal, report = {k: v for k, v in c.baseline.items() if k not in KEPT_OUT}, None
        fitted.update(rows=rows, failed=failed, cal=cal, report=report)
        return cal

    if a.url:
        c = http_context(a)
        settle(asyncio.run(over_http(c.base + "/v1/systemone", bodies, a.concurrency, a.api_key)) if bodies else [])
        task_responses = []
    else:
        srv, c = local_context(a)
        _, task_responses = asyncio.run(two_passes(srv, bodies, task_bodies, a.concurrency, settle))
    finish(a, c, fitted, targets, small, len(bodies), units, len(task_bodies), task_responses)


def finish(a, c, fitted, targets, small, n_bodies, units, n_task_bodies, task_responses):
    """After the readouts: the task thresholds, the saved rows, the file's provenance (fitted_on), the report and the
    file."""
    rows, failed, cal, report = fitted["rows"], fitted["failed"], fitted["cal"], fitted["report"]
    trows, tfailed = task_rows(units, task_responses) if units else ([], [])
    if tfailed:
        print(f"warning: {len(tfailed)} task request(s) failed and are left out:\n  " + "\n  ".join(tfailed[:5]),
              file=sys.stderr)
    if units and not trows:
        raise SystemExit("no task decisions to fit")
    shipped_tasks = c.baseline.get("task_thresholds") or {}
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
        head = dict(basal_calibrate_rows=1, fmt=c.fmt, kind=c.kind, quant=c.quant, vllm_preset=c.preset,
                    model=str(c.model), engine=c.engine, data=Path(a.input).name, requests=n_bodies - len(failed),
                    baseline=str(c.base_file) if c.baseline else None, orders=c.orders)
        head.update(layout=a.layout, adapter=c.adapter, revision=c.revision, model_files=c.files,
                    model_content=c.content,
                    baseline_why=c.base_why if c.baseline else None)  # (its thresholds left out: the refit too)
        with open(a.save_rows, "w", encoding="utf-8") as f:
            for x in [head] + rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    cal["engine"] = engine_name(c.kind, c.quant, c.preset)
    good = clean_rows(rows)[0]
    cal["fitted_on"] = dict(data=Path(a.input).name, decisions=sum(r["prim"] != "multi" for r in good),
                            requests=n_bodies - len(failed),
                            fit=report["n_fit"] if report else 0, cert=report["n_cert"] if report else 0,
                            per_type={t: r["n"] for t, r in report["types"].items()} if report else {},
                            multi=sum(r["prim"] == "multi" for r in good),
                            date=datetime.date.today().isoformat(), model=c.model, engine=c.engine,
                            split="FIT 70% / CERT 30% by group", tool=f"basal-calibrate {__version__}")
    if c.revision:
        cal["fitted_on"]["revision"] = c.revision
    if c.revision and c.files:  # basal-serve applies it to another revision with the same model files (model_files)
        cal["fitted_on"]["model_files"] = c.files
    if c.content:  # a local folder: matched by its files, not its name (server.identity_mismatch)
        cal["fitted_on"]["model_content"] = c.content
    if report and report["kept"]:  # their temperatures and thresholds are the compared file's, not fitted here
        cal["fitted_on"].update(kept_shipped=report["kept"], kept_shipped_from=kept_from(c.base_file, c.baseline))
    if c.adapter:  # basal-serve applies the file only with this adapter (calibration_mismatch)
        cal["fitted_on"]["adapter"] = c.adapter
    if task_report:
        cal["fitted_on"]["tasks"] = {t: e["n"] for t, e in task_report.items()}
        cal["fitted_on"]["task_requests"] = n_task_bodies - len(tfailed)
    if c.orders is not None:  # basal-serve applies the file only to these orders (or loads CALIBRATION.o<N>.json)
        cal["fitted_on"]["orders"] = c.orders
    if getattr(a, "layout", None) not in (None, "canonical"):  # basal-serve applies it only in this layout (2.4.1)
        cal["fitted_on"]["layout"] = a.layout
    cal["certified_on"] = "CERT split of fitted_on.data (30% by group)"
    cal["baseline"] = (str(a.calibration or f"{a.model}/{c.base_file.name}") if c.baseline else "none") + \
        (f" ({c.baseline['source_tag']})" if c.baseline.get("source_tag") else "")
    if not rows:
        cal["decision_calibration"] = (f"kept from {cal['baseline']} (no decision questions in the input)" if c.baseline
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
        print(f"\nwrote {a.out} (calibration file {c.fmt}, engine {cal['engine']})\n"
              f"  basal-serve --model {a.model or '<model>'} --mode {c.mode}"
              + (f" --adapter {a.adapter}" if getattr(a, "adapter", None) else "") + f" --calibration {a.out}")
    summary(small + cert_notes(report, task_report, targets))
    if a.strict and report:
        strict_check(report, a.min_family)


if __name__ == "__main__":
    main()
