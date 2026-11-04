"""The rank question type (experimental, 2.5): rank thousands of candidates in one request, beyond the 255 options of
a choice question.

  {"type": "rank", "instructions": "Does {candidate} contain information needed to answer the user's question?",
   "candidates": {key: text} | [text, ...] | [{"key": ..., "text": ..., "group": ...}, ...],   (10,000 per request)
   "label": "Passage", "list_title": "Passages", "chunk": 50, "top": 20, "choice_instructions": "Which passage ...?",
   "choice_context": "auto" | "screen" | "shortlist", "order": "state_first" | "candidates_first",
   "prior": {key: weight}, "limit": 100}

Screening: the candidates are numbered in a list ("Passage 7: <text>", under a "[<group>]" line when they have one) and
packed, in their order, into requests of `chunk` candidates (a group stays in one request when it fits). Each request's
state is the request's own state and its list; every candidate in it is one yes / no question ("{candidate}" in the
instructions names it), so the request reads its state once (SOAM) and judges its candidates side by side. All requests
go through the server's queue together. A candidate's score is its calibrated P(yes), times its `prior` weight.

Final choice (`top` >= 2): one choice question over the top `top` candidates by score, whose options are their names in
the list; its probabilities reorder them, the others keep their score order. Its context is the screening request's
state when one request held every candidate (`screen`: with --state-cache-gb that state is cached, so only the question
is computed), or the request's state with a list of only the shortlisted candidates (`shortlist`); `auto` picks
`screen` when it can. `candidates_first` puts the list before the request's state, so requests about the same
candidates share a cached prefix (--state-cache-gb) across different states.

Answer: {"type": "rank", "ranking": [{"key", "score", "choice"?, "yes"?}, ...] (best first; "choice" on the shortlisted
ones, "yes" the P(yes) before the prior), "top": key, "candidates": N, "requests": n, "experimental": true}. Every other
question of the request is answered as usual over the request's state.
"""
import asyncio
import json
import math
import time

from .prompt import RequestError, lang_of

MAX_CANDIDATES = 10_000  # per request, all its rank questions together (basal-serve --rank-max-candidates)
MAX_CHUNK = 1_000
TOP = 20  # shortlist of the final choice (at most the model's options per question)
CHUNK = 50  # candidates per request: the gain of a shared context levels off at about 50 (chunk 0 falls back to it)
CONTEXTS = ("auto", "screen", "shortlist")
ORDERS = ("state_first", "candidates_first")
LABEL = {"en": "Candidate", "pl": "Kandydat"}
CHOICE = {"en": "Which {label} is the best match?", "pl": "Która pozycja najlepiej pasuje?"}
MARGIN = 96  # tokens kept free for a question and its options when a whole list must fit one request (chunk 0)


def is_rank(q):
    return isinstance(q, dict) and q.get("type") == "rank"


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _int(name, q, k, default, lo, hi):
    v = q.get(k, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise RequestError(f"question {name!r}: {k} must be an integer from {lo} to {hi}")
    return v


def _weight(v):
    """A prior weight: a finite number >= 0 (an infinite one, 1e999 in JSON or an integer beyond the largest float, made
    the scores inf or NaN and the answer no JSON: HTTP 500)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v) and v >= 0
    except OverflowError:
        return False


def parse(name, q, max_options, lang, max_candidates=MAX_CANDIDATES):
    """A rank question -> its spec; RequestError (HTTP 422) for an invalid one."""
    for k in ("ask_if", "depends_on", "coverage", "max_error", "evidence"):
        if q.get(k) is not None:
            raise RequestError(f"question {name!r}: {k} is not available on a rank question")
    instr = q.get("instructions")
    if not isinstance(instr, str) or "{candidate}" not in instr:
        raise RequestError(f"question {name!r}: a rank question needs instructions that name the candidate with "
                           "{candidate}, e.g. \"Does {candidate} answer the question?\"")
    c = q.get("candidates")
    if isinstance(c, dict):
        cands = [(str(k), _text(v), None) for k, v in c.items()]
    elif isinstance(c, list):
        cands = []
        for i, x in enumerate(c):
            if isinstance(x, dict):
                if "text" not in x:
                    raise RequestError(f"question {name!r}: candidate {i} has no text")
                g = x.get("group")
                cands.append((str(x.get("key", i)), _text(x["text"]), None if g is None else _text(g)))
            else:
                cands.append((str(i), _text(x), None))
    else:
        raise RequestError(f"question {name!r}: candidates must be an object {{key: text}} or a list")
    if not cands:
        raise RequestError(f"question {name!r}: no candidates")
    if len(cands) > max_candidates:
        raise RequestError(f"question {name!r}: {len(cands)} candidates, more than the {max_candidates} this server "
                           "takes per request (basal-serve --rank-max-candidates)")
    keys = [k for k, _, _ in cands]
    if len(set(keys)) != len(keys):
        raise RequestError(f"question {name!r}: candidate keys must be unique")
    top = _int(name, q, "top", min(TOP, max_options), 0, max_options)
    if top == 1:
        raise RequestError(f"question {name!r}: top must be 0 (no choice) or 2..{max_options}")
    ctx, order = q.get("choice_context", "auto"), q.get("order", "state_first")
    if ctx not in CONTEXTS:
        raise RequestError(f"question {name!r}: choice_context must be one of {', '.join(CONTEXTS)}")
    if order not in ORDERS:
        raise RequestError(f"question {name!r}: order must be one of {', '.join(ORDERS)}")
    prior = q.get("prior") or {}
    if not isinstance(prior, dict) or not all(_weight(v) for v in prior.values()):
        raise RequestError(f"question {name!r}: prior must be an object {{key: finite weight >= 0}}")
    unknown = set(map(str, prior)) - set(keys)
    if unknown:
        raise RequestError(f"question {name!r}: prior names unknown candidates: {sorted(unknown)[:5]}")
    label = q.get("label", LABEL[lang])
    title = q.get("list_title")
    choice = q.get("choice_instructions", CHOICE[lang].format(label=str(label).lower()))
    for k, v in (("label", label), ("list_title", title), ("choice_instructions", choice)):
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise RequestError(f"question {name!r}: {k} must be a non-empty string")
    return dict(name=name, instructions=instr, cands=cands, label=label, title=title, choice=choice, top=top,
                chunk=_int(name, q, "chunk", CHUNK, 0, MAX_CHUNK), context=ctx, order=order,
                prior={str(k): float(v) for k, v in prior.items()},
                limit=_int(name, q, "limit", len(cands), 1, max_candidates))


def chunks(cands, size):
    """Candidate indices packed in order into requests of at most `size`; a group (consecutive candidates with the
    same group) starts a new request rather than being split, unless it is larger than a request itself."""
    out, cur, i = [], [], 0
    while i < len(cands):
        j = i + 1
        while j < len(cands) and cands[i][2] is not None and cands[j][2] == cands[i][2]:
            j += 1
        group = list(range(i, j))
        if cur and len(cur) + len(group) > size:
            out.append(cur)
            cur = []
        while len(group) > size:
            out.append(group[:size])
            group = group[size:]
        cur += group
        i = j
    if cur:
        out.append(cur)
    return out


def listing(cands, idx, label, title):
    """The numbered list of the candidates idx (1, 2, ... in this list) under their group lines -> text, {idx: name}."""
    lines, names, group = ([title + ":"] if title else []), {}, None
    for n, i in enumerate(idx, 1):
        g = cands[i][2]
        if g is not None and g != group:
            lines.append(f"[{g}]")
        group = g
        names[i] = f"{label} {n}"
        lines.append(f"{names[i]}: {cands[i][1]}")
    return "\n".join(lines), names


def compose(state, text, order):
    if not state:
        return text
    return f"{text}\n\n{state}" if order == "candidates_first" else f"{state}\n\n{text}"


def fits(srv, state, spec):
    """Whether the whole list fits one request of the server's context (chunk 0); True when the limit is unknown."""
    limit = getattr(srv, "context", None)
    if not limit:
        return True
    text, _ = listing(spec["cands"], range(len(spec["cands"])), spec["label"], spec["title"])
    n = len(srv.tok(compose(state, text, spec["order"]), add_special_tokens=False).input_ids)
    return n + len(srv.tok(spec["instructions"], add_special_tokens=False).input_ids) + MARGIN <= limit[0]


async def sub(srv, body):
    """A sub-request through Server.decide, counted in flight (GET /health) like a client request."""
    srv.in_flight += 1
    try:
        out = await srv.decide(body)
    finally:
        srv.in_flight -= 1
    if out["usage"].get("truncated"):
        raise RequestError("a rank request is longer than the context (--context-overflow truncate would cut its "
                           "list); lower chunk")
    return out


async def screen(srv, state, spec, fields):
    """Every candidate's P(yes) -> (scores, the requests' bodies and lists, their responses)."""
    size = spec["chunk"] or (len(spec["cands"]) if fits(srv, state, spec) else CHUNK)
    parts, bodies = [], []
    for idx in chunks(spec["cands"], size):
        text, names = listing(spec["cands"], idx, spec["label"], spec["title"])
        qs = {str(n): {"type": "noul", "instructions": spec["instructions"].replace("{candidate}", names[i])}
              for n, i in enumerate(idx)}
        bodies.append(dict(fields, state=compose(state, text, spec["order"]), questions=qs))
        parts.append((idx, names))
    outs = await asyncio.gather(*(sub(srv, b) for b in bodies))
    yes = [0.0] * len(spec["cands"])
    for (idx, _), out in zip(parts, outs):
        for n, i in enumerate(idx):
            yes[i] = float(out["answers"][str(n)]["noul"])
    return yes, bodies, parts, list(outs)


async def rank_one(srv, state, spec, fields):
    """One rank question -> (its answer, its sub-responses)."""
    yes, bodies, parts, outs = await screen(srv, state, spec, fields)
    cands, prior = spec["cands"], spec["prior"]
    score = [y * prior.get(k, 1.0) for y, (k, _, _) in zip(yes, cands)]
    order = sorted(range(len(cands)), key=lambda i: -score[i])  # stable: ties keep the candidates' order
    choice = {}
    k = min(spec["top"], len(cands))
    if k >= 2:
        short = sorted(order[:k])  # the list's order
        if spec["context"] == "screen" and len(parts) > 1:
            raise RequestError(f"question {spec['name']!r}: choice_context screen needs every candidate in one request "
                               f"(they took {len(parts)}; raise chunk or use shortlist)")
        if spec["context"] != "shortlist" and len(parts) == 1:
            body, names = bodies[0], parts[0][1]
        else:
            text, names = listing(cands, short, spec["label"], spec["title"])
            body = dict(fields, state=compose(state, text, spec["order"]))
        crit = {str(i): names[i] for i in short}
        q = {"type": "choice", "instructions": spec["choice"], "criteria": crit, "option_keys": "hide"}
        out = await sub(srv, dict(body, questions={"choice": q}))
        outs.append(out)
        probs = out["answers"]["choice"]["probabilities"]
        choice = {i: float(probs[str(i)]) for i in short}
        head = sorted(order[:k], key=lambda i: -choice[i])
        order = head + order[k:]
    ranking = []
    for i in order[: spec["limit"]]:
        e = {"key": cands[i][0], "score": score[i]}
        if i in choice:
            e["choice"] = choice[i]
        if prior:
            e["yes"] = yes[i]
        ranking.append(e)
    return {"type": "rank", "ranking": ranking, "top": cands[order[0]][0], "candidates": len(cands),
            "requests": len(outs), "experimental": True}, outs


async def decide(srv, body):
    """A request with rank questions (Server.decide sends it here): the rank questions through their own requests, the
    others as one request over the state; the usage sums them all."""
    t0 = time.perf_counter()
    if body.get("media") is not None or body.get("images") is not None:
        raise RequestError("rank questions do not take media")
    state = _text(body["state"]) if body["state"] is not None else ""
    lang = body.get("lang") or lang_of(state)
    if lang not in ("pl", "en"):
        raise RequestError('lang must be "pl" or "en"')
    cap = getattr(srv, "rank_max", MAX_CANDIDATES)
    specs = [parse(n, q, srv.max_options, lang, cap) for n, q in body["questions"].items() if is_rank(q)]
    total = sum(len(s["cands"]) for s in specs)
    if total > cap:  # every 50 candidates are one request through the queue every client shares
        raise RequestError(f"{total} rank candidates in this request, more than the {cap} this server takes per "
                           "request (basal-serve --rank-max-candidates)")
    others = {n: q for n, q in body["questions"].items() if not is_rank(q)}
    fields = {k: v for k, v in body.items() if k not in ("state", "questions")}
    jobs = [rank_one(srv, state, s, fields) for s in specs]
    if others:
        jobs.append(srv.decide(dict(body, questions=others)))
    res = await asyncio.gather(*jobs)
    answers, subs = {}, []
    if others:
        rest = res.pop()
        subs.append(rest)
    for s, (a, outs) in zip(specs, res):
        answers[s["name"]] = a
        subs += outs
    if others:
        answers.update(rest["answers"])
    answers = {n: answers[n] for n in body["questions"]}  # the request's order
    usage = {"input_tokens": sum(o["usage"]["input_tokens"] for o in subs), "output_tokens": 0,
             "questions": len(body["questions"]), "branches": sum(o["usage"]["branches"] for o in subs),
             "requests": len(subs), "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
             "longest_prompt_tokens": max(o["usage"].get("longest_prompt_tokens", 0) for o in subs)}
    return {"model": srv.name, "answers": answers, "usage": usage}
