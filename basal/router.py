"""basal-router (2.5): several basal-serve replicas (one per GPU) behind one HTTP endpoint.

A request goes to the replica its state is routed to, so the questions over one document meet the same replica's
state sharing (SOAM) and state cache: consistent hashing of the state's text over a ring of the ready replicas, with
bounded load (a replica already holding more than LOAD_FACTOR x the mean in-flight requests passes the request on
along the ring). Requests without a state go to the less loaded of two random replicas. Replicas are polled at
/health; one answering 503 (starting) or not answering gets no requests until it is ready again. With --gpus, the
router exits (status 1) when every replica it started has exited, and the replicas it started end with it: each runs
in its own process group, which the router ends when it stops (SIGTERM, SIGINT, SIGHUP, an error) and, on Linux, which
gets SIGTERM when the router is killed.

Off by default (the replicas' answers then pass through untouched):
  --escalate URL ... --escalate-below P   a cascade: questions whose top probability (calibrated, as the first replica
                                          answers) is under P are asked again of the escalation replicas (a bigger
                                          model); every answer says "answered_by": "first" or "escalated". Repeated,
                                          the pairs are ordered levels: the first replica is level 1, the k-th
                                          --escalate level k + 1, and a question goes on from level k to level k + 1
                                          only while its top probability at level k is under the k-th
                                          --escalate-below; every answer then also gives its "level" and "model" (as
                                          that replica reports it). A level that is down or does not answer 200 leaves
                                          the previous answers. Multi and task answers never escalate.
  --blend URL ... --blend-weights W0 W1 ...
                                          a blend: every question is also asked of each --blend pool (one model
                                          each; repeat the flag), and a plain choice / noul / score answer gets the
                                          weighted mean of the models' probabilities (W0 for --replicas, then one per
                                          --blend, in order): choice, noul, score and confidence follow from it, and
                                          the answer says "blend": {"models": [...], "weights": [...]} and
                                          "calibrated": false (no model's thresholds hold for it: refit them on the
                                          blend's output). A model that is down or does not answer 200 is left out
                                          (the weights of the rest are used). Every question costs a call per model.
                                          Multi, act, tool, task answers, questions with coverage / max_error /
                                          evidence keep the first replica's answer. Not with --escalate.
  --fallback URL --fallback-model NAME --fallback-below T
                                          an LLM after the last level (or the single model): a plain choice question
                                          whose top probability there is under T goes to an OpenAI-compatible chat
                                          endpoint (URL: its base, e.g. http://127.0.0.1:8100/v1), with a prompt the
                                          router builds from the document, the question and the options, and a JSON
                                          schema that admits only the options' letters. Its answer says "answered_by":
                                          "fallback", the "model", "calibrated": false, option probabilities from the
                                          endpoint's logprobs (renormalised over the options; null without them) and
                                          the basal answer it replaced. Greedy (temperature 0, seed 0), up to
                                          --fallback-max-tokens (8192) tokens. Down, --fallback-timeout (60 s), an
                                          answer that does not parse or is not an option: the basal answer stays.
                                          Multi, task,
                                          coverage / max_error / evidence questions and requests with media never go.
                                          --fallback-extra JSON is merged into the chat request (e.g. a thinking
                                          switch); BASAL_FALLBACK_API_KEY, if set, is sent as a bearer token.
  --route SCRIPT=URL[,URL]                requests whose state is mostly in that script (latin, cyrillic, greek,
                                          arabic, hebrew, cjk) go to those replicas instead

  basal-router --replicas http://127.0.0.1:8001 http://127.0.0.1:8002 --port 8000
  basal-router --gpus 0,1,2,3 --port 8000 -- --model Remek/basal-2.0-mini --mode fast   (starts one basal-serve per
                                                                                          GPU on ports 8001..)
  basal-router --replicas http://127.0.0.1:8001 --escalate http://127.0.0.1:8002 --escalate-below 0.8 \
               --escalate http://127.0.0.1:8003 --escalate-below 0.9   (pico on 8001, mini on 8002, ultra on 8003)
  basal-router --replicas http://127.0.0.1:8003 --blend http://127.0.0.1:8002 --blend http://127.0.0.1:8001 \
               --blend-weights 0.6 0.2 0.2   (ultra first: answers that are not blended are its own)
GET /health: 200 when a replica of --replicas / --gpus (the pool of ordinary requests) is ready, else 503 (the ready
replicas of every pool in its "pools"); /metrics: per replica in flight, requests routed by
state / by load, failures, readiness, answers per cascade level, blended answers, fallback calls by result
(Prometheus); /v1/models: the first ready replica's of that pool. --ui on: GET /ui, one live page of the router and every replica
(each replica's own panels when it runs with --ui on); GET /ui/state, its JSON.

Request ids: the client's X-Request-Id (or a req_<hex> the router makes when it has none or an invalid one) goes to
every replica the request reaches and comes back as the response's x-request-id, so the replicas' logs and the client
name one request alike.

API keys: a client's Authorization / X-API-Key header is passed on, as it came, to every replica its request reaches
(--replicas, --escalate, --blend, --route; never to --fallback), so replicas started with --api-keys (the same key file
on each) check it as if called directly. The router checks no key itself: its /health and /metrics stay open.
"""
import argparse
import asyncio
import atexit
import bisect
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time

from .metrics import prometheus

LOAD_FACTOR = 1.25
VNODES = 64
POLL_S = 2.0
SERVE = [sys.executable, "-m", "basal.server"]  # what --gpus starts per GPU (spawn)


def state_key(body):
    """The routing key of a request body: its state with whitespace normalised (None without a state, or for a body
    that is not JSON or nested too deeply to read: the replica answers that client error)."""
    try:
        state = json.loads(body).get("state")
    except (ValueError, AttributeError, RecursionError):
        return None
    if not isinstance(state, str) or not state.strip():
        return None
    return re.sub(r"\s+", " ", state.strip())


SCRIPTS = ("latin", "cyrillic", "greek", "arabic", "hebrew", "cjk")


def script_of(text):
    """The script most of the letters of text are in (SCRIPTS, or "other"; None without letters). A state that is not a
    string is read as its JSON text, as basal-serve shows it to the model (keys too)."""
    import unicodedata
    if text is not None and not isinstance(text, str):
        text = _text(text)
    count = {}
    for ch in text or "":
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            sc = next((x for x in SCRIPTS[:5] if name.startswith(x.upper())), None)
            if sc is None:
                sc = "cjk" if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) else "other"
            count[sc] = count.get(sc, 0) + 1
    return max(count, key=count.get) if count else None


def top_probability(answer):
    """The top probability of an answer (its option probabilities), None for answers without one (multi, tasks)."""
    p = answer.get("probabilities") if isinstance(answer, dict) else None
    if answer.get("type") == "multi" or not isinstance(p, dict) or not p:
        return None
    return max(p.values())


BLENDS = ("choice", "noul", "score")
PLAIN = ("coverage", "max_error", "evidence", "multi")  # question fields whose answers are not blended
CERTIFIED = ("abstain", "max_error", "answer_set", "coverage", "coverage_note", "calibration")  # one model's calibration


def blendable(question, answer):
    """A plain choice / noul / score answer (its option probabilities only) to a question without limits or evidence."""
    return (isinstance(answer, dict) and answer.get("type") in BLENDS and isinstance(answer.get("probabilities"), dict)
            and bool(answer["probabilities"]) and isinstance(question, dict) and not any(k in question for k in PLAIN))


def blend_answer(answer, members):
    """answer (the first replica's) with the weighted mean of members' probabilities, [(weight, model, answer)] in
    order: p[k] = sum(w * p_m[k]) / sum(w); choice / noul / score / confidence recomputed from p as basal-serve does.
    The blend is not calibrated: "calibrated": false, and no field that rests on one model's calibration (CERTIFIED)."""
    total = sum(w for w, _, _ in members)
    p = {k: sum(w * a["probabilities"][k] for w, _, a in members) / total for k in answer["probabilities"]}
    out = {k: v for k, v in answer.items() if k not in CERTIFIED}
    out.update(probabilities=p, confidence=max(p.values()))
    if answer["type"] == "choice":
        out["choice"] = max(p, key=p.get)
    elif answer["type"] == "noul":
        out["noul"] = p["true"]
    else:
        out["score"] = float(sum(i * v for i, v in enumerate(p.values())))
    out["blend"] = {"models": [m for _, m, _ in members], "weights": [w for w, _, _ in members]}
    out["calibrated"] = False
    return out


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FALLBACK_RESULTS = ("answered", "error", "timeout", "unparseable", "not_an_option")
RETRY_MIN_S = 1.0  # a fallback call is retried (once) only with at least this much of --fallback-timeout left
FALLBACK_SYSTEM = ("You answer a question about a document. Read the document, then pick the one option that answers "
                   'the question. Reply with JSON only: {"answer": "<the letter of the option>"}.')


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def fallback_options(question, answer):
    """[(key, text)] of a plain choice question in the answer's order, the texts as basal-serve shows them to the model
    (option_keys); None when the question does not go to the fallback (another type, limits, evidence, multi, keys
    that do not match the answer's, fewer than 2 or more than 26 options)."""
    if not (isinstance(answer, dict) and answer.get("type") == "choice" and isinstance(answer.get("probabilities"), dict)
            and isinstance(question, dict) and question.get("type", "choice") == "choice"
            and not any(k in question for k in PLAIN)):
        return None
    crit = question.get("criteria") or {}
    if isinstance(crit, list):
        crit = {k: None for k in crit}
    if not isinstance(crit, dict) or [str(k) for k in crit] != list(answer["probabilities"]) or not 2 <= len(crit) <= 26:
        return None
    texts = [str(k) if v is None else _text(v) for k, v in crit.items()]
    dup = {t for t in texts if texts.count(t) > 1}
    hide = question.get("option_keys") == "hide"
    return [(str(k), t if v is None or (hide and t not in dup) else f"{k}: {t}")
            for (k, v), t in zip(crit.items(), texts)]


def fallback_messages(state, question, options):
    """The chat messages for the fallback: the document, the question and the options as lettered lines."""
    q = question.get("instructions")
    lines = "\n".join(f"{LETTERS[i]}. {t}" for i, (_, t) in enumerate(options))
    user = (f"Document:\n{_text(state)}\n\nQuestion:\n{_text(q) if q else 'Which option fits the document?'}\n\n"
            f"Options:\n{lines}")
    return [{"role": "system", "content": FALLBACK_SYSTEM}, {"role": "user", "content": user}]


def fallback_request(model, messages, n, extra=None, max_tokens=8192, seed=0):
    """The chat completion request: greedy (temperature 0, a fixed seed), a token budget with room for a model's
    thinking, logprobs, and a JSON schema that admits only the n options' letters; extra merged last."""
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(LETTERS[:n])}},
              "required": ["answer"], "additionalProperties": False}
    body = {"model": model, "messages": messages, "temperature": 0, "seed": seed, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": 20,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
    body.update(extra or {})
    return body


def _letter_probs(entries, letter, n):
    """Probabilities of the n letters at the token that carries the answer's letter (the last '"answer": "' of the
    output), from its top logprobs renormalised over the letters (letters not among them: 0); None if not found."""
    toks = [(e.get("token") or "") if isinstance(e, dict) else "" for e in entries]
    text = "".join(toks)
    hits = list(re.finditer(r'"answer"\s*:\s*"', text))
    if not hits:
        return None
    v, pos = hits[-1].end(), 0
    for i, t in enumerate(toks):
        if pos <= v < pos + len(t):
            break
        pos += len(t)
    else:
        return None
    pre, chosen = text[pos:v], toks[i]
    if chosen[len(pre):len(pre) + 1] != letter or not isinstance(entries[i], dict):
        return None
    rest = chosen[len(pre) + 1:]
    lp = {letter: entries[i].get("logprob")}
    for alt in entries[i].get("top_logprobs") or []:
        t = alt.get("token") or ""
        c = t[len(pre):len(pre) + 1]
        if t.startswith(pre) and c and c in LETTERS[:n] and t[len(pre) + 1:] == rest and alt.get("logprob") is not None:
            lp[c] = max(lp.get(c) if lp.get(c) is not None else -math.inf, alt["logprob"])
    lp = {c: x for c, x in lp.items() if isinstance(x, (int, float)) and math.isfinite(x)}  # (a masked letter: -inf)
    if letter not in lp:
        return None
    top = max(lp.values())
    ex = {c: math.exp(x - top) for c, x in lp.items()}
    total = sum(ex.values())
    return [ex.get(c, 0.0) / total for c in LETTERS[:n]]


def fallback_parse(data, n):
    """A chat completion -> (option index or None, probabilities or None, result): strict, the content must be
    exactly {"answer": "<one of the n letters>"}."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        ans = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None, "unparseable"
    if not isinstance(ans, dict) or set(ans) != {"answer"} or not isinstance(ans["answer"], str):
        return None, None, "unparseable"
    if len(ans["answer"]) != 1 or ans["answer"] not in LETTERS[:n]:
        return None, None, "not_an_option"
    entries = ((choice.get("logprobs") or {}).get("content") or []) if isinstance(choice, dict) else []
    return LETTERS.index(ans["answer"]), _letter_probs(entries, ans["answer"], n) if entries else None, "answered"


KEY_HEADERS = ("authorization", "x-api-key")  # a client's API key, as basal-serve --api-keys reads it


def key_headers(raw):
    """The client's API key headers among raw ASGI headers, values as bytes, the last of each name (as basal-serve
    reads them): passed on to the replicas, never to the --fallback LLM. {} without them."""
    return {k: v.strip() for k, v in ((n.decode("latin1").lower(), v) for n, v in raw) if k in KEY_HEADERS}


def _h(text):
    # (surrogatepass: a state with a lone surrogate, "\ud800" in its JSON, still has a key; the replica refuses it)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()[:8], "big")


class Ring:
    """Consistent hashing with bounded load over replica names (VNODES points per replica)."""

    def __init__(self, names, vnodes=VNODES):
        self.points = sorted((_h(f"{n}#{v}"), n) for n in names for v in range(vnodes))
        self.keys = [p[0] for p in self.points]

    def pick(self, key, load, ready, factor=LOAD_FACTOR):
        """The replica for key: the first ready one clockwise from hash(key) whose load stays within factor x the
        mean load (counting this request); None when no replica is ready."""
        names = [n for n in dict.fromkeys(n for _, n in self.points) if n in ready]
        if not names:
            return None
        cap = math.ceil(factor * (sum(load.get(n, 0) for n in names) + 1) / len(names))  # (at least 1)
        i = bisect.bisect(self.keys, _h(key)) % len(self.points)
        seen = set()
        for j in range(len(self.points)):
            n = self.points[(i + j) % len(self.points)][1]
            if n in ready and n not in seen:
                seen.add(n)
                if load.get(n, 0) + 1 <= cap:
                    return n
        return min(names, key=lambda n: load.get(n, 0))


class Router:
    def __init__(self, replicas, transport=None, factor=LOAD_FACTOR, seed=0, escalate=(), below=0.0, routes=None,
                 levels=None, blend=None, blend_weight=1.0, fallback=None):
        """escalate, below: one cascade level; levels: [(URLs, threshold), ...] for levels 2, 3, ... instead;
        blend: [(URLs, weight), ...] asked beside the replicas (weight blend_weight); fallback: {url, model, below,
        timeout, extra} of the LLM after the last level."""
        import httpx
        levels = [(list(u), b) for u, b in (levels if levels is not None else [(escalate, below)]) if u]
        self.levels = [(f"level:{k}", b) for k, (_, b) in enumerate(levels, start=2)]  # (level k's pool, k - 1's threshold)
        blend = [(list(u), w) for u, w in (blend or []) if u]
        self.blends = [(f"blend:{k}", w) for k, (_, w) in enumerate(blend, start=2)]
        self.blend_weight = blend_weight
        self.blended = {}
        self.fallback = fallback
        self.fallback_counts = {k: 0 for k in FALLBACK_RESULTS}
        self.pools = {"main": list(replicas), **{f"level:{k}": u for k, (u, _) in enumerate(levels, start=2)},
                      **{f"blend:{k}": u for k, (u, _) in enumerate(blend, start=2)},
                      **{f"script:{k}": list(v) for k, v in (routes or {}).items()}}
        self.replicas = list(dict.fromkeys(u for urls in self.pools.values() for u in urls))
        self.rings = {name: Ring(urls) for name, urls in self.pools.items() if urls}
        self.ring = self.rings.get("main")
        self.answered = {"first": 0, "escalated": 0}
        self.by_level = {k: 0 for k in range(1, len(self.levels) + 2)}
        self.scripts = {}
        self.ready = set()
        self.load = {r: 0 for r in self.replicas}
        self.routed = {(r, how): 0 for r in self.replicas for how in ("state", "load")}
        self.failed = {r: 0 for r in self.replicas}
        self.factor = factor
        self.rng = random.Random(seed)
        self.dead = False
        self.client = httpx.AsyncClient(timeout=None, transport=transport,
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))

    def choose(self, key, pool="main"):
        """(replica, how) for a request with routing key key (None: by load), among the replicas of pool."""
        if key is not None:
            r = self.rings[pool].pick(key, self.load, self.ready, self.factor)
            return r, "state"
        ready = sorted(self.ready & set(self.pools[pool]))
        if not ready:
            return None, "load"
        a, b = self.rng.choice(ready), self.rng.choice(ready)
        return (a if self.load[a] <= self.load[b] else b), "load"

    async def poll(self):
        """Readiness of every replica (GET /health: 200 = ready)."""
        async def one(r):
            try:
                ok = (await self.client.get(r + "/health", timeout=5)).status_code == 200
            except Exception:  # noqa: BLE001  (down or still binding)
                ok = False
            (self.ready.add if ok else self.ready.discard)(r)
        await asyncio.gather(*(one(r) for r in self.replicas))

    async def forward(self, path, body, headers):
        """-> (status, content, content type): the request to the replica its pool and state give, then (--escalate)
        its low-confidence questions to the escalation replicas."""
        pool = "main"
        if any(name.startswith("script:") for name in self.rings):  # --route
            try:
                sc = script_of(json.loads(body).get("state"))
            except (ValueError, AttributeError, TypeError):  # (not JSON, not an object: the replica refuses it)
                sc = None
            if sc is not None:
                self.scripts[sc] = self.scripts.get(sc, 0) + 1
                pool = f"script:{sc}" if f"script:{sc}" in self.rings else pool
        status, content, ctype = await self._send(pool, path, body, headers)
        level = {}
        if self.levels and status == 200:
            content = await self._cascade(path, body, content, headers, level)
        if self.blends and status == 200:
            content = await self._blend(path, body, content, headers)
        if self.fallback and status == 200:
            content = await self._fallback(body, content, level if self.levels else None)
        return status, content, ctype

    async def _fallback(self, body, content, level=None):
        """The answers of the last level (level: the cascade's level of each question; None: one model) whose top
        probability is under --fallback-below asked of the LLM, concurrently; its answer replaces the basal one."""
        try:
            req, resp = json.loads(body), json.loads(content)
            answers = resp["answers"]
            qs = req.get("questions") or {}
            if req.get("media") is not None or req.get("images") is not None:  # (images: the alias of media)
                return content
            go = []
            for q, a in answers.items():
                opts = fallback_options(qs.get(q), a) if q in qs else None
                if opts and (level is None or level.get(q) == len(self.levels) + 1) \
                        and top_probability(a) < self.fallback["below"]:
                    go.append((q, a, opts))
        except (ValueError, KeyError, TypeError, AttributeError):
            return content
        if not go:
            return content
        res = await asyncio.gather(*(self._ask_fallback(req.get("state"), qs[q], opts) for q, _, opts in go))
        answered = []
        for (q, a, opts), (i, probs, result, model) in zip(go, res):
            self.fallback_counts[result] += 1
            if i is None:
                continue
            keys = [k for k, _ in opts]
            p = None if probs is None else dict(zip(keys, probs))
            answers[q] = {"type": "choice", "choice": keys[i], "probabilities": p,
                          "confidence": None if p is None else max(p.values()), "answered_by": "fallback",
                          "model": model, "calibrated": False,
                          "basal": {"choice": a.get("choice"), "confidence": top_probability(a),
                                    "model": a.get("model", resp.get("model"))}}
            answered.append(q)
        resp["fallback"] = {"below": self.fallback["below"], "asked": [q for q, _, _ in go], "answered": answered}
        return json.dumps(resp, ensure_ascii=False).encode()

    async def _ask_fallback(self, state, question, options):
        """-> (option index or None, probabilities or None, result, model) of one fallback call."""
        import httpx
        f = self.fallback
        body = fallback_request(f["model"], fallback_messages(state, question, options), len(options), f.get("extra"),
                                f.get("max_tokens", 8192))
        headers = {"content-type": "application/json"}
        if os.environ.get("BASAL_FALLBACK_API_KEY"):
            headers["authorization"] = "Bearer " + os.environ["BASAL_FALLBACK_API_KEY"]
        deadline = time.monotonic() + f.get("timeout", 60.0)  # (both attempts together)
        for attempt in (0, 1):
            left = deadline - time.monotonic()
            try:
                r = await asyncio.wait_for(self.client.post(
                    f["url"].rstrip("/") + "/chat/completions", content=json.dumps(body).encode(), headers=headers,
                    timeout=left), left)
                break
            except (httpx.TimeoutException, asyncio.TimeoutError):
                return None, None, "timeout", None
            except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError):
                # once more, in the time left (at least RETRY_MIN_S): a pooled connection the endpoint closed as it
                # was reused, at its keep-alive
                if attempt or deadline - time.monotonic() < RETRY_MIN_S:
                    return None, None, "error", None
            except Exception:  # noqa: BLE001  (down, refused)
                return None, None, "error", None
        if r.status_code != 200:
            return None, None, "error", None
        try:
            data = r.json()
        except ValueError:
            return None, None, "unparseable", None
        i, probs, result = fallback_parse(data, len(options))
        return i, probs, result, (data.get("model") if isinstance(data, dict) else None) or f["model"]

    async def _blend(self, path, body, content, headers):
        """The first replica's answers, the blendable ones mixed with the --blend pools' (asked at once, concurrently)."""
        try:
            req, resp = json.loads(body), json.loads(content)
            answers = resp["answers"]
            qs = req.get("questions") or {}
            want = [q for q, a in answers.items() if q in qs and blendable(qs[q], a)]
        except (ValueError, KeyError, TypeError, AttributeError):
            return content
        if not want:
            return content
        sub = json.dumps(dict(req, questions={q: qs[q] for q in want})).encode()
        outs = await asyncio.gather(*(self._send(pool, path, sub, headers) for pool, _ in self.blends))
        others = []
        for (_, w), (st, c, _) in zip(self.blends, outs):
            try:
                r = json.loads(c) if st == 200 else None
                if isinstance(r, dict) and isinstance(r.get("answers"), dict):
                    others.append((w, r.get("model"), r["answers"]))
            except ValueError:
                continue
        for q in want:
            a = answers[q]
            keys = list(a["probabilities"])
            members = [(self.blend_weight, resp.get("model"), a)] + [
                (w, m, x[q]) for w, m, x in others if isinstance(x.get(q), dict) and x[q].get("type") == a["type"]
                and isinstance(x[q].get("probabilities"), dict) and list(x[q]["probabilities"]) == keys]
            members = [m for m in members if m[0] > 0]  # (a weight of 0: the model takes no part, the first one too)
            if len(members) > 1:
                answers[q] = blend_answer(members[0][2], members)
            elif members:  # only one model answered (or has a weight): its answer as it is
                answers[q] = dict(members[0][2], blend={"models": [members[0][1]], "weights": [members[0][0]]})
            self.blended[len(members)] = self.blended.get(len(members), 0) + 1
        return json.dumps(resp, ensure_ascii=False).encode()

    async def _cascade(self, path, body, content, headers, level_out=None):
        """The answers of the first replica (level 1), with those under level k's threshold asked again of level k + 1,
        level by level. A level that is down or does not answer 200 leaves the previous answers (and ends the chain).
        With one level the response is as before; with more, every answer also gives its level and model."""
        try:
            req, resp = json.loads(body), json.loads(content)
            answers = resp["answers"]
        except (ValueError, KeyError, TypeError):
            return content
        named = len(self.levels) > 1
        level = {} if level_out is None else level_out
        for q, a in answers.items():
            if isinstance(a, dict):
                a["answered_by"] = "first"
                if named:
                    a["level"], a["model"] = 1, resp.get("model")
                level[q] = 1
        done = []
        for k, (pool, below) in enumerate(self.levels, start=1):
            low = [q for q, a in answers.items() if q in (req.get("questions") or {}) and level.get(q) == k
                   and (top_probability(a) if isinstance(a, dict) else None) is not None and top_probability(a) < below]
            if not low:
                break
            sub = dict(req, questions={q: req["questions"][q] for q in low})
            st, c2, _ = await self._send(pool, path, json.dumps(sub).encode(), headers)
            if st != 200:
                break
            r2 = json.loads(c2)
            for q, a in (r2.get("answers") or {}).items():
                if q in low and isinstance(a, dict):
                    a["answered_by"] = "escalated"
                    if named:
                        a["level"], a["model"] = k + 1, r2.get("model")
                    answers[q] = a
                    level[q] = k + 1
                    if k == 1:
                        done.append(q)
        self.answered["escalated"] += len(done)
        self.answered["first"] += sum(1 for a in answers.values() if isinstance(a, dict)) - len(done)
        for k in level.values():
            self.by_level[k] += 1
        below = [b for _, b in self.levels] if named else self.levels[0][1]
        resp["cascade"] = {"below": below, "escalated": done}
        return json.dumps(resp, ensure_ascii=False).encode()

    async def _send(self, pool, path, body, headers):
        r, how = self.choose(state_key(body), pool)
        if r is None:
            return 503, b'{"error": "no basal replica is ready"}', "application/json"
        self.load[r] += 1
        self.routed[(r, how)] += 1
        try:
            resp = await self.client.post(r + path, content=body, headers=headers)
            if resp.status_code >= 500:
                self.failed[r] += 1
            return resp.status_code, resp.content, resp.headers.get("content-type", "application/json")
        except Exception as e:  # noqa: BLE001
            self.failed[r] += 1
            self.ready.discard(r)
            return 502, json.dumps({"error": f"replica {r}: {type(e).__name__}"}).encode(), "application/json"
        finally:
            self.load[r] -= 1

    def metrics(self):
        return prometheus([
            ("basal_router_replica_ready", "1 when the replica answers /health with 200", "gauge",
             [({"replica": r}, int(r in self.ready)) for r in self.replicas]),
            ("basal_router_in_flight", "requests the router has in flight at the replica", "gauge",
             [({"replica": r}, self.load[r]) for r in self.replicas]),
            ("basal_router_routed_total", "requests routed, by the state's hash or by load", "counter",
             [({"replica": r, "by": how}, n) for (r, how), n in self.routed.items()]),
            ("basal_router_failed_total", "requests the replica answered with 5xx or not at all", "counter",
             [({"replica": r}, n) for r, n in self.failed.items()])]
            + ([("basal_router_answers_total", "answers by the first replica / escalated (--escalate)", "counter",
                 [({"answered_by": k}, n) for k, n in self.answered.items()])] if self.levels else [])
            + ([("basal_router_level_answers_total", "answers by cascade level (1: the first replica)", "counter",
                 [({"level": str(k)}, n) for k, n in self.by_level.items()])] if len(self.levels) > 1 else [])
            + ([("basal_router_fallback_total", "questions sent to the --fallback LLM, by result (answered: its answer "
                 "replaced the basal one)", "counter", [({"result": k}, n) for k, n in self.fallback_counts.items()])]
               if self.fallback else [])
            + ([("basal_router_blend_answers_total", "answers blended (--blend), by the number of models in the blend",
                 "counter", [({"models": str(k)}, n) for k, n in sorted(self.blended.items())])] if self.blends else [])
            + ([("basal_router_script_total", "requests by the script of their state (--route)", "counter",
                 [({"script": k}, n) for k, n in sorted(self.scripts.items())])] if self.scripts else []))


def app(router, poll_s=None, procs=(), on_dead=lambda: signal.raise_signal(signal.SIGTERM), on_stop=None, ui=None):
    """The router's app. procs: the replicas it started (--gpus); on_stop: called at shutdown (main: Replicas.stop);
    ui: the basal.router_ui.RouterUI of --ui on (its decision routes recorded, GET /ui/state, and the page at /ui)."""
    from contextlib import asynccontextmanager
    poll_s = POLL_S if poll_s is None else poll_s

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route

    async def post(request):
        headers = {"content-type": "application/json", **key_headers(request.headers.raw)}
        if request.scope.get("basal.request_id"):  # (request_ids below: the client's, or one it made)
            headers["x-request-id"] = request.scope["basal.request_id"]
        status, content, ctype = await router.forward(request.url.path, await request.body(), headers)
        return Response(content, status_code=status, media_type=ctype)

    def main_ready():
        """The ready replicas of the pool of ordinary requests: an escalation or blend pool alone serves none."""
        return sorted(r for r in router.pools["main"] if r in router.ready)

    async def health(_):
        up = bool(main_ready())
        body = {"status": "ok" if up else "starting", "ready": len(router.ready),
                "replicas": len(router.replicas), "in_flight": sum(router.load.values()),
                "pools": {name: sum(r in router.ready for r in urls) for name, urls in router.pools.items()}}
        return JSONResponse(body, status_code=200 if up else 503)

    async def models(request):
        for r in main_ready():
            try:
                resp = await router.client.get(r + "/v1/models", timeout=10, headers=key_headers(request.headers.raw))
                return Response(resp.content, status_code=resp.status_code, media_type="application/json")
            except Exception:  # noqa: BLE001
                continue
        return JSONResponse({"error": "no basal replica is ready"}, status_code=503)

    async def metrics(_):
        return PlainTextResponse(router.metrics(), media_type="text/plain; version=0.0.4")

    @asynccontextmanager
    async def lifespan(_app):
        async def loop():
            while True:
                await router.poll()
                if procs and all(p.poll() is not None for p in procs):
                    router.dead = True
                    print("basal-router: every replica exited", file=sys.stderr, flush=True)
                    on_dead()  # SIGTERM: uvicorn shuts down, main exits with status 1
                    return
                await asyncio.sleep(poll_s)
        task = asyncio.get_running_loop().create_task(loop())
        yield
        task.cancel()
        await router.client.aclose()
        if on_stop is not None:  # (uvicorn's shutdown: before it re-raises the signal that stopped it)
            await asyncio.get_running_loop().run_in_executor(None, on_stop)

    from starlette.middleware import Middleware

    from .auth import request_ids
    ids = [Middleware(request_ids)]  # x-request-id on every response, passed on to the replicas
    routes = [Route("/v1/systemone", post, methods=["POST"]), Route("/v1/basal", post, methods=["POST"]),
              Route("/health", health), Route("/v1/models", models), Route("/metrics", metrics)]
    if ui is None:
        return Starlette(routes=routes, lifespan=lifespan, middleware=ids)
    return ui.page(Starlette(routes=ui.routes(routes), lifespan=lifespan, middleware=ids))


class Replicas:
    """The basal-serve processes of --gpus. stop() ends each one's process group (SIGTERM, then SIGKILL after
    `timeout` seconds), once, from whichever exit path of the router comes first: the app's shutdown, main's signal
    handler (uvicorn re-raises the signal it caught after its own shutdown, which would end the process before main's
    finally), main's finally, atexit."""

    def __init__(self, procs=()):
        self.procs, self.stopped = list(procs), False

    def stop(self, timeout=30):
        if self.stopped:
            return
        self.stopped = True
        for p in self.procs:
            _end_group(p)
        end = time.monotonic() + timeout
        for p in self.procs:
            try:
                p.wait(timeout=max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for p in self.procs:  # a replica past the timeout, or a process a replica left in its group
            _end_group(p, hard=True)
            p.wait()


def _end_group(p, hard=False):
    """SIGTERM (hard: SIGKILL) to a replica's process group; without process groups (Windows) to the replica."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(p.pid, signal.SIGKILL if hard else signal.SIGTERM)
        elif p.poll() is None:
            p.kill() if hard else p.terminate()
    except (ProcessLookupError, PermissionError):
        pass


def _death_signal():
    """preexec_fn of a replica on Linux: SIGTERM when the router dies, however it dies (prctl PR_SET_PDEATHSIG);
    None elsewhere. libc is loaded here, before the fork."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()

    def child():
        prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:  # the router already died
            os._exit(1)
    return child


def spawn(gpus, port0, serve_args):
    """One basal-serve per GPU (CUDA_VISIBLE_DEVICES), on ports port0, port0 + 1, ... -> (processes, URLs). Each in
    its own process group (Replicas.stop ends the replica and whatever it started) that, on Linux, gets SIGTERM when
    the router dies. The replicas listen on 127.0.0.1 only (the router reaches them there; basal-serve's own default,
    0.0.0.0, would let the network bypass the router) unless the arguments after -- give --host."""
    procs, urls, pre = [], [], _death_signal()
    host = [] if any(x == "--host" or x.startswith("--host=") for x in serve_args) else ["--host", "127.0.0.1"]
    for i, g in enumerate(gpus):
        port = port0 + i
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([*SERVE, "--port", str(port), *host, *serve_args], env=env,
                                      start_new_session=hasattr(os, "killpg"), preexec_fn=pre))
        urls.append(f"http://127.0.0.1:{port}")
    return procs, urls


def levels(escalate, below):
    """The cascade levels of --escalate URL ... (repeated: levels 2, 3, ...) and --escalate-below P (one per
    --escalate, in order; none: 0.8 for each) -> [(URLs, threshold), ...]; [] without escalation replicas."""
    if not any(escalate):
        return []
    if not all(escalate):
        raise SystemExit("basal-router: --escalate without URLs among several --escalate levels")
    if below and len(below) != len(escalate):
        raise SystemExit(f"basal-router: {len(escalate)} --escalate levels and {len(below)} --escalate-below "
                         "thresholds (give one per level, in the same order, or none for 0.8)")
    return [(list(u), b) for u, b in zip(escalate, below or [0.8] * len(escalate))]


def blends(blend, weights, cascade):
    """The blend of --blend URL ... (repeated: one model each) and --blend-weights W0 W1 ... (W0 for --replicas, then
    one per --blend) -> (first replica's weight, [(URLs, weight), ...]); (1.0, []) without --blend."""
    if not blend:
        if weights is not None:
            raise SystemExit("basal-router: --blend-weights without --blend")
        return 1.0, []
    if cascade:
        raise SystemExit("basal-router: --blend and --escalate do not go together")
    if not all(blend):
        raise SystemExit("basal-router: --blend without URLs")
    if weights is None or len(weights) != 1 + len(blend):
        raise SystemExit(f"basal-router: --blend-weights needs {1 + len(blend)} weights (--replicas first, then one "
                         "per --blend, in order)")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise SystemExit("basal-router: --blend-weights must be finite, >= 0 and not all 0")
    return weights[0], [(list(u), w) for u, w in zip(blend, weights[1:])]


def fallback_config(a, blend):
    """--fallback URL --fallback-model NAME [--fallback-below T --fallback-timeout S --fallback-max-tokens N
    --fallback-extra JSON] -> the Router's fallback dict; None without --fallback."""
    given = [f for f in ("fallback_model", "fallback_below", "fallback_timeout", "fallback_max_tokens", "fallback_extra")
             if getattr(a, f) is not None]
    if not a.fallback:
        if given:
            raise SystemExit("basal-router: --fallback-model / -below / -timeout / -max-tokens / -extra need --fallback URL")
        return None
    if blend:
        raise SystemExit("basal-router: --fallback and --blend do not go together")
    if not a.fallback_model:
        raise SystemExit("basal-router: --fallback needs --fallback-model NAME (the model the endpoint serves)")
    below = 0.5 if a.fallback_below is None else a.fallback_below
    timeout = 60.0 if a.fallback_timeout is None else a.fallback_timeout
    max_tokens = 8192 if a.fallback_max_tokens is None else a.fallback_max_tokens
    if not 0 < below <= 1 or not timeout > 0 or max_tokens < 1:
        raise SystemExit("basal-router: --fallback-below must be in (0, 1], --fallback-timeout > 0, "
                         "--fallback-max-tokens >= 1")
    try:
        extra = json.loads(a.fallback_extra) if a.fallback_extra else {}
    except ValueError:
        extra = None
    if not isinstance(extra, dict):
        raise SystemExit("basal-router: --fallback-extra must be a JSON object")
    return {"url": a.fallback, "model": a.fallback_model, "below": below, "timeout": timeout, "max_tokens": max_tokens,
            "extra": extra}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    serve_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", nargs="*", default=[], help="URLs of running basal-serve replicas")
    ap.add_argument("--gpus", default=None, help="start one basal-serve per GPU, e.g. 0,1,2,3 (arguments after --)")
    ap.add_argument("--replica-port", dest="replica_port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--load-factor", dest="load_factor", type=float, default=LOAD_FACTOR)
    ap.add_argument("--escalate", nargs="*", action="append", default=[],
                    help="escalation replicas (a bigger model) for the cascade; repeat for further levels")
    ap.add_argument("--escalate-below", dest="escalate_below", type=float, action="append", default=[],
                    help="--escalate: questions whose top probability is under this are asked again there (default "
                         "0.8); one per --escalate, in the same order")
    ap.add_argument("--route", action="append", default=[], help="SCRIPT=URL[,URL]: states in that script go there")
    ap.add_argument("--blend", nargs="+", action="append", default=[],
                    help="replicas of one more model whose probabilities are blended in; repeat for each model")
    ap.add_argument("--blend-weights", dest="blend_weights", nargs="+", type=float, default=None,
                    help="--blend: the weight of --replicas, then of each --blend, in order")
    ap.add_argument("--fallback", default=None, help="base URL of an OpenAI-compatible chat endpoint (an LLM) for "
                    "questions still unsure after the last level, e.g. http://127.0.0.1:8100/v1")
    ap.add_argument("--fallback-model", dest="fallback_model", default=None, help="--fallback: the model it serves")
    ap.add_argument("--fallback-below", dest="fallback_below", type=float, default=None,
                    help="--fallback: questions whose top probability is under this go (default 0.5)")
    ap.add_argument("--fallback-timeout", dest="fallback_timeout", type=float, default=None,
                    help="--fallback: seconds per call (default 60); then the basal answer stays")
    ap.add_argument("--fallback-max-tokens", dest="fallback_max_tokens", type=int, default=None,
                    help="--fallback: the token budget of a call, thinking included (default 8192; the endpoint's "
                         "context must hold the prompt plus this)")
    ap.add_argument("--fallback-extra", dest="fallback_extra", default=None,
                    help='--fallback: JSON merged into the chat request, e.g. a thinking switch: '
                         '\'{"chat_template_kwargs": {"enable_thinking": true}}\'')
    ap.add_argument("--ui", choices=["off", "on"], default="off",
                    help="also serve one live page of the router and every replica at /ui: replicas, readiness, "
                         "routing, requests in flight, failures, traffic through the router since start and its recent "
                         "requests (metadata, never text), and each replica's own status, traffic, state cache and GPU "
                         "telemetry from its /ui/state when the replica runs with --ui on (after -- with --gpus); "
                         "each request adds one append to an in-memory ring. Open like /metrics; the replicas' parts "
                         "need a key when they run with --api-keys (the page asks for it)")
    a = ap.parse_args(argv)
    cascade = levels(a.escalate, a.escalate_below)
    first, blend = blends(a.blend, a.blend_weights, cascade)
    fallback = fallback_config(a, blend)
    procs, urls = ([], list(a.replicas))
    if a.gpus:
        procs, more = spawn(a.gpus.split(","), a.replica_port, serve_args)
        urls += more
    replicas = Replicas(procs)
    atexit.register(replicas.stop)  # (from here: a SystemExit below ends the replicas too)
    if not urls:
        raise SystemExit("basal-router: no replicas (--replicas URL ... or --gpus 0,1 -- <basal-serve arguments>)")
    import uvicorn
    print(f"basal-router: {len(urls)} replicas, port {a.port}", flush=True)
    routes = {}
    for x in a.route:
        k, _, v = x.partition("=")
        if k not in SCRIPTS + ("other",) or not v:
            raise SystemExit(f"basal-router: --route {x!r}: SCRIPT=URL[,URL], SCRIPT one of {', '.join(SCRIPTS)}, other")
        routes[k] = v.split(",")
    router = Router(urls, factor=a.load_factor, levels=cascade, routes=routes, blend=blend, blend_weight=first,
                    fallback=fallback)
    ui = None
    if a.ui == "on":  # basal/router_ui.py: the page at /ui, its state at /ui/state
        from .router_ui import RouterUI
        ui = RouterUI(router, a, vars(ap.parse_args([])))
        print(f"basal-router: web UI at http://{a.host}:{a.port}/ui (open to whoever reaches this port, as /metrics "
              "is; replicas started with --api-keys need the key on the page)", flush=True)

    def ended(sig, _frame):
        """SIGTERM / SIGINT / SIGHUP outside uvicorn, or re-raised by it after its shutdown: the replicas end first."""
        replicas.stop()
        if router.dead:
            raise SystemExit(1)
        signal.signal(sig, signal.SIG_DFL)
        signal.raise_signal(sig)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), ended)
    try:
        uvicorn.run(app(router, procs=procs, on_stop=replicas.stop, ui=ui), host=a.host, port=a.port,
                    log_level="warning")
    finally:
        replicas.stop()
    if router.dead:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
