"""Python client of basal-serve: every question and task type, blocking (Basal) or asyncio (AsyncBasal).

  from basal.client import Basal, choice, yes_no
  b = Basal("http://127.0.0.1:8000")
  b.choice(state, "Which department should handle this?", {"billing": "Billing", "tech": "Technical support"})
  b.decide(state, {"dept": choice("Which department?", {"billing": "Billing", "tech": "Technical support"}),
                   "urgent": yes_no("Is it urgent?")})      # several questions over one state: one request (SOAM)

  async with AsyncBasal("http://127.0.0.1:8000") as b:
      a = await b.yes_no(state, "Is it urgent?")

A method returns the answer of its question (API.md, "Response"); decide returns the answers by name, request the whole
response (with "usage"). Keyword options of a method go into the question ("option_keys", "max_error", "rank", ...);
request fields ("facts", "lang", "think", "media", ...) are client defaults or keywords of decide.

Errors follow the server's status codes, all a BasalError (a RuntimeError) with .status: InvalidRequest (422, change the
request), OutOfMemory (503, the GPU ran out of memory; retried `retries` times, waiting backoff, 2 x backoff, ...) and
ServerError (500 and any other status).

It also talks to other servers of the same /v1/systemone request shape, such as vLLM's structured-diffusion example
(examples/features/structured_diffusion): `api_key` sends "Authorization: Bearer <key>", choice criteria given as a
list of keys are sent as {key: null} (what basal-serve reads a list as), and a noul answer that carries only "noul"
gets "probabilities" and "confidence" from it, as basal-serve computes them. The confidences of another server are
not calibrated by basal: basal's certified thresholds (max_error) do not apply to them.
"""
import asyncio
import time

import httpx


def complete(answer):
    """A noul answer with only "noul" (p of true) -> also "probabilities" {"true", "false"} and "confidence"; every
    other answer (basal-serve's, or null for a question another server skipped) as it is."""
    if isinstance(answer, dict) and answer.get("type") == "noul" and "probabilities" not in answer:
        p = float(answer["noul"])
        answer = dict(answer, probabilities={"true": p, "false": 1.0 - p}, confidence=max(p, 1.0 - p))
    return answer


class BasalError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class InvalidRequest(BasalError):
    """422: the request is invalid and must be changed."""


class OutOfMemory(BasalError):
    """503: the GPU ran out of memory; the same request may succeed later or on another server."""


class ServerError(BasalError):
    """500 or any other status."""


def _error(r):
    try:
        message = r.json().get("error", r.text)
    except (ValueError, AttributeError):
        message = r.text
    cls = {422: InvalidRequest, 503: OutOfMemory}.get(r.status_code, ServerError)
    return cls(r.status_code, message)


# Questions, for decide with several of them; the keyword options are the question's own fields (API.md, TASKS.md).

def choice(question, criteria, **opts):
    """criteria: {key: description} or a list of keys."""
    if isinstance(criteria, list) and all(isinstance(k, str) for k in criteria):
        criteria = dict.fromkeys(criteria)  # basal-serve reads a list of keys so; other servers need the map
    return {"type": "choice", "instructions": question, "criteria": criteria, **opts}


def yes_no(question, true=None, false=None, **opts):
    """true / false: optional descriptions of the two sides."""
    crit = {k: v for k, v in (("true", true), ("false", false)) if v}
    return {"type": "noul", "instructions": question, **({"criteria": crit} if crit else {}), **opts}


def score(question, levels, **opts):
    """levels: lowest first."""
    return {"type": "score", "instructions": question, "criteria": levels, **opts}


def multi(question, labels, **opts):
    """labels: {key: description} or a list; opts: threshold, min, max, objective, exclusive, fallback, ..."""
    return {"type": "multi", "instructions": question, "criteria": labels, **opts}


def act(question, outcomes, costs, **opts):
    """outcomes: as choice's criteria ({"true": ..., "false": ...} for a yes/no question); costs: {action: {outcome:
    cost}} or {"wrong": c, "defer": c}; opts: max_error, defer_action."""
    return {"type": "act", "instructions": question, "criteria": outcomes, "costs": costs, **opts}


def tool(tools, question=None, **opts):
    """tools: OpenAI, MCP or Anthropic tool definitions; opts: extra, rank, max_error, multi, description_chars."""
    return {"type": "tool", "tools": tools, **({"instructions": question} if question else {}), **opts}


def rank(question, candidates, **opts):
    """question: names the candidate with {candidate}; candidates: {key: text}, [text] or [{"key", "text", "group"}],
    up to the server's --rank-max-candidates (10,000) per request; opts: top, chunk, label, list_title, choice_instructions, choice_context, order, prior, limit
    (basal/rank.py, experimental). A large ranking takes many engine calls: give the client a timeout to match."""
    return {"type": "rank", "instructions": question, "candidates": candidates, **opts}


def classify(labels, **opts):
    """labels: {name: rubric}, a list of names or of {"name", "rubric"}; opts: none, multi_label, ..."""
    return {"type": "classify", "labels": labels, **opts}


def verify(claims, **opts):
    """claims: {key: statement} or a list; opts: binary."""
    return {"type": "verify", "claims": claims, **opts}


def compress(question=None, **opts):
    """opts: unit, keep_ratio, threshold."""
    return {"type": "compress", **({"instructions": question} if question else {}), **opts}


def extract(schema, **opts):
    """schema: a JSON schema of the fields to copy from the state."""
    return {"type": "extract", "schema": schema, **opts}


class _Client:
    """What Basal and AsyncBasal share: the request body and one method per question type (in AsyncBasal each
    returns a coroutine)."""

    def __init__(self, url, early_exit, facts, lang, retries, backoff):
        self.base = url.rstrip("/")
        self.url = self.base + "/v1/systemone"
        self.defaults = {k: v for k, v in (("early_exit", early_exit), ("facts", facts), ("lang", lang)) if v}
        if not isinstance(retries, int) or retries < 0:
            raise ValueError(f"retries must be an integer >= 0, not {retries!r}")
        self.retries, self.backoff = retries, backoff

    def _body(self, state, questions, fields):
        return {"state": state, "questions": questions, **self.defaults,
                **{k: v for k, v in fields.items() if v is not None}}

    def choice(self, state, question, criteria, **opts):
        return self._one(state, choice(question, criteria, **opts))

    def yes_no(self, state, question, true=None, false=None, **opts):
        return self._one(state, yes_no(question, true, false, **opts))

    def score(self, state, question, levels, **opts):
        return self._one(state, score(question, levels, **opts))

    def multi(self, state, question, labels, **opts):
        return self._one(state, multi(question, labels, **opts))

    def act(self, state, question, outcomes, costs, **opts):
        return self._one(state, act(question, outcomes, costs, **opts))

    def tool(self, state, tools, question=None, **opts):
        return self._one(state, tool(tools, question, **opts))

    def rank(self, state, question, candidates, **opts):
        return self._one(state, rank(question, candidates, **opts))

    def classify(self, state, labels, **opts):
        return self._one(state, classify(labels, **opts))

    def verify(self, state, claims, **opts):
        return self._one(state, verify(claims, **opts))

    def compress(self, state, question=None, **opts):
        return self._one(state, compress(question, **opts))

    def extract(self, state, schema, **opts):
        return self._one(state, extract(schema, **opts))


class Basal(_Client):
    """A blocking client. http: an httpx.Client of your own (headers, transport, proxies); timeout and transport apply
    otherwise. api_key: sent as "Authorization: Bearer <key>"."""

    def __init__(self, url="http://127.0.0.1:8000", timeout=60, early_exit=None, facts=None, lang=None, retries=0,
                 backoff=0.5, http=None, transport=None, api_key=None):
        super().__init__(url, early_exit, facts, lang, retries, backoff)
        self.http = http or httpx.Client(timeout=timeout, transport=transport)
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    def _send(self, method, url, **kw):
        for attempt in range(self.retries + 1):
            r = self.http.request(method, url, **kw)
            if r.status_code == 200:
                return r.json()
            e = _error(r)
            if not isinstance(e, OutOfMemory) or attempt == self.retries:
                raise e
            time.sleep(self.backoff * 2 ** attempt)

    def request(self, body):
        """The whole response to a request body (answers, model, usage)."""
        return self._send("POST", self.url, json=body)

    def decide(self, state, questions, **fields):
        """{name: answer} for {name: question}; fields: request fields (facts, lang, think, media, ...)."""
        return {k: complete(a) for k, a in self.request(self._body(state, questions, fields))["answers"].items()}

    def _one(self, state, q):
        return self.decide(state, {"q": q})["q"]

    def preset(self, state, name, lang=None, max_error=None):
        """The answers to a ready question set (basal/presets.py: triage, moderation, compliance, guard_input,
        guard_output, route_intent) about state; max_error adds the abstain flag to every answer. The wording is in
        `lang`, else the client's `lang`, else the language of the state."""
        from .presets import questions
        return self.decide(state, questions(name, lang or self.defaults.get("lang"), state, max_error))

    def guard(self, state, kind="input", policy="strict", lang=None, max_error=None):
        """Run the guard_input / guard_output battery on state -> presets.guard(): {"action": pass | review | block |
        support, "reason", "checks"} plus "answers". max_error takes the thresholds from the model's calibration."""
        from .presets import guard
        answers = self.preset(state, "guard_" + kind, lang, max_error)
        return {**guard(answers, policy, max_error, expect=kind), "answers": answers}

    def route(self, state, lang=None, min_confidence=0.6, max_error=None):
        """Run route_intent on state -> presets.route(): the intent, or "human" on low confidence / abstain."""
        from .presets import route
        answers = self.preset(state, "route_intent", lang, max_error)
        return {**route(answers, min_confidence, max_error), "answers": answers}

    def models(self):
        return self._send("GET", self.base + "/v1/models")["models"]

    def health(self):
        return self._send("GET", self.base + "/health")

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncBasal(_Client):
    """The asyncio client: the methods of Basal as coroutines. http: an httpx.AsyncClient of your own."""

    def __init__(self, url="http://127.0.0.1:8000", timeout=60, early_exit=None, facts=None, lang=None, retries=0,
                 backoff=0.5, http=None, transport=None, api_key=None):
        super().__init__(url, early_exit, facts, lang, retries, backoff)
        self.http = http or httpx.AsyncClient(timeout=timeout, transport=transport)
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    async def _send(self, method, url, **kw):
        for attempt in range(self.retries + 1):
            r = await self.http.request(method, url, **kw)
            if r.status_code == 200:
                return r.json()
            e = _error(r)
            if not isinstance(e, OutOfMemory) or attempt == self.retries:
                raise e
            await asyncio.sleep(self.backoff * 2 ** attempt)

    async def request(self, body):
        return await self._send("POST", self.url, json=body)

    async def decide(self, state, questions, **fields):
        answers = (await self.request(self._body(state, questions, fields)))["answers"]
        return {k: complete(a) for k, a in answers.items()}

    async def _one(self, state, q):
        return (await self.decide(state, {"q": q}))["q"]

    async def preset(self, state, name, lang=None, max_error=None):
        from .presets import questions
        return await self.decide(state, questions(name, lang or self.defaults.get("lang"), state, max_error))

    async def guard(self, state, kind="input", policy="strict", lang=None, max_error=None):
        from .presets import guard
        answers = await self.preset(state, "guard_" + kind, lang, max_error)
        return {**guard(answers, policy, max_error, expect=kind), "answers": answers}

    async def route(self, state, lang=None, min_confidence=0.6, max_error=None):
        from .presets import route
        answers = await self.preset(state, "route_intent", lang, max_error)
        return {**route(answers, min_confidence, max_error), "answers": answers}

    async def models(self):
        return (await self._send("GET", self.base + "/v1/models"))["models"]

    async def health(self):
        return await self._send("GET", self.base + "/health")

    async def aclose(self):
        await self.http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()
