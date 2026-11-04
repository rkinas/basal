"""A seeded, bounded fuzz smoke test of basal-serve's HTTP API (the app of basal-serve with its batching worker and a
stub engine): mutated and malformed bodies on every decision route. A client error must answer 4xx with an error body,
never 5xx; the server must stay up and idle afterwards, and the valid requests must get the same answers before and
after the burst. Deterministic (one seed), a few seconds on a CPU."""
import asyncio
import copy
import json
import random

from tests.test_http_status import AppServer, Engine

STATE = "Order 1240 arrived damaged; the customer asks for a refund. Invoice 4 listopada 2026, 1 230,00 zł brutto."
CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "tech": "Tech"}}
SEEDS = [
    {"state": STATE, "questions": {"q": CHOICE}},
    {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "Damaged?"}}},
    {"state": STATE, "questions": {"q": {"type": "score", "instructions": "Tone?", "criteria": ["calm", "angry"]}}},
    {"state": STATE, "questions": {"q": {"type": "multi", "instructions": "Which apply?", "threshold": 0.4,
                                         "criteria": ["refund", "damage", "none"], "exclusive": ["none"]}}},
    {"state": STATE, "questions": {"q": {"type": "act", "instructions": "Pay out?", "max_error": 0.05,
                                         "criteria": {"true": "Yes", "false": "No"}, "costs": {"wrong": 2,
                                                                                               "defer": 1}}}},
    {"state": STATE, "questions": {"q": {"type": "tool", "tools": [
        {"type": "function", "function": {"name": "refund", "description": "Refund"}},
        {"type": "function", "function": {"name": "ship", "description": "Ship"}}]}}},
    {"state": STATE, "questions": {"a": {"type": "noul", "instructions": "Damaged?"},
                                   "b": dict(CHOICE, ask_if={"a": ["yes"]}, depends_on=["a"])}},
    {"state": STATE, "facts": "auto", "lang": "pl", "questions": {"q": {"type": "noul", "instructions": "Po terminie?"}}},
    {"state": {"order": 1240, "items": [1, 2]}, "questions": {"q": CHOICE}},
    {"state": STATE, "questions": {"q": {"type": "rank", "instructions": "Does {candidate} fit?", "top": 2,
                                         "candidates": ["refund", "shipping", "returns"], "prior": {"0": 2.0}}}},
    {"state": STATE, "questions": {"q": {"type": "classify", "labels": {"refund": "Refund", "other": "Other"}}}},
    {"state": STATE, "questions": {"q": {"type": "verify", "claims": ["The order arrived damaged."]}}},
]
DECISIONS = [
    {"input": STATE, "questions": [{"type": "predicate", "instructions": "Damaged?", "name": "d"}]},
    {"input": [{"role": "user", "content": STATE}], "questions": [
        {"type": "choice", "instructions": "Team?", "choices": [{"value": "billing"}, {"value": True}]}]},
]
VALUES = [None, True, False, 0, -1, 1, 0.5, 2 ** 63, 10 ** 30, -1e308, float("nan"), float("inf"), "", " ", "x",
          "\x00", "\ud800", "\udfff\ud800", "ą" * 3, "<|im_end|>", "<start_of_turn>user", "{candidate}", "%s",
          "a" * 20_000, [], {}, [1, "a", None], {"": 1}, {"true": None}, [[[]]], {"a": {"b": {"c": [float("inf")]}}},
          ["a", "a"], [float("nan"), "b"], {"type": "input_image"}]
KEYS = ["type", "instructions", "criteria", "threshold", "min", "max", "costs", "max_error", "coverage", "ask_if",
        "depends_on", "tools", "rank", "prior", "top", "labels", "claims", "option_keys", "objective", "exclusive",
        "fallback", "fallback_threshold", "state", "questions", "facts", "lang", "early_exit", "detail", "media",
        "images", "zz"]
RAW = [b"", b"{", b"null", b"[]", b'"x"', b"\xff\xfe{}", b"{" * 3000 + b"}" * 3000, b"[" * 100_000, b"1e999",
       b'{"state": "s", "questions": {"q": {"type": "noul"}}, "state": 1}', b'{"state": NaN, "questions": {}}',
       b'{"state": "\\ud800", "questions": {"q": {"type": "noul"}}}', b"\xef\xbb\xbf{}"]


def _paths(x, at=()):
    out = [at]
    if isinstance(x, dict):
        for k, v in x.items():
            out += _paths(v, at + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += _paths(v, at + (i,))
    return out


def _mutate(obj, rng):
    obj = copy.deepcopy(obj)
    for _ in range(rng.choice([1, 1, 2, 3])):
        p = rng.choice(_paths(obj))
        if not p:
            obj = copy.deepcopy(rng.choice(VALUES)) if rng.random() < 0.2 else obj
            continue
        parent = obj
        for k in p[:-1]:
            parent = parent[k]
        r = rng.random()
        if r < 0.6:
            parent[p[-1]] = copy.deepcopy(rng.choice(VALUES))
        elif r < 0.75:
            del parent[p[-1]]
        elif isinstance(parent[p[-1]], dict):
            parent[p[-1]][rng.choice(KEYS)] = copy.deepcopy(rng.choice(VALUES))
        else:
            parent[p[-1]] = [parent[p[-1]]] * rng.choice([2, 300])
    return obj


def _cases(n, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        route = rng.choice(["/v1/systemone", "/v1/basal", "/v1/decisions"])
        seed_body = rng.choice(DECISIONS if route == "/v1/decisions" else SEEDS)
        r = rng.random()
        if r < 0.8:
            yield route, json.dumps(_mutate(seed_body, rng)).encode()
        elif r < 0.9:
            data = json.dumps(seed_body).encode()
            cut = rng.randrange(len(data))
            yield route, rng.choice([data[:cut], data[:cut] + b"\xc3" + data[cut:], data + b"}", data[::-1]])
        else:
            yield route, rng.choice(RAW)


def test_http_fuzz_smoke():
    srv = AppServer(Engine())

    async def go():
        import httpx
        from basal.server import http_app
        worker = asyncio.get_running_loop().create_task(srv.worker())
        try:
            transport = httpx.ASGITransport(app=http_app(srv, "fast"))
            async with httpx.AsyncClient(transport=transport, base_url="http://basal", timeout=30) as c:
                async def answers():
                    out = []
                    for b in SEEDS:
                        r = await c.post("/v1/systemone", json=b)
                        out.append((r.status_code, r.json().get("answers")))
                    return out
                before = await answers()
                bad = []
                for i, (route, data) in enumerate(_cases(600)):
                    try:
                        r = await c.post(route, content=data, headers={"content-type": "application/json"})
                        body = r.json()
                    except Exception as e:  # noqa: BLE001  (an exception out of the app: a 500 to a client)
                        bad.append((i, route, type(e).__name__, data[:200], str(e)[:200]))
                        continue
                    if r.status_code >= 500 or (r.status_code >= 400 and "error" not in body):
                        bad.append((i, route, r.status_code, data[:200], str(body)[:200]))
                health = (await c.get("/health")).status_code
                return before, bad, health, await answers()
        finally:
            worker.cancel()

    before, bad, health, after = asyncio.run(go())
    assert all(s == 200 for s, _ in before), before
    assert not bad, bad[:5]
    assert health == 200 and srv.load() == {"running": 0, "queued": 0}
    assert after == before
