"""A long state costs its tokenizations once, off the event loop: the option codes are read at the prompt's end
(answer_end) instead of once per code over the whole prompt, and a request whose prompts hold over PREP_INLINE of text
is prepared in the preparation thread while the loop answers /health."""
import asyncio
import threading
import time

from test_http_status import AppServer, Engine
from test_server import Tok

from basal import server as S
from basal.prompt import FORMATS, LETTER_TAIL, answer_end, letter_ids, render


class Counting(Tok):
    def __init__(self):
        self.chars = 0

    def __call__(self, text, add_special_tokens=False):
        self.chars += sum(map(len, text)) if isinstance(text, list) else len(text)
        return super().__call__(text, add_special_tokens)


def test_the_codes_of_a_long_prompt_are_read_at_its_end():
    tok = Counting()
    state = "Faktura 1240 z 15 marca 2024 r.\n" * 30000  # ~1 MB
    prompt = render(tok, state, "Która?", ["a", "b", "c"], "pl", 1)
    end = answer_end(tok, prompt)
    assert len(end) < 3 * LETTER_TAIL and prompt.endswith(end) and end.startswith("\n")
    tok.chars = 0
    k = len(FORMATS[1]["codes"])
    ids = letter_ids(tok, prompt, k, 1)
    assert ids == [ord(c) for c in FORMATS[1]["codes"][:k]]
    assert tok.chars < 3 * len(prompt)  # (was k + 1 times the whole prompt)
    short = render(tok, "x", "Która?", ["a", "b"], "pl", 1)
    assert answer_end(tok, short) is short  # a short prompt is read whole, as before


def test_health_answers_while_a_100_kib_state_is_prepared(monkeypatch):
    release = threading.Event()
    real = AppServer.encode

    def slow_encode(self, branches):
        release.wait(10)
        return real(self, branches)
    monkeypatch.setattr(AppServer, "encode", slow_encode)
    srv = AppServer(Engine())
    srv.context = (10**6, "test context")
    body = {"state": "Faktura 1240. " * 7500, "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}}
    assert S.PREP_INLINE < len(body["state"]) * 2  # (two option orders)

    async def go(c):
        timer = threading.Timer(3.0, release.set)
        timer.start()
        t = time.perf_counter()
        req = asyncio.create_task(c.post("/v1/systemone", json=body))
        await asyncio.sleep(0.2)
        h = await c.get("/health")
        waited = time.perf_counter() - t
        release.set()
        timer.cancel()
        return h, waited, await req
    from test_overload import _with_app
    h, waited, r = asyncio.run(_with_app(srv, go))
    assert h.status_code == 200 and waited < 1.0 and r.status_code == 200
