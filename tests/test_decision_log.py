"""--log-decisions: the request key covers the media and the language of a request, and every line is strict JSON."""
import asyncio
import json
import math

from basal.server import Server, parser
from tests.test_calibrate import engine  # noqa: F401 (fixture)


class Log:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)


OUT = {"answers": {"q": {"type": "noul", "probabilities": [0.5, 0.5]}}}
Q = {"q": {"type": "noul", "question": "ok?"}}


def logged(body, out=OUT):
    srv = Server.__new__(Server)
    srv.name, srv.log = "m", Log()
    srv.log_decision(body, out)
    return srv.log.lines[-1]


def key(body):
    return json.loads(logged(body))["request_sha256"]


def test_two_images_with_the_same_text_have_different_keys():
    a = key({"state": "same", "questions": Q, "media": [{"type": "image", "data": "AAAA"}]})
    b = key({"state": "same", "questions": Q, "media": [{"type": "image", "data": "BBBB"}]})
    assert a != b


def test_a_different_language_is_a_different_key():
    assert key({"state": "same", "questions": Q, "lang": "pl"}) != key({"state": "same", "questions": Q, "lang": "en"})


def test_a_nan_probability_is_logged_as_strict_json():
    nan = {"answers": {"q": {"type": "noul", "probabilities": [math.nan, 1.0]}}}
    json.loads(logged({"state": "s", "questions": Q}, nan), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_a_request_without_media_or_language_keeps_its_key():
    assert key({"state": "s", "questions": Q}) == key({"state": "s", "questions": Q})


def refuse(token):
    raise ValueError(f"not strict JSON: {token}")


def test_a_request_without_media_or_language_keeps_the_key_of_2_6():
    """Decision logs of earlier releases still join by key for these requests (the key of basal 2.6 for this one)."""
    body = {"state": "Umowa z 3 maja.", "questions": Q}
    old = "266f86a1117e3da1a22996ab78541cf37b1e973325f55f897424ff18673b8896"
    assert key(body) == key(dict(body, lang=None)) == old
    assert key(dict(body, lang="pl")) != old and key(dict(body, media=[{"type": "image", "data": "AAAA"}])) != old


def test_the_images_alias_is_keyed_as_the_media_it_stands_for():
    """The think server logs the request as it came, with its "images" alias: an image sent as a {content_type,
    base64} object is keyed by its data, as the media item decide() turns it into, not as an empty one."""
    def img(data):
        return key({"state": "s", "questions": Q, "images": [{"content_type": "image/png", "base64": data}]})

    def media(data):
        return key({"state": "s", "questions": Q, "media": [{"type": "image", "data": data}]})
    assert img("AAAA") != img("BBBB") and img("AAAA") == media("AAAA")
    url = "data:image/png;base64,AAAA"
    assert key({"state": "s", "questions": Q, "images": [url]}) == media(url)


def test_nan_and_inf_are_logged_as_null():
    out = {"answers": {"q": {"type": "noul", "probabilities": [math.nan, math.inf], "confidence": -math.inf},
                       "r": {"type": "choice", "choice": "a", "probabilities": (0.25, 0.75), "confidence": 0.75}}}
    rec = json.loads(logged({"state": "s", "questions": Q}, out), parse_constant=refuse)
    assert rec["answers"]["q"] == {"type": "noul", "probabilities": [None, None], "confidence": None}
    assert rec["answers"]["r"] == {"type": "choice", "choice": "a", "probabilities": [0.25, 0.75], "confidence": 0.75}


def test_a_served_request_is_logged_with_its_language(tmp_path, engine):  # noqa: F811
    log = tmp_path / "decisions.jsonl"
    srv = Server(parser().parse_args(["--mode", "eager", "--model", "stub", "--log-decisions", str(log)]))
    noul = {"type": "noul", "instructions": "Covered?", "criteria": {"true": "Yes ~0.9", "false": "No ~0.1"}}
    body = {"state": "The policy covers fire.", "questions": {"q": noul}}

    async def go():
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            for b in (body, dict(body, lang="pl"), dict(body, lang="en"), body):
                await srv.decide(b)
        finally:
            w.cancel()
    asyncio.run(go())
    keys = [json.loads(line, parse_constant=refuse)["request_sha256"]
            for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(keys) == 4 and len(set(keys[:3])) == 3 and keys[3] == keys[0]
