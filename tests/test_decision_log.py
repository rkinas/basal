"""--log-decisions: the request key covers the media and the language of a request, and every line is strict JSON."""
import json
import math

from basal.server import Server


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
