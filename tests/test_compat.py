"""The request shape of vLLM's structured-diffusion /v1/systemone on basal-serve: depends_on / ask_if (a post-filter:
a skipped question is answered null), the "images" alias of media and --error-format openai. Without these fields and
the flag, requests and answers are as before."""
import asyncio

import httpx
import pytest

from basal.server import Server, error_body, http_app, images_as_media, parser
from tests.test_calibrate import engine  # noqa: F401  (a pytest fixture: the stub engine behind --mode eager)
from tests.test_http_status import AppServer, Engine
from tests.test_server import FakeServer


def noul(p, **kw):
    return dict(type="noul", instructions="Is it?", criteria={"true": f"Yes ~{p}", "false": f"No ~{1 - p}"}, **kw)


CHOICE = dict(type="choice", instructions="Which team?", option_keys="hide",
              criteria={"billing": "Billing ~0.2", "tech": "Tech ~0.8"})
SCORE = dict(type="score", instructions="Tone?", criteria=["calm ~0.1", "annoyed ~0.7", "furious ~0.2"])


@pytest.fixture
def ask(engine):  # noqa: F811
    srv = Server(parser().parse_args(["--mode", "eager"]))

    def run(questions, **body):
        async def go():
            w = asyncio.get_running_loop().create_task(srv.worker())
            try:
                return await srv.decide({"state": "x", "questions": questions, **body})
            finally:
                w.cancel()
        return asyncio.run(go())
    return run


def test_ask_if_answers_null_when_the_condition_fails(ask):
    out = ask({"urgent": noul(0.9), "team": dict(CHOICE, ask_if={"urgent": ["yes"]}),
               "tone": dict(SCORE, ask_if={"team": ["billing"]}),
               "late": dict(noul(0.6), ask_if={"tone": ["annoyed ~0.7"], "urgent": [True]})})
    a = out["answers"]
    assert a["team"]["choice"] == "tech" and a["tone"] is None and a["late"] is None   # late: tone was skipped
    assert out["usage"]["skipped"] == ["tone", "late"]
    out = ask({"urgent": noul(0.2), "team": dict(CHOICE, ask_if={"urgent": ["yes", "true"]}),
               "tone": dict(SCORE, ask_if={"urgent": ["no"]}), "lvl": dict(SCORE, ask_if={"tone": ["1"]})})
    assert out["answers"]["team"] is None and out["answers"]["tone"]["probabilities"]["1"] > 0.5
    assert out["answers"]["lvl"] is not None and out["usage"]["skipped"] == ["team"]  # a level key or its text


def test_without_the_fields_nothing_changes(ask):
    plain = ask({"urgent": noul(0.9), "team": CHOICE})
    assert "skipped" not in plain["usage"]
    dep = ask({"urgent": noul(0.9), "team": dict(CHOICE, depends_on=["urgent"])})   # depends_on only orders
    for out in (plain, dep):
        out["usage"].pop("latency_ms")
    assert dep == plain


@pytest.mark.parametrize("questions,message", [
    ({"a": noul(0.9, ask_if={"b": ["yes"]})}, "depends on unknown question 'b'"),
    ({"a": noul(0.9, depends_on=["a"])}, "depends on unknown question 'a'"),
    ({"a": noul(0.9, depends_on="b"), "b": noul(0.5)}, "depends_on must be a list"),
    ({"a": noul(0.9, ask_if={"b": []}), "b": noul(0.5)}, "non-empty list"),
    ({"a": noul(0.9, ask_if={"b": ["maybe"]}), "b": noul(0.5)}, "ask_if values ['maybe'] for 'b'"),
    ({"a": noul(0.9, ask_if={"b": ["sales"]}), "b": CHOICE}, "must be among ['billing', 'tech']"),
    ({"a": noul(0.9, depends_on=["b"]), "b": noul(0.5, ask_if={"a": ["no"]})}, "a cycle among ['a', 'b']"),
    ({"a": noul(0.9, ask_if={"b": ["defer"]}),
      "b": {"type": "act", "instructions": "?", "criteria": {"true": "y", "false": "n"},
            "costs": {"wrong": 2, "defer": 1}}}, "not a choice, noul or score question"),
])
def test_ask_if_refusals(ask, questions, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        ask(questions)


def test_images_alias():
    assert images_as_media(["data:image/png;base64,iVBO", {"content_type": "image/jpeg", "base64": "/9j/"}]) == [
        {"type": "image", "data": "data:image/png;base64,iVBO"}, {"type": "image", "data": "/9j/"}]
    for bad in ([], "x", ["http://a/b.png"], [{"content_type": "text/plain", "base64": "QQ=="}]):
        with pytest.raises(ValueError, match="images"):
            images_as_media(bad)
    run = lambda body: asyncio.run(FakeServer([0.5, 0.5])._run({"state": "x", "questions": {"q": noul(0.5)}, **body}))
    for body in ({"images": ["data:image/png;base64,iVBO"]}, {"media": [{"type": "image", "data": "iVBO"}]}):
        with pytest.raises(ValueError, match="this server does not take media"):   # the alias reaches media
            run(body)
    with pytest.raises(ValueError, match="media or images, not both"):
        run({"images": ["data:image/png;base64,iVBO"], "media": [{"type": "image", "data": "iVBO"}]})


def test_error_format():
    from basal.prompt import RequestError
    e = RequestError("missing field 'state'")
    assert error_body(e) == ({"error": "missing field 'state'"}, 422)
    assert error_body(e, "openai") == ({"error": {"message": "missing field 'state'", "type": "validation_error",
                                                  "code": 422}}, 422)
    assert error_body(RuntimeError("boom"), "openai")[0]["error"]["type"] == "server_error"
    assert parser().parse_args([]).error_format == "basal"

    async def go(fmt):
        srv = AppServer(Engine())
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            app = http_app(srv, "fast") if fmt is None else http_app(srv, "fast", error_format=fmt)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://basal") as c:
                return await c.post("/v1/systemone", json={"questions": {}})
        finally:
            w.cancel()
    default, basal, openai = (asyncio.run(go(f)) for f in (None, "basal", "openai"))
    assert default.status_code == basal.status_code == openai.status_code == 422
    assert default.content == basal.content == b'{"error":"missing field \'state\'"}'
    assert openai.json()["error"]["message"] == "missing field 'state'"
