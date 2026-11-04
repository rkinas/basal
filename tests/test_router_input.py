"""basal-router input handling: --route reads a state of any JSON type as the text the replicas read; the --fallback
LLM never takes a request with images (the "images" alias of "media")."""
import asyncio

import pytest
from starlette.testclient import TestClient
from test_router import FB, OPTS, SMALL, _ask, _ch, cascade_transport, fallback_transport

from basal import router as R


@pytest.mark.parametrize("state", [42, 1.5, True, None, [], {}, ["Звіт", "про"], {"звіт": "інцидент"}, "Raport"])
def test_route_takes_a_state_of_any_json_type(state):
    main, cyr = "http://main", "http://cyr"
    rt = R.Router([main], transport=cascade_transport({main: 0.9, cyr: 0.9}), routes={"cyrillic": [cyr]})
    with TestClient(R.app(rt, poll_s=0.05), raise_server_exceptions=False) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", json={"state": state, "questions": {"q": {"type": "choice"}}})
    assert r.status_code == 200
    cyrillic = isinstance(state, (list, dict)) and bool(state)
    assert r.json()["replica"] == (cyr if cyrillic else main)        # the JSON text the model reads, keys too


def test_script_of_reads_json_values_as_their_text():
    assert R.script_of(42) is None and R.script_of(None) is None and R.script_of(True) == "latin"  # "true"
    assert R.script_of({"звіт": "інцидент"}) == "cyrillic" and R.script_of(["Raport", "z"]) == "latin"


@pytest.mark.parametrize("field", ["media", "images"])
def test_fallback_never_takes_a_request_with_images(field):
    asked = {}
    low = _ch(0.3)
    conf = {SMALL: {"q": low}}
    media = {"media": [{"type": "image", "data": "x"}], "images": ["data:image/png;base64,eA=="]}[field]
    r, _ = _ask(R.Router([SMALL], transport=fallback_transport(conf, asked), fallback=FB),
                {"state": "s", "questions": {"q": {"criteria": OPTS}}, field: media})
    assert r.status_code == 200 and r.json()["answers"]["q"] == low and "llm" not in asked
