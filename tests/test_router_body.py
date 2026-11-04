"""basal-router reads a request body once (parse_body): a body it cannot read as a request (nested beyond the parser,
a lone surrogate, not JSON) goes to a replica as it came, whose own error the client gets, never a 500 of the
router; --route names a state's script as before, each distinct character looked up once."""
import asyncio
import random
import unicodedata

import httpx
import pytest
from starlette.testclient import TestClient

from basal import router as R

MAIN, CYR = "http://main", "http://cyr"


def replica_transport(seen):
    """A replica that refuses what basal-serve refuses with a 422, else answers; it records the bodies it got."""
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.append(req.content)
        return httpx.Response(422, json={"error": "invalid request", "code": "INVALID_REQUEST"})
    return httpx.MockTransport(handler)


DEEP = b'{"state": ' + b"[" * 50000 + b"]" * 50000 + b', "questions": {"q": {"type": "choice"}}}'
SURROGATE = b'{"state": "Raport \\udead", "questions": {"q": {"type": "choice"}}}'


@pytest.mark.parametrize("body", [DEEP, SURROGATE, b'{"state": "x", "questions": ', b"[1, 2]"])
@pytest.mark.parametrize("kw", [{"routes": {"cyrillic": [CYR]}}, {"escalate": ["http://big"], "below": 0.8}, {}])
def test_a_body_the_router_cannot_read_goes_to_a_replica_as_it_came(body, kw):
    seen = []
    rt = R.Router([MAIN], transport=replica_transport(seen), **kw)
    with TestClient(R.app(rt, poll_s=0.05), raise_server_exceptions=False) as c:
        asyncio.run(rt.poll())
        r = c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_REQUEST" and seen == [body]


def _script_per_char(text):  # the reading script_of keeps: every letter named, scripts counted in text order
    count = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            sc = next((x for x in R.SCRIPTS[:5] if name.startswith(x.upper())), None)
            sc = sc or ("cjk" if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) else "other")
            count[sc] = count.get(sc, 0) + 1
    return max(count, key=count.get) if count else None


def test_script_of_names_each_character_once_with_the_same_result():
    rng = random.Random(0)
    pools = ["abcxyzĄęł", "абвгдїє", "αβγδ", "אבג", "ابت", "漢字かなカナ한", "ᚠᚢᚦ", "  12,.!"]
    for _ in range(300):
        text = "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randrange(0, 40)))
        assert R.script_of(text) == _script_per_char(text), text
    assert R.script_of("ab аб") == "latin" and R.script_of("аб ab") == "cyrillic"  # a tie: the script seen first
