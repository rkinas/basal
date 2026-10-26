"""basal/examples/local_backend.py: the request it sends and how it reads answers and abstain, against a stub server."""
import importlib.util
import json
from pathlib import Path

import httpx

spec = importlib.util.spec_from_file_location("local_backend", Path(__file__).parents[1] / "basal/examples/local_backend.py")
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def stub(monkeypatch, answer):
    seen = {}

    def post(url, json=None, timeout=None):
        seen.update(url=url, body=json)
        return httpx.Response(200, json={"answers": {"q": answer}}, request=httpx.Request("POST", url))
    monkeypatch.setattr(lb.httpx, "post", post)
    return seen


def test_ask_choice_returns_the_key_and_confidence(monkeypatch):
    seen = stub(monkeypatch, {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.9}, "confidence": 0.9})
    assert lb.ask_choice("http://h:1/", "s", "?", {"refund": "r"}) == ("refund", 0.9)
    assert seen["url"] == "http://h:1/v1/systemone"
    assert "max_error" not in seen["body"]["questions"]["q"]


def test_abstain_gives_none(monkeypatch):
    seen = stub(monkeypatch, {"type": "choice", "choice": "refund", "confidence": 0.4, "abstain": True})
    assert lb.ask_choice("http://h:1", "s", "?", {"refund": "r"}, max_error=0.05) == (None, 0.4)
    assert seen["body"]["questions"]["q"]["max_error"] == 0.05
