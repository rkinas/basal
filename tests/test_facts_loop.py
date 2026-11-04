"""facts=auto runs off the event loop."""
import asyncio
import threading

import pytest
from test_server import FakeServer

import basal.server as server_module


def test_facts_auto_runs_off_the_event_loop(monkeypatch):
    seen = {}

    def inject_facts(text):
        seen["worker"] = threading.get_ident()
        raise RuntimeError("stop")

    async def go():
        seen["loop"] = threading.get_ident()
        body = {"state": "Umowa z 31.12.2025 na 30 dni.", "facts": "auto",
                "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}}}
        with pytest.raises(RuntimeError, match="stop"):
            await FakeServer([0.5, 0.5]).decide(body)

    monkeypatch.setattr(server_module, "inject_facts", inject_facts)
    asyncio.run(go())
    assert seen["worker"] != seen["loop"]
