"""--think auto with every request basal-serve takes (a stub readout 2, no vLLM): a question skipped by ask_if (null)
and task / rank questions do not break the think pass, whatever the request's think; only plain questions think. The
sidecar's log file is closed with it."""
import asyncio
import threading

import pytest
from test_server import FakeServer

from basal import think as T

Q = {"type": "choice", "criteria": ["a", "b"]}


class Probe(T.ThinkServer, FakeServer):
    """ThinkServer.decide over FakeServer's engine; readout 2 answers [0.9, 0.1] for every question it is given."""

    def __init__(self):
        FakeServer.__init__(self, lambda p, ids: [1 / len(ids)] * len(ids))
        self.think, self.think_cal, self.think_cal_name, self.think_log = "auto", {}, "off", None
        self.thought = []

    async def think_one(self, q):
        self.thought.append(q["name"])
        return ([0.9, 0.1], "note", 1, 0), None


@pytest.mark.parametrize("mode", ["off", "auto", "force"])
def test_a_question_skipped_by_ask_if_is_null(mode):
    s = Probe()
    out = asyncio.run(s._run({"state": "text", "think": mode,
                              "questions": {"parent": Q, "child": dict(Q, ask_if={"parent": ["b"]})}}))
    assert out["answers"]["child"] is None and out["usage"]["skipped"] == ["child"]
    assert out["answers"]["parent"]["meta"]["think"]["used"] is (mode != "off")


def test_task_questions_do_not_think_and_plain_ones_still_do():
    s = Probe()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    out = asyncio.run(s._run({"state": "Alice works here.", "think": "force",
                              "questions": {"x": {"type": "extract", "schema": schema}, "q": Q}}))
    assert "x" in out["answers"] and out["answers"]["q"]["think"]["tokens"] == 1
    assert "q" in s.thought and "x" not in s.thought


def test_the_sidecar_closes_its_log(tmp_path, monkeypatch):
    class Proc:
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(T.atexit, "register", lambda f: None)
    car = T.Sidecar("model", log=str(tmp_path / "vllm.log"))
    f = car.log_file
    assert not f.closed
    car.close()
    assert f.closed


def test_facts_are_read_off_the_event_loop(monkeypatch):
    """facts=auto: the think pass computes the facts of the state readout 1 saw in a worker thread, as basal-serve
    does, so a long state does not stall the event loop."""
    s = Probe()
    seen = []

    def facts(text):
        seen.append(threading.get_ident())
        return text
    monkeypatch.setattr(T, "inject_facts", facts)

    async def go():
        return threading.get_ident(), await s._run({"state": "Umowa z 1.02.2024 r.", "facts": "auto", "think": "force",
                                                    "questions": {"q": Q}})
    loop, out = asyncio.run(go())
    assert seen and loop not in seen and out["answers"]["q"]["think"]["tokens"] == 1
