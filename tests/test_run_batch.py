"""basal-run: the batch runner against a fake server (httpx.MockTransport): request lines are sent with all their
fields, a bad line is its own error, the input is read and the answers written as they go (a bounded window), an
interrupted run keeps its answers so far, and the input is never the output."""
import argparse
import asyncio
import json

import httpx
import pytest

from basal import run as run_module

URL = "http://basal/v1/systemone"


def args(tmp_path, concurrency=2, **kw):
    return argparse.Namespace(input=str(tmp_path / "in.jsonl"), output=str(tmp_path / "out.jsonl"), url=URL,
                              concurrency=concurrency, early_exit=None, preset=None, lang=None, **kw)


def write(path, lines):
    path.write_text("".join((x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)) + "\n" for x in lines),
                    encoding="utf-8")


def answer(body):
    qs = body["questions"]
    return {"model": "m", "usage": {"latency_ms": 1.0}, "answers": {
        k: {"type": q.get("type", "choice"), "confidence": 0.9,
            "probabilities": {"true": 0.9, "false": 0.1} if q.get("type") == "noul"
            else {c: (0.9 if i == 0 else 0.1 / max(1, len(q["criteria"]) - 1)) for i, c in enumerate(q["criteria"])}}
        for k, q in qs.items()}}


def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


SIMPLE = {"state": "Order 1240 arrived damaged.", "question": "Refund?", "options": ["yes", "no"], "gold": 0}


def test_output_equal_to_input_is_refused(tmp_path):
    write(tmp_path / "in.jsonl", [SIMPLE])
    before = (tmp_path / "in.jsonl").read_bytes()
    for out in (tmp_path / "in.jsonl", tmp_path / "." / "in.jsonl"):
        a = args(tmp_path)
        a.output = str(out)
        with pytest.raises(SystemExit, match="is the --input file"):
            asyncio.run(run_module.run(a, transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    (tmp_path / "link.jsonl").symlink_to(tmp_path / "in.jsonl")
    a = args(tmp_path)
    a.output = str(tmp_path / "link.jsonl")
    with pytest.raises(SystemExit, match="is the --input file"):
        asyncio.run(run_module.run(a, transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert (tmp_path / "in.jsonl").read_bytes() == before


def test_a_bad_line_is_its_own_error_and_the_run_goes_on(tmp_path, capsys):
    write(tmp_path / "in.jsonl", [SIMPLE, {"state": "no options", "question": "?"}, "{not json", "[1, 2]",
                                  {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}},
                                  dict(SIMPLE, id="last")])
    asyncio.run(run_module.run(args(tmp_path), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=answer(json.loads(r.content))))))
    out = read(tmp_path / "out.jsonl")
    assert [r["id"] for r in out] == [0, 1, 2, 3, 4, "last"]
    assert ["error" in r for r in out] == [False, True, True, True, False, False]
    assert "KeyError" in out[1]["error"] and "not JSON" in out[2]["error"] and "not a JSON object" in out[3]["error"]
    assert out[0]["correct"] is True and out[5]["prediction"] == 0
    s = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert s["items"] == 6 and s["errors"] == 3 and s["accuracy"] == 1.0
    assert not (tmp_path / "out.jsonl.part").exists()


def test_input_is_read_as_it_goes_with_a_bounded_window(tmp_path, monkeypatch):
    """Memory follows --concurrency, not the input: at most a few times --concurrency lines are read ahead of the
    answers (the whole file used to be read, and every answer held, before the first line was written)."""
    n, conc = 300, 2
    write(tmp_path / "in.jsonl", [dict(SIMPLE, id=i) for i in range(n)])
    read_so_far, answered, ahead = [0], [0], []
    lines = run_module.lines

    def counted(path):
        for x in lines(path):
            read_so_far[0] += 1
            yield x
    monkeypatch.setattr(run_module, "lines", counted)

    def handler(r):
        ahead.append(read_so_far[0] - answered[0])
        answered[0] += 1
        return httpx.Response(200, json=answer(json.loads(r.content)))
    asyncio.run(run_module.run(args(tmp_path, conc), transport=httpx.MockTransport(handler)))
    assert [r["id"] for r in read(tmp_path / "out.jsonl")] == list(range(n))
    assert max(ahead) <= 4 * conc + 1


def test_an_interrupted_run_keeps_the_answers_so_far(tmp_path, capsys):
    write(tmp_path / "in.jsonl", [dict(SIMPLE, id=i) for i in range(20)])
    calls = [0]

    def handler(r):
        calls[0] += 1
        if calls[0] == 6:
            raise RuntimeError("the client died")
        return httpx.Response(200, json=answer(json.loads(r.content)))
    with pytest.raises(RuntimeError, match="the client died"):
        asyncio.run(run_module.run(args(tmp_path, concurrency=1), transport=httpx.MockTransport(handler)))
    assert [r["id"] for r in read(tmp_path / "out.jsonl.part")] == [0, 1, 2, 3, 4]
    assert not (tmp_path / "out.jsonl").exists()
    assert "the 5 answers so far are in" in capsys.readouterr().err
