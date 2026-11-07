"""basal-run: the batch runner against a fake server (httpx.MockTransport): request lines are sent with all their
fields, a bad line is its own error, the input is read and the answers written as they go (a bounded window), an
interrupted run keeps its answers so far, and the input is never the output."""
import argparse
import asyncio
import json
import sys

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


def test_request_lines_keep_every_request_field(tmp_path):
    """A full-request line is sent as it is (media, facts, lang, model, detail, think, images and fields a later
    server may read), without the line's own id / gold / group / family / _notes."""
    req = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}, "early_exit": "0.99",
           "media": [{"type": "image", "data": "aGk="}], "facts": "auto", "lang": "pl", "model": "basal-x",
           "detail": "high", "think": "off", "future_field": {"x": 1}}
    line = dict(req, id="r1", gold={"q": "yes"}, group="g", family="f", _note="mine")
    write(tmp_path / "in.jsonl", [line])
    seen = []

    def handler(r):
        seen.append(json.loads(r.content))
        return httpx.Response(200, json=answer(seen[-1]))
    asyncio.run(run_module.run(args(tmp_path), transport=httpx.MockTransport(handler)))
    assert seen == [req]
    assert run_module.to_request({"state": "s", "questions": {}, "images": ["data:image/png;base64,aGk="]})[
        "images"] == ["data:image/png;base64,aGk="]


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
        if r.method != "POST":  # (GET /v1/models: the server it answers for)
            return httpx.Response(404)
        calls[0] += 1
        if calls[0] == 6:
            raise RuntimeError("the client died")
        return httpx.Response(200, json=answer(json.loads(r.content)))
    with pytest.raises(RuntimeError, match="the client died"):
        asyncio.run(run_module.run(args(tmp_path, concurrency=1), transport=httpx.MockTransport(handler)))
    assert [r["id"] for r in read(tmp_path / "out.jsonl.part")] == [0, 1, 2, 3, 4]
    assert not (tmp_path / "out.jsonl").exists()
    assert "the 5 answers so far are in" in capsys.readouterr().err


# -- resuming an interrupted run (2.8) -------------------------------------------------------------------------------
class Server:
    """A fake basal server: GET /v1/models names `model`; POST answers, counted, and raises at POST `die_at` (the run
    is interrupted there)."""

    def __init__(self, model="m", die_at=None):
        self.model, self.die_at, self.posts = model, die_at, []

    def __call__(self, r):
        if r.method == "GET":
            return httpx.Response(200, json={"models": [{"name": self.model, "mode": "fast"}]})
        body = json.loads(r.content)
        self.posts.append(body["state"])
        if len(self.posts) == self.die_at:
            raise RuntimeError("the client died")
        return httpx.Response(200, json=answer(body))


ITEMS = [dict(SIMPLE, id=i, state=f"Order {i} arrived damaged.") for i in range(20)]
REST = [f"Order {i} arrived damaged." for i in range(5, 20)]


def go(tmp_path, server, concurrency=1, **kw):
    a = args(tmp_path, concurrency)
    for k, v in kw.items():
        setattr(a, k, v)
    asyncio.run(run_module.run(a, transport=httpx.MockTransport(server)))


def reference(tmp_path):
    """The output of the same input, answered in one run."""
    ref = tmp_path / "ref"
    ref.mkdir()
    write(ref / "in.jsonl", ITEMS)
    go(ref, Server())
    return read(ref / "out.jsonl")


def interrupted(tmp_path, at=6):
    write(tmp_path / "in.jsonl", ITEMS)
    with pytest.raises(RuntimeError, match="the client died"):
        go(tmp_path, Server(die_at=at))
    assert [r["id"] for r in read(tmp_path / "out.jsonl.part")] == list(range(at - 1))


def test_the_same_command_goes_on_from_the_part(tmp_path, capsys):
    """Run again after an interruption, the command sends only the lines after the answers in <output>.part (with more
    clients too) and the output is the one an uninterrupted run writes; the summary counts every line."""
    want = reference(tmp_path)
    interrupted(tmp_path)
    assert "the same command goes on from there" in capsys.readouterr().err
    again = Server()
    go(tmp_path, again, concurrency=4)
    assert again.posts == REST
    assert read(tmp_path / "out.jsonl") == want
    out = capsys.readouterr()
    assert "5 answers kept, going on from line 6" in out.err
    s = json.loads(out.out.strip().splitlines()[-1])
    assert s["items"] == 20 and s["errors"] == 0 and s["accuracy"] == 1.0
    assert not (tmp_path / "out.jsonl.part").exists() and not (tmp_path / "out.jsonl.part.json").exists()


def test_a_line_cut_short_and_the_errors_after_the_last_answer_are_sent_again(tmp_path):
    """A kill can leave half a line at the end, and a server that went away leaves errors for the lines after the
    last answer: both are sent again. An error between answers stays as it is (that line failed on its own)."""
    want = reference(tmp_path)
    interrupted(tmp_path)
    part = tmp_path / "out.jsonl.part"
    lines = part.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = json.dumps(run_module.failed(2, "a line that failed")) + "\n"
    lines += [json.dumps(run_module.failed(i, "ConnectError")) + "\n" for i in (5, 6)]
    part.write_text("".join(lines) + '{"id": 7, "answers": {"q": {"ty', encoding="utf-8")
    again = Server()
    go(tmp_path, again)
    assert again.posts == REST
    out = read(tmp_path / "out.jsonl")
    assert out[2]["error"] == "a line that failed"
    assert out[:2] + out[3:] == want[:2] + want[3:]


@pytest.mark.parametrize("change,said", [
    ("input", "another --input"), ("early_exit", "other options: --early-exit"), ("preset", "other options: --preset"),
    ("server", "another server"), ("record", "no record of its run")])
def test_a_part_of_another_run_is_refused(tmp_path, change, said):
    """A .part from another input, other options, another server, or without its record is not continued: nothing is
    sent and it stays as it was. --restart replaces it."""
    interrupted(tmp_path)
    part = tmp_path / "out.jsonl.part"
    before = part.read_bytes()
    kw, server = {}, Server()
    if change == "input":
        write(tmp_path / "in.jsonl", ITEMS + [dict(SIMPLE, id=20)])
    elif change == "early_exit":
        kw["early_exit"] = "0.99"
    elif change == "preset":
        kw["preset"] = "support"
    elif change == "server":
        server = Server(model="another")
    else:
        (tmp_path / "out.jsonl.part.json").unlink()
    with pytest.raises(SystemExit, match=said) as e:
        go(tmp_path, server, **kw)
    assert "--restart replaces it" in str(e.value)
    assert server.posts == [] and part.read_bytes() == before
    if change in ("server", "record"):
        go(tmp_path, server, restart=True)
        assert len(server.posts) == 20 and [r["id"] for r in read(tmp_path / "out.jsonl")] == list(range(20))


# -- a server that stops answering (2.8) -------------------------------------------------------------------------------
class Down(Server):
    """A fake basal server that does not answer the lines in `down` (by their number): a connection error, or an HTTP
    `status` (502 / 503 / 504) with a body that is not JSON, as a starting server or a proxy gives."""

    def __init__(self, down=(), status=None):
        super().__init__()
        self.down, self.status = set(down), status

    def __call__(self, r):
        if r.method == "POST" and int(json.loads(r.content)["state"].split()[1]) in self.down:
            self.posts.append(json.loads(r.content)["state"])
            if self.status:
                return httpx.Response(self.status, text="Starting")
            raise httpx.ConnectError("connection refused")
        return super().__call__(r)


def test_a_server_that_goes_away_stops_the_run_and_the_same_command_resumes(tmp_path, capsys):
    """The server stops answering at line 8: after --stop-after lines it did not answer the run stops (instead of
    failing all the rest in seconds and ending as if done), writes none of them, and keeps the .part; once the server
    is back the same command sends the lines from 8 on, and the output is the one of an uninterrupted run."""
    want = reference(tmp_path)
    write(tmp_path / "in.jsonl", ITEMS)
    with pytest.raises(Exception, match="answered none of the last 8 lines .*connection refused") as e:
        go(tmp_path, Down(down=range(7, 20)), concurrency=4, stop_after=8)
    assert e.type.__name__ == "ServerGone"
    assert [r["id"] for r in read(tmp_path / "out.jsonl.part")] == list(range(7))
    assert not (tmp_path / "out.jsonl").exists()
    assert "the same command goes on from there" in capsys.readouterr().err
    back = Server()
    go(tmp_path, back, concurrency=4)
    assert back.posts == [f"Order {i} arrived damaged." for i in range(7, 20)]
    assert read(tmp_path / "out.jsonl") == want


def test_a_few_unanswered_lines_between_answers_are_error_lines(tmp_path):
    """Fewer than --stop-after lines the server did not answer, between answers, are each an error line and the run
    goes on to the end, as before."""
    write(tmp_path / "in.jsonl", ITEMS)
    go(tmp_path, Down(down={3, 4, 11}), concurrency=4, stop_after=3)
    out = read(tmp_path / "out.jsonl")
    assert [r["id"] for r in out] == list(range(20))
    assert [i for i, r in enumerate(out) if "error" in r] == [3, 4, 11]


def test_a_starting_server_counts_as_not_answering_and_stop_after_0_never_stops(tmp_path):
    """HTTP 503 with a body that is not JSON (a server starting up) is a line the server did not answer ("HTTP 503");
    --stop-after 0 keeps the old behaviour: every such line an error line, and the run ends."""
    write(tmp_path / "in.jsonl", ITEMS)
    with pytest.raises(Exception, match="the last: HTTP 503") as e:
        go(tmp_path, Down(down=range(5, 20), status=503), stop_after=4)
    assert e.type.__name__ == "ServerGone"
    assert [r["id"] for r in read(tmp_path / "out.jsonl.part")] == list(range(5))
    go(tmp_path, Down(down=range(5, 20), status=503), stop_after=0, restart=True)
    out = read(tmp_path / "out.jsonl")
    assert len(out) == 20 and all(r["error"] == "HTTP 503" for r in out[5:])


def test_basal_run_exits_with_the_reason_when_the_server_went_away(tmp_path, monkeypatch, capsys):
    write(tmp_path / "in.jsonl", ITEMS)
    real = httpx.AsyncClient
    monkeypatch.setattr(run_module.httpx, "AsyncClient",
                        lambda **kw: real(**dict(kw, transport=httpx.MockTransport(Down(down=range(20))))))
    monkeypatch.setattr(sys, "argv", ["basal-run", "--input", str(tmp_path / "in.jsonl"), "--output",
                                      str(tmp_path / "out.jsonl"), "--url", URL, "--stop-after", "5"])
    with pytest.raises(SystemExit, match="basal-run: the server answered none of the last 5 lines"):
        run_module.main()
    assert (tmp_path / "out.jsonl.part").read_text() == "" and not (tmp_path / "out.jsonl").exists()
