"""basal-bench's command line without a model: a fake engine with no memory tracking (as GGUF, Ollama, llama.cpp and
the CPU), too few items, and the exit status of python -m basal.bench --identity."""
import json
import runpy
import sys

import pytest

from basal import bench as bench_module


class FakeEngine:
    tok = None  # no device, no memory tracking: memory_gb is NaN

    def run(self, prompts, ids):
        return [[0.7, 0.3] for _ in prompts]

    def run_shared(self, items):
        return [[[0.7, 0.3], [0.4, 0.6]] for _ in items]


def strict(text):
    def refuse(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(text, parse_constant=refuse)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_module, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr(bench_module, "model_format", lambda md: 1)
    monkeypatch.setattr(bench_module, "build", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(bench_module, "device_name", lambda be=None: "cpu")
    monkeypatch.setattr(bench_module, "groups_for", lambda tok, qs, fmt=1: [
        (q, [[0, 1], [1, 0]], ["p0", "p1"], [[1, 2], [1, 2]]) for q in qs])

    def questions(n):
        f = tmp_path / f"q{n}.jsonl"
        f.write_text("".join(json.dumps({"state": f"s{i}", "question": "?", "options": ["a", "b"], "gold": i % 2})
                             + "\n" for i in range(n)), encoding="utf-8")
        return str(f)
    return questions


def bench(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["basal-bench", "--modes", "eager", *argv])
    return bench_module.main()


def test_out_is_valid_json_when_memory_is_not_measured(fake, tmp_path, monkeypatch, capsys):
    out = tmp_path / "rows.json"
    bench(monkeypatch, "--questions", fake(8), "--lat-n", "2", "--out", str(out))
    rows = strict(out.read_text(encoding="utf-8"))  # NaN was written for mem_gb
    assert rows[0]["mode"] == "eager" and rows[0]["n"] == 8 and rows[0]["mem_gb"] is None
    assert "nan" in capsys.readouterr().out.splitlines()[-1]  # the table still shows it as not measured


@pytest.mark.parametrize("n", [0, 3, 5])
def test_too_few_items_is_a_message_not_a_crash(fake, monkeypatch, n):
    with pytest.raises(SystemExit, match=f"basal-bench: {n} simple items .* at least 6"):
        bench(monkeypatch, "--questions", fake(n))


def test_too_few_items_in_the_prompt_format(fake, monkeypatch):
    monkeypatch.setattr(bench_module, "groups_for", lambda tok, qs, fmt=1: [
        (q, [[0, 1], [1, 0]], ["p0", "p1"], [[1, 2], [1, 2]]) for q in qs[:4]])
    with pytest.raises(SystemExit, match="4 of the 8 items fit"):
        bench(monkeypatch, "--questions", fake(8))


@pytest.mark.parametrize("flag", ["--lat-n", "--n"])
def test_counts_below_one_are_refused(fake, monkeypatch, capsys, flag):
    with pytest.raises(SystemExit) as e:
        bench(monkeypatch, "--questions", fake(8), flag, "0")
    assert e.value.code == 2 and "must be at least 1" in capsys.readouterr().err


@pytest.mark.parametrize("gate, status", [("pass", 0), ("fail", 1)])
def test_python_m_identity_exits_with_the_gate(monkeypatch, gate, status):
    """The Makefile runs python -m basal.bench; its __main__ dropped main()'s status, so a failed gate exited 0."""
    import basal.identity
    monkeypatch.setattr(basal.identity, "main", lambda argv: [{"gate": gate}])
    monkeypatch.setattr(sys, "argv", ["basal.bench", "--identity"])
    monkeypatch.delitem(sys.modules, "basal.bench", raising=False)
    with pytest.raises(SystemExit) as e:
        runpy.run_module("basal.bench", run_name="__main__")
    assert e.value.code == status
