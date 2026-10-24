import threading

import pytest

from basal.ui.stats import Stats


@pytest.fixture
def stats(tmp_path):
    s = Stats(tmp_path / "stats.json")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))


def test_best_answer_probability_per_question_type(stats):
    stats.record("m", {"answers": {
        "c": {"type": "choice", "choice": "a", "confidence": 0.8},
        "n": {"type": "noul", "noul": 0.1, "confidence": 0.9},
        "s": {"type": "multi", "selected": ["x"], "set_confidence": 0.5},
        "t": {"type": "tool", "tool": "search", "probabilities": {"search": 0.6, "none": 0.4}},
        "u": {"type": "choice", "choice": "b"}}, "usage": {"input_tokens": 30}}, 10.0)
    stats.record("m", {"answers": [{"type": "predicate", "name": "p", "probability": 0.2}]}, 20.0)
    m = stats.snapshot()["models"]["m"]
    assert (m["requests"], m["questions"], m["errors"], m["input_tokens"]) == (2, 6, 0, 30)
    assert m["avg_best_prob"] == pytest.approx((0.8 + 0.9 + 0.5 + 0.6 + 0.8) / 5)
    assert m["by_type"]["choice"] == {"questions": 2, "avg_best_prob": pytest.approx(0.8)}  # the answer without one
    assert m["by_type"]["predicate"]["avg_best_prob"] == pytest.approx(0.8)
    assert m["avg_latency_ms"] == 15.0


def test_errors_and_latency_percentiles(stats):
    for ms in range(1, 101):
        stats.record("m", {"answers": {}}, float(ms))
    stats.record("m", {"error": "bad request"}, 1.0, ok=False)
    m = stats.snapshot()["models"]["m"]
    assert (m["requests"], m["errors"], m["avg_best_prob"]) == (101, 1, None)
    assert (m["p50_ms"], m["p95_ms"]) == (50.0, 95.0)


def test_persistence_round_trip_and_reset_of_one_model(tmp_path, stats):
    stats.record("a", {"answers": {"q": {"type": "noul", "confidence": 0.7}}}, 5.0)
    stats.record("b", {"answers": {}}, 5.0)
    stats.add_benchmark("a", {"p50_ms": 1.0})
    stats.flush()
    again = Stats(tmp_path / "stats.json")
    assert again.snapshot() == stats.snapshot()
    again.reset("a")
    assert list(again.snapshot()["models"]) == ["b"]
    assert again.snapshot()["benchmarks"] == {"a": [{"p50_ms": 1.0}]}  # benchmark results are not statistics
    again.close()
    assert list(Stats(tmp_path / "stats.json").snapshot()["models"]) == ["b"]


def test_benchmark_results_keep_the_last_twenty(stats):
    for i in range(22):
        stats.add_benchmark("a", {"run": i})
    assert [r["run"] for r in stats.snapshot()["benchmarks"]["a"]] == list(range(2, 22))


def test_corrupt_file_starts_empty(tmp_path):
    (tmp_path / "stats.json").write_text("{not json")
    s = Stats(tmp_path / "stats.json")
    assert s.snapshot() == {"models": {}, "benchmarks": {}}
    s.close()


def test_two_instances_on_one_file_add_up(tmp_path):
    a, b = Stats(tmp_path / "stats.json"), Stats(tmp_path / "stats.json")
    a.record("m", {"answers": {"q": {"confidence": 0.5}}}, 10.0)
    b.record("m", {"answers": {"q": {"confidence": 1.0}}}, 30.0)
    b.record("n", {"answers": {}}, 1.0)
    a.flush()
    b.flush()
    a.reset("n")  # removes n in the file, keeps m's merged counts
    b.add_benchmark("m", {"p50_ms": 1.0})
    merged = Stats(tmp_path / "stats.json").snapshot()
    assert merged["models"]["m"]["requests"] == 2 and merged["models"]["m"]["avg_best_prob"] == 0.75
    assert "n" not in merged["models"] and merged["benchmarks"] == {"m": [{"p50_ms": 1.0}]}
    for s in (a, b):
        s.close()


def test_record_during_reset_remains_in_delta_and_is_not_lost(stats, monkeypatch):
    from basal.ui import settings

    write = settings._write_private
    started = threading.Event()
    workers = []

    def record_before_persist(path, text):
        if not workers:
            def record():
                started.set()
                stats.record("m", {"answers": {}}, 2.0)
            worker = threading.Thread(target=record)
            workers.append(worker)
            worker.start()
            assert started.wait(1)
        write(path, text)

    monkeypatch.setattr(settings, "_write_private", record_before_persist)
    stats.reset()
    workers[0].join(timeout=1)
    assert not workers[0].is_alive()
    assert stats.snapshot()["models"]["m"]["requests"] == 1
    stats.flush()
    assert stats.snapshot()["models"]["m"]["requests"] == 1


def test_snapshot_reads_another_instances_flushed_counters(tmp_path):
    a, b = Stats(tmp_path / "shared.json"), Stats(tmp_path / "shared.json")
    b.record("m", {"answers": {}}, 1.0)
    b.flush()
    assert a.snapshot()["models"]["m"]["requests"] == 1
    a.close()
    b.close()
