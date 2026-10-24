"""Per-model statistics of the decisions basal-ui answered (playground, API, MCP) and the benchmark results, kept in
<settings stem>.stats.json next to the settings file (merged under the settings lock: several basal-ui processes on
one settings file add up)."""
import json
import threading
import time
from pathlib import Path

SAMPLES = 1000  # latencies kept per model for the percentiles
BENCHMARKS = 20  # results kept per model (benchmarks, load tests, determinism checks, mode comparisons)
FLUSH_S = 5


def answer_items(answers) -> list[tuple[str, dict]]:
    """(name, answer) pairs of /v1/systemone answers (a dict by name) or /v1/decisions answers (a list)."""
    if isinstance(answers, dict):
        return [(name, a) for name, a in answers.items() if isinstance(a, dict)]
    if isinstance(answers, list):
        return [(str(a.get("name", i)), a) for i, a in enumerate(answers) if isinstance(a, dict)]
    return []


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def best_probability(answer: dict):
    """The probability of the returned answer: confidence (choice, noul, score, act), set_confidence (multi), the
    larger side of a predicate's probability, else the largest probability (tool); None when there is none."""
    for key in ("confidence", "set_confidence"):
        if _number(answer.get(key)):
            return float(answer[key])
    if _number(answer.get("probability")):
        return max(float(answer["probability"]), 1 - float(answer["probability"]))
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        values = [v for v in probabilities.values() if _number(v)]
        if values:
            return float(max(values))
    return None


def _percentile(samples, q):
    if not samples:
        return None
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _empty():
    return {"requests": 0, "questions": 0, "errors": 0, "input_tokens": 0, "latency_ms_sum": 0.0,
            "latency_samples": [], "best_prob_sum": 0.0, "best_prob_n": 0, "first_used": None, "last_used": None,
            "by_type": {}}


def _read(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded.get("models"), dict) and isinstance(loaded.get("benchmarks"), dict):
            return loaded
    except (OSError, ValueError, AttributeError):
        pass
    return {"models": {}, "benchmarks": {}}


def _merge(into: dict, delta: dict) -> None:
    """Add the counters of delta (one model's record) to into."""
    for key in ("requests", "questions", "errors", "input_tokens", "latency_ms_sum", "best_prob_sum", "best_prob_n"):
        into[key] += delta[key]
    into["latency_samples"] = [*into["latency_samples"], *delta["latency_samples"]][-SAMPLES:]
    into["first_used"] = min(filter(None, (into["first_used"], delta["first_used"])), default=None)
    into["last_used"] = max(filter(None, (into["last_used"], delta["last_used"])), default=None)
    for kind, t in delta["by_type"].items():
        target = into["by_type"].setdefault(kind, {"questions": 0, "best_prob_sum": 0.0, "best_prob_n": 0})
        for key in target:
            target[key] += t[key]


class Stats:
    """Counts go into `delta` and are merged into the file under the settings lock at every flush, so two basal-ui
    processes on one settings file add up instead of overwriting each other; `data` mirrors the file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.data = _read(self.path)
        self.delta: dict = {}
        self._stop = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    def record(self, model_id: str, response_json, latency_ms: float, ok: bool = True) -> None:
        now = time.time()
        with self.lock:
            entry = self.delta.setdefault(model_id, _empty())
            entry["first_used"] = entry["first_used"] or now
            entry["last_used"] = now
            entry["requests"] += 1
            entry["latency_ms_sum"] += latency_ms
            entry["latency_samples"] = [*entry["latency_samples"], round(latency_ms, 2)][-SAMPLES:]
            if not ok or not isinstance(response_json, dict):
                entry["errors"] += 1
                return
            usage = response_json.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
                entry["input_tokens"] += usage["input_tokens"]
            for _name, answer in answer_items(response_json.get("answers")):
                kind = answer.get("type") if isinstance(answer.get("type"), str) else "other"
                by_type = entry["by_type"].setdefault(kind, {"questions": 0, "best_prob_sum": 0.0, "best_prob_n": 0})
                entry["questions"] += 1
                by_type["questions"] += 1
                p = best_probability(answer)
                if p is not None:
                    for target in (entry, by_type):
                        target["best_prob_sum"] += p
                        target["best_prob_n"] += 1

    def _update_file(self, change) -> None:
        """Under the settings lock: read the file, merge this process's delta, apply change(data), write, mirror."""
        from . import settings

        def go():
            with self.lock:
                delta, self.delta = self.delta, {}
            data = _read(self.path)
            try:
                for model_id, entry in delta.items():
                    _merge(data["models"].setdefault(model_id, _empty()), entry)
                change(data)
                settings._write_private(self.path, json.dumps(data))
            except BaseException:
                with self.lock:  # keep the counts for the next flush
                    for model_id, entry in delta.items():
                        _merge(self.delta.setdefault(model_id, _empty()), entry)
                raise
            with self.lock:
                self.data = data

        with self.write_lock:
            settings._locked(go)()

    def add_benchmark(self, model_id: str, result: dict) -> None:
        def change(data):
            runs = data["benchmarks"].get(model_id, [])
            data["benchmarks"][model_id] = [*runs, result][-BENCHMARKS:]
        self._update_file(change)

    def snapshot(self) -> dict:
        from . import settings

        def read():
            data = _read(self.path)
            with self.lock:
                self.data = data
                merged = json.loads(json.dumps(data))
                for model_id, entry in self.delta.items():
                    _merge(merged["models"].setdefault(model_id, _empty()), entry)
            return merged

        merged = settings._locked(read)()
        def avg(total, n):
            return total / n if n else None

        models = {}
        for model_id, e in merged["models"].items():
            models[model_id] = {
                **{k: e[k] for k in ("requests", "questions", "errors", "input_tokens", "first_used", "last_used")},
                "avg_latency_ms": avg(e["latency_ms_sum"], e["requests"]),
                "p50_ms": _percentile(e["latency_samples"], 0.5),
                "p95_ms": _percentile(e["latency_samples"], 0.95),
                "avg_best_prob": avg(e["best_prob_sum"], e["best_prob_n"]),
                "by_type": {kind: {"questions": t["questions"],
                                   "avg_best_prob": avg(t["best_prob_sum"], t["best_prob_n"])}
                            for kind, t in e["by_type"].items()}}
        return {"models": models, "benchmarks": merged["benchmarks"]}

    def reset(self, model_id: str | None = None) -> None:
        from . import settings

        def go():
            with self.lock:
                previous = self.delta if model_id is None else {
                    model_id: self.delta.pop(model_id)} if model_id in self.delta else {}
                if model_id is None:
                    self.delta = {}
                try:
                    data = _read(self.path)
                    if model_id is None:
                        data["models"].clear()
                    else:
                        data["models"].pop(model_id, None)
                    settings._write_private(self.path, json.dumps(data))
                except BaseException:
                    for key, entry in previous.items():
                        _merge(self.delta.setdefault(key, _empty()), entry)
                    raise
                self.data = data

        with self.write_lock:
            settings._locked(go)()


    def flush(self) -> None:
        with self.lock:
            if not self.delta:
                return
        self._update_file(lambda data: None)

    def _flush_loop(self):
        while not self._stop.wait(FLUSH_S):
            try:
                self.flush()
            except OSError:
                pass  # the delta is kept; the next flush retries

    def close(self) -> None:
        self._stop.set()
        self.flush()
