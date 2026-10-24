"""Measurements of one model from basal-ui: the benchmark preset (latency, throughput and accuracy over the bundled
examples or a custom JSONL set, docs/BENCHMARK_SETS.md), configurable load tests and determinism checks (basal-loadtest
against the model's basal-serve) and engine-mode comparisons (basal-bench, the model loaded in basal-ui's process)."""
import asyncio
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from basal import bench as offline
from basal import loadtest
from basal.bench import DEFAULT_QUESTIONS
from basal.loadtest import body

from .jobs import Cancelled

MIN_ITEMS = 10
WARMUP = 5


def load_items(paths) -> list[dict]:
    """Simple-format items {"state", "question", "options", "gold"?} of each file, in file order, tagged with the
    file's stem; ValueError names a malformed line."""
    items = []
    for path in paths:
        path = Path(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not JSON ({exc})") from exc
            if isinstance(item, dict) and isinstance(item.get("options"), list) and len(item["options"]) >= 2 \
                    and isinstance(item.get("state"), (str, dict)) and isinstance(item.get("question"), str):
                items.append({**item, "_file": path.stem})
    return items


def resolve_set(spec) -> tuple[str, list[str]]:
    """"bundled" or a path to a .jsonl file or a directory of them -> (label, files); ValueError says why not."""
    if spec in (None, "", "bundled"):
        return "bundled", list(DEFAULT_QUESTIONS)
    if not isinstance(spec, str):
        raise ValueError("set must be \"bundled\" or a path")
    path = Path(spec).expanduser()
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path} contains no .jsonl files")
    try:
        items = load_items(files)
    except OSError as exc:
        raise ValueError(f"cannot read {exc.filename}: {exc.strerror}") from exc
    if len(items) < MIN_ITEMS:
        raise ValueError(f"{path}: {len(items)} items with state, question and options; a benchmark set needs at "
                         f"least {MIN_ITEMS} (docs/BENCHMARK_SETS.md)")
    return str(path), [str(f) for f in files]


def _correct(item, response_json):
    """None without gold or an answer, else whether the most probable option is the gold one."""
    gold = item.get("gold")
    if not isinstance(gold, int) or isinstance(gold, bool):
        return None
    try:
        probabilities = response_json["answers"]["q0"]["probabilities"]
        return max(probabilities, key=probabilities.get) == f"option_{gold + 1}"
    except (KeyError, TypeError, ValueError):
        return None


async def run_benchmark(fleet, model_id, paths, label, job, n_seq=30, n_conc=120, concurrency=8) -> dict:
    """Warm-up, n_seq sequential requests (latency), n_conc at `concurrency` (throughput); every first answer to an
    item with `gold` is scored. Requests go through fleet.adecide with source "benchmark": not counted in the
    statistics or the history. Progress and cancellation through the job (basal/ui/jobs.py)."""
    items = load_items(paths)
    if not items:
        raise ValueError(f"{label}: no items")
    n = WARMUP + n_seq + n_conc
    order = [i % len(items) for i in range(n)]  # small sets are reused
    scored: dict[int, bool] = {}
    error_statuses = {}
    progress = job.progress

    def check():
        if job.cancel.is_set():
            raise Cancelled()

    def score(index, response):
        if index in scored or response.status_code != 200:
            return
        result = _correct(items[index], response.json())
        if result is not None:
            scored[index] = result

    async with httpx.AsyncClient(timeout=300) as client:
        async def ask(index):
            check()
            t0 = time.perf_counter()
            response = await fleet.adecide(model_id, body(items[index]), source="benchmark", client=client)
            if response.status_code >= 400:
                key = str(response.status_code)
                error_statuses[key] = error_statuses.get(key, 0) + 1
                if error_statuses[key] == 1:
                    job.log.append(f"HTTP {key}: {response.text[:1000]} [request {response.headers.get('x-request-id', 'unavailable')}]")
            return time.perf_counter() - t0, response

        for done, index in enumerate(order[:WARMUP]):
            progress("warm-up", done, WARMUP)
            _, response = await ask(index)
            response.raise_for_status()
            score(index, response)
        latencies = []
        for done, index in enumerate(order[WARMUP:WARMUP + n_seq]):
            progress("sequential", done, n_seq)
            seconds, response = await ask(index)
            response.raise_for_status()
            latencies.append(seconds * 1000)
            score(index, response)
        semaphore = asyncio.Semaphore(concurrency)
        finished = 0

        async def one(index):
            nonlocal finished
            async with semaphore:
                try:
                    _, response = await ask(index)
                except httpx.HTTPError:
                    return None
                finished += 1
                progress("concurrent", finished, n_conc)
                return index, response

        progress("concurrent", 0, n_conc)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(i) for i in order[WARMUP + n_seq:]])
        wall = time.perf_counter() - t0
    check()  # a cancel during the last in-flight requests must not store a result
    ok, tokens = 0, 0
    for result in results:
        if result is None or result[1].status_code != 200:
            continue
        index, response = result
        ok += 1
        usage = response.json().get("usage") or {}
        tokens += usage.get("input_tokens", 0) or 0
        score(index, response)
    latencies.sort()
    by_file: dict[str, list[bool]] = {}
    for index, correct in scored.items():
        by_file.setdefault(items[index]["_file"], []).append(correct)
    values = list(scored.values())
    return {"model": model_id, "set": label, "items": len(items), "when": time.time(),
            "p50_ms": statistics.median(latencies) if latencies else None,
            "p95_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
            "mean_ms": statistics.fmean(latencies) if latencies else None,
            "decisions_per_s": ok / wall if wall else None, "questions_per_s": ok / wall if wall else None,
            "input_tokens_per_s": tokens / wall if wall else None, "errors": len(results) - ok,
            "error_statuses": error_statuses,
            "accuracy": {"overall": sum(values) / len(values) if values else None,
                         "by_file": {stem: sum(v) / len(v) for stem, v in sorted(by_file.items())}}}


def benchmark_job(job, fleet, model_id, paths, label):
    """The benchmark job's target: run it, keep the result in the statistics file (last BENCHMARKS per model)."""
    result = asyncio.run(run_benchmark(fleet, model_id, paths, label, job))
    if fleet.stats is not None:
        fleet.stats.add_benchmark(model_id, result)
    return result


LOADTEST_DEFAULTS = {"synthetic": None, "long_frac": 0.0, "seed": 0, "questions_per_request": 1, "n_seq": 200,
                     "n_conc": 1000, "concurrency": 32, "early_exit": None, "determinism": 0}


def _int(params, key, lo, hi=None):
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < lo or (hi is not None and value > hi):
        raise ValueError(f"{key} must be an integer" + (f" between {lo} and {hi}" if hi is not None else f" >= {lo}"))
    return value


def loadtest_params(params, kind, early_exit_levels) -> dict:
    """The parameters of a load test or determinism check merged over LOADTEST_DEFAULTS; ValueError names the bad one."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    unknown = set(params) - set(LOADTEST_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameter: {sorted(unknown)[0]}")
    p = {**LOADTEST_DEFAULTS, **params}
    for key in ("n_seq", "n_conc", "concurrency"):
        _int(p, key, 1)
    _int(p, "questions_per_request", 1, 5)
    _int(p, "seed", 0)
    _int(p, "determinism", 0)
    if kind == "determinism" and p["determinism"] < 1:
        p["determinism"] = 50
    if kind == "loadtest":
        p["determinism"] = 0
    if p["synthetic"] is not None:
        _int(p, "synthetic", 50)
    if isinstance(p["long_frac"], bool) or not isinstance(p["long_frac"], (int, float)) or not 0 <= p["long_frac"] <= 1:
        raise ValueError("long_frac must be between 0 and 1")
    if p["early_exit"] is not None and p["early_exit"] not in early_exit_levels:
        raise ValueError(f"early_exit must be one of {early_exit_levels}" if early_exit_levels else
                         "the model has no early exits (basal-serve --mode fast-exit)")
    return p


def loadtest_job(job, fleet, model_id, label, paths, params):
    """basal-loadtest (or --determinism N) against the model's basal-serve; the result is kept with the benchmarks."""
    runner = fleet.ready(model_id)

    def progress(phase, done, total):
        job.check()
        job.progress(phase, done, total)

    a = SimpleNamespace(url=runner.base_url + "/v1/systemone", headers=runner._headers(), questions=paths,
                        progress=progress, out=None, **params)
    if params["determinism"]:
        kind, out = "determinism", asyncio.run(loadtest.determinism(a, loadtest.determinism_bodies(a)))
    else:
        kind, out = "loadtest", asyncio.run(loadtest.run(a))
    out.pop("url", None)
    result = {"kind": kind, "model": model_id, "set": label, "params": params, "when": time.time(), **out}
    if fleet.stats is not None:
        fleet.stats.add_benchmark(model_id, result)
    return result


def comparison_modes(available) -> list[str]:
    """The modes a comparison may run: the available basal-serve modes (basal/ui/options.available_modes) plus the
    basal-bench references eager-fp32, vllm@deterministic and fast-exit@<policy> where their engine is available."""
    modes = [m["mode"] for m in available if m["available"]]
    out = (["eager-fp32"] if "eager" in modes else []) + modes
    if "vllm" in modes:
        out.append("vllm@deterministic")
    if "fast-exit" in modes:
        out += [f"fast-exit@{p}" for p in ("off", "0.999", "0.995", "0.99", "0.98")]
    return out


def comparison_memory(modes, est_memory_bytes) -> int:
    """The memory a comparison holds at its peak: one mode at a time, eager-fp32 at twice the bf16 weights."""
    return (2 if "eager-fp32" in modes else 1) * (est_memory_bytes or 0)


def modes_job(job, fleet, variant, local_dir, modes, n, lat_n, options):
    """basal-bench in basal-ui's process: every mode loaded, measured and freed in turn; the first mode that runs is the
    reference of the agreement, a mode that fails to load or run is kept with its error. The memory reserved by the
    caller is released at the end. Kept with the benchmarks."""
    try:
        md = offline.resolve(str(local_dir), metadata_only=all(
            offline.MODES.get(m.split("@")[0], ("",))[0] in offline.GGUF_KINDS for m in modes))
        qs = offline.load_questions(DEFAULT_QUESTIONS, n)
        fmt = offline.model_format(md)
        gguf = str(Path(local_dir) / variant.gguf_file) if variant.gguf_file else None
        ref, ref_mode, rows = None, None, []
        for i, mode in enumerate(modes):
            job.check()
            job.progress(mode, i, len(modes))
            job.log.append(f"{mode}: loading")
            try:
                row, top = offline.measure_mode(mode, md, qs, fmt, lat_n, ref_mode or mode, ref, None, gguf,
                                                options.get("ollama_model"), options["ollama_url"],
                                                options["llamacpp_url"], options.get("mlx_packed", False))
            except (Exception, SystemExit) as exc:  # noqa: BLE001 — the other modes still run
                rows.append({"mode": mode, "error": str(exc)})
                job.log.append(f"{mode}: {exc}")
                continue
            if ref is None:
                ref, ref_mode = top, mode
            if mode in ("mlx", "mlx-q8"):
                row["mlx_packed"] = options.get("mlx_packed", False)
            rows.append(row)
            job.log.append(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
        job.progress("done", len(modes), len(modes))
        if not any("error" not in r for r in rows):
            raise RuntimeError("; ".join(f"{r['mode']}: {r['error']}" for r in rows))
    finally:
        fleet.release("modes")
    result = {"kind": "modes", "model": variant.id, "set": "bundled", "items": len(qs), "when": time.time(),
              "modes": rows}
    if fleet.stats is not None:
        fleet.stats.add_benchmark(variant.id, result)
    return result
