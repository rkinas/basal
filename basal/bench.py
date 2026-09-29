"""Offline benchmark of a serving mode (no HTTP): latency, throughput, agreement with an fp32 reference, accuracy.

  basal-bench --model Remek/basal-1.0-4.5B --modes eager-fp32 fast fp8              # bundled examples
  basal-bench --model Remek/basal-1.0-4.5B --modes eager-fp32 fast --questions my_items.jsonl

Questions file (JSONL): {"state": ..., "question": ..., "options": [...], "gold": <index, optional>}.
Measured per mode (same definitions as in the technical report):
  lat1_ms  median latency of one decision with ONE option order at batch size 1
  lat2_ms  median latency of one decision with BOTH option orders at batch size 1 (what the server does by default)
  dec_s    two-order decisions per second when 32 option-order passes are processed together
  agree    share of decisions whose top option equals the first mode's (use eager-fp32 first as the reference)
  acc      accuracy of the averaged two-order decision (if gold is given)
"""
import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

from .engine import EagerBackend, ExitGraphBackend, GraphBackend, SharedBackend, VLLMBackend, default_device, resolve, sync
from .prompt import letter_ids, render
from .server import MODES


def build(mode, md, vllm_model=None):
    if mode == "eager-fp32":
        return EagerBackend(md, "float32")
    if mode.startswith("fast-exit@"):
        return ExitGraphBackend(md, "bfloat16", compile=True, default_policy=mode.split("@")[1])
    if mode.startswith("mps-"):  # mps-float16 etc.: the mps mode in another dtype
        return SharedBackend(md, mode.split("-", 1)[1])
    kind, quant, comp = MODES[mode]
    if kind == "eager":
        return EagerBackend(md, "bfloat16")
    if kind == "shared":
        return SharedBackend(md, "bfloat16")
    if kind == "vllm":
        return VLLMBackend(vllm_model or md)
    if kind == "exit":
        return ExitGraphBackend(md, "bfloat16", quant, compile=comp)
    return GraphBackend(md, "bfloat16", quant, compile=comp, shared=True)


EXAMPLES = Path(__file__).resolve().parent / "examples"  # installed with the package
DEFAULT_QUESTIONS = [str(EXAMPLES / f) for f in ("questions.jsonl", "choice.jsonl", "noul.jsonl", "score.jsonl")]


def load_questions(paths, n, seed=0):
    """Simple-format items {"state", "question", "options", "gold"?} from one or more JSONL files (full-request lines
    with "questions" are skipped)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    qs = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    qs = [q for q in qs if "options" in q]
    random.Random(seed).shuffle(qs)
    return qs[:n]


def groups_for(tok, qs):
    out = []
    for q in qs:
        k = len(q["options"])
        perms = [list(range(k)), list(range(k))[::-1]]
        prompts = [render(tok, q["state"], q["question"], [q["options"][c] for c in p]) for p in perms]
        ids = letter_ids(tok, prompts[0], k)
        if ids:
            out.append((q, perms, prompts, [ids, ids]))
    return out


def decisions(groups, res):
    out = []
    for (q, perms, _, _), probs in zip(groups, res):
        canon = [0.0] * len(q["options"])
        for perm, p in zip(perms, probs):
            for k, j in enumerate(perm):
                canon[j] += p[k] / len(perms)
        out.append(canon)
    return out


def cold(be):
    """vLLM keeps finished prompts in its prefix cache; a repeated prompt would be answered almost for free. Clear it so
    that every timed call computes its prompt (the second option order may still reuse the first order's prefix, as in
    real serving)."""
    if hasattr(be, "llm"):
        be.llm.reset_prefix_cache()


def timed(fn, reps, be=None):
    dev = getattr(be, "dev", default_device())
    lat = []
    for _ in range(reps):
        cold(be)
        sync(dev); t = time.perf_counter()
        fn()
        sync(dev); lat.append(time.perf_counter() - t)
    return lat


def device_name(dev):
    if dev == "cuda":
        return torch.cuda.get_device_name(0)
    if dev == "mps":
        import platform
        import subprocess
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            return f"mps ({platform.machine()})"
    return "cpu"


def reset_memory(dev):
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    elif dev == "mps":
        torch.mps.empty_cache()


def peak_memory_gb(dev):
    """CUDA: peak allocated. MPS has no peak counter: memory held by the Metal driver at the end of the run."""
    if dev == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return float("nan")


def measure(be, groups, lat_n):
    res = []
    for i in range(0, len(groups), 32):
        res += be.run_shared([(g[2], g[3]) for g in groups[i: i + 32]])
    dec = decisions(groups, res)
    one, two = [], []
    for g in groups[: lat_n + 5]:
        be.run([g[2][0]], [g[3][0]]); be.run_shared([(g[2], g[3])])  # warm-up of the shapes
        one += timed(lambda: be.run([g[2][0]], [g[3][0]]), 1, be)
        two += timed(lambda: be.run_shared([(g[2], g[3])]), 1, be)
    one, two = sorted(one[5:]), sorted(two[5:])
    chunk = groups[: 16 * 8]
    be.run_shared([(g[2], g[3]) for g in chunk[:16]])
    t = sum(timed(lambda: [be.run_shared([(g[2], g[3]) for g in chunk[i: i + 16]]) for i in range(0, len(chunk), 16)], 1, be))
    return dec, dict(lat1_ms=statistics.median(one) * 1000, lat2_ms=statistics.median(two) * 1000,
                     lat2_p95_ms=two[int(0.95 * len(two))] * 1000, dec_s=len(chunk) / t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Remek/basal-1.0-4.5B")
    ap.add_argument("--vllm-model", dest="vllm_model", default=None, help="checkpoint for --modes vllm (e.g. the NVFP4 repo)")
    ap.add_argument("--modes", nargs="+", default=["eager-fp32", "fast"],
                    help="eager-fp32, eager, fast, fast-nocompile, fp8, nvfp4, vllm, fast-exit@off|0.999|0.995|0.99|0.98; "
                         "Apple Silicon: eager-fp32, eager, mps, mps-float16")
    ap.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS,
                    help="JSONL file(s) with simple items (default: the bundled examples, 44 items)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lat-n", dest="lat_n", type=int, default=100)
    ap.add_argument("--out", default=None, help="write results as JSON")
    a = ap.parse_args()
    md = resolve(a.model)
    vm = resolve(a.vllm_model) if a.vllm_model else None
    qs = load_questions(a.questions, a.n)
    ref, rows = None, []
    dev = default_device()
    for mode in a.modes:
        reset_memory(dev)
        t0 = time.time()
        be = build(mode, md, vm)
        load_s = time.time() - t0
        groups = groups_for(be.tok, qs)
        dec, r = measure(be, groups, min(a.lat_n, len(groups) - 5))
        top = [max(range(len(d)), key=d.__getitem__) for d in dec]
        r = dict(mode=mode, gpu=device_name(dev), n=len(groups), load_s=round(load_s, 1), **r, mem_gb=peak_memory_gb(dev))
        if all("gold" in g[0] for g in groups):
            r["acc"] = sum(t == g[0]["gold"] for t, g in zip(top, groups)) / len(groups)
        if ref is None:
            ref = top
        else:
            r["agree"] = sum(x == y for x, y in zip(top, ref)) / len(ref)
        rows.append(r)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        if hasattr(be, "llm"):
            del be.llm
        del be; reset_memory(dev)
    print(f"\n{'mode':<18}{'lat2 ms':>9}{'lat1 ms':>9}{'dec/s':>8}{'agree':>8}{'acc':>7}{'GB':>6}")
    for r in rows:
        print(f"{r['mode']:<18}{r['lat2_ms']:>9.1f}{r['lat1_ms']:>9.1f}{r['dec_s']:>8.1f}"
              f"{r.get('agree', float('nan')):>8.3f}{r.get('acc', float('nan')):>7.3f}{r['mem_gb']:>6.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
