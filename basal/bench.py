"""Offline benchmark of serving modes (no HTTP): latency, throughput, agreement with the first mode, accuracy.

  basal-bench --model Remek/basal-1.5-4.5B --modes eager-fp32 fast fp8              # bundled examples
  basal-bench --model Remek/basal-1.5-4.5B --modes eager-fp32 fast --questions my_items.jsonl

Questions file (JSONL): {"state": ..., "question": ..., "options": [...], "gold": <index, optional>}.
Measured per mode (same definitions as in the technical report):
  lat1_ms  median latency of one decision with ONE option order at batch size 1
  lat2_ms  median latency of one decision with BOTH option orders at batch size 1 (what the server does by default)
  dec_s    two-order decisions per second when 32 option-order passes are processed together
  agree    share of decisions whose top option equals the first mode's (default reference: eager-fp32; on Apple
           Silicon bf16 mps, not fp32)
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

from .engine import (EagerBackend, ExitGraphBackend, GGUFBackend, GraphBackend, LlamaCppBackend, MLXBackend,
                     MPSBackend, OllamaBackend, SGLangBackend, VLLMBackend, resolve)
from .prompt import letter_ids, render
from .server import MODES, default_mode

GGUF_KINDS = ("gguf", "ollama", "llamacpp")  # the engine serves GGUF weights: --model gives tokenizer and calibration


def build(mode, md, vllm_model=None, gguf=None, ollama_model=None, ollama_url="http://127.0.0.1:11434",
          llamacpp_url="http://127.0.0.1:8080"):
    if mode == "eager-fp32":
        return EagerBackend(md, "float32")
    if mode.startswith("fast-exit@"):
        return ExitGraphBackend(md, "bfloat16", compile=True, default_policy=mode.split("@")[1])
    if mode == "gguf" or mode.startswith("gguf@"):
        path = mode.split("@", 1)[1] if "@" in mode else gguf
        if not path:
            raise SystemExit("mode gguf needs --gguf <file.gguf> (or gguf@<file.gguf>)")
        return GGUFBackend(md, path)
    if mode in ("tpu", "tpu-fp32"):
        from .tpu import TPUBackend
        return TPUBackend(md, "float32" if mode == "tpu-fp32" else "bfloat16", warm=False)
    kind, quant, comp = MODES[mode]
    if kind == "eager":
        return EagerBackend(md, "bfloat16")
    if kind == "vllm":
        return VLLMBackend(vllm_model or md)
    if kind == "sglang":
        return SGLangBackend(md)
    if kind == "mlx":
        return MLXBackend(md, quant=quant)
    if kind == "mps":
        return MPSBackend(md, "bfloat16")
    if kind == "ollama":
        return OllamaBackend(md, ollama_url, ollama_model)
    if kind == "llamacpp":
        return LlamaCppBackend(md, llamacpp_url)
    if kind == "exit":
        return ExitGraphBackend(md, "bfloat16", quant, compile=comp)
    return GraphBackend(md, "bfloat16", quant, compile=comp, shared=True)


def default_modes():
    """A reference that fits the device, then the serving default: eager-fp32 on CUDA and CPU, bf16 mps on Apple
    Silicon (the fp32 4.5B needs about 18 GB)."""
    mode = default_mode()
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        return [mode] if mode == "mps" else ["mps", mode]
    return ["eager-fp32", mode]


def device_name(be=None):
    if hasattr(be, "device_name"):
        return be.device_name()
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    import platform
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor() or "cpu"


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    try:
        import mlx.core as mx
        mx.clear_cache(); mx.reset_peak_memory()
    except ImportError:
        pass


def memory_gb(be):
    """Peak allocated memory (CUDA, MLX, TPU); for MPS the memory held by the Metal driver at the end of the run; NaN
    for backends without memory tracking (GGUF, Ollama, llama.cpp server, CPU)."""
    if isinstance(be, MLXBackend):
        return be.mx.get_peak_memory() / 2**30
    if hasattr(be, "memory_gb"):
        return be.memory_gb()
    dev = getattr(be, "dev", None)
    if dev is not None:
        kind = torch.device(dev).type
        if kind == "cuda":
            return torch.cuda.max_memory_allocated() / 2**30
        if kind == "mps":
            return torch.mps.driver_allocated_memory() / 2**30
    if hasattr(be, "llm") and torch.cuda.is_available():  # vLLM / SGLang
        return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def sync(be):
    dev = getattr(be, "dev", None)
    if dev is not None:
        kind = torch.device(dev).type
        if kind == "cuda":
            torch.cuda.synchronize()
        elif kind == "mps":
            torch.mps.synchronize()


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
        clear = getattr(be.llm, "reset_prefix_cache", None) or getattr(be.llm, "flush_cache", None)  # vLLM / SGLang
        if clear is not None:
            clear()


def timed(fn, reps, be=None):
    lat = []
    for _ in range(reps):
        cold(be)
        sync(be); t = time.perf_counter()
        fn()
        sync(be); lat.append(time.perf_counter() - t)
    return lat


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
    ap.add_argument("--model", default="Remek/basal-1.5-4.5B")
    ap.add_argument("--vllm-model", dest="vllm_model", default=None, help="checkpoint for --modes vllm (e.g. the NVFP4 repo)")
    ap.add_argument("--gguf", default=None, help="GGUF weights for --modes gguf (several files: gguf@<file> per mode)")
    ap.add_argument("--ollama-model", dest="ollama_model", default=None, help="Ollama model name for --modes ollama")
    ap.add_argument("--ollama-url", dest="ollama_url", default="http://127.0.0.1:11434")
    ap.add_argument("--llamacpp-url", dest="llamacpp_url", default="http://127.0.0.1:8080")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="default: eager-fp32 and the serving default (fast on CUDA); on Apple Silicon bf16 mps and mlx "
                         "(--modes eager-fp32 mps mlx for fp32 agreement if memory allows). Modes: eager-fp32, eager, "
                         "fast, fast-nocompile, fp8, nvfp4, vllm, sglang, fast-exit@off|0.999|0.995|0.99|0.98; Apple "
                         "Silicon: mlx, mlx-q8, mps; GGUF: gguf, gguf@<file.gguf>, ollama, llamacpp; "
                         "Google TPU: tpu, tpu-fp32")
    ap.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS,
                    help="JSONL file(s) with simple items (default: the bundled examples, 44 items)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lat-n", dest="lat_n", type=int, default=100)
    ap.add_argument("--out", default=None, help="write results as JSON")
    a = ap.parse_args()
    a.modes = a.modes or default_modes()
    md = resolve(a.model, metadata_only=all(MODES.get(m.split("@")[0], ("",))[0] in GGUF_KINDS for m in a.modes))
    vm = resolve(a.vllm_model) if a.vllm_model else None
    qs = load_questions(a.questions, a.n)
    ref, rows = None, []
    for mode in a.modes:
        reset_memory()
        t0 = time.time()
        be = build(mode, md, vm, a.gguf, a.ollama_model, a.ollama_url, a.llamacpp_url)
        load_s = time.time() - t0
        groups = groups_for(be.tok, qs)
        dec, r = measure(be, groups, min(a.lat_n, len(groups) - 5))
        top = [max(range(len(d)), key=d.__getitem__) for d in dec]
        r = dict(mode=mode, reference_mode=a.modes[0], gpu=device_name(be), n=len(groups), load_s=round(load_s, 1),
                 **r, mem_gb=memory_gb(be))
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
        del be; gc.collect(); reset_memory()
    print(f"\n{'mode':<18}{'lat2 ms':>9}{'lat1 ms':>9}{'dec/s':>8}{'agree':>8}{'acc':>7}{'GB':>6}")
    for r in rows:
        print(f"{r['mode']:<18}{r['lat2_ms']:>9.1f}{r['lat1_ms']:>9.1f}{r['dec_s']:>8.1f}"
              f"{r.get('agree', float('nan')):>8.3f}{r.get('acc', float('nan')):>7.3f}{r['mem_gb']:>6.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
