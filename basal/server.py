"""HTTP server for typed decisions (System One-compatible JSON interface).

  POST /v1/systemone   {"state": "...", "questions": {"q1": {"type": "choice"|"noul"|"score", "instructions": "...",
                        "criteria": {...} | [...], "option_keys": "show"|"hide" (optional, default "show")}},
                        "early_exit": "off"|"0.99"|... (optional)}
  GET  /v1/models
  GET  /health

  basal-serve --model Remek/basal-1.0-4.5B --mode fast --port 8000
"""
import argparse
import asyncio
import json
import time

import torch

from .engine import (EagerBackend, ExitGraphBackend, GGUFBackend, GraphBackend, MLXBackend, MPSBackend, VLLMBackend,
                     resolve)
from .ollama import OllamaBackend
from .prompt import MAX_OPTIONS, lang_of, letter_ids, render

RELEASE_DATE = "2026-10-01"

MODES = {
    # mode: (backend, quantisation, compile)
    "eager": ("eager", None, False),        # reference PyTorch forward, any GPU (CUDA, Apple MPS) or CPU
    "fast": ("graph", None, True),          # bf16 + torch.compile + CUDA graphs + shared prefix (recommended)
    "fast-nocompile": ("graph", None, False),  # same without torch.compile (faster start-up, ~1.4x slower on H100)
    "fast-exit": ("exit", None, True),      # "fast" + trained early exits, policy chosen per request
    "fp8": ("graph", "fp8", True),          # "fast" with torchao FP8 (Hopper / Blackwell)
    "nvfp4": ("graph", "nvfp4", True),      # "fast" with torchao NVFP4 (Blackwell, experimental)
    "vllm": ("vllm", None, False),          # vLLM, for the ModelOpt FP8 / NVFP4 checkpoints
    "mlx": ("mlx", None, False),            # Apple Silicon: MLX bf16 + shared prefix (recommended on Mac)
    "mlx-q8": ("mlx", "q8", False),         # "mlx" with 8-bit weights (less memory, not faster)
    "mps": ("mps", None, False),            # Apple Silicon: PyTorch MPS + shared prefix, no graphs
    "gguf": ("gguf", None, False),          # llama.cpp on a GGUF file (--gguf; Metal, CUDA or CPU) + shared prefix
    "ollama": ("ollama", None, False),     # Ollama safetensors import; raw prompts + next-token logprobs
}


def default_mode():
    """fast on CUDA, MLX on Apple Silicon when mlx and mlx-lm are installed, MPS otherwise."""
    if torch.cuda.is_available():
        return "fast"
    if torch.backends.mps.is_available():
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            return "mps"
    return "eager"


def validate_quant(mode, quant):
    """Reject quantisation overrides that the selected backend cannot apply."""
    if quant is None:
        return
    kind = MODES[mode][0]
    if quant == "q8" and kind == "mlx":
        return
    if quant in ("fp8", "nvfp4") and kind in ("graph", "exit"):
        return
    supported = "q8 only for MLX; fp8 and nvfp4 only for graph and exit modes"
    raise SystemExit(f"--quant {quant} is not supported with --mode {mode} ({supported})")


def _text(x):
    """Strings as they are; structured values (objects, lists, numbers) as compact JSON."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def named_options(crit, option_keys="show"):
    """{key: description} -> option texts shown to the model.

    option_keys="show" (default): every described option is shown as "key: description", because only the caller knows
    whether a key is meaningful (a service code "B", a decision name "approve"); the server never guesses. A key without
    a description is shown by itself.
    option_keys="hide": the caller states that the keys are placeholders and the descriptions alone define the options
    (the format of the training data and of our PL/EN evaluation clients); descriptions that are not unique within the
    question still get their key, otherwise the options could not be told apart."""
    if option_keys not in ("show", "hide"):
        raise ValueError(f'option_keys must be "show" or "hide", got {option_keys!r}')
    texts = [str(k) if v is None else _text(v) for k, v in crit.items()]
    dup = {t for t in texts if texts.count(t) > 1}
    shown = [t if v is None or (option_keys == "hide" and t not in dup) else f"{k}: {t}"
             for (k, v), t in zip(crit.items(), texts)]
    return list(crit), shown


def to_items(state, questions):
    """Questions of a request -> readout items (options as a list) + keys to assemble the answer."""
    state = _text(state)
    out = []
    for name, q in questions.items():
        t = q.get("type", "choice")
        instr = _text(q.get("instructions", ""))
        lang = lang_of(state + instr)
        if t == "noul":  # yes/no; optional criteria {"true": ..., "false": ...}; answer = P(true)
            crit = q.get("criteria") or {}
            keys = ["true", "false"]
            opts = [_text(crit.get("true") or ("Tak" if lang == "pl" else "Yes")),
                    _text(crit.get("false") or ("Nie" if lang == "pl" else "No"))]
        elif t == "score":  # ordered levels (list or {key: description})
            crit = q.get("criteria") or q.get("levels") or []
            if isinstance(crit, dict):
                keys, opts = named_options(crit, q.get("option_keys", "show"))
            else:
                keys, opts = [str(i) for i in range(len(crit))], [_text(v) for v in crit]
        else:  # choice: {key: description} or [keys]
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {k: None for k in crit}
            keys, opts = named_options(crit, q.get("option_keys", "show"))
        if not 2 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"question {name!r}: {len(opts)} options (supported: 2..{MAX_OPTIONS})")
        out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=opts, lang=lang))
    return out


def models_payload(name, mode, policies):
    """GET /v1/models in the official ModelMetadataList shape, plus the serving mode and early-exit levels."""
    return {"models": [{"name": name, "release_date": RELEASE_DATE,
                        "description": "basal typed-decision model (choice / noul / score), Polish and English",
                        "mode": mode, "early_exit": sorted(policies)}]}


class Server:
    def __init__(self, a):
        validate_quant(a.mode, a.quant)
        if a.mode == "ollama" and not a.ollama_model:
            raise SystemExit("--mode ollama needs --ollama-model <name> (an Ollama safetensors import of --model)")
        kind, quant, comp = MODES[a.mode]
        md = resolve(a.model, a.revision, weights=kind not in ("gguf", "ollama"))
        self.name = a.name or a.model.rstrip("/").split("/")[-1]
        quant = a.quant or quant
        if kind == "eager":
            self.backend = EagerBackend(md, a.dtype)
        elif kind == "exit":
            self.backend = ExitGraphBackend(md, a.dtype, quant, compile=comp, heads_dir=a.exit_heads,
                                            default_policy=a.early_exit)
        elif kind == "vllm":
            self.backend = VLLMBackend(md, a.dtype, mem=a.gpu_memory)
        elif kind == "mlx":
            self.backend = MLXBackend(md, a.dtype, quant)
        elif kind == "mps":
            self.backend = MPSBackend(md, a.dtype, shared=a.orders == 2)
        elif kind == "gguf":
            if not a.gguf:
                raise SystemExit("--mode gguf needs --gguf <file.gguf> (--model gives tokenizer and calibration)")
            self.backend = GGUFBackend(md, a.gguf)
        elif kind == "ollama":
            self.backend = OllamaBackend(md, a.ollama_model, a.ollama_url)
        else:
            self.backend = GraphBackend(md, a.dtype, quant, compile=comp, shared=a.orders == 2)
        self.tok = self.backend.tok
        cal = md / "CALIBRATION.json"
        self.temps = {} if a.no_calibration or not cal.exists() else json.loads(cal.read_text()).get("temperature_per_prim", {})
        self.orders = a.orders
        self.letters = {}
        self.queue, self.max_batch, self.wait = asyncio.Queue(), a.max_batch, a.wait_ms / 1000

    def jobs_for(self, q):
        k = len(q["options"])
        jobs = []
        for perm in ([list(range(k)), list(range(k))[::-1]][: self.orders]):
            prompt = render(self.tok, q["state"], q["question"], [q["options"][c] for c in perm], q["lang"])
            if k not in self.letters:  # letter ids at the answer position depend only on the number of options
                self.letters[k] = letter_ids(self.tok, prompt, k)
            jobs.append((perm, prompt, self.letters[k]))
        return jobs

    async def worker(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            # adaptive batching: take everything already waiting (requests queue up while the GPU is busy); wait
            # extra time only if --wait-ms > 0, so an idle server answers immediately
            while len(batch) < self.max_batch and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            t_end = loop.time() + self.wait
            while self.wait > 0 and len(batch) < self.max_batch:
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), max(0.0, t_end - loop.time())))
                except asyncio.TimeoutError:
                    break
            try:
                probs = [None] * len(batch)
                for pol in {b[3] for b in batch}:  # requests with the same early-exit policy run together
                    ix = [i for i, b in enumerate(batch) if b[3] == pol]
                    out = await loop.run_in_executor(None, self.backend.run_shared,
                                                     [(batch[i][0], batch[i][1]) for i in ix], pol)
                    for i, o in zip(ix, out):
                        probs[i] = o
                for b, p in zip(batch, probs):
                    b[2].set_result(p)
            except Exception as e:  # noqa: BLE001
                for b in batch:
                    if not b[2].done():
                        b[2].set_exception(e)

    async def decide(self, body):
        t0 = time.perf_counter()
        pol = body.get("early_exit")
        if pol is not None:
            pol = str(pol)
            allowed = sorted(getattr(self.backend, "policies", {}))
            if pol not in allowed:
                raise ValueError(f"early_exit={pol!r} not available (server mode must be fast-exit); allowed: {allowed}")
        qs = to_items(body["state"], body["questions"])
        loop = asyncio.get_running_loop()
        pend, n_tok = [], 0
        for q in qs:  # one queue entry = all option orders of one question (shared prefix)
            jobs = self.jobs_for(q)
            if any(ids is None for _, _, ids in jobs):
                raise ValueError("option letters are not single tokens for this tokenizer")
            fut = loop.create_future()
            n_tok += sum(len(self.tok(p, add_special_tokens=False).input_ids) for _, p, _ in jobs)
            await self.queue.put(([p for _, p, _ in jobs], [ids for _, _, ids in jobs], fut, pol))
            pend.append((q, [perm for perm, _, _ in jobs], fut))
        answers = {}
        for q, perms, fut in pend:
            canon = []
            for perm, p_pos in zip(perms, await fut):
                c = [0.0] * len(perm)
                for k, j in enumerate(perm):
                    c[j] = p_pos[k]
                canon.append(c)
            p = torch.tensor([sum(x) / len(canon) for x in zip(*canon)])
            T = self.temps.get(q["type"], 1.0)  # calibrated temperature per question type
            if T != 1.0:
                p = torch.softmax(torch.log(p.clamp_min(1e-12)) / T, -1)
            pl = p.tolist()
            probs = {k: float(v) for k, v in zip(q["keys"], pl)}
            conf = float(max(pl))
            if q["type"] == "noul":
                ans = {"type": "noul", "noul": probs["true"], "probabilities": probs, "confidence": conf}
            elif q["type"] == "score":
                ans = {"type": "score", "score": float(sum(i * v for i, v in enumerate(pl))),
                       "legend": dict(zip(q["keys"], q["options"])), "probabilities": probs, "confidence": conf}
            else:
                ans = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
            answers[q["name"]] = ans
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": n_tok, "output_tokens": 0, "questions": len(qs),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}


def parser():
    ap = argparse.ArgumentParser(description="basal typed-decision server")
    ap.add_argument("--model", default="Remek/basal-1.0-4.5B", help="local directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None, help="model name reported in responses (default: last part of --model)")
    ap.add_argument("--mode", choices=list(MODES), default=None,
                    help="default: fast on CUDA, mlx on Apple Silicon (mps without mlx or mlx-lm), eager otherwise")
    ap.add_argument("--quant", choices=["fp8", "nvfp4", "q8"], default=None,
                    help="override the quantisation of the mode (fp8 / nvfp4: CUDA modes, q8: mlx)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--orders", type=int, choices=[1, 2], default=2,
                    help="2 = ask in original and reversed option order and average (default, reduces order sensitivity)")
    ap.add_argument("--early-exit", dest="early_exit", default="off", help="default policy for --mode fast-exit")
    ap.add_argument("--exit-heads", dest="exit_heads", default=None, help="exit heads dir (default: <model>/exit_heads)")
    ap.add_argument("--no-calibration", dest="no_calibration", action="store_true")
    ap.add_argument("--gpu-memory", dest="gpu_memory", type=float, default=0.6, help="vLLM memory fraction")
    ap.add_argument("--gguf", default=None, help="GGUF weights for --mode gguf (converted from --model, see docs/GGUF.md)")
    ap.add_argument("--ollama-model", default=None, help="Ollama safetensors import name for --mode ollama")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama API root for --mode ollama")
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=64)
    ap.add_argument("--wait-ms", dest="wait_ms", type=float, default=0.0,
                    help="extra time to wait for more requests before a forward (default 0: adaptive batching)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def main():
    a = parser().parse_args()
    a.mode = a.mode or default_mode()
    validate_quant(a.mode, a.quant)
    from contextlib import asynccontextmanager

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    t0 = time.time()
    srv = Server(a)
    print(f"basal: model {a.model} mode {a.mode} ready in {time.time() - t0:.0f}s on port {a.port}", flush=True)

    async def systemone(request):
        try:
            return JSONResponse(await srv.decide(await request.json()))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=422)

    async def models(_):
        return JSONResponse(models_payload(srv.name, a.mode, getattr(srv.backend, "policies", {})))

    async def health(_):
        return JSONResponse({"status": "ok"})

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.get_running_loop().create_task(srv.worker())
        yield
        task.cancel()

    app = Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]), Route("/v1/models", models),
                            Route("/health", health)], lifespan=lifespan)
    try:  # uvloop + httptools when available (uvicorn[standard]); plain asyncio otherwise
        import httptools  # noqa: F401
        import uvloop  # noqa: F401
        fast = dict(loop="uvloop", http="httptools")
    except ImportError:
        fast = {}
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **fast)


if __name__ == "__main__":
    main()
