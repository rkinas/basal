"""HTTP server for typed decisions (System One-compatible JSON interface).

  POST /v1/systemone   {"state": "...", "questions": {"q1": {"type": "choice"|"noul"|"score"|"act"|"multi",
                        "instructions": "...", "criteria": {...} | [...], "option_keys": "show"|"hide" (optional),
                        "evidence": true (optional; models with evidence_head.pt: supporting spans of the state)}},
                        "early_exit": "off"|"0.99"|... (optional), "facts": "off"|"auto" (optional)}
  POST /v1/basal       same as /v1/systemone
  GET  /v1/models
  GET  /health

State once, ask many (SOAM): every question of a request, in both option orders (and every label of a `multi`), is one
branch over the state; the state is computed once per request. `act` and `multi` are engine-level types:
  act    the underlying question (criteria = outcomes; {"true", "false"} keys = yes/no) plus "actions" and "costs"
         (cost[action][outcome], or the short form {"wrong": W, "defer": D}); returns the minimum-expected-cost action
  multi  one yes/no branch per label ("does label X apply?"); independent probabilities, the selected set
         (>= "threshold", default 0.5, at most "max", at least "min") and its confidence

  basal-serve --model Remek/basal-1.5-4.5B --mode fast --port 8000
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import torch

from .engine import (EagerBackend, ExitGraphBackend, GGUFBackend, GraphBackend, LlamaCppBackend, MLXBackend,
                     MPSBackend, OllamaBackend, SGLangBackend, VLLMBackend, resolve)
from .evidence import Evidence
from .facts import inject as inject_facts
from .prompt import MAX_OPTIONS, lang_of, letter_ids, render

RELEASE_DATE = "2026-10-05"  # the engine's release date (GET /v1/models), not the served model's

MODES = {
    # mode: (backend, quantisation, compile)
    "eager": ("eager", None, False),        # reference PyTorch forward, any GPU (CUDA, Apple MPS) or CPU
    "fast": ("graph", None, True),          # bf16 + torch.compile + CUDA graphs + shared prefix (recommended)
    "fast-nocompile": ("graph", None, False),  # same without torch.compile (faster start-up, ~1.4x slower on H100)
    "fast-exit": ("exit", None, True),      # "fast" + trained early exits, policy chosen per request
    "fp8": ("graph", "fp8", True),          # "fast" with torchao FP8 (Hopper / Blackwell)
    "nvfp4": ("graph", "nvfp4", True),      # "fast" with torchao NVFP4 (Blackwell, experimental)
    "vllm": ("vllm", None, False),          # vLLM, for the ModelOpt FP8 / NVFP4 checkpoints
    "sglang": ("sglang", None, False),      # SGLang offline engine (1.5)
    "mlx": ("mlx", None, False),            # Apple Silicon, MLX (1.5): the MLX fp8 / fp4 exports or bf16 weights
    "mlx-q8": ("mlx", "q8", False),         # "mlx" with 8-bit weights made at load time (less memory, not faster)
    "mps": ("mps", None, False),            # Apple Silicon: PyTorch MPS + shared prefix, no graphs
    "gguf": ("gguf", None, False),          # llama.cpp in process on a GGUF file (--gguf; Metal, CUDA or CPU)
    "ollama": ("ollama", None, False),      # Ollama with the GGUF exports (1.5); --ollama-url, --ollama-model
    "llamacpp": ("llamacpp", None, False),  # llama.cpp server with the GGUF exports (1.5); --llamacpp-url
    "tpu": ("tpu", None, False),            # Google TPU: JAX / XLA + shared prefix, one executable per shape
}


def default_mode():
    """fast on CUDA, mlx on Apple Silicon when mlx and mlx-lm are installed, mps otherwise there, tpu on a TPU VM
    with JAX, else eager."""
    if torch.cuda.is_available():
        return "fast"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            return "mps"
    from .tpu import tpu_available
    if tpu_available():
        return "tpu"
    return "eager"


def validate_quant(mode, quant):
    """Reject quantisation overrides that the selected backend cannot apply (before any model is loaded)."""
    if quant is None:
        return
    kind = MODES[mode][0]
    if quant == "q8" and kind == "mlx" or quant in ("fp8", "nvfp4") and kind in ("graph", "exit"):
        return
    raise SystemExit(f"--quant {quant} is not supported with --mode {mode} "
                     "(q8 only for mlx; fp8 and nvfp4 only for the CUDA graph modes)")


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


MULTI_Q = {"pl": "{q}\nCzy dotyczy: „{label}”?", "en": "{q}\nDoes this apply: \"{label}\"?"}


def yes_no(lang, crit=None):
    crit = crit or {}
    return (["true", "false"], [_text(crit.get("true") or ("Tak" if lang == "pl" else "Yes")),
                                _text(crit.get("false") or ("Nie" if lang == "pl" else "No"))])


def to_items(state, questions):
    """Questions of a request -> readout items (options as a list) + keys to assemble the answer. Every item has
    `branches` = [(question text, options)]: one for the decision types, one per label for `multi`."""
    state = _text(state)
    out = []
    for name, q in questions.items():
        t = q.get("type", "choice")
        instr = _text(q.get("instructions", ""))
        lang = lang_of(state + instr)
        extra = {}
        if t == "multi":  # one yes/no branch per label
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {k: None for k in crit}
            if not crit:
                raise ValueError(f"question {name!r}: multi needs criteria")
            keys = list(crit)
            labels = [str(k) if v is None else _text(v) for k, v in crit.items()]
            yn = yes_no(lang)[1]
            branches = [(MULTI_Q[lang].format(q=instr, label=lab), yn) for lab in labels]
            extra = dict(threshold=float(q.get("threshold", 0.5)), max=q.get("max"), min=q.get("min"), labels=labels)
            out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=yn, lang=lang,
                            branches=branches, **extra))
            continue
        if t == "act":
            crit = q.get("criteria") or {}
            if isinstance(crit, dict) and set(crit) <= {"true", "false"} or q.get("base") == "noul":
                keys, opts = yes_no(lang, crit if isinstance(crit, dict) else None)
                base = "noul"
            else:
                base = q.get("base", "choice")
                if isinstance(crit, list):
                    crit = {k: None for k in crit}
                keys, opts = named_options(crit, q.get("option_keys", "show"))
            extra = dict(base=base, act=act_spec(name, q, keys))
        elif t == "noul":  # yes/no; optional criteria {"true": ..., "false": ...}; answer = P(true)
            keys, opts = yes_no(lang, q.get("criteria"))
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
        out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=opts, lang=lang,
                        branches=[(instr, opts)], option_keys=q.get("option_keys", "show"), **extra))
    return out


def act_spec(name, q, outcomes):
    """Cost matrix C[action][outcome] from the full form or the short form {"wrong": W, "defer": D}."""
    costs = q.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise ValueError(f"question {name!r}: act needs costs")
    if set(costs) <= {"wrong", "defer"} and not isinstance(costs.get("wrong"), dict):
        w = float(costs["wrong"])
        C = {y: {z: 0.0 if z == y else w for z in outcomes} for y in outcomes}
        if "defer" in costs:
            C["defer"] = {z: float(costs["defer"]) for z in outcomes}
        return dict(costs=C, defer="defer" if "defer" in costs else None, max_error=q.get("max_error"))
    for a, row in costs.items():
        if not isinstance(row, dict) or set(row) != set(outcomes):
            raise ValueError(f"question {name!r}: costs[{a!r}] must give a cost for every outcome {outcomes}")
    C = {a: {z: float(v) for z, v in row.items()} for a, row in costs.items()}
    defer = q.get("defer_action") or next((a for a in ("defer", "human") if a in C), None)
    return dict(costs=C, defer=defer, max_error=q.get("max_error"))


def act_answer(q, probs, conf, cal):
    """Minimum expected cost action; with max_error, automatic actions below the certified threshold are refused."""
    spec = q["act"]
    exp = {a: sum(probs[z] * c for z, c in row.items()) for a, row in spec["costs"].items()}
    action = min(exp, key=exp.get)
    ans = {"type": "act", "action": action, "expected_costs": {a: round(v, 6) for a, v in exp.items()},
           "answer": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
    if spec["max_error"] is not None:
        thr = (cal.get("thresholds") or {}).get(str(spec["max_error"]), {}).get("confidence")
        if thr is None:
            ans["max_error"] = "no certified threshold for this target"
        elif conf < thr and action != spec["defer"]:
            if spec["defer"] is None:
                raise ValueError("max_error needs a defer action (\"defer\"/\"human\" or defer_action)")
            ans["action"], ans["refused"] = spec["defer"], action
    validated = bool(cal.get("temperature_per_prim")) and (q["base"] == "noul" or q.get("option_keys") == "hide")
    ans["calibration"] = "validated" if validated else "unvalidated"
    return ans


def multi_answer(q, p_yes):
    probs = dict(zip(q["keys"], p_yes))
    order = sorted(probs, key=probs.get, reverse=True)
    sel = [k for k in order if probs[k] >= q["threshold"]]
    if q["max"] is not None:
        sel = sel[: int(q["max"])]
    if q["min"] is not None and len(sel) < int(q["min"]):
        sel = order[: int(q["min"])]
    conf = 1.0
    for k, v in probs.items():
        conf *= v if k in sel else 1 - v
    return {"type": "multi", "selected": sel, "probabilities": probs, "threshold": q["threshold"],
            "set_confidence": conf}


def models_payload(name, mode, policies):
    """GET /v1/models in the official ModelMetadataList shape, plus the serving mode and early-exit levels."""
    return {"models": [{"name": name, "release_date": RELEASE_DATE,
                        "description": "basal typed-decision model (choice / noul / score / act / multi), Polish and English",
                        "mode": mode, "early_exit": sorted(policies)}]}


class Server:
    def __init__(self, a):
        validate_quant(a.mode, a.quant)
        kind, quant, comp = MODES[a.mode]
        if kind == "gguf" and not a.gguf:
            raise SystemExit("--mode gguf needs --gguf <file.gguf> (--model gives tokenizer and calibration)")
        # GGUF modes: the engine serves the GGUF weights, so only tokenizer, template and calibration are downloaded
        md = resolve(a.model, a.revision, metadata_only=kind in ("gguf", "ollama", "llamacpp"))
        self.name = a.name or a.model.rstrip("/").split("/")[-1]
        quant = a.quant or quant
        if kind == "eager":
            self.backend = EagerBackend(md, a.dtype, device=a.device)
        elif kind == "exit":
            self.backend = ExitGraphBackend(md, a.dtype, quant, compile=comp, heads_dir=a.exit_heads,
                                            default_policy=a.early_exit)
        elif kind == "vllm":
            self.backend = VLLMBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len)
        elif kind == "sglang":
            self.backend = SGLangBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len)
        elif kind == "mlx":
            self.backend = MLXBackend(md, quant=quant)
        elif kind == "mps":
            self.backend = MPSBackend(md, a.dtype, shared=a.orders == 2)
        elif kind == "gguf":
            self.backend = GGUFBackend(md, a.gguf)
        elif kind == "ollama":
            self.backend = OllamaBackend(md, a.ollama_url, a.ollama_model, parallel=a.http_parallel)
        elif kind == "llamacpp":
            self.backend = LlamaCppBackend(md, a.llamacpp_url, parallel=a.http_parallel)
        elif kind == "tpu":
            from .tpu import TPUBackend
            self.backend = TPUBackend(md, a.dtype, shared=a.orders == 2, max_len=a.max_len)
        else:
            self.backend = GraphBackend(md, a.dtype, quant, compile=comp, shared=a.orders == 2)
        self.tok = self.backend.tok
        model = getattr(self.backend, "model", None)  # evidence head: needs the in-process HF model
        self.evidence = Evidence(md, model, self.tok) if isinstance(model, torch.nn.Module) else None
        cal = md / f"CALIBRATION.{kind}.json"  # 1.5: each engine has its own calibration file when shipped
        cal = cal if kind in ("vllm", "sglang", "mlx", "ollama", "llamacpp") and cal.exists() else md / "CALIBRATION.json"
        cal = Path(a.calibration) if a.calibration else cal
        self.cal = {} if a.no_calibration or not cal.exists() else json.loads(cal.read_text())
        self.temps = self.cal.get("temperature_per_prim", {})
        self.orders = a.orders
        self.soam = a.soam == "on"
        self.log = open(a.log_decisions, "a", buffering=1) if a.log_decisions else None
        self.letters = {}
        self.queue, self.max_batch, self.wait = asyncio.Queue(), a.max_batch, a.wait_ms / 1000

    def jobs_for(self, q, question=None, options=None):
        """Prompts of one branch in every option order: [(perm, prompt, letter ids)]."""
        question = q["question"] if question is None else question
        options = q["options"] if options is None else options
        k = len(options)
        jobs = []
        for perm in ([list(range(k)), list(range(k))[::-1]][: self.orders]):
            prompt = render(self.tok, q["state"], question, [options[c] for c in perm], q["lang"])
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

    def canonical(self, q_type, perms, outs):
        """Position probabilities of every option order -> mean in the canonical order, then the type temperature."""
        canon = []
        for perm, p_pos in zip(perms, outs):
            c = [0.0] * len(perm)
            for k, j in enumerate(perm):
                c[j] = p_pos[k]
            canon.append(c)
        p = torch.tensor([sum(x) / len(canon) for x in zip(*canon)])
        T = self.temps.get(q_type, 1.0)  # calibrated temperature per question type
        if T != 1.0:
            p = torch.softmax(torch.log(p.clamp_min(1e-12)) / T, -1)
        return p.tolist()

    async def decide(self, body):
        t0 = time.perf_counter()
        pol = body.get("early_exit")
        if pol is not None:
            pol = str(pol)
            allowed = sorted(getattr(self.backend, "policies", {}))
            if pol not in allowed:
                raise ValueError(f"early_exit={pol!r} not available (server mode must be fast-exit); allowed: {allowed}")
        state = body["state"]
        if body.get("facts", "off") not in ("off", "auto"):
            raise ValueError('facts must be "off" or "auto"')
        if body.get("facts") == "auto":  # append computed calendar / arithmetic facts (PL dates, gaps, VAT)
            state = inject_facts(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
        qs = to_items(state, body["questions"])
        branches = []  # (question index, branch index, perms, prompts, letter ids)
        for i, q in enumerate(qs):
            for bi, (question, options) in enumerate(q["branches"]):
                jobs = self.jobs_for(q, question, options)
                if any(ids is None for _, _, ids in jobs):
                    raise ValueError("option letters are not single tokens for this tokenizer")
                branches.append((i, bi, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs]))
        flat = [p for b in branches for p in b[3]]
        toks = self.tok(flat, add_special_tokens=False).input_ids  # tokenize once; the engine takes the ids
        n_tok = sum(map(len, toks))
        if not getattr(self.backend, "takes_text", False):  # Ollama takes the text and tokenizes it itself
            j = 0
            for b in branches:
                b[3][:] = toks[j: j + len(b[3])]; j += len(b[3])
        loop = asyncio.get_running_loop()
        # SOAM: the whole request is one queue entry (one packed row: state once, every branch after it);
        # otherwise one entry per branch (all option orders of one question share the prefix)
        entries = [branches] if self.soam else [[b] for b in branches]
        futs = []
        for e in entries:
            fut = loop.create_future()
            await self.queue.put(([p for b in e for p in b[3]], [x for b in e for x in b[4]], fut, pol))
            futs.append(fut)
        outs = []
        for e, fut in zip(entries, futs):
            res, j = await fut, 0
            for b in e:
                outs.append(res[j: j + len(b[3])]); j += len(b[3])
        per_q = [[] for _ in qs]
        for b, o in zip(branches, outs):
            per_q[b[0]].append((b[2], o))
        answers = {}
        for q, brs in zip(qs, per_q):
            if q["type"] == "multi":
                ans = multi_answer(q, [self.canonical("noul", perms, o)[0] for perms, o in brs])
                answers[q["name"]] = ans
                continue
            perms, o = brs[0]
            pl = self.canonical(q.get("base", q["type"]), perms, o)
            probs = {k: float(v) for k, v in zip(q["keys"], pl)}
            conf = float(max(pl))
            if q["type"] == "act":
                ans = act_answer(q, probs, conf, self.cal)
            elif q["type"] == "noul":
                ans = {"type": "noul", "noul": probs["true"], "probabilities": probs, "confidence": conf}
            elif q["type"] == "score":
                ans = {"type": "score", "score": float(sum(i * v for i, v in enumerate(pl))),
                       "legend": dict(zip(q["keys"], q["options"])), "probabilities": probs, "confidence": conf}
            else:
                ans = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
            answers[q["name"]] = ans
        ev_q = [q for q in qs if body["questions"][q["name"]].get("evidence")]
        if ev_q:  # evidence spans: character offsets into the request's state (facts block excluded)
            if not (self.evidence and self.evidence.available):
                raise ValueError("evidence is not available for this model (no evidence_head.pt)")
            limit = len(_text(body["state"])) if body.get("facts") == "auto" else None
            for q in ev_q:
                if q["type"] == "multi":
                    raise ValueError(f"question {q['name']!r}: evidence is not supported for multi")
                answers[q["name"]]["evidence"] = await loop.run_in_executor(
                    None, self.evidence.spans, q["state"], q["question"], q["options"], q["lang"], limit)
        out = {"model": self.name, "answers": answers,
               "usage": {"input_tokens": n_tok, "output_tokens": 0, "questions": len(qs), "branches": len(branches),
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}
        if self.log:
            self.log_decision(body, out)
        return out

    def log_decision(self, body, out):
        """Propensity log (one JSON line per request): the served probabilities q(a) of every question, so production
        feedback can later be used off-policy. `reviewed` is filled in by the caller's feedback, not here."""
        h = hashlib.sha256(json.dumps([body.get("state"), body.get("questions")], sort_keys=True,
                                      ensure_ascii=False, default=str).encode()).hexdigest()
        rec = {"ts": time.time(), "model": self.name, "request_sha256": h, "reviewed": None,
               "answers": {k: {f: v for f, v in a.items() if f in ("type", "choice", "action", "selected",
                                                                    "probabilities", "confidence")}
                           for k, a in out["answers"].items()}}
        self.log.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parser():
    ap = argparse.ArgumentParser(description="basal typed-decision server")
    ap.add_argument("--model", default="Remek/basal-1.5-4.5B", help="local directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None, help="model name reported in responses (default: last part of --model)")
    ap.add_argument("--mode", choices=list(MODES), default=None,
                    help="default: fast on CUDA, mlx on Apple Silicon (mps without mlx / mlx-lm), tpu on a TPU VM, "
                         "eager otherwise")
    ap.add_argument("--quant", choices=["fp8", "nvfp4", "q8"], default=None,
                    help="override the quantisation of the mode (fp8 / nvfp4: CUDA graph modes, q8: mlx)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="device for --mode eager (default: BASAL_DEVICE, else cuda, else Apple Silicon mps, else cpu)")
    ap.add_argument("--orders", type=int, choices=[1, 2], default=2,
                    help="2 = ask in original and reversed option order and average (default, reduces order sensitivity)")
    ap.add_argument("--early-exit", dest="early_exit", default="off", help="default policy for --mode fast-exit (basal-1.0 models, which ship exit heads)")
    ap.add_argument("--exit-heads", dest="exit_heads", default=None, help="exit heads dir (default: <model>/exit_heads)")
    ap.add_argument("--no-calibration", dest="no_calibration", action="store_true")
    ap.add_argument("--calibration", default=None,
                    help="calibration file (default: <model>/CALIBRATION.<engine>.json if shipped, else CALIBRATION.json)")
    ap.add_argument("--soam", choices=["on", "off"], default="on",
                    help="on = the whole request (every question and option order) is one packed row over the state")
    ap.add_argument("--log-decisions", dest="log_decisions", default=None, help="append a propensity log (JSONL)")
    ap.add_argument("--gpu-memory", dest="gpu_memory", type=float, default=0.6, help="vLLM memory fraction")
    ap.add_argument("--max-len", dest="max_len", type=int, default=4096,
                    help="--mode vllm / sglang / tpu: context length in tokens (longer states are refused; raise it for "
                         "long documents)")
    ap.add_argument("--gguf", default=None, help="--mode gguf: GGUF weights (a -GGUF file or one converted from --model)")
    ap.add_argument("--ollama-url", dest="ollama_url", default="http://127.0.0.1:11434", help="--mode ollama")
    ap.add_argument("--ollama-model", dest="ollama_model", default=None,
                    help="--mode ollama: Ollama model name (the -GGUF repositories); --model gives the tokenizer")
    ap.add_argument("--llamacpp-url", dest="llamacpp_url", default="http://127.0.0.1:8080", help="--mode llamacpp")
    ap.add_argument("--http-parallel", dest="http_parallel", type=int, default=1,
                    help="--mode ollama / llamacpp: concurrent requests (match OLLAMA_NUM_PARALLEL / llama-server -np)")
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

    app = Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]),
                            Route("/v1/basal", systemone, methods=["POST"]), Route("/v1/models", models),
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
