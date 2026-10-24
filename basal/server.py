"""HTTP server for typed decisions (System One-compatible JSON interface).

  POST /v1/systemone   {"state": "...", "questions": {"q1": {"type": "choice"|"noul"|"score"|"act"|"multi"|"tool",
                        "instructions": "...", "criteria": {...} | [...], "option_keys": "show"|"hide" (optional),
                        "evidence": true (optional; models with evidence_head.pt: supporting spans of the state)}},
                        "early_exit": "off"|"0.99"|... (optional), "facts": "off"|"auto" (optional)}
  POST /v1/basal       same as /v1/systemone; also the experimental task types classify, verify, compress, extract
                       (basal/tasks.py)
  GET  /v1/models
  GET  /health
  GET  /metrics

State once, ask many (SOAM): every question of a request, in both option orders (and every label of a `multi`), is one
branch over the state; the state is computed once per request. `act`, `multi` and `tool` are engine-level types:
  act    the underlying question (criteria = outcomes; {"true", "false"} keys = yes/no) plus "actions" and "costs"
         (cost[action][outcome], or the short form {"wrong": W, "defer": D}); returns the minimum-expected-cost action
  multi  one yes/no branch per label ("does label X apply?"); independent probabilities, the selected set and its
         confidence. "objective": "threshold" (default: >= "threshold", default 0.5, at most "max", at least "min"),
         "exact" (p >= 0.5), "iou" / "f1" (the set with the highest expected IoU / F1; "min" / "max" bound its size);
         "exclusive": [labels selected only alone], "fallback": a label selected only when nothing else is
         ("fallback_threshold")
  tool   options built from OpenAI / MCP tool definitions ("tools") plus "extra" (answer directly / ask the user / no
         suitable tool); the chosen tool, "rank", "multi": true for the set of tools, "max_error" (basal/tools.py)

  basal-serve --model Remek/basal-1.5-4.5B --mode fast --port 8000
"""
import argparse
import asyncio
import functools
import gc
import hashlib
import json
import sys
import math
import os
import re
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .engine import (ATTN, EAGER_FORWARDS, EagerBackend, ExitGraphBackend, GGUFBackend, GraphBackend, LlamaCppBackend,
                     MLXBackend, MPSBackend, OllamaBackend, SGLangBackend, VLLM_PRESETS, VLLMBackend, fp8_refused,
                     gemma_fast_kernels, recurrent_layers, recurrent_refused, resolve, set_fp8_fast_accum)
from .evidence import Evidence
from .facts import inject as inject_facts
from .gdn_graph import default_fast as gdn_default_fast, is_gdn, refused as gdn_refused
from . import tasks
from .prompt import (FORMATS, LAYOUTS, MAX_OPTIONS, RequestError, lang_of, letter_ids, max_options, media_messages,
                     model_format, model_layout,
                     reads_request, render)
from .sets import label_set_id
from .sets import select as select_set
from .sets import validate as validate_set
from .tools import QUESTION as TOOL_Q
from .tools import QUESTION_SET as TOOL_SET_Q
from .tools import answer as tool_answer
from .tools import parse as parse_tools
from .tools import set_answer as tool_set_answer

RELEASE_DATE = "2026-10-22"  # the engine's release date (GET /v1/models), not the served model's

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
    "strix": ("strix", None, False),       # AMD Strix Halo: Triton attention / FFN kernels, one sequence per chunk
    "ollama": ("ollama", None, False),      # Ollama with the GGUF exports (1.5); --ollama-url, --ollama-model
    "llamacpp": ("llamacpp", None, False),  # llama.cpp server with the GGUF exports (1.5); --llamacpp-url
    "tpu": ("tpu", None, False),            # Google TPU (experimental, basal[tpu]): JAX / XLA, SOAM, one program per shape
}


def default_mode(model=None, revision=None):
    """fast on CUDA, mlx on Apple Silicon when mlx and mlx-lm are installed, mps otherwise there, tpu on a TPU VM
    with basal[tpu] installed, else eager. basal-omni (Gemma 4; model: its directory or repo id) gets fast only on the
    GPU classes where its fast configuration passed its gate (gemma_fast_kernels: Hopper with flash_attn_3, Blackwell
    workstation GPUs), eager on other GPUs until they pass theirs (2.2.0 had no working default for it there: fast
    stopped at "'Gemma4Model' object has no attribute 'layers'"). A model with recurrent-state layers gets eager on CUDA
    and Apple Silicon (mps refuses it, mlx is not checked on it), except a Qwen3.5 model (Gated DeltaNet), which gets
    fast on the GPU class where the graph engine's port passed its gate (gdn_graph.default_fast: Hopper with fla)."""
    if torch.cuda.is_available():
        if model is not None and _recurrent(model, revision):
            return "fast" if _gdn(model, revision) and gdn_default_fast() else "eager"
        if model is not None and gemma_fast_kernels() is None and _gemma4(model, revision):
            return "eager"
        return "fast"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        if model is not None and _recurrent(model, revision):
            return "eager"
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


def _gemma4(model, revision=None):
    """A Gemma 4 model (basal-omni), from its config.json; False when that cannot be read (the server says why)."""
    from .gemma4_graph import is_gemma4_dir
    try:
        return is_gemma4_dir(resolve(model, revision, metadata_only=True))
    except Exception:  # offline without the model cached, a wrong name: the model load reports it
        return False


def _recurrent(model, revision=None):
    """The recurrent-state layers of a model (recurrent_layers), from its config.json; None when that cannot be read
    (the model load reports it)."""
    try:
        return recurrent_layers(resolve(model, revision, metadata_only=True))
    except Exception:
        return None


def _gdn(model, revision=None):
    """A Qwen3.5 model of Gated DeltaNet and attention layers (gdn_graph.is_gdn: the recurrent models the CUDA graph
    engine serves), from its config.json; False when that cannot be read."""
    try:
        return is_gdn(resolve(model, revision, metadata_only=True))
    except Exception:
        return False


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
        raise RequestError(f'option_keys must be "show" or "hide", got {option_keys!r}')
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


def check_body(body):
    """The shape of a decision request (its questions are read by to_items) -> RequestError (HTTP 422)."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    for k in ("state", "questions"):
        if k not in body:
            raise RequestError(f"missing field {k!r}")
    if not isinstance(body["questions"], dict) or not all(isinstance(q, dict) for q in body["questions"].values()):
        raise RequestError('"questions" must be an object of {name: question object}')


@reads_request
def to_items(state, questions, max_opts=MAX_OPTIONS, force_lang=None):
    """Questions of a request -> readout items (options as a list) + keys to assemble the answer. Every item has
    `branches` = [(question text, options)]: one for the decision types, one per label for `multi`. max_opts: the
    model's prompt format limit (10 for format 1, 255 for format 2). force_lang: the request's "lang" ("pl" / "en")
    for every question's template, yes/no words and branch text, instead of the language detected from its text."""
    state = _text(state)
    out = []
    state_pl = not force_lang and lang_of(state) == "pl"  # once per request, not once per question
    for name, q in questions.items():
        t = q.get("type", "choice")
        instr = _text(q.get("instructions", ""))
        lang = force_lang or ("pl" if state_pl else lang_of(instr))
        extra = {}
        if t == "multi":  # one yes/no branch per label
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {k: None for k in crit}
            if not crit:
                raise RequestError(f"question {name!r}: multi needs criteria")
            keys = list(crit)
            labels = [str(k) if v is None else _text(v) for k, v in crit.items()]
            yn = yes_no(lang)[1]
            branches = [(MULTI_Q[lang].format(q=instr, label=lab), yn) for lab in labels]
            objective = q.get("objective", "threshold")
            validate_set(name, objective, keys, q.get("threshold"), q.get("min"), q.get("max"), q.get("exclusive"),
                         q.get("fallback"), q.get("fallback_threshold"))
            extra = dict(threshold=float(q.get("threshold", 0.5)), max=q.get("max"), min=q.get("min"), labels=labels,
                         objective=objective, exclusive=q.get("exclusive") or [], fallback=q.get("fallback"),
                         fallback_threshold=q.get("fallback_threshold"),
                         fallback_threshold_source="request" if q.get("fallback_threshold") is not None else None)
            out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=yn, lang=lang,
                            branches=branches, **extra))
            continue
        if t == "tool":  # options from tool definitions (basal/tools.py); "multi": true = one yes/no branch per tool
            keys, opts, names = parse_tools(name, q, lang)
            extra = dict(base="choice", tools=names, rank=q.get("rank"), max_error=q.get("max_error"),
                         multi=bool(q.get("multi")))
            if extra["multi"]:
                instr = instr or TOOL_SET_Q[lang]
                objective = q.get("objective", "threshold")
                validate_set(name, objective, keys, q.get("threshold"), q.get("min"), q.get("max"), q.get("exclusive"),
                             q.get("fallback"), q.get("fallback_threshold"))
                yn = yes_no(lang)[1]
                # the extra options can only be chosen alone (an agent either calls tools or does one of them)
                excl = list(dict.fromkeys([k for k in keys if k not in names] + (q.get("exclusive") or [])))
                extra.update(base="noul", threshold=float(q.get("threshold", 0.5)), max=q.get("max"), min=q.get("min"),
                             labels=opts, objective=objective, exclusive=excl, fallback=q.get("fallback"),
                             fallback_threshold=q.get("fallback_threshold"),
                             fallback_threshold_source="request" if q.get("fallback_threshold") is not None else None)
                out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=yn, lang=lang,
                                branches=[(MULTI_Q[lang].format(q=instr, label=o), yn) for o in opts], **extra))
                continue
            instr = instr or TOOL_Q[lang]
        elif t == "act":
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
        if not 2 <= len(opts) <= max_opts:
            what = " (tools and extra options)" if t == "tool" else ""
            raise RequestError(f"question {name!r}: {len(opts)} options{what} (supported: 2..{max_opts})")
        out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=opts, lang=lang,
                        branches=[(instr, opts)], option_keys=q.get("option_keys", "show"), **extra))
    return out


def act_spec(name, q, outcomes):
    """Cost matrix C[action][outcome] from the full form or the short form {"wrong": W, "defer": D}."""
    costs = q.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise RequestError(f"question {name!r}: act needs costs")
    if set(costs) <= {"wrong", "defer"} and not isinstance(costs.get("wrong"), dict):
        w = float(costs["wrong"])
        C = {y: {z: 0.0 if z == y else w for z in outcomes} for y in outcomes}
        if "defer" in costs:
            C["defer"] = {z: float(costs["defer"]) for z in outcomes}
        return dict(costs=C, defer="defer" if "defer" in costs else None, max_error=q.get("max_error"))
    for a, row in costs.items():
        if not isinstance(row, dict) or set(row) != set(outcomes):
            raise RequestError(f"question {name!r}: costs[{a!r}] must give a cost for every outcome {outcomes}")
    C = {a: {z: float(v) for z, v in row.items()} for a, row in costs.items()}
    defer = q.get("defer_action") or next((a for a in ("defer", "human") if a in C), None)
    return dict(costs=C, defer=defer, max_error=q.get("max_error"))


NO_CERTIFICATE = "no certified threshold for this type at this max_error"


def certified(cal, max_error, q_type):
    """The certified confidence threshold of a max_error target for a question type: the type's own
    (thresholds[target].per_type[type]), else None, and the type's decisions with that max_error are deferred. The
    threshold pooled over the types (thresholds[target].confidence) is informational and never applied: a type without
    its own certificate can run above the target under it. Only a format-1 file written before per-type thresholds
    (no per_type at all, the basal-1.5 files) keeps its pooled threshold, its only certificate."""
    entry = (cal.get("thresholds") or {}).get(str(float(max_error))) or {}
    if "per_type" not in entry and cal.get("prompt", 1) == 1:
        return entry.get("confidence")
    return ((entry.get("per_type") or {}).get(q_type) or {}).get("confidence")


def act_answer(q, probs, conf, cal):
    """Minimum expected cost action; with max_error, automatic actions below the certified threshold of the question's
    type are refused, and all of them when the type has no certified threshold for that target (`deferred`)."""
    spec = q["act"]
    exp = {a: sum(probs[z] * c for z, c in row.items()) for a, row in spec["costs"].items()}
    action = min(exp, key=exp.get)
    ans = {"type": "act", "action": action, "expected_costs": {a: round(v, 6) for a, v in exp.items()},
           "answer": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
    if spec["max_error"] is not None:
        thr = certified(cal, spec["max_error"], q["base"])
        if thr is None:
            ans["max_error"] = NO_CERTIFICATE
        if (thr is None or conf < thr) and action != spec["defer"]:
            if spec["defer"] is None:
                raise RequestError("max_error needs a defer action (\"defer\"/\"human\" or defer_action)")
            ans["action"], ans["refused"] = spec["defer"], action
        ans["deferred"] = ans["action"] == spec["defer"]
    validated = bool(cal.get("temperature_per_prim") or cal.get("temperature_per_bucket")) and (
        q["base"] == "noul" or q.get("option_keys") == "hide")
    ans["calibration"] = "validated" if validated else "unvalidated"
    return ans


def multi_answer(q, p_yes):
    probs = dict(zip(q["keys"], p_yes))
    objective = q.get("objective", "threshold")
    sel, score, applied = select_set(probs, objective, q["threshold"], q["min"], q["max"], q.get("exclusive"),
                                     q.get("fallback"), q.get("fallback_threshold"))
    conf = 1.0
    for k, v in probs.items():
        conf *= v if k in sel else 1 - v
    if objective == "threshold":  # the 1.5 response (plus constraints_applied when the request has constraints)
        ans = {"type": "multi", "selected": sel, "probabilities": probs, "threshold": q["threshold"],
               "set_confidence": conf}
    else:
        ans = {"type": "multi", "selected": sel, "probabilities": probs, "objective": objective,
               "expected_score": score, "set_confidence": conf}
    if applied is not None:
        if q.get("fallback") is not None:  # request, the calibration file's fit for this label set, or the default
            applied["fallback_threshold_source"] = q.get("fallback_threshold_source") or "default"
        ans["constraints_applied"] = applied
    return ans


def stored_fallback(cal, q):
    """A multi question with a fallback and no fallback_threshold of its own takes the threshold basal-calibrate fitted
    for the same label set (calibration file 2's `multi` section), if any."""
    if q.get("fallback") is None or q.get("fallback_threshold") is not None:
        return
    entry = (cal.get("multi") or {}).get(label_set_id(q["keys"], q["labels"], q["fallback"])) or {}
    if entry.get("fallback_threshold") is not None:
        q["fallback_threshold"], q["fallback_threshold_source"] = float(entry["fallback_threshold"]), "calibration"


def temperature(cal, q_type, k):
    """Calibrated temperature of a question type with k options. Calibration file 2 (format-2 models) may give
    temperature_per_bucket[type]["lo-hi"] for option-count buckets ("2-10", "11-26", "27-255"); a type or count without
    a bucket uses temperature_per_prim[type] (the only form in 1.5 files), else 1."""
    for b, T in ((cal.get("temperature_per_bucket") or {}).get(q_type) or {}).items():
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= k <= hi:
            return float(T)
    return float((cal.get("temperature_per_prim") or {}).get(q_type, 1.0))


def check_calibration(cal, fmt, path):
    """Malformed buckets or a file fitted for another prompt format are refused at start-up, not on a request."""
    for t, bs in (cal.get("temperature_per_bucket") or {}).items():
        for b in bs:
            if not re.fullmatch(r"\d+-\d+", b) or int(b.split("-")[0]) > int(b.split("-")[1]):
                raise SystemExit(f"{path}: temperature_per_bucket[{t!r}] has a bad bucket {b!r} (\"lo-hi\", e.g. \"2-10\")")
    if cal.get("prompt", fmt) != fmt:
        raise SystemExit(f"{path} was fitted for prompt format {cal['prompt']}, the model uses format {fmt}")


def calibration_name(kind, quant, preset=None):
    """Engine name of the per-engine calibration file CALIBRATION.<engine>.json: the quantised CUDA modes (fp8, nvfp4),
    the engines other than the basal bf16 path and a --vllm-preset other than the default (vllm.<preset>); None =
    CALIBRATION.json."""
    if kind in ("graph", "exit") and quant in ("fp8", "nvfp4"):
        return quant
    if kind == "vllm" and preset not in (None, "default"):
        return f"vllm.{preset}"
    return kind if kind in ("vllm", "sglang", "mlx", "gguf", "ollama", "llamacpp", "tpu") else None


DETERMINISTIC_KINDS = ("graph", "exit", "eager", "mps", "mlx", "gguf", "strix", "tpu")  # engines that run in this process


def deterministic_setup(kind):
    """--deterministic: one request per forward (its own packed rows and shapes, never batched with other requests) and
    deterministic kernels, so an answer does not depend on load or on which requests run at the same time. Must run
    before the first cuBLAS call (the workspace setting is read when the handle is created)."""
    if kind not in DETERMINISTIC_KINDS:
        raise SystemExit("--deterministic needs an engine that runs in this process (fast, fast-nocompile, fast-exit, "
                         "fp8, nvfp4, eager, mps, mlx, gguf, strix, tpu); vLLM, SGLang, Ollama and llama.cpp server "
                         "batch requests their own way")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    set_fp8_fast_accum(False)  # FP8 GEMMs with full accumulation (fast accumulation rounds by kernel choice)


def _answer(fut, res):
    if not fut.done():
        fut.set_result(res)


def out_of_memory(e):
    """The GPU ran out of memory: torch.OutOfMemoryError (CUDA, MPS), or a CUDA / Metal error that says so."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and (
        "out of memory" in str(e).lower() or "OutOfMemory" in str(e))


def http_status(e):
    """The status code of a failed request: 422 for an invalid request (RequestError: the client must change it),
    503 when the GPU ran out of memory (a retry may succeed: the server frees the request's memory), 500 for any other
    server error."""
    if isinstance(e, RequestError):
        return 422
    return 503 if out_of_memory(e) else 500


def _fail(futs, e):
    """Requests of a failed engine call get its error. Out of GPU memory: the frames of the failed forward are cleared
    first; through the traceback (and the futures that hold it, a reference cycle) they kept its tensors alive until
    a garbage collection, so the handler's empty_cache could not return them and the next requests ran out of memory
    too (ultra in eager: a 2k-token request right after an 82k-token one, 20 Oct)."""
    if out_of_memory(e):
        traceback.clear_frames(e.__traceback__)
    for f in futs:
        if not f.done():
            f.set_exception(e)


class LaneGate:
    """The GPU between the two lanes of --long-lane, main lane first. The long lane runs one decoder layer at a time and
    calls between(i) before each layer: it keeps at most two layers queued on the GPU (so the main lane never waits
    behind more than that), and when the main lane has work, the long lane has run for at least `slice_s` since it
    last resumed and its recent share of the GPU's busy time is at least `share`, it lets its queued layers finish and
    waits for the main lane's next engine call to end (or for the main lane to go idle). Below its share it keeps
    running, so long requests advance under constant short traffic; the busy times decay with a 2 s time constant."""

    TAU = 2.0  # seconds: memory of the busy-time shares

    def __init__(self, slice_s, cuda=True, share=0.5):
        self.slice, self.cuda, self.share = slice_s, cuda, share
        self.cv = threading.Condition()
        self.pending = 0    # main lane: queue entries not yet answered (queued or in an engine call)
        self.calls = 0      # main lane: engine calls finished
        self.resumed, self.last, self.yields = 0.0, None, 0
        self.busy = {"long": 0.0, "main": 0.0}
        self.t_busy = self.mark = time.perf_counter()

    def _account(self, lane, dt):
        now = time.perf_counter()
        k = math.exp(-(now - self.t_busy) / self.TAU)
        for x in self.busy:
            self.busy[x] *= k
        self.busy[lane] += dt
        self.t_busy = now

    def long_share(self):
        total = self.busy["long"] + self.busy["main"]
        return self.busy["long"] / total if total > 0 else 0.0

    def main_add(self, n):
        with self.cv:
            self.pending += n

    def main_done(self, n, seconds=0.0):
        with self.cv:
            self.pending -= n
            self.calls += 1
            self._account("main", seconds)
            self.cv.notify_all()

    def start(self):
        """A long-lane engine call begins: its slice starts now."""
        self.resumed = self.mark = time.perf_counter()
        self.last = None

    def between(self, i):
        if self.cuda and self.last is not None:
            self.last.synchronize()  # the layer before the previous one is done: at most two layers queued
        now = time.perf_counter()
        with self.cv:
            self._account("long", now - self.mark)
        self.mark = now
        if self.pending and now - self.resumed >= self.slice and self.long_share() >= self.share:
            if self.cuda:
                torch.cuda.current_stream().synchronize()
            with self.cv:
                target = self.calls + 1
                self.cv.wait_for(lambda: self.calls >= target or self.pending <= 0)
            self.yields += 1
            self.resumed = self.mark = time.perf_counter()
        if self.cuda:
            self.last = torch.cuda.Event()
            self.last.record()


def lean_arg(value):
    """--lean on / pass2 / off -> the GraphBackend argument (True, "pass2", False)."""
    return {"on": True, "off": False}.get(value, value)


def lean_tree(lean, tree, kind, compiled=True, quant=None):
    """--lean / --tree auto (opt-in; v2.0.0 defaults: off / dense) -> the values in effect. The compiled CUDA graph
    engine without torchao quantisation (--mode fast, bf16 or FP8 checkpoints) runs pass 2 with the lean layer and the
    FlexAttention trie mask (faster wherever pass 2 runs; dev v2 accuracy with two orders >= the dense mask's - 0.05
    points, bf16 and FP8); other engines and --lean on keep the dense mask."""
    if lean == "auto":
        lean = "pass2" if kind == "graph" and compiled and not quant and tree in ("auto", "flex") else "off"
    if tree == "auto":
        tree = "flex" if lean == "pass2" else "dense"
    return lean, tree


def refuse_lean_flex(lean, tree):
    """--lean on with --tree flex (both explicit, so refused before a mode is chosen or a model resolved)."""
    if lean == "on" and tree == "flex":
        raise SystemExit("--tree flex runs with --lean pass2; with --lean on its FlexAttention falls back to an "
                         "uncompiled kernel during CUDA graph capture and the engine fails to start")


def merge_arg(value):
    """--merge auto / on / off -> the GraphBackend argument ("auto", True, False)."""
    return {"on": True, "off": False}.get(value, "auto")


def default_orders(orders, fmt):
    """--orders: 1, 2, "auto" or "2le<N>" (both orders for questions with at most N options, one above); default 2
    for prompt-1 models (as in 1.5) and 1 for prompt-2 models."""
    if orders is None:
        return 2 if fmt == 1 else 1
    return orders if orders == "auto" or str(orders).startswith("2le") else int(orders)


def orders_arg(value):
    """--orders values: 1, 2, auto, 2le<N> (N >= 2)."""
    if value in ("1", "2", "auto") or (value.startswith("2le") and value[3:].isdigit() and int(value[3:]) >= 2):
        return value
    raise argparse.ArgumentTypeError(f"--orders must be 1, 2, auto or 2le<N> (N >= 2), got {value!r}")


def orders_for(orders, k):
    """Option orders of a question with k options under --orders (auto: 1 in the first pass)."""
    if orders == 2:
        return 2
    if isinstance(orders, str) and orders.startswith("2le"):
        return 2 if k <= int(orders[3:]) else 1
    return 1


def models_payload(name, mode, policies, temperatures=None, prompt_format=None, orders=None, media=(),
                   calibration_warning=None, vllm_preset=None, layout=None):
    """GET /v1/models in the official ModelMetadataList shape, plus the serving mode, early-exit levels, the
    temperatures in use ({} = uncalibrated probabilities, what basal-calibrate needs from a running server), the
    model's prompt format, the option orders served, why the --calibration file may not belong to the model and the
    --vllm-preset (--mode vllm)."""
    entry = {"name": name, "release_date": RELEASE_DATE,
             "description": "basal typed-decision model (choice / noul / score / act / multi / tool), Polish and "
                            "English",
             "mode": mode, "early_exit": sorted(policies), "temperature_per_prim": dict(temperatures or {})}
    if prompt_format is not None:
        entry["prompt_format"] = prompt_format
    if media:  # --media on: the media the served model takes
        entry["media"] = list(media)
    if orders is not None:
        entry["orders"] = orders
    if calibration_warning:
        entry["calibration_warning"] = calibration_warning
    if vllm_preset:
        entry["vllm_preset"] = vllm_preset
    if layout not in (None, "canonical"):  # the prompt layout served (2.4.1; basal-calibrate fits its file)
        entry["layout"] = layout
    return {"models": [entry]}


def refuse_layout_caches(layout, a):
    """--state-cache-gb and --prefix-cache cache the prompt up to the end of the state, which only the canonical layout
    starts with."""
    if getattr(a, "state_cache_gb", 0) or getattr(a, "prefix_cache", "off") == "on":
        raise SystemExit(f"layout {layout}: --state-cache-gb and --prefix-cache need the canonical layout (they cache "
                         "the prompt up to the end of the state, which only that layout starts with)")


def calibration_layout(cal):
    """The prompt layout a calibration file was fitted on: fitted_on.layout (basal-calibrate --layout), else a top-level
    "layout" (the training side's files), else canonical."""
    fitted = cal.get("fitted_on")
    return (fitted.get("layout") if isinstance(fitted, dict) else None) or cal.get("layout") or "canonical"


def calibration_orders(cal):
    """The option orders a calibration file was fitted with: fitted_on.orders (basal-calibrate, where fitted_on is an
    object) or a top-level "orders" (the training side's files, where fitted_on is a text); None when the file does
    not say (the published basal-1.5 files)."""
    fitted = cal.get("fitted_on")
    o = fitted.get("orders") if isinstance(fitted, dict) and fitted.get("orders") is not None else cal.get("orders")
    return None if o is None else o if o == "auto" or str(o).startswith("2le") else int(o)


def model_identity(name, md):
    """(model, revision) of a served model: the repo id and commit of a Hugging Face cache snapshot, else the name as
    given and no revision."""
    p = Path(md) if md is not None else None
    if p is not None and p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        return p.parent.parent.name[len("models--"):].replace("--", "/"), p.name
    return str(name), None


def calibration_mismatch(cal, name, md):
    """Why a file written by basal-calibrate for another model, or for another revision of this one, does not belong
    to the served model; None when it does or the file does not say (files that ship with a model)."""
    fitted = cal.get("fitted_on")
    if not isinstance(fitted, dict) or not fitted.get("model"):
        return None
    model, rev = model_identity(name, md)
    key = lambda s: str(s).rstrip("/").split("/")[-1].lower()  # a repo id, a local copy or a served name
    if key(fitted["model"]) != key(model):
        return f"it was fitted on {fitted['model']}, the served model is {model}"
    if fitted.get("revision") and rev and fitted["revision"] != rev:
        return f"it was fitted on revision {fitted['revision'][:12]} of {model}, the served revision is {rev[:12]}"
    return None


def load_calibration(md, kind, path, quant, orders, off=False, preset=None, layout="canonical"):
    """The calibration basal-serve applies -> (calibration dict, its file, None or why it is unvalidated). A file
    fitted on other option orders than the served ones gives wrong temperatures and thresholds: the file for the
    served orders, CALIBRATION[.<engine>].o<orders>.json, is used when the model ships one; otherwise the file is kept
    for its temperatures without its thresholds, fitted fallback thresholds and task thresholds (max_error defers,
    tasks report automated: null), and the answers are marked unvalidated. An explicit --calibration file is not replaced."""
    layout = layout or "canonical"
    cal = calibration_file(md, kind, path, quant, preset, layout)
    data = {} if off or not cal.exists() else json.loads(cal.read_text(encoding="utf-8"))
    if data and calibration_layout(data) != layout:  # another layout's thresholds do not hold here (2.4.1)
        why = f"{cal.name} was fitted on the {calibration_layout(data)} layout, served with --layout {layout}"
        return {k: v for k, v in data.items() if k not in ("thresholds", "multi", "task_thresholds")}, cal, why
    fitted = calibration_orders(data)
    alt = cal.with_name(f"{cal.stem}.o{orders}{cal.suffix}")
    if fitted is not None and fitted != orders and not path and alt.exists():
        cal, data = alt, json.loads(alt.read_text(encoding="utf-8"))
        fitted = calibration_orders(data)
    if fitted is None or fitted == orders:
        return data, cal, None
    why = f"{cal.name} was fitted on {fitted} option order(s), served with {orders}"
    return {k: v for k, v in data.items() if k not in ("thresholds", "multi", "task_thresholds")}, cal, why


def calibration_file(md, kind, path=None, quant=None, preset=None, layout="canonical"):
    """The calibration file basal-serve loads: `path` (--calibration), else CALIBRATION.<engine>.json when the model
    ships one for the engine (calibration_name; a --vllm-preset without a file of its own: CALIBRATION.vllm.json),
    else CALIBRATION.json. Another --layout than canonical: CALIBRATION[.<engine>].<layout>.json first (2.4.1; without
    one the canonical file is loaded and its thresholds are not applied, load_calibration)."""
    if path:
        return Path(path)
    layout = layout or "canonical"
    # 1.5: each engine has its own calibration file when shipped (fp8 too); 2.2.1: each vLLM preset
    engines = dict.fromkeys((calibration_name(kind, quant, preset), calibration_name(kind, quant)))
    for sfx in ([f".{layout}"] if layout != "canonical" else []) + [""]:
        for eng in engines:
            if eng and (md / f"CALIBRATION.{eng}{sfx}.json").exists():
                return md / f"CALIBRATION.{eng}{sfx}.json"
        if sfx and (md / f"CALIBRATION{sfx}.json").exists():
            return md / f"CALIBRATION{sfx}.json"
    return md / "CALIBRATION.json"


def model_context(md):
    """The model's context in tokens (max_position_embeddings in its config.json), None when unknown."""
    try:
        cfg = json.loads((Path(md) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    n = cfg.get("max_position_embeddings") or (cfg.get("text_config") or {}).get("max_position_embeddings")
    return int(n) if n else None


def context_limit(md, backend):
    """(longest prompt in tokens, what sets it): the model's context, or the engine's own limit when that is lower
    (vLLM / SGLang --max-len, Ollama num_ctx, llama-server's context per slot); None when neither is known."""
    limits = [(n, why) for n, why in ((model_context(md), "the model's context"),
                                      (getattr(backend, "context", None), getattr(backend, "context_reason", None)))
              if n]
    return min(limits, key=lambda x: x[0]) if limits else None


def temper(p, T):
    """The calibrated temperature of a question type on a probability vector (a float32 tensor), as served."""
    return torch.softmax(torch.log(p.clamp_min(1e-12)) / T, -1) if T != 1.0 else p


class Server:
    LONG_BATCH, LONG_BATCH_TOKENS = 4, 32768  # --long-lane: long requests (and their tokens) per long-lane engine call
    SCHED_CALL_TOKENS = 8192  # --scheduler: prompt tokens of long requests per main-lane engine call (at least one)
    TRUNC_MARK, TRUNC_MARGIN = "\n[…]\n", 32  # --context-overflow truncate: the cut in the state, spare tokens
    # GET /health: client requests in flight (serve) and the calls of ask whose queue entries all wait for an engine
    # call (waiting: future of a queue entry -> [True while its call waits])
    in_flight = queued = 0
    waiting = None

    def __init__(self, a):
        validate_quant(a.mode, a.quant)
        kind, quant, comp = MODES[a.mode]
        if kind == "gguf" and not a.gguf:
            raise SystemExit("--mode gguf needs --gguf <file.gguf> (--model gives tokenizer and calibration)")
        if a.calibration and not a.no_calibration and not Path(a.calibration).exists():
            raise SystemExit(f"--calibration {a.calibration}: no such file")  # a typo must not serve uncalibrated
        if a.deterministic:
            deterministic_setup(kind)
        elif getattr(a, "fp8_accum", "fast") == "full":  # FP8 checkpoints: the same GEMM arithmetic as --deterministic
            set_fp8_fast_accum(False)
        if getattr(a, "state_cache_gb", 0) and (a.deterministic or kind not in ("graph", "exit")):
            raise SystemExit("--state-cache-gb needs the CUDA graph engine (fast, fast-nocompile, fp8, nvfp4) and is "
                             "not combined with --deterministic (a cached state changes the shapes of the forwards)")
        if getattr(a, "layout", None) not in (None, "canonical"):
            refuse_layout_caches(a.layout, a)
        if getattr(a, "prefix_cache", "off") == "on" and (a.deterministic or kind != "graph"):
            raise SystemExit("--prefix-cache needs the CUDA graph engine (fast, fast-nocompile, fp8, nvfp4) and is not "
                             "combined with --deterministic")
        a.lean, a.tree = lean_tree(getattr(a, "lean", "off"), getattr(a, "tree", "dense"), kind, comp, a.quant or quant)
        refuse_lean_flex(a.lean, a.tree)
        if getattr(a, "batches", "default") not in ("default", "auto") and kind not in ("graph", "exit"):
            raise SystemExit("--batches needs the CUDA graph engine (fast, fast-nocompile, fast-exit, fp8, nvfp4)")
        if getattr(a, "vllm_preset", "default") != "default" and kind != "vllm":
            raise SystemExit("--vllm-preset needs --mode vllm (the CUDA graph engine has --deterministic)")
        attn = getattr(a, "attn", "auto")
        if attn not in ("auto", "fa2") and kind not in ("graph", "exit"):
            raise SystemExit(f"--attn {attn} needs the CUDA graph engine (--mode fast, fast-nocompile, fast-exit, fp8, "
                             "nvfp4)")
        if getattr(a, "mlx_packed", False) and kind != "mlx":
            raise SystemExit("--mlx-packed needs --mode mlx (or mlx-q8)")
        if kind == "mps" and a.device == "cuda":
            raise SystemExit("--mode mps runs on Apple Silicon or the CPU (--device mps / cpu); on CUDA: --mode fast")
        if kind in ("graph", "exit"):  # before the weights are downloaded (mps: MPSBackend, when it shares prefixes)
            what = _recurrent(a.model, a.revision)
            gdn = bool(what) and _gdn(a.model, a.revision)
            if what and (kind == "exit" or not gdn):
                raise SystemExit(recurrent_refused(a.model, what, gdn))
            if gdn:  # Qwen3.5 in the graph engine (basal 2.4): bf16, plain and branched rows
                why = gdn_refused(a.quant or quant, getattr(a, "state_cache_gb", 0),
                                  getattr(a, "prefix_cache", "off") == "on", getattr(a, "batches", "default"),
                                  lean_arg(getattr(a, "lean", "off")), getattr(a, "tree", "dense"),
                                  getattr(a, "readout_layer", "off"), getattr(a, "template_prefix", "off"))
                if not why and getattr(a, "long_lane", "off") == "on":
                    why = "--long-lane is not available for Qwen3.5 (Gated DeltaNet) models"
                if why:
                    raise SystemExit(f"{a.model}: {why}")
        if kind in ("graph", "exit") and (a.quant or quant) == "fp8" and fp8_refused():
            raise SystemExit(fp8_refused())
        # GGUF modes: the engine serves the GGUF weights, so only tokenizer, template and calibration are downloaded
        md = resolve(a.model, a.revision, metadata_only=kind in ("gguf", "ollama", "llamacpp"))
        self.fmt = model_format(md)  # prompt format of the model (basal.json "prompt"; 1 when absent)
        self.max_options = max_options(self.fmt)
        self.orders = default_orders(a.orders, self.fmt)
        # the prompt layout (prompt.LAYOUTS; 2.4.1): --layout, else the model's basal.json "layout", else canonical
        self.layout = getattr(a, "layout", None) or model_layout(md)
        if self.layout != "canonical" and self.fmt != 2:
            raise SystemExit(f"layout {self.layout} needs a prompt-format-2 model (basal.json \"prompt\": 2)")
        if self.layout != "canonical":
            refuse_layout_caches(self.layout, a)
        self.tau, self.band = a.auto_tau, tuple(a.auto_band)
        shared = self.orders != 1 or a.soam == "on"  # prefix sharing: option orders, or the questions of a request
        self.name = a.name or a.model.rstrip("/").split("/")[-1]
        quant = a.quant or quant
        if getattr(a, "media", "off") == "on" and kind not in ("eager", "vllm"):
            raise SystemExit("--media on needs --mode eager or vllm (the media paths of basal-omni)")
        if getattr(a, "ple_offload", "off") == "on" and kind != "eager":
            raise SystemExit("--ple-offload on needs --mode eager")
        if getattr(a, "eager_attn", "math") != "math" and kind != "eager":
            raise SystemExit("--eager-attn needs --mode eager (the CUDA graph engine takes --attn)")
        if kind == "eager":
            self.backend = EagerBackend(md, a.dtype, device=a.device, media=getattr(a, "media", "off") == "on",
                                        forward=getattr(a, "eager_forward", None),
                                        ple_offload=getattr(a, "ple_offload", "off") == "on",
                                        attn=getattr(a, "eager_attn", "math"))
        elif kind == "exit":
            self.backend = ExitGraphBackend(md, a.dtype, quant, compile=comp, heads_dir=a.exit_heads,
                                            default_policy=a.early_exit, attn=attn, merge=merge_arg(a.merge))
        elif kind == "vllm":
            self.backend = VLLMBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len, max_options=self.max_options,
                                       preset=getattr(a, "vllm_preset", "default"),
                                       fast_prefill=getattr(a, "vllm_fast_prefill", "auto") == "auto",
                                       media=getattr(a, "media", "off") == "on")
        elif kind == "sglang":
            self.backend = SGLangBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len)
        elif kind == "mlx":
            self.backend = MLXBackend(md, quant=quant, packed=getattr(a, "mlx_packed", False))
        elif kind == "mps":
            self.backend = MPSBackend(md, a.dtype, shared=shared, device=a.device)
        elif kind == "gguf":
            self.backend = GGUFBackend(md, a.gguf)
        elif kind == "strix":
            from .strix import StrixHaloFastBackend
            self.backend = StrixHaloFastBackend(md, a.dtype)
        elif kind == "ollama":
            self.backend = OllamaBackend(md, a.ollama_url, a.ollama_model, parallel=a.http_parallel)
        elif kind == "llamacpp":
            self.backend = LlamaCppBackend(md, a.llamacpp_url, parallel=a.http_parallel)
        elif kind == "tpu":
            from .tpu import TPUBackend
            self.backend = TPUBackend(md, a.dtype, shared=shared, max_len=a.max_len)
        else:
            self.backend = GraphBackend(md, a.dtype, quant, compile=comp, shared=shared, attn=attn,
                                        state_cache_gb=getattr(a, "state_cache_gb", 0),
                                        merge=merge_arg(a.merge), lean=lean_arg(getattr(a, "lean", "off")),
                                        tree=getattr(a, "tree", "dense"),
                                        prefix=getattr(a, "prefix_cache", "off") == "on",
                                        batches=getattr(a, "batches", "default"),
                                        readout_layer=getattr(a, "readout_layer", "off"),
                                        template=getattr(a, "template_prefix", "off"),
                                        gemma_attn=getattr(a, "gemma_attn", "auto"))
        self.tok = self.backend.tok
        self.states_cached = bool(getattr(a, "state_cache_gb", 0))  # --state-cache-gb: requests are cut at their state
        model = getattr(self.backend, "model", None)  # evidence head: needs the in-process HF model
        self.evidence = Evidence(md, model, self.tok, self.fmt) if isinstance(model, torch.nn.Module) else None
        self.md, self.kind = md, kind
        self.context = context_limit(md, self.backend)
        self.cal, cal, self.cal_unvalidated = load_calibration(md, kind, a.calibration, quant, self.orders,
                                                                a.no_calibration, getattr(a, "vllm_preset", None),
                                                                self.layout)
        check_calibration(self.cal, self.fmt, cal)
        self.cal_mismatch = calibration_mismatch(self.cal, a.model, md) if a.calibration and not a.no_calibration \
            else None  # a file of your own (--calibration); the files in the model directory belong to it
        if self.cal_mismatch:
            if getattr(a, "strict_calibration", False):
                raise SystemExit(f"{cal}: {self.cal_mismatch} (--strict-calibration)")
            print(f"basal: warning: {cal.name}: {self.cal_mismatch}; its thresholds were certified on that model "
                  "(refit with basal-calibrate, or --strict-calibration to refuse such a file)", flush=True)
        if self.cal_unvalidated:
            print(f"basal: warning: {self.cal_unvalidated}; its answers are marked \"calibration\": \"unvalidated\" "
                  "and max_error defers (fit a file for this serving with basal-calibrate, or serve with the file's "
                  "orders and layout)", flush=True)
        self.temps = self.cal.get("temperature_per_prim", {})
        self.soam = a.soam == "on"
        self.log = open(a.log_decisions, "a", buffering=1, encoding="utf-8") if a.log_decisions else None
        self.letters = {}
        self.queue, self.max_batch, self.wait = asyncio.Queue(), a.max_batch, a.wait_ms / 1000
        self.window = a.batch_window_ms / 1000
        if a.deterministic:  # one queue entry (a whole request with SOAM) per forward
            self.max_batch, self.wait, self.window = 1, 0.0, 0.0
        self.overflow = getattr(a, "context_overflow", "refuse")  # a prompt over the context: truncate the state
        self.trunc_head = getattr(a, "truncate_head", 1 / 3)
        if not 0 <= self.trunc_head <= 1:
            raise SystemExit("--truncate-head must be between 0 and 1")
        self.backlog = None  # --scheduler on: long requests waiting for a main-lane call (schedule)
        if getattr(a, "scheduler", "off") in ("on", "auto") and kind == "graph" and not a.deterministic \
                and getattr(a, "long_lane", "off") != "on":
            self.backlog, self.sched_tokens, self.sched_rate = [], a.long_tokens, 1 / 30000
        self.long_tokens = None  # --long-lane: requests whose longest prompt has more tokens go to the long lane
        if getattr(a, "long_lane", "off") == "on":
            if kind != "graph":
                raise SystemExit("--long-lane needs the CUDA graph engine (fast, fast-nocompile, fp8, nvfp4)")
            self.long_lane(a.long_tokens, a.long_slice_ms / 1000, cuda=True, share=getattr(a, "long_share", 0.5))

    def long_lane(self, tokens, slice_s, cuda=True, share=0.5):
        """Turn on the long lane: requests whose longest prompt has more than `tokens` tokens are answered by
        long_worker, the others batch in the main lane as before (LaneGate shares the GPU)."""
        self.long_tokens, self.gate = tokens, LaneGate(slice_s, cuda, share)
        self.long_entries, self.long_ready, self.long_rate = [], asyncio.Event(), 1 / 30000  # s per token, updated

    def jobs_for(self, q, question=None, options=None, perms=None):
        """Prompts of one branch in every option order of the first pass (or in `perms`): [(perm, prompt, code ids)]."""
        question = q["question"] if question is None else question
        options = q["options"] if options is None else options
        k = len(options)
        if perms is None:
            perms = [list(range(k)), list(range(k))[::-1]][: orders_for(self.orders, k)]
        jobs = []
        for perm in perms:
            prompt = render(self.tok, q["state"], question, [options[c] for c in perm], q["lang"], self.fmt,
                            getattr(self, "layout", "canonical"))
            jobs.append((perm, prompt, self.code_ids(prompt, k)))
        return jobs

    def branch(self, i, bi, q, perms=None):
        """Branch bi of question i -> (i, bi, perms, prompts, code ids)."""
        jobs = self.jobs_for(q, *q["branches"][bi], perms)
        if any(ids is None for _, _, ids in jobs):
            raise ValueError("option letters are not single tokens for this tokenizer")
        return (i, bi, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs])

    def encode(self, branches):
        """Tokenize the prompts of all branches at once (the engine takes the ids) -> (number of prompt tokens, the
        longest prompt of each branch)."""
        toks = self.tok([p for b in branches for p in b[3]], add_special_tokens=False).input_ids
        longest, j = [], 0
        for b in branches:
            longest.append(max(map(len, toks[j: j + len(b[3])])))
            if not getattr(self.backend, "takes_text", False):  # Ollama takes the text and tokenizes it itself
                b[3][:] = toks[j: j + len(b[3])]
            j += len(b[3])
        return sum(map(len, toks)), longest

    def truncate_state(self, text, excess):
        """--context-overflow truncate: the state without its middle, `excess` tokens (plus TRUNC_MARGIN for the marker
        and merges at the cuts) shorter: its first --truncate-head share of the kept tokens, TRUNC_MARK, the rest from
        its end. -> (state, what was cut) or (None, None) when too little of the state would be left."""
        ids = self.tok(text, add_special_tokens=False).input_ids
        keep = len(ids) - excess - self.TRUNC_MARGIN
        if keep < self.TRUNC_MARGIN:
            return None, None
        head = int(round(keep * self.trunc_head))
        tail = keep - head
        state = self.tok.decode(ids[:head]) + self.TRUNC_MARK + (self.tok.decode(ids[-tail:]) if tail else "")
        return state, {"state_tokens": len(ids), "head_tokens": head, "tail_tokens": tail,
                       "dropped_tokens": len(ids) - keep, "head_share": self.trunc_head}

    def check_context(self, qs, branches, longest):
        """Every prompt must fit the context: one prompt is the state, one question (a branch) and its options, so
        with SOAM the state counts once plus the longest question. Too long -> HTTP 422 with the count and the limit."""
        limit = getattr(self, "context", None)
        if not limit or not longest:
            return
        i = max(range(len(longest)), key=longest.__getitem__)
        if longest[i] <= limit[0]:
            return
        q = qs[branches[i][0]]
        hint = "shorten the state, the question or the options"
        if q["type"] == "tool":
            hint = ("lower description_chars, give fewer tools, or shortlist the tools first (an embedding or keyword "
                    "prefilter to about 30-50)")
        raise RequestError(f"question {q['name']!r}: the prompt is {longest[i]} tokens (state, question and options), "
                           f"more than the {limit[0]} tokens of {limit[1]}; {hint}")

    async def serve(self, body):
        """One client request (POST /v1/systemone, /v1/basal or an MCP tool call): the body's shape, then decide.
        Counted as in flight from here to its answer or error (GET /health)."""
        self.in_flight += 1
        try:
            check_body(body)
            return await self.decide(body)
        finally:
            self.in_flight -= 1

    def load(self):
        """GET /health: "queued" = requests waiting for the engine (every queue entry of their current engine work
        still in the main queue, the --scheduler backlog or the --long-lane queue); "running" = the other requests in
        flight (in an engine call, or in the server's own work before and after it). running + queued = 0: idle."""
        return {"running": self.in_flight - self.queued, "queued": self.queued}

    def started(self, entries):
        """Queue entries that go into an engine call: their requests are no longer queued (GET /health)."""
        for e in entries:
            wait = self.waiting.pop(e[2], None) if self.waiting else None
            if wait and wait[0]:
                wait[0] = False
                self.queued -= 1

    async def ask(self, branches, pol, cut=None, lane="main"):
        """Position probabilities of every prompt of every branch: [[probs per prompt] per branch]. cut: the number of
        tokens of the request's state prefix (the state cache's unit), passed to the engine with every entry. lane
        "long": the whole request is one entry of the long lane (--long-lane)."""
        loop = asyncio.get_running_loop()
        # SOAM: the whole request is one queue entry (one packed row: state once, every branch after it);
        # otherwise one entry per branch (all option orders of one question share the prefix)
        entries = [branches] if self.soam or lane == "long" else [[b] for b in branches]
        futs, wait = [], [True]  # this call waits (queued) until one of its entries starts an engine call
        if self.waiting is None:
            self.waiting = {}
        self.queued += 1
        if lane == "main" and getattr(self, "gate", None) is not None:
            self.gate.main_add(len(entries))
        for e in entries:
            fut = loop.create_future()
            self.waiting[fut] = wait
            entry = ([p for b in e for p in b[3]], [x for b in e for x in b[4]], fut, pol, cut)
            if lane == "long":
                lens = [len(p) for p in entry[0]]
                shared = cut or 0  # the state once, then every prompt's own tokens
                self.long_entries.append(entry + (time.perf_counter(), shared + sum(n - shared for n in lens)))
                self.long_ready.set()
            else:
                await self.queue.put(entry)
            futs.append(fut)
        outs = []
        try:
            for e, fut in zip(entries, futs):
                res, j = await fut, 0
                for b in e:
                    outs.append(res[j: j + len(b[3])]); j += len(b[3])
        finally:
            for fut in futs:
                self.waiting.pop(fut, None)
            if wait[0]:
                self.queued -= 1
        return outs

    def cut_of(self, q, branches):
        """Where the engine's state cache cuts a request's prompts (None: no cut). --state-cache-gb: after the state.
        --prefix-cache: after the template prefix (chat template, system prompt, the user header up to the state) when
        everything below it is short (all prompts together at most the engine's PREFIX_SFX[-1] tokens past the
        prefix), unless the request's state is cached itself."""
        cache = getattr(self.backend, "state_cache", None)
        if cache is None:
            return None
        cut = self.state_cut(q) if getattr(self, "states_cached", True) else None
        if getattr(self.backend, "prefix", False):
            if cut and cache.has(cache.key(branches[0][3][0][:cut])):
                return cut
            t = self.template_cut(q, branches[0][3][0])
            if t and sum(len(p) - t for b in branches for p in b[3]) <= self.backend.PREFIX_SFX[-1]:
                return t
        return cut

    def template_cut(self, q, ids):
        """Tokens of the template prefix of a question's prompts (everything before the state: chat template, system
        prompt and the user header "State:" / "Stan:"), checked against the prompt's token ids; None when not found."""
        head = FORMATS[self.fmt]["templates"][q["lang"]]["user"].split("{state}")[0]
        text = render(self.tok, q["state"], q["question"], q["options"], q["lang"], self.fmt)
        i = text.find(head + q["state"]) if q["state"] else -1
        if i < 0:
            return None
        memo = self.__dict__.setdefault("_template_ids", {})
        pre = memo.get(text[: i + len(head)])
        if pre is None:
            pre = memo[text[: i + len(head)]] = self.tok(text[: i + len(head)], add_special_tokens=False).input_ids
        n = len(pre)
        while n and list(ids[:n]) != pre[:n]:  # the last token merges with what follows: end earlier
            n -= 1
        return n or None

    def state_cut(self, q):
        """Tokens of the state prefix (chat template up to the end of the state) of a question's prompts, where the
        state cache keys a request; None when it cannot be found (the prompt does not contain the state as rendered).
        The state is found after the user message's header ("State:" / "Stan:"), not where its text first occurs (a
        short state can also be a word of the system prompt)."""
        head = FORMATS[self.fmt]["templates"][q["lang"]]["user"].split("{state}")[0]
        text = render(self.tok, q["state"], q["question"], q["options"], q["lang"], self.fmt)
        i = text.find(head + q["state"]) if q["state"] else -1
        if i < 0:
            return None
        full = self.tok(text, add_special_tokens=False).input_ids
        pre = self.tok(text[: i + len(head) + len(q["state"])], add_special_tokens=False).input_ids
        while pre and full[: len(pre)] != pre:  # the last token merges with what follows: end one token earlier
            pre = pre[:-1]
        return len(pre) or None

    def uncertain(self, q, perms, outs):
        """--orders auto: does a branch get its second option order? Yes/no branches (noul, multi labels, act on a
        yes/no question) when P(true) is inside the band, the others when the top probability is below tau (both on
        the calibrated first-order probabilities)."""
        t = "noul" if q["type"] == "multi" else q.get("base", q["type"])
        p = self.canonical(t, perms, outs)
        if t == "noul":
            return self.band[0] <= p[0] <= self.band[1]
        return max(p) < self.tau

    def code_ids(self, prompt, k):
        """Option code ids at the answer position. They depend only on the number of options, so the ids of all codes of
        the format are computed once (first prompt) and sliced; per k if one of them is not a single token."""
        if k not in self.letters:
            if "all" not in self.letters:
                self.letters["all"] = letter_ids(self.tok, prompt, self.max_options, self.fmt)
            full = self.letters["all"]
            self.letters[k] = full[:k] if full else letter_ids(self.tok, prompt, k, self.fmt)
        return self.letters[k]

    async def worker(self):
        loop = asyncio.get_running_loop()
        # engine calls in flight: 2 for the CUDA graph engine (one call's host work overlaps the other's forwards)
        slots = asyncio.Semaphore(max(1, getattr(self.backend, "inflight", 1)))
        last = 1  # queue entries in the previous call
        while True:
            await slots.acquire()
            # with the scheduler and long requests waiting, the call starts without waiting for a new request
            batch = [] if getattr(self, "backlog", None) else [await self.queue.get()]
            # adaptive batching: take everything already waiting (requests queue up while the GPU is busy). An idle
            # server answers immediately; under load (the previous call had several requests) it waits up to
            # --batch-window-ms for as many as before, since concurrent clients send their next requests right after
            # their answers and would otherwise be split into a small and a large forward. --wait-ms > 0: always wait.
            while len(batch) < self.max_batch and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            window = getattr(self, "window", 0.0) if last > 1 else 0.0
            t_end = loop.time() + max(self.wait, window)
            while len(batch) < self.max_batch and (self.wait > 0 or len(batch) < last):
                left = t_end - loop.time()
                if left <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), left))
                except asyncio.TimeoutError:
                    break
            if getattr(self, "backlog", None) is not None:
                batch = self.schedule(batch)
            last = max(1, len(batch))
            loop.create_task(self.forward(batch, slots))

    def schedule(self, entries):
        """--scheduler on: the queue entries of the next engine call. Short requests (longest prompt at most
        --long-tokens) go into the call at once; long ones wait in the backlog and join calls in HRRN order (response
        ratio (waiting + expected service) / expected service: shorter long requests first, and one that has waited
        long enough before shorter ones), at most SCHED_CALL_TOKENS of them per call (at least one). A short request
        then waits behind at most one call's bounded long work instead of every long request queued before it."""
        now = time.perf_counter()
        call = []
        for e in entries:
            n = max(map(len, e[0]), default=0)
            if n > self.sched_tokens:
                self.backlog.append((e, now, n))
            else:
                call.append(e)
        if self.backlog:
            rate = self.sched_rate
            self.backlog.sort(key=lambda x: ((now - x[1]) / (x[2] * rate) + 1, -x[2]), reverse=True)
            budget, keep = self.SCHED_CALL_TOKENS, []
            for x in self.backlog:
                if len(call) < self.max_batch and (budget == self.SCHED_CALL_TOKENS or x[2] <= budget):
                    call.append(x[0])
                    budget -= x[2]
                else:
                    keep.append(x)
            self.backlog = keep
        return call

    async def long_worker(self):
        """The long lane (--long-lane): up to LONG_BATCH requests per engine call, the waiting ones with the highest
        response ratios first ((waiting time + expected service time) / expected service time, HRRN: shorter long
        requests go first, and one that has waited long enough goes before shorter ones). Its engine call (run_long) runs in a thread of its own
        and gives the GPU to the main lane between decoder layers (LaneGate)."""
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(1, thread_name_prefix="basal-long-lane")
        self.long_ready = asyncio.Event()  # bound to this loop
        if self.long_entries:
            self.long_ready.set()
        while True:
            await self.long_ready.wait()
            now, rate = time.perf_counter(), self.long_rate
            order = sorted(self.long_entries, key=lambda e: ((now - e[5]) / (e[6] * rate) + 1, -e[6]), reverse=True)
            batch, total = [], 0  # the highest response ratios first, up to LONG_BATCH requests / LONG_BATCH_TOKENS
            for e in order:
                if batch and (len(batch) >= self.LONG_BATCH or total + e[6] > self.LONG_BATCH_TOKENS
                              or e[3] != batch[0][3]):
                    continue
                batch.append(e)
                total += e[6]
            for e in batch:
                self.long_entries.remove(e)
            self.started(batch)
            if not self.long_entries:
                self.long_ready.clear()
            groups = [(e[0], e[1]) + ((e[4],) if e[4] else ()) for e in batch]
            t0 = time.perf_counter()
            self.gate.start()
            try:
                out = await loop.run_in_executor(pool, functools.partial(self.backend.run_long, groups, batch[0][3],
                                                                         between=self.gate.between))
                for e, o in zip(batch, out):
                    _answer(e[2], o)
                self.long_rate = 0.8 * self.long_rate + 0.2 * (time.perf_counter() - t0) / max(total, 1)
            except Exception as ex:  # noqa: BLE001
                _fail([e[2] for e in batch], ex)

    def lane_of(self, branches, longest, cut):
        """--long-lane: a request whose longest prompt has more than --long-tokens tokens goes to the long lane, unless
        its state is in the state cache (then only its questions and options are computed, which is short work)."""
        if getattr(self, "long_tokens", None) is None or max(longest, default=0) <= self.long_tokens:
            return "main"
        cache = getattr(self.backend, "state_cache", None)
        if cache is not None and cut and cache.has(cache.key(branches[0][3][0][:cut])):
            return "main"
        return "long"

    async def forward(self, batch, slots):
        """One engine call for a batch of queue entries. A streaming engine answers every entry as soon as its forwards
        are read out, so its clients can send their next requests while the rest of the call still runs."""
        loop = asyncio.get_running_loop()
        t_call = time.perf_counter()
        if not batch:
            slots.release()
            return
        self.started(batch)
        try:
            for pol in {b[3] for b in batch}:  # requests with the same early-exit policy run together
                ix = [i for i, b in enumerate(batch) if b[3] == pol]
                kw = {}
                if getattr(self.backend, "streams", False):
                    kw["done"] = lambda j, res, ix=ix: loop.call_soon_threadsafe(_answer, batch[ix[j]][2], res)
                groups = []
                for i in ix:  # (prompts, code ids[, state prefix length for the state cache])
                    cut = batch[i][4] if len(batch[i]) > 4 else None
                    groups.append((batch[i][0], batch[i][1]) + ((cut,) if cut else ()))
                out = await loop.run_in_executor(None, functools.partial(self.backend.run_shared, groups, pol, **kw))
                for i, o in zip(ix, out):
                    _answer(batch[i][2], o)
        except Exception as e:  # noqa: BLE001
            _fail([b[2] for b in batch], e)
        finally:
            slots.release()
            if getattr(self, "gate", None) is not None:
                self.gate.main_done(len(batch), time.perf_counter() - t_call)
            if getattr(self, "backlog", None) is not None:  # seconds per prompt token, for the HRRN ratios
                tokens = sum(max(map(len, b[0]), default=0) for b in batch)
                self.sched_rate = 0.8 * self.sched_rate + 0.2 * (time.perf_counter() - t_call) / max(tokens, 1)

    def canonical(self, q_type, perms, outs):
        """Position probabilities of every option order -> mean in the canonical order, then the type temperature."""
        canon = []
        for perm, p_pos in zip(perms, outs):
            c = [0.0] * len(perm)
            for k, j in enumerate(perm):
                c[j] = p_pos[k]
            canon.append(c)
        p = torch.tensor([sum(x) / len(canon) for x in zip(*canon)])
        T = temperature(self.cal, q_type, len(p))  # calibrated temperature per question type (and option count)
        return temper(p, T).tolist()

    async def decide(self, body):
        t0 = time.perf_counter()
        orig, plans = body, None
        if any(tasks.is_task(q) for q in body["questions"].values()):  # classify / verify / compress / extract
            body, plans = tasks.expand(body, self.max_options)
        pol = body.get("early_exit")
        if pol is not None:
            pol = str(pol)
            allowed = sorted(getattr(self.backend, "policies", {}))
            if pol not in allowed:
                raise RequestError(f"early_exit={pol!r} not available (server mode must be fast-exit); "
                                   f"allowed: {allowed}")
        state = body["state"]
        if body.get("facts", "off") not in ("off", "auto"):
            raise RequestError('facts must be "off" or "auto"')
        if body.get("facts") == "auto":  # append computed calendar / arithmetic facts (PL dates, gaps, VAT)
            state = inject_facts(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
        m = self.media_of(body, pol)
        lang = body.get("lang")  # the prompt language when the text alone does not show it (a state that is only media)
        if lang is not None and lang not in ("pl", "en"):
            raise RequestError('lang must be "pl" or "en"')
        qs = to_items(state, body["questions"], self.max_options, force_lang=lang)
        for q in qs:
            stored_fallback(self.cal, q)
        cap = getattr(self.backend, "max_codes", None)  # an engine that reads fewer option codes (Ollama: 20)
        for q in qs:
            if cap is not None and len(q["options"]) > cap:
                raise RequestError(f"question {q['name']!r}: {len(q['options'])} options, but "
                                   f"{type(self.backend).__name__} reads at most {cap} "
                                   f"({self.backend.max_codes_reason})")
        # (question index, branch index, perms, prompts, code ids): every branch in the first pass's option orders
        branches = [self.branch(i, bi, q) for i, q in enumerate(qs) for bi in range(len(q["branches"]))]
        truncation, cut, lane, n_media = None, None, "main", 0
        if m is not None:  # media: one forward per prompt (eager) or one vLLM call per request; no queue or truncation
            outs, n_tok, longest, n_media = await self.ask_media(qs, branches, m)
        else:
            qs, branches, outs, n_tok, longest, truncation, cut, lane = await self.ask_text(body, qs, branches, pol)
        second = []
        if self.orders == "auto":  # the reversed option order, only for the branches that are uncertain in order 1
            redo = [j for j, (b, o) in enumerate(zip(branches, outs)) if self.uncertain(qs[b[0]], b[2], o)]
            second = [self.branch(branches[j][0], branches[j][1], qs[branches[j][0]], [branches[j][2][0][::-1]])
                      for j in redo]
            if second:
                if m is not None:
                    o2, n2, _, nm2 = await self.ask_media(qs, second, m)
                    n_tok, n_media = n_tok + n2, n_media + nm2
                else:
                    n_tok += self.encode(second)[0]
                    o2 = await self.ask(second, pol, cut, lane)
                for j, b2, o in zip(redo, second, o2):
                    branches[j][2].extend(b2[2]); outs[j] = outs[j] + o
        out = await self.assemble(body, qs, branches, outs, m, n_tok, n_media, longest, second, truncation, lane, t0)
        if plans:  # the task answers from their decision answers; extract's spans need a second round of decisions
            r2, a2 = tasks.second_round(out, orig, plans, self.max_options), None
            if r2 is not None:
                out2 = await self.decide(r2)
                a2 = out2["answers"]
                for k in ("input_tokens", "branches"):
                    out["usage"][k] = out["usage"].get(k, 0) + out2["usage"].get(k, 0)
            out = tasks.finish(out, orig, plans, self.cal, a2)
            out["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    @reads_request
    def media_of(self, body, pol):
        """The request's media decoded (basal/media.py), or None without media."""
        if body.get("media") is None:
            return None
        from .media import decode, detail_tokens
        kinds = getattr(self.backend, "media_kinds", ())
        if not kinds:
            raise RequestError("this server does not take media (basal-serve --mode eager or vllm with --media on, and "
                               "a model that has an image / audio encoder)")
        if pol is not None:
            raise RequestError("early_exit is not available with media")
        if any(q.get("evidence") for q in body["questions"].values() if isinstance(q, dict)):
            raise RequestError("evidence is not available with media")
        soft = detail_tokens(body.get("detail", "default"))
        images, audio, order = decode(body["media"], kinds, getattr(self.backend, "max_audio_s", None))
        return dict(images=images, audio=audio, kinds=order, max_soft_tokens=soft, detail=body.get("detail", "default"))

    async def ask_media(self, qs, branches, m):
        """Position probabilities of every prompt of every branch, with the request's media -> (outs, prompt tokens,
        the longest prompt of each branch, media tokens)."""
        if getattr(self, "layout", "canonical") != "canonical":
            raise RequestError(f"images and audio are read in the canonical layout only (this server: --layout "
                               f"{self.layout})")
        msgs, ids, where = [], [], []
        for j, b in enumerate(branches):
            q = qs[b[0]]
            question, options = q["branches"][b[1]]
            for perm, lid in zip(b[2], b[4]):
                msgs.append(media_messages(q["state"], question, [options[c] for c in perm], m["kinds"], q["lang"],
                                           self.fmt))
                ids.append(lid)
                where.append(j)
        limit = getattr(self, "context", None)
        res = await asyncio.get_running_loop().run_in_executor(None, functools.partial(
            self.backend.run_media, msgs, ids, m["images"], m["audio"], m["max_soft_tokens"], limit[0] if limit else None))
        outs, longest = [[] for _ in branches], [0] * len(branches)
        for j, (p, n, _) in zip(where, res):
            outs[j].append(p)
            longest[j] = max(longest[j], n)
        return outs, sum(n for _, n, _ in res), longest, sum(nm for _, _, nm in res)

    async def ask_text(self, body, qs, branches, pol):
        """The text path (as before media): prompts tokenized, the context checked (or the state truncated, which
        rebuilds the questions and branches), the queue -> (questions, branches, outs, prompt tokens, the longest
        prompt of each branch, truncation, state cut, lane)."""
        n_tok, longest = self.encode(branches)
        truncation, limit, orig, excess = None, getattr(self, "context", None), qs[0]["state"], 0
        for _ in range(2):  # --context-overflow truncate: the state's head and tail, the questions and options whole
            if getattr(self, "overflow", "refuse") != "truncate" or not limit or max(longest, default=0) <= limit[0]:
                break
            excess += max(longest) - limit[0]
            state, t = self.truncate_state(orig, excess)
            if t is None:
                break
            truncation = t
            langs = [q["lang"] for q in qs]
            qs = to_items(state, body["questions"], self.max_options, force_lang=body.get("lang"))
            for q, lang in zip(qs, langs):
                q["lang"] = lang  # the language of the whole state
                stored_fallback(self.cal, q)
            branches = [self.branch(i, bi, q) for i, q in enumerate(qs) for bi in range(len(q["branches"]))]
            n_tok, longest = self.encode(branches)
        self.check_context(qs, branches, longest)
        cut = self.cut_of(qs[0], branches)
        lane = self.lane_of(branches, longest, cut)
        outs = await self.ask(branches, pol, cut, lane)
        return qs, branches, outs, n_tok, longest, truncation, cut, lane

    async def assemble(self, body, qs, branches, outs, m, n_tok, n_media, longest, second, truncation, lane, t0):
        """Answers and usage of a request from the position probabilities of its branches."""
        loop = asyncio.get_running_loop()
        per_q = [[] for _ in qs]
        for b, o in zip(branches, outs):
            per_q[b[0]].append((b[2], o))
        answers = {}
        for q, brs in zip(qs, per_q):
            if q["type"] == "multi" or q.get("multi"):
                ans = multi_answer(q, [self.canonical("noul", perms, o)[0] for perms, o in brs])
                answers[q["name"]] = tool_set_answer(q, ans) if q["type"] == "tool" else ans
                continue
            perms, o = brs[0]
            pl = self.canonical(q.get("base", q["type"]), perms, o)
            probs = {k: float(v) for k, v in zip(q["keys"], pl)}
            conf = float(max(pl))
            if q["type"] == "act":
                ans = act_answer(q, probs, conf, self.cal)
            elif q["type"] == "tool":
                ans = tool_answer(q, probs, conf, self.cal)
            elif q["type"] == "noul":
                ans = {"type": "noul", "noul": probs["true"], "probabilities": probs, "confidence": conf}
            elif q["type"] == "score":
                ans = {"type": "score", "score": float(sum(i * v for i, v in enumerate(pl))),
                       "legend": dict(zip(q["keys"], q["options"])), "probabilities": probs, "confidence": conf}
            else:
                ans = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
            answers[q["name"]] = ans
        ev_q = [q for q in qs if body["questions"][q["name"]].get("evidence")]
        if ev_q and truncation:
            raise RequestError("evidence is not available for a state truncated to the context (shorten the state)")
        if ev_q:  # evidence spans: character offsets into the request's state (facts block excluded)
            if not (self.evidence and self.evidence.available):
                raise RequestError("evidence is not available for this model (no evidence_head.pt)")
            limit = len(_text(body["state"])) if body.get("facts") == "auto" else None
            for q in ev_q:
                if q["type"] == "multi" or q.get("multi"):
                    raise RequestError(f"question {q['name']!r}: evidence is not supported for multi")
                answers[q["name"]]["evidence"] = await loop.run_in_executor(
                    None, self.evidence.spans, q["state"], q["question"], q["options"], q["lang"], limit)
        if getattr(self, "cal_unvalidated", None):  # the calibration was fitted on other option orders
            for ans in answers.values():
                ans["calibration"] = "unvalidated"
        elif m is not None:  # the temperatures and certified thresholds were fitted on text, not on media
            for ans in answers.values():
                ans["calibration"] = "text"
        out = {"model": self.name, "answers": answers,
               "usage": {"input_tokens": n_tok, "output_tokens": 0, "questions": len(qs), "branches": len(branches),
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}
        if m is not None:
            out["usage"]["media"] = {"images": len(m["images"]), "audio": len(m["audio"]), "media_tokens": n_media,
                                     "detail": m["detail"]}
        if self.orders == "auto":
            out["usage"]["second_order_branches"] = len(second)
        out["usage"]["longest_prompt_tokens"] = max(longest, default=0)
        if truncation:
            out["usage"]["truncated"] = truncation
        if getattr(self, "long_tokens", None) is not None:
            out["usage"]["lane"] = lane
        if self.log:
            self.log_decision(body, out)
        return out

    def log_decision(self, body, out):
        """Propensity log (one JSON line per request): the served probabilities q(a) of every question, so production
        feedback can later be used off-policy. `reviewed` is filled in by the caller's feedback, not here."""
        h = hashlib.sha256(json.dumps([body.get("state"), body.get("questions")], sort_keys=True,
                                      ensure_ascii=False, default=str).encode()).hexdigest()
        rec = {"ts": time.time(), "model": self.name, "request_sha256": h, "reviewed": None,
               "answers": {k: {f: v for f, v in a.items() if f in ("type", "choice", "action", "selected", "tool",
                                                                    "tools", "probabilities", "confidence")}
                           for k, a in out["answers"].items()}}
        self.log.write(json.dumps(rec, ensure_ascii=False) + "\n")


def add_engine_args(ap):
    """The engine options of basal-serve (basal-calibrate runs the same engine in process with them)."""
    ap.add_argument("--model", default="Remek/basal-1.5-4.5B", help="local directory or Hugging Face repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", choices=list(MODES), default=None,
                    help="default: fast on CUDA, mlx on Apple Silicon (mps without mlx / mlx-lm), tpu on a TPU VM "
                         "with basal[tpu], eager otherwise")
    ap.add_argument("--quant", choices=["fp8", "nvfp4", "q8"], default=None,
                    help="override the quantisation of the mode (fp8 / nvfp4: CUDA graph modes, q8: mlx)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--media", choices=["off", "on"], default="off",
                    help="on: requests may carry images / audio (\"media\", basal/media.py); --mode eager or vllm "
                         "and a model with an image / audio encoder (basal-omni, Gemma 4)")
    ap.add_argument("--eager-forward", dest="eager_forward", choices=list(EAGER_FORWARDS), default=None,
                    help="--mode eager with a Gemma 4 model (basal-omni): basal = the engine's own forward (the shared "
                         "prefix of a request once, the last layers at the answer rows only; default), hf = the "
                         "transformers forward of basal 2.2.0 (default: BASAL_EAGER_FORWARD, else basal)")
    ap.add_argument("--eager-attn", dest="eager_attn", default="math",
                    help="--mode eager with a Gemma 4 model and the engine's own forward: the attention kernel per layer "
                         "type. math (default) = the math kernel everywhere, as basal 2.2; auto (opt-in) = on Hopper "
                         "FlashAttention 3 in the sliding-window layers (the memory-efficient kernel without "
                         "flash_attn_3) and the memory-efficient kernel in the global layers, the math kernel elsewhere; "
                         "or sliding=K,full=K with K in sdpa, efficient, cudnn, fa3, fa4")
    ap.add_argument("--ple-offload", dest="ple_offload", choices=["off", "on"], default="off",
                    help="--mode eager with Gemma 4 E2B / E4B (basal-omni mini): keep the per-layer embedding table in "
                         "host memory and copy only the rows of each forward's tokens (E4B: 5.6 GB less GPU memory)")
    ap.add_argument("--mlx-packed", dest="mlx_packed", action="store_true",
                    help="--mode mlx, opt-in: a group of a Llama model (basal-1.x, 2.x; static RoPE) whose state and "
                         "option orders fit in 512 tokens runs as one packed forward (explicit positions, block mask) "
                         "instead of a KV cache plus a continuation; faster on short prompts, exact in fp32, but in "
                         "bf16 / 8-bit the probabilities move slightly, so check it against CALIBRATION.mlx.json "
                         "(fitted on the default path) before relying on the calibrated thresholds")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="device for --mode eager (default: BASAL_DEVICE, else cuda, else Apple Silicon mps, else cpu) "
                         "and --mode mps (mps or cpu; default: mps on Apple Silicon, else cpu: a request's state "
                         "computed once for all its prompts, where eager computes it once per prompt)")
    ap.add_argument("--layout", choices=LAYOUTS, default=None,
                    help="prompt layout (prompt-format-2 models trained on it): canonical = the "
                         "state, then the question (questions share the state; state cache); state_twice = the state "
                         "read twice; qfirst = the question also before the state, one pass per question (the state "
                         "is read knowing the question: slower with several questions, no state cache). Default: the "
                         "model's basal.json \"layout\", else canonical. Loads CALIBRATION.<layout>.json when the model "
                         "ships one")
    ap.add_argument("--orders", type=orders_arg, default=None,
                    help="1 = the option order given; 2 = also the reversed order, averaged (shares the state); auto = "
                         "order 1, and the reversed order only where order 1 is uncertain (--auto-tau, --auto-band); "
                         "2le<N> = both orders for questions with at most N options, one above (e.g. 2le3). "
                         "Default: 2 for prompt-format-1 models, 1 for prompt-format-2 models")
    ap.add_argument("--auto-tau", dest="auto_tau", type=float, default=0.6,
                    help="--orders auto: a second order when the top probability is below this (choice, score)")
    ap.add_argument("--auto-band", dest="auto_band", type=float, nargs=2, default=[0.2, 0.8],
                    help="--orders auto: a second order when P(true) of a yes/no branch is inside this band")
    ap.add_argument("--early-exit", dest="early_exit", default="off", help="default policy for --mode fast-exit (basal-1.0 models, which ship exit heads)")
    ap.add_argument("--exit-heads", dest="exit_heads", default=None, help="exit heads dir (default: <model>/exit_heads)")
    ap.add_argument("--soam", choices=["on", "off"], default="on",
                    help="on = the whole request (every question and option order) is one packed row over the state")
    ap.add_argument("--gpu-memory", dest="gpu_memory", type=float, default=0.6, help="vLLM memory fraction")
    ap.add_argument("--vllm-preset", dest="vllm_preset", choices=list(VLLM_PRESETS), default="default",
                    help="--mode vllm: default = as 2.2; deterministic = batch-invariant kernels without the prefix "
                         "cache and chunked prefill (VLLM_BATCH_INVARIANT: answers independent of the other requests, "
                         "lower throughput). Loads CALIBRATION.vllm.<preset>.json when the model ships one, else "
                         "CALIBRATION.vllm.json")
    ap.add_argument("--vllm-fast-prefill", dest="vllm_fast_prefill", choices=["auto", "off"], default="auto",
                    help="--mode vllm: auto = layers that reuse earlier layers' keys and values (Gemma 4 E2B / E4B) "
                         "run at the answer position only during prefill (vLLM's kv_sharing_fast_prefill); off = as 2.2")
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
    ap.add_argument("--attn", choices=list(ATTN), default="auto",
                    help="attention kernel of the CUDA graph engine: auto (FlashAttention 3 on Hopper when installed, "
                         "else 2; FP8 checkpoints and --mode fp8: 2), fa2 (torch SDPA), fa3, fa4 (flash-attn-4), cudnn")
    ap.add_argument("--gemma-attn", dest="gemma_attn", choices=["auto", "math", "flex", "fa3", "fa3+math"],
                    default="auto",
                    help="CUDA graph engine, Gemma 4 (basal-omni): the attention kernel of its layers: math (torch "
                         "SDPA's math kernel, as --mode eager), flex (FlexAttention with block masks: sliding window, "
                         "trie), fa3 (FlashAttention 3 in the sliding-window layers' plain rows, flex elsewhere; "
                         "Hopper), fa3+math (FlashAttention 3 in the sliding layers, math in the global ones). auto: "
                         "fa3 in --mode fast where FlashAttention 3 runs (with merged projections and compact rows), "
                         "flex on Blackwell workstation GPUs (sm_120), else math")
    ap.add_argument("--merge", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graph engine: one q/k/v and one gate/up GEMM per layer (bf16: the same products; FP8 "
                         "checkpoints: the merged weights re-quantised per tensor). auto: on for bf16 weights, off for "
                         "FP8 checkpoints")
    ap.add_argument("--lean", choices=["auto", "on", "pass2", "off"], default="off",
                    help="CUDA graph engine, Llama models: one compiled layer function for every layer and shape "
                         "(basal/lean.py) instead of transformers' module tree, in both passes or in pass 2 only. "
                         "auto: pass2 on the CUDA graph engine (with --tree auto / flex), else off")
    ap.add_argument("--tree", choices=["auto", "dense", "flex"], default="dense",
                    help="with --lean pass2: the pass-2 trie mask as a dense additive mask (KV heads repeated) or as a "
                         "FlexAttention mask over the KV heads as they are. auto: flex with --lean pass2, else dense")
    ap.add_argument("--state-cache-gb", dest="state_cache_gb", type=float, default=0,
                    help="CUDA graph engine: keep the keys and values of request states on the GPU up to this many GB "
                         "(least recently used evicted); a later request with the same state computes only its "
                         "questions and options (0: off)")
    ap.add_argument("--prefix-cache", dest="prefix_cache", choices=["on", "off"], default="off",
                    help="CUDA graph engine: compute the template prefix (chat template, system prompt, the header up "
                         "to the state) once and run short requests over its cached keys and values")
    ap.add_argument("--batches", choices=["auto", "compact", "default", "fine"], default="auto",
                    help="CUDA graph engine: compact = single prompts packed back to back into one row (FlashAttention "
                         "3 varlen on Hopper; elsewhere, sm80+, torch's varlen attention, Llama models), no padding to "
                         "graph lengths and row counts; default = rows padded to graph lengths; fine = more row "
                         "counts (less padding of short batches, longer start-up); auto = compact where FlashAttention "
                         "3 runs, else default")
    ap.add_argument("--readout-layer", dest="readout_layer", choices=["auto", "on", "off"], default="off",
                    help="CUDA graph engine with compact rows, Llama models: the last decoder layer computes keys, values "
                         "and attention for every token and its o_proj and MLP at the readout positions only (auto: on; off by "
                         "default: +1.6% at best on the H100, below the v2.0 speed gate)")
    ap.add_argument("--template-prefix", dest="template_prefix", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graph engine with compact rows: the template prefix of the prompts (chat template, system "
                         "prompt, the user header up to the state) is computed once at start-up and every prompt runs "
                         "only its own tokens over its keys and values (auto: on)")
    ap.add_argument("--fp8-accum", dest="fp8_accum", choices=["fast", "full"], default="fast",
                    help="FP8 checkpoints in the CUDA graph engine: cuBLAS fast accumulation (rounding depends on the "
                         "batch's kernel) or full accumulation (decisions independent of batching, as --deterministic)")
    ap.add_argument("--context-overflow", dest="context_overflow", choices=["truncate", "refuse"], default="truncate",
                    help="a prompt longer than the context: truncate = drop the middle of the state (its head and tail "
                         "kept, the questions and options whole; marked in usage.truncated), refuse = HTTP 422")
    ap.add_argument("--truncate-head", dest="truncate_head", type=float, default=1 / 3,
                    help="--context-overflow truncate: the share of the kept state tokens taken from its start (the "
                         "rest from its end)")
    ap.add_argument("--scheduler", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graph engine: requests whose longest prompt has more than --long-tokens tokens wait in a "
                         "backlog and join engine calls in HRRN order, a bounded amount per call, so short requests do "
                         "not queue behind them (auto: on; off with --long-lane or --deterministic)")
    ap.add_argument("--long-lane", dest="long_lane", choices=["on", "off"], default="off",
                    help="CUDA graph engine: requests whose longest prompt has more than --long-tokens tokens run in a "
                         "lane of their own, one at a time and one decoder layer at a time, giving the GPU to the "
                         "batches of short requests (whose answers then do not wait for a long document)")
    ap.add_argument("--long-tokens", dest="long_tokens", type=int, default=2048,
                    help="--long-lane: the prompt length (tokens) above which a request is long")
    ap.add_argument("--long-share", dest="long_share", type=float, default=0.5,
                    help="--long-lane: the long lane yields to waiting short batches only once its recent share of "
                         "the GPU's busy time reaches this (0: always yield, 1: never)")
    ap.add_argument("--long-slice-ms", dest="long_slice_ms", type=float, default=10.0,
                    help="--long-lane: the long lane runs at least this long before it gives the GPU to a waiting "
                         "short batch")
    ap.add_argument("--deterministic", action="store_true",
                    help="answers independent of load: one request per forward, deterministic kernels, FP8 GEMMs with "
                         "full accumulation (lower throughput under concurrency; in-process engines only)")
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=64)
    ap.add_argument("--wait-ms", dest="wait_ms", type=float, default=0.0,
                    help="extra time to wait for more requests before a forward (default 0: adaptive batching)")
    ap.add_argument("--batch-window-ms", dest="batch_window_ms", type=float, default=3.0,
                    help="under load, wait up to this long for as many requests as the previous forward had (0: off)")
    return ap


def parser():
    ap = argparse.ArgumentParser(description="basal typed-decision server")
    add_engine_args(ap)
    ap.add_argument("--name", default=None, help="model name reported in responses (default: last part of --model)")
    ap.add_argument("--no-calibration", dest="no_calibration", action="store_true")
    ap.add_argument("--calibration", default=None,
                    help="calibration file (default: <model>/CALIBRATION.<engine>.json if shipped, else "
                         "CALIBRATION.json; fit your own with basal-calibrate)")
    ap.add_argument("--strict-calibration", dest="strict_calibration", action="store_true",
                    help="refuse to start when the --calibration file was fitted on another model or revision "
                         "(default: a warning)")
    ap.add_argument("--log-decisions", dest="log_decisions", default=None, help="append a propensity log (JSONL)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mcp", choices=["off", "on"], default="off",
                    help="also serve MCP (streamable HTTP) at /mcp for agents: tools decide and decide_many over the "
                         "same queue (needs basal[mcp]; no authentication, like the HTTP API: bind --host 127.0.0.1 "
                         "for agents on this machine)")
    from .think import add_think_args  # --think off|auto (experimental; off = this server unchanged)
    add_think_args(ap)
    return ap


def http_app(srv, mode, mcp=None):
    """The app of basal-serve: POST /v1/systemone and /v1/basal, GET /v1/models, /health and /metrics;
    mcp: (routes, lifespan) of --mcp on. A failed request answers {"error": message} with http_status:
    422 invalid request, 503 out of GPU memory, 500 any other server error."""
    from contextlib import asynccontextmanager, nullcontext

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from . import __version__

    labels = [v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') for v in (__version__, srv.name, mode)]
    build_info = f'basal_build_info{{version="{labels[0]}",model="{labels[1]}",mode="{labels[2]}"}} 1'
    cache_snapshot, cache_updated = {}, 0

    def failed(e):
        print(f"basal: request failed: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
        if isinstance(e, torch.OutOfMemoryError):  # the failed forward's blocks back to the GPU: kept by the
            torch.cuda.empty_cache()  # allocator, they failed the next requests too (ultra, 20 Oct)
        return JSONResponse({"error": str(e)}, status_code=http_status(e))

    async def systemone(request):
        try:
            body = await request.json()
        except ValueError as e:  # not JSON (or not UTF-8)
            return failed(RequestError(f"the request body is not valid JSON: {e}"))
        try:
            return JSONResponse(await srv.serve(body))
        except Exception as e:  # noqa: BLE001
            return failed(e)

    async def models(_):
        return JSONResponse(models_payload(srv.name, mode, getattr(srv.backend, "policies", {}), srv.temps,
                                           srv.fmt, srv.orders, getattr(srv.backend, "media_kinds", ()),
                                           getattr(srv, "cal_mismatch", None), getattr(srv.backend, "preset", None),
                                           getattr(srv, "layout", None)))

    async def health(_):
        return JSONResponse({"status": "ok", **srv.load()})

    async def metrics(_):
        nonlocal cache_snapshot, cache_updated
        lines = ["# HELP basal_build_info Engine version, served model and mode.", "# TYPE basal_build_info gauge",
                 build_info]

        def add(name, kind, help_text, value):
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name} {value}"))

        load = srv.load()
        add("basal_requests_running", "gauge", "Requests in flight outside the engine queue.", load["running"])
        add("basal_requests_queued", "gauge", "Requests waiting for the engine.", load["queued"])
        cache = getattr(srv.backend, "state_cache", None)
        if cache is not None:
            snapshot = cache.stats(blocking=False)
            if snapshot is not None:
                cache_snapshot, cache_updated = snapshot, time.time()
            for key, kind, suffix in (("entries", "gauge", "entries"), ("used_bytes", "gauge", "used_bytes"),
                                      ("budget_bytes", "gauge", "budget_bytes"), ("hits", "counter", "hits_total"),
                                      ("misses", "counter", "misses_total"), ("evictions", "counter", "evictions_total"),
                                      ("prefix_hits", "counter", "prefix_hits_total")):
                if key in cache_snapshot:
                    add(f"basal_state_cache_{suffix}", kind, f"State cache {key.replace('_', ' ')}.", cache_snapshot[key])
            add("basal_state_cache_snapshot_timestamp_seconds", "gauge",
                "Unix timestamp of the last successful cache snapshot, or 0 before the first.", cache_updated)
        return Response("\n".join(lines) + "\n",
                        headers={"content-type": "text/plain; version=0.0.4; charset=utf-8"})

    mcp_routes, mcp_run = mcp or ([], None)

    @asynccontextmanager
    async def lifespan(_app):
        tasks = [asyncio.get_running_loop().create_task(srv.worker())]
        if srv.long_tokens is not None:
            tasks.append(asyncio.get_running_loop().create_task(srv.long_worker()))
        async with mcp_run() if mcp_run else nullcontext():
            yield
        for task in tasks:
            task.cancel()

    return Starlette(routes=[Route("/v1/systemone", systemone, methods=["POST"]),
                             Route("/v1/basal", systemone, methods=["POST"]), Route("/v1/models", models),
                             Route("/health", health), Route("/metrics", metrics), *mcp_routes], lifespan=lifespan)


def main():
    a = parser().parse_args()
    refuse_lean_flex(a.lean, a.tree)  # before default_mode, which resolves the model on CUDA GPUs without FA3
    a.mode = a.mode or default_mode(a.model, a.revision)
    validate_quant(a.mode, a.quant)
    if a.mcp == "on":
        from .mcp_server import available
        missing = available()
        if missing:
            raise SystemExit(missing)
    import uvicorn

    t0 = time.time()
    if a.think == "off":
        srv = Server(a)
    else:
        from .think import ThinkServer
        srv = ThinkServer(a)
    attn = getattr(srv.backend, "attn", None)
    rows = getattr(srv.backend, "compact_kernel", None)  # compact rows: their kernel, and the template prefix with them
    info = (f" attention {attn}" if attn else "") + (f" compact rows {rows}" if rows else "") \
        + (" template prefix" if rows and getattr(srv.backend, "template", None) else "")
    # everything built so far (weights, graphs, compiled code) lives as long as the server: out of the garbage
    # collector's sight, so a full collection no longer walks it (up to 1.4 s each at the end of start-up on 4.5B, and
    # a 2.9 s stall of one request under nsys, 20 Oct)
    gc.collect()
    gc.freeze()
    print(f"basal: model {a.model} mode {a.mode}{info} ready in {time.time() - t0:.0f}s on port {a.port}", flush=True)

    mcp = None
    if a.mcp == "on":
        from .mcp_server import http
        mcp = http(srv, a.host)
        print(f"basal: MCP (streamable HTTP) at http://{a.host}:{a.port}/mcp", flush=True)
    app = http_app(srv, a.mode, mcp)
    try:  # uvloop + httptools when available (uvicorn[standard]); plain asyncio otherwise
        import httptools  # noqa: F401
        import uvloop  # noqa: F401
        fast = dict(loop="uvloop", http="httptools")
    except ImportError:
        fast = {}
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **fast)


if __name__ == "__main__":
    main()
