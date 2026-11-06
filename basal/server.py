"""HTTP server for typed decisions (System One-compatible JSON interface).

  POST /v1/systemone   {"state": "...", "questions": {"q1": {"type": "choice"|"noul"|"score"|"act"|"multi"|"tool",
                        "instructions": "...", "criteria": {...} | [...], "option_keys": "show"|"hide" (optional),
                        "evidence": true (optional; models with evidence_head.pt: supporting spans of the state)}},
                        "early_exit": "off"|"0.99"|... (optional), "facts": "off"|"auto" (optional)}
                       also the shape of vLLM's structured-diffusion example: "ask_if" / "depends_on" on a question (a
                       skipped question is answered null) and "images" for image media
  POST /v1/basal       same as /v1/systemone; also the experimental task types classify, verify, compress, extract
                       (basal/tasks.py) and rank (any number of candidates, basal/rank.py; on /v1/systemone too)
  POST /v1/decisions   OpenAI's Decisions API (predicate / choice / score), decided as /v1/systemone (basal/decisions.py)
  GET  /v1/models
  GET  /health

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
import itertools
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

from . import adapter as adapter_module
from .engine import (ATTN, EAGER_FORWARDS, EagerBackend, ExitGraphBackend, GGUFBackend, GraphBackend, LlamaCppBackend,
                     FP8_BF16_LAYERS, MLXBackend, MPSBackend, OllamaBackend, SGLangBackend,
                     VLLM_PRESETS, VLLMBackend, cache_signature, fp8_refused, gemma_fast_kernels, recurrent_layers,
                     recurrent_refused, resolve, set_fp8_fast_accum)
from .evidence import Evidence
from .metrics import prometheus  # noqa: F401  (GET /metrics; torch-free, so basal-router imports it alone)
from .facts import MAX_CHARS as FACTS_MAX_CHARS, inject as inject_facts
from .gdn_graph import default_fast as gdn_default_fast, is_gdn, refused as gdn_refused
from . import rank, tasks
from .prompt import (FORMATS, LAYOUTS, MAX_OPTIONS, OptionPrompt, RequestError, defang, lang_of, letter_ids,
                     max_options, media_messages, model_format, model_layout, model_option_isolation, model_two_way,
                     once_ids, option_tokens, reads_request, render, tokens_once)
from .sets import label_set_id
from .sets import select as select_set
from .sets import validate as validate_set
from .tools import QUESTION as TOOL_Q
from .tools import QUESTION_SET as TOOL_SET_Q
from .tools import answer as tool_answer
from .tools import parse as parse_tools
from .tools import set_answer as tool_set_answer

RELEASE_DATE = "2026-11-05"  # the engine's release date (GET /v1/models), not the served model's

MODES = {
    # mode: (backend, quantisation, compile)
    "eager": ("eager", None, False),        # reference PyTorch forward, any GPU (CUDA, Apple MPS) or CPU
    "fast": ("graph", None, True),          # bf16 + torch.compile + CUDA graphs + shared prefix (recommended)
    "fast-nocompile": ("graph", None, False),  # same without torch.compile (faster start-up, ~1.4x slower on H100)
    "fast-exit": ("exit", None, True),      # "fast" + trained early exits, policy chosen per request
    "fp8": ("graph", "fp8", True),          # "fast" with torchao FP8 (Hopper / Blackwell)
    "nvfp4": ("graph", "nvfp4", True),      # "fast" with torchao NVFP4 (Blackwell, experimental)
    "int8": ("graph", "int8", True),        # "fast" with block-scaled int8 (basal-omni ultra; Ampere and newer)
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


NVFP4_AMAX = "NVFP4_AMAX.json"  # a model folder's own calibrated input amax for --mode nvfp4 (basal-omni ultra)


def nvfp4_args(a, md=None, quant=None):
    """--nvfp4-amax / --nvfp4-scope as GraphBackend arguments (none when not given). Without --nvfp4-amax, --mode
    nvfp4 reads the model folder's NVFP4_AMAX.json when it ships one."""
    path = getattr(a, "nvfp4_amax", None)
    if not path and quant == "nvfp4" and md is not None and (Path(md) / NVFP4_AMAX).exists():
        path = Path(md) / NVFP4_AMAX
    if not path:
        return {}
    from .gemma4_graph import load_nvfp4_amax
    return dict(nvfp4_amax=load_nvfp4_amax(str(path)), nvfp4_scope=getattr(a, "nvfp4_scope", "all"))


def nvfp4_calibration_refused(md, kind, quant, a):
    """None, or why basal-omni in NVFP4 or int8 is refused: it serves only with its own calibration
    (CALIBRATION.nvfp4.json / CALIBRATION.int8.json in the model folder, or --calibration). NVFP4 changes about 2% of
    the top answers and makes them more confident, so the thresholds fitted on bf16 readouts do not hold; int8 passed
    its accuracy check with its own file."""
    if quant not in ("nvfp4", "int8") or kind not in ("graph", "exit") or a.calibration \
            or getattr(a, "no_calibration", False):
        return None
    from .gemma4_graph import is_gemma4_dir
    if not is_gemma4_dir(md) or (Path(md) / f"CALIBRATION.{quant}.json").exists():
        return None
    name = quant.upper() if quant == "nvfp4" else quant
    return (f"--mode {quant} with basal-omni needs its own calibration: CALIBRATION.{quant}.json in the model folder "
            f"or --calibration <file> (fitted on {name} readouts; the bf16 thresholds do not hold)")


def default_mode(model=None, revision=None):
    """fast on CUDA, mlx on Apple Silicon when mlx and mlx-lm are installed, mps otherwise there, tpu on a TPU VM
    with basal[tpu] installed, else eager. basal-omni (Gemma 4; model: its directory or repo id) gets fast only on the
    GPU classes where its fast configuration passed its gate (gemma_fast_kernels: Hopper with flash_attn_3, Blackwell
    workstation GPUs), eager on other GPUs until they pass theirs (2.2.0 had no working default for it there: fast
    stopped at "'Gemma4Model' object has no attribute 'layers'"). A model with recurrent-state layers gets eager on CUDA
    and Apple Silicon (mps refuses it; mlx serves Qwen3.5 only when asked, --mode mlx), except a Qwen3.5 model (Gated
    DeltaNet), which gets fast on the GPU class where the graph engine's port passed its gate (gdn_graph.default_fast:
    Hopper with fla)."""
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


def _config_unreadable(probe, model, e):
    """The warning for a model whose config.json a probe could not read: the probe answers its default, the cause
    (a permission error, a full disk) is no longer swallowed with it."""
    print(f"basal: warning: {probe}: cannot read the config of {model}: {type(e).__name__}: {str(e)[:300]}",
          file=sys.stderr, flush=True)


def _gemma4(model, revision=None):
    """A Gemma 4 model (basal-omni), from its config.json; False when that cannot be read (the server says why)."""
    from .gemma4_graph import is_gemma4_dir
    try:
        return is_gemma4_dir(resolve(model, revision, metadata_only=True), strict=True)
    except Exception as e:  # offline without the model cached, a wrong name: the model load reports it
        _config_unreadable("_gemma4", model, e)
        return False


def _recurrent(model, revision=None):
    """The recurrent-state layers of a model (recurrent_layers), from its config.json; None when that cannot be read
    (the model load reports it)."""
    try:
        return recurrent_layers(resolve(model, revision, metadata_only=True), strict=True)
    except Exception as e:
        _config_unreadable("_recurrent", model, e)
        return None


def _gdn(model, revision=None):
    """A Qwen3.5 model of Gated DeltaNet and attention layers (gdn_graph.is_gdn: the recurrent models the CUDA graph
    engine serves), from its config.json; False when that cannot be read."""
    try:
        return is_gdn(resolve(model, revision, metadata_only=True), strict=True)
    except Exception as e:
        _config_unreadable("_gdn", model, e)
        return False


def validate_quant(mode, quant):
    """Reject quantisation overrides that the selected backend cannot apply (before any model is loaded)."""
    if quant is None:
        return
    kind = MODES[mode][0]
    if quant == "q8" and kind == "mlx" or quant in ("fp8", "nvfp4", "int8") and kind in ("graph", "exit"):
        return
    raise SystemExit(f"--quant {quant} is not supported with --mode {mode} "
                     "(q8 only for mlx; fp8, nvfp4 and int8 only for the CUDA graph modes)")


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


def listed_options(name, crit):
    """A list of option keys -> {key: None}. A key that is a non-finite number (NaN, or 1e999 read as inf) is refused:
    it became a key of the answer's probabilities, which are then no JSON (HTTP 500 after the engine's run)."""
    if any(isinstance(k, float) and not math.isfinite(k) for k in crit):
        raise RequestError(f"question {name!r}: an option key must be a string or a finite number")
    return {k: None for k in crit}


def check_body(body):
    """The shape of a decision request (its questions are read by to_items) -> RequestError (HTTP 422)."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    for k in ("state", "questions"):
        if k not in body:
            raise RequestError(f"missing field {k!r}")
    if not isinstance(body["questions"], dict) or not all(isinstance(q, dict) for q in body["questions"].values()):
        raise RequestError('"questions" must be an object of {name: question object}')
    if not body["questions"]:  # (an empty object reached the engine and failed there with a 500)
        raise RequestError('"questions" must name at least one question')


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
        if t not in ("choice", "noul", "score", "act", "multi", "tool"):
            raise RequestError(f"question {name!r}: unknown type {t!r}")
        instr = _text(q.get("instructions", ""))
        lang = force_lang or ("pl" if state_pl else lang_of(instr))
        extra = {}
        lim = limits(name, q, t)
        if t == "multi":  # one yes/no branch per label
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = listed_options(name, crit)
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
                if base not in ("choice", "noul"):
                    raise RequestError(f"question {name!r}: base must be \"choice\" or \"noul\", got {base!r}")
                if isinstance(crit, list):
                    crit = listed_options(name, crit)
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
                crit = listed_options(name, crit)
            keys, opts = named_options(crit, q.get("option_keys", "show"))
        if not 2 <= len(opts) <= max_opts:
            what = " (tools and extra options)" if t == "tool" else ""
            raise RequestError(f"question {name!r}: {len(opts)} options{what} (supported: 2..{max_opts})")
        out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=opts, lang=lang,
                        branches=[(instr, opts)], option_keys=q.get("option_keys", "show"), **extra, **lim))
    return out


def _cost(name, v):
    """A cost of an act question: a finite number (the legacy coercion of numeric text stays)."""
    try:
        c = float(v)
    except (TypeError, ValueError, OverflowError):  # (OverflowError: an integer beyond the largest float)
        c = math.inf if isinstance(v, int) else None
        if c is None:
            raise RequestError(f"question {name!r}: a cost must be a number, got {v!r}") from None
    if not math.isfinite(c):
        raise RequestError(f"question {name!r}: a cost must be a finite number, got {v!r}")
    return c


def act_spec(name, q, outcomes):
    """Cost matrix C[action][outcome] from the full form or the short form {"wrong": W, "defer": D}."""
    max_error = q.get("max_error")
    if max_error is not None:
        try:
            max_error = float(max_error)
        except (TypeError, ValueError, OverflowError):
            raise RequestError(f"question {name!r}: max_error must be a number") from None
    costs = q.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise RequestError(f"question {name!r}: act needs costs")
    if set(costs) <= {"wrong", "defer"} and not isinstance(costs.get("wrong"), dict):
        w = _cost(name, costs["wrong"])
        C = {y: {z: 0.0 if z == y else w for z in outcomes} for y in outcomes}
        if "defer" in costs:
            C["defer"] = {z: _cost(name, costs["defer"]) for z in outcomes}
        return dict(costs=C, defer="defer" if "defer" in costs else None, max_error=max_error)
    for a, row in costs.items():
        if not isinstance(row, dict) or set(row) != set(outcomes):
            raise RequestError(f"question {name!r}: costs[{a!r}] must give a cost for every outcome {outcomes}")
    C = {a: {z: _cost(name, v) for z, v in row.items()} for a, row in costs.items()}
    defer = q.get("defer_action") or next((a for a in ("defer", "human") if a in C), None)
    return dict(costs=C, defer=defer, max_error=max_error)


NO_CERTIFICATE = "no certified threshold for this type at this max_error"
PLAIN = ("choice", "noul", "score")


def answer_ids(q):
    """The ask_if values that name each answer of a choice / noul / score item -> its canonical answer: noul yes / no
    (or true / false), a choice option key, a score level key or its text."""
    if q["type"] == "noul":
        return {"yes": "true", "true": "true", "no": "false", "false": "false"}
    ids = {k: k for k in q["keys"]}
    if q["type"] == "score":
        ids.update({o: k for k, o in zip(q["keys"], q["options"]) if o not in ids})
    return ids


def answer_id(q, ans):
    """The canonical answer of a plain answer: noul "true" / "false" (true on a tie), the choice, the top score level."""
    if q["type"] == "noul":
        return "true" if ans["probabilities"]["true"] >= ans["probabilities"]["false"] else "false"
    return ans["choice"] if q["type"] == "choice" else max(ans["probabilities"], key=ans["probabilities"].get)


def ask_gates(questions, qs):
    """The questions' depends_on / ask_if (as in vLLM's structured-diffusion example) -> [(name, {question: canonical
    answers})] of the questions with an ask_if, each after every question it names. Checked: known names, no cycle,
    ask_if naming a choice / noul / score question, values among its answers (answer_ids). basal-serve answers every
    question over the state in one pass, so depends_on puts no earlier answer into a prompt; a question whose ask_if
    fails is answered null afterwards (skip_answers)."""
    if not any(isinstance(q, dict) and ("depends_on" in q or "ask_if" in q) for q in questions.values()):
        return []
    plain = {q["name"]: q for q in qs if q["type"] in PLAIN}
    deps, gates = {}, {}
    for name, q in questions.items():
        d, ask = q.get("depends_on") or [], q.get("ask_if") or {}
        if not isinstance(d, list) or not all(isinstance(x, str) for x in d):
            raise RequestError(f"question {name!r}: depends_on must be a list of question names")
        if not isinstance(ask, dict) or not all(isinstance(v, list) and v for v in ask.values()):
            raise RequestError(f"question {name!r}: ask_if must map a question name to a non-empty list of its answers")
        for x in list(d) + list(ask):
            if x not in questions or x == name:
                raise RequestError(f"question {name!r}: depends on unknown question {x!r}")
        for x, vals in ask.items():
            if x not in plain:
                raise RequestError(f"question {name!r}: ask_if names {x!r}, not a choice, noul or score question")
            ids = answer_ids(plain[x])
            vals = [("true" if v else "false") if isinstance(v, bool) else v for v in vals]
            bad = [v for v in vals if not isinstance(v, str) or v not in ids]
            if bad:
                raise RequestError(f"question {name!r}: ask_if values {bad} for {x!r} must be among {sorted(ids)}")
            gates.setdefault(name, {})[x] = {ids[v] for v in vals}
        deps[name] = set(d) | set(ask)
    order, done = [], set()
    while len(done) < len(deps):
        ready = [n for n in deps if n not in done and deps[n] <= done]
        if not ready:
            raise RequestError(f"depends_on / ask_if: a cycle among {sorted(set(deps) - done)}")
        done.update(ready)
        order += [(n, gates[n]) for n in ready if n in gates]
    return order


def skip_answers(answers, gates, qs):
    """ask_if as a post-filter: a question whose ask_if fails (the named question's answer is not among the values, or
    it was skipped itself) is answered null -> the skipped names."""
    by = {q["name"]: q for q in qs}
    skipped = []
    for name, ask in gates:
        if any(answers.get(x) is None or answer_id(by[x], answers[x]) not in ok for x, ok in ask.items()):
            answers[name] = None
            skipped.append(name)
    return skipped


def images_as_media(images):
    """"images" (data:image/... URLs or {"content_type": "image/...", "base64": ...}, as in vLLM's structured-diffusion
    example) -> "media" image items."""
    if not isinstance(images, list) or not images:
        raise RequestError("images must be a non-empty list of data:image/... URLs or {content_type, base64} objects")
    out = []
    for i, im in enumerate(images):
        if isinstance(im, str) and im.startswith("data:image/"):
            out.append({"type": "image", "data": im})
        elif isinstance(im, dict) and str(im.get("content_type", "")).startswith("image/") \
                and isinstance(im.get("base64"), str):
            out.append({"type": "image", "data": im["base64"]})
        else:
            raise RequestError(f"images[{i}]: a data:image/... URL or an object with content_type and base64")
    return out


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


NO_CONFORMAL = "no conformal threshold for this type at this coverage: the set is every option"


def answer_set(p, p_min):
    """Indices of the conformal answer set of probabilities p: every option with p >= p_min, the most probable first
    (ties in option order) and always in; every option when p_min is None."""
    order = sorted(range(len(p)), key=lambda i: -p[i])
    return order if p_min is None else [i for j, i in enumerate(order) if j == 0 or p[i] >= p_min]


def limits(name, q, t):
    """The optional "coverage" (conformal answer set) and "max_error" (abstain flag) of a choice / noul / score
    question -> extra item fields; refused on the other types (act and tool read their own max_error)."""
    plain = t in ("choice", "noul", "score")
    if q.get("coverage") is not None and not plain:
        raise RequestError(f"question {name!r}: coverage is supported on choice, noul and score questions")
    out = {}
    for k in ("coverage", "max_error") if plain else ():
        v = q.get(k)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1:
            raise RequestError(f"question {name!r}: {k} must be a number between 0 and 1")
        out[k] = float(v)
    return out


def plain_limits(ans, q, probs, conf, cal):
    """A plain answer with "coverage": its answer_set (the keys whose probability reaches the type's conformal p_min
    at that level, calibration conformal.levels[L].per_type[type]) and the coverage; with "max_error": "abstain",
    true when the confidence is below the type's certified threshold for that target or it has none."""
    if "coverage" in q:
        level = ((cal.get("conformal") or {}).get("levels") or {}).get(str(q["coverage"])) or {}
        p_min = ((level.get("per_type") or {}).get(q["type"]) or {}).get("p_min")
        keys = list(probs)
        ans["answer_set"] = [keys[i] for i in answer_set([probs[k] for k in keys], p_min)]
        ans["coverage"] = q["coverage"]
        if p_min is None:
            ans["coverage_note"] = NO_CONFORMAL
    if "max_error" in q:
        thr = certified(cal, q["max_error"], q["type"])
        if thr is None:
            ans["max_error"] = NO_CERTIFICATE
        ans["abstain"] = thr is None or conf < thr
    return ans


def certified_table(cal):
    """{max_error target: {question type: certified confidence threshold}} of a calibration ({} without thresholds):
    what a client needs to tell automatable decisions from those to review."""
    table = {}
    for target in (cal.get("thresholds") or {}):
        row = {t: c for t in ("choice", "noul", "score") if (c := certified(cal, float(target), t)) is not None}
        if row:
            table[target] = row
    return table


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
    """Engine name of the per-engine calibration file CALIBRATION.<engine>.json: the quantised CUDA modes (fp8, nvfp4,
    int8), the engines other than the basal bf16 path and a --vllm-preset other than the default (vllm.<preset>); None =
    CALIBRATION.json."""
    if kind in ("graph", "exit") and quant in ("fp8", "nvfp4", "int8"):
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


def lanes_refused(a, kind):
    """Why basal-serve --lanes N cannot run as asked (None: it can; Qwen3.5 models are refused where they are found)."""
    n = getattr(a, "lanes", 1)
    if n < 1:
        return "--lanes must be at least 1"
    if n == 1:
        return None
    if not a.deterministic:
        return "--lanes needs --deterministic (without it, concurrent requests share forwards)"
    if kind != "graph":
        return "--lanes needs the CUDA graph engine (--mode fast, fast-nocompile, fp8, nvfp4)"
    if getattr(a, "long_lane", "off") == "on":
        return "--lanes is not combined with --long-lane"
    if lean_arg(getattr(a, "lean", "off")):
        return "--lanes is not combined with --lean"
    if os.environ.get("BASAL_EXPERIMENTAL_LANES") != "1":
        return ("--lanes is experimental in this version: under concurrent requests its lanes can return wrong answers "
                "or stop answering; BASAL_EXPERIMENTAL_LANES=1 turns it on for development only")
    return None


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


def states_cached(a, backend):
    """--state-cache-gb: requests are cut at their state, unless the engine leaves the cache off (block FP8: the
    request path is then the one without a cache)."""
    return bool(getattr(a, "state_cache_gb", 0)) and not getattr(backend, "state_cache_off", False)


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


def served_orders(a, fmt):
    """The option orders served: --order-average K (K orders, their mean), else --orders (default_orders)."""
    k = getattr(a, "order_average", None)
    if k is None:
        return default_orders(a.orders, fmt)
    if a.orders is not None:
        raise SystemExit("--order-average K replaces --orders: give one of them")
    return k


def orders_arg(value):
    """--orders values: 1, 2, auto, 2le<N> (N >= 2)."""
    if value in ("1", "2", "auto") or (value.startswith("2le") and value[3:].isdigit() and int(value[3:]) >= 2):
        return value
    raise argparse.ArgumentTypeError(f"--orders must be 1, 2, auto or 2le<N> (N >= 2), got {value!r}")


def orders_for(orders, k):
    """Option orders of a question with k options under --orders (auto: 1 in the first pass) or --order-average K (at
    most the k! distinct orders)."""
    if orders == 2:
        return 2
    if isinstance(orders, int) and orders > 2:
        return min(orders, math.factorial(k))
    if isinstance(orders, str) and orders.startswith("2le"):
        return 2 if k <= int(orders[3:]) else 1
    return 1


def option_orders(k, n):
    """The first n option orders (permutations of range(k)): the given order, the reversed one, then each rotation
    followed by its reverse, then the other permutations in lexicographic order. --orders 2 asks the first two;
    --order-average K the first K, so that with 2k orders every option is read at every position twice."""
    first, seen, out = list(range(k)), set(), []
    for perm in itertools.chain((first, first[::-1]), *(((first[r:] + first[:r]), (first[r:] + first[:r])[::-1])
                                                          for r in range(1, k)), map(list, itertools.permutations(first))):
        if len(out) == n:
            break
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            out.append(list(perm))
    return out


def order_average_arg(value):
    """--order-average K: K >= 2 option orders."""
    if value.isdigit() and int(value) >= 2:
        return int(value)
    raise argparse.ArgumentTypeError(f"--order-average needs K >= 2 option orders, got {value!r}")


def models_payload(name, mode, policies, temperatures=None, prompt_format=None, orders=None, media=(),
                   calibration_warning=None, vllm_preset=None, layout=None, two_way=None, option_isolation=False,
                   certified=None, context=None, adapter=None, buckets=None):
    """GET /v1/models in the official ModelMetadataList shape, plus the serving mode, early-exit levels, the
    temperatures in use ({} = uncalibrated probabilities, what basal-calibrate needs from a running server), the
    model's prompt format, the option orders served, why the --calibration file may not belong to the model, the
    --vllm-preset (--mode vllm), the two-way limits of a model trained with two-way reading, whether its options
    are read in isolated blocks and the --adapter identity (name@hash)."""
    entry = {"name": name, "release_date": RELEASE_DATE,
             "description": "basal typed-decision model (choice / noul / score / act / multi / tool), Polish and "
                            "English",
             "mode": mode, "early_exit": sorted(policies), "temperature_per_prim": dict(temperatures or {})}
    entry["temperature_per_bucket"] = buckets or {}
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
    if two_way:
        entry["two_way"] = dict(two_way)
    if option_isolation:
        entry["option_isolation"] = True
    if certified is not None:
        entry["certified"] = certified
    if context is not None:
        entry["context"] = context
    if adapter:  # the LoRA adapter merged at load (basal-calibrate --url records it)
        entry["adapter"] = adapter
    return {"models": [entry]}


# --adapter: the engines that refuse it, and why
ADAPTER_REFUSED = dict.fromkeys(("gguf", "ollama", "llamacpp"),
                                "serves weights the engine cannot merge into (a GGUF file / another process); fuse "
                                "them with basal-fuse-adapter and convert with basal-export-gguf")
ADAPTER_REFUSED["exit"] = ("reads exit heads trained on the base model's hidden states, whose thresholds do not hold "
                           "for the adapter's; serve it with --mode fast")
# by quantisation of the CUDA graph engine: served only with a calibration (and NVFP4 activation scales) measured on
# the model itself, which the base model's files are not for the adapter's weights
ADAPTER_REFUSED["nvfp4"] = ADAPTER_REFUSED["int8"] = (
    "serves a model with its own calibration for that mode (and NVFP4 activation scales), measured on the base model's "
    "weights, not the adapter's; fuse it with basal-fuse-adapter and calibrate the fused checkpoint for this mode")


def adapter_refused(name, mode, kind):
    """Why an engine refuses --adapter (basal-serve, basal-bench); kind: a key of ADAPTER_REFUSED."""
    return f"--adapter {name}: {mode} {ADAPTER_REFUSED[kind]}"


def adapter_refused_kind(kind, quant):
    """The ADAPTER_REFUSED key of a mode's engine kind and quantisation, None when it takes --adapter."""
    return kind if kind in ADAPTER_REFUSED else quant if quant in ADAPTER_REFUSED else None


IMAGE_BUDGETS = (70, 140, 280, 560, 1120)  # soft tokens per image Gemma 4's vision tower takes


def reread_refused(threshold, fmt, image=None, media=False):
    """Why --reread / --reread-image cannot be used (None: they can): a probability threshold, the state_twice layout
    of format 2; an image budget of Gemma 4's vision tower, for a server that takes media."""
    if image is not None:
        if threshold is None:
            return "--reread-image needs --reread <threshold> (the answers it reads again)"
        if image not in IMAGE_BUDGETS:
            return f"--reread-image takes an image token budget ({', '.join(map(str, IMAGE_BUDGETS))})"
        if not media:
            return "--reread-image needs --media on (a server that reads images)"
    if threshold is None:
        return None
    if not 0 < threshold <= 1:
        return "--reread takes a probability threshold in (0, 1]"
    if fmt == 1:
        return ("--reread reads the state twice (the state_twice layout of prompt format 2); this model uses prompt "
                "format 1")
    return None


def two_way_refused(kind, a):
    """Why a two-way model (basal.json "two_way") cannot be served with these options, or None. It reads the state
    with each question: nothing is shared between questions (SOAM) or cached before the question (the state and prefix
    caches); the engine's own eager Gemma 4 forward and the CUDA graph engine (fast, fast-nocompile) read two-way."""
    if a.soam == "on":
        return "--soam on: a two-way model reads the state with each question, so the questions cannot share it"
    if getattr(a, "state_cache_gb", 0) or getattr(a, "prefix_cache", "off") == "on":
        return ("--state-cache-gb and --prefix-cache keep a state read before its question; a two-way model reads it "
                "with the question")
    if kind not in ("eager", "graph"):
        return ("a two-way model (basal.json \"two_way\") is served with --mode eager, fast or fast-nocompile in this "
                "version")
    if kind == "graph" and getattr(a, "long_lane", "off") == "on":
        return "--long-lane: not with a two-way model (its long prompts are read one way, like the others)"
    if getattr(a, "eager_forward", None) == "hf":
        return "a two-way model needs the engine's own Gemma 4 forward (--eager-forward basal)"
    if getattr(a, "reread", None) is not None:
        return ("--reread reads the state twice (the state_twice layout); a two-way model reads it once, with each "
                "question, and was not trained on that layout")
    return None


def option_isolation_refused(kind, a):
    """Why a model trained with isolated options (basal.json "option_isolation") cannot be served with these options,
    or None. Its prompts are read whole by the engine's own eager Gemma 4 forward (nothing shared or cached before the
    options; text only) in this version."""
    if a.soam == "on":
        return "--soam on: a model with isolated options reads each prompt whole in this version"
    if getattr(a, "state_cache_gb", 0) or getattr(a, "prefix_cache", "off") == "on":
        return "--state-cache-gb and --prefix-cache: not with isolated options in this version"
    if kind != "eager":
        return "a model with isolated options (basal.json \"option_isolation\") is served with --mode eager"
    if getattr(a, "eager_forward", None) == "hf":
        return "a model with isolated options needs the engine's own Gemma 4 forward (--eager-forward basal)"
    if getattr(a, "media", "off") == "on":
        return "--media on: not with isolated options in this version (text prompts only)"
    if getattr(a, "reread", None) is not None:
        return ("--reread reads the state twice (the state_twice layout), which a model with isolated options was not "
                "trained on")
    return None


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


def calibration_mismatch(cal, name, md, adapter=None):
    """Why a file written by basal-calibrate for another model, or for another revision of this one, or with another
    --adapter (or none), does not belong to the served model; None when it does or the file does not say (files that
    ship with a model)."""
    fitted = cal.get("fitted_on")
    if not isinstance(fitted, dict) or not fitted.get("model"):
        return None
    model, rev = model_identity(name, md)
    # both Hugging Face snapshots (a revision on each side): the whole repo id, so OrgA/x is not OrgB/x; otherwise the
    # last part of the path, so a local copy matches its repo id
    full = bool(fitted.get("revision") and rev)
    key = lambda s: (str(s) if full else Path(str(s)).name).lower()
    if key(fitted["model"]) != key(model):
        return f"it was fitted on {fitted['model']}, the served model is {model}"
    if fitted.get("revision") and rev and fitted["revision"] != rev:
        return f"it was fitted on revision {fitted['revision'][:12]} of {model}, the served revision is {rev[:12]}"
    fa, sa = fitted.get("adapter"), adapter.identity if adapter else None
    if fa != sa:
        return f"it was fitted with adapter {fa or 'none'}, the served adapter is {sa or 'none'}"
    return None


# the parts of a file fitted on other option orders, another layout or another model that do not hold for the served
# ones
UNVALIDATED_OUT = ("thresholds", "multi", "task_thresholds", "conformal")


def unvalidated_reason(data, name, orders, layout="canonical"):
    """Why the thresholds of calibration `data` (file `name`) do not hold for the served option orders and layout, None
    when they do (or the file does not say): the rule of load_calibration, shared with basal-calibrate --from-rows."""
    if not data:
        return None
    if calibration_layout(data) != (layout or "canonical"):  # another layout's thresholds do not hold here (2.4.1)
        return f"{name} was fitted on the {calibration_layout(data)} layout, served with --layout {layout or 'canonical'}"
    fitted = calibration_orders(data)
    if fitted is not None and orders is not None and fitted != orders:
        return f"{name} was fitted on {fitted} option order(s), served with {orders}"
    return None


def load_calibration(md, kind, path, quant, orders, off=False, preset=None, layout="canonical", adapter=None):
    """The calibration basal-serve applies -> (calibration dict, its file, None or why it is unvalidated). A file
    fitted on other option orders than the served ones gives wrong temperatures and thresholds: the file for the
    served orders, CALIBRATION[.<engine>].o<orders>.json, is used when the model ships one; otherwise the file is kept
    for its temperatures without its thresholds, fitted fallback thresholds, task thresholds and conformal levels
    (max_error defers, tasks report automated: null, answer sets hold every option), and the answers are marked
    unvalidated. An explicit --calibration file is not replaced. adapter (--adapter): the adapter directory's own files
    come first; the model's file is the base model's, kept the same way as a file of other orders. An explicit
    --calibration file is applied as it is under an adapter too (it may be fitted with one); calibration_mismatch checks
    its fitted_on.adapter."""
    layout = layout or "canonical"
    cal = calibration_file(md, kind, path, quant, preset, layout, adapter.path if adapter else None)
    data = {} if off or not cal.exists() else json.loads(cal.read_text(encoding="utf-8"))
    if data and adapter and not path and cal.parent != adapter.path:
        why = f"{cal.name} belongs to the base model, served with --adapter {adapter.name}"
        return {k: v for k, v in data.items() if k not in UNVALIDATED_OUT}, cal, why
    fitted = calibration_orders(data)
    alt = cal.with_name(f"{cal.stem}.o{orders}{cal.suffix}")
    if calibration_layout(data) == layout and fitted is not None and fitted != orders and not path and alt.exists():
        return load_calibration(md, kind, alt, quant, orders, off, preset, layout, adapter)
    why = unvalidated_reason(data, cal.name, orders, layout)
    if why is None:
        return data, cal, None
    return {k: v for k, v in data.items() if k not in UNVALIDATED_OUT}, cal, why


def calibration_file(md, kind, path=None, quant=None, preset=None, layout="canonical", adapter_dir=None):
    """The calibration file basal-serve loads: `path` (--calibration), else CALIBRATION.<engine>.json when the model
    ships one for the engine (calibration_name; a --vllm-preset without a file of its own: CALIBRATION.vllm.json),
    else CALIBRATION.json. Another --layout than canonical: CALIBRATION[.<engine>].<layout>.json first (2.4.1; without
    one the canonical file is loaded and its thresholds are not applied, load_calibration). adapter_dir (--adapter):
    any of these files in the adapter directory (fitted with the adapter) before the model's."""
    if path:
        return Path(path)
    layout = layout or "canonical"
    # 1.5: each engine has its own calibration file when shipped (fp8 too); 2.2.1: each vLLM preset
    engines = dict.fromkeys((calibration_name(kind, quant, preset), calibration_name(kind, quant)))
    for d in ([Path(adapter_dir)] if adapter_dir else []) + [md]:
        for sfx in ([f".{layout}"] if layout != "canonical" else []) + [""]:
            for eng in engines:
                if eng and (d / f"CALIBRATION.{eng}{sfx}.json").exists():
                    return d / f"CALIBRATION.{eng}{sfx}.json"
            if (sfx or d != md) and (d / f"CALIBRATION{sfx}.json").exists():
                return d / f"CALIBRATION{sfx}.json"
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
        if getattr(a, "compile_scope", "model") == "layer" and (kind != "graph" or not comp):
            raise SystemExit(f"--compile-scope layer: the CUDA graph engine's compiled modes (--mode fast), not "
                             f"--mode {a.mode}")
        # --adapter: read (and refused when it is not a plain LoRA) before any weights are downloaded
        self.adapter = adapter_module.read(a.adapter) if getattr(a, "adapter", None) else None
        if self.adapter and adapter_refused_kind(kind, a.quant or quant):
            raise SystemExit(adapter_refused(self.adapter.name, f"--mode {a.mode}",
                                             adapter_refused_kind(kind, a.quant or quant)))
        if kind == "gguf" and not a.gguf:
            raise SystemExit("--mode gguf needs --gguf <file.gguf> (--model gives tokenizer and calibration)")
        if a.calibration and not a.no_calibration and not Path(a.calibration).exists():
            raise SystemExit(f"--calibration {a.calibration}: no such file")  # a typo must not serve uncalibrated
        why = lanes_refused(a, kind)  # before any model is read
        if why:
            raise SystemExit(why)
        if a.deterministic:
            deterministic_setup(kind)
        elif getattr(a, "fp8_accum", "fast") == "full":  # FP8 checkpoints: the same GEMM arithmetic as --deterministic
            set_fp8_fast_accum(False)
        if getattr(a, "state_cache_bits", 16) != 16 and (kind != "graph" or not (
                getattr(a, "state_cache_gb", 0) or getattr(a, "prefix_cache", "off") == "on")):
            raise SystemExit("--state-cache-bits 8 needs --state-cache-gb or --prefix-cache on in the CUDA graph "
                             "engine (fast, fast-nocompile, fp8, int8, nvfp4)")
        if getattr(a, "state_cache_lock", "off") == "on" and (kind != "graph" or not getattr(a, "state_cache_gb", 0)):
            raise SystemExit("--state-cache-lock on needs --state-cache-gb in the CUDA graph engine (fast, "
                             "fast-nocompile, fp8, int8, nvfp4)")
        if getattr(a, "state_cache_gb", 0) and (a.deterministic or kind not in ("graph", "exit", "mlx")):
            raise SystemExit("--state-cache-gb needs the CUDA graph engine (fast, fast-nocompile, fp8, nvfp4) or "
                             "--mode mlx and is not combined with --deterministic (a cached state changes the shapes of "
                             "the forwards)")
        if getattr(a, "state_cache_dir", None) is not None:
            if not (getattr(a, "state_cache_gb", 0) or getattr(a, "prefix_cache", "off") == "on"):
                raise SystemExit("--state-cache-dir needs --state-cache-gb or --prefix-cache on")
            if kind not in ("graph", "mlx"):
                raise SystemExit("--state-cache-dir works with the CUDA graph engine (fast, fast-nocompile, fp8, int8, "
                                 "nvfp4) and --mode mlx")
        if getattr(a, "layout", None) not in (None, "canonical"):
            refuse_layout_caches(a.layout, a)
        if getattr(a, "prefix_cache", "off") == "on" and (a.deterministic or kind not in ("graph", "mlx")):
            raise SystemExit("--prefix-cache needs the CUDA graph engine (fast, fast-nocompile, fp8, nvfp4) or --mode "
                             "mlx and is not combined with --deterministic")
        a.lean, a.tree = lean_tree(getattr(a, "lean", "off"), getattr(a, "tree", "dense"), kind, comp, a.quant or quant)
        refuse_lean_flex(a.lean, a.tree)
        if getattr(a, "batches", "default") not in ("default", "auto") and kind not in ("graph", "exit"):
            raise SystemExit("--batches needs the CUDA graph engine (fast, fast-nocompile, fast-exit, fp8, nvfp4)")
        if getattr(a, "fp8_scaling", "auto") == "block" and (a.quant or quant) != "fp8":
            raise SystemExit("--fp8-scaling block needs --mode fp8")
        if getattr(a, "fp8_bf16_layers", FP8_BF16_LAYERS) < 0:
            raise SystemExit("--fp8-bf16-layers: give 0 or more layers")
        if getattr(a, "quant_load", "gpu") != "gpu" and (kind != "graph" or (a.quant or quant) not in ("fp8", "nvfp4",
                                                                                                        "int8")):
            raise SystemExit("--quant-load cpu needs --mode fp8, int8 or nvfp4 (the CUDA graph engine)")
        if getattr(a, "long_lane_gdn", "off") == "on" and getattr(a, "long_lane", "off") != "on":
            raise SystemExit("--long-lane-gdn on needs --long-lane on")
        if getattr(a, "graph_memory_gb", None) is not None:
            if kind != "graph":
                raise SystemExit("--graph-memory-gb needs the CUDA graph engine (fast, fast-nocompile, fp8, int8, "
                                 "nvfp4)")
            if not a.graph_memory_gb > 0:
                raise SystemExit(f"--graph-memory-gb {a.graph_memory_gb:g}: give a size in GiB above 0")
        if getattr(a, "vllm_preset", "default") != "default" and kind != "vllm":
            raise SystemExit("--vllm-preset needs --mode vllm (the CUDA graph engine has --deterministic)")
        if getattr(a, "global_attn", "flex") != "flex" and kind != "graph":
            raise SystemExit("--global-attn needs the CUDA graph engine (--mode fast, fast-nocompile, fp8, nvfp4)")
        if kind == "vllm" and VLLM_PRESETS.get(getattr(a, "vllm_preset", "default")):
            what = _recurrent(a.model, a.revision)  # before vLLM starts: its batch-invariant mode raises in the engine core
            if what:
                raise SystemExit(f"{a.model}: --vllm-preset {a.vllm_preset}: vLLM's batch-invariant kernels do not cover "
                                 f"{what} (a recurrent state, such as Qwen3.5's Gated DeltaNet); serve it with the default "
                                 "preset (answers can vary run to run on near-ties) or --mode eager (exact)")
        attn = getattr(a, "attn", "auto")
        if attn not in ("auto", "fa2") and kind not in ("graph", "exit"):
            raise SystemExit(f"--attn {attn} needs the CUDA graph engine (--mode fast, fast-nocompile, fast-exit, fp8, "
                             "nvfp4)")
        if getattr(a, "mlx_packed", False) and kind != "mlx":
            raise SystemExit("--mlx-packed needs --mode mlx (or mlx-q8)")
        if getattr(a, "mlx_forward", "mlx-lm") != "mlx-lm" and kind != "mlx":
            raise SystemExit("--mlx-forward needs --mode mlx (or mlx-q8)")
        if getattr(a, "mlx_compute", "auto") != "auto" and kind != "mlx":
            raise SystemExit("--mlx-compute needs --mode mlx")
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
                                  getattr(a, "readout_layer", "off"), getattr(a, "template_prefix", "off"),
                                  disk=getattr(a, "state_cache_dir", None) is not None)
                # --long-lane on Qwen3.5: bf16 answers 0.12 pt further from fp32 than eager's (five questions per state),
                # so only with --long-lane-gdn on
                if not why and getattr(a, "long_lane", "off") == "on" and getattr(a, "long_lane_gdn", "off") != "on":
                    why = ("--long-lane is not available for Qwen3.5 (Gated DeltaNet) models in this version by "
                           "default; --long-lane-gdn on serves it anyway (its bf16 answers to five questions per state "
                           "agree with float32 0.12 pt less often than --mode eager's)")
                if not why and getattr(a, "lanes", 1) > 1:
                    why = "--lanes is not available for Qwen3.5 (Gated DeltaNet) models"
                if why:
                    raise SystemExit(f"{a.model}: {why}")
        if kind in ("graph", "exit") and (a.quant or quant) == "fp8" and fp8_refused():
            raise SystemExit(fp8_refused())
        # GGUF modes: the engine serves the GGUF weights, so only tokenizer, template and calibration are downloaded
        md = resolve(a.model, a.revision, metadata_only=kind in ("gguf", "ollama", "llamacpp"))
        self.fmt = model_format(md)  # prompt format of the model (basal.json "prompt"; 1 when absent)
        self.max_options = max_options(self.fmt)
        self.rank_max = getattr(a, "rank_max_candidates", rank.MAX_CANDIDATES)  # rank candidates per request
        self.orders = served_orders(a, self.fmt)
        # the prompt layout (prompt.LAYOUTS; 2.4.1): --layout, else the model's basal.json "layout", else canonical
        self.layout = getattr(a, "layout", None) or model_layout(md)
        if self.layout != "canonical" and self.fmt != 2:
            raise SystemExit(f"layout {self.layout} needs a prompt-format-2 model (basal.json \"prompt\": 2)")
        if self.layout != "canonical":
            refuse_layout_caches(self.layout, a)
        # two-way reading (basal.json "two_way"): each prompt read whole, so SOAM is off (its default) and no cache
        self.two_way = model_two_way(md)
        if self.two_way and two_way_refused(kind, a):
            raise SystemExit(f"{a.model}: {two_way_refused(kind, a)}")
        # isolated options (basal.json "option_isolation"): prompts tokenized piece by piece and read whole
        self.iso = model_option_isolation(md)
        if self.iso and (self.two_way or self.fmt != 2):
            raise SystemExit(f"{a.model}: isolated options need prompt format 2 and one-way reading")
        if self.iso and option_isolation_refused(kind, a):
            raise SystemExit(f"{a.model}: {option_isolation_refused(kind, a)}")
        self.soam = (a.soam or ("off" if self.two_way or self.iso else "on")) == "on"
        self.tau, self.band = a.auto_tau, tuple(a.auto_band)
        # --reread T (2.5): a choice or yes / no answer of a short request whose top probability is below T is read again
        # in the state_twice layout and the two readouts are combined (reread_pass, combined)
        self.reread = getattr(a, "reread", None)
        self.reread_combine = getattr(a, "reread_combine", "mean")
        self.reread_max = getattr(a, "reread_max_tokens", 2048)
        # --reread-image B: a request with an image re-reads its uncertain answers with the image at B soft tokens
        self.reread_image = getattr(a, "reread_image", None)
        why = reread_refused(self.reread, self.fmt, self.reread_image, getattr(a, "media", "off") == "on")
        if why:
            raise SystemExit(why)
        shared = self.orders != 1 or self.soam  # prefix sharing: option orders, or the questions of a request
        base = Path(a.model).name or Path(a.model).resolve().name  # --model . : the directory's name
        self.name = a.name or (f"{base}+{self.adapter.name}" if self.adapter else base)
        quant = a.quant or quant
        why = nvfp4_calibration_refused(md, kind, quant, a)
        if why:
            raise SystemExit(why)
        if getattr(a, "media", "off") == "on" and kind not in ("eager", "vllm"):
            raise SystemExit("--media on needs --mode eager or vllm (the media paths of basal-omni)")
        if getattr(a, "ple_offload", "off") == "on" and kind != "eager":
            raise SystemExit("--ple-offload on needs --mode eager")
        if getattr(a, "eager_attn", "math") != "math" and kind != "eager":
            raise SystemExit("--eager-attn needs --mode eager (the CUDA graph engine takes --attn)")
        self.prepared = None  # --prepared: the compiled kernels of an earlier start on this GPU class (basal-prepare)
        if kind in ("graph", "exit") and comp and getattr(a, "prepared", "auto") != "off" and torch.cuda.is_available():
            from . import prepared
            fields = prepared.key_fields(md, a.mode, a.dtype, attn)
            self.prepared = prepared.startup(md, fields, getattr(a, "prepared", "auto"), a.model, a.revision)
            print(f"basal: prepared start-up: {self.prepared or 'none for this GPU and version (basal-prepare writes it)'}",
                  flush=True)
        # --state-cache-dir: the compatibility signature of the cached states' files (engine.cache_signature)
        disk_kwargs = dict(state_cache_dir=getattr(a, "state_cache_dir", None),
                           state_cache_disk_gb=getattr(a, "state_cache_disk_gb", 20.0),
                           state_cache_signature=None)
        if disk_kwargs["state_cache_dir"] is not None:
            kv_dtype = "bf16" if kind == "mlx" else {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}.get(
                a.dtype, a.dtype)
            disk_kwargs["state_cache_signature"] = cache_signature(md, *model_identity(a.model, md), self.adapter,
                                                                   quant, kv_dtype)
        if kind == "eager":
            self.backend = EagerBackend(md, a.dtype, device=a.device, media=getattr(a, "media", "off") == "on",
                                        forward=getattr(a, "eager_forward", None),
                                        ple_offload=getattr(a, "ple_offload", "off") == "on",
                                        attn=getattr(a, "eager_attn", "math"), adapter=self.adapter)
        elif kind == "exit":
            self.backend = ExitGraphBackend(md, a.dtype, quant, compile=comp, heads_dir=a.exit_heads,
                                            default_policy=a.early_exit, attn=attn, merge=merge_arg(a.merge))
        elif kind == "vllm":
            self.backend = VLLMBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len, max_options=self.max_options,
                                       preset=getattr(a, "vllm_preset", "default"),
                                       fast_prefill=getattr(a, "vllm_fast_prefill", "auto") == "auto",
                                       media=getattr(a, "media", "off") == "on",
                                       overlap=getattr(a, "vllm_overlap", "off") == "on", adapter=self.adapter)
        elif kind == "sglang":
            self.backend = SGLangBackend(md, a.dtype, mem=a.gpu_memory, max_len=a.max_len, adapter=self.adapter)
        elif kind == "mlx":
            self.backend = MLXBackend(md, quant=quant, packed=getattr(a, "mlx_packed", False),
                                      forward=getattr(a, "mlx_forward", "mlx-lm"),
                                      compute=getattr(a, "mlx_compute", "auto"),
                                      state_cache_gb=getattr(a, "state_cache_gb", 0),
                                      prefix=getattr(a, "prefix_cache", "off") == "on", adapter=self.adapter,
                                      **disk_kwargs)
        elif kind == "mps":
            self.backend = MPSBackend(md, a.dtype, shared=shared, device=a.device, adapter=self.adapter)
        elif kind == "gguf":
            self.backend = GGUFBackend(md, a.gguf)
        elif kind == "strix":
            from .strix import StrixHaloFastBackend
            self.backend = StrixHaloFastBackend(md, a.dtype, adapter=self.adapter)
        elif kind == "ollama":
            self.backend = OllamaBackend(md, a.ollama_url, a.ollama_model, parallel=a.http_parallel)
        elif kind == "llamacpp":
            self.backend = LlamaCppBackend(md, a.llamacpp_url, parallel=a.http_parallel)
        elif kind == "tpu":
            from .tpu import TPUBackend
            self.backend = TPUBackend(md, a.dtype, shared=shared, max_len=a.max_len, adapter=self.adapter)
        else:
            scope = getattr(a, "compile_scope", "model")
            self.backend = GraphBackend(md, a.dtype, quant, compile="layer" if comp and scope == "layer" else comp,
                                        shared=shared, attn=attn,
                                        state_cache_gb=getattr(a, "state_cache_gb", 0),
                                        merge=merge_arg(a.merge), lean=lean_arg(getattr(a, "lean", "off")),
                                        tree=getattr(a, "tree", "dense"),
                                        prefix=getattr(a, "prefix_cache", "off") == "on",
                                        batches=getattr(a, "batches", "default"),
                                        readout_layer=getattr(a, "readout_layer", "off"),
                                        template=getattr(a, "template_prefix", "off"),
                                        gemma_attn=getattr(a, "gemma_attn", "auto"),
                                        global_attn=getattr(a, "global_attn", "flex"),
                                        fp8_scaling=getattr(a, "fp8_scaling", "auto"),
                                        fp8_bf16_layers=getattr(a, "fp8_bf16_layers", FP8_BF16_LAYERS),
                                        quant_load=getattr(a, "quant_load", "gpu"),
                                        state_cache_bits=getattr(a, "state_cache_bits", 16),
                                        graph_memory_gb=getattr(a, "graph_memory_gb", None), **nvfp4_args(a, md, quant),
                                        adapter=self.adapter, **disk_kwargs)
            if getattr(a, "state_cache_lock", "off") == "on":  # before lane() copies the backend
                self.backend.state_cache_lock = True
        self.lanes = None  # --lanes N: the backend and N - 1 copies (GraphBackend.lane), each serving one request
        if getattr(a, "lanes", 1) > 1:
            t = time.time()
            self.lanes = [self.backend] + [self.backend.lane() for _ in range(a.lanes - 1)]
            self.free_lanes = list(self.lanes)
            print(f"basal: --lanes {a.lanes}: {a.lanes - 1} more copies of the graphs in {time.time() - t:.0f} s",
                  flush=True)
        if self.adapter:  # every backend has consumed the factors (merged, or written to its PEFT directory)
            self.adapter.deltas.clear()
        self.tok = self.backend.tok
        self.once = once_ids(self.tok)  # the shared part of a request's prompts tokenized once (prompt.tokens_once)
        self.states_cached = states_cached(a, self.backend)
        model = getattr(self.backend, "model", None)  # evidence head: needs the in-process HF model
        # the head was trained on the base model's final-layer states: not read under --adapter
        self.evidence = Evidence(md, model, self.tok, self.fmt) if isinstance(model, torch.nn.Module) \
            and not self.adapter else None
        self.md, self.kind = md, kind
        self.context = context_limit(md, self.backend)
        self.cal, cal, self.cal_unvalidated = load_calibration(md, kind, a.calibration, quant, self.orders,
                                                                a.no_calibration, getattr(a, "vllm_preset", None),
                                                                self.layout, self.adapter)
        check_calibration(self.cal, self.fmt, cal)
        self.cal_file = cal  # --ui on: named in the configuration view
        # a file of your own (--calibration) or the adapter's (fitted with an adapter); the files in the model directory
        # belong to it
        own = a.calibration or (bool(self.adapter) and cal.parent == self.adapter.path)
        self.cal_mismatch = calibration_mismatch(self.cal, a.model, md, self.adapter) if own and not a.no_calibration \
            else None
        unvalidated = self.cal_unvalidated
        if self.cal_mismatch:
            if getattr(a, "strict_calibration", False):
                raise SystemExit(f"{cal}: {self.cal_mismatch} (--strict-calibration)")
            # its thresholds were certified on that model: kept for the temperatures only, as a file of other orders
            self.cal = {k: v for k, v in self.cal.items() if k not in UNVALIDATED_OUT}
            self.cal_unvalidated = unvalidated or f"{cal.name}: {self.cal_mismatch}"
            print(f"basal: warning: {cal.name}: {self.cal_mismatch}; its thresholds were certified on that model, so "
                  "they are not applied: max_error defers and the answers are marked \"calibration\": \"unvalidated\" "
                  "(refit with basal-calibrate, or --strict-calibration to refuse such a file)", flush=True)
        if unvalidated:
            base_file = self.adapter and not a.calibration and cal.parent != self.adapter.path
            print(f"basal: warning: {self.cal_unvalidated}; its answers are marked \"calibration\": \"unvalidated\" "
                  "and max_error defers (" + ("fit a file for the adapter with basal-calibrate --adapter and put it in "
                                              "the adapter directory)" if base_file else
                                              "fit a file for this serving with basal-calibrate, or serve with the "
                                              "file's orders and layout)"), flush=True)
        self.temps = self.cal.get("temperature_per_prim", {})
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

    def jobs_for(self, q, question=None, options=None, perms=None, layout=None):
        """Prompts of one branch in every option order of the first pass (or in `perms`), in the served layout (or
        `layout`): [(perm, prompt, code ids)]."""
        question = q["question"] if question is None else question
        options = q["options"] if options is None else options
        k = len(options)
        if perms is None:
            n = orders_for(self.orders, k)
            perms = [list(range(k)), list(range(k))[::-1]][:n] if n <= 2 else option_orders(k, n)
        jobs = []
        for perm in perms:
            prompt = render(self.tok, q["state"], question, [options[c] for c in perm], q["lang"], self.fmt,
                            layout or getattr(self, "layout", "canonical"))
            if getattr(self, "iso", False):  # its options go with it: the engine tokenizes it piece by piece
                prompt = OptionPrompt(prompt)
                prompt.options = [options[c] for c in perm]
            jobs.append((perm, prompt, self.code_ids(prompt, k)))
        return jobs

    def branch(self, i, bi, q, perms=None, layout=None):
        """Branch bi of question i -> (i, bi, perms, prompts, code ids)."""
        jobs = self.jobs_for(q, *q["branches"][bi], perms, layout)
        if any(ids is None for _, _, ids in jobs):
            raise ValueError("option letters are not single tokens for this tokenizer")
        return (i, bi, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs])

    def encode(self, branches):
        """Tokenize the prompts of all branches at once (the engine takes the ids) -> (number of prompt tokens, the
        longest prompt of each branch)."""
        flat = [p for b in branches for p in b[3]]
        if getattr(self, "iso", False):  # isolated options: the pieces one by one, with their token counts
            toks = [option_tokens(self.tok, p, p.options, self.fmt) for p in flat]
        else:  # the part the prompts share (template and state) once, then each prompt's own tail; else each whole
            once = getattr(self, "once", None)
            toks = (once and tokens_once(self.tok, flat, once)) or self.tok(flat, add_special_tokens=False).input_ids
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
        # otherwise one entry per branch (all option orders of one question share the prefix). A two-way model: one
        # entry, so the engine sees the request's record (its one-way fallback above max_record tokens)
        entries = [branches] if self.soam or lane == "long" or getattr(self, "two_way", None) \
            else [[b] for b in branches]
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
                if not fut.done():  # cancelled, or another entry failed: its queued work is dropped (forward)
                    fut.cancel()
            if wait[0]:
                self.queued -= 1
        return outs

    def cut_of(self, q, branches):
        """Where the engine's state cache cuts a request's prompts (None: no cut). --state-cache-gb: after the state.
        --prefix-cache: after the template prefix (chat template, system prompt, the user header up to the state) when
        everything below it is short (all prompts together at most the engine's PREFIX_SFX[-1] tokens past the
        prefix), unless the request's state is cached itself. A Qwen3.5 engine (one recurrent state per pass-2 row:
        below the cut it would read the state once per prompt) cuts single-prompt requests only."""
        cache = getattr(self.backend, "state_cache", None)
        if cache is None:
            return None
        cut = self.state_cut(q) if getattr(self, "states_cached", True) else None
        if getattr(self.backend, "prefix", False):
            if cut and cache.has(cache.key(branches[0][3][0][:cut])):
                return cut
            t = self.template_cut(q, branches[0][3][0])
            if t and sum(len(p) - t for b in branches for p in b[3]) <= self.backend.PREFIX_SFX[-1] and (
                    not getattr(self.backend, "row_per_leaf", False) or sum(len(b[3]) for b in branches) == 1):
                return t
        return cut

    def template_cut(self, q, ids):
        """Tokens of the template prefix of a question's prompts (everything before the state: chat template, system
        prompt and the user header "State:" / "Stan:"), checked against the prompt's token ids; None when not found."""
        head = FORMATS[self.fmt]["templates"][q["lang"]]["user"].split("{state}")[0]
        text = render(self.tok, q["state"], q["question"], q["options"], q["lang"], self.fmt)
        i = text.find(head + defang(self.tok, q["state"])) if q["state"] else -1  # as rendered (special tokens broken)
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
        shown = defang(self.tok, q["state"])  # the state as rendered (special tokens broken)
        i = text.find(head + shown) if q["state"] else -1
        if i < 0:
            return None
        full = self.tok(text, add_special_tokens=False).input_ids
        pre = self.tok(text[: i + len(head) + len(shown)], add_special_tokens=False).input_ids
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
        slots = asyncio.Semaphore(len(self.lanes) if getattr(self, "lanes", None)
                                  else max(1, getattr(self.backend, "inflight", 1)))
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
            self.long_entries[:] = [e for e in self.long_entries if not e[2].done()]  # (cancelled: dropped)
            if not self.long_entries:
                self.long_ready.clear()
                continue
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
        live = [b for b in batch if not b[2].done()]  # (a cancelled request's entries cost the engine nothing)
        lanes = getattr(self, "lanes", None)
        backend = self.free_lanes.pop() if lanes else self.backend  # (slots: never more calls than lanes)
        try:
            for pol in {b[3] for b in live}:  # requests with the same early-exit policy run together
                ix = [i for i, b in enumerate(live) if b[3] == pol]
                kw = {}
                if getattr(backend, "streams", False):
                    kw["done"] = lambda j, res, ix=ix: loop.call_soon_threadsafe(_answer, live[ix[j]][2], res)
                groups = []
                for i in ix:  # (prompts, code ids[, state prefix length for the state cache])
                    cut = live[i][4] if len(live[i]) > 4 else None
                    groups.append((live[i][0], live[i][1]) + ((cut,) if cut else ()))
                out = await loop.run_in_executor(None, functools.partial(backend.run_shared, groups, pol, **kw))
                for i, o in zip(ix, out):
                    _answer(live[i][2], o)
        except Exception as e:  # noqa: BLE001
            _fail([b[2] for b in live], e)
        finally:
            if lanes:
                self.free_lanes.append(backend)
            slots.release()
            if getattr(self, "gate", None) is not None:
                self.gate.main_done(len(batch), time.perf_counter() - t_call)
            if getattr(self, "backlog", None) is not None and live:  # seconds per prompt token, for the HRRN ratios
                tokens = sum(max(map(len, b[0]), default=0) for b in live)
                self.sched_rate = 0.8 * self.sched_rate + 0.2 * (time.perf_counter() - t_call) / max(tokens, 1)

    @staticmethod
    def order_mean(perms, outs):
        """Position probabilities of every option order -> their mean in the canonical order (a tensor)."""
        canon = []
        for perm, p_pos in zip(perms, outs):
            c = [0.0] * len(perm)
            for k, j in enumerate(perm):
                c[j] = p_pos[k]
            canon.append(c)
        return torch.tensor([sum(x) / len(canon) for x in zip(*canon)])

    def canonical(self, q_type, perms, outs):
        """Position probabilities of every option order -> mean in the canonical order, then the type temperature."""
        p = self.order_mean(perms, outs)
        T = temperature(self.cal, q_type, len(p))  # calibrated temperature per question type (and option count)
        return temper(p, T).tolist()

    def combined(self, q_type, perms, outs, perms2, outs2):
        """--reread: the first and the second (state_twice, or the image at --reread-image tokens) readout -> one
        distribution, then the type temperature:
        the mean of their log-probabilities (renormalised), or the second read alone (--reread-combine second)."""
        p2 = self.order_mean(perms2, outs2)
        if self.reread_combine == "second":
            p = p2
        else:
            p = torch.softmax((self.order_mean(perms, outs).clamp_min(1e-12).log() + p2.clamp_min(1e-12).log()) / 2, -1)
        return temper(p, temperature(self.cal, q_type, len(p))).tolist()

    def to_reread(self, qs, branches, outs):
        """The branches --reread reads again: single-branch choice and yes / no questions whose calibrated top
        probability is below the threshold."""
        redo = []
        for j, (b, o) in enumerate(zip(branches, outs)):
            q = qs[b[0]]
            if q["type"] in ("choice", "noul") and not q.get("multi") and len(q["branches"]) == 1 \
                    and max(self.canonical(q["type"], b[2], o)) < self.reread:
                redo.append(j)
        return redo

    async def reread_image_pass(self, qs, branches, outs, m):
        """--reread-image B: a request with an image whose uncertain answers (to_reread) were read with fewer image
        tokens reads them again with every image at B soft tokens (canonical layout, the option orders read so far)
        -> ({branch index: (perms, position probabilities)}, prompt tokens, media tokens)."""
        if not m["images"] or m["max_soft_tokens"] >= self.reread_image:
            return {}, 0, 0
        redo = self.to_reread(qs, branches, outs)
        if not redo:
            return {}, 0, 0
        o2, n_tok, _, n_media = await self.ask_media(qs, [branches[j] for j in redo],
                                                     dict(m, max_soft_tokens=self.reread_image))
        return {j: (branches[j][2], o) for j, o in zip(redo, o2)}, n_tok, n_media

    async def reread_pass(self, qs, branches, outs, longest, pol, lane):
        """--reread T: the single-branch choice and yes / no questions of a request whose longest prompt has at most
        --reread-max-tokens tokens and whose calibrated top probability is below T, read again in the state_twice
        layout (in the option orders read so far) -> ({branch index: (perms, position probabilities)}, prompt tokens)."""
        if max(longest, default=0) > self.reread_max:
            return {}, 0
        redo = self.to_reread(qs, branches, outs)
        if not redo:
            return {}, 0
        again = [self.branch(branches[j][0], branches[j][1], qs[branches[j][0]], list(branches[j][2]), "state_twice")
                 for j in redo]
        n_tok = self.encode(again)[0]
        o2 = await self.ask(again, pol, None, lane)  # (no state-cache cut: the state comes twice)
        return {j: (b2[2], o) for j, b2, o in zip(redo, again, o2)}, n_tok

    async def decide(self, body):
        if any(rank.is_rank(q) for q in body["questions"].values()):  # ranking beyond 255 options (basal/rank.py)
            return await rank.decide(self, body)
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
            # (a regex pass over the whole state, linear in its length: off the event loop, which answers /health
            # meanwhile)
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            if len(text) > FACTS_MAX_CHARS:
                raise RequestError(f'facts "auto" reads a state of at most {FACTS_MAX_CHARS} characters, this one has '
                                   f'{len(text)}: send "facts": "off" or a shorter state')
            state = await asyncio.to_thread(inject_facts, text)
        if body.get("images") is not None:  # the alias of vLLM's structured-diffusion example
            if body.get("media") is not None:
                raise RequestError("give media or images, not both")
            body = dict(body, media=images_as_media(body["images"]))
        # (a request with media is decoded off the event loop; without, media_of is None at once)
        m = self.media_of(body, pol) if body.get("media") is None else \
            await asyncio.get_running_loop().run_in_executor(None, self.media_of, body, pol)
        lang = body.get("lang")  # the prompt language when the text alone does not show it (a state that is only media)
        if lang is not None and lang not in ("pl", "en"):
            raise RequestError('lang must be "pl" or "en"')
        qs = to_items(state, body["questions"], self.max_options, force_lang=lang)
        gates = ask_gates(orig["questions"], qs)
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
        reread = None
        if getattr(self, "reread", None) is not None and m is None:  # a second read of the uncertain short answers
            reread, n2 = await self.reread_pass(qs, branches, outs, longest, pol, lane)
            n_tok += n2
        elif getattr(self, "reread_image", None) is not None and m is not None:  # ... with the image at more tokens
            reread, n2, nm2 = await self.reread_image_pass(qs, branches, outs, m)
            n_tok, n_media = n_tok + n2, n_media + nm2
        out = await self.assemble(body, qs, branches, outs, m, n_tok, n_media, longest, second, truncation, lane, t0,
                                  reread)
        if plans:  # the task answers from their decision answers; extract's spans need a second round of decisions
            r2, a2 = tasks.second_round(out, orig, plans, self.max_options), None
            if r2 is not None:
                out2 = await self.decide(r2)
                a2 = out2["answers"]
                for k in ("input_tokens", "branches"):
                    out["usage"][k] = out["usage"].get(k, 0) + out2["usage"].get(k, 0)
            out = tasks.finish(out, orig, plans, self.cal, a2)
            out["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if gates:
            skipped = skip_answers(out["answers"], gates, qs)
            if skipped:
                out["usage"]["skipped"] = skipped
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

    def media_pool(self):
        """The thread of the engine's media calls: the engines read one media request at a time (a lock), so requests
        waiting for it wait here, never in the default executor's threads, which the text batches need."""
        if getattr(self, "_media_pool", None) is None:
            self._media_pool = ThreadPoolExecutor(1, thread_name_prefix="basal-media")
        return self._media_pool

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
                                           self.fmt, self.tok))
                ids.append(lid)
                where.append(j)
        limit = getattr(self, "context", None)
        res = await asyncio.get_running_loop().run_in_executor(self.media_pool(), functools.partial(
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

    async def assemble(self, body, qs, branches, outs, m, n_tok, n_media, longest, second, truncation, lane, t0,
                       reread=None):
        """Answers and usage of a request from the position probabilities of its branches (and of the second reads of
        --reread: {branch index: (perms, position probabilities)})."""
        loop = asyncio.get_running_loop()
        per_q, again = [[] for _ in qs], {}
        for j, (b, o) in enumerate(zip(branches, outs)):
            per_q[b[0]].append((b[2], o))
            if reread and j in reread:
                again[b[0]] = reread[j]
        answers = {}
        for qi, (q, brs) in enumerate(zip(qs, per_q)):
            if q["type"] == "multi" or q.get("multi"):
                ans = multi_answer(q, [self.canonical("noul", perms, o)[0] for perms, o in brs])
                answers[q["name"]] = tool_set_answer(q, ans) if q["type"] == "tool" else ans
                continue
            perms, o = brs[0]
            if qi in again:
                pl = self.combined(q.get("base", q["type"]), perms, o, *again[qi])
            else:
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
            if qi in again:
                ans["reread"] = True
            if q["type"] in ("choice", "noul", "score") and ("coverage" in q or "max_error" in q):
                plain_limits(ans, q, probs, conf, self.cal)
            answers[q["name"]] = ans
        ev_q = [q for q in qs if body["questions"][q["name"]].get("evidence")]
        if ev_q and truncation:
            raise RequestError("evidence is not available for a state truncated to the context (shorten the state)")
        if ev_q:  # evidence spans: character offsets into the request's state (facts block excluded)
            if not (self.evidence and self.evidence.available):
                raise RequestError("evidence is not available with --adapter (the evidence head was trained on the base "
                                   "model's states)" if getattr(self, "adapter", None) else
                                   "evidence is not available for this model (no evidence_head.pt)")
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
        if getattr(self, "reread", None) is not None and (m is None or getattr(self, "reread_image", None) is not None):
            out["usage"]["reread_branches"] = len(reread or {})
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
    ap.add_argument("--adapter", default=None,
                    help="a LoRA adapter directory merged into the model's weights at load (mlx-lm: adapter_config.json "
                         "+ adapters.safetensors; PEFT: adapter_config.json + adapter_model.safetensors); vLLM / SGLang "
                         "take it natively; not with GGUF, Ollama, llama.cpp (basal-fuse-adapter, then "
                         "basal-export-gguf) or FP8 checkpoints")
    ap.add_argument("--mode", choices=list(MODES), default=None,
                    help="default: fast on CUDA, mlx on Apple Silicon (mps without mlx / mlx-lm), tpu on a TPU VM "
                         "with basal[tpu], eager otherwise")
    ap.add_argument("--quant", choices=["fp8", "nvfp4", "int8", "q8"], default=None,
                    help="override the quantisation of the mode (fp8 / nvfp4 / int8: CUDA graph modes, q8: mlx)")
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
    ap.add_argument("--mlx-forward", dest="mlx_forward", choices=["mlx-lm", "basal"], default="mlx-lm",
                    help="--mode mlx: mlx-lm (default) = mlx-lm's forward; basal (opt-in, 2.5) = basal's own forward "
                         "over mlx-lm's layers for Llama and Gemma 4: the last layers at the readout rows only, merged "
                         "projections checked bit for bit at load; the probabilities equal mlx-lm's up to float "
                         "rounding, not bit for bit. Other architectures always use mlx-lm's")
    ap.add_argument("--mlx-compute", dest="mlx_compute", choices=["auto", "bf16"], default="auto",
                    help="--mode mlx, Qwen3.5 (Gated DeltaNet) models such as basal-omni-pico: auto (default) = "
                         "float32 compute from the bf16 weights; bf16 (opt-in) = compute in bf16: faster, but further "
                         "from float32 (Apple M4 Pro, 1,700 held-out single prompts: 0.136 s per prompt against 0.171 "
                         "s for the GGUF Q8_0 path, 1.25x; the same top answer as CUDA eager float32 on 99.41%%, 1 "
                         "changed where its margin is >= 0.05; 17,000 readouts of five questions in two option orders: "
                         "99.33%%, 8 changed). Other models compute in their weights' dtype")
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
    ap.add_argument("--order-average", dest="order_average", type=order_average_arg, default=None, metavar="K",
                    help="the mean of the probabilities over K option orders (the given order, the reversed one, then "
                         "each rotation and its reverse), every order reading the request's state once; off by "
                         "default. --order-average 2 = --orders 2. A calibration fitted on other orders keeps its "
                         "temperatures only and the answers are marked unvalidated, unless the model ships "
                         "CALIBRATION.o<K>.json (basal-calibrate --order-average K fits one)")
    ap.add_argument("--auto-tau", dest="auto_tau", type=float, default=0.6,
                    help="--orders auto: a second order when the top probability is below this (choice, score)")
    ap.add_argument("--auto-band", dest="auto_band", type=float, nargs=2, default=[0.2, 0.8],
                    help="--orders auto: a second order when P(true) of a yes/no branch is inside this band")
    ap.add_argument("--reread", type=float, default=None,
                    help="a second read of uncertain answers (prompt format 2): a choice or yes/no answer of a short "
                         "request (--reread-max-tokens) whose calibrated top probability is below this is read again "
                         "with the state twice (the state_twice layout) and the two readouts are combined. Off by "
                         "default")
    ap.add_argument("--reread-combine", dest="reread_combine", choices=["mean", "second"], default="mean",
                    help="--reread: mean of the two readouts' log-probabilities (default), or the second read alone")
    ap.add_argument("--reread-max-tokens", dest="reread_max_tokens", type=int, default=2048,
                    help="--reread: only requests whose longest first-read prompt has at most this many tokens")
    ap.add_argument("--reread-image", dest="reread_image", type=int, default=None,
                    help="with --reread and --media on: a request with an image reads its uncertain answers again with "
                         "every image at this many soft tokens (Gemma 4: 70, 140, 280, 560 or 1120; the request's "
                         "\"detail\" sets the first read's) and combines the reads as --reread does. Off by default")
    ap.add_argument("--early-exit", dest="early_exit", default="off", help="default policy for --mode fast-exit (basal-1.0 models, which ship exit heads)")
    ap.add_argument("--exit-heads", dest="exit_heads", default=None, help="exit heads dir (default: <model>/exit_heads)")
    ap.add_argument("--prepared", default="auto",
                    help="compiled modes: load the kernels of an earlier start on this GPU class (basal-prepare) before "
                         "the engine compiles. auto (default) = the model's prepared/<key> or the user cache when the key "
                         "matches; off; or a directory")
    ap.add_argument("--soam", choices=["on", "off"], default=None,
                    help="on = the whole request (every question and option order) is one packed row over the state "
                         "(default; off for a two-way model, basal.json \"two_way\")")
    ap.add_argument("--gpu-memory", dest="gpu_memory", type=float, default=0.6, help="vLLM memory fraction")
    ap.add_argument("--vllm-preset", dest="vllm_preset", choices=list(VLLM_PRESETS), default="default",
                    help="--mode vllm: default = as 2.2; deterministic = batch-invariant kernels without the prefix "
                         "cache and chunked prefill (VLLM_BATCH_INVARIANT: answers independent of the other requests, "
                         "lower throughput). Loads CALIBRATION.vllm.<preset>.json when the model ships one, else "
                         "CALIBRATION.vllm.json")
    ap.add_argument("--vllm-fast-prefill", dest="vllm_fast_prefill", choices=["auto", "off"], default="auto",
                    help="--mode vllm: auto = layers that reuse earlier layers' keys and values (Gemma 4 E2B / E4B) "
                         "run at the answer position only during prefill (vLLM's kv_sharing_fast_prefill); off = as 2.2")
    ap.add_argument("--vllm-overlap", dest="vllm_overlap", choices=["on", "off"], default="off",
                    help="--mode vllm: on = several engine calls in flight through vLLM's async engine, batched together "
                         "by its scheduler (the GPU does not wait between one batch's answers and the next batch); off "
                         "= one LLM.generate call at a time. on is opt-in: a little faster, and its answers differ from "
                         "off's (RTX PRO 6000 Max-Q, 16 clients, basal-omni-1.0-mini and 1.0-max: 1.01-1.07x the "
                         "questions per second, 1.21x on mini's development set; the same top answer on 98.17-99.65%%, "
                         "2-6 changed where off's margin is >= 0.05, per 600-2,000 readouts). With --media on it "
                         "batches requests with media too (basal-omni-1.1-max, one H100, 8 clients: new images "
                         "1.18-1.49x the requests per second; audio clips about the same as off (0.89-1.23x); the same "
                         "top answer as off on 99.65-100%%, accuracy within 0.35 points)")
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
    ap.add_argument("--global-attn", dest="global_attn", choices=["flex", "triton"], default="flex",
                    help="CUDA graph engine, basal-omni (Gemma 4): the kernel of the 512-wide global layers on plain "
                         "rows (single prompts and the state of a multi-question request): flex (FlexAttention, as "
                         "before) or triton (basal's causal kernel for 512-wide heads, faster on long states; Hopper); "
                         "pass 2 and compact rows keep flex. triton is opt-in: faster, but answers differ from flex "
                         "(H100, five questions over 24k-32k-token states, 4 clients: basal-omni-1.0-mini 1.10x, the "
                         "same top answer on 99.67%%, none changed where flex's margin is >= 0.05; 1.1-max 1.16x, "
                         "98.33%%, 4 of 600 changed; 1.0-ultra 1.03x, 99.0%%, 2 of 600)")
    ap.add_argument("--merge", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graph engine: one q/k/v and one gate/up GEMM per layer (bf16: the same products; FP8 "
                         "checkpoints: the merged weights re-quantised per tensor). auto: on for bf16 weights, off for "
                         "FP8 checkpoints")
    ap.add_argument("--lean", choices=["auto", "on", "pass2", "off"], default="off",
                    help="CUDA graph engine, Llama models: one compiled layer function for every layer and shape "
                         "(basal/lean.py) instead of transformers' module tree, in both passes or in pass 2 only. "
                         "auto: pass2 on the CUDA graph engine (with --tree auto / flex), else off")
    ap.add_argument("--compile-scope", dest="compile_scope", choices=["model", "layer"], default="model",
                    help="CUDA graph engine, Llama models, compiled modes: torch.compile the transformers model as a "
                         "whole (model) or each decoder layer, one compiled layer function for all of them (layer: "
                         "start-up compiles one layer instead of the whole model; kernels fuse within a layer only, "
                         "so probabilities move by bf16 rounding). layer is opt-in: a shorter start-up, but answers "
                         "differ from model's (H100, basal-2.0-mini, cold caches: 47 s instead of 228 s, the same top "
                         "answer on 98.69%% of 840 readouts, none changed where the margin is >= 0.05; two model-scope "
                         "cold starts agree on 98.3-98.5%%)")
    ap.add_argument("--tree", choices=["auto", "dense", "flex"], default="dense",
                    help="with --lean pass2: the pass-2 trie mask as a dense additive mask (KV heads repeated) or as a "
                         "FlexAttention mask over the KV heads as they are. auto: flex with --lean pass2, else dense")
    ap.add_argument("--state-cache-gb", dest="state_cache_gb", type=float, default=0,
                    help="CUDA graph engine and --mode mlx: keep the keys and values of request states on the GPU "
                         "up to this many GB (least recently used evicted); a later request with the same state "
                         "computes only its questions and options, one whose state begins with a cached one only its "
                         "new tokens (0: off)")
    ap.add_argument("--state-cache-lock", dest="state_cache_lock", choices=["off", "on"], default="off",
                    help="--state-cache-gb in the CUDA graph engine: on continues cached states under the engine's "
                         "launch lock, which removes a rare HTTP 500 under concurrent requests (a torch.compile "
                         "race: 1 of 8,740 requests at 8 clients without it, 0 with it); measured ~8%% lower "
                         "throughput at 8 clients (basal-omni-1.0-max, H100: 8.19 vs 8.88 requests/s, p50 630 vs "
                         "577 ms). off (default): as in 2.4.1")
    ap.add_argument("--state-cache-bits", dest="state_cache_bits", type=int, choices=[16, 8], default=16,
                    help="--state-cache-gb / --prefix-cache: 16 keeps the cached keys and values as computed; 8 stores "
                         "them as 8-bit integers with a scale per group (keys per channel, values per token): about "
                         "1.7x more cached states per GB, answers close to (not equal to) the uncached ones")
    ap.add_argument("--state-cache-dir", dest="state_cache_dir", default=None,
                    help="with --state-cache-gb / --prefix-cache on: also keep every cached state as a .safetensors "
                         "file under this directory (one subdirectory per model signature), read back on a memory "
                         "miss, also by later server processes; least recently used files evicted at "
                         "--state-cache-disk-gb")
    ap.add_argument("--state-cache-disk-gb", dest="state_cache_disk_gb", type=float, default=20.0,
                    help="--state-cache-dir: GiB of state files kept on disk (default 20)")
    ap.add_argument("--prefix-cache", dest="prefix_cache", choices=["on", "off"], default="off",
                    help="CUDA graph engine and --mode mlx: compute the template prefix (chat template, system "
                         "prompt, the header up to the state) once and run short requests over its cached keys and "
                         "values")
    ap.add_argument("--batches", choices=["auto", "compact", "default", "fine"], default="auto",
                    help="CUDA graph engine: compact = single prompts packed back to back into one row (FlashAttention "
                         "3 varlen on Hopper; elsewhere, sm80+, torch's varlen attention, Llama models; basal-omni with "
                         "the flex kernels: flex block masks per prompt), no padding to "
                         "graph lengths and row counts; default = rows padded to graph lengths; fine = more row "
                         "counts (less padding of short batches, longer start-up); auto = compact where FlashAttention "
                         "3 runs, else default. basal-omni's flex compact rows (Blackwell workstation GPUs) are opt-in: "
                         "faster, but answers differ from the default rows (RTX PRO 6000 Max-Q, 16 clients, single "
                         "questions: 1.17-1.20x on basal-omni-1.0-mini, 1.21-1.27x on 1.0-max; the same top answer on "
                         "98.0-99.75%%, up to 4 changed where the default's margin is >= 0.05, per 600-2,000 readouts)")
    ap.add_argument("--readout-layer", dest="readout_layer", choices=["auto", "on", "off"], default="off",
                    help="CUDA graph engine with compact rows, Llama models: the last decoder layer computes keys, values "
                         "and attention for every token and its o_proj and MLP at the readout positions only (auto: on; off by "
                         "default: +1.6%% at best on the H100, below the v2.0 speed gate)")
    ap.add_argument("--template-prefix", dest="template_prefix", choices=["auto", "on", "off"], default="auto",
                    help="CUDA graph engine with compact rows: the template prefix of the prompts (chat template, system "
                         "prompt, the user header up to the state) is computed once at start-up and every prompt runs "
                         "only its own tokens over its keys and values (auto: on)")
    ap.add_argument("--fp8-scaling", dest="fp8_scaling", choices=["auto", "row", "block"], default="auto",
                    help="--mode fp8: row = torchao, one scale per token and per weight row; block = scales per 128 x "
                         "128 weight block and per 128 features of each token (basal-omni ultra only; other "
                         "basal-omni models are refused); auto: block for basal-omni, row otherwise")
    ap.add_argument("--fp8-bf16-layers", dest="fp8_bf16_layers", type=int, default=FP8_BF16_LAYERS,
                    help="--mode fp8 with block scaling, int8, nvfp4: the first and the last this many decoder layers "
                         f"keep bf16 weights (default {FP8_BF16_LAYERS}; 0 = every layer quantised)")
    ap.add_argument("--quant-load", dest="quant_load", choices=["gpu", "cpu"], default="gpu",
                    help="--mode fp8 (block scaling) / int8 / nvfp4 with basal-omni: where the bf16 weights are held "
                         "while they are quantised. gpu: all on the GPU first (needs their full size); cpu: in host "
                         "memory, one decoder layer at a time to the GPU, quantised there (GPU memory about the "
                         "quantised size: ultra in FP8 or int8 on cards under 64 GB; the same weights). With "
                         "--graph-memory-gb N it answers as any GPU given the same N")
    ap.add_argument("--graph-memory-gb", dest="graph_memory_gb", type=float, default=None,
                    help="the CUDA graph engine: size the graph shapes (the grid, the pass-2 graphs, the longest state "
                         "run as one forward) for this much free GPU memory after the weights, in GiB, instead of the "
                         "free memory measured at start-up (default). Two GPUs given the same value run the same "
                         "shapes and give bit-identical answers; more than is free after the weights is refused")
    ap.add_argument("--nvfp4-amax", dest="nvfp4_amax", default=None,
                    help="--mode nvfp4 with basal-omni ultra (other basal-omni models are refused): the calibrated "
                         "input amax of each decoder linear (JSON, {\"amax\": {\"<layer>.<proj>\": value}}; default: the "
                         "model's NVFP4_AMAX.json): NVFP4 weights and activations (basal/nvfp4.py); --fp8-bf16-layers "
                         "keeps the edge layers bf16. Needs the model's CALIBRATION.nvfp4.json or --calibration")
    ap.add_argument("--nvfp4-scope", dest="nvfp4_scope", choices=["all", "mlp"], default="all",
                    help="--mode nvfp4 with --nvfp4-amax: all decoder projections, or gate / up / down only")
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
    ap.add_argument("--long-lane-gdn", dest="long_lane_gdn", choices=["on", "off"], default="off",
                    help="--long-lane with a Qwen3.5 (Gated DeltaNet) model such as basal-omni-pico: on = serve the "
                         "long lane (opt-in), off = refuse it. Its bf16 answers are a little further from float32 than "
                         "--mode eager's (H100, 1,700 held-out decisions, five questions per state: the same top "
                         "answer as float32 on 99.39%% against eager bf16's 99.51%%; single questions 99.71%%, two option "
                         "orders 99.65%%, both 99.46%%, within 0.1 pt of eager bf16's); no speed measured. No effect on other "
                         "models")
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
    ap.add_argument("--lanes", type=int, default=1,
                    help="--deterministic with the CUDA graph engine: N requests at once, each in its own copy of the "
                         "graphs on its own CUDA stream (GPU memory and start-up grow with N). Experimental, not for "
                         "production in this version (its lanes can return wrong answers under concurrency): needs "
                         "BASAL_EXPERIMENTAL_LANES=1")
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
                         "(default: a warning, and its thresholds are not applied)")
    ap.add_argument("--log-decisions", dest="log_decisions", default=None, help="append a propensity log (JSONL)")
    ap.add_argument("--rank-max-candidates", dest="rank_max_candidates", type=int, default=rank.MAX_CANDIDATES,
                    metavar="N", help="rank questions (experimental): at most N candidates per request, all its rank "
                                      "questions together (HTTP 422 above; every 50 take one engine request through "
                                      "the shared queue). Default %(default)s")
    ap.add_argument("--api-keys", default=None, metavar="PATH",
                    help="JSON API key file; when set, require a key for every endpoint except GET /health")
    ap.add_argument("--cors-origins", default="", help="comma-separated browser origins allowed by CORS")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--error-format", dest="error_format", choices=["basal", "openai"], default="basal",
                    help="body of a failed request: {\"error\": message} (basal) or {\"error\": {\"message\", \"type\", "
                         "\"code\"}} (openai, as OpenAI-style servers answer); the status codes are the same")
    ap.add_argument("--mcp", choices=["off", "on"], default="off",
                    help="also serve MCP (streamable HTTP) at /mcp for agents: tools decide and decide_many over the "
                         "same queue (needs basal[mcp]; without --api-keys no authentication, like the HTTP API: bind "
                         "--host 127.0.0.1 for agents on this machine; with --api-keys /mcp needs a key too)")
    ap.add_argument("--ui", choices=["off", "on", "full"], default="off",
                    help="also serve a live page of this server at /ui: status, traffic since start, state cache, GPU "
                         "and host telemetry, configuration and the recent requests (their metadata, never their "
                         "text), polled from /ui/state; each request adds one append to an in-memory ring. full: the "
                         "page also shows the last 200 requests and their answers (kept in memory, visible to whoever "
                         "can open the page). Open one page at a time: each open page reads the state once a second, "
                         "and more pages can slow requests down (one H100, 8 clients: p50 +0.4-2.2%% with 1-2 pages, "
                         "+5.2%% with full and 1 page; 1 client: answers byte-identical; docs/SERVING.md, Live web "
                         "page). With --api-keys the page asks for a key; without, it is open to whoever reaches the "
                         "port")
    from .think import add_think_args  # --think off|auto (experimental; off = this server unchanged)
    add_think_args(ap)
    return ap


class Starting:
    """basal-serve while the model loads and warms up (2.5): the port answers every request with HTTP 503
    {"status": "starting"} (a router or a load balancer sees a server that is not ready, not a closed port), from a
    small thread server; stop() frees the port for the real one. Start-up stays in the main thread (CUDA graphs,
    torch.compile)."""

    def __init__(self, host, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def _answer(self):
                body = b'{"status": "starting"}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Retry-After", "5")
                self.end_headers()
                if self.command != "HEAD":  # HEAD gets the status and the headers of GET, never a body
                    self.wfile.write(body)

            def __getattr__(self, name):  # any method (TRACE, CONNECT, a made-up one) gets the same answer, not a 501
                if name.startswith("do_"):
                    return self._answer
                raise AttributeError(name)

            def log_message(self, *_):
                pass
        ThreadingHTTPServer.allow_reuse_address = True
        try:
            self.httpd = ThreadingHTTPServer((host, port), H)
        except OSError as e:  # the port is taken: start without it (uvicorn reports the port when it binds)
            print(f"basal: warning: no 503 answers during start-up on port {port}: {e}", flush=True)
            self.httpd = None
            return
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        # a process forked during start-up (vLLM's engine core) would hold a copy of this listening socket for its
        # whole life, so the port stayed taken after stop() and the real server's bind failed (address already in
        # use): the child closes its copy at once
        sock = self.httpd.socket
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=sock.close)

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def error_body(e, error_format="basal"):
    """The body and status of a failed request: {"error": message, "code": code} (a 422 adds "detail"); with
    --error-format openai {"error": {"message", "type", "code"}}, the shape of OpenAI-style servers (vLLM's
    structured-diffusion example among them)."""
    status = http_status(e)
    if error_format == "openai":
        kind = "validation_error" if status == 422 else "server_error"
        return {"error": {"message": str(e), "type": kind, "code": status}}, status
    code = {422: "INVALID_REQUEST", 503: "OUT_OF_MEMORY", 500: "INTERNAL_ERROR"}[status]
    body = {"error": str(e), "code": code}
    if status == 422:
        body["detail"] = [{"loc": ["body"], "msg": str(e), "type": "value_error"}]
    return body, status


def http_app(srv, mode, mcp=None, error_format="basal", ui=None):
    """The app of basal-serve: POST /v1/systemone and /v1/basal, POST /v1/decisions (OpenAI's Decisions API), GET
    /v1/models, /health and /metrics; mcp: (routes,
    lifespan) of --mcp on. A failed request answers error_body with http_status: 422 invalid request, 503 out of GPU memory, 500
    any other server error. ui: the basal.ui.UI of --ui on (its decision routes recorded, and GET /ui/state)."""
    from contextlib import asynccontextmanager, nullcontext

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route
    from . import __version__
    from . import decisions as dec
    counts = {"requests": 0, "failed": 0, "seconds": 0.0}  # GET /metrics
    cache_seen = {"stats": None, "at": 0}  # the last state-cache snapshot /metrics read, and its Unix time

    def failed(e):
        print(f"basal: request failed: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
        if isinstance(e, torch.OutOfMemoryError):  # the failed forward's blocks back to the GPU: kept by the
            torch.cuda.empty_cache()  # allocator, they failed the next requests too (ultra, 20 Oct)
        body, status = error_body(e, error_format)
        return JSONResponse(body, status_code=status)

    async def systemone(request):
        t = time.perf_counter()
        counts["requests"] += 1  # (a body that is not JSON counts too, as on /v1/decisions)
        try:
            body = await request.json()
        except (ValueError, RecursionError) as e:  # not JSON (or not UTF-8), or nested too deeply
            counts["failed"] += 1
            counts["seconds"] += time.perf_counter() - t
            return failed(RequestError(f"the request body is not valid JSON: {e}"))
        try:
            return JSONResponse(await srv.serve(body))
        except Exception as e:  # noqa: BLE001
            counts["failed"] += 1
            return failed(e)
        finally:
            counts["seconds"] += time.perf_counter() - t

    async def decisions(request):
        """POST /v1/decisions (OpenAI's Decisions API, basal/decisions.py): the System One request it stands for,
        decided as POST /v1/systemone; errors in OpenAI's shape (400 for an invalid request)."""
        t = time.perf_counter()
        counts["requests"] += 1
        try:
            try:
                body = await request.json()
            except (ValueError, RecursionError) as e:  # not JSON (or not UTF-8), or nested too deeply
                raise RequestError(f"the request body is not valid JSON: {e}") from None
            s1, specs = dec.to_systemone(body)
            return JSONResponse(dec.to_decision(await srv.serve(s1), specs))
        except Exception as e:  # noqa: BLE001
            counts["failed"] += 1
            print(f"basal: request failed: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
            if isinstance(e, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
            status = 400 if isinstance(e, RequestError) else http_status(e)
            return JSONResponse(dec.error_body(e, status), status_code=status)
        finally:
            counts["seconds"] += time.perf_counter() - t

    async def models(_):
        return JSONResponse(models_payload(srv.name, mode, getattr(srv.backend, "policies", {}), srv.temps,
                                           srv.fmt, srv.orders, getattr(srv.backend, "media_kinds", ()),
                                           getattr(srv, "cal_mismatch", None), getattr(srv.backend, "preset", None),
                                           getattr(srv, "layout", None), getattr(srv, "two_way", None),
                                           getattr(srv, "iso", False),
                                           certified=None if getattr(srv, "cal_mismatch", None) else
                                           certified_table(srv.cal),
                                           context={"limit": srv.context[0] if srv.context else None,
                                                    "overflow": srv.overflow, "truncate_head": srv.trunc_head},
                                           adapter=srv.adapter.identity if getattr(srv, "adapter", None) else None,
                                           buckets=srv.cal.get("temperature_per_bucket")))

    async def health(_):
        return JSONResponse({"status": "ok", **srv.load()})

    async def metrics(_):
        """Prometheus metrics (2.5): the engine version and the served model, requests waiting / running (also under
        vLLM's names, which routers such as llm-d read), requests served / failed and their time, the state cache when
        there is one. The cache is read without waiting for its lock: while the engine holds it, the last snapshot is
        reported again (its time in basal_state_cache_snapshot_timestamp_seconds)."""
        load = srv.load()
        info = {"version": __version__, "model": srv.name, "mode": mode, "layout": getattr(srv, "layout", "canonical")}
        m = [("basal_build_info", "engine version, served model, mode and prompt layout", "gauge", [(info, 1)]),
             ("basal_requests_waiting", "requests waiting for the engine", "gauge", [({}, load["queued"])]),
             ("basal_requests_running", "requests in flight, not waiting", "gauge", [({}, load["running"])]),
             ("vllm:num_requests_waiting", "as basal_requests_waiting (vLLM's name)", "gauge", [({}, load["queued"])]),
             ("vllm:num_requests_running", "as basal_requests_running (vLLM's name)", "gauge", [({}, load["running"])]),
             ("basal_requests_total", "requests served (POST /v1/systemone, /v1/basal, /v1/decisions)", "counter",
              [({}, counts["requests"])]),
             ("basal_requests_failed_total", "requests answered with an error", "counter", [({}, counts["failed"])]),
             ("basal_request_seconds_total", "time spent on those requests", "counter",
              [({}, round(counts["seconds"], 6))])]
        cache = getattr(srv.backend, "state_cache", None)
        if cache is not None:
            st = cache.stats(blocking=False)
            if st is not None:
                cache_seen.update(stats=st, at=round(time.time(), 3))
            st = cache_seen["stats"]
            if st is not None:
                m += [("basal_state_cache_bytes", "bytes of cached states", "gauge", [({}, st["used_bytes"])]),
                      ("basal_state_cache_budget_bytes", "--state-cache-gb", "gauge", [({}, st["budget_bytes"])]),
                      ("basal_state_cache_entries", "cached states", "gauge", [({}, st["entries"])]),
                      ("basal_state_cache_hits_total", "requests whose state was cached", "counter",
                       [({}, st["hits"])]),
                      ("basal_state_cache_misses_total", "requests whose state was not", "counter",
                       [({}, st["misses"])]),
                      ("basal_state_cache_prefix_hits_total", "states that began with a cached one", "counter",
                       [({}, st["prefix_hits"])]),
                      ("basal_state_cache_evictions_total", "states evicted for the budget", "counter",
                       [({}, st["evictions"])])]
                if "disk_entries" in st:  # --state-cache-dir
                    m += [("basal_state_cache_disk_bytes", "bytes of state files on disk", "gauge",
                           [({}, st["disk_bytes"])]),
                          ("basal_state_cache_disk_budget_bytes", "--state-cache-disk-gb", "gauge",
                           [({}, st["disk_budget_bytes"])]),
                          ("basal_state_cache_disk_entries", "state files on disk", "gauge", [({}, st["disk_entries"])]),
                          ("basal_state_cache_disk_loads_total", "states read from disk", "counter",
                           [({}, st["disk_loads"])]),
                          ("basal_state_cache_disk_prefix_loads_total", "states read from disk as a prefix", "counter",
                           [({}, st["disk_prefix_loads"])]),
                          ("basal_state_cache_disk_writes_total", "state files written", "counter",
                           [({}, st["disk_writes"])]),
                          ("basal_state_cache_disk_write_drops_total", "state files not written (queue full)",
                           "counter", [({}, st["disk_write_drops"])]),
                          ("basal_state_cache_disk_errors_total", "state files that could not be written or read",
                           "counter", [({}, st["disk_errors"])]),
                          ("basal_state_cache_disk_evictions_total", "state files evicted for the disk budget",
                           "counter", [({}, st["disk_evictions"])])]
            m.append(("basal_state_cache_snapshot_timestamp_seconds", "Unix time of the cache values above (0: none "
                      "read yet)", "gauge", [({}, cache_seen["at"])]))
        return PlainTextResponse(prometheus(m), media_type="text/plain; version=0.0.4")

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
        cache = getattr(srv.backend, "state_cache", None)
        if cache is not None:  # --state-cache-dir: the states queued for writing reach the disk
            cache.close()

    routes = [Route("/v1/systemone", systemone, methods=["POST"]), Route("/v1/basal", systemone, methods=["POST"]),
              Route("/v1/decisions", decisions, methods=["POST"]), Route("/v1/models", models),
              Route("/health", health), Route("/metrics", metrics), *mcp_routes]
    return Starlette(routes=ui.routes(routes, counts) if ui is not None else routes, lifespan=lifespan)


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
    starting = Starting(a.host, a.port)  # 503 on the port until the model is ready
    try:
        if a.think == "off":
            srv = Server(a)
        else:
            from .think import ThinkServer
            srv = ThinkServer(a)
    except BaseException as e:  # a failed start-up exits (exit 1), the port closes: never a half-started server
        starting.stop()
        if isinstance(e, SystemExit):
            raise
        traceback.print_exc()
        raise SystemExit(f"basal: start-up failed: {type(e).__name__}: {e}") from None
    attn = getattr(srv.backend, "attn", None)
    rows = getattr(srv.backend, "compact_kernel", None)  # compact rows: their kernel, and the template prefix with them
    fwd = getattr(srv.backend, "forward", None)  # MLX: basal's forward (what it merged) or mlx-lm's
    conv = getattr(srv.backend, "conv_kernel", None)  # Qwen3.5: the Gated DeltaNet convolution's kernel
    info = (f" attention {attn}" if attn else "") + (f" compact rows {rows}" if rows else "") \
        + (" template prefix" if rows and getattr(srv.backend, "template", None) else "") \
        + (f" convolution {conv}" if conv else "") \
        + (f" forward {fwd}" if isinstance(fwd, str) else "") \
        + (" compute bfloat16" if getattr(srv.backend, "compute", None) == "bfloat16" else "") \
        + (f" adapter {srv.adapter.name} ({srv.backend.adapter_modules} layers)" if srv.adapter else "") \
        + ((f" calibration {srv.cal_file}" if srv.cal else " uncalibrated") if srv.adapter else "")
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
    ui = None
    if a.ui != "off":  # basal/ui.py: the page at /ui, its state at /ui/state
        from .ui import UI
        ui = UI(srv, a, a.mode, vars(parser().parse_args([])), t0, full=a.ui == "full")
        print(f"basal: web UI at http://{a.host}:{a.port}/ui" + (" (/ui/state needs an API key)" if a.api_keys else
              " (open to whoever reaches this port; --api-keys to require a key)")
              + ("; --ui full keeps the last 200 requests' text in memory for it" if a.ui == "full" else ""), flush=True)
    from .auth import ApiKeys, protect, request_ids
    origins = [origin.strip() for origin in a.cors_origins.split(",") if origin.strip()]
    app = protect(http_app(srv, a.mode, mcp, a.error_format, ui), ApiKeys(a.api_keys) if a.api_keys else None, origins)
    app = request_ids(ui.page(app) if ui is not None else app)
    try:  # uvloop + httptools when available (uvicorn[standard]); plain asyncio otherwise
        import httptools  # noqa: F401
        import uvloop  # noqa: F401
        fast = dict(loop="uvloop", http="httptools")
    except ImportError:
        fast = {}
    starting.stop()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **fast)


if __name__ == "__main__":
    main()
