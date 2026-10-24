"""Load options of a model in basal-ui: the engine mode and the basal-serve runtime settings (attention, determinism,
option orders, batching, caching, scheduling, context policy, the CUDA graph engine's kernel switches, the Gemma 4
eager forward, calibration strictness, thinking, early exit). DEFAULTS are basal-serve's own defaults, and args()
emits only the options that differ from them. Incompatible combinations are not re-checked here: the child refuses
them with basal-serve's own message, shown on the model's row and in its log."""
import re
from types import SimpleNamespace

from basal.engine import VLLM_PRESETS
from basal.gdn_graph import refused as gdn_refused
from basal.prompt import LAYOUTS
from basal.server import MODES, lean_arg, lean_tree, refuse_layout_caches, validate_quant

from .catalog import _cuda_graph

# basal-omni-pico and other Qwen3.5 models (Variant.recurrent), basal-serve 2.4 (engine.recurrent_refused, gdn_graph)
RECURRENT_REASON = ("recurrent-state layers (Gated DeltaNet, Qwen3.5): fast-exit and mps keep attention keys and values "
                    "only and would answer wrongly; basal-serve refuses them. Use fast (CUDA), eager, vllm or gguf")
MLX_RECURRENT_REASON = "MLX is not checked on recurrent-state models (Qwen3.5): use eager"
FLA_REASON = ("the CUDA graph engine runs Qwen3.5's Gated DeltaNet layers with flash-linear-attention: pip install "
              "flash-linear-attention (or serve with --mode eager)")

DEFAULTS = {
    "mode": None, "quant": None, "revision": None, "attn": "auto", "gemma_attn": "auto", "deterministic": False, "orders": None,
    "auto_tau": 0.6, "auto_band": [0.2, 0.8], "soam": True, "max_batch": 64, "wait_ms": 0.0, "batch_window_ms": 3.0,
    "state_cache_gb": 0.0, "prefix_cache": False, "scheduler": "auto", "long_lane": False, "long_tokens": 2048,
    "long_share": 0.5, "long_slice_ms": 10.0, "context_overflow": "truncate", "truncate_head": 1 / 3, "max_len": None,
    "vllm_preset": "default", "gpu_memory": 0.6, "ollama_url": "http://127.0.0.1:11434", "ollama_model": None,
    "llamacpp_url": "http://127.0.0.1:8080", "http_parallel": 1, "device": None, "think": False, "think_tokens": 256,
    "think_python": None, "think_url": None, "early_exit": "off", "mlx_packed": False, "layout": None,
    # the CUDA graph engine's kernel switches
    "merge": "auto", "lean": "off", "tree": "dense", "batches": "auto", "readout_layer": "off", "template_prefix": "auto",
    "fp8_accum": "fast",
    # --mode eager with a Gemma 4 model (basal-omni)
    "eager_attn": "math", "eager_forward": None, "ple_offload": False,
    "vllm_fast_prefill": "auto", "dtype": "bfloat16", "exit_heads": None, "strict_calibration": False,
    "think_calibration": None, "think_model": None, "think_adapter": None, "think_memory": 0.3, "think_max_len": 8192,
    "think_log": "basal-think-vllm.log",
}
CHOICES = {
    "quant": (None, "fp8", "nvfp4", "q8"), "attn": ("auto", "fa2", "fa3", "fa4", "cudnn"),
    "gemma_attn": ("auto", "math", "flex", "fa3", "fa3+math"), "scheduler": ("auto", "on", "off"),
    "context_overflow": ("truncate", "refuse"), "vllm_preset": tuple(VLLM_PRESETS), "device": (None, "cuda", "mps", "cpu"),
    "early_exit": ("off", "0.999", "0.995", "0.99", "0.98"),
    "merge": ("auto", "on", "off"), "lean": ("auto", "on", "pass2", "off"), "tree": ("auto", "dense", "flex"),
    "batches": ("auto", "compact", "default", "fine"), "readout_layer": ("auto", "on", "off"),
    "template_prefix": ("auto", "on", "off"), "fp8_accum": ("fast", "full"), "eager_forward": (None, "basal", "hf"),
    "vllm_fast_prefill": ("auto", "off"), "dtype": ("bfloat16", "float16", "float32"), "layout": (None, *LAYOUTS),
}
BOOLS = ("deterministic", "soam", "prefix_cache", "long_lane", "think", "ple_offload", "strict_calibration", "mlx_packed")
# key: (minimum, maximum, integer, minimum excluded)
NUMBERS = {
    "auto_tau": (0, 1, False, False), "max_batch": (1, None, True, False), "wait_ms": (0, None, False, False),
    "batch_window_ms": (0, None, False, False), "state_cache_gb": (0, None, False, False),
    "long_tokens": (1, None, True, False), "long_share": (0, 1, False, False), "long_slice_ms": (0, None, False, False),
    "truncate_head": (0, 1, False, False), "gpu_memory": (0, 1, False, True), "http_parallel": (1, None, True, False),
    "think_tokens": (1, None, True, False), "think_memory": (0, 1, False, True), "think_max_len": (512, None, True, False),
}
TEXTS = ("ollama_url", "ollama_model", "llamacpp_url", "think_python", "think_url", "exit_heads", "think_calibration",
         "think_model", "think_adapter", "think_log")
ORDERS = re.compile(r"1|2|auto|2le(?:[2-9]|[1-9]\d*)\Z")
KERNELS = "sdpa|efficient|cudnn|fa3|fa4"
EAGER_ATTN = re.compile(rf"math|auto|sliding=({KERNELS}),full=({KERNELS})\Z")
# the order of the Engine section; the GPU graph modes first
MODE_ORDER = ("fast", "fast-nocompile", "fast-exit", "fp8", "nvfp4", "eager", "vllm", "sglang", "mlx", "mlx-q8", "mps",
              "gguf", "strix", "tpu", "ollama", "llamacpp")


def _number(key, value):
    lo, hi, integer, open_lo = NUMBERS[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and int(value) != value):
        raise ValueError(f"load option {key} must be {'an integer' if integer else 'a number'}")
    if (value <= lo if open_lo else value < lo) or (hi is not None and value > hi):
        raise ValueError(f"load option {key} must be {'above' if open_lo else 'at least'} {lo}"
                         + (f" and at most {hi}" if hi is not None else ""))
    return int(value) if integer else float(value)


def validate(options: dict | None, default_mode: str | None = None) -> dict:
    """The options merged over DEFAULTS; ValueError names the offending option. A quantisation is checked against the
    chosen mode, else the model's default mode."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("options must be an object")  # noqa: TRY004 — validation exposes a ValueError API
    out = dict(DEFAULTS)
    for key, value in options.items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown load option: {key}")
        if key == "mode":
            if value is not None and value not in MODES:
                raise ValueError(f"load option mode must be one of {list(MODES)}")
        elif key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError(f"load option {key} must be one of {[c for c in CHOICES[key] if c is not None]}")
        elif key in BOOLS:
            if not isinstance(value, bool):
                raise ValueError(f"load option {key} must be true or false")
        elif key in NUMBERS:
            value = _number(key, value)
        elif key == "revision":
            value = (value.strip() or None) if isinstance(value, str) else value
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9A-Za-z._/-]{1,64}", value)):
                raise ValueError("load option revision must match [0-9A-Za-z._/-]{1,64}")
        elif key in TEXTS:
            if value is not None and not isinstance(value, str):
                raise ValueError(f"load option {key} must be a string")
            value = value.strip() or None if isinstance(value, str) else None
            if value is None and DEFAULTS[key] is not None:
                value = DEFAULTS[key]
        elif key == "orders":
            if value is not None and not (isinstance(value, str) and ORDERS.fullmatch(value)):
                raise ValueError("load option orders must be 1, 2, auto or 2le<N> (N >= 2)")
        elif key == "auto_band":
            if (not isinstance(value, list) or len(value) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                                      for x in value) or not 0 <= value[0] <= value[1] <= 1):
                raise ValueError("load option auto_band must be [low, high] with 0 <= low <= high <= 1")
            value = [float(x) for x in value]
        elif key == "max_len":
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 512 <= value <= 131072):
                raise ValueError("load option max_len must be an integer between 512 and 131072")
        elif key == "eager_attn" and not (isinstance(value, str) and EAGER_ATTN.fullmatch(value)):
            raise ValueError(f"load option eager_attn must be math, auto or sliding=K,full=K with K in {KERNELS}")
        out[key] = value
    if out["quant"] and (out["mode"] or default_mode):
        try:
            validate_quant(out["mode"] or default_mode, out["quant"])
        except SystemExit as exc:  # basal-serve's own message
            raise ValueError(str(exc)) from None
    return out


def non_default(options: dict) -> dict:
    """The options that differ from basal-serve's defaults."""
    return {k: v for k, v in options.items() if k in DEFAULTS and v != DEFAULTS[k]}


def _value(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def args(options: dict) -> list[str]:
    """basal-serve flags of the non-default options; the mode is applied by the runner (serve_args's --mode)."""
    flags = []
    for key, value in non_default(validate(options)).items():
        flag = "--" + key.replace("_", "-")
        if key in ("mode", "revision"):
            continue
        if key in ("deterministic", "strict_calibration", "mlx_packed"):
            flags.append(flag)
        elif key in ("soam", "prefix_cache", "long_lane", "ple_offload"):
            flags += [flag, "on" if value else "off"]
        elif key == "think":
            flags += [flag, "auto"]
        elif key == "auto_band":
            flags += [flag, *map(_value, value)]
        else:
            flags += [flag, _value(value)]
    return flags





def available_modes(v, hw, assessment) -> list[dict]:
    """Every basal-serve mode in MODE_ORDER with whether this machine can serve the variant with it and why not."""
    graph = _cuda_graph(v, hw)
    cap = hw.cuda_capability or (0, 0)
    mlx_port = v.port.startswith("MLX") or v.port == "bf16"
    reasons = {
        "fast": graph, "fast-nocompile": graph,
        "fast-exit": graph or (None if v.family == "basal-1.0" else "exit heads ship with basal-1.0"),
        "fp8": graph or fp8_unavailable(hw),
        "nvfp4": graph or (None if cap >= (10, 0) else "NVFP4 needs a Blackwell GPU (capability 10.0+)"),
        "eager": None if v.port == "bf16" else f"the eager engine reads bf16 weights, not {v.port}",
        "vllm": ("install basal[vllm] (own environment, docs/INSTALL.md)" if "vllm" not in hw.engines
                 else None if hw.platform == "cuda" else "vLLM needs an NVIDIA GPU"),
        "sglang": ("install basal[sglang] in its own environment (docs/INSTALL.md)" if "sglang" not in hw.engines
                   else None if hw.platform == "cuda" else "SGLang needs an NVIDIA GPU"),
        "mlx": ("MLX runs on Apple Silicon only" if hw.platform != "apple" else
                "install basal[mlx]" if "mlx" not in hw.engines else
                None if mlx_port else f"MLX reads MLX ports or bf16 weights, not {v.port}"),
        "mps": None if v.port == "bf16" else f"mps reads bf16 weights, not {v.port}",
        "gguf": ("install basal[gguf] (llama-cpp-python)" if "gguf" not in hw.engines
                 else None if v.port == "GGUF" else "needs a GGUF variant"),
        "strix": (None if "strix" in hw.engines else
                  "needs AMD Strix Halo with ROCm PyTorch (docs/benchmarks/STRIX_HALO.md)"),
        "tpu": None if "tpu" in hw.engines else "needs a TPU VM with basal[tpu]",
        "ollama": None, "llamacpp": None,  # external services: their URL fields are required then
    }
    reasons["mlx-q8"] = reasons["mlx"] or (None if v.port in ("bf16", "MLX") else
                                           f"{v.port} weights are quantised already (mlx-q8 quantises at load)")
    if v.recurrent:
        for m in ("fast", "fast-nocompile"):
            reasons[m] = reasons[m] or (None if "fla" in hw.engines else FLA_REASON)
        for m in ("fp8", "nvfp4"):  # refused on every GPU: the model's own reason first
            reasons[m] = gdn_refused(m, device=False)
        reasons["fast-exit"] = reasons["mps"] = RECURRENT_REASON
        reasons["mlx"] = reasons["mlx-q8"] = MLX_RECURRENT_REASON
    return [{"mode": m, "available": reasons[m] is None, "reason": reasons[m], "default": m == assessment.mode}
            for m in MODE_ORDER]


def available_media_modes(assessment, hw) -> list[dict]:
    """The modes of a load with media (--media on): eager wherever basal[media] is installed, vLLM on CUDA with
    basal[vllm] (basal-serve refuses the checkpoints vLLM reads differently, basal-omni-1.1-max)."""
    reason = assessment.media_reason if assessment.media_mode is None else None
    vllm = reason or (None if hw.platform == "cuda" and "vllm" in hw.engines else
                      "vLLM media serving needs an NVIDIA GPU with basal[vllm] (docs/OMNI.md)")
    return [{"mode": "eager", "available": reason is None, "reason": reason, "default": True},
            {"mode": "vllm", "available": vllm is None, "reason": vllm, "default": False}]


THINK_REASON = "thinking is available for basal-2.0 4.5B with basal[vllm] (docs/THINKING.md)"


def think_unavailable(v, hw, options=None) -> str | None:
    """Why `think` cannot be turned on for the variant here (None: it can). The reasoning model runs on vLLM: in
    basal-ui's environment, or another one named by think_python, or an already running server at think_url."""
    if v.family != "basal-2.0" or v.tier != "4.5B":
        return THINK_REASON
    options = options or {}
    if "vllm" not in hw.engines and not (options.get("think_python") or options.get("think_url")):
        return THINK_REASON + "; or give the Python of an environment with vLLM, or a running think server's URL"
    return None


def mlx_packed_unavailable(v, hw):
    if hw.platform != "apple" or "mlx" not in hw.engines:
        return "MLX packed needs Apple Silicon with basal[mlx]"
    if v.family not in ("basal-1.0", "basal-1.5", "basal-2.0") or not (v.port == "bf16" or v.port.startswith("MLX")):
        return "MLX packed needs a Llama basal-1.x or basal-2.0 model in bf16 or MLX format"
    return None


def fp8_unavailable(hw):
    """Why fp8 cannot run here, None when it can: basal-serve ≥ 2.3.2 stops --mode fp8 / --quant fp8 before loading
    below compute capability 8.9 (engine.fp8_refused)."""
    if hw.platform == "cuda" and (hw.cuda_capability or (0, 0)) < (8, 9):
        return "FP8 matrix multiplies need compute capability 8.9+ (Ada, Hopper, Blackwell); use fast (bf16) on this GPU"
    return None


def check_combination(v, hw, options, default_mode, media=False):
    """ValueError for the experimental options this variant or machine cannot use (merged options)."""
    mode = options["mode"] or ("eager" if media else default_mode)
    if options["mlx_packed"]:
        reason = mlx_packed_unavailable(v, hw)
        if media or mode not in ("mlx", "mlx-q8"):
            reason = "--mlx-packed needs --mode mlx or mlx-q8 without media"
        if reason:
            raise ValueError(reason)
    if options["batches"] not in ("auto", "default") and MODES[mode][0] not in ("graph", "exit"):
        raise ValueError("--batches compact/fine needs the CUDA graph engine")
    if options["layout"] not in (None, "canonical"):  # basal-serve 2.4.1: the caches hold the canonical prompt only
        try:
            refuse_layout_caches(options["layout"], SimpleNamespace(
                state_cache_gb=options["state_cache_gb"], prefix_cache="on" if options["prefix_cache"] else "off"))
        except SystemExit as exc:  # basal-serve's own message
            raise ValueError(str(exc)) from None
    kind, mode_quant, compiled = MODES[mode]
    quant = options["quant"] or mode_quant
    if v.recurrent and kind == "graph":  # basal-serve 2.4: Qwen3.5 in the graph engine runs bf16 plain/branched rows
        lean, tree = lean_tree(options["lean"], options["tree"], kind, compiled, quant)
        reason = gdn_refused(quant, options["state_cache_gb"], options["prefix_cache"], options["batches"],
                             lean_arg(lean), tree, options["readout_layer"], options["template_prefix"], device=False)
        if not reason and options["long_lane"]:
            reason = "--long-lane is not available for Qwen3.5 (Gated DeltaNet) models"
        if reason:
            raise ValueError(reason)
    if quant == "fp8" and kind in ("graph", "exit"):
        reason = fp8_unavailable(hw)
        if reason:
            raise ValueError(reason)
    if mode == "mps" and options["device"] == "cuda":
        raise ValueError("--mode mps runs on Apple Silicon or CPU; choose --device mps or cpu")
    if options["device"] == "mps" and "mps" not in hw.engines:
        raise ValueError("--device mps needs an available Apple Silicon MPS device")
    if options["think"]:
        reason = think_unavailable(v, hw, options)
        if reason:
            raise ValueError(reason)
    if options["early_exit"] != "off" and (options["mode"] or default_mode) != "fast-exit":
        raise ValueError("early_exit needs --mode fast-exit (the exit heads of basal-1.0, bf16 on CUDA)")
