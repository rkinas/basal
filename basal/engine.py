"""Inference backends. Every backend exposes the same two calls:

  run(prompts, ids_list)          -> one probability list per prompt (softmax over its option letters)
  run_shared(groups[, policy])    -> groups = [(prompts, ids_list)]; the prompts of one group share a prefix (all option
                                     orders of one question, or every branch of a whole request: state once, ask many)

A prompt is a string or a list of token ids (the server tokenizes once and passes ids).

Backends:
  EagerBackend      plain PyTorch forward (reference, any GPU, Apple MPS or CPU); Gemma 4 (basal-omni) through
                    basal/gemma4.py: the shared prefix of a group once, the last layers at the answer rows only
  GraphBackend      static shapes + CUDA graphs, optional torch.compile, shared prefix for the two option orders,
                    token-budget batching, optional torchao FP8 / NVFP4 quantisation
  ExitGraphBackend  GraphBackend split into graph segments at trained early-exit layers; exit policy per request
  MPSBackend        GraphBackend's shared prefix + token-budget batching without graphs, on Apple MPS or the CPU
  VLLMBackend       vLLM: bf16 checkpoints, ModelOpt FP8 / NVFP4 ports, a repeatable preset, basal-omni media
  SGLangBackend     SGLang offline engine (1.5): the same letter readout from the log-probabilities of the option tokens
  MLXBackend        Apple Silicon, MLX / mlx-lm (1.5): MLX fp8 / fp4 exports or bf16 weights (optional 8-bit at load
                    time); shared-prefix KV cache
  GGUFBackend       a GGUF file in process through llama.cpp (llama-cpp-python; Metal, CUDA or CPU); shared prefix as
                    llama.cpp sequences
  OllamaBackend     Ollama with the GGUF exports (1.5): raw prompt, one token, top log-probabilities of the letters
  LlamaCppBackend   llama.cpp server with the GGUF exports (1.5): token-id prompt, `n_probs` log-probabilities
  TPUBackend        (basal/tpu.py, extra basal[tpu]; experimental) the same SOAM packing in JAX / XLA on Google TPU
"""
import functools
import json
import os
import re
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import PREFILL, RequestError


METADATA_FILES = ["*.json", "*.jinja", "*.model", "*.txt", "*.md", "Modelfile*"]  # tokenizer, template, calibration


def resolve(name, revision=None, metadata_only=False):
    """Local directory or Hugging Face repo id (downloaded once to the HF cache). metadata_only: tokenizer, chat
    template and calibration files only (the Ollama / llama.cpp modes, where the engine serves the GGUF weights)."""
    p = Path(name)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision, allow_patterns=METADATA_FILES if metadata_only else None))


def default_device():
    """BASAL_DEVICE if set, else cuda, else Apple Silicon (mps), else cpu."""
    import os
    if os.environ.get("BASAL_DEVICE"):
        return os.environ["BASAL_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eager_sdpa_kernels(device):
    """The SDPA kernels the eager backend's transformers forward may use: None (every enabled one), or, on CUDA with
    torch older than 2.14.1, the enabled ones without the memory-efficient kernel. That kernel returns wrong attention
    for keys / values expanded from one KV head (zero stride, transformers' repeat_kv with a mask) at query lengths
    = 1 mod 32 (head size 256 / 512) or 1 mod 64 (128): pytorch/pytorch#191937, fixed in 2.14.1; basal-omni max was off
    by 6.2 in code log-prob on a 1,857-token prompt (2026-10-07)."""
    from torch.torch_version import TorchVersion
    if torch.device(device).type != "cuda" or TorchVersion(torch.__version__) >= "2.14.1":
        return None
    from torch.nn.attention import SDPBackend
    on = ((SDPBackend.FLASH_ATTENTION, torch.backends.cuda.flash_sdp_enabled()),
          (SDPBackend.CUDNN_ATTENTION, torch.backends.cuda.cudnn_sdp_enabled()),
          (SDPBackend.MATH, torch.backends.cuda.math_sdp_enabled()))
    return [k for k, enabled in on if enabled]


def _memo_media(proc):
    """The processor prepares a request's images / audio once for all its prompts (questions and option orders): at
    1,120 soft tokens the image preprocessing took ~2.8 s a call against 0.65 s for the forward (measured
    2026-10-07). Keyed on the media objects, which the memo keeps alive (their ids are not reused); one request at a
    time (EagerBackend.media_lock)."""
    for name in ("_process_images", "_process_audio"):
        f, last = getattr(proc, name, None), {}
        if f is None:
            continue

        def wrapped(x, _f=f, _last=last, **kw):
            flat = [i for sub in x for i in (sub if isinstance(sub, (list, tuple)) else [sub])]
            key = (tuple(id(i) for i in flat), repr(sorted(kw.items())))
            if _last.get("key") != key:
                _last.update(key=key, ref=flat, val=_f(x, **kw))
            return _last["val"]
        setattr(proc, name, wrapped)
    return proc


class EagerBackend:
    # one forward at a time (text batches and media requests share _head_in), one media request at a time (the
    # processor's media memo); class defaults, so a backend built without __init__ (tests) has them too
    lock = threading.Lock()
    media_lock = threading.Lock()
    g4 = None  # basal/gemma4.py: the own Gemma 4 forward (shared prefixes, answer rows), None = the transformers forward
    sdpa_kernels = None  # eager_sdpa_kernels: the SDPA kernels of the transformers forward (None: every enabled one)

    def __init__(self, model_dir, dtype="bfloat16", device=None, media=False, forward=None, ple_offload=False,
                 attn="math"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        dev = device or default_device()
        # media (basal-omni, --media on): the model with its image / audio encoders and the processor; text requests
        # read the same decoder as AutoModelForCausalLM does
        self.media_kinds, self.max_audio_s, self.proc = (), None, None
        cls = AutoModelForCausalLM
        if media:
            from .media import audio_window, kinds_of
            self.media_kinds = kinds_of(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")))
            if not self.media_kinds:
                raise SystemExit("--media on: the model has no image or audio encoder (config.json: vision_config, "
                                 "audio_config)")
            from transformers import AutoModelForImageTextToText, AutoProcessor
            cls, self.proc = AutoModelForImageTextToText, AutoProcessor.from_pretrained(model_dir)
            pc = Path(model_dir) / "processor_config.json"
            self.max_audio_s = audio_window(json.loads(pc.read_text(encoding="utf-8"))) if pc.exists() else None
            _memo_media(self.proc)
        self.lock, self.media_lock = threading.Lock(), threading.Lock()
        self.model = cls.from_pretrained(model_dir, dtype=getattr(torch, dtype))
        if ple_offload:  # --ple-offload on (Gemma 4 E2B / E4B): the per-layer embedding table stays in host memory
            from .gemma4 import offload_ple
            if not offload_ple(self.model):
                raise SystemExit("--ple-offload on: the model has no per-layer embeddings (Gemma 4 E2B / E4B)")
        self.model = self.model.to(dev).eval()
        if torch.device(dev).type == "cuda":
            # no cuDNN attention for any model (the graph engine never uses it either): it builds an execution plan for
            # every new prompt length (H100, torch 2.11: 573 ms for a new length, 20 ms for a repeated one), so
            # a Qwen3 model and basal-2.0 in eager paid ~0.45 s a request
            torch.backends.cuda.enable_cudnn_sdp(False)
        from .gemma4 import is_gemma4
        if is_gemma4(self.model):
            # Gemma 4 (basal-omni): SDPA's memory-efficient kernel returned wrong attention on real activations (omni
            # kernel check 2026-10-07: expanded keys / values at query lengths = 1 mod 32, pytorch/pytorch#191937, as
            # the transformers forward passes max's one global KV head); the math kernel equals eager attention there
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            # basal 2.3: the engine's own Gemma 4 forward (basal/gemma4.py) unless --eager-forward hf (or
            # BASAL_EAGER_FORWARD=hf), the transformers forward of 2.2.0
            import os
            forward = forward or os.environ.get("BASAL_EAGER_FORWARD") or "basal"
            if forward not in EAGER_FORWARDS:
                raise ValueError(f"eager forward {forward!r} (one of {', '.join(EAGER_FORWARDS)})")
            if forward == "basal":
                from .gemma4 import Gemma4Forward, attention_plan
                self.g4 = Gemma4Forward(self.model, attn=attention_plan(attn))
        self.dev = dev
        self.sdpa_kernels = eager_sdpa_kernels(dev)
        self.prefill = PREFILL
        # option-code logits in fp32, as in GraphBackend._rows: the output head's input (the normalised last hidden
        # state) is taken from a hook and multiplied by the fp32 code rows; the model's final logit softcap (Gemma)
        # is applied in fp32 as well, so only the model body runs in the serving dtype
        self.head = self.model.get_output_embeddings()
        cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self._head_in = None
        self.head.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._head_in = inputs[0]

    def _forward(self, **kw):
        """The transformers forward on the SDPA kernels of sdpa_kernels."""
        if self.sdpa_kernels is None:
            return self.model(**kw)
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(self.sdpa_kernels):
            return self.model(**kw)

    def encode(self, prompts):
        """Strings -> token id lists (one batched tokenizer call); id lists pass through."""
        txt = [k for k, p in enumerate(prompts) if isinstance(p, str)]
        out = list(prompts)
        if txt:
            for k, ids in zip(txt, self.tok([prompts[k] for k in txt], add_special_tokens=False).input_ids):
                out[k] = ids
        return out

    # memory bounds of the eager forward (the attention scores are B x heads x L x L on the math kernel): prompts longer
    # than CHUNK run alone, prefilled CHUNK tokens at a time through the KV cache; shorter ones are batched by length up
    # to MAX_BATCH_TOKENS padded tokens (basal-omni 2026-10-08: concurrent long contracts made the server run out of
    # GPU memory and fail its requests)
    CHUNK = 2048
    MAX_BATCH_TOKENS = 16384

    def run(self, prompts, ids_list):
        if self.g4 is not None:
            return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]
        with self.lock:
            return self._run(prompts, ids_list)

    @torch.no_grad()
    def run_media(self, msgs_list, ids_list, images, audio, max_soft_tokens, max_tokens=None):
        """Prompts of one request with media (basal/media.py; the same images / audio in every prompt), one forward
        each, as the basal-omni evaluation reads them: [(position probabilities, prompt tokens, media tokens)]. A prompt over
        max_tokens -> ValueError before any forward."""
        with self.media_lock:
            encs = []
            for msgs in msgs_list:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) + self.prefill
                kw = dict(text=[text], return_tensors="pt")
                if images:
                    kw.update(images=[images], images_kwargs=dict(max_soft_tokens=max_soft_tokens))
                if audio:
                    kw.update(audio=audio)
                enc = self.proc(**kw)
                ids = enc["input_ids"][0]
                n_media = int(sum((ids == t).sum() for t in (getattr(self.proc, "image_token_id", None),
                                                             getattr(self.proc, "audio_token_id", None)) if t is not None))
                if max_tokens and ids.shape[0] > max_tokens:
                    raise RequestError(f"the prompt with its media is {ids.shape[0]} tokens, more than the "
                                       f"{max_tokens} tokens of the context; send fewer or shorter media, or detail "
                                       "\"default\"")
                encs.append((enc, int(ids.shape[0]), n_media))
            if self.g4 is not None:  # the towers once, the prompts' shared prefix (template, media, state) once
                with self.lock:
                    feats = self.g4.media_features(encs[0][0])
                    hid = self.g4.hidden([[e["input_ids"][0].tolist() for e, _, _ in encs]], media=feats)[0]
                    ps = self._codes(torch.stack(hid).float(), ids_list)
                return [(p, n, n_media) for p, (_, n, n_media) in zip(ps, encs)]
            out = []
            for (enc, n, n_media), lid in zip(encs, ids_list):
                x = {k: (v.to(self.dev, self.model.dtype) if v.is_floating_point() else v.to(self.dev))
                     for k, v in enc.items() if isinstance(v, torch.Tensor)}
                with self.lock:
                    self._forward(**x, logits_to_keep=1)
                    p = self._codes(self._head_in[:, -1, :].float(), [lid])[0]
                out.append((p, n, n_media))
            return out

    @torch.no_grad()
    def _run(self, prompts, ids_list):
        enc = self.encode(prompts)
        out, cur = [None] * len(enc), []
        for k in sorted(range(len(enc)), key=lambda k: len(enc[k])):
            if len(enc[k]) > self.CHUNK:
                out[k] = self._chunked(enc[k], ids_list[k])
                continue
            if cur and (len(cur) + 1) * len(enc[k]) > self.MAX_BATCH_TOKENS:
                for j, p in zip(cur, self._batch([enc[j] for j in cur], [ids_list[j] for j in cur])):
                    out[j] = p
                cur = []
            cur.append(k)
        if cur:
            for j, p in zip(cur, self._batch([enc[j] for j in cur], [ids_list[j] for j in cur])):
                out[j] = p
        return out

    def _chunked(self, e, lid):
        """One long prompt, prefilled in CHUNK-token parts through the KV cache (exact up to float rounding)."""
        from transformers import DynamicCache
        # a hybrid model (Qwen3.5: Gated DeltaNet and attention layers) needs its layer types in the cache, so the
        # linear-attention layers carry their conv and recurrent states from part to part (an empty DynamicCache()
        # raised IndexError there); attention-only models (Llama, Gemma 4) keep the plain cache
        cfg = self.model.config
        types = getattr(cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg, "layer_types", None) or ()
        cache = DynamicCache(config=cfg) if "linear_attention" in types else DynamicCache()
        ids = torch.tensor([list(e)], device=self.dev)
        for s0 in range(0, ids.shape[1], self.CHUNK):
            part = ids[:, s0:s0 + self.CHUNK]
            pos = torch.arange(s0, s0 + part.shape[1], device=self.dev)[None]
            cache = self._forward(input_ids=part, position_ids=pos, past_key_values=cache, use_cache=True,
                                  logits_to_keep=1).past_key_values
        return self._codes(self._head_in[:, -1, :].float(), [lid])[0]

    def _batch(self, enc, ids_list):
        L = max(map(len, enc))
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(e)) + list(e) for e in enc], device=self.dev)
        att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=self.dev)
        # positions that skip the left pads, so a padded prompt reads as it does alone: on Gemma 4 (basal-omni) the
        # default arange moved a 44-option distribution by 0.12 with 1,458 pads (RoPE is relative only up to rounding)
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        self._forward(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1)
        return self._codes(self._head_in[:, -1, :].float(), ids_list)

    def _codes(self, h, ids_list):
        out = []
        for b, lid in enumerate(ids_list):  # softmax over the code rows = softmax over the codes of the full log-softmax
            sel = torch.tensor(lid, device=h.device)
            z = h[b] @ self.head.weight.index_select(0, sel).float().T
            if self.head.bias is not None:
                z = z + self.head.bias.index_select(0, sel).float()
            if self.softcap:
                z = torch.tanh(z / self.softcap) * self.softcap
            out.append(torch.softmax(z, -1).tolist())
        return out

    def run_shared(self, groups, policy=None):
        if self.g4 is not None:  # Gemma 4: the shared prefix of each group once (SOAM)
            with self.lock, torch.no_grad():
                hid = self.g4.hidden([self.encode(g[0]) for g in groups])
                flat = self._codes(torch.stack([h for g in hid for h in g]).float(), [x for g in groups for x in g[1]])
        else:
            flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


EAGER_FORWARDS = ("basal", "hf")
ATTN = ("auto", "fa2", "fa3", "fa4", "cudnn")
_ATTN = {"impl": "fa2", "direct": None}  # the attention of causal forwards without a mask (process-wide)


def attention_available(impl):
    """None if `impl` can run here, else why not (an optional package, the GPU)."""
    if impl == "fa2":
        return None
    if not torch.cuda.is_available():
        return "needs CUDA"
    major = torch.cuda.get_device_capability()[0]
    if impl == "fa3":
        if major != 9:
            return "FlashAttention 3 runs on Hopper (sm_90) only"
        try:
            import flash_attn_interface  # noqa: F401  (the flash_attn_3 wheel)
        except ImportError:
            return ("the flash_attn_3 package is not installed (uv pip install flash_attn_3 --extra-index-url "
                    "https://download.pytorch.org/whl/cu128)")
        return None
    if impl == "fa4":
        if major < 9:
            return "FlashAttention 4 runs on Hopper and Blackwell (sm_90+)"
        try:
            from flash_attn.cute import interface  # noqa: F401  (the flash-attn-4 wheel)
        except ImportError:
            return "the flash-attn-4 package is not installed (pip install --pre flash-attn-4)"
        return None
    if impl == "cudnn":
        return None if torch.backends.cudnn.is_available() else "cuDNN is not available"
    if impl == "flex":  # Gemma 4's FlexAttention: Inductor generates its kernel with Triton (not in Windows wheels)
        import importlib.util
        return None if importlib.util.find_spec("triton") else "FlexAttention needs Triton (the triton package)"
    return f"unknown attention {impl!r} (one of {', '.join(ATTN)})"


def auto_attention(fp8=False):
    """--attn auto: FlashAttention 3 on Hopper when the flash_attn_3 wheel is installed (gated on an H100: bf16 decisions
    against the plain path >= 0.99 on every path, dev v2 accuracy within 0.2 points of FA2), else torch's built-in
    FlashAttention 2. fp8 (an FP8 checkpoint or --mode fp8): FlashAttention 2, the checked path (the FP8 ports were
    gated, measured and calibrated on it); with FA3, 3-8% of FP8 decisions differ from the fa2 rows, mostly near
    ties, and accuracy moved by -0.8 to +1.4 points against them (dev sample, H100, 2026-10-12); --attn fa3 still
    selects FA3. FA4 and cuDNN run when asked for (--attn fa4 / cudnn) until they are gated on Blackwell."""
    if fp8:
        return "fa2"
    return "fa3" if attention_available("fa3") is None else "fa2"


def gemma_fast_kernels():
    """The kernels of basal-omni's (Gemma 4) compiled fast forward where it passed its gate on this GPU class, else None
    (the math kernel, uncompiled; eager without --mode): fa3 where FlashAttention 3 runs (Hopper; E3, 20 Oct), flex on
    Blackwell workstation GPUs (sm_120: FlexAttention with the smaller tiles of 99 KB of shared memory per block; RTX
    PRO 6000, 20 Oct)."""
    if attention_available("flex"):  # both run FlexAttention (fa3: in the 512-wide global layers)
        return None
    if attention_available("fa3") is None:
        return "fa3"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        return "flex"
    return None


def set_attention(choice="auto", fp8=False):
    """Choose the kernel of the causal forwards without a mask (plain prompts and the root pass of branched units):
    fa2 = torch SDPA's built-in FlashAttention 2, fa3 = FlashAttention 3 in SDPA's flash slot (torch's registry),
    fa4 = flash-attn-4 called directly, cudnn = SDPA's cuDNN kernel. Masked attention (the branch pass) and fp32 keep
    SDPA. Process-wide; call before any graph is captured. fp8: the model runs FP8 GEMMs (auto_attention).
    -> the attention in use."""
    impl = auto_attention(fp8) if choice == "auto" else choice
    why = attention_available(impl)
    if why:
        raise ValueError(f"--attn {impl}: {why}")
    if _ATTN["impl"] == "fa3" and impl != "fa3":
        torch.nn.attention.restore_flash_attention_impl()
    if impl == "fa3" and _ATTN["impl"] != "fa3":
        torch.nn.attention.activate_flash_attention_impl("FA3")
    _ATTN.update(impl=impl, direct=_direct_attention(impl))
    return impl


def _direct_attention(impl):
    """A causal attention for the kernels that SDPA does not route to: [b, heads, L, d] in, [b, L, heads, d] out."""
    if impl == "fa4":
        from flash_attn.cute import interface as fa4

        @torch.compiler.disable
        def run(q, k, v, scale):
            o = fa4.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scale,
                                    causal=True)
            return o[0] if isinstance(o, (tuple, list)) else o
        return run
    if impl == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        @torch.compiler.disable
        def run(q, k, v, scale):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale,
                                                                     enable_gqa=True)
            return o.transpose(1, 2).contiguous()
        return run
    return None


def torch_varlen_available(dtype):
    """None if torch's varlen attention (torch.nn.attention.varlen.varlen_attn: torch's built-in FlashAttention 2 on
    packed sequences, with GQA) can run compact rows here, else why not."""
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability()[0] < 8:
        return "needs an sm80+ GPU"
    if dtype not in (torch.bfloat16, torch.float16):
        return "needs bf16 / fp16 activations"
    if not torch.backends.cuda.is_flash_attention_available():  # Windows wheels: built without USE_FLASH_ATTENTION
        return "this torch is built without FlashAttention"
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError:
        return "needs torch with torch.nn.attention.varlen"
    import inspect
    if "enable_gqa" not in inspect.signature(varlen_attn).parameters:
        return "this torch's varlen_attn has no enable_gqa"
    return None


def compact_refused(why):
    return ("--batches compact needs FlashAttention 3 (--attn fa3 / auto on Hopper; Gemma 4: --gemma-attn fa3) or "
            f"torch's varlen attention ({why})")


# config.json "layer_types" of attention layers; any other kind (Qwen3.5's linear_attention: Gated DeltaNet, Mamba,
# conv) carries a recurrent state from token to token
ATTENTION_LAYERS = {"full_attention", "sliding_attention", "chunked_attention"}
RECURRENT_MODEL_TYPES = ("qwen3_5", "qwen3_next", "mamba", "jamba", "zamba", "bamba", "falcon_h1", "granitemoehybrid",
                         "nemotron_h", "lfm2", "recurrent_gemma", "rwkv")


def recurrent_layers(model_dir):
    """The recurrent-state layers of a model (Gated DeltaNet / linear attention as in Qwen3.5, Mamba, conv), read from
    its config.json before any weights are -> e.g. "18 linear_attention layers" or "model_type qwen3_5_text"; None for
    an attention-only decoder or without a readable config.json."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cfgs = (cfg, cfg.get("text_config") or {})  # multimodal models: the decoder's settings in text_config
    for c in cfgs:
        kinds = [k for k in c.get("layer_types") or () if k not in ATTENTION_LAYERS]
        if kinds:
            return ", ".join(f"{kinds.count(k)} {k}" for k in sorted(set(kinds))) + " layers"
    for c in cfgs:
        if str(c.get("model_type") or "").startswith(RECURRENT_MODEL_TYPES):
            return f"model_type {c['model_type']}"
    return None


def recurrent_refused(model, what, gdn=False):
    """The refusal of the shared-prefix engines for a model with recurrent-state layers. Their prefix tries, two passes,
    compact rows and caches hold attention keys and values only: a Gated DeltaNet layer would restart from zero state
    below a shared prefix and run across the prompts of a packed row, so the answers would be wrong without any error
    (v2.3.1 with the 0.8B pico, 21 Oct). Since 2.4 the CUDA graph engine carries the recurrent state of Qwen3.5 decoders
    (gdn: basal/gdn_graph.py, --mode fast and fast-nocompile); fast-exit and mps still refuse them, and every other
    recurrent model stays refused by all of them."""
    if gdn:
        return (f"{model}: {what} (recurrent state: Gated DeltaNet, Qwen3.5). --mode fast-exit and --mode mps keep "
                "attention keys and values only and would answer wrongly; serve it with --mode fast (CUDA), --mode "
                "eager (exact) or --mode vllm")
    return (f"{model}: {what} (recurrent state: Gated DeltaNet / linear attention, as in Qwen3.5). The CUDA graph "
            "engine (--mode fast, fast-nocompile, fast-exit, fp8, nvfp4) carries the recurrent state of Qwen3.5 "
            "decoders only, and --mode mps of none: they would answer wrongly; serve it with --mode eager (exact) or "
            "--mode vllm (faster), or --mode gguf with a GGUF file")


def fp8_refused():
    """Why torchao FP8 (--mode fp8, --quant fp8) cannot run here, None if it can: its FP8 matrix multiplies need a GPU
    of compute capability 8.9 or newer (Ada, Hopper, Blackwell); on older GPUs torchao failed only after the model was
    loaded (RTX 3090, #33)."""
    if not torch.cuda.is_available():
        return "--mode fp8 needs a CUDA GPU of compute capability 8.9 or newer (Ada, Hopper, Blackwell)"
    cc = torch.cuda.get_device_capability()
    if tuple(cc) < (8, 9):
        return (f"--mode fp8 needs a GPU of compute capability 8.9 or newer (Ada, Hopper, Blackwell) for its FP8 "
                f"matrix multiplies; {torch.cuda.get_device_name()} is sm_{cc[0]}{cc[1]}. Serve with --mode fast "
                "(bf16) on this GPU")
    return None


def _copy_same(dst, src):
    """dst.copy_(src) for a captured graph's shared output: the same dtype and shape, so the copy is exact."""
    if dst.dtype != src.dtype or dst.shape != src.shape:
        raise RuntimeError(f"graph output {tuple(src.shape)} {src.dtype} does not fit its buffer {tuple(dst.shape)} "
                           f"{dst.dtype}")
    return dst.copy_(src)


def _graph_tokens(key):
    """A graph's size for the capture order: (tokens it runs, pass 2: query-key pairs). key: ("p", rows, length),
    ("c", tokens) or ("x", rows, branch length, root length)."""
    if key[0] == "p":
        return key[1] * key[2], 0
    if key[0] == "c":
        return key[1], 0
    return key[1] * key[2], key[1] * key[2] * (key[3] + key[2])


def _varlen_attention(q, k, v, cu, scale, layer=0, impl="fa3"):
    """Causal attention of compact rows: [1, heads, T, d] in (prompts packed back to back, cu = their cumulative token
    counts, int32, empty prompts at the end), [1, T, heads, d] out. impl (the backend's compact_kernel): "fa3" =
    FlashAttention 3's varlen kernel, "torch varlen" = torch's varlen_attn (torch's FlashAttention 2, any sm80+ GPU);
    both are torch custom ops, so the compiled forward has no graph break. FA3: the same rows as the padded forward on
    an H100.
    With a template prefix cu = (cu_q, cu_k, gather, prefix): every prompt's keys / values are its template prefix's
    (prefix[layer] = (K, V) [prefix tokens, kv heads, d], computed at start-up) followed by its own; gather picks them
    from prefix + own tokens, and the queries (own tokens) attend bottom-right causally."""
    T = q.shape[2]
    k, v = k[0].transpose(0, 1), v[0].transpose(0, 1)
    if isinstance(cu, tuple):
        cu, cu_k, gather, prefix = cu
        pk, pv = prefix[layer]
        k = torch.cat([pk, k]).index_select(0, gather)
        v = torch.cat([pv, v]).index_select(0, gather)
    else:
        cu_k = cu
    if impl == "torch varlen":
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q[0].transpose(0, 1), k, v, cu, cu_k, T, k.shape[0], scale=scale, window_size=(-1, 0),
                           enable_gqa=True)[None]
    import flash_attn_interface as fa3
    o = fa3.flash_attn_varlen_func(q[0].transpose(0, 1), k, v, cu, cu_k, T, k.shape[0], softmax_scale=scale,
                                   causal=True)
    return (o[0] if isinstance(o, (tuple, list)) else o)[None]


def basal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                    basal_kv=None, basal_prefix=None, basal_cu=None, basal_varlen="fa3", basal_tree=None, **kwargs):
    """torch SDPA with additions for the two-pass forward of branched units and for compact rows (the kwargs pass
    through the transformers model forward): basal_kv[layer] = (K, V) buffers that receive this layer's keys and
    values (the root pass), basal_prefix[layer] = (K, V) of a computed prefix that the queries also attend to (the
    branch pass), basal_cu = cumulative token counts of prompts packed into one row (each attends to itself only),
    basal_varlen = the kernel of those rows (_varlen_attention's impl), basal_tree = the branch pass without dense
    masks: the queries attend to the prefix and, per trie block, to their own ancestors separately, merged by
    log-sum-exp (no concatenated keys, no [N, P + N] mask); _block_plan's batches on the CPU, a _VarlenTree on CUDA.
    A causal forward without a mask uses the kernel chosen by set_attention (--attn)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if basal_cu is not None:
        return _varlen_attention(query, key, value, basal_cu, scaling, module.layer_idx, basal_varlen), None
    if basal_tree is not None:
        k, v = basal_prefix[module.layer_idx]
        if isinstance(basal_tree, _VarlenTree):
            return _cascade_varlen(query, key, value, k, v, basal_tree, scaling), None
        o_p, l_p = _attention_lse(query, k, v, scaling)
        o_b, l_b = _branch_attention(query, key, value, scaling, basal_tree)
        return _merge_attention(o_p, l_p, o_b, l_b).transpose(1, 2).contiguous(), None
    if basal_kv is not None:
        k, v = basal_kv[module.layer_idx]
        k.copy_(key); v.copy_(value)
    if basal_prefix is not None:
        k, v = basal_prefix[module.layer_idx]
        key, value = torch.cat([k, key], 2), torch.cat([v, value], 2)
    direct = _ATTN["direct"]
    if (direct is not None and attention_mask is None and basal_prefix is None and query.shape[2] > 1
            and query.dtype in (torch.bfloat16, torch.float16)):
        return direct(query, key, value, scaling), None
    groups = getattr(module, "num_key_value_groups", 1)
    if attention_mask is not None and key.shape[1] == 1 and groups > 1:
        # one key / value head: transformers' repeat_kv would hand SDPA a stride-0 view, and the memory-efficient
        # kernel (the dispatcher's pick for a masked call) then returns a wrong last query row on torch < 2.14.1 at
        # query lengths 1 mod 64 (head size 128; 1 mod 32 at 256 and 512): pytorch/pytorch#191937, fp32 last rows off
        # by ~3 on an H100 (20 Oct). The heads are copied here instead (basal-2.0 has 2 or 8 key / value heads, which
        # repeat_kv already copies)
        key, value = (x.expand(-1, groups, -1, -1).contiguous() for x in (key, value))
        module = SimpleNamespace(num_key_value_groups=1, is_causal=getattr(module, "is_causal", True))
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kwargs)


def _attention_lse(q, k, v, scale, mask=None):
    """Attention and its log-sum-exp on the CPU (SDPA's CPU flash kernel, the one SDPA picks there): q [B, H, S, d],
    k / v [B, Hkv, L, d], mask bool broadcastable to [B, 1, S, L] or None -> (o [B, H, S, d], lse [B, H, S] fp32).
    Without a mask the H / Hkv query heads of each key / value head are folded into the query length (grouped-query
    attention without copying keys)."""
    flash = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    B, H, S, d = q.shape
    r = H // k.shape[1]
    if mask is None:
        o, lse = flash(q.reshape(B, k.shape[1], r * S, d), k, v, scale=scale)
        return o.reshape(B, H, S, d), lse.reshape(B, H, S)
    # finfo.min, not -inf: on a row of -inf only the kernel returns lse 0 instead of -inf (_block_plan gives every row
    # a key, so this is a guard)
    add = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(~mask, torch.finfo(q.dtype).min)
    if r > 1:
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    return flash(q, k, v, attn_mask=add, scale=scale)


def _merge_attention(o1, l1, o2, l2):
    """Two attentions over disjoint key sets with their log-sum-exp -> the attention over the union: the softmax over
    all the keys, summed in another order. Weights in fp32; a part that saw no key weighs 0."""
    lse = torch.logaddexp(l1, l2)
    w1, w2 = torch.exp(l1 - lse)[..., None], torch.exp(l2 - lse)[..., None]
    return (torch.where(w1 > 0, o1.float() * w1, 0.0) + torch.where(w2 > 0, o2.float() * w2, 0.0)).to(o1.dtype)


def _block_plan(seg, parents, waste=1.5):
    """The branch pass per trie block, without a trie mask: block b's queries see the tokens of its ancestor blocks
    below the root and its own earlier tokens, i.e. keys = (ancestors' tokens, b's tokens) with a causal mask aligned at
    the end. seg: block of every token below the root (0..; each block contiguous, flattened depth first), parents: the
    parent block of every block (-1 = the root); a slice of whole top-level subtrees works as well. -> [(queries
    [n, m], keys [n, k], mask bool [n, 1, m, k])]: token indices (len(seg) = padding) of blocks of similar shape
    batched, padded to the batch's longest, at most `waste` x the real query-key pairs per batch (one block alone may
    exceed it)."""
    N = len(seg)
    span = {}
    for i, s in enumerate(seg):
        span[s] = (span.get(s, (i,))[0], i + 1)
    items = []
    for blk, (a, b) in span.items():
        anc, p = [], parents[blk]
        while p >= 0:
            anc = list(range(*span[p])) + anc
            p = parents[p]
        items.append((list(range(a, b)), anc))
    items.sort(key=lambda x: (len(x[0]), len(x[1])))
    plan, cur = [], []
    for it in items + [None]:
        if it is not None:
            nxt = cur + [it]
            m, k = max(len(q) for q, _ in nxt), max(len(q) + len(an) for q, an in nxt)
            if not cur or len(nxt) * m * k <= waste * sum(len(q) * (len(q) + len(an)) for q, an in nxt):
                cur = nxt
                continue
        m = max(len(q) for q, _ in cur)
        k = max(len(q) + len(an) for q, an in cur)
        qi = torch.full((len(cur), m), N, dtype=torch.long)
        ki = torch.full((len(cur), k), N, dtype=torch.long)
        mask = torch.zeros((len(cur), 1, m, k), dtype=torch.bool)
        i, j = torch.arange(m)[:, None], torch.arange(k)[None]
        for r, (q, an) in enumerate(cur):
            qi[r, : len(q)] = torch.tensor(q)
            ki[r, : len(an) + len(q)] = torch.tensor(an + q)
            mask[r, 0] = (j <= len(an) + i) & (j < len(an) + len(q))
            mask[r, 0, len(q):, 0] = True  # padding queries see one key (no row without keys)
        plan.append((qi, ki, mask))
        cur = [it] if it is not None else []
    return plan


def attention_only(cfg):
    """A model whose layers all attend to every earlier key (the Llama-based basal models): no sliding-window or
    linear-attention layers, so every key a branch token sees is in the root or in its block's ancestors and the branch
    pass can run as a cascade (_cascade on the CPU, _cascade_varlen on CUDA)."""
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return set(getattr(cfg, "layer_types", None) or ["full_attention"]) == {"full_attention"} \
        and not getattr(cfg, "sliding_window", None)


def cascade_ok(cfg):
    """The CPU branch pass of _cascade: attention-only models (attention_only); models with sliding-window or
    linear-attention layers keep the dense pass over the transformers cache, as does a torch without the CPU flash
    kernel _attention_lse calls (a private aten op; basal pins only torch>=2.11)."""
    return attention_only(cfg) and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu")


def _branch_attention(q, k, v, scale, plan):
    """The branch part of the branch pass: q [1, H, N, d], k / v [1, Hkv, N, d] (the tokens below the root) and
    _block_plan's batches -> (o [1, H, N, d], lse [1, H, N] fp32)."""
    _, H, N, d = q.shape
    o = q.new_empty(H, N + 1, d)
    lse = torch.empty(H, N + 1, dtype=torch.float32, device=q.device)
    qp, kp, vp = (torch.cat([x[0], x.new_zeros(x.shape[1], 1, d)], 1) for x in (q, k, v))  # [heads, N + 1, d]
    for qi, ki, mask in plan:
        ob, lb = _attention_lse(qp[:, qi].transpose(0, 1), kp[:, ki].transpose(0, 1), vp[:, ki].transpose(0, 1),
                                scale, mask)
        o[:, qi] = ob.transpose(0, 1)
        lse[:, qi] = lb.transpose(0, 1)
    return o[None, :, :N], lse[None, :, :N]


class _VarlenTree(NamedTuple):
    """The branch pass of B rows of S tokens on CUDA (_cascade_varlen), built by _varlen_tree. Root part: every token
    of row r over the first used[r] keys of its root buffer (cu_qr / cu_kr: S / Lk tokens per row). Branch part: per
    trie block, its queries (flat indices qi, cu_q) over its ancestors' and own tokens (flat indices ki, cu_k), causal
    at the end; full: no padding tokens (qi is every token in order); fused: the merge compiled (_merge_fused)."""
    cu_qr: torch.Tensor
    cu_kr: torch.Tensor
    used: torch.Tensor
    qi: torch.Tensor
    ki: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    full: bool
    fused: bool = False


def _varlen_tree(seg, parents, roots, Lk, to_dev, fused=False):
    """seg [B, S] numpy: the block of every token below the root (0..; contiguous, flattened depth first; -1 =
    padding), parents: per row the parent block of every block (-1 = the root), roots: the root's tokens per row, Lk:
    the length of the root key / value buffers, fused: merge with _merge_fused -> _VarlenTree on the device (to_dev:
    host array -> device)."""
    B, S = seg.shape
    qi, ki, cq, ck = [], [], [0], [0]
    for r in range(B):
        s = seg[r]
        n = int((s >= 0).sum())
        starts = np.flatnonzero(np.r_[True, s[1:n] != s[: n - 1]]) if n else np.zeros(0, dtype=np.int64)
        runs = list(zip(starts.tolist(), np.r_[starts[1:], n].tolist()))
        span = {int(s[a]): (a, b) for a, b in runs}
        for a, b in runs:
            keys, p = [np.arange(a, b)], parents[r][int(s[a])]
            while p >= 0:
                keys.insert(0, np.arange(*span[p]))
                p = parents[r][p]
            ki.append(r * S + np.concatenate(keys))
            qi.append(r * S + np.arange(a, b))
            cq.append(cq[-1] + b - a)
            ck.append(ck[-1] + len(ki[-1]))
    qi, ki = np.concatenate(qi), np.concatenate(ki)
    i32 = lambda x: to_dev(np.asarray(x, dtype=np.int32))  # noqa: E731
    return _VarlenTree(i32(np.arange(B + 1) * S), i32(np.arange(B + 1) * Lk), i32(roots), to_dev(qi), to_dev(ki),
                       i32(cq), i32(ck), int(np.diff(cq).max()), int(np.diff(ck).max()), len(qi) == B * S, fused)


def _varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal=False, used=None):
    """torch's varlen attention (FlashAttention 2 on packed sequences, grouped-query) with its log-sum-exp: q [Tq, H,
    d], k / v [Tk, Hkv, d], cu_q / cu_k cumulative lengths (int32), used: the keys each sequence uses (seqused_k) or
    None, causal: aligned at the end of each sequence -> (o [Tq, H, d], lse [Tq, H] fp32). The log-sum-exp comes as
    [H, Tq] (torch 2.13, RTX 4090). The op varlen_attn (torch.nn.attention.varlen) runs, with the arguments it passes,
    called directly: varlen_attn wraps it in a Python custom op and allocates a random-number state per call, host
    time that an uncompiled pass 2 paid twice per layer (_cascade_varlen), more than the kernels took on a fast GPU."""
    o, lse, *_ = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, False, scale=scale, window_size_left=-1,
        window_size_right=0 if causal else -1, seqused_k=used)
    return o, (lse.transpose(0, 1) if lse.shape == (q.shape[1], q.shape[0]) else lse).float()


def _cascade_varlen(q, k, v, kp, vp, tree, scale):
    """The branch pass on CUDA: q [B, H, S, d], k / v [B, Hkv, S, d] (the rows' own tokens), kp / vp [B, Hkv, Lk, d]
    (the roots' key / value buffers, row r's root in its first tree.used[r] keys) -> [B, S, H, d]. The root part reads
    the buffers with no mask and no repeated heads, the branch part only each block's ancestors and own tokens; a
    padding token sees the root only, as in the dense mask."""
    B, H, S, d = q.shape
    Hkv, Lk = kp.shape[1], kp.shape[2]
    qf = q.transpose(1, 2).reshape(B * S, H, d)
    o_r, l_r = _varlen(qf, kp.transpose(1, 2).reshape(B * Lk, Hkv, d), vp.transpose(1, 2).reshape(B * Lk, Hkv, d),
                       tree.cu_qr, tree.cu_kr, S, Lk, scale, used=tree.used)
    kf, vf = (x.transpose(1, 2).reshape(B * S, Hkv, d) for x in (k, v))
    o_b, l_b = _varlen(qf if tree.full else qf[tree.qi], kf[tree.ki], vf[tree.ki], tree.cu_q, tree.cu_k, tree.max_q,
                       tree.max_k, scale, causal=True)
    if not tree.full:
        o_b = qf.new_zeros(B * S, H, d).index_copy_(0, tree.qi, o_b)
        l_b = torch.full((B * S, H), float("-inf"), device=q.device).index_copy_(0, tree.qi, l_b)
    return (_merge_fused() if tree.fused else _merge_attention)(o_r, l_r, o_b, l_b).view(B, S, H, d)


@functools.cache
def _merge_fused():
    """_merge_attention compiled for _cascade_varlen (dynamic shapes; a GraphBackend that compiles, with Triton): one
    kernel per layer instead of ~15 small ones (log-sum-exp, the two weights, casts, products, guards, sum, cast back),
    which the uncompiled pass 2 launched one by one: on a fast GPU the GPU then waits for the host, and basal-2.0-mini's
    40-question requests ran 0.90x as fast as the dense pass on an H100 (#39). The same operations in the same order in
    fp32 (Inductor's exp is libdevice's, as ATen's). GraphBackend._warm_cascade compiles it at start-up."""
    return torch.compile(_merge_attention, dynamic=True, fullgraph=True)


def causal_sdpa(model):
    """Attention through torch SDPA under a name transformers has no mask builder for: a forward without a mask runs
    with SDPA's is_causal (the flash kernel), a prepared 4D mask is used as given. Under torch.compile transformers
    would otherwise materialise the causal mask of every plain forward (it never skips it while tracing), which forces
    the slower masked kernel: 1.6x the time of a plain 8k-token forward on an H100."""
    from transformers import AttentionInterface
    AttentionInterface.register("basal_sdpa", basal_attention)
    model.config._attn_implementation = "basal_sdpa"
    return model


class StateCache:
    """Keys and values of request states (the root of a unit cut at the state boundary: template and state), keyed by a
    hash of their token ids, kept on the GPU within a byte budget with least-recently-used eviction. A request whose
    state is cached runs only its questions and options (pass 2 over the cached keys and values). An entry stored with
    its token ids is also found by a state that only begins with it (a growing agent trace or chat history, or an edited
    one): longest() finds the cached state sharing the longest prefix, through chained hashes of 256-token blocks."""

    BLOCK = 256  # tokens per block of the prefix index

    def __init__(self, budget_bytes, salt=b""):
        from collections import OrderedDict
        self.budget, self.used, self.salt = int(budget_bytes), 0, salt
        self.entries, self.lock = OrderedDict(), threading.Lock()
        self.index = {}  # chained hash of a state's first i blocks -> the key of an entry beginning with them
        self.hits = self.misses = self.evictions = self.prefix_hits = 0

    def _blocks(self, tokens):
        """Chained hashes of the full BLOCK-token blocks of tokens: the i-th covers tokens[: BLOCK * (i + 1)]."""
        import hashlib
        a, h, out = np.asarray(tokens, dtype=np.int64), b"", []
        for i in range(len(a) // self.BLOCK):
            h = hashlib.blake2b(h + a[i * self.BLOCK: (i + 1) * self.BLOCK].tobytes(), digest_size=16,
                                key=self.salt).digest()
            out.append(h)
        return out

    def longest(self, tokens):
        """The cached state sharing the longest prefix with tokens -> (its key, its keys / values, the length of the
        common prefix, its length), or None (no common full block). Found by the longest common block, then extended
        token by token: a state that extends a cached one shares all of it, an edited one up to the edit."""
        blocks = self._blocks(tokens)
        with self.lock:
            for i in range(len(blocks) - 1, -1, -1):
                key = self.index.get(blocks[i])
                if key in self.entries and self.entries[key][2] is not None:
                    break
            else:
                return None
            kv, _, own = self.entries[key]
            t, n = np.asarray(tokens, dtype=np.int64), min(len(own), len(tokens))
            m = (i + 1) * self.BLOCK
            diff = np.nonzero(own[m:n] != t[m:n])[0]
            m = m + int(diff[0]) if len(diff) else n
            self.entries.move_to_end(key)
            self.prefix_hits += 1
            return key, kv, m, len(own)

    def _drop(self, key):
        """Removes an entry and its blocks from the index (lock held)."""
        _, size, own = self.entries.pop(key)
        self.used -= size
        if own is not None:
            for h in self._blocks(own):
                if self.index.get(h) == key:
                    del self.index[h]

    def key(self, tokens):
        import hashlib
        return hashlib.blake2b(np.asarray(tokens, dtype=np.int64).tobytes(), digest_size=16, key=self.salt).digest()

    def get(self, key):
        with self.lock:
            e = self.entries.get(key)
            if e is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return e[0]

    def has(self, key):
        """Whether a state is cached (no recency update, not counted as a hit or a miss)."""
        with self.lock:
            return key in self.entries

    def put(self, key, kv, tokens=None, replaces=None):
        """Stores a state's keys / values; with its token ids (tokens) it is also found by states that begin with it.
        replaces: the key of an entry this one extends to its end (an agent loop keeps one entry, not one per step)."""
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
        if size > self.budget:
            return
        with self.lock:
            if replaces in self.entries:
                self._drop(replaces)
            if key in self.entries:
                return
            own = None if tokens is None else np.asarray(tokens, dtype=np.int64)
            self.entries[key] = (kv, size, own)
            self.used += size
            if own is not None:
                for h in self._blocks(own):
                    self.index[h] = key
            while self.used > self.budget:
                self._drop(next(iter(self.entries)))
                self.evictions += 1

    def stats(self):
        with self.lock:
            return dict(entries=len(self.entries), used_bytes=self.used, budget_bytes=self.budget, hits=self.hits,
                        misses=self.misses, evictions=self.evictions, prefix_hits=self.prefix_hits)


class GraphBackend(EagerBackend):
    """Fast path. Inputs are padded on the RIGHT into a small set of static shapes, one CUDA graph per shape, all graphs
    in one memory pool, captured at start-up (lazy capture under load would stall the server). Every group of prompts
    (all option orders of a question, or every branch of a request: SOAM) becomes the compressed prefix trie of its
    prompts ("unit"), and the units of all groups of a call -- of all concurrent requests -- are batched together,
    sorted by length, into forwards of at most TOKEN_BUDGET padded tokens:

    - a unit that is a single prompt runs in a plain graph: causal attention without a mask (the flash kernel);
      padding after the last real token cannot influence it, so the hidden state is read at the last real position.
      Longer than the largest graph: the same forward without a graph.
    - a unit with several branches runs in two passes. Pass 1 is the same plain graph over the unit's root (the state
      and whatever all its prompts share), which also writes every layer's keys and values into a shared buffer.
      Pass 2 runs every other block of the trie as one row per unit over those keys and values: positions continue
      from the root, and a block mask lets each branch see the root, its ancestors and itself only. Exactly equivalent
      to separate forwards, the state (usually most of the tokens) is computed once, and the long part runs with the
      flash kernel; pass 2 is short and replays a graph of the normal forward (up to 8 rows x 64 / 128 / 256 branch
      tokens over roots up to the largest of LENS), or runs as a normal forward for other shapes. Llama on CUDA, where
      such a forward's rows hold CASCADE_TOKENS or more tokens (compress, many questions) and torch's varlen attention
      runs (sm80+, bf16 / fp16, seqused_k; _cascade_why): a cascade instead of the block mask, the roots' keys and
      values and each block's ancestors in two varlen calls merged by log-sum-exp (_cascade_varlen).

    All forwards of a call are launched back to back; host buffers are pinned and copied asynchronously, and each
    forward's readout is copied back behind it, so a group is reported as soon as its forwards are done while later
    ones still run (run_shared's `done` callback; the server answers the request then)."""

    LENS = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
            2304, 2560, 2816, 3072]
    PLAIN_LENS = LENS + [3584, 4096, 5120, 6144, 8192]
    BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]  # rows per graph
    TOKEN_BUDGET = 12288  # max rows x bucket length per forward
    SUFFIX = 32                   # pass-2 rows without a graph are padded to a multiple of this
    BRANCH_TOKENS = 8192          # most tokens below the root of one packed row (more: split, the root once per part)
    # Llama on CUDA: a pass 2 without a graph whose longest row has at least this many tokens runs as a cascade
    # (_cascades). Below it the dense masked call is faster: with the cascade 5-10 questions over 1.5k tokens took
    # 0.83-0.98x, 40 questions 1.07-1.11x, compress requests 1.15-1.39x (basal-2.0-mini / 4.5B, RTX 4090 and A100,
    # 22 Oct)
    CASCADE_TOKENS = 1024
    SUFFIX_LENS = [64, 128, 256]  # pass-2 branch lengths with graphs, for roots up to the largest of LENS
    TREE_ROWS = [1, 2, 4, 8]      # pass-2 graph rows: branched units per forward with such roots
    # --batches fine: more row counts for the plain graphs (less padding of short batches, more graphs at start-up)
    FINE_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    # --prefix-cache: the template prefix of a short request (chat template, system prompt and the user header up to the
    # state) comes from the state cache, and the rest of the request runs as pass 2 over it. Its pass-2 graphs: roots up
    # to PREFIX_ROOT tokens, rows below the root up to PREFIX_SFX[-1], up to PREFIX_ROWS[-1] units per forward.
    PREFIX_ROWS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    PREFIX_SFX = [64, 96, 128, 160, 192, 256]
    PREFIX_ROOT = 256
    PREFIX_GB = 0.25              # state-cache budget when only the template prefixes are kept
    # --batches compact (FlashAttention 3): single prompts packed back to back into one row with varlen attention, no
    # padding of rows to LENS and of row counts to BATCHES; one graph per total token count of COMPACT_LENS (fine
    # steps: a prompt alone pads to the next multiple of 16 below 512 tokens), up to COMPACT_SEQS prompts per forward
    COMPACT_LENS = (list(range(16, 512, 16)) + list(range(512, 1024, 32)) + list(range(1024, 2048, 64))
                    + list(range(2048, 4096, 128)) + list(range(4096, 8192, 256)) + list(range(8192, 12289, 512)))
    COMPACT_SEQS = 32
    compact = False
    compact_kernel = None  # with compact rows: "fa3" or "torch varlen" (_varlen_attention's impl, the start-up line)
    readout_layer = False  # --readout-layer on (with compact rows): the last layer's o_proj / MLP at the readouts only
    template = ()          # --template-prefix on (with compact rows): template prefixes computed at start-up
    prefix = False
    lean, flex = None, False      # the lean layer (basal/lean.py) and its FlexAttention pass 2, when enabled
    state_cache = None            # StateCache: keys / values of request states, reused across requests
    gemma = None                  # basal/gemma4_graph.py: the forward of a Gemma 4 model (basal-omni)
    softcap = None                # the model's final logit softcap (Gemma), applied to the fp32 code logits
    _out = None                   # Llama graphs: the output buffers their captures share (_shared_out)
    _side = None                  # the stream the captures warm up on (one: each new stream pins a cuBLAS workspace)
    _deferred = None              # warm(): graphs warmed up, captured the largest first (_capture_deferred)
    fuse_merge = False            # the cascade's merge compiled (_merge_fused): with torch.compile on, Llama, Triton

    def __new__(cls, model_dir=None, *args, **kwargs):
        """GraphBackend(model_dir, ...) of a Qwen3.5 model (Gated DeltaNet and attention layers, from its config.json)
        builds GDNGraphBackend (basal/gdn_graph.py, basal 2.4), which carries the recurrent state through both passes;
        the subclasses (fast-exit, mps, gguf) keep their own class."""
        if cls is GraphBackend and model_dir is not None:
            from .gdn_graph import GDNGraphBackend, is_gdn
            if is_gdn(model_dir):
                return super().__new__(GDNGraphBackend)
        return super().__new__(cls)

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, shared=True, warm=True, attn="auto",
                 merge="auto", lean=False, tree="dense", state_cache_gb=0, prefix=False, batches="default",
                 readout_layer="off", template="off", gemma_attn="auto"):
        what = recurrent_layers(model_dir)  # before anything is loaded
        if what:
            from .gdn_graph import is_gdn
            raise ValueError(recurrent_refused(model_dir, what, is_gdn(model_dir)))
        if quant == "fp8" and fp8_refused():
            raise ValueError(fp8_refused())
        torch.backends.cuda.enable_cudnn_sdp(False)  # before capture: a captured graph keeps its attention kernel
        from . import gemma4_graph
        # every captured graph keeps its own cuBLAS workspace (32 MB on an H100): basal-omni-1.1-max's ~400 graphs held
        # 37 GB of graph pool and ran out of memory in warm-up; 8 MB workspaces made no difference to its forwards
        # (1.1-max, H100, 20 Oct). basal-2.0 (2.3.1): the same speed, 0.6-0.9 GB less GPU memory, 4.5B's start-up
        # 452 s instead of 554 s, dev v2 top answers as two runs differ (H100, 20 Oct). Read when the first cuBLAS
        # handle is made, so set before any GEMM; a value the user set is kept, and --deterministic has set :4096:8 before
        # (deterministic cuBLAS needs it)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")
        self.fp8_checkpoint = modelopt_fp8(model_dir)  # the -FP8 repositories: FP8 weights with static scales
        # likewise: the kernel of the causal forwards (--attn; auto keeps FlashAttention 2 for FP8 GEMMs)
        self.attn = set_attention(attn, fp8=self.fp8_checkpoint or quant == "fp8")
        if batches == "compact" and self.attn != "fa3":  # refused before the weights are loaded
            why = ("Gemma 4 needs --gemma-attn fa3" if gemma4_graph.is_gemma4_dir(model_dir)
                   else torch_varlen_available(getattr(torch, dtype)))
            if why:
                raise ValueError(compact_refused(why))
        if gemma_attn in ("flex", "fa3") and attention_available("flex") and gemma4_graph.is_gemma4_dir(model_dir):
            raise ValueError(f"--gemma-attn {gemma_attn}: {attention_available('flex')}")  # fa3: flex in global layers
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        if self.fp8_checkpoint:
            if quant not in (None, "fp8"):
                raise ValueError(f"{model_dir} is an FP8 checkpoint; serve it without --quant {quant}")
            self.model, quant = load_modelopt_fp8(model_dir, "cuda", getattr(torch, dtype)), None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).cuda().eval()
        self.dev = next(self.model.parameters()).device
        self.prefill = PREFILL
        if gemma4_graph.is_gemma4(self.model):
            self._gemma_init(gemma_attn, quant, compile, merge)
        else:
            causal_sdpa(self.model)
            self._llama_init(merge, quant, lean, tree, compile)
        self.shared = shared
        self._wrows = {}
        if batches == "fine":
            self.BATCHES = self.FINE_BATCHES
        # compact rows need a varlen kernel: FlashAttention 3's (Hopper; auto: compact where it runs; Gemma 4 with its
        # fa3 kernels: FA3 varlen in the sliding layers, a flex block mask per prompt in the 512-wide global ones), or,
        # with --batches compact on other GPUs, torch's varlen_attn (its FlashAttention 2, sm80+; Llama models)
        fa3 = self.attn == "fa3" and (self.gemma is None or self.gemma.kernels.get("sliding_attention") == "fa3"
                                      and self.gemma.kernels.get("full_attention") != "math")
        why = torch_varlen_available(getattr(torch, dtype)) if self.gemma is None else "Gemma 4 needs --gemma-attn fa3"
        torch_varlen = not fa3 and batches == "compact" and why is None
        self.compact = batches in ("compact", "auto") and (fa3 or torch_varlen) and self.dev.type == "cuda"
        self.compact_kernel = ("torch varlen" if torch_varlen else "fa3") if self.compact else None  # per backend
        if batches == "compact" and not self.compact:
            raise ValueError(compact_refused(why))
        if self.compact and self.gemma is not None:  # its forward reads out at one row per prompt
            self.fwd_c, self.readout_layer = self.gemma.forward_compact, True
        elif self.compact:
            self.fwd_c = torch.compile(self._forward_compact, dynamic=True) if compile else self._forward_compact
            last = self.model.model.layers[-1]
            self.readout_layer = readout_layer in ("on", "auto") and type(last).__name__ == "LlamaDecoderLayer" \
                and type(last.self_attn).__name__ == "LlamaAttention" and type(last.mlp).__name__ == "LlamaMLP"
        self.prefix = bool(prefix) and shared
        gb = max(state_cache_gb, self.PREFIX_GB if self.prefix else 0)
        self.state_cache = StateCache(gb * 2 ** 30) if gb and shared else None
        self._alloc_kv()
        self.template = []  # --template-prefix (with compact rows): [(start in template_kv, prefix token ids)]
        if self.compact and template in ("on", "auto") and self.gemma is None:  # (Gemma 4's template: 7 tokens)
            self._template_prefixes(model_dir)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        if warm:
            self.warm()
        self.one_forward = self._one_forward_tokens()

    def _gemma_init(self, kernels, quant, compile, merge):
        """basal-omni (Gemma 4): the forward of basal/gemma4_graph.py under the same packing, graphs and caches. Its
        own layer code reads the code rows at the readout positions (`ro`), so its plain graphs take them as an
        input; pass 2 with flex takes the trie as a TreeMask.

        auto (--gemma-attn, --merge): the configuration gated on an H100 (E3, 20 Oct: dev v2 against eager bf16,
        mini 0.9933 / 1.1-max 0.9891 of top answers equal, 0.9990-0.9998 at margin >= 0.2; automated at 1% within
        -0.11 points, each engine its own calibration; 1.1-max 3.4x eager's decisions/s, mini 6.1x): fa3 kernels,
        merged projections and (--batches auto) compact rows where FlashAttention 3 runs and the engine compiles; flex
        and merged projections on Blackwell workstation GPUs (gemma_fast_kernels); else the math kernel, uncompiled
        (E2's gate)."""
        from . import gemma4_graph
        why = gemma4_graph.supports(self.model)
        if why:
            raise ValueError(why)
        for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):  # text only here (31B: ~1.7 GB)
            if getattr(self.model.model, name, None) is not None:
                setattr(self.model.model, name, None)
        torch.cuda.empty_cache()
        if quant:  # NVFP4: its vLLM ports failed (15 Oct); torchao FP8 on 1.1-max failed the port gate (2-10 options at
            # margin >= 0.2: 0.9892, bar 0.9893; 20 Oct). The -FP8 checkpoints run in --mode fast / fp8 as stored
            raise ValueError(f"--mode {quant} quantisation at load time is not available for basal-omni (Gemma 4); "
                             "serve its -FP8 checkpoint instead")
        if kernels == "auto":
            kernels = (gemma_fast_kernels() if compile else None) or "math"
        if merge == "auto":
            merge = kernels != "math"
        if "fa3" in str(kernels) and attention_available("fa3"):
            raise ValueError(f"--gemma-attn {kernels}: {attention_available('fa3')}")
        # the math kernel runs uncompiled in --mode fast too: the configuration its gate measured (E2, 20 Oct:
        # agreement with eager 0.998-0.999 on dev v2, mini and 1.1-max); the compiled forward comes with the flex /
        # fa3 kernels
        compile = compile and kernels != "math"
        # merged projections: bf16 linears only (an FP8 checkpoint's FP8Linear layers keep theirs)
        self.gemma = gemma4_graph.Gemma4Graph(self.model, kernels, compile=compile, merge=merge is True)
        self.softcap, self.merged, self.lean = self.gemma.softcap, self.gemma.merged, None
        self.fwd, self.fwd_sfx = self.gemma.forward, self.gemma.forward_suffix
        self.flex = all(k != "math" for k in self.gemma.kernels.values())

    def _llama_init(self, merge, quant, lean, tree, compile):
        # one q / k / v and one gate / up GEMM per layer (before torchao, which quantises the merged linears as such).
        # auto: on for bf16 / fp32 weights (gated: same bf16 identity, dev v2 accuracy +0.01 points); off for FP8
        # checkpoints and torchao quantisation until their gate decides
        if merge == "auto":
            merge = not self.fp8_checkpoint and not quant
        self.merged = merge_projections(self.model) if merge else 0
        if quant:
            quantize(self.model, quant)
        self.fwd, self.fwd_sfx = self._forward, self._forward_suffix
        self.lean = None
        if lean:  # one compiled layer function for every layer and shape (basal/lean.py): both passes, or pass 2 only
            from . import lean as lean_module
            if lean_module.supports(self.model):
                self.lean = lean_module.Lean(self.model, compile=compile)
                self.fwd_sfx = self.lean.forward_suffix
                if lean != "pass2":
                    self.fwd = self.lean.forward
        # pass 2 with the lean layer: the trie as a FlexAttention mask (no dense mask, no repeated KV heads)
        self.flex = self.lean is not None and tree == "flex"
        import importlib.util
        self.fuse_merge = bool(compile) and importlib.util.find_spec("triton") is not None
        if compile and self.fwd == self._forward:  # fuse normalisation / rotary / activation kernels, capture graphs
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
            # pass 2 stays uncompiled: compiled graphs for its (rows, branch length, root length) grid took minutes
            # to capture; its graphs replay the normal forward
            self.fwd = torch.compile(self._forward, dynamic=True)

    @property
    def _dtype(self):
        """The dtype of the model body (its token embeddings)."""
        m = self.gemma.lm if self.gemma is not None else self.model.model
        return m.embed_tokens.weight.dtype

    def _alloc_kv(self):
        """The keys / values buffer that pass 1 writes and pass 2 reads: per layer, room for the largest plain graph
        and for the largest pass-2 graph's rows over the longest root. kv_shapes: (kv heads, head size) per layer
        with its own keys and values (Gemma 4: per layer type; its KV-shared layers read those of an earlier layer)."""
        if self.gemma is not None:
            self.kv_shapes = list(self.gemma.kv_shapes)
            self._fit_grid()
        else:
            cfg = self.model.config
            H = cfg.num_key_value_heads
            D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            self.kv_shapes = [(H, D)] * len(self.model.model.layers)
        self.kv_shape = self.kv_shapes[0]
        n = self._kv_tokens()
        dt = self._dtype
        # zeros, not empty: below a cached root the masked tail of its rows is never written, and a masked NaN still
        # poisons the readout (0 x NaN)
        self._kv = [tuple(torch.zeros(n * H * D, dtype=dt, device=self.dev) for _ in "kv") for H, D in self.kv_shapes] \
            if self.shared else None

    PASS2_POOL = 1 << 30  # Gemma 4: the most the pass-2 graphs over roots longer than LENS may add to the GPU memory
    # pass-2 graphs are captured only while at least this much GPU memory stays free (_capture_deferred): the forwards
    # without a graph (states beyond the grid, pass 2 of other shapes) need room at run time
    PASS2_RESERVE = 3 << 29  # 1.5 GiB
    KV_SHARE = 0.15  # Gemma 4: the most of the free GPU memory (after the weights) its key / value buffers may take

    def _fit_grid(self):
        """Gemma 4: every layer with its own keys and values keeps them for every token (31B: 0.9 MB per token in
        bf16, 12B: 0.34 MB), so buffers for the largest graph shapes could leave the activations no room next to the
        weights. Its pass-2 graphs are captured only where pass 1 can fill them (rows x root length within
        TOKEN_BUDGET; warm), so the buffers hold max(TOKEN_BUDGET, PLAIN_LENS[-1]) tokens, and both shrink until they
        take at most KV_SHARE of the free memory; longer units run without a graph, as beyond PLAIN_LENS."""
        if self.dev.type != "cuda" or not self.shared:
            return
        per = sum(2 * H * D for H, D in self.kv_shapes) * self._dtype.itemsize
        cap = int(self.KV_SHARE * torch.cuda.mem_get_info(self.dev)[0] / per)
        if max(self.TOKEN_BUDGET, self.PLAIN_LENS[-1]) > cap:  # 31B on an 80 GB GPU (a quarter of free memory
            # left 76 GB allocated and ran out in warm-up: graph pools and activations need the rest, 20 Oct)
            self.TOKEN_BUDGET = min(self.TOKEN_BUDGET, max(cap // 512 * 512, 1024))
            self.PLAIN_LENS = [x for x in self.PLAIN_LENS if x <= max(cap, self.PLAIN_LENS[0])]
            self.LENS = [x for x in self.LENS if x <= self.PLAIN_LENS[-1]]
            # compact rows hold no keys / values, but their activations and graph pool grow with the row: 31B with
            # compact rows of 12,288 tokens ran out of memory in warm-up (12.5 GB of graph pool, 20 Oct)
            self.COMPACT_LENS = [x for x in self.COMPACT_LENS if x <= self.TOKEN_BUDGET]

    def _kv_tokens(self):
        """Tokens of the key / value buffer: the largest plain graph and the largest captured pass-2 graph's rows over
        its root (_pass2_ok; Gemma 4: within TOKEN_BUDGET, see _fit_grid)."""
        if self.gemma is not None:
            return max(self.TOKEN_BUDGET, self.PLAIN_LENS[-1])
        n = [self.TOKEN_BUDGET, self.PLAIN_LENS[-1]]
        n += [b * L for b in self.TREE_ROWS for L in self.LENS if self._pass2_ok(b, L)]
        if self.prefix:
            n += [b * L for L in self.LENS if L <= self.PREFIX_ROOT for S in self.PREFIX_SFX for b in self.PREFIX_ROWS
                  if b * S <= self.TOKEN_BUDGET]
        return max(n)

    def _pass2_ok(self, b, L):
        """Whether pass 1 can produce a batch for the pass-2 graph of b rows over roots of length L: that graph (b =
        TREE_ROWS[i]) serves TREE_ROWS[i - 1] + 1 to b units, whose roots pass 1 runs padded to BATCHES rows x L
        within TOKEN_BUDGET (_chunks; units with cached roots are chunked the same way). Llama: 12 of the 276 default
        pass-2 graphs (8 rows over roots of 2,304-3,072 tokens) could never run, and the key / value buffer held
        24,576 tokens for them instead of 16,384. Gemma 4: rows x root within TOKEN_BUDGET (_fit_grid)."""
        if self.gemma is not None:
            return b * L <= self.TOKEN_BUDGET
        prev = ([0] + self.TREE_ROWS)[self.TREE_ROWS.index(b)]
        return self._bucket(prev + 1, self.BATCHES) * L <= self.TOKEN_BUDGET

    def _shared_out(self, name, n, dtype):
        """A flat buffer of at least n elements shared by the captures of one graph family (Llama): a captured graph's
        output stays allocated in the graph pool for as long as the graph lives, so 169 plain graphs kept 169 hidden
        state tensors [b, L, D] (and 276 pass-2 graphs their outputs and masks) although every forward's readout is
        copied out right after its replay, under _launch, on the same stream. RTX 4090, 21 Oct, with the one warm-up
        stream and _pass2_ok: basal-2.0-mini 12.7 -> 10.0 GB after warm-up, basal-2.0-4.5B 24.1 -> 20.5 GB, the
        probabilities bit-identical (420 of 420 readouts each), the same decisions per second."""
        if self._out is None:
            self._out = {}
        buf = self._out.get(name)
        if buf is None or buf.numel() < n or buf.dtype != dtype:
            buf = self._out[name] = torch.zeros(n, dtype=dtype, device=self.dev)
        return buf

    def _kv_new(self, L):
        """Fresh per-layer (K, V) buffers [1, kv heads, L, head size] (a root longer than the largest graph). Gemma 4:
        a sliding layer's last `window` tokens only (kv_lens: pass 2 reads no further back), so a long state takes a
        fifth of the memory on 31B (0.9 MB per token: its 10k-token states ran out of memory on an 80 GB H100 next to
        59 GB of weights, 20 Oct); the state cache keeps the same."""
        dt = self._dtype
        lens = self.gemma.kv_lens(L) if self.gemma is not None else [L] * len(self.kv_shapes)
        return [(torch.empty((1, H, n, D), dtype=dt, device=self.dev), torch.empty((1, H, n, D), dtype=dt,
                                                                                  device=self.dev))
                for (H, D), n in zip(self.kv_shapes, lens)]

    @torch.no_grad()
    def _template_prefixes(self, model_dir):
        """--template-prefix: the tokens every prompt of a language starts with (chat template, system prompt and the
        user header up to the state; one token less, which can merge with the state's first) and their keys / values
        in every layer, computed once: template_kv[layer] = (K, V) [prefix tokens of all languages, kv heads, d]. A
        compact prompt that starts with one runs only its own tokens over them."""
        from .prompt import FORMATS, model_format, render
        fmt = model_format(model_dir)
        H, D = self.kv_shape
        dt = self.model.model.embed_tokens.weight.dtype
        kvs, a = [], 0
        for lang, t in FORMATS[fmt]["templates"].items():
            head = t["user"].split("{state}")[0]
            state = "Stan klienta: zgłoszenie" if lang == "pl" else "Customer state: a request"
            text = render(self.tok, state, "Q?", ["a", "b"], lang, fmt)
            i = text.find(head + state)
            if i < 0:
                continue
            ids = self.tok(text[: i + len(head)], add_special_tokens=False).input_ids
            full = self.tok(text, add_special_tokens=False).input_ids
            n = len(ids) - 1
            while n and full[:n] != ids[:n]:
                n -= 1
            if n < 2:
                continue
            kv = [(torch.empty((1, H, n, D), dtype=dt, device=self.dev), torch.empty((1, H, n, D), dtype=dt,
                                                                                     device=self.dev))
                  for _ in self.model.model.layers]
            self._forward(self._to_dev(torch.tensor([full[:n]])), kv)
            kvs.append(kv)
            self.template.append((a, tuple(full[:n])))
            a += n
        if not self.template:
            return
        self.template.sort(key=lambda x: -len(x[1]))  # the longest match first
        self.template_tokens, self.template_max = a, max(len(x[1]) for x in self.template)
        self.template_kv = [tuple(torch.cat([kv[i][j][0].transpose(0, 1) for kv in kvs]).contiguous() for j in (0, 1))
                            for i in range(len(self.model.model.layers))]

    def _kv_views(self, b, L):
        """Per-layer (K, V) views [b, kv heads, L, head dim] at the start of the buffer (one layout for every graph,
        so that the compiled forward is compiled once)."""
        if self._kv is None:
            return None
        return [(k[: b * H * L * D].view(b, H, L, D), v[: b * H * L * D].view(b, H, L, D))
                for (k, v), (H, D) in zip(self._kv, self.kv_shapes)]

    def grid(self, lens):
        """(rows, length) shapes of one graph family."""
        return [(b, L) for L in lens for b in self.BATCHES if b * L <= self.TOKEN_BUDGET or b == 1]

    def warm(self):
        """Capture every plain graph of the grid, and with `shared` the pass-2 graphs for roots up to the largest of
        LENS (Gemma 4: of PLAIN_LENS, _warm_long_sfx). Every graph is warmed up here in this order and captured
        afterwards (_capture_deferred): the plain and compact graphs the largest first, then the pass-2 graphs the
        smallest first while PASS2_RESERVE stays free; Gemma 4's pass-2 graphs over the longer roots after those, one at
        a time, as _warm_long_sfx keeps the pool growth they add within PASS2_POOL and the reserve free."""
        self._deferred = {}  # key -> (function, static inputs), in warm-up order
        if self.compact:  # plain graphs only for pass 1 of branched units (at most TREE_ROWS[-1] roots)
            for T in self.COMPACT_LENS:
                self._graph_c(T)
        for b, L in self.grid(self.PLAIN_LENS):
            if not self.compact or self.shared and b <= self.TREE_ROWS[-1]:
                self._graph(b, L)
        if self.shared:
            for b in self.TREE_ROWS:
                for L in self.LENS:
                    for S in self.SUFFIX_LENS:
                        if b * S <= self.TOKEN_BUDGET and self._pass2_ok(b, L):
                            self._graph_sfx(b, S, L)
            if self.prefix:  # pass 2 of short requests over a cached template prefix
                for L in (x for x in self.LENS if x <= self.PREFIX_ROOT):
                    for S in self.PREFIX_SFX:
                        for b in self.PREFIX_ROWS:
                            if b * S <= self.TOKEN_BUDGET:
                                self._graph_sfx(b, S, L)
            if self.gemma is not None:
                self._capture_deferred()
                self._warm_long_sfx()
        if self._deferred is not None:
            self._capture_deferred()
        # beyond the largest graph the forwards run uncompiled; their first use sets up kernels and memory for long
        # shapes (3 s of the first long request on an H100), so a long prompt and a long branched unit run here once
        n = self.PLAIN_LENS[-1] + 64
        cfg = self.model.config
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg  # Gemma 4: the decoder's settings
        if getattr(cfg, "max_position_embeddings", 0) >= n + 8:
            root = [self.tok.pad_token_id] * n
            self.run_shared([([root + [5]], [[10, 11]])] + ([([root + [5, 6], root + [6, 5]], [[10, 11], [10, 11]])]
                                                             if self.shared else []))
            if self.gemma is not None and self.flex and self.shared:
                self._warm_pass2(n)
        if self.shared and self.fuse_merge and self._cascade_why is None:
            self._warm_cascade()

    def _warm_cascade(self):
        """Compile the cascade's merge (_merge_fused) at start-up, not in the first request that runs a cascade: on tiny
        rows, without padding and with (the log-sum-exp of the branch part comes as the kernel returns it, or scattered
        into a full-length one), the two layouts a cascade passes."""
        H, (Hkv, d) = self.model.config.num_attention_heads, self.kv_shape
        for seg in ([[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 1, -1, -1]]):
            seg = np.array(seg)
            B, S = seg.shape
            q = torch.zeros((B, H, S, d), dtype=self._dtype, device=self.dev)
            k = torch.zeros((B, Hkv, S, d), dtype=self._dtype, device=self.dev)
            _cascade_varlen(q, k, k, k, k, _varlen_tree(seg, [[-1, -1]] * B, [S] * B, S, self._to_dev, True), 1.0)

    def _warm_long_sfx(self):
        """Gemma 4: pass-2 graphs over the roots of the plain graphs longer than LENS (states of 3k-8k tokens: the
        questions of a multi-question request over them ran kernel by kernel, mini's GPU idle 11-13% of the time under
        load, nsys 20 Oct), rows x root within TOKEN_BUDGET, while the graph pool grows by at most PASS2_POOL."""
        r0 = torch.cuda.memory_reserved(self.dev) if self.dev.type == "cuda" else 0
        for L in (x for x in self.PLAIN_LENS if x > self.LENS[-1]):
            for b in (x for x in self.TREE_ROWS if x * L <= self.TOKEN_BUDGET):
                for S in self.SUFFIX_LENS:
                    if (self.dev.type == "cuda" and torch.cuda.memory_reserved(self.dev) - r0 > self.PASS2_POOL
                            or not self._room()):
                        return
                    self._measured(self._graph_sfx, b, S, L)

    def _warm_pass2(self, n):
        """Gemma 4: pass 2 without a graph calls one compiled flex_attention, which compiles anew for each shape class
        it has not seen (one row or several, fresh key / value buffers or the graphs', readout rows or all, a second
        size of a dimension makes it dynamic): 1.8-2.4 s each, under the launch lock, the first time a class came up
        under load (nsys, 20 Oct: 4-5 such stalls in the first multi-question run on mini and 1.1-max). The classes
        run here once: branched units of five 64-token branches over a root beyond the largest graph (fresh buffers)
        and over a root of the plain graphs (their buffers), one and two units per forward."""
        pad = self.tok.pad_token_id

        def unit(L):  # five branches of 64 tokens below an L-token root
            return ([[pad] * L + [5 + k] + [7] * 63 for k in range(5)], [[10, 11]] * 5)
        L = self.LENS[-1] + 64  # a root of the plain graphs (none on ultra, whose grid ends at LENS: _fit_grid)
        if L <= self.PLAIN_LENS[-1] and 2 * self._bucket(L, self.PLAIN_LENS) <= self.TOKEN_BUDGET:
            self.run_shared([unit(L)])
            self.run_shared([unit(L), unit(L)])
        self.run_shared([unit(n)])

    # -- helpers -------------------------------------------------------------------------------------------------
    def _bucket(self, n, xs):
        for x in xs:
            if n <= x:
                return x
        raise ValueError(f"too long: {n} > {xs[-1]}")

    def _chunks(self, lens, grid_lens=None, extra=None, rows=None):
        """Indices sorted by length, cut into chunks with padded size <= TOKEN_BUDGET (short: big batches); an item
        longer than the largest length of grid_lens (default LENS) is a chunk of its own. extra: a second length per
        item (pass 2 of branched units) whose rows x longest must also stay within the budget; rows: most per chunk."""
        grid_lens = grid_lens or self.LENS
        rows = min(rows or self.BATCHES[-1], self.BATCHES[-1])
        order = sorted(range(len(lens)), key=lambda k: lens[k])
        out, cur = [], []
        for k in order:
            nxt = cur + [k]
            if cur and (len(nxt) > rows or lens[k] > grid_lens[-1] or
                        self._bucket(len(nxt), self.BATCHES) * self._bucket(lens[k], grid_lens) > self.TOKEN_BUDGET or
                        extra is not None and len(nxt) * max(extra[j] for j in nxt) > self.TOKEN_BUDGET):
                out.append(cur); nxt = [k]
            cur = nxt
        return out + ([cur] if cur else [])

    def _to_dev(self, x):
        """Host array / tensor -> device through pinned memory, asynchronously (a copy from pageable memory would wait
        for every forward already queued)."""
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        return x.pin_memory().to(self.dev, non_blocking=True) if self.dev.type == "cuda" else x.to(self.dev)

    @torch.no_grad()
    def _forward(self, ids, kv=None, ro=None):
        """Plain causal forward; kv: per-layer (K, V) buffers [rows, kv heads, length, head dim] it fills (pass 1).
        Gemma 4: uncompiled; ro [rows, R]: the readout positions -> [rows, R, D]."""
        if self.gemma is not None:
            return self.gemma.forward(ids, kv, ro, compiled=False)
        return self.model.model(input_ids=ids, use_cache=False, basal_kv=kv).last_hidden_state

    @torch.no_grad()
    def _forward_compact(self, ids, pos, cu, ro=None, cu_k=None, gather=None):
        """Compact rows: prompts packed back to back in one row [1, T] (positions from 0 per prompt, cu = cumulative
        token counts), each attending to itself only -> hidden states [1, T, D]. ro (--readout-layer on): the readout
        positions [R]; the last decoder layer then computes keys / values and attention for every position and the
        o_proj, MLP and final norm at the readout positions only -> [R, D]. cu_k / gather (--template-prefix on): the
        prompts continue the template prefix computed at start-up (see _varlen_attention)."""
        m = self.model.model
        if cu_k is not None:
            cu = (cu, cu_k, gather, self.template_kv)
        if ro is None:
            return m(input_ids=ids, position_ids=pos, use_cache=False, basal_cu=cu,
                     basal_varlen=self.compact_kernel).last_hidden_state
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        h = m.embed_tokens(ids)
        pe = m.rotary_emb(h, position_ids=pos)
        layers = m.layers[: m.config.num_hidden_layers]
        for layer in layers[:-1]:
            h = layer(h, attention_mask=None, position_embeddings=pe, position_ids=pos, past_key_values=None,
                      use_cache=False, basal_cu=cu, basal_varlen=self.compact_kernel)
        last, a = layers[-1], layers[-1].self_attn
        x = last.input_layernorm(h)
        if hasattr(a, "qkv_proj"):
            q, k, v = a.qkv_proj(x).split(a.qkv_sizes, -1)
        else:
            q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        shape = (*x.shape[:-1], -1, a.head_dim)
        q, k = apply_rotary_pos_emb(q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2), *pe)
        o = _varlen_attention(q, k, v.view(shape).transpose(1, 2), cu, a.scaling, a.layer_idx,
                              self.compact_kernel)[0, ro]  # [R, heads, d]
        h = h[0, ro] + a.o_proj(o.reshape(o.shape[0], -1))
        h = h + last.mlp(last.post_attention_layernorm(h))
        return m.norm(h)

    @torch.no_grad()
    def _forward_masked(self, ids, mask, pos):
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state

    @torch.no_grad()
    def _forward_suffix(self, ids, mask, pos, prefix, ro=None, kv=None):
        """Pass 2: rows that continue computed prefixes (per-layer (K, V) [rows, kv heads, prefix length, head dim]);
        mask [rows, 1, length, prefix length + length] (Gemma 4 with flex: a TreeMask). Gemma 4: uncompiled; ro as in
        _forward; kv: buffers that receive the rows' own keys / values (_continue)."""
        if self.gemma is not None:
            return self.gemma.forward_suffix(ids, mask, pos, prefix, ro, compiled=False, kv=kv)
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                                basal_prefix=prefix, basal_kv=kv).last_hidden_state

    @torch.no_grad()
    def _forward_cascade(self, ids, pos, prefix, tree):
        """Pass 2 as a cascade (_cascade_varlen): ids / pos [rows, length] host arrays, prefix: the roots' per-layer
        (K, V) [rows, kv heads, Lk, head dim], tree: _varlen_tree of the rows -> hidden states [rows, length, D]."""
        return self.model.model(input_ids=self._to_dev(ids), position_ids=self._to_dev(pos), use_cache=False,
                                basal_prefix=prefix, basal_tree=tree).last_hidden_state

    def _cascades(self, n):
        """Whether a pass 2 without a graph whose longest row has n tokens runs as a cascade (_cascade_varlen):
        CASCADE_TOKENS or more, where _cascade_why has no objection."""
        return n >= self.CASCADE_TOKENS and self._cascade_why is None

    @functools.cached_property
    def _cascade_why(self):
        """None if pass 2 can run as a cascade here, else why not: CUDA, a Llama-style model with attention layers
        only (attention_only; the CPU flash op that cascade_ok also asks for plays no part here), torch's varlen
        attention with grouped-query heads and per-sequence key counts in this dtype, the transformers forward (not
        --lean, which has its own pass 2)."""
        if self.dev.type != "cuda" or self.gemma is not None or self.lean is not None:
            return "CUDA graph engine, Llama, without --lean"
        if not attention_only(self.model.config):
            return "sliding-window or linear-attention layers"
        why = torch_varlen_available(self._dtype)
        if why is None:
            import inspect
            from torch.nn.attention.varlen import varlen_attn
            if "seqused_k" not in inspect.signature(varlen_attn).parameters:
                why = "this torch's varlen_attn has no seqused_k"
        return why

    one_forward = 0  # tokens of the longest state run as one forward (_one_forward_tokens; 0: every one in chunks)
    # of the free memory after warm-up a state beyond the largest graph may take in one forward (else in chunks; one
    # that runs out of memory anyway is redone in chunks): ultra on an 80 GB H100 has 8-12 GB free after warm-up, and
    # one forward of 32k tokens ran in one start-up and out of memory in another (20 Oct): at the edge, so chunked
    ONE_FORWARD_SHARE = 0.5

    def _one_forward_tokens(self):
        """The longest prompt or state beyond the largest graph that runs as one forward (2.3.0's path, the fastest);
        longer ones are prefilled in chunks (_continue). From the free memory after warm-up and an estimate of one
        forward's memory per token: the MLP's gate / up rows, a few hidden-size rows, and the keys / values a root keeps
        (Gemma 4: its global layers; sliding layers keep their window). Ultra: 32k tokens on an 80 GB H100."""
        if self.dev.type != "cuda":
            return 1 << 30
        cfg = self.gemma.cfg if self.gemma is not None else self.model.config
        b = self._dtype.itemsize
        kept = self.gemma.kv_lens(1 << 30) if self.gemma is not None else [1 << 30] * len(self.kv_shapes)
        kv = sum(2 * H * D * b for (H, D), t in zip(self.kv_shapes, kept) if t >= 1 << 30)
        per = (2 * cfg.intermediate_size + 4 * cfg.hidden_size) * b + kv
        n = int(self.ONE_FORWARD_SHARE * torch.cuda.mem_get_info(self.dev)[0] / per)
        print(f"basal: states beyond {n} tokens are prefilled in chunks of {self.CHUNK}", flush=True)
        return n

    CHUNK = 2048  # tokens per forward of a long state (and of a state's new part over a cached one): _continue

    @torch.no_grad()
    def _continue(self, tokens, prefix=None, m=0):
        """tokens[m:] at positions m.. over prefix (per own layer (K, V) [kv heads, length, head size] of tokens[:m];
        Gemma 4's sliding layers may hold its last tokens only) in forwards of at most CHUNK tokens, each over the keys
        and values before it, so the activations scale with the chunk, not the state (basal-omni-1.0-ultra's
        76k-token states ran out of memory as one forward on an 80 GB H100, 20 Oct). The keys / values go into
        buffers allocated once: the whole length for layers that keep every token, window + MARGIN + CHUNK for
        Gemma 4's sliding layers (no copy of the growing prefix per chunk) -> (per own layer (K, V) [kv heads,
        length, head size] of all of tokens, sliding layers their last window + MARGIN (kv_lens); the hidden state of
        the last token [D]). Uncompiled, as beyond the largest graph."""
        dt, L, h = self._dtype, len(tokens), None
        keep = self.gemma.kv_lens(L) if self.gemma is not None else [L] * len(self.kv_shapes)
        full = [t >= L for t in keep]
        bufs, cur = [], []  # per layer: (K, V) [1, kv heads, room, d]; tokens held (positions s - cur .. s - 1)
        for (H, D), t, f, pre in zip(self.kv_shapes, keep, full, prefix or [None] * len(keep)):
            room = L if f else min(L, t + self.CHUNK)
            kb = torch.empty((1, H, room, D), dtype=dt, device=self.dev)
            vb = torch.empty((1, H, room, D), dtype=dt, device=self.dev)
            c = 0
            if pre is not None:
                c = min(pre[0].shape[1], m if f else t)
                kb[0, :, :c].copy_(pre[0][:, pre[0].shape[1] - c:])
                vb[0, :, :c].copy_(pre[1][:, pre[1].shape[1] - c:])
            bufs.append((kb, vb))
            cur.append(c)
        for s in range(m, L, self.CHUNK):
            e = min(L, s + self.CHUNK)
            n = e - s
            ids = self._to_dev(torch.tensor([tokens[s:e]]))
            ro = self._to_dev(torch.tensor([[n - 1]])) if self.gemma is not None else None
            out_kv = [(kb[:, :, c: c + n], vb[:, :, c: c + n]) for (kb, vb), c in zip(bufs, cur)]
            if s == 0:  # from position 0: a plain causal forward
                out = self._forward(ids, out_kv, ro)
            else:  # the chunk as one block of a trie over the keys / values before it: positions continue from s
                pos = self._to_dev(torch.arange(s, e)[None])
                if self.flex and self.gemma is not None:
                    mask = self._tree_mask(np.zeros((1, n), dtype=np.int64), [[-1]], [s], n)
                else:
                    m2 = self._mask_from_seg(self._to_dev(torch.zeros((1, n), dtype=torch.long)), [[-1]])
                    mask = torch.cat([torch.zeros((1, 1, n, s), dtype=m2.dtype, device=self.dev), m2], -1)
                pre = [(kb[:, :, :c], vb[:, :, :c]) for (kb, vb), c in zip(bufs, cur)]
                out = self._forward_suffix(ids, mask, pos, pre, ro, kv=out_kv)
            h = out[0, -1]
            for i, ((kb, vb), c, t, f) in enumerate(zip(bufs, cur, keep, full)):
                c += n
                if not f and c > t:  # a sliding layer: its last t tokens to the front
                    kb[:, :, :t] = kb[:, :, c - t: c].clone()
                    vb[:, :, :t] = vb[:, :, c - t: c].clone()
                    c = t
                cur[i] = c
        if self.dev.type == "cuda" and L - m > self.CHUNK:  # the allocator's blocks of a long state back to the GPU
            torch.cuda.empty_cache()  # (after one 32k-token ultra request the next ones ran out of memory)
        # (sliding layers copied out of their room of window + MARGIN + CHUNK: what the state cache holds is what it counts)
        return [(kb[0, :, :c], vb[0, :, :c]) if f else (kb[0, :, :c].clone(), vb[0, :, :c].clone())
                for (kb, vb), c, f in zip(bufs, cur, full)], h

    def _usable(self, kv, m, n):
        """The common prefix m of a cached state of n tokens that its keys / values can continue (0: none): Gemma 4's
        sliding layers of a long state hold its last tokens only, which must reach a window back from m."""
        if self.gemma is None:
            return m
        for (k, _), t in zip(kv, self.gemma.kv_lens(n)):
            if t < n and n - k.shape[1] > max(0, m - self.gemma.window):
                return 0
        return m

    def _extend_states(self, units, keys, hits):
        """Incremental states: a unit whose state (root) is not cached but begins with a cached state's first m tokens
        (a growing agent trace or chat history, or an edited one up to the edit) computes only tokens m.. over that
        state's keys / values (_continue); the result is cached, replacing the old entry when it extends it to its
        end, and used as a hit."""
        if not keys or self.state_cache is None:
            return
        for k, key in enumerate(keys):
            if key is None or k in hits:
                continue
            pk = units[k][0]
            root = list(pk[0][: pk[2].index(1)])
            found = self.state_cache.longest(root)
            if found is None:
                continue
            old, kv, m, n = found
            if old == key:  # cached by an earlier unit of this call
                hits[k] = kv
                continue
            m = self._usable(kv, m, n)
            if m < self.state_cache.BLOCK or m >= len(root):
                continue
            prefix = []
            for (ck, cv), t in zip(kv, self.gemma.kv_lens(n) if self.gemma is not None else [n] * len(kv)):
                o = n - ck.shape[1]  # position of the entry's first stored token (sliding layers of a long state)
                prefix.append((ck[:, : m - o], cv[:, : m - o]))
            new, _ = self._continue(root, prefix, m)
            self.state_cache.put(key, new, tokens=root, replaces=old if m == n else None)
            hits[k] = new

    @torch.no_grad()
    def _forward_layered(self, ids, kv=None, between=None):
        """The plain causal forward (`_forward`, uncompiled) one decoder layer at a time, calling between(i) before
        layer i (the long lane gives the GPU to the main lane there): the transformers model's own loop, the same
        modules with the same arguments, so the same result. Gemma 4: its own forward, uncompiled."""
        if self.gemma is not None:
            return self.gemma.forward(ids, kv, between=between, compiled=False)
        from transformers.masking_utils import create_causal_mask
        m = self.model.model
        h = m.embed_tokens(ids)
        pos = torch.arange(h.shape[1], device=h.device)[None]
        mask = create_causal_mask(config=m.config, inputs_embeds=h, attention_mask=None, past_key_values=None,
                                  position_ids=pos)
        pe = m.rotary_emb(h, position_ids=pos)
        for i, layer in enumerate(m.layers[: m.config.num_hidden_layers]):
            if between is not None:
                between(i)
            h = layer(h, attention_mask=mask, position_embeddings=pe, position_ids=pos, past_key_values=None,
                      use_cache=False, basal_kv=kv)
        return m.norm(h)

    def _capture(self, key, fn, *static):
        """Warm fn up and capture it as graph `key`; in warm() the capture waits for _capture_deferred, and a key
        already waiting is not warmed up again (with --prefix-cache the two pass-2 families share some shapes)."""
        if self._deferred is not None:
            if key not in self._deferred:
                self._warm_up(fn, static)
                self._deferred[key] = (fn, static)
            return None
        self._warm_up(fn, static)
        return self._record(key, fn, static)

    def _warm_up(self, fn, static):
        """One call of fn on the side stream before its capture: what a capture cannot do happens at a forward's first
        call (torch.compile compiles, or recompiles for a shape its guards reject; cuBLAS sets up its handle and
        workspace; Triton loads its kernels). The second and third calls (plain graphs 3, pass-2 graphs 2 before) ran
        the same kernels again and only took start-up time: RTX 4060 Ti, 22 Oct, warm() 468 -> 323 s on basal-2.0-mini
        and 1,094 -> 680 s on basal-2.0-4.5B, the probabilities bit-identical (420 readouts each)."""
        if self._side is None:
            self._side = torch.cuda.Stream()
        st = self._side; st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fn(*static)
        torch.cuda.current_stream().wait_stream(st)

    def _record(self, key, fn, static):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn(*static)
        self.graphs[key] = (g, *static, out)
        return self.graphs[key]

    def _capture_deferred(self):
        """warm(): capture the graphs warmed up so far, the one with the most tokens first. Every graph's memory stays
        in the shared pool; what a graph frees at the end of its capture, a later capture reuses, but only in blocks no
        larger than the freed ones, so captured from the smallest up (the grid's order: compact graphs of 16, 32, ...
        12,288 tokens) each larger graph allocated new blocks and the pool grew to many times the largest graph's
        working set. torch.compile still sees the shapes in the grid's order (the warm-up calls in _capture), so the
        compiled kernels are those of capturing in that order. RTX 4090, basal-2.0-mini, --batches compact, 21 Oct:
        graph pool 9.0 -> 1.0 GiB, 17.6 -> 9.4 GB after warm-up, the probabilities bit-identical.

        The pass-2 graphs come last, the smallest (the commonest shapes) first, and only while PASS2_RESERVE of GPU
        memory stays free after the capture (_room: free memory less what the last capture took): each keeps 10-20 MB
        outside the pool for its kernels (basal-2.0-mini / 4.5B, RTX 4090), 264 of them about 5 GB on basal-2.0-4.5B,
        which therefore did not start on a 16 GB card. A pass 2 without its graph runs uncompiled, as one beyond the
        grid does (_tree_hidden); the plain and compact graphs, which a forward cannot do without, are all captured."""
        deferred, self._deferred = self._deferred, None
        keys = sorted(deferred, key=_graph_tokens, reverse=True)
        for key in (k for k in keys if k[0] != "x"):
            self._measured(self._record, key, *deferred[key])
        pass2 = [k for k in reversed(keys) if k[0] == "x"]
        if pass2 and self.dev.type == "cuda":
            torch.cuda.empty_cache()  # (blocks the warm-up calls left cached are no graph's)
        for n, key in enumerate(pass2):
            if not self._room():
                print(f"basal: {n} of {len(pass2)} pass-2 graphs captured, {self.PASS2_RESERVE >> 20} MiB of GPU "
                      "memory kept free; the other shapes run pass 2 without a graph", flush=True)
                break
            self._measured(self._record, key, *deferred[key])

    _capture_cost = 0  # the GPU memory the last measured capture took (_measured): the estimate for the next one

    def _room(self):
        """Whether one more pass-2 graph fits: the free GPU memory less what the last capture took stays at or above
        PASS2_RESERVE (always off CUDA)."""
        free = self._free_memory()
        return free is None or free - self._capture_cost >= self.PASS2_RESERVE

    def _measured(self, capture, *args):
        """capture(*args), keeping the GPU memory it took as _capture_cost."""
        before = self._free_memory()
        capture(*args)
        if before is not None:
            self._capture_cost = max(before - self._free_memory(), 0)

    def _free_memory(self):
        """Free GPU memory in bytes (None off CUDA: no budget)."""
        return torch.cuda.mem_get_info(self.dev)[0] if self.dev.type == "cuda" else None

    def _graph(self, b, L):
        """Plain graph (b rows, length L): (graph, ids, kv buffers or None, hidden states [b, L, D]); Gemma 4: (graph,
        ids, kv buffers or None, readout position of every row [b, 1], hidden states there [b, 1, D])."""
        key = ("p", b, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        if self.gemma is not None:
            ro = torch.zeros((b, 1), dtype=torch.long, device=self.dev)
            return self._capture(key, self.fwd, ids, self._kv_views(b, L), ro)
        out = self._shared_out("plain", max(b * L for b, L in self.grid(self.PLAIN_LENS)) * self._hidden_size,
                               self._dtype)[: b * L * self._hidden_size].view(b, L, -1)
        fwd = self.fwd

        def plain(i, kv):  # the output in the buffer every plain graph shares (_shared_out)
            return _copy_same(out, fwd(i, kv))
        return self._capture(key, plain, ids, self._kv_views(b, L))

    def _graph_c(self, T):
        """Compact graph of T tokens: (graph, ids, positions, cumulative counts, hidden states)."""
        key = ("c", T)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((1, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(T, device=self.dev)[None].contiguous()
        cu = torch.full((self.COMPACT_SEQS + 2,), T, dtype=torch.int32, device=self.dev)
        cu[0] = 0
        # readout positions: one per prompt (duplicates share it), the rest point at 0
        ro = torch.zeros(self.COMPACT_SEQS, dtype=torch.long, device=self.dev) if self.readout_layer else None
        if self.template:  # key counts and the gather of prefix + own keys / values (no prompt continues a prefix)
            cu_k = cu.clone()
            gather = torch.zeros(T + self.COMPACT_SEQS * self.template_max, dtype=torch.long, device=self.dev)
            gather[:T] = self.template_tokens + torch.arange(T, device=self.dev)
            return self._capture(key, self._compact_out(T, ro), ids, pos, cu, ro, cu_k, gather)
        return self._capture(key, self._compact_out(T, ro), ids, pos, cu, *([ro] if ro is not None else []))

    def _compact_out(self, T, ro):
        """The compact forward of T tokens, its hidden states [1, T, D] in the buffer every compact graph shares
        (_shared_out; 104 graphs of up to 12,288 tokens kept 0.8 GB on basal-2.0-mini). With readout positions it
        returns [COMPACT_SEQS, D] (a few KB per graph) as it is."""
        if ro is not None or self.gemma is not None:
            return self.fwd_c
        D = self._hidden_size
        out = self._shared_out("compact", max(self.COMPACT_LENS) * D, self._dtype)[: T * D].view(1, T, D)
        fwd = self.fwd_c

        def compact(*s):
            return _copy_same(out, fwd(*s))
        return compact

    def _graph_sfx(self, b, S, L):
        """Pass-2 graph (b rows of S tokens over root keys / values of length L, the layout of the plain graphs):
        (graph, ids, mask, positions, kv, hidden states)."""
        key = ("x", b, S, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, S), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(L, L + S, device=self.dev)[None].expand(b, S).contiguous()
        dt = self._dtype
        if self.flex:
            from .lean import TreeMask
            mask = TreeMask(torch.zeros(b, dtype=torch.long, device=self.dev),
                            torch.full((b, S), S, dtype=torch.long, device=self.dev),
                            torch.zeros((b, S + 1, S + 1), dtype=torch.bool, device=self.dev))
        elif self.gemma is not None:
            mask = torch.zeros((b, 1, S, L + S), dtype=dt, device=self.dev)
        else:  # the mask is written whole before every replay: one buffer for every pass-2 graph (_shared_out)
            mask = self._shared_out("pass2 mask", self._pass2_max(lambda b, S, L: b * S * (L + S)), dt)[
                : b * S * (L + S)].view(b, 1, S, L + S)
        if self.gemma is not None or self.flex:
            return self._capture(key, self.fwd_sfx, ids, mask, pos, self._kv_views(b, L))
        out = self._shared_out("pass2", self._pass2_max(lambda b, S, L: b * S) * self._hidden_size, dt)[
            : b * S * self._hidden_size].view(b, S, -1)
        fwd = self.fwd_sfx

        def suffix(i, m, p, kv):
            return _copy_same(out, fwd(i, m, p, kv))
        return self._capture(key, suffix, ids, mask, pos, self._kv_views(b, L))

    @property
    def _hidden_size(self):
        return self.model.config.hidden_size

    def _pass2_max(self, size):
        """The largest size(b, S, L) over the pass-2 graphs warm captures (Llama)."""
        shapes = [(b, S, L) for b in self.TREE_ROWS for L in self.LENS for S in self.SUFFIX_LENS
                  if b * S <= self.TOKEN_BUDGET and self._pass2_ok(b, L)]
        if self.prefix:
            shapes += [(b, S, L) for L in self.LENS if L <= self.PREFIX_ROOT for S in self.PREFIX_SFX
                       for b in self.PREFIX_ROWS if b * S <= self.TOKEN_BUDGET]
        return max(size(*s) for s in shapes)

    def _tree_mask(self, seg, parents, roots, S):
        """The trie of pass-2 rows as a TreeMask (basal/lean.py): root tokens per row, block per token (S: padding),
        ancestors per block."""
        from .lean import TreeMask
        anc = np.zeros((len(parents), S + 1, S + 1), dtype=bool)
        for r, par in enumerate(parents):
            if par:
                anc[r, : len(par), : len(par)] = self._ancestors([par], len(par))[0]
        s = np.where(seg >= 0, seg, S)
        return TreeMask(self._to_dev(torch.tensor(roots)), self._to_dev(torch.from_numpy(s)),
                        self._to_dev(torch.from_numpy(anc)))

    @staticmethod
    def _ancestors(parents, NB):
        """Per row, anc[k, j] = block j is block k or one of its ancestors; index NB - 1 is padding (sees nothing)."""
        anc = np.zeros((len(parents), NB, NB), dtype=bool)
        for r, par in enumerate(parents):
            a = anc[r]
            for k, pk in enumerate(par):  # parents precede children (depth-first order)
                if pk >= 0:
                    a[k] = a[pk]
                a[k, k] = True
        return anc

    def _mask_from_seg(self, seg, parents=None):
        """seg [b, L]: block id of every token, -1 = padding; parents: per row, the parent block of every block (-1 =
        root). A token sees earlier tokens of its own block and all tokens of its ancestor blocks (the prefix trie of
        the prompts), so every readout equals a separate forward of its own prompt. Without parents: block 0 is the
        shared prefix of all other blocks. -> additive 4D mask [b, 1, L, L]."""
        b, L = seg.shape
        if parents is None:
            nb = int(seg.max()) + 1
            parents = [[-1] + [0] * (nb - 1)] * b
        NB = max(map(len, parents)) + 1  # last index: padding, sees nothing
        anc = self._to_dev(self._ancestors(parents, NB))
        s = torch.where(seg >= 0, seg, NB - 1)
        allow = anc[torch.arange(b, device=seg.device)[:, None, None], s[:, :, None], s[:, None, :]]
        allow &= torch.ones(L, L, dtype=torch.bool, device=seg.device).tril()[None]
        allow |= torch.eye(L, dtype=torch.bool, device=seg.device)[None]
        dt = next(self.model.parameters()).dtype
        return torch.where(allow, 0.0, torch.finfo(dt).min).to(dt)[:, None]

    @staticmethod
    def _pack(toks, cut=None):
        """Token lists of one group -> (ids, positions, block ids, readout positions, block parents): the compressed
        prefix trie of the prompts, flattened depth-first. Shared prefixes (the state for all questions, a question's
        text for its option orders) are computed once; positions are each token's index in its own prompt. cut: the
        root block ends after this many tokens (the state prefix of a request, the unit of the state cache) when every
        prompt is longer."""
        import os
        ids, pos, seg, parent, last = [], [], [], [], [None] * len(toks)
        cut = cut if cut and all(len(t) > cut for t in toks) and len({tuple(t[:cut]) for t in toks}) == 1 else None

        def build(members, start, par):
            end = start + len(os.path.commonprefix([toks[m][start:] for m in members]))
            if par == -1 and cut is not None:
                end = min(end, cut)
            blk = len(parent)
            parent.append(par)
            ids.extend(toks[members[0]][start:end]); pos.extend(range(start, end)); seg.extend([blk] * (end - start))
            kids = {}
            for m in members:
                if len(toks[m]) == end:
                    last[m] = len(ids) - 1
                else:
                    kids.setdefault(toks[m][end], []).append(m)
            for ms in kids.values():
                build(ms, end, blk)

        roots = {}
        for m, tk in enumerate(toks):
            roots.setdefault(tk[0], []).append(m)
        for ms in roots.values():
            build(ms, 0, -1)
        return ids, pos, seg, last, parent

    def _rows(self, lids):
        """Answer-row slice of the LM head for the letters in lids, and each readout's column indices (padded, masked).
        softmax over the letters of the full log-softmax equals softmax over these rows' logits (row-only readout). The
        rows are kept in fp32: the code logits are computed in fp32 from the hidden state (bf16 logits of magnitude
        16-32 have steps of 0.125, which clumps the probabilities)."""
        key = tuple(sorted({i for ids in lids for i in ids}))
        if key not in self._wrows:
            self._wrows[key] = self.model.lm_head.weight.index_select(0, self._to_dev(torch.tensor(key))).float()
        col = {t: j for j, t in enumerate(key)}
        K = max(map(len, lids))
        idx = self._to_dev(torch.tensor([[col[i] for i in ids] + [0] * (K - len(ids)) for ids in lids]))
        pad = self._to_dev(torch.tensor([[False] * len(ids) + [True] * (K - len(ids)) for ids in lids]))
        return self._wrows[key], idx, pad

    def _letters(self, z, idx, pad, lids):
        p = torch.softmax(z.float().gather(1, idx).masked_fill(pad, float("-inf")), -1).tolist()  # one device sync
        return [row[: len(ids)] for row, ids in zip(p, lids)]

    def _code_logits(self, h, W):
        """fp32 logits of the code rows W from normalised hidden states h, with the model's final softcap (Gemma)."""
        z = h.float() @ W.T
        return torch.tanh(z / self.softcap) * self.softcap if self.softcap else z

    def _readout(self, h, lids):
        W, idx, pad = self._rows(lids)
        return self._letters(self._code_logits(h, W), idx, pad, lids)  # last_hidden_state is already normalised

    def _host_rows(self, packs, idx, b, L):
        """Packed rows -> padded host arrays ids / positions / block ids [b, L], readout rows and columns, and the
        block parents of every row."""
        h_ids = np.full((b, L), self.tok.pad_token_id, dtype=np.int64)
        h_pos = np.tile(np.arange(L, dtype=np.int64), (b, 1))
        h_seg = np.full((b, L), -1, dtype=np.int64)
        rows, cols, parents = [], [], [[] for _ in range(b)]
        for r, k in enumerate(idx):
            t, pp, sg, last, parents[r] = packs[k]
            h_ids[r, : len(t)] = t; h_pos[r, : len(t)] = pp; h_seg[r, : len(t)] = sg
            rows += [r] * len(last); cols += last
        return torch.from_numpy(h_ids), torch.from_numpy(h_pos), torch.from_numpy(h_seg), rows, cols, parents

    def _fill(self, packs, idx, b, L, ids, mask, pos):
        h_ids, h_pos, h_seg, rows, cols, parents = self._host_rows(packs, idx, b, L)
        ids.copy_(self._to_dev(h_ids), non_blocking=True)
        pos.copy_(self._to_dev(h_pos), non_blocking=True)
        mask.copy_(self._mask_from_seg(self._to_dev(h_seg), parents))
        return self._to_dev(torch.tensor(rows)), self._to_dev(torch.tensor(cols))

    def _eager_shared(self, pack, lids):
        """Readout of one packed row in a single eager masked forward (the reference for the masked graphs)."""
        t, pp, sg, last, par = pack
        h = self._forward_masked(torch.tensor([t], device=self.dev),
                                 self._mask_from_seg(torch.tensor([sg], device=self.dev), [par]),
                                 torch.tensor([pp], device=self.dev))[0, last]
        return self._readout(h, lids)

    def _plain_hidden(self, packs, idx):
        """Single-prompt units -> hidden states at their readout positions (a plain graph, or an eager forward without
        a mask when longer than the largest plain graph)."""
        n = max(len(packs[k][0]) for k in idx)
        if n > self.PLAIN_LENS[-1]:
            assert len(idx) == 1
            t, _, _, last, _ = packs[idx[0]]
            if n > self.one_forward:  # in chunks (_continue)
                assert all(i == len(t) - 1 for i in last)  # a single prompt (or duplicates) reads out at its last token
                h = self._continue(list(t))[1][None].expand(len(last), -1).clone()
                torch.cuda.empty_cache()
                return h
            try:
                if self.gemma is not None:  # a single prompt (or duplicates) reads out at its last token
                    return self._forward(self._to_dev(torch.tensor([t])), ro=self._to_dev(torch.tensor([[last[0]]])))[
                        0, [0] * len(last)]
                return self._forward(self._to_dev(torch.tensor([t])))[0, last]
            except torch.OutOfMemoryError as e:  # beyond what the estimate allowed for: in chunks after all
                traceback.clear_frames(e.__traceback__)  # (the failed forward's tensors, held by the error until then)
                torch.cuda.empty_cache()
                self.one_forward = min(self.one_forward, n - 1)
                return self._plain_hidden(packs, idx)
        L, b = self._bucket(n, self.PLAIN_LENS), self._bucket(len(idx), self.BATCHES)
        g, ids, _, *ro, out = self._graph(b, L)
        h_ids, _, _, rows, cols, _ = self._host_rows(packs, idx, b, L)
        ids.copy_(self._to_dev(h_ids), non_blocking=True)
        if ro:  # Gemma 4: the forward reads out at each row's last token (duplicates share it)
            h_ro = np.zeros((b, 1), dtype=np.int64)
            for r, c in zip(rows, cols):
                h_ro[r, 0] = c
            ro[0].copy_(self._to_dev(h_ro), non_blocking=True)
            cols = [0] * len(rows)
        g.replay()
        return out[self._to_dev(torch.tensor(rows)), self._to_dev(torch.tensor(cols))]

    def _compact_hidden(self, packs, idx):
        """Single-prompt units packed into one compact row (the graph of the next COMPACT_LENS total; the padding after
        the last prompt is one more prompt nobody reads) -> hidden states at their readout positions."""
        pre = {}  # prompt -> (start of its template prefix in template_kv, prefix tokens): only its own tokens run
        if self.template:
            packs = list(packs)
            for j, k in enumerate(idx):
                t, pp, sg, last, par = packs[k]
                for a, ids_p in self.template:
                    P = len(ids_p)
                    if len(t) > P and tuple(t[:P]) == ids_p:
                        pre[j] = (a, P)
                        packs[k] = (t[P:], pp[P:], sg[P:], [i - P for i in last], par)
                        break
        n = [len(packs[k][0]) for k in idx]
        T = self._bucket(sum(n), self.COMPACT_LENS)
        g, ids, pos, cu, *extra, out = self._graph_c(T)
        ro = extra[0] if self.readout_layer else None
        h_ids = np.full(T, self.tok.pad_token_id, dtype=np.int64)
        h_pos = np.arange(T, dtype=np.int64)
        h_cu = np.full(self.COMPACT_SEQS + 2, T, dtype=np.int32)
        h_ro = np.zeros(self.COMPACT_SEQS, dtype=np.int64)
        if self.template:
            h_cuk = np.zeros(self.COMPACT_SEQS + 2, dtype=np.int32)
            h_g = np.zeros(T + self.COMPACT_SEQS * self.template_max, dtype=np.int64)
            ok = 0  # keys so far
        cols, rows, o = [], [], 0
        for j, (k, m) in enumerate(zip(idx, n)):
            t, pp, _, last, _ = packs[k]
            h_ids[o: o + m] = t
            h_pos[o: o + m] = pp
            h_cu[j] = o
            h_ro[j] = o + m - 1  # a single prompt (or duplicates) is read out at its last token
            cols += [o + i for i in last]
            rows += [j] * len(last)
            if self.template:
                h_cuk[j] = ok
                if j in pre:  # the template prefix's keys / values, then the prompt's own
                    a, P = pre[j]
                    h_g[ok: ok + P] = np.arange(a, a + P)
                    ok += P
                h_g[ok: ok + m] = self.template_tokens + np.arange(o, o + m)
                ok += m
            o += m
        h_cu[len(idx)] = o
        h_pos[o:] = np.arange(T - o)  # the padding: one more prompt from position 0
        ids.copy_(self._to_dev(h_ids[None]), non_blocking=True)
        pos.copy_(self._to_dev(h_pos[None]), non_blocking=True)
        cu.copy_(self._to_dev(h_cu), non_blocking=True)
        if ro is not None:
            ro.copy_(self._to_dev(h_ro), non_blocking=True)
        if self.template:
            h_cuk[len(idx)] = ok
            h_g[ok: ok + T - o] = self.template_tokens + np.arange(o, T)
            h_cuk[len(idx) + 1:] = ok + T - o
            extra[-2].copy_(self._to_dev(h_cuk), non_blocking=True)
            extra[-1].copy_(self._to_dev(h_g), non_blocking=True)
        g.replay()
        if ro is not None:
            return out[self._to_dev(torch.tensor(rows))]
        return out[0, self._to_dev(torch.tensor(cols))]

    @torch.no_grad()
    def _tree_hidden(self, packs, idx, keys=None, hits=None):
        """Branched units (one root each) -> hidden states at their readout positions, in two passes: their roots in a
        plain graph that also writes the keys and values (an uncompiled-shape forward beyond the largest graph), then
        all other blocks, one row per unit, over those keys and values with the trie mask. keys / hits: per unit, the
        state-cache key of its root and the cached keys / values; when every root is cached, pass 1 is skipped and the
        cached keys / values go where pass 1 would have written them; the roots computed here are stored. Gemma 4:
        pass 1 reads out at the root's last token (the only readout a root can hold: a prompt that is the shared
        prefix itself) and pass 2 without a graph at the readout positions only."""
        P = [packs[k][2].index(1) for k in idx]  # block 0 (the root) comes first and is contiguous
        b = len(idx)
        n2 = max(len(packs[k][0]) - p for k, p in zip(idx, P))  # longest part below the root
        keys, hits = keys or [None] * b, hits or [None] * b
        # pass-2 shapes: the template-prefix family (--prefix-cache: short roots and rows, more units per forward) or
        # the branched units of SOAM / option orders
        short = self.prefix and max(P) <= self.PREFIX_ROOT and n2 <= self.PREFIX_SFX[-1]
        rows, sfx = (self.PREFIX_ROWS, self.PREFIX_SFX) if short else (self.TREE_ROWS, self.SUFFIX_LENS)
        h1 = None
        if all(e is not None for e in hits):  # every root from the state cache: no pass 1
            if max(P) > self.PLAIN_LENS[-1]:
                assert b == 1
                L = P[0]
                kv = self._kv_new(L)
            else:
                L = self._bucket(max(P), self.PLAIN_LENS)
                B = max(b, self._bucket(b, rows) if L <= self.LENS[-1] else b)
                kv = self._kv_views(B if B * L <= self._kv_tokens() else b, L)  # the rows of a pass-2 graph if any
            same = {}  # rows with the same cached root (a template prefix): one broadcast copy per layer
            for r, e in enumerate(hits):
                same.setdefault(id(e), (e, P[r], []))[2].append(r)
            for e, p, rs in same.values():
                # a cached entry may hold the root's last tokens only (Gemma 4's sliding layers, _kv_new): they end
                # where the root ends (a buffer of the graph) or the buffer does (a fresh one as long as the entry)
                if len(rs) == 1:
                    for (kb, vb), (ke, ve) in zip(kv, e):
                        z = min(p, kb.shape[2])
                        kb[rs[0], :, z - ke.shape[1]: z].copy_(ke, non_blocking=True)
                        vb[rs[0], :, z - ke.shape[1]: z].copy_(ve, non_blocking=True)
                    continue
                ix = self._to_dev(torch.tensor(rs))
                for (kb, vb), (ke, ve) in zip(kv, e):
                    z = min(p, kb.shape[2])
                    kb[:, :, z - ke.shape[1]: z].index_copy_(0, ix, ke[None].expand(len(rs), *ke.shape))
                    vb[:, :, z - ke.shape[1]: z].index_copy_(0, ix, ve[None].expand(len(rs), *ve.shape))
        elif max(P) > self.PLAIN_LENS[-1]:
            # a root longer than the largest graph runs uncompiled, like a long single prompt: the compiled forward
            # recompiles for these fresh key / value buffers (44-109 s on the first long request on an H100), the
            # uncompiled one costs a few percent of a forward that is rare and long anyway
            assert b == 1
            L, B = P[0], 1
            if L > self.one_forward:  # in chunks (_continue); pass 1 holds the root's last token only
                kv, h1 = self._continue(list(packs[idx[0]][0][:L]))
                kv, h1 = [(k[None], v[None]) for k, v in kv], h1[None, None]
            else:
                try:
                    kv = self._kv_new(L)
                    ro = self._to_dev(torch.tensor([[L - 1]])) if self.gemma is not None else None
                    h1 = self._forward(self._to_dev(torch.tensor([packs[idx[0]][0][:L]])), kv, ro)
                except torch.OutOfMemoryError as e:  # beyond what the estimate allowed for: in chunks after all
                    kv = None
                    traceback.clear_frames(e.__traceback__)  # (the failed forward's tensors, held by the error)
                    torch.cuda.empty_cache()
                    self.one_forward = min(self.one_forward, L - 1)
                    return self._tree_hidden(packs, idx, keys, hits)
        else:
            L, B = self._bucket(max(P), self.PLAIN_LENS), self._bucket(b, self.BATCHES)
            g, ids, kv, *ro, h1 = self._graph(B, L)
            h_ids = np.full((B, L), self.tok.pad_token_id, dtype=np.int64)
            for r, k in enumerate(idx):
                h_ids[r, : P[r]] = packs[k][0][: P[r]]
            ids.copy_(self._to_dev(h_ids), non_blocking=True)
            if ro:
                ro[0].copy_(self._to_dev(np.array([[p - 1] for p in P] + [[0]] * (B - b), dtype=np.int64)),
                            non_blocking=True)
            g.replay()
        if h1 is not None and self.state_cache is not None:  # store the roots computed here
            for r, key in enumerate(keys):
                if key is not None and hits[r] is None:
                    self.state_cache.put(key, [(kb[r, :, : P[r]].clone(), vb[r, :, : P[r]].clone()) for kb, vb in kv],
                                         tokens=packs[idx[r]][0][: P[r]])
        S = next((x for x in sfx if n2 <= x), None)
        graph = None
        # a pass-2 graph, rows padded (stale buffer rows are masked); Gemma 4: over the longer plain graphs' roots too
        if S and (L <= self.LENS[-1] or self.gemma is not None) and b <= rows[-1]:
            Bc = self._bucket(b, rows)
            graph = self.graphs.get(("x", Bc, S, L))
        if graph is not None:
            B = Bc
            kv = self._kv_views(B, L)
        else:
            S, B = -(-n2 // self.SUFFIX) * self.SUFFIX, b
            kv = [(k[:b], v[:b]) for k, v in kv]
        ids2 = np.full((B, S), self.tok.pad_token_id, dtype=np.int64)
        pos2 = np.tile(np.arange(L, L + S, dtype=np.int64), (B, 1))
        seg2 = np.full((B, S), -1, dtype=np.int64)
        par2, sel = [[] for _ in range(B)], []  # sel: (pass, row, column) of every readout, in unit order
        for r, (k, p) in enumerate(zip(idx, P)):
            t, pp, sg, last, par = packs[k]
            n = len(t) - p
            ids2[r, :n], seg2[r, :n] = t[p:], [x - 1 for x in sg[p:]]  # blocks 1.. -> 0..: the trie below the root
            pos2[r, :n], pos2[r, n:] = pp[p:], pp[-1] + 1 + np.arange(S - n)
            par2[r] = [-1 if q <= 0 else q - 1 for q in par[1:]]
            # Gemma 4: pass 1 holds the root's last token only (column 0 of its readout rows)
            sel += [(1, r, i if self.gemma is None and h1.shape[1] > 1 else 0) if i < p else (2, r, i - p) for i in last]
        # FlexAttention only in the pass-2 graphs captured at start-up: an uncaptured shape (long roots, many or long
        # branches) runs the dense pass 2 uncompiled, as without flex. The compiled flex layer specialises on shape
        # classes, and a request of a new class compiled for 1-4 s (SPEED.md sections 6 and 11). Gemma 4: its flex
        # kernel takes the TreeMask in uncaptured shapes too (one compiled flex_attention of dynamic shapes)
        if self.flex and (graph is not None or self.gemma is not None):  # root tokens, blocks, ancestors
            tm = self._tree_mask(seg2, par2, P + [0] * (B - b), S)
            if graph is None:
                h2, sel = self._gemma_suffix(ids2, tm, pos2, kv, sel, B)
            else:
                g2, s_ids, s_mask, s_pos, _, h2 = graph
                s_ids.copy_(self._to_dev(ids2), non_blocking=True)
                s_pos.copy_(self._to_dev(pos2), non_blocking=True)
                for dst, src in zip(s_mask, tm):
                    dst.copy_(src)
                g2.replay()
        elif graph is None and self._cascades(n2):
            h2 = self._forward_cascade(ids2, pos2, kv, _varlen_tree(seg2, par2, P, kv[0][0].shape[2], self._to_dev,
                                                                    self.fuse_merge))
        else:
            m2 = self._mask_from_seg(self._to_dev(seg2), par2)                             # [B, 1, S, S]
            Pr = self._to_dev(torch.tensor(P + [0] * (B - b)))
            root = Pr[:, None] > torch.arange(L, device=self.dev)[None]                     # [B, L]
            pre = torch.where(root, 0.0, torch.finfo(m2.dtype).min).to(m2.dtype)[:, None, None].expand(B, 1, S, L)
            if graph is None and self.gemma is not None:
                h2, sel = self._gemma_suffix(ids2, torch.cat([pre, m2], -1), pos2, kv, sel, B)
            elif graph is None:
                fwd = self._forward_suffix if self.flex else self.fwd_sfx
                h2 = fwd(self._to_dev(ids2), torch.cat([pre, m2], -1), self._to_dev(pos2), kv)
            else:
                g2, s_ids, s_mask, s_pos, _, h2 = graph
                s_ids.copy_(self._to_dev(ids2), non_blocking=True)
                s_pos.copy_(self._to_dev(pos2), non_blocking=True)
                s_mask[..., :L].copy_(pre); s_mask[..., L:].copy_(m2)
                g2.replay()
        out = torch.empty((len(sel), h2.shape[-1]), dtype=h2.dtype, device=self.dev)
        for src, h in ((1, h1), (2, h2)):
            j = [n for n, x in enumerate(sel) if x[0] == src]
            if j:
                assert h is not None, "a readout inside a cached root"
                rows = self._to_dev(torch.tensor([sel[n][1] for n in j]))
                cols = self._to_dev(torch.tensor([sel[n][2] for n in j]))
                out[self._to_dev(torch.tensor(j))] = h[rows, cols]
        if max(P) > max(self.PLAIN_LENS[-1], self.one_forward) and self.dev.type == "cuda":  # a root in chunks: its
            del kv, h1, h2  # passes' cached blocks back to the GPU (an 82k-token ultra request left 6 GB of them
            torch.cuda.empty_cache()  # behind; after every long root it cost ultra's multi profile 4%, 20 Oct)
        return out

    def _gemma_suffix(self, ids2, mask, pos2, kv, sel, B):
        """Gemma 4, pass 2 without a graph: uncompiled, the layers from the last own-key one on at the readout rows
        only. -> (hidden states [B, R, D], sel with each pass-2 readout's slot in its row)."""
        ro2, slot = self._readout_slots(sel, B)
        h2 = self._forward_suffix(self._to_dev(ids2), mask, self._to_dev(pos2), kv, self._to_dev(ro2))
        return h2, [(x[0], x[1], slot[n]) if x[0] == 2 else x for n, x in enumerate(sel)]

    @staticmethod
    def _readout_slots(sel, B):
        """The pass-2 readouts of _tree_hidden ((2, row, column) in sel) as readout positions per row [B, R] (padded
        with each row's first; R = the most of any row) and the slot of each readout in its row (None for pass 1)."""
        cols = [[] for _ in range(B)]
        slot = [None] * len(sel)
        for n, (src, r, c) in enumerate(sel):
            if src == 2:
                if c not in cols[r]:
                    cols[r].append(c)
                slot[n] = cols[r].index(c)
        R = max(1, max(map(len, cols)))
        return np.array([c + [c[0] if c else 0] * (R - len(c)) for c in cols], dtype=np.int64), slot

    @torch.no_grad()
    def _tree_cache(self, pack):
        """A branched unit in two eager passes over a transformers KV cache (the MPS and early-exit backends, whose
        attention has no key / value buffers): the root with causal attention, then every other block as one packed
        row over that cache with the trie mask; on the CPU the branch pass of _cascade instead. -> hidden states at the
        readout positions."""
        t, pp, sg, last, par = pack
        P = sg.index(1) if 1 in sg else len(t)  # block 0 (the root) comes first and is contiguous
        out = self.model.model(input_ids=self._to_dev(torch.tensor([t[:P]])), use_cache=True)
        h = [None] * len(last)
        for m, i in enumerate(last):
            if i < P:  # a prompt that ends inside the root (a duplicate of the shared prefix)
                h[m] = out.last_hidden_state[0, i]
        rest = [m for m, i in enumerate(last) if i >= P]
        if rest and self.dev.type == "cpu" and cascade_ok(self.model.config):
            h2 = self._cascade(t[P:], pp[P:], [s - 1 for s in sg[P:]], [-1 if q <= 0 else q - 1 for q in par[1:]],
                               out.past_key_values)
        elif rest:
            N = len(t) - P
            seg2 = torch.tensor([[s - 1 for s in sg[P:]]])  # blocks 1.. -> 0..; their trie without the root
            par2 = [[-1 if q <= 0 else q - 1 for q in par[1:]]]
            m2 = self._mask_from_seg(self._to_dev(seg2), par2)  # [1, 1, N, N]
            m2 = torch.cat([torch.zeros((1, 1, N, P), dtype=m2.dtype, device=m2.device), m2], -1)  # root: visible
            h2 = self.model.model(input_ids=self._to_dev(torch.tensor([t[P:]])), attention_mask=m2,
                                  position_ids=self._to_dev(torch.tensor([pp[P:]])), past_key_values=out.past_key_values,
                                  use_cache=True).last_hidden_state[0]
        for m in rest:
            h[m] = h2[last[m] - P]
        return torch.stack(h)

    def _cascade(self, toks, pos, seg, parents, cache):
        """The branch pass of _tree_cache on the CPU: the tokens below the root (toks, their positions pos, their blocks
        seg 0.., the blocks' parents with -1 = the root) attend to the root's keys / values in the KV cache as they are
        and, per trie block, to their own ancestors (basal_attention's basal_tree), merged by log-sum-exp: no copied
        keys and no [N, P + N] mask, so a unit need not be split (_oversize) and the root runs once. Parts of whole
        top-level subtrees (each sees only the root and itself) of at most BRANCH_TOKENS tokens (one larger subtree:
        a part of its own) bound the activations. -> hidden states of the tokens below the root [N, D]."""
        from transformers import AttentionInterface
        AttentionInterface.register("basal_sdpa", basal_attention)
        prefix = [(layer.keys, layer.values) for layer in cache.layers]
        top = []  # the top-level subtree of every block (a block comes after its parent)
        for b, p in enumerate(parents):
            top.append(b if p < 0 else top[p])
        runs, a = [], 0  # top-level subtrees: contiguous runs of tokens (the trie is flattened depth first)
        for i in range(1, len(seg) + 1):
            if i == len(seg) or top[seg[i]] != top[seg[i - 1]]:
                runs.append((a, i))
                a = i
        parts = [list(runs[0])]
        for a, b in runs[1:]:
            if b - parts[-1][0] > self.BRANCH_TOKENS:
                parts.append([a, b])
            else:
                parts[-1][1] = b
        cfg = self.model.config  # basal_attention for this pass only (the MPS backend runs one call at a time)
        impl, cfg._attn_implementation = cfg._attn_implementation, "basal_sdpa"
        try:
            return torch.cat([self.model.model(input_ids=self._to_dev(torch.tensor([toks[a:b]])),
                                               position_ids=self._to_dev(torch.tensor([pos[a:b]])), use_cache=False,
                                               basal_prefix=prefix, basal_tree=_block_plan(seg[a:b], parents)
                                               ).last_hidden_state[0] for a, b in parts])
        finally:
            cfg._attn_implementation = impl

    def _long(self, pack, lids):
        """Readout of a unit longer than the largest graph: one forward without a mask (a single prompt) or the two
        passes over the root's KV cache (branches)."""
        if len(pack[4]) == 1:
            return self._readout(self._forward(self._to_dev(torch.tensor([pack[0]])))[0, pack[3]], lids)
        return self._readout(self._tree_cache(pack), lids)

    def _oversize(self, pack, n):
        """A packed row of n readouts that must be split: prompts without one common root (no shared prefix to
        compute once; the two passes need one root), or more than BRANCH_TOKENS tokens below the root. Pass 2 builds
        dense masks of (tokens below the root)^2 and (tokens below the root) x (root): many long questions over one
        state (32 questions of 1-2k tokens each) made them tens of GB and ran out of memory."""
        root = pack[2].index(1) if 1 in pack[2] else len(pack[0])
        return pack[4].count(-1) > 1 or len(pack[0]) - root > self.BRANCH_TOKENS

    def _units(self, groups):
        """Groups -> packed rows ("units"). A group whose packed row must be split is bisected (exact: every branch
        still sees its own prompt only). Without `shared` every prompt is a unit of its own. -> [(pack, lids, group,
        first)]."""
        units = []
        for k, g in enumerate(groups):
            prompts, lids = g[0], g[1]
            cut = g[2] if len(g) > 2 and getattr(self, "state_cache", None) is not None else None
            toks = self.encode(prompts)
            if not getattr(self, "shared", True):
                units += [(self._pack([tk]), [x], k, a) for a, (tk, x) in enumerate(zip(toks, lids))]
                continue
            todo = [(0, len(toks))]
            while todo:
                a, b = todo.pop()
                pk = self._pack(toks[a:b], cut)
                if self._oversize(pk, b - a) and b - a > 1:
                    m = (a + b) // 2
                    todo += [(m, b), (a, m)]
                else:
                    units.append((pk, lids[a:b], k, a))
        return units

    @staticmethod
    def _assemble(groups, units, probs):
        res = [[None] * len(g[0]) for g in groups]
        for (pk, lids, k, a), pr in zip(units, probs):
            res[k][a: a + len(lids)] = pr
        return res

    # -- public calls --------------------------------------------------------------------------------------------
    streams = True  # run_shared(..., done=callback) reports every group as soon as its forwards are read out
    inflight = 2    # calls the server may run at once: one call's host work overlaps the other's forwards
    # a forward's input copies, replay and readout are enqueued by one call at a time, all on the device's default
    # stream (every thread's current stream; nothing here sets another), so the graphs' static inputs, the key / value
    # buffer and the shared graph outputs (_shared_out) are reused only after the previous call's reads, in launch order
    _launch = threading.Lock()

    @torch.no_grad()
    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    def _forwards(self, units, cached=()):
        """Units -> [(unit indices, kind)] in launch order: single prompts in plain graphs (short first), then branched
        units in two passes, batched by the length of their roots; units whose root is in the state cache (`cached`)
        are batched separately ("cached": pass 2 only)."""
        packs = [u[0] for u in units]
        plain = [k for k, pk in enumerate(packs) if len(pk[4]) == 1]  # one block: a single prompt (or duplicates)
        out = []
        if self.compact:  # short first, up to COMPACT_SEQS prompts and the largest compact graph per forward
            plain.sort(key=lambda k: len(packs[k][0]))
            cur, tot = [], 0
            for k in plain:
                n = len(packs[k][0])
                if n > self.PLAIN_LENS[-1]:
                    out.append(([k], "plain"))
                    continue
                if cur and (len(cur) == self.COMPACT_SEQS or tot + n > self.COMPACT_LENS[-1]):
                    out.append((cur, "compact")); cur, tot = [], 0
                cur.append(k); tot += n
            out += [(cur, "compact")] if cur else []
            plain = []
        out += [([plain[j] for j in idx], "plain") for idx in self._chunks([len(packs[k][0]) for k in plain],
                                                                           self.PLAIN_LENS)]
        for kind in ("tree", "cached"):
            tree = [k for k, pk in enumerate(packs) if len(pk[4]) > 1 and (k in cached) == (kind == "cached")]
            if kind == "cached" and self.prefix:  # short requests over a cached template prefix: by length below it
                short = [k for k in tree if packs[k][2].index(1) <= self.PREFIX_ROOT
                         and len(packs[k][0]) - packs[k][2].index(1) <= self.PREFIX_SFX[-1]]
                rest = [len(packs[k][0]) - packs[k][2].index(1) for k in short]
                out += [([short[j] for j in idx], kind) for idx in self._chunks(rest, self.PREFIX_SFX, None,
                                                                                self.PREFIX_ROWS[-1])]
                tree = [k for k in tree if k not in set(short)]
            roots = [packs[k][2].index(1) for k in tree]
            rest = [len(packs[k][0]) - r for k, r in zip(tree, roots)]
            out += [([tree[j] for j in idx], kind) for idx in self._chunks(roots, self.PLAIN_LENS, rest,
                                                                           self.TREE_ROWS[-1])]
        return out

    def _cache_keys(self, groups, units):
        """Per unit, the state-cache key of its root when the root is the state prefix its group was cut at (None
        otherwise), and the cached keys / values of the units whose root is in the cache."""
        if self.state_cache is None:
            return None, {}
        keys, hits = [], {}
        for k, (pk, _, g, _) in enumerate(units):
            cut = groups[g][2] if len(groups[g]) > 2 else None
            root = pk[2].index(1) if len(pk[4]) > 1 else None
            key = self.state_cache.key(pk[0][:root]) if cut and root == cut else None
            keys.append(key)
            if key is not None:
                e = self.state_cache.get(key)
                if e is not None:
                    hits[k] = e
        return keys, hits

    @torch.no_grad()
    def run_shared(self, groups, policy=None, done=None):
        """All forwards of the call are launched first; each one's option probabilities are copied to pinned host
        memory behind it, so the readouts are collected in launch order while later forwards still run. done(k, res)
        (optional) is called for group k as soon as all its units are read out."""
        units = self._units(groups)
        packs = [u[0] for u in units]
        cuda = self.dev.type == "cuda"
        pending = []
        keys, hits = self._cache_keys(groups, units)
        self._extend_states(units, keys, hits)
        for idx, kind in self._forwards(units, hits):
            with self._launch:  # the static buffers of a graph are shared by concurrent calls
                if kind == "compact":
                    h = self._compact_hidden(packs, idx)
                elif kind == "plain":
                    h = self._plain_hidden(packs, idx)
                elif keys is None:
                    h = self._tree_hidden(packs, idx)
                else:
                    h = self._tree_hidden(packs, idx, [keys[k] for k in idx], [hits.get(k) for k in idx])
                W, cols, pad = self._rows([x for k in idx for x in units[k][1]])
                p = torch.softmax(self._code_logits(h, W).gather(1, cols).masked_fill(pad, float("-inf")), -1)
                if cuda:
                    host = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
                    host.copy_(p, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record()
                    pending.append((idx, host, ev))
                else:
                    pending.append((idx, p, None))
        res = [[None] * len(g[0]) for g in groups]
        left = [0] * len(groups)
        for u in units:
            left[u[2]] += 1
        for idx, host, ev in pending:
            if ev is not None:
                ev.synchronize()
            rows, j = host.tolist(), 0
            for k in idx:
                _, lids, g, a = units[k]
                res[g][a: a + len(lids)] = [r[: len(x)] for r, x in zip(rows[j: j + len(lids)], lids)]
                j += len(lids)
                left[g] -= 1
                if left[g] == 0 and done is not None:
                    done(g, res[g])
        return res


    @torch.no_grad()
    def _long_hidden(self, pack, key=None, hit=None, between=None):
        """One unit of the long lane -> hidden states at its readout positions: a single prompt in one layered forward;
        a branched unit in the two passes over fresh key / value buffers (pass 1 layered, or the cached state; pass 2
        uncompiled). The static buffers of the main lane's graphs are never touched."""
        t, pp, sg, last, par = pack
        if len(par) == 1:
            return self._forward_layered(self._to_dev(torch.tensor([t])), None, between)[0, last]
        P = sg.index(1)  # block 0 (the root) comes first and is contiguous
        h1 = None
        if hit is not None:
            kv = [(k[None], v[None]) for k, v in hit]
        else:
            kv = self._kv_new(P)
            h1 = self._forward_layered(self._to_dev(torch.tensor([t[:P]])), kv, between)[0]
            if key is not None and self.state_cache is not None:
                self.state_cache.put(key, [(k[0].clone(), v[0].clone()) for k, v in kv], tokens=t[:P])
        n = len(t) - P
        seg2, par2 = np.array([[x - 1 for x in sg[P:]]]), [[-1 if q <= 0 else q - 1 for q in par[1:]]]
        if self._cascades(n):
            h2 = self._forward_cascade(np.array([t[P:]]), np.array([pp[P:]]), kv,
                                       _varlen_tree(seg2, par2, [P], kv[0][0].shape[2], self._to_dev,
                                                    self.fuse_merge))[0]
        else:
            m2 = self._mask_from_seg(self._to_dev(torch.from_numpy(seg2)), par2)                # [1, 1, n, n]
            m2 = torch.cat([torch.zeros((1, 1, n, P), dtype=m2.dtype, device=self.dev), m2], -1)  # the root: visible
            h2 = self._forward_suffix(self._to_dev(torch.tensor([t[P:]])), m2, self._to_dev(torch.tensor([pp[P:]])),
                                      kv)[0]
        assert h1 is not None or min(last) >= P, "a readout inside a cached root"
        return torch.stack([h1[i] if i < P else h2[i - P] for i in last])

    @torch.no_grad()
    def run_long(self, groups, policy=None, between=None):
        """The long lane's engine call (basal-serve --long-lane): every unit of the groups on its own, uncompiled and
        one decoder layer at a time (between(i) before layer i), over fresh key / value buffers, so it can run while
        the main lane's graphs replay; the state cache is shared with the main lane. -> the option probabilities of
        every prompt, as run_shared."""
        units = self._units(groups)
        keys, hits = self._cache_keys(groups, units)
        self._extend_states(units, keys, hits)
        probs = [self._readout(self._long_hidden(pk, keys[k] if keys else None, hits.get(k), between), lids)
                 for k, (pk, lids, _, _) in enumerate(units)]
        return self._assemble(groups, units, probs)


class ExitHead(torch.nn.Module):
    """Trained early-exit head: RMSNorm (initialised from the final norm) + residual low-rank adapter, followed by the
    frozen LM head restricted to the option letters."""

    def __init__(self, final_norm, d, r=256):
        super().__init__()
        self.norm = type(final_norm)(d, eps=final_norm.variance_epsilon)
        self.a = torch.nn.Linear(d, r, bias=False)
        self.b = torch.nn.Linear(r, d, bias=False)

    def forward(self, h):
        x = self.norm(h.float())
        return x + self.b(self.a(x))


class ExitGraphBackend(GraphBackend):
    """Shared-prefix fast path split into CUDA-graph segments at the exit layers. After each segment the exit heads
    score the letters at the readout positions; if every readout in the chunk reaches the threshold of that layer the
    chunk stops there (a batch stops only as a whole). `policy` selects the thresholds per request: "off" = always the
    final layer, or an agreement level calibrated in advance ("0.999", "0.995", "0.99", "0.98")."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time

    LENS = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072]
    BATCHES = [1, 2, 4, 8, 16, 32]

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, heads_dir=None, default_policy="off",
                 attn="auto", merge="auto"):
        super().__init__(model_dir, dtype, quant, compile=False, shared=True, warm=False, attn=attn, merge=merge)
        hd = Path(heads_dir) if heads_dir else Path(model_dir) / "exit_heads"
        ck = torch.load(hd / "heads.pt", map_location="cpu")
        th = json.loads((hd / "thresholds.json").read_text(encoding="utf-8"))
        self.policies = {k: {int(L): t for L, t in v["taus"].items() if t <= 1.0} for k, v in th["levels"].items()}
        self.policies["off"] = {}
        self.default_policy = default_policy
        self.exits = sorted({L for pol in self.policies.values() for L in pol})
        H = self.model.config.hidden_size
        self.heads = torch.nn.ModuleDict({str(L): ExitHead(self.model.model.norm, H, ck["rank"]) for L in ck["layers"]})
        self.heads.load_state_dict(ck["state"])
        self.heads = self.heads.to(self.dev).eval()
        n = len(self.model.model.layers)
        self.bounds = [0] + self.exits + [n]
        self.segs = [self._seg_fn(a, b, b == n) for a, b in zip(self.bounds, self.bounds[1:])]
        if compile:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
            self.segs = [torch.compile(f, dynamic=True) for f in self.segs]
        self.stats = {}
        for L in self.LENS:
            for b in self.BATCHES:
                if b * L <= self.TOKEN_BUDGET or b == 1:
                    self._graph_exit(b, L)

    def _seg_fn(self, a, b, last):
        mm = self.model.model

        @torch.no_grad()
        def f(h, mask, pos, cos, sin):
            for layer in mm.layers[a:b]:
                h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
                h = h[0] if isinstance(h, tuple) else h
            return mm.norm(h) if last else h
        return f

    def _graph_exit(self, b, L):
        key = ("e", b, L)
        if key in self.graphs:
            return self.graphs[key]
        mm = self.model.model
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(L, device=self.dev)[None].expand(b, L).contiguous()
        mask = self._mask_from_seg(torch.ones((b, L), dtype=torch.long, device=self.dev))

        @torch.no_grad()
        def emb():
            h = mm.embed_tokens(ids)
            cos, sin = mm.rotary_emb(h, pos)
            return h, cos, sin
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                h, cos, sin = emb()
                for f in self.segs:
                    h = f(h, mask, pos, cos, sin)
        torch.cuda.current_stream().wait_stream(st)
        graphs, outs = [], []
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            h, cos, sin = emb()
            h = self.segs[0](h, mask, pos, cos, sin)
        graphs.append(g); outs.append(h)
        for f in self.segs[1:]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                h = f(outs[-1], mask, pos, cos, sin)
            graphs.append(g); outs.append(h)
        self.graphs[key] = (graphs, ids, mask, pos, outs)
        return self.graphs[key]

    @torch.no_grad()
    def run(self, prompts, ids_list, policy=None):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)], policy)]

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        taus = self.policies[policy or self.default_policy]
        units = self._units(groups)
        res = [None] * len(units)
        for idx in self._chunks([len(u[0][0]) for u in units]):
            if len(units[idx[-1]][0][0]) > self.LENS[-1]:
                for k in idx:
                    res[k] = self._long(units[k][0], units[k][1])
                continue
            L = self._bucket(max(len(units[k][0][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            graphs, ids, mask, pos, outs = self._graph_exit(b, L)
            ri, ci = self._fill([u[0] for u in units], idx, b, L, ids, mask, pos)
            lids = [x for k in idx for x in units[k][1]]
            W, cols, pad = self._rows(lids)
            probs, depth = None, self.bounds[-1]
            for s, g in enumerate(graphs):
                g.replay()
                if s < len(self.exits) and self.exits[s] in taus:
                    Lx = self.exits[s]
                    z = self.heads[str(Lx)](outs[s][ri, ci]).to(W.dtype) @ W.T
                    p = torch.softmax(z.float().gather(1, cols).masked_fill(pad, float("-inf")), -1)
                    if float(p.max(-1).values.min()) >= taus[Lx]:
                        probs, depth = [row[: len(x)] for row, x in zip(p.tolist(), lids)], Lx
                        break
            if probs is None:
                probs = self._letters(outs[-1][ri, ci].float() @ W.T, cols, pad, lids)
            key = (policy or self.default_policy, depth)
            self.stats[key] = self.stats.get(key, 0) + len(idx)
            j = 0
            for k in idx:
                n = len(units[k][1]); res[k] = probs[j: j + n]; j += n
        return self._assemble(groups, units, res)


class MPSBackend(GraphBackend):
    """The shared-prefix packing (prefix trie, SOAM) and token-budget batching of GraphBackend, run as plain forwards
    without CUDA graphs: on the Apple GPU through PyTorch MPS, or on the CPU (no MPS, or device="cpu"), where a
    request's state is then computed once instead of once per prompt as in EagerBackend. Rows are padded on the right
    to the longest packed row of the chunk."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time

    def __init__(self, model_dir, dtype="bfloat16", shared=True, device=None):
        what = recurrent_layers(model_dir) if shared else None  # without sharing: EagerBackend's forwards (exact)
        if what:
            from .gdn_graph import is_gdn
            raise ValueError(recurrent_refused(model_dir, what, is_gdn(model_dir)))
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        device = device or ("mps" if mps else "cpu")
        EagerBackend.__init__(self, model_dir, dtype, device)
        self.dev = torch.device(device)
        self.shared = shared
        self._wrows = {}

    def _oversize(self, pack, n):
        """On the CPU (cascade_ok models) the branch pass runs in parts of whole top-level subtrees over one root
        (_cascade): a unit is split only when its prompts have no common root or one top-level subtree (a question with
        its option orders, ...) holds more than BRANCH_TOKENS tokens; splitting then moves its shared block into the
        root of the smaller units."""
        if self.dev.type != "cpu" or not cascade_ok(self.model.config):
            return GraphBackend._oversize(self, pack, n)
        top, size = [], {}  # the top-level subtree of every block (block 0 = the root); its tokens
        for b, p in enumerate(pack[4]):
            top.append(b if p <= 0 else top[p])
        for s in pack[2]:
            if s > 0:
                size[top[s]] = size.get(top[s], 0) + 1
        return pack[4].count(-1) > 1 or max(size.values(), default=0) > self.BRANCH_TOKENS

    @torch.no_grad()
    def run(self, prompts, ids_list):
        if not self.shared:
            return EagerBackend.run(self, prompts, ids_list)
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        if not self.shared:
            return EagerBackend.run_shared(self, groups)
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._chunks([len(u[0][0]) for u in units]):
            if len(units[idx[-1]][0][0]) > self.LENS[-1]:
                for k in idx:
                    probs[k] = self._long(units[k][0], units[k][1])
                continue
            L = max(len(units[k][0][0]) for k in idx)
            ids, pos, seg, rows, cols, parents = self._host_rows([u[0] for u in units], idx, len(idx), L)
            h = self._forward_masked(ids.to(self.dev), self._mask_from_seg(seg.to(self.dev), parents), pos.to(self.dev))
            pr = self._readout(h[rows, cols], [x for k in idx for x in units[k][1]])
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)


VLLM_PRESETS = {
    # --vllm-preset: repeatable. The weights are the checkpoint's own (bf16, or a ModelOpt FP8 / NVFP4 port): bf16
    # weights quantised to FP8 at load time cost basal-omni 0.5-9.5 points of automation at 1%
    "default": False,
    "deterministic": True,   # batch-invariant kernels, no prefix cache, no chunked prefill: answers independent of load
}


def vllm_args(cfg, preset="default", media=False, fast_prefill=True):
    """vLLM engine arguments and environment of a --vllm-preset for a model's config.json -> (kwargs, env).
    fast_prefill:
    models whose last layers reuse earlier layers' keys and values (Gemma 4 E2B / E4B: num_kv_shared_layers) run those
    layers at the answer position only during prefill (vLLM's kv_sharing_fast_prefill, YOCO); the generated token, the
    only one basal reads, is the same, prompt log-probabilities would not be. media (--media on): at most
    media.MAX_ITEMS items per prompt of the media the model takes, no video. A text-only server still loads the
    encoders: without them (language_model_only) vLLM 0.31 picks another attention backend for Gemma 4 on an H100
    (FlashAttention instead of Triton + FlashAttention) and keeps fewer KV tokens (mini: 911k instead of 978k)."""
    if preset not in VLLM_PRESETS:
        raise ValueError(f"unknown --vllm-preset {preset!r} (one of: {', '.join(VLLM_PRESETS)})")
    invariant = VLLM_PRESETS[preset]
    text = cfg.get("text_config") or cfg
    kw, env = {}, {}
    if invariant:
        # batch-invariant kernels without the prefix cache and without chunked prefill: with them, a request's numbers
        # depend on the states earlier requests left in the cache and on where the scheduler cut a long prompt (the
        # 31B FP8 port, 2,000 decisions at 32 in flight, two runs: 96.9% the same top answer with the prefix cache,
        # 99.95% without; basal-omni-1.1-max without it: every decision with 2-10 options bit-identical, 95.6% of
        # those with 11-127). Triton attention: vLLM's batch-invariant FlashAttention is FlashAttention 2, which takes
        # heads up to 256 wide (Gemma 4's global layers: 512; vLLM 0.31)
        env["VLLM_BATCH_INVARIANT"] = "1"
        kw.update(enable_prefix_caching=False, enable_chunked_prefill=False, attention_backend="TRITON_ATTN")
    if fast_prefill and text.get("num_kv_shared_layers"):
        kw["kv_sharing_fast_prefill"] = True
    if media:
        from .media import MAX_ITEMS, kinds_of
        kw["limit_mm_per_prompt"] = {k: MAX_ITEMS if k in kinds_of(cfg) else 0 for k in ("image", "audio", "video")}
    return kw, env


class VLLMBackend:
    """vLLM backend: bf16 checkpoints and the ModelOpt FP8 / NVFP4 ports, with the engine settings of a --vllm-preset
    (VLLM_PRESETS, vllm_args). One generated token restricted to the option letters; with
    logprobs_mode="processed_logprobs" the log-probabilities are computed after that restriction, so the softmax over
    the letters equals the readout of the other backends. Where vLLM supports it, logprob_token_ids asks for exactly
    the letter ids: vllm-metal ignores the logprobs mode and returns the top-k of the unrestricted vocabulary, which
    can miss a letter. vLLM's automatic prefix caching shares the state between the two option orders. max_options:
    the most option codes one readout asks for (vLLM refuses more log-probabilities than its max_logprobs, default 20,
    so it is raised to the model's prompt format limit: 255 for format 2). vLLM 0.31 accepts at most 128
    logprob_token_ids; a question with more options takes the top-k of the restricted distribution instead, which on
    CUDA (processed log-probabilities) are exactly the allowed codes."""

    LOGPROB_TOKEN_IDS = 128  # longest logprob_token_ids list vLLM accepts
    lock = threading.Lock()  # one vLLM call at a time (text batches and media requests); a class default for tests
    media_kinds, max_audio_s = (), None

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096, max_options=10, preset="default",
                 fast_prefill=True, media=False):
        cfg_file = Path(model_dir) / "config.json"
        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        kw, env = vllm_args(cfg, preset, media=media, fast_prefill=fast_prefill)
        if media:  # --media on: the media the model takes (basal/media.py), as EagerBackend
            from .media import audio_window, kinds_of
            self.media_kinds = kinds_of(cfg)
            if not self.media_kinds:
                raise SystemExit("--media on: the model has no image or audio encoder (config.json: vision_config, "
                                 "audio_config)")
            if "Gemma4UnifiedForConditionalGeneration" in (cfg.get("architectures") or ()):
                # basal-omni-1.1-max against --mode eager --media on (2.3 media check set): images 0.942 the same
                # answer, accuracy -1.3 points; 19 of 285 audio clips expanded to other token counts
                raise SystemExit("--media on with --mode vllm does not serve Gemma 4 Unified models (basal-omni-1.1-max): "
                                 "vLLM 0.31 reads their images and audio differently from the model's processor; use "
                                 "--mode eager --media on")
            pc = Path(model_dir) / "processor_config.json"
            self.max_audio_s = audio_window(json.loads(pc.read_text(encoding="utf-8"))) if pc.exists() else None
        os.environ.update(env)  # before vLLM starts its engine process, which reads it
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.preset, self.engine_args, self.lock = preset, kw, threading.Lock()
        from vllm import LLM
        self.llm = LLM(**{**dict(model=str(model_dir), dtype=dtype, gpu_memory_utilization=mem, max_model_len=max_len,
                                 enable_prefix_caching=True, logprobs_mode="processed_logprobs",
                                 max_logprobs=max(16, max_options)), **kw})
        self.context = max_len - 1  # longest prompt: max_model_len holds the one generated token too
        self.context_reason = f"vLLM's --max-len {max_len}"

    def run(self, prompts, ids_list):
        from vllm.inputs import TokensPrompt
        reqs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in EagerBackend.encode(self, prompts)]
        return [p for p, _ in self._generate(reqs, ids_list)]

    def _generate(self, reqs, ids_list):
        """One vLLM call -> [(softmax over the option codes, vLLM's output)]."""
        from vllm import SamplingParams
        import inspect
        exact = "logprob_token_ids" in inspect.signature(SamplingParams).parameters
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(ids), allowed_token_ids=list(ids),
                              **(dict(logprob_token_ids=list(ids)) if exact and len(ids) <= self.LOGPROB_TOKEN_IDS
                                 else {}))
               for ids in ids_list]
        with self.lock:
            outs = self.llm.generate(reqs, sps, use_tqdm=False)
        res = []
        for o, ids in zip(outs, ids_list):
            lp = o.outputs[0].logprobs[0]
            x = torch.tensor([lp[i].logprob if i in lp else -1e9 for i in ids], dtype=torch.float32)
            res.append((torch.softmax(x, -1).tolist(), o))
        return res

    def run_media(self, msgs_list, ids_list, images, audio, max_soft_tokens, max_tokens=None):
        """Prompts of one request with media, as EagerBackend.run_media reads them: the chat template's text with one
        placeholder per item, which vLLM's Gemma 4 processor expands with the Hugging Face processor (detail: the
        image's soft tokens). All prompts go into one vLLM call, so they share the request's media and state through
        the prefix cache: [(position probabilities, prompt tokens, media tokens)]."""
        from .media import SAMPLE_RATE
        mm = {k: v for k, v in (("image", list(images)), ("audio", [(a, SAMPLE_RATE) for a in audio])) if v}
        reqs = [{"prompt": self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False) + self.prefill,
                 "multi_modal_data": mm, "mm_processor_kwargs": {"max_soft_tokens": max_soft_tokens}}
                for msgs in msgs_list]
        try:
            outs = self._generate(reqs, ids_list)
        except ValueError as e:  # vLLM: "The decoder prompt (length N) is longer than the maximum model length ..."
            n = re.search(r"\(length (\d+)\) is longer than", str(e))
            if not n:
                raise
            raise RequestError(f"the prompt with its media is {n.group(1)} tokens, more than the {max_tokens or '?'} "
                               "tokens of the context; send fewer or shorter media, or detail \"default\"") from None
        media_ids = {t for t in (getattr(self.tok, "image_token_id", None), getattr(self.tok, "audio_token_id", None))
                     if t is not None}
        return [(p, len(o.prompt_token_ids), sum(t in media_ids for t in o.prompt_token_ids)) for p, o in outs]

    def run_shared(self, groups, policy=None):
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


class SGLangBackend:
    """SGLang offline engine (1.5). One generated token; the engine returns the log-probabilities of the option letters
    at the answer position (`token_ids_logprob`), and the softmax over them equals the readout of the other backends
    (a softmax over a subset of log-softmax values is the softmax over the same logits). RadixAttention shares the
    state between the option orders of one question and between the questions of one request."""

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096):
        import sglang
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.llm = sglang.Engine(model_path=str(model_dir), dtype=dtype, mem_fraction_static=mem,
                                 context_length=max_len, log_level="error")
        self.context = max_len - 1  # longest prompt: the context holds the one generated token too
        self.context_reason = f"SGLang's --max-len {max_len}"

    @staticmethod
    def _letters(meta, ids):
        """meta_info["output_token_ids_logprobs"][0] -> logprob per requested id ((logprob, token_id, text) tuples)."""
        got = {int(x[1]): float(x[0]) for x in meta["output_token_ids_logprobs"][0]}
        return [got.get(i, -1e9) for i in ids]

    def run(self, prompts, ids_list):
        enc = [list(ids) for ids in EagerBackend.encode(self, prompts)]
        outs = self.llm.generate(input_ids=enc, sampling_params=[{"max_new_tokens": 1, "temperature": 0.0}] * len(enc),
                                 return_logprob=True, logprob_start_len=-1,
                                 token_ids_logprob=[list(ids) for ids in ids_list])
        return [torch.softmax(torch.tensor(self._letters(o["meta_info"], ids), dtype=torch.float32), -1).tolist()
                for o, ids in zip(outs, ids_list)]

    def run_shared(self, groups, policy=None):
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


def _split(flat, groups):
    out, j = [], 0
    for g in groups:
        out.append(flat[j: j + len(g[0])]); j += len(g[0])
    return out


def mlx_config_overrides(cfg):
    """transformers 5 writes the rotary settings as `rope_parameters`; mlx-lm reads `rope_theta` / `rope_scaling` only
    (without this override it silently uses rope_theta = 10000). Gemma 4 (basal-omni): transformers 5 writes the global
    layers' head size (and KV heads) in `per_layer_config`; mlx-lm reads `global_head_dim` /
    `num_global_key_value_heads` only (without, it assumes 512). A multimodal config gets its `text_config` back whole."""
    if isinstance(cfg.get("text_config"), dict):
        t = mlx_config_overrides(cfg["text_config"])
        return {"text_config": {**cfg["text_config"], **t}} if t else {}
    rp = cfg.get("rope_parameters") or {}
    out = {}
    if "rope_theta" not in cfg and "rope_theta" in rp:
        out["rope_theta"] = rp["rope_theta"]
    if not cfg.get("rope_scaling") and rp.get("rope_type", "default") != "default":
        out["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
    types = cfg.get("layer_types") or []
    full = [c for i, c in (cfg.get("per_layer_config") or {}).items()
            if int(i) < len(types) and types[int(i)] == "full_attention"]
    for key, name in (("global_head_dim", "head_dim"), ("num_global_key_value_heads", "num_key_value_heads")):
        vals = {c[name] for c in full if name in c}
        if cfg.get(key) is None and len(vals) == 1:
            out[key] = vals.pop()
    return out


class _PackedRow:
    """mlx-lm's cache interface for the packed rows of MLXBackend._packed: one position per token (offset [B, L], read
    by the wrapped RoPE) and the block mask; nothing is stored (keys and values pass through)."""

    def __init__(self, pos, mask):
        self.offset, self.mask = pos, mask

    def make_mask(self, n, return_array=False, window_size=None):
        return self.mask

    def update_and_fetch(self, keys, values):
        return keys, values


class MLXBackend:
    """Apple Silicon (MLX / mlx-lm): the MLX exports (the -MLX-8bit / -MLX-fp4 repositories) or a bf16 Hugging Face
    directory. Same prompts, tokens and letter readout as the PyTorch backends. The shared prefix of a group (both
    option orders of a question; with SOAM every question and order of a request, i.e. the state) is computed once into
    a KV cache, and the remaining suffixes run as one right-padded batch continuing that cache (the default). With
    packed=True (basal-serve / basal-bench --mlx-packed, opt-in), a group of a Llama decoder (basal-1.x, 2.x) with a
    static RoPE (default, linear, llama3) whose prefix and all suffixes fit in PACK_TOKENS runs instead as one packed
    row (the prefix, then every suffix, with explicit positions and a block mask), several groups per forward; other
    groups (Gemma 4, long groups, sequence-length-dependent RoPE) keep the KV cache. Either way a suffix sees the prefix
    and its own earlier tokens only, so every readout equals a separate forward of its prompt; the packed row sums in a
    different order, so in bf16 / quantised weights its probabilities differ slightly from the KV-cache path (identical
    decisions in fp32) and the shipped CALIBRATION.mlx.json was fitted on the KV-cache path. All MLX work runs on one
    dedicated thread (the server calls the backend from an executor)."""

    MAX_ROWS = 16      # suffix rows per forward
    TOKEN_BUDGET = 8192  # rows x padded length per forward without a shared prefix
    PACK_TOKENS = 512  # longest packed row (prefix + all suffixes of a group); longer groups use the KV cache (above
    # about 600 tokens a forward is compute-bound and the packed row's dense mask makes it no faster, M5 Max)
    PACK_BUDGET = 2048  # packed rows x padded length per forward: on an M4 Pro 16 rows of ~250 tokens in one forward
    # (TOKEN_BUDGET) gave 9-11% fewer decisions/s than chunks of 2,048 tokens (44 bench examples, two orders)
    CACHE_BYTES = 2 << 30  # MLX buffer cache limit (and wired headroom above the weights)

    def __init__(self, model_dir, max_rows=None, quant=None, packed=False):
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.prefill = PREFILL
        self.max_rows = max_rows or self.MAX_ROWS
        self.pool.submit(self._load, Path(model_dir), quant, packed).result()

    def _load(self, md, quant=None, packed=False):
        """quant: None, or "q8": MLX affine 8-bit weights (group size 64) of the decoder blocks' linear layers, made at
        load time from bf16 weights (the embeddings and the LM head stay bf16); the -MLX-8bit exports need no quant."""
        import mlx.core as mx
        from mlx_lm.utils import load_model
        self.mx = mx
        cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
        if quant not in (None, "q8"):
            raise ValueError(f"unknown MLX quantisation {quant!r} (q8)")
        # lazy for q8: the weights are materialised once, after quantisation (never the full bf16 copy)
        self.model, self.config = load_model(md, lazy=bool(quant), model_config=mlx_config_overrides(cfg))
        # the final logit softcap of the decoder (Gemma) is applied to the fp32 code logits
        self.softcap = getattr(self.lm.args, "final_logit_softcapping", None)
        self.quantization = self.config.get("quantization")
        if quant:
            if self.quantization:
                raise ValueError(f"{md} is already quantised ({self.quantization}); use it without --quant")
            import mlx.nn as nn
            nn.quantize(self.model, group_size=64, bits=8,
                        class_predicate=lambda path, m: isinstance(m, nn.Linear)
                        and path.removeprefix("language_model.").startswith("model.layers."))
            mx.eval(self.model.parameters())
            self.quantization = {"group_size": 64, "bits": 8}
        # keep the weights resident (wired: under memory pressure macOS would page them out and every forward would
        # wait for them) and the buffer cache small (by default it grows towards the whole working set, one entry per
        # new shape, and on a 16-24 GB Mac pushes everything else into swap)
        from mlx.utils import tree_flatten
        weights = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        if mx.metal.is_available():
            rec = mx.device_info().get("max_recommended_working_set_size", weights)
            mx.set_wired_limit(min(rec, weights + self.CACHE_BYTES))
        mx.set_cache_limit(self.CACHE_BYTES)
        self.packed = bool(packed) and self._position_rope()

    @staticmethod
    def _static_rope(rope):
        """RoPE layers whose frequencies do not depend on the sequence length (mlx-lm: default and linear = nn.RoPE,
        llama3), so that rotating each token as its own length-1 row equals the contiguous run. Not packed: dynamic NTK
        (its base grows with the length of the call), longrope / su, yarn and anything else."""
        import mlx.nn as nn
        from mlx_lm.models.rope_utils import Llama3RoPE
        return type(rope) in (nn.RoPE, Llama3RoPE)

    def _position_rope(self):
        """Llama decoders without sliding windows and with a static RoPE (_static_rope): wraps each layer's RoPE so that an offset of shape [B, L] (one
        position per token, _PackedRow) rotates every token at its own position, each token as its own row of the same
        mx.fast.rope (bit-identical to a contiguous run); any other offset (the KV cache path) is passed on. Returns
        whether the packed rows can be used."""
        import mlx.nn as nn
        from mlx_lm.models import llama
        if not isinstance(self.lm, llama.Model) or self.lm.model.swa_idx is not None:
            return False
        if not all(self._static_rope(layer.self_attn.rope) for layer in self.lm.model.layers):
            return False  # the KV-cache path

        class PositionRoPE(nn.Module):
            def __init__(self, rope):
                super().__init__()
                self.rope = rope

            def __call__(self, x, offset=0):
                if getattr(offset, "ndim", 0) != 2:
                    return self.rope(x, offset=offset)
                B, H, L, D = x.shape
                y = self.rope(x.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D), offset=offset.reshape(-1))
                return y.reshape(B, L, H, D).transpose(0, 2, 1, 3)

        for layer in self.lm.model.layers:
            layer.self_attn.rope = PositionRoPE(layer.self_attn.rope)
        return True

    def encode(self, prompts):
        return EagerBackend.encode(self, prompts)

    @property
    def lm(self):
        """The decoder: the model itself (Llama, Bielik) or the text model of a multimodal wrapper (Gemma 4,
        basal-omni; mlx-lm drops its image / audio towers)."""
        return getattr(self.model, "language_model", self.model)

    def _head(self, h):
        m = self.lm
        return m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)

    def _code_rows(self, key):
        """fp32 rows of the output head for the code ids in key (cached), as GraphBackend._rows: the code logits are
        computed in fp32 from the hidden state (bf16 logits of magnitude 16-32 step by 0.125, which clumps the
        probabilities). None for a quantised head (its logits come from the head itself)."""
        rows = self.__dict__.setdefault("_wrows", {})
        if key not in rows:
            m = self.lm
            head = m.model.embed_tokens if m.args.tie_word_embeddings else m.lm_head
            rows[key] = None if hasattr(head, "scales") else head.weight[self.mx.array(key)].astype(self.mx.float32)
        return rows[key]

    def _letters(self, h, lids):
        """Hidden states [n, d] at the readout positions -> softmax over each readout's codes (float32)."""
        mx = self.mx
        key = tuple(sorted({i for ids in lids for i in ids}))
        W = self._code_rows(key)
        if W is None:
            z, col = self._head(h).astype(mx.float32), None
        else:
            z, col = h.astype(mx.float32) @ W.T, {t: j for j, t in enumerate(key)}
        if self.softcap:
            z = mx.tanh(z / self.softcap) * self.softcap
        K = max(map(len, lids))
        idx = mx.array([[col[i] if col else i for i in ids] + [col[ids[0]] if col else ids[0]] * (K - len(ids))
                        for ids in lids])
        pad = mx.array([[False] * len(ids) + [True] * (K - len(ids)) for ids in lids])
        z = mx.where(pad, -mx.inf, mx.take_along_axis(z, idx, axis=1))
        p = mx.softmax(z, axis=-1).tolist()
        return [row[: len(ids)] for row, ids in zip(p, lids)]

    def _forward(self, rows, prefix=None):
        """Token rows (right-padded into one batch) -> hidden state at each row's last token. `prefix`: per-layer
        (keys, values) of a shared prefix that every row continues (positions start after it)."""
        mx = self.mx
        from mlx_lm.models.cache import KVCache
        B, S = len(rows), max(map(len, rows))
        pad = self.tok.pad_token_id
        ids = mx.array([list(r) + [pad] * (S - len(r)) for r in rows])
        cache = None
        if prefix is not None:
            cache = []
            for k, v in prefix:
                c = KVCache()
                c.keys, c.values, c.offset = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0), k.shape[2]
                cache.append(c)
        h = self.lm.model(ids, cache=cache)
        return h[mx.arange(B), mx.array([len(r) - 1 for r in rows])]

    def _prefix(self, toks):
        mx = self.mx
        from mlx_lm.models.cache import KVCache, make_prompt_cache
        # plain caches: a sliding-window layer (Gemma 4) gets a rotating cache from mlx-lm, which keeps only the last
        # window of the prefix; here every layer keeps the whole prefix and the window is applied by the mask
        cache = [KVCache() for _ in make_prompt_cache(self.model)]
        self.lm.model(mx.array([toks]), cache=cache)
        kv = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache]
        mx.eval(kv)
        return kv

    @staticmethod
    def _shared_len(toks):
        """Tokens of the prefix shared by all prompts of a group (each prompt keeps at least its last token)."""
        return min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1)

    def _packed(self, groups):
        """Groups -> hidden states at every prompt's last token, one forward: each group is one row (its shared prefix
        once, then every suffix; each suffix's positions continue from the end of the prefix, and the mask lets it see
        the prefix and its own earlier tokens only), the rows right-padded into one batch."""
        mx = self.mx
        rows = []
        for toks in groups:
            P = self._shared_len(toks)
            ids, pos, seg, last = list(toks[0][:P]), list(range(P)), [0] * P, []
            for s, t in enumerate(toks, 1):
                ids += t[P:]; pos += range(P, len(t)); seg += [s] * (len(t) - P); last.append(len(ids) - 1)
            rows.append((ids, pos, seg, last))
        L, pad = max(len(r[0]) for r in rows), self.tok.pad_token_id
        ids = mx.array([r[0] + [pad] * (L - len(r[0])) for r in rows])
        pos = mx.array([r[1] + [0] * (L - len(r[1])) for r in rows])
        seg = mx.array([r[2] + [-1] * (L - len(r[2])) for r in rows])  # -1: padding (sees itself only)
        si, sj = seg[:, :, None], seg[:, None, :]
        mask = mx.tril(mx.ones((L, L), dtype=mx.bool_)) & (sj >= 0) & ((sj == 0) | (sj == si))
        mask = mask | mx.eye(L, dtype=mx.bool_)
        h = self.lm.model(ids, cache=[_PackedRow(pos, mask[:, None])] * len(self.lm.model.layers))
        return h[mx.array([b for b, r in enumerate(rows) for _ in r[3]]), mx.array([j for r in rows for j in r[3]])]

    def _group(self, toks, lids):
        """One group: shared prefix once, then the suffixes in batches of max_rows."""
        P = self._shared_len(toks)
        prefix = self._prefix(toks[0][:P]) if P > 0 else None
        out = []
        for i in range(0, len(toks), self.max_rows):
            h = self._forward([t[P:] for t in toks[i: i + self.max_rows]], prefix)
            out += self._letters(h, lids[i: i + self.max_rows])
        return out

    def _chunks(self, lens, budget=None):
        """Indices of rows of these lengths, sorted by length, in right-padded batches of at most max_rows rows and
        `budget` (default TOKEN_BUDGET) tokens."""
        chunks, cur, budget = [], [], budget or self.TOKEN_BUDGET
        for k in sorted(range(len(lens)), key=lambda k: lens[k]):
            if cur and (len(cur) >= self.max_rows or (len(cur) + 1) * lens[k] > budget):
                chunks.append(cur); cur = []
            cur.append(k)
        return chunks + [cur]

    def _singles(self, toks, lids):
        """Unrelated prompts: right-padded batches of similar length, no cache."""
        res = [None] * len(toks)
        for c in self._chunks([len(t) for t in toks]):
            for j, p in zip(c, self._letters(self._forward([toks[j] for j in c]), [lids[j] for j in c])):
                res[j] = p
        return res

    def _run_shared(self, groups):
        enc = [(self.encode(p), x) for p, x in groups]
        out = [None] * len(groups)
        single = [k for k, (t, _) in enumerate(enc) if len(t) == 1]
        if single:
            for k, p in zip(single, self._singles([enc[k][0][0] for k in single], [enc[k][1][0] for k in single])):
                out[k] = [p]
        if self.packed:  # packed row length: the prefix once, then every suffix
            n = {k: sum(map(len, t)) - (len(t) - 1) * self._shared_len(t)
                 for k, (t, _) in enumerate(enc) if out[k] is None}
            fit = [k for k in n if n[k] <= self.PACK_TOKENS]
            for c in self._chunks([n[k] for k in fit], self.PACK_BUDGET) if fit else []:
                ks = [fit[j] for j in c]
                probs = self._letters(self._packed([enc[k][0] for k in ks]), [x for k in ks for x in enc[k][1]])
                for k in ks:
                    out[k], probs = probs[: len(enc[k][1])], probs[len(enc[k][1]):]
        for k, (t, x) in enumerate(enc):
            if out[k] is None:
                out[k] = self._group(t, x)
        return out

    def run_shared(self, groups, policy=None):
        return self.pool.submit(self._run_shared, groups).result()

    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]


class GGUFBackend(GraphBackend):
    """A GGUF file in process through llama.cpp (llama-cpp-python: Metal on Apple Silicon, CUDA or CPU elsewhere). Only
    the weights come from the GGUF file; tokenizer, chat template and CALIBRATION.json come from the Hugging Face model
    directory, so the token ids are exactly those of the other backends. The packed prefix trie of a group (as in
    GraphBackend: the state once, every question and option order after it) becomes llama.cpp sequences: every readout
    has its own sequence and every token belongs to the sequences of all readouts below its block, so a shared prefix
    is computed once and each readout sees its own prompt only. All rows of a chunk go through one llama_decode
    (unified KV cache, cleared per chunk).

    Hybrid and recurrent architectures (Qwen3.5's Gated DeltaNet layers: GGUF architecture qwen35; Mamba, RWKV) keep
    a recurrent state per sequence, which a token shared by several sequences cannot have; on the CPU llama.cpp
    returned non-finite logits whenever several sequences shared one llama_decode (the 0.8B pico's Q8_0: 300 of 300
    held-out prompts at 16 per decode, 0 of 300 at one; llama-cpp-python 0.3.36, 22 Oct). Such a model decodes one prompt
    per llama_decode, from its first token (N_SEQ = 1: every group is split into single prompts), so a request's
    state is computed once per question and option order."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time

    N_CTX = 16384  # KV cells per chunk: room for a single long prompt
    context, context_reason = N_CTX, f"the in-process llama.cpp context ({N_CTX} cells)"
    N_SEQ = 64     # readouts (sequences) per chunk
    # GGUF architectures with recurrent-state layers (beside llama.cpp's own is_recurrent / is_hybrid flags)
    RECURRENT_ARCHS = ("qwen35", "qwen3next", "mamba", "rwkv", "jamba", "falcon-h1", "plamo2", "granitehybrid",
                       "nemotron_h", "lfm2", "kimi-linear")

    def __init__(self, model_dir, gguf, n_gpu_layers=-1):
        import logging

        import llama_cpp as C
        logging.getLogger("llama-cpp-python").setLevel(logging.ERROR)  # llama.cpp load / Metal info lines
        self.C = C
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.shared = True
        C.llama_backend_init()
        mp = C.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        self.cmodel = C.llama_model_load_from_file(str(gguf).encode(), mp)
        if not self.cmodel:
            raise ValueError(f"llama.cpp could not load {gguf}")
        self.arch = self._meta("general.architecture") or ""
        self.recurrent = bool(C.llama_model_is_recurrent(self.cmodel) or C.llama_model_is_hybrid(self.cmodel)
                              or self.arch.startswith(self.RECURRENT_ARCHS)
                              or self._meta(f"{self.arch}.ssm.state_size"))
        if self.recurrent:
            self.N_SEQ = 1
        # a full model directory: its architecture must match the GGUF file; the -GGUF repositories carry tokenizer and
        # calibration without config.json (their .gguf files belong to them)
        cp = Path(model_dir) / "config.json"
        cfg = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
        cfg = cfg.get("text_config") or cfg  # Gemma 4 (basal-omni): the decoder's settings are in text_config
        for name, llama_name in (("num_hidden_layers", "llama_model_n_layer"), ("hidden_size", "llama_model_n_embd")):
            if not cfg:
                break
            expected = cfg.get(name)
            if expected is None:
                raise ValueError(f"{model_dir}/config.json is missing {name}")
            actual = getattr(C, llama_name)(self.cmodel)
            if actual != expected:
                raise ValueError(f"{gguf}: {llama_name.removeprefix('llama_model_')} is {actual}, "
                                 f"but {model_dir}/config.json specifies {name}={expected}")
        self.n_vocab = C.llama_vocab_n_tokens(C.llama_model_get_vocab(self.cmodel))
        expected_vocab = cfg.get("vocab_size")
        if expected_vocab is not None and self.n_vocab != expected_vocab:
            raise ValueError(f"{gguf}: vocabulary is {self.n_vocab}, "
                             f"but {model_dir}/config.json specifies vocab_size={expected_vocab}")
        if self.n_vocab < len(self.tok):
            raise ValueError(f"{gguf}: vocabulary of {self.n_vocab} tokens, tokenizer of {model_dir} has {len(self.tok)}")
        cp = C.llama_context_default_params()
        cp.n_ctx = cp.n_batch = self.N_CTX
        cp.n_ubatch = 512
        cp.n_seq_max = self.N_SEQ
        cp.kv_unified = True  # sequences share the cells of their common prefix
        cp.swa_full = True  # sliding-window layers (Gemma 4) keep every cell: a shared prefix cell stays readable
        # llama.cpp's own default is 4 threads (GGML_DEFAULT_N_THREADS); PyTorch's count is the physical cores (or
        # OMP_NUM_THREADS / torch.set_num_threads). Each output element stays one dot product in one thread: same logits.
        cp.n_threads = cp.n_threads_batch = torch.get_num_threads()
        self.ctx = C.llama_init_from_model(self.cmodel, cp)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self.batch = C.llama_batch_init(self.N_CTX, 0, self.N_SEQ)  # a prefix token belongs to every readout below it

    def _meta(self, key):
        """A metadata value of the GGUF file as a string, None if absent."""
        import ctypes
        buf = ctypes.create_string_buffer(256)
        n = self.C.llama_model_meta_val_str(self.cmodel, key.encode(), buf, ctypes.sizeof(buf))
        return buf.value.decode() if n >= 0 else None

    def __del__(self):
        C = getattr(self, "C", None)
        if C is None:
            return
        if getattr(self, "batch", None) is not None:
            C.llama_batch_free(self.batch)
        if getattr(self, "ctx", None):
            C.llama_free(self.ctx)
        if getattr(self, "cmodel", None):
            C.llama_model_free(self.cmodel)

    def _oversize(self, pack, n):
        return len(pack[0]) > self.N_CTX or n > self.N_SEQ

    def _decode_chunks(self, units):
        """Units sorted by length, cut into chunks of at most N_CTX tokens and N_SEQ readouts."""
        out, cur, n_tok, n_seq = [], [], 0, 0
        for k in sorted(range(len(units)), key=lambda k: len(units[k][0][0])):
            t, r = len(units[k][0][0]), len(units[k][1])
            if t > self.N_CTX:
                raise RequestError(f"prompt too long for the GGUF backend: {t} tokens (max {self.N_CTX})")
            if cur and (n_tok + t > self.N_CTX or n_seq + r > self.N_SEQ):
                out.append(cur); cur, n_tok, n_seq = [], 0, 0
            cur.append(k); n_tok += t; n_seq += r
        return out + ([cur] if cur else [])

    def _decode(self, packs, idx):
        """One llama_decode for the packed rows idx -> readout logits [n_readouts, n_vocab] (float32)."""
        C, b = self.C, self.batch
        n, base, reads = 0, 0, []
        for k in idx:
            t, pos, seg, last, parent = packs[k]
            below = [[] for _ in parent]  # sequences (readouts) below every block of the trie
            for r, i in enumerate(last):
                blk = seg[i]
                while blk >= 0:
                    below[blk].append(base + r); blk = parent[blk]
            for tok, p, s in zip(t, pos, seg):
                b.token[n], b.pos[n], b.logits[n] = tok, p, 0
                b.n_seq_id[n] = len(below[s])
                for j, q in enumerate(below[s]):
                    b.seq_id[n][j] = q
                n += 1
            for i in last:
                b.logits[n - len(t) + i] = 1
                reads.append(n - len(t) + i)
            base += len(last)
        b.n_tokens = n
        C.llama_memory_clear(C.llama_get_memory(self.ctx), False)
        if C.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        return np.stack([np.ctypeslib.as_array(C.llama_get_logits_ith(self.ctx, r), shape=(self.n_vocab,))
                         for r in reads])

    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    def run_shared(self, groups, policy=None):
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._decode_chunks(units):
            lg = self._decode([u[0] for u in units], idx)
            pr = []
            for m, ids in enumerate(x for k in idx for x in units[k][1]):
                z = lg[m, list(ids)].astype(np.float64)
                e = np.exp(z - z.max())
                pr.append((e / e.sum()).tolist())
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)


class _HTTPLetters:
    """Shared part of the HTTP backends (Ollama, llama.cpp server): one request per prompt (one generated token with
    the top log-probabilities at the answer position), sent sequentially so that the engine's prompt cache reuses the
    state between the option orders and the questions of a request; `parallel` > 1 sends that many at once (for an
    engine started with several slots / OLLAMA_NUM_PARALLEL)."""

    TOP = 20  # top log-probabilities requested; the option letters dominate the answer position
    RETRIES = 3  # dropped connections retried per request
    max_codes = None  # most options per question the engine can read (None: no limit of its own)
    context = None  # longest prompt in tokens the engine takes (None: unknown; the model's context applies)

    def __init__(self, model_dir, url, parallel=1, timeout=120.0):
        import httpx
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        self.parallel = parallel
        self.pool = None
        if parallel > 1:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=parallel)

    def _post(self, path, body):
        """POST with a few retries on dropped connections (an engine restarting or closing an idle keep-alive)."""
        import time

        import httpx
        for k in range(self.RETRIES + 1):
            try:
                r = self.http.post(path, json=body)
                r.raise_for_status()
                return r.json()
            except httpx.TransportError:  # connection refused / reset, server disconnected
                if k == self.RETRIES:
                    raise
                time.sleep(0.5 * 2 ** k)

    def run(self, prompts, ids_list):
        jobs = list(zip(prompts, ids_list))
        lps = list(self.pool.map(lambda j: self._one(*j), jobs)) if self.pool else [self._one(*j) for j in jobs]
        return [torch.softmax(torch.tensor(x, dtype=torch.float32), -1).tolist() for x in lps]

    def run_shared(self, groups, policy=None):
        return _split(self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]]), groups)

    def close(self):
        self.http.close()


class OllamaBackend(_HTTPLetters):
    """Ollama (GGUF exports: the -GGUF repositories; Apple Silicon, CPU, CUDA). The prompt is rendered and
    calibrated here (the model directory gives the tokenizer, chat template and CALIBRATION.json) and sent as a raw
    prompt (`raw: true`: no Ollama template) to /api/generate with one generated token and `logprobs` /
    `top_logprobs`; Ollama tokenizes it with the GGUF vocabulary. Letters missing from the top list get -1e9 (their
    log-probability is below the TOP-th token, i.e. practically zero next to the letters that are listed)."""

    takes_text = True  # Ollama accepts text only; the server skips its own tokenization
    max_codes = 20  # Ollama refuses top_logprobs > 20, so a question can have at most 20 options
    max_codes_reason = "Ollama returns at most 20 top log-probabilities per position; use --mode gguf or llamacpp"

    def __init__(self, model_dir, url="http://127.0.0.1:11434", model=None, parallel=1, num_ctx=4096, keep_alive="30m"):
        super().__init__(model_dir, url, parallel)
        if not model:
            raise ValueError("--ollama-model is required for --mode ollama")
        self.name, self.num_ctx, self.keep_alive = model, num_ctx, keep_alive
        self.letter_text = {}
        self.checked = False
        self.context = num_ctx - 1  # Ollama truncates a longer prompt silently; num_ctx holds the generated token
        self.context_reason = f"Ollama's num_ctx {num_ctx}"

    def _raw(self, prompt):
        """The rendered prompt starts with the BOS text; the GGUF exports set add_bos_token, so Ollama adds BOS itself
        (sending it as text too would give two)."""
        bos = self.tok.bos_token
        return prompt[len(bos):] if bos and prompt.startswith(bos) else prompt

    def _text(self, i):
        if i not in self.letter_text:
            self.letter_text[i] = self.tok.decode([i])
        return self.letter_text[i]

    def body(self, prompt):
        return {"model": self.name, "prompt": prompt, "raw": True, "stream": False, "logprobs": True,
                "top_logprobs": self.TOP, "keep_alive": self.keep_alive,
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx}}

    @staticmethod
    def parse(resp, letters):
        """/api/generate response -> log-probability of each letter text at the first generated position."""
        first = resp["logprobs"][0]
        got = {x["token"]: float(x["logprob"]) for x in first.get("top_logprobs") or []}
        got.setdefault(first["token"], float(first["logprob"]))
        return [got.get(t, -1e9) for t in letters]

    def _one(self, prompt, ids):
        if not isinstance(prompt, str):
            prompt = self.tok.decode(prompt)
        out = self._post("/api/generate", self.body(self._raw(prompt)))
        if not self.checked:  # first request: Ollama's token count must equal ours (same tokenizer, one BOS)
            self.checked = True
            n, got = len(self.tok(prompt, add_special_tokens=False).input_ids), out.get("prompt_eval_count")
            if got is not None and got != n:
                import warnings
                warnings.warn(f"Ollama evaluated {got} prompt tokens, the basal tokenizer gives {n}: the GGUF vocabulary "
                              "differs from the model's tokenizer (use the GGUF files of the -GGUF repositories)")
        return self.parse(out, [self._text(i) for i in ids])


class LlamaCppBackend(_HTTPLetters):
    """llama.cpp server (`llama-server -m model.gguf`): the same readout from /completion with the prompt given as
    token ids (exactly the server's tokens) and `n_probs`; the log-probabilities are those of the raw logits
    (`post_sampling_probs` off). The prompt cache (`cache_prompt`) shares the state between requests of one slot.
    llama.cpp has no cap on `n_probs` below the vocabulary, so a question with k options asks for max(20, k + 10)."""

    def __init__(self, model_dir, url="http://127.0.0.1:8080", parallel=1, timeout=120.0):
        super().__init__(model_dir, url, parallel, timeout)
        n = self.slot_context()
        if n:
            self.context, self.context_reason = n - 1, f"llama-server's context per slot ({n} tokens)"

    def slot_context(self):
        """The context of one slot from GET /props (llama-server -c divided by -np), None for a server without it."""
        try:
            props = self.http.get("/props").json()
        except Exception:  # noqa: BLE001  (an older server, or one behind a proxy: the model's context applies)
            return None
        n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
        return int(n) if n else None

    def body(self, ids, k=0):
        return {"prompt": list(ids), "n_predict": 1, "n_probs": max(self.TOP, k + 10), "temperature": 0.0,
                "cache_prompt": True, "post_sampling_probs": False, "return_tokens": False}

    @staticmethod
    def parse(resp, ids):
        first = resp["completion_probabilities"][0]
        got = {int(x["id"]): float(x["logprob"]) for x in first.get("top_logprobs") or []}
        got.setdefault(int(first["id"]), float(first["logprob"]))
        return [got.get(int(i), -1e9) for i in ids]

    def _one(self, prompt, ids):
        toks = prompt if not isinstance(prompt, str) else self.tok(prompt, add_special_tokens=False).input_ids
        return self.parse(self._post("/completion", self.body(toks, len(ids))), ids)


def modelopt_fp8(model_dir):
    """True for a ModelOpt FP8 checkpoint (the -FP8 repositories): FP8 linear weights with static per-tensor weight and
    input scales."""
    p = Path(model_dir) / "config.json"
    q = (json.loads(p.read_text(encoding="utf-8")).get("quantization_config") or {}) if p.exists() else {}
    return q.get("quant_method") == "modelopt" and q.get("quant_algo") == "FP8"


def _fp8_cast_ops(x, scale):
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


_FP8_CAST = []
# FP8 GEMMs use cuBLAS's fast (reduced-precision) accumulation, whose rounding depends on the kernel cuBLAS picks for
# the shape (batch) and on deterministic algorithms: a 4.5B FP8 checkpoint on a dev set scored 0.6915 with
# --deterministic, 0.6969 batched, 0.6978 one request at a time. --deterministic turns it off (set_fp8_fast_accum):
# full accumulation gives 0.6958 with and without batching and the same decisions in both
FP8_FAST_ACCUM = [True]


def set_fp8_fast_accum(on):
    FP8_FAST_ACCUM[0] = bool(on)


def fp8_cast(x, scale):
    """x / scale as saturated FP8 e4m3. On CUDA one fused kernel (compiled once, any shape), also where the model runs
    uncompiled (pass 2): there the three separate ops of every linear layer were most of the work."""
    if not x.is_cuda:
        return _fp8_cast_ops(x, scale)
    if not _FP8_CAST:
        _FP8_CAST.append(torch.compile(_fp8_cast_ops, dynamic=True))
    return _FP8_CAST[0](x, scale)


class FP8Linear(torch.nn.Module):
    """A linear layer of a ModelOpt FP8 checkpoint: the FP8 weight and its per-tensor scale, the input cast to FP8
    with the checkpoint's static per-tensor input scale, and one FP8 GEMM (torch._scaled_mm, bias fused, bf16 out);
    the same arithmetic as vLLM's kernels for these checkpoints. Off CUDA the product is emulated in fp32."""

    def __init__(self, weight, weight_scale, input_scale, bias=None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.float().reshape(()))
        self.register_buffer("input_scale", input_scale.float().reshape(()))
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        shape, dt = x.shape, x.dtype
        x = x.reshape(-1, self.in_features)
        q = fp8_cast(x, self.input_scale)
        if x.is_cuda:
            m = q.shape[0]
            if m % 16:  # FP8 GEMM operands: rows padded to a multiple of 16
                q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, -m % 16)).view(torch.float8_e4m3fn)
            y = torch._scaled_mm(q, self.weight.t(), scale_a=self.input_scale, scale_b=self.weight_scale,
                                 bias=None if self.bias is None else self.bias.to(dt), out_dtype=dt,
                                 use_fast_accum=FP8_FAST_ACCUM[0])
            y = y[:m]
        else:
            y = (q.float() * self.input_scale) @ (self.weight.float() * self.weight_scale).t()
            y = (y if self.bias is None else y + self.bias.float()).to(dt)
        return y.reshape(*shape[:-1], self.out_features)


def merged_linear(mods):
    """One linear for several that read the same input (q / k / v, gate / up): bf16 weights and biases concatenated,
    the same products in one GEMM. FP8: the merged weight re-quantised with one per-tensor scale (its amax / 448) and
    the largest of the input scales (they are equal when calibrated on the same activation)."""
    bias = None if mods[0].bias is None else torch.cat([m.bias.detach() for m in mods])
    if isinstance(mods[0], FP8Linear):
        w = torch.cat([m.weight.float() * m.weight_scale for m in mods])
        s = (w.abs().amax() / 448.0).clamp_min(1e-12)
        q = (w / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return FP8Linear(q, s, torch.stack([m.input_scale for m in mods]).max(), bias)
    w = torch.cat([m.weight.detach() for m in mods])
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device="meta", dtype=w.dtype)
    lin.weight = torch.nn.Parameter(w, requires_grad=False)
    if bias is not None:
        lin.bias = torch.nn.Parameter(bias, requires_grad=False)
    return lin


def _merged_attention(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                      **kwargs):
    """transformers' LlamaAttention.forward with one q / k / v projection."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, k, v = self.qkv_proj(hidden_states).split(self.qkv_sizes, -1)
    query_states, key_states = q.view(hidden_shape).transpose(1, 2), k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attention = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    attn_output, attn_weights = attention(self, query_states, key_states, value_states, attention_mask,
                                          dropout=0.0 if not self.training else self.attention_dropout,
                                          scaling=self.scaling, **kwargs)
    return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights


def _merged_mlp(self, x):
    """transformers' LlamaMLP.forward with one gate / up projection."""
    gate, up = self.gate_up_proj(x).chunk(2, -1)
    return self.down_proj(self.act_fn(gate) * up)


def merge_projections(model):
    """Merge every Llama layer's q / k / v and gate / up projections into one linear each (merged_linear): five GEMMs per
    layer instead of seven. Layers of another architecture, or whose projections differ in kind (FP8 and bf16), keep
    theirs. -> the number of merged layers."""
    import types
    n = 0
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        if type(a).__name__ != "LlamaAttention" or type(m).__name__ != "LlamaMLP":
            continue
        qkv, gu = [a.q_proj, a.k_proj, a.v_proj], [m.gate_proj, m.up_proj]
        if len({type(x) for x in qkv + gu}) != 1 or len({x.bias is None for x in qkv}) != 1 \
                or len({x.bias is None for x in gu}) != 1:
            continue
        a.qkv_sizes = [x.out_features for x in qkv]
        a.qkv_proj = merged_linear(qkv)
        del a.q_proj, a.k_proj, a.v_proj
        a.forward = types.MethodType(_merged_attention, a)
        m.gate_up_proj = merged_linear(gu)
        del m.gate_proj, m.up_proj
        m.forward = types.MethodType(_merged_mlp, m)
        n += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n


def load_modelopt_fp8(model_dir, device="cuda", dtype=torch.bfloat16):
    """A ModelOpt FP8 checkpoint as a transformers model whose decoder linears are FP8Linear (embeddings, norms and
    the LM head stay in bf16, as in the checkpoint)."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    try:
        from transformers.initialization import no_init_weights  # transformers 5
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    cfg = AutoConfig.from_pretrained(model_dir)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    with torch.device(device), no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    sd = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        sd.update(load_file(str(f), device=str(device)))
    fp8 = set()
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Linear) and f"{name}.weight_scale" in sd:
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child,
                    FP8Linear(sd.pop(f"{name}.weight"), sd.pop(f"{name}.weight_scale"), sd.pop(f"{name}.input_scale"),
                              sd.pop(f"{name}.bias", None)))
            fp8.add(name)
    if not fp8:
        raise ValueError(f"{model_dir}: no FP8 linear layers (weight_scale / input_scale) in the checkpoint")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.rsplit(".", 1)[0] not in fp8]
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" in missing:
        model.tie_weights()  # the output head is the embedding (Gemma 4): not in the checkpoint
        missing.remove("lm_head.weight")
    if missing or unexpected:
        raise ValueError(f"{model_dir}: checkpoint does not match the model (missing {missing[:5]}, unexpected "
                         f"{unexpected[:5]})")
    return model.eval()


def quantize(model, quant):
    """On-the-fly torchao quantisation of the decoder layers (the LM head stays in bf16).
    fp8   -- dynamic FP8 activations + FP8 weights, per-row scales (Ada, Hopper, Blackwell)
    nvfp4 -- NVFP4 activations + weights (Blackwell); for best FP4 speed prefer the ModelOpt NVFP4 checkpoint + vLLM"""
    from torchao.quantization import quantize_
    if quant == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif quant == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        cfg = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    else:
        raise ValueError(f"unknown quantisation {quant!r} (fp8 | nvfp4)")
    quantize_(model.model, cfg)
