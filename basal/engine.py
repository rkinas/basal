"""Inference backends. Every backend exposes the same two calls:

  run(prompts, ids_list)          -> one probability list per prompt (softmax over its option letters)
  run_shared(groups[, policy])    -> groups = [(prompts, ids_list)]; the prompts of one group share a prefix (all option
                                     orders of one question, or every branch of a whole request: state once, ask many)

A prompt is a string or a list of token ids (the server tokenizes once and passes ids).

Backends:
  EagerBackend      plain PyTorch forward (reference, any GPU, Apple MPS or CPU)
  GraphBackend      static shapes + CUDA graphs, optional torch.compile, shared prefix for the two option orders,
                    token-budget batching, optional torchao FP8 / NVFP4 quantisation
  ExitGraphBackend  GraphBackend split into graph segments at trained early-exit layers; exit policy per request
  MPSBackend        GraphBackend's shared prefix + token-budget batching on Apple MPS (PyTorch, no graphs)
  VLLMBackend       vLLM, for ModelOpt FP8 / NVFP4 checkpoints (native low-precision kernels)
  SGLangBackend     SGLang offline engine (1.5): the same letter readout from the log-probabilities of the option tokens
  MLXBackend        Apple Silicon, MLX / mlx-lm (1.5): MLX fp8 / fp4 exports or bf16 weights (optional 8-bit at load
                    time); shared-prefix KV cache
  GGUFBackend       a GGUF file in process through llama.cpp (llama-cpp-python; Metal, CUDA or CPU); shared prefix as
                    llama.cpp sequences
  OllamaBackend     Ollama with the GGUF exports (1.5): raw prompt, one token, top log-probabilities of the letters
  LlamaCppBackend   llama.cpp server with the GGUF exports (1.5): token-id prompt, `n_probs` log-probabilities
  TPUBackend        (basal/tpu.py, extra basal[tpu]; experimental) the same SOAM packing in JAX / XLA on Google TPU
"""
import json
import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import PREFILL


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

    def __init__(self, model_dir, dtype="bfloat16", device=None, media=False):
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
            self.media_kinds = kinds_of(json.loads((Path(model_dir) / "config.json").read_text()))
            if not self.media_kinds:
                raise SystemExit("--media on: the model has no image or audio encoder (config.json: vision_config, "
                                 "audio_config)")
            from transformers import AutoModelForImageTextToText, AutoProcessor
            cls, self.proc = AutoModelForImageTextToText, AutoProcessor.from_pretrained(model_dir)
            pc = Path(model_dir) / "processor_config.json"
            self.max_audio_s = audio_window(json.loads(pc.read_text())) if pc.exists() else None
            _memo_media(self.proc)
        self.lock, self.media_lock = threading.Lock(), threading.Lock()
        self.model = cls.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(dev).eval()
        if str(getattr(self.model.config, "model_type", "")).startswith("gemma4"):
            # Gemma 4 (basal-omni): SDPA's memory-efficient kernel returned wrong attention on real activations in some
            # shapes (omni kernel check 2026-10-07); the math kernel equals eager attention there
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.dev = dev
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
    # GPU memory and answer HTTP 422)
    CHUNK = 2048
    MAX_BATCH_TOKENS = 16384

    def run(self, prompts, ids_list):
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
                    raise ValueError(f"the prompt with its media is {ids.shape[0]} tokens, more than the {max_tokens} "
                                     "tokens of the context; send fewer or shorter media, or detail \"default\"")
                encs.append((enc, int(ids.shape[0]), n_media))
            out = []
            for (enc, n, n_media), lid in zip(encs, ids_list):
                x = {k: (v.to(self.dev, self.model.dtype) if v.is_floating_point() else v.to(self.dev))
                     for k, v in enc.items() if isinstance(v, torch.Tensor)}
                with self.lock:
                    self.model(**x, logits_to_keep=1)
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
        ids, cache = torch.tensor([list(e)], device=self.dev), DynamicCache()
        for s0 in range(0, ids.shape[1], self.CHUNK):
            part = ids[:, s0:s0 + self.CHUNK]
            pos = torch.arange(s0, s0 + part.shape[1], device=self.dev)[None]
            cache = self.model(input_ids=part, position_ids=pos, past_key_values=cache, use_cache=True,
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
        self.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1)
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
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


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


def _varlen_attention(q, k, v, cu, scale, layer=0):
    """Causal attention of compact rows: [1, heads, T, d] in (prompts packed back to back, cu = their cumulative token
    counts, int32, empty prompts at the end), [1, T, heads, d] out. FlashAttention 3's varlen kernel (a torch custom
    op, so the compiled forward has no graph break): the same rows as the padded forward on an H100.
    With a template prefix cu = (cu_q, cu_k, gather, prefix): every prompt's keys / values are its template prefix's
    (prefix[layer] = (K, V) [prefix tokens, kv heads, d], computed at start-up) followed by its own; gather picks them
    from prefix + own tokens, and the queries (own tokens) attend bottom-right causally."""
    import flash_attn_interface as fa3
    T = q.shape[2]
    k, v = k[0].transpose(0, 1), v[0].transpose(0, 1)
    if isinstance(cu, tuple):
        cu, cu_k, gather, prefix = cu
        pk, pv = prefix[layer]
        k = torch.cat([pk, k]).index_select(0, gather)
        v = torch.cat([pv, v]).index_select(0, gather)
    else:
        cu_k = cu
    o = fa3.flash_attn_varlen_func(q[0].transpose(0, 1), k, v, cu, cu_k, T, k.shape[0], softmax_scale=scale,
                                   causal=True)
    return (o[0] if isinstance(o, (tuple, list)) else o)[None]


def basal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                    basal_kv=None, basal_prefix=None, basal_cu=None, **kwargs):
    """torch SDPA with additions for the two-pass forward of branched units and for compact rows (the kwargs pass
    through the transformers model forward): basal_kv[layer] = (K, V) buffers that receive this layer's keys and
    values (the root pass), basal_prefix[layer] = (K, V) of a computed prefix that the queries also attend to (the
    branch pass), basal_cu = cumulative token counts of prompts packed into one row (each attends to itself only).
    A causal forward without a mask uses the kernel chosen by set_attention (--attn)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if basal_cu is not None:
        return _varlen_attention(query, key, value, basal_cu, scaling, module.layer_idx), None
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
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kwargs)


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
    state is cached runs only its questions and options (pass 2 over the cached keys and values)."""

    def __init__(self, budget_bytes, salt=b""):
        from collections import OrderedDict
        self.budget, self.used, self.salt = int(budget_bytes), 0, salt
        self.entries, self.lock = OrderedDict(), threading.Lock()
        self.hits = self.misses = self.evictions = 0

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

    def put(self, key, kv):
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
        if size > self.budget:
            return
        with self.lock:
            if key in self.entries:
                return
            self.entries[key] = (kv, size)
            self.used += size
            while self.used > self.budget:
                _, (_, s) = self.entries.popitem(last=False)
                self.used -= s
                self.evictions += 1

    def stats(self):
        with self.lock:
            return dict(entries=len(self.entries), used_bytes=self.used, budget_bytes=self.budget, hits=self.hits,
                        misses=self.misses, evictions=self.evictions)


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
      tokens over roots up to the largest of LENS), or runs as a normal forward for other shapes.

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
    readout_layer = False  # --readout-layer on (with compact rows): the last layer's o_proj / MLP at the readouts only
    template = ()          # --template-prefix on (with compact rows): template prefixes computed at start-up
    prefix = False
    lean, flex = None, False      # the lean layer (basal/lean.py) and its FlexAttention pass 2, when enabled
    state_cache = None            # StateCache: keys / values of request states, reused across requests

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, shared=True, warm=True, attn="auto",
                 merge="auto", lean=False, tree="dense", state_cache_gb=0, prefix=False, batches="default",
                 readout_layer="off", template="off"):
        torch.backends.cuda.enable_cudnn_sdp(False)  # before capture: a captured graph keeps its attention kernel
        self.fp8_checkpoint = modelopt_fp8(model_dir)  # the -FP8 repositories: FP8 weights with static scales
        # likewise: the kernel of the causal forwards (--attn; auto keeps FlashAttention 2 for FP8 GEMMs)
        self.attn = set_attention(attn, fp8=self.fp8_checkpoint or quant == "fp8")
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
        causal_sdpa(self.model)
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
        if compile and self.fwd == self._forward:  # fuse normalisation / rotary / activation kernels, capture graphs
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
            # pass 2 stays uncompiled: compiled graphs for its (rows, branch length, root length) grid took minutes
            # to capture; its graphs replay the normal forward
            self.fwd = torch.compile(self._forward, dynamic=True)
        self.shared = shared
        self._wrows = {}
        if batches == "fine":
            self.BATCHES = self.FINE_BATCHES
        # compact rows need FlashAttention 3's varlen kernel (Hopper); auto: compact where it runs
        self.compact = batches in ("compact", "auto") and self.attn == "fa3" and self.dev.type == "cuda"
        if batches == "compact" and not self.compact:
            raise ValueError("--batches compact needs FlashAttention 3 (--attn fa3 / auto on Hopper)")
        if self.compact:
            self.fwd_c = torch.compile(self._forward_compact, dynamic=True) if compile else self._forward_compact
            last = self.model.model.layers[-1]
            self.readout_layer = readout_layer in ("on", "auto") and type(last).__name__ == "LlamaDecoderLayer" \
                and type(last.self_attn).__name__ == "LlamaAttention" and type(last.mlp).__name__ == "LlamaMLP"
        self.prefix = bool(prefix) and shared
        gb = max(state_cache_gb, self.PREFIX_GB if self.prefix else 0)
        self.state_cache = StateCache(gb * 2 ** 30) if gb and shared else None
        self._alloc_kv()
        self.template = []  # --template-prefix (with compact rows): [(start in template_kv, prefix token ids)]
        if self.compact and template in ("on", "auto"):
            self._template_prefixes(model_dir)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        if warm:
            self.warm()

    def _alloc_kv(self):
        """The keys / values buffer that pass 1 writes and pass 2 reads: per layer, room for the largest plain graph
        and for the largest pass-2 graph's rows over the longest root."""
        cfg = self.model.config
        H = cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        n = max(self.TOKEN_BUDGET, self.PLAIN_LENS[-1], self.TREE_ROWS[-1] * self.LENS[-1]) * H * D
        dt = self.model.model.embed_tokens.weight.dtype
        self.kv_shape = (H, D)
        # zeros, not empty: below a cached root the masked tail of its rows is never written, and a masked NaN still
        # poisons the readout (0 x NaN)
        self._kv = [(torch.zeros(n, dtype=dt, device=self.dev), torch.zeros(n, dtype=dt, device=self.dev))
                    for _ in self.model.model.layers] if self.shared else None

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
        H, D = self.kv_shape
        return [(k[: b * H * L * D].view(b, H, L, D), v[: b * H * L * D].view(b, H, L, D)) for k, v in self._kv]

    def grid(self, lens):
        """(rows, length) shapes of one graph family."""
        return [(b, L) for L in lens for b in self.BATCHES if b * L <= self.TOKEN_BUDGET or b == 1]

    def warm(self):
        """Capture every plain graph of the grid, and with `shared` the pass-2 graphs for roots up to the largest of
        LENS."""
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
                        if b * S <= self.TOKEN_BUDGET:
                            self._graph_sfx(b, S, L)
            if self.prefix:  # pass 2 of short requests over a cached template prefix
                for L in (x for x in self.LENS if x <= self.PREFIX_ROOT):
                    for S in self.PREFIX_SFX:
                        for b in self.PREFIX_ROWS:
                            if b * S <= self.TOKEN_BUDGET:
                                self._graph_sfx(b, S, L)
        # beyond the largest graph the forwards run uncompiled; their first use sets up kernels and memory for long
        # shapes (3 s of the first long request on an H100), so a long prompt and a long branched unit run here once
        n = self.PLAIN_LENS[-1] + 64
        if getattr(self.model.config, "max_position_embeddings", 0) >= n + 8:
            root = [self.tok.pad_token_id] * n
            self.run_shared([([root + [5]], [[10, 11]])] + ([([root + [5, 6], root + [6, 5]], [[10, 11], [10, 11]])]
                                                             if self.shared else []))

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
    def _forward(self, ids, kv=None):
        """Plain causal forward; kv: per-layer (K, V) buffers [rows, kv heads, length, head dim] it fills (pass 1)."""
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
            return m(input_ids=ids, position_ids=pos, use_cache=False, basal_cu=cu).last_hidden_state
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        h = m.embed_tokens(ids)
        pe = m.rotary_emb(h, position_ids=pos)
        layers = m.layers[: m.config.num_hidden_layers]
        for layer in layers[:-1]:
            h = layer(h, attention_mask=None, position_embeddings=pe, position_ids=pos, past_key_values=None,
                      use_cache=False, basal_cu=cu)
        last, a = layers[-1], layers[-1].self_attn
        x = last.input_layernorm(h)
        if hasattr(a, "qkv_proj"):
            q, k, v = a.qkv_proj(x).split(a.qkv_sizes, -1)
        else:
            q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        shape = (*x.shape[:-1], -1, a.head_dim)
        q, k = apply_rotary_pos_emb(q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2), *pe)
        o = _varlen_attention(q, k, v.view(shape).transpose(1, 2), cu, a.scaling, a.layer_idx)[0, ro]  # [R, heads, d]
        h = h[0, ro] + a.o_proj(o.reshape(o.shape[0], -1))
        h = h + last.mlp(last.post_attention_layernorm(h))
        return m.norm(h)

    @torch.no_grad()
    def _forward_masked(self, ids, mask, pos):
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state

    @torch.no_grad()
    def _forward_suffix(self, ids, mask, pos, prefix):
        """Pass 2: rows that continue computed prefixes (per-layer (K, V) [rows, kv heads, prefix length, head dim]);
        mask [rows, 1, length, prefix length + length]."""
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                                basal_prefix=prefix).last_hidden_state

    @torch.no_grad()
    def _forward_layered(self, ids, kv=None, between=None):
        """The plain causal forward (`_forward`, uncompiled) one decoder layer at a time, calling between(i) before
        layer i (the long lane gives the GPU to the main lane there): the transformers model's own loop, the same
        modules with the same arguments, so the same result."""
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

    def _capture(self, key, fn, *static, warmup=3):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(warmup):
                fn(*static)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn(*static)
        self.graphs[key] = (g, *static, out)
        return self.graphs[key]

    def _graph(self, b, L):
        """Plain graph (b rows, length L): (graph, ids, kv buffers or None, hidden states)."""
        key = ("p", b, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        return self._capture(key, self.fwd, ids, self._kv_views(b, L))

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
            return self._capture(key, self.fwd_c, ids, pos, cu, ro, cu_k, gather)
        return self._capture(key, self.fwd_c, ids, pos, cu, *([ro] if ro is not None else []))

    def _graph_sfx(self, b, S, L):
        """Pass-2 graph (b rows of S tokens over root keys / values of length L, the layout of the plain graphs):
        (graph, ids, mask, positions, kv, hidden states)."""
        key = ("x", b, S, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, S), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(L, L + S, device=self.dev)[None].expand(b, S).contiguous()
        dt = self.model.model.embed_tokens.weight.dtype
        if self.flex:
            from .lean import TreeMask
            mask = TreeMask(torch.zeros(b, dtype=torch.long, device=self.dev),
                            torch.full((b, S), S, dtype=torch.long, device=self.dev),
                            torch.zeros((b, S + 1, S + 1), dtype=torch.bool, device=self.dev))
        else:
            mask = torch.zeros((b, 1, S, L + S), dtype=dt, device=self.dev)
        return self._capture(key, self.fwd_sfx, ids, mask, pos, self._kv_views(b, L), warmup=2)

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

    def _readout(self, h, lids):
        W, idx, pad = self._rows(lids)
        return self._letters(h.float() @ W.T, idx, pad, lids)  # last_hidden_state is already normalised

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
            return self._forward(self._to_dev(torch.tensor([t])))[0, last]
        L, b = self._bucket(n, self.PLAIN_LENS), self._bucket(len(idx), self.BATCHES)
        g, ids, _, out = self._graph(b, L)
        h_ids, _, _, rows, cols, _ = self._host_rows(packs, idx, b, L)
        ids.copy_(self._to_dev(h_ids), non_blocking=True)
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
        cached keys / values go where pass 1 would have written them; the roots computed here are stored."""
        H, D = self.kv_shape
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
                kv = [(torch.empty((1, H, L, D), dtype=k.dtype, device=self.dev),
                       torch.empty((1, H, L, D), dtype=k.dtype, device=self.dev)) for k, _ in self._kv]
            else:
                L = self._bucket(max(P), self.PLAIN_LENS)
                kv = self._kv_views(max(b, self._bucket(b, rows) if L <= self.LENS[-1] else b), L)
            same = {}  # rows with the same cached root (a template prefix): one broadcast copy per layer
            for r, e in enumerate(hits):
                same.setdefault(id(e), (e, P[r], []))[2].append(r)
            for e, p, rs in same.values():
                if len(rs) == 1:
                    for (kb, vb), (ke, ve) in zip(kv, e):
                        kb[rs[0], :, :p].copy_(ke, non_blocking=True)
                        vb[rs[0], :, :p].copy_(ve, non_blocking=True)
                    continue
                ix = self._to_dev(torch.tensor(rs))
                for (kb, vb), (ke, ve) in zip(kv, e):
                    kb[:, :, :p].index_copy_(0, ix, ke[None].expand(len(rs), *ke.shape))
                    vb[:, :, :p].index_copy_(0, ix, ve[None].expand(len(rs), *ve.shape))
        elif max(P) > self.PLAIN_LENS[-1]:
            # a root longer than the largest graph runs uncompiled, like a long single prompt: the compiled forward
            # recompiles for these fresh key / value buffers (44-109 s on the first long request on an H100), the
            # uncompiled one costs a few percent of a forward that is rare and long anyway
            assert b == 1
            L, B = P[0], 1
            kv = [(torch.empty((1, H, L, D), dtype=k.dtype, device=self.dev),
                   torch.empty((1, H, L, D), dtype=k.dtype, device=self.dev)) for k, _ in self._kv]
            h1 = self._forward(self._to_dev(torch.tensor([packs[idx[0]][0][:L]])), kv)
        else:
            L, B = self._bucket(max(P), self.PLAIN_LENS), self._bucket(b, self.BATCHES)
            g, ids, kv, h1 = self._graph(B, L)
            h_ids = np.full((B, L), self.tok.pad_token_id, dtype=np.int64)
            for r, k in enumerate(idx):
                h_ids[r, : P[r]] = packs[k][0][: P[r]]
            ids.copy_(self._to_dev(h_ids), non_blocking=True)
            g.replay()
        if h1 is not None and self.state_cache is not None:  # store the roots computed here
            for r, key in enumerate(keys):
                if key is not None and hits[r] is None:
                    self.state_cache.put(key, [(kb[r, :, : P[r]].clone(), vb[r, :, : P[r]].clone()) for kb, vb in kv])
        S = next((x for x in sfx if n2 <= x), None)
        graph = None
        if S and L <= self.LENS[-1] and b <= rows[-1]:  # a pass-2 graph, rows padded (stale buffer rows are masked)
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
            sel += [(1, r, i) if i < p else (2, r, i - p) for i in last]
        # FlexAttention only in the pass-2 graphs captured at start-up: an uncaptured shape (long roots, many or long
        # branches) runs the dense pass 2 uncompiled, as without flex. The compiled flex layer specialises on shape
        # classes, and a request of a new class compiled for 1-4 s (SPEED.md sections 6 and 11)
        if self.flex and graph is not None:  # the trie as FlexAttention's mask: root tokens, blocks, ancestors
            tm = self._tree_mask(seg2, par2, P + [0] * (B - b), S)
            g2, s_ids, s_mask, s_pos, _, h2 = graph
            s_ids.copy_(self._to_dev(ids2), non_blocking=True)
            s_pos.copy_(self._to_dev(pos2), non_blocking=True)
            for dst, src in zip(s_mask, tm):
                dst.copy_(src)
            g2.replay()
        else:
            m2 = self._mask_from_seg(self._to_dev(seg2), par2)                             # [B, 1, S, S]
            Pr = self._to_dev(torch.tensor(P + [0] * (B - b)))
            root = Pr[:, None] > torch.arange(L, device=self.dev)[None]                     # [B, L]
            pre = torch.where(root, 0.0, torch.finfo(m2.dtype).min).to(m2.dtype)[:, None, None].expand(B, 1, S, L)
            if graph is None:
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
        return out

    @torch.no_grad()
    def _tree_cache(self, pack):
        """A branched unit in two eager passes over a transformers KV cache (the MPS and early-exit backends, whose
        attention has no key / value buffers): the root with causal attention, then every other block as one packed
        row over that cache with the trie mask. -> hidden states at the readout positions."""
        t, pp, sg, last, par = pack
        P = sg.index(1) if 1 in sg else len(t)  # block 0 (the root) comes first and is contiguous
        out = self.model.model(input_ids=self._to_dev(torch.tensor([t[:P]])), use_cache=True)
        h = [None] * len(last)
        for m, i in enumerate(last):
            if i < P:  # a prompt that ends inside the root (a duplicate of the shared prefix)
                h[m] = out.last_hidden_state[0, i]
        rest = [m for m, i in enumerate(last) if i >= P]
        if rest:
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
    _launch = threading.Lock()  # a forward's input copies, replay and readout are enqueued by one call at a time

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
                p = torch.softmax((h.float() @ W.T).gather(1, cols).masked_fill(pad, float("-inf")), -1)
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
        H, D = self.kv_shape
        P = sg.index(1)  # block 0 (the root) comes first and is contiguous
        h1 = None
        if hit is not None:
            kv = [(k[None], v[None]) for k, v in hit]
        else:
            kv = [(torch.empty((1, H, P, D), dtype=k.dtype, device=self.dev),
                   torch.empty((1, H, P, D), dtype=k.dtype, device=self.dev)) for k, _ in self._kv]
            h1 = self._forward_layered(self._to_dev(torch.tensor([t[:P]])), kv, between)[0]
            if key is not None and self.state_cache is not None:
                self.state_cache.put(key, [(k[0].clone(), v[0].clone()) for k, v in kv])
        n = len(t) - P
        m2 = self._mask_from_seg(self._to_dev(torch.tensor([[x - 1 for x in sg[P:]]])),
                                 [[-1 if q <= 0 else q - 1 for q in par[1:]]])                 # [1, 1, n, n]
        m2 = torch.cat([torch.zeros((1, 1, n, P), dtype=m2.dtype, device=self.dev), m2], -1)   # the root: visible
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
        th = json.loads((hd / "thresholds.json").read_text())
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
    """Apple GPU through PyTorch MPS: the shared-prefix packing (prefix trie, SOAM) and token-budget batching of
    GraphBackend, run as plain forwards (MPS has no CUDA graphs). Rows are padded on the right to the longest packed
    row of the chunk."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time

    def __init__(self, model_dir, dtype="bfloat16", shared=True, device="mps"):
        EagerBackend.__init__(self, model_dir, dtype, device)
        self.dev = torch.device(device)
        self.shared = shared
        self._wrows = {}

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


class VLLMBackend:
    """vLLM backend for ModelOpt FP8 / NVFP4 checkpoints. One generated token restricted to the option letters; with
    logprobs_mode="processed_logprobs" the log-probabilities are computed after that restriction, so the softmax over
    the letters equals the readout of the other backends. Where vLLM supports it, logprob_token_ids asks for exactly
    the letter ids: vllm-metal ignores the logprobs mode and returns the top-k of the unrestricted vocabulary, which
    can miss a letter. vLLM's automatic prefix caching shares the state between the two option orders. max_options:
    the most option codes one readout asks for (vLLM refuses more log-probabilities than its max_logprobs, default 20,
    so it is raised to the model's prompt format limit: 255 for format 2). vLLM 0.31 accepts at most 128
    logprob_token_ids; a question with more options takes the top-k of the restricted distribution instead, which on
    CUDA (processed log-probabilities) are exactly the allowed codes."""

    LOGPROB_TOKEN_IDS = 128  # longest logprob_token_ids list vLLM accepts

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096, max_options=10):
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.llm = LLM(model=str(model_dir), dtype=dtype, gpu_memory_utilization=mem, max_model_len=max_len,
                       enable_prefix_caching=True, logprobs_mode="processed_logprobs",
                       max_logprobs=max(16, max_options))
        self.context = max_len - 1  # longest prompt: max_model_len holds the one generated token too
        self.context_reason = f"vLLM's --max-len {max_len}"

    def run(self, prompts, ids_list):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        reqs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in EagerBackend.encode(self, prompts)]
        import inspect
        exact = "logprob_token_ids" in inspect.signature(SamplingParams).parameters
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(ids), allowed_token_ids=list(ids),
                              **(dict(logprob_token_ids=list(ids)) if exact and len(ids) <= self.LOGPROB_TOKEN_IDS
                                 else {}))
               for ids in ids_list]
        res = []
        for o, ids in zip(self.llm.generate(reqs, sps, use_tqdm=False), ids_list):
            lp = o.outputs[0].logprobs[0]
            x = torch.tensor([lp[i].logprob if i in lp else -1e9 for i in ids], dtype=torch.float32)
            res.append(torch.softmax(x, -1).tolist())
        return res

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
    (without this override it silently uses rope_theta = 10000)."""
    rp = cfg.get("rope_parameters") or {}
    out = {}
    if "rope_theta" not in cfg and "rope_theta" in rp:
        out["rope_theta"] = rp["rope_theta"]
    if not cfg.get("rope_scaling") and rp.get("rope_type", "default") != "default":
        out["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
    return out


class MLXBackend:
    """Apple Silicon (MLX / mlx-lm): the MLX exports (the -MLX-8bit / -MLX-fp4 repositories) or a bf16 Hugging Face
    directory. Same prompts, tokens and letter readout as the PyTorch backends. The shared prefix of a group (both
    option orders of a question; with SOAM every question and order of a request, i.e. the state) is computed once into
    a KV cache, and the remaining suffixes run as one right-padded batch continuing that cache: a suffix sees the
    prefix and its own earlier tokens only, so every readout equals a separate forward of its prompt. All MLX work
    runs on one dedicated thread (the server calls the backend from an executor)."""

    MAX_ROWS = 16      # suffix rows per forward
    TOKEN_BUDGET = 8192  # rows x padded length per forward without a shared prefix
    CACHE_BYTES = 2 << 30  # MLX buffer cache limit (and wired headroom above the weights)

    def __init__(self, model_dir, max_rows=None, quant=None):
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.prefill = PREFILL
        self.max_rows = max_rows or self.MAX_ROWS
        self.pool.submit(self._load, Path(model_dir), quant).result()

    def _load(self, md, quant=None):
        """quant: None, or "q8": MLX affine 8-bit weights (group size 64) of the decoder blocks' linear layers, made at
        load time from bf16 weights (the embeddings and the LM head stay bf16); the -MLX-8bit exports need no quant."""
        import mlx.core as mx
        from mlx_lm.utils import load_model
        self.mx = mx
        cfg = json.loads((md / "config.json").read_text())
        if quant not in (None, "q8"):
            raise ValueError(f"unknown MLX quantisation {quant!r} (q8)")
        # lazy for q8: the weights are materialised once, after quantisation (never the full bf16 copy)
        self.model, self.config = load_model(md, lazy=bool(quant), model_config=mlx_config_overrides(cfg))
        self.quantization = self.config.get("quantization")
        if quant:
            if self.quantization:
                raise ValueError(f"{md} is already quantised ({self.quantization}); use it without --quant")
            import mlx.nn as nn
            nn.quantize(self.model, group_size=64, bits=8,
                        class_predicate=lambda path, m: isinstance(m, nn.Linear) and path.startswith("model.layers."))
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

    def encode(self, prompts):
        return EagerBackend.encode(self, prompts)

    def _head(self, h):
        m = self.model
        return m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)

    def _code_rows(self, key):
        """fp32 rows of the output head for the code ids in key (cached), as GraphBackend._rows: the code logits are
        computed in fp32 from the hidden state (bf16 logits of magnitude 16-32 step by 0.125, which clumps the
        probabilities). None for a quantised head (its logits come from the head itself)."""
        rows = self.__dict__.setdefault("_wrows", {})
        if key not in rows:
            m = self.model
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
        h = self.model.model(ids, cache=cache)
        return h[mx.arange(B), mx.array([len(r) - 1 for r in rows])]

    def _prefix(self, toks):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        self.model.model(mx.array([toks]), cache=cache)
        kv = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache]
        mx.eval(kv)
        return kv

    def _group(self, toks, lids):
        """One group: shared prefix once, then the suffixes in batches of max_rows."""
        import os
        P = min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1)
        prefix = self._prefix(toks[0][:P]) if P > 0 else None
        out = []
        for i in range(0, len(toks), self.max_rows):
            h = self._forward([t[P:] for t in toks[i: i + self.max_rows]], prefix)
            out += self._letters(h, lids[i: i + self.max_rows])
        return out

    def _singles(self, toks, lids):
        """Unrelated prompts: right-padded batches of similar length, no cache."""
        chunks, cur = [], []
        for k in sorted(range(len(toks)), key=lambda k: len(toks[k])):
            if cur and (len(cur) >= self.max_rows or (len(cur) + 1) * len(toks[k]) > self.TOKEN_BUDGET):
                chunks.append(cur); cur = []
            cur.append(k)
        res = [None] * len(toks)
        for c in chunks + [cur]:
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
    (unified KV cache, cleared per chunk)."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time

    N_CTX = 16384  # KV cells per chunk: room for a single long prompt
    context, context_reason = N_CTX, f"the in-process llama.cpp context ({N_CTX} cells)"
    N_SEQ = 64     # readouts (sequences) per chunk

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
        # a full model directory: its architecture must match the GGUF file; the -GGUF repositories carry tokenizer and
        # calibration without config.json (their .gguf files belong to them)
        cp = Path(model_dir) / "config.json"
        cfg = json.loads(cp.read_text()) if cp.exists() else {}
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
        self.ctx = C.llama_init_from_model(self.cmodel, cp)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self.batch = C.llama_batch_init(self.N_CTX, 0, self.N_SEQ)  # a prefix token belongs to every readout below it

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
                raise ValueError(f"prompt too long for the GGUF backend: {t} tokens (max {self.N_CTX})")
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
    q = (json.loads(p.read_text()).get("quantization_config") or {}) if p.exists() else {}
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
