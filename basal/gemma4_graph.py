"""The Gemma 4 (basal-omni) forward of the CUDA graph engine (GraphBackend: --mode fast, fast-nocompile; basal 2.3).
GraphBackend keeps its machinery (static shapes and one CUDA graph per shape, the prefix trie of a request in one
packed row, pass 1 over the root into key / value buffers, pass 2 over every branch with the trie mask, the state
cache); for a Gemma 4 model (Gemma4 / Gemma4Unified, text: E2B, E4B, 12B, 31B) it replays this forward instead of the
Llama one:

  forward(ids, kv=None, ro=None)                    plain causal rows from position 0 (single prompts, pass-1 roots)
  forward_suffix(ids, mask, pos, prefix, ro=None)   pass 2: rows over root keys / values with the trie mask

What differs from Llama, layer by layer as transformers' Gemma4TextDecoderLayer (the same modules are called):

- **Layer types.** Sliding-window layers (window W: E2B / E4B 512, 12B / 31B 1024; head size 256) between global layers
  (head size 512, partial rotary embeddings, on 12B / 31B one projection for keys and values: attention_k_eq_v, the
  values normalised but not rotated). A token of a packed row sees its ancestors in the trie at the positions they have
  in its own prompt, so in sliding layers the window is a condition on positions added to the trie mask: a branch
  token sees a root key at position j only when its own position is < j + W, exactly as in its prompt alone.
- **Readout rows.** From the last layer with its own keys and values on, a layer's output matters only where it is
  read: E2B / E4B reuse the keys and values of earlier layers in their last layers (E4B: the last 18 of 42), and the
  attention and MLP of the last own-key layer only feed those (on 12B / 31B the last layer is that layer). With `ro`
  (the readout positions of every row) these layers run at the readout rows only, their queries over the keys and
  values of every token; same maths, ~40% less compute on E4B.
- **Per-layer inputs** (E2B / E4B), the embedding scale, q / k / v norms, the four norms around attention and MLP, the
  layer scalar: transformers' own modules and methods.
- **Attention kernels** per layer type (`kernels`, {"sliding_attention": k, "full_attention": k}): "math" = torch
  SDPA's math kernel with a boolean mask, the kernel EagerBackend runs for Gemma 4 (the reference; queries in parts,
  so that the scores of a long row fit); "flex" = FlexAttention with block masks of the same conditions; "fa3" =
  FlashAttention 3 (sliding layers' plain rows; flex where a mask is needed). flex and fa3 run in compiled forwards
  only (uncompiled, flex_attention is a slow reference); uncompiled forwards run math.
- The output head reads the code rows in fp32 with the final logit softcap (GraphBackend.softcap).

Attention is causal for text; the bidirectional attention of image tokens (use_bidirectional_attention "vision") is
not used here, since this path takes text only (media run in the eager backend).
"""
from collections import namedtuple

import torch
import torch.nn.functional as F


def text_model(model):
    """The decoder of a Gemma 4 model: Gemma4(Unified)ForConditionalGeneration -> model.language_model,
    Gemma4ForCausalLM -> model. (The same helper as basal/gemma4.py of the eager backend.)"""
    return getattr(model.model, "language_model", model.model)


def is_gemma4(model):
    return str(getattr(model.config, "model_type", "")).startswith("gemma4")


def is_gemma4_dir(model_dir):
    """A model directory with a Gemma 4 config.json (before loading it)."""
    import json
    from pathlib import Path
    p = Path(model_dir) / "config.json"
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("model_type", "")).startswith("gemma4")
    except (OSError, ValueError):
        return False


def supports(model):
    """None if this forward runs the model, else why not (the MoE layers of 26B-A4B are not ported)."""
    if not is_gemma4(model):
        return "not a Gemma 4 model"
    if getattr(text_model(model).config, "enable_moe_block", False):
        return "Gemma 4 MoE layers (26B-A4B) are not in the CUDA graph engine; use --mode eager or vllm"
    return None


KERNELS = ("math", "flex", "fa3")
_FLEX = []


def _compile(fn):
    """torch.compile in Inductor's deterministic mode (TORCHINDUCTOR_DETERMINISTIC=0: without it). Otherwise Inductor
    times some reduction kernels at their first launch and keeps the faster config: the RMSNorm kernels ran with a
    reduction block of 4,096 or 2,048 (Inductor adds the halved block when a kernel needs many registers), whichever
    was faster by timings within 1% of each other, and the block sets the order of the sum. The choice is not always
    cached, so another process can time it again: basal-omni-1.0-mini on an H100, 2.3.3, two runs on fresh compile
    caches differed in 485 of 840 identity readouts (top answer in 7), a cold and a warm cache in 400 (top answer in
    4; 22 Oct). In this mode every such kernel has one config, chosen without timing. Set per compile: Dynamo resets the
    global switch after every frame it traces (it restores torch.use_deterministic_algorithms, which sets it)."""
    import os
    return torch.compile(fn, options={"deterministic": os.environ.get("TORCHINDUCTOR_DETERMINISTIC", "1") == "1"})


def _flex_attention():
    """flex_attention as it is inside the compiled forward (Inductor generates its kernel there); from uncompiled code
    (fast-nocompile, shapes without a graph, the long lane) one compiled instance (automatic dynamic shapes), since
    uncompiled flex_attention is a slow reference implementation."""
    from torch.nn.attention.flex_attention import flex_attention
    if torch.compiler.is_compiling():
        return flex_attention
    if not _FLEX:
        _FLEX.append(_compile(flex_attention))
    return _FLEX[0]


LayerC = namedtuple("LayerC", "heads kv_heads d eps sizes")  # constants of a layer (sizes: of a merged q / k / v)


def _rms(x, w, eps):
    """Gemma4RMSNorm: in fp32, x * (mean(x^2) + eps)^-0.5, times the weight (None: the values' norm), back to x's
    dtype."""
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (y * w.float() if w is not None else y).type_as(x)


def _rot(x, cos, sin):
    """apply_rotary_pos_emb of Gemma 4 on [b, L, heads, d] with cos / sin [b, L, d]."""
    cos, sin = cos[:, :, None], sin[:, :, None]
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


def _pre(h, P, C, cos, sin, kb=None, vb=None, pk=None, pv=None):
    """A decoder layer up to attention: the input norm and the projections -> (q [b, L, heads x d] as projected,
    k, v [b, kv heads, P + L, d]: normed, keys rotated, written to kb / vb and after the prefix pk / pv; None for a
    KV-shared layer). transformers' Gemma4TextAttention, with attention_k_eq_v (values = the key projection)."""
    from .lean import linear
    x = _rms(h, P["ln_in"], C.eps)
    if C.sizes is not None:
        q, k, *v = linear(x, P["qkv"]).split(C.sizes, -1)
        v = v[0] if v else k
    else:
        q = linear(x, P["q"])
        if not C.kv_heads:
            return q, None, None
        k = linear(x, P["k"])
        v = linear(x, P["v"]) if "v" in P else k
    b, L = x.shape[:2]
    k = _rot(_rms(k.view(b, L, C.kv_heads, C.d), P["k_norm"], C.eps), cos, sin).transpose(1, 2)
    v = _rms(v.view(b, L, C.kv_heads, C.d), None, C.eps).transpose(1, 2)
    if kb is not None:
        kb.copy_(k)
        vb.copy_(v)
    if pk is not None:
        k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
    return q, k, v


def _post(h, q, k, v, P, C, cos, sin, mask, pli):
    """The rest of a decoder layer (transformers' Gemma4TextDecoderLayer): queries normed and rotated, attention
    with the kernel of the mask, o_proj, the four norms around attention and MLP, the per-layer input, the layer
    scalar."""
    from .lean import linear
    b, L = h.shape[:2]
    q = _rot(_rms(q.view(b, L, C.heads, C.d), P["q_norm"], C.eps), cos, sin).transpose(1, 2)
    h = h + _rms(linear(Gemma4Graph._attend(q, k, v, mask), P["o"]), P["ln_attn"], C.eps)
    x = _rms(h, P["ln_pre"], C.eps)
    if "gu" in P:
        g, u = linear(x, P["gu"]).chunk(2, -1)
    else:
        g, u = linear(x, P["gate"]), linear(x, P["up"])
    h = h + _rms(linear(F.gelu(g, approximate="tanh") * u, P["down"]), P["ln_ff"], C.eps)
    if pli is not None:
        g = F.gelu(linear(h, P["ple_gate"]), approximate="tanh") * pli
        h = h + _rms(linear(g, P["ple_proj"]), P["ln_ple"], C.eps)
    return (h * P["scalar"]).to(h.dtype)  # in place in transformers (h *= scalar): the same rounding


class Gemma4Graph:
    # "math": queries per SDPA call, so that the scores (rows x heads x queries x keys, fp32) stay within SCORES
    # elements: in CUDA graphs every captured shape keeps its own peak in the graph pool (1.1-max with 2,048-query
    # parts: 33 GB in the pool, out of memory before warm-up ended, 2026-10-20)
    SCORES = 2 ** 26
    BLOCK = 128   # "flex": block size of the block masks
    SMEM = None   # shared memory per block of the GPU (flex's tiles, _flex)

    def __init__(self, model, kernels="math", compile=False, merge=False):
        lm = self.lm = text_model(model)
        cfg = self.cfg = lm.config
        n = cfg.num_hidden_layers
        self.layers = list(lm.layers[:n])
        self.types = list(cfg.layer_types[:n])
        self.window = cfg.sliding_window
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        dev = lm.embed_tokens.weight.device
        if dev.type == "cuda":
            Gemma4Graph.SMEM = torch.cuda.get_device_properties(dev).shared_memory_per_block_optin
        self.own = [not layer.self_attn.is_kv_shared_layer for layer in self.layers]
        assert all(self.own[: sum(self.own)]), "layers with their own keys and values come first"
        self.n_kv = sum(self.own)
        # the last layer with its own keys and values; a shared layer reads the last such layer of its type (the
        # layer transformers marks store_full_length_kv)
        self.last_kv = self.n_kv - 1
        self.src = {i: max(j for j in range(self.n_kv) if self.types[j] == t)
                    for i, t in enumerate(self.types) if not self.own[i]}
        self.store = set(self.src.values())
        # per own layer: (key / value heads, head size), the shape of its key / value buffers
        self.kv_shapes = [(a.k_proj.out_features // a.head_dim, a.head_dim)
                          for a in (layer.self_attn for layer in self.layers[: self.n_kv])]
        self.ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        if kernels in ("fa3", "fa3+math"):  # FlashAttention 3 in the sliding-window layers; flex or math in the
            kernels = {"sliding_attention": "fa3", "full_attention": "flex" if kernels == "fa3" else "math"}  # global
        if isinstance(kernels, str):
            kernels = {t: kernels for t in set(self.types)}
        for k in kernels.values():
            if k not in KERNELS:
                raise ValueError(f"unknown Gemma 4 attention kernel {k!r} (one of {', '.join(KERNELS)})")
        self.kernels = kernels
        if kernels.get("full_attention") == "fa3":
            raise ValueError("fa3 runs the sliding-window layers only (FlashAttention 3 takes heads up to 256 wide)")
        self.flex = any(k != "math" for k in kernels.values())  # fa3 is flex where a mask is needed
        self.qkv, self.gu = {}, {}  # merged projections per layer: (linear, output sizes)
        self.merged = self._merge() if merge else 0
        if compile or self.flex:  # flex: uncompiled code calls one compiled flex_attention (_flex_attention), an
            # entry per mask kind, head size and rows-only pass: 13 for mini's long states, past the default 8, after
            # which flex_attention runs its unfused reference (fast-nocompile, 2026-10-20)
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
        self.args = self._layer_args()
        # one compiled function per layer part, with automatic dynamic shapes (a static compile, then the dimensions
        # that change: rows and tokens). The whole forward compiled as one graph took 15+ minutes of start-up on mini
        # (each recompile of 42 layers; and with dynamic=True every dimension went symbolic and tracing did not end)
        self.compiled = compile
        if compile:
            self.pre, self.post, self.embed = _compile(_pre), _compile(_post), _compile(self._embed)

    @staticmethod
    def _fuse(mods):
        """One linear for several that read the same input: their weights concatenated into a new tensor, and the
        modules' own weights become views of it (transformers' forward of the same model still runs; no second
        copy). -> (linear, output sizes)."""
        w = torch.cat([x.weight.detach() for x in mods])
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=w.dtype)
        lin.weight = torch.nn.Parameter(w, requires_grad=False)
        o = 0
        for x in mods:
            x.weight = torch.nn.Parameter(w[o: o + x.out_features], requires_grad=False)
            o += x.out_features
        return lin, [x.out_features for x in mods]

    @torch.no_grad()
    def _merge(self):
        """One q / k / v GEMM per layer with its own keys and values (q / k with attention_k_eq_v) and one gate / up
        GEMM per layer, for plain bf16 / fp32 linears without bias (FP8 and torchao linears keep theirs). -> the
        number of layers merged."""
        def plain(mods):
            return all(type(x) is torch.nn.Linear and x.bias is None for x in mods) \
                and len({x.weight.dtype for x in mods}) == 1
        n = 0
        for i, layer in enumerate(self.layers):
            a, m = layer.self_attn, layer.mlp
            if self.own[i]:
                qkv = [a.q_proj, a.k_proj] + ([a.v_proj] if a.v_proj is not None else [])
                if plain(qkv):
                    self.qkv[i] = self._fuse(qkv)
            if plain([m.gate_proj, m.up_proj]):
                self.gu[i] = self._fuse([m.gate_proj, m.up_proj])
            n += i in self.qkv or i in self.gu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n

    # -- the layer --------------------------------------------------------------------------------------------------
    def _rope(self, pos, t, dtype):
        """cos / sin [b, L, head size] of layer type t at positions pos [b, L], as Gemma4TextRotaryEmbedding (fp32,
        then the model's dtype)."""
        r = self.lm.rotary_emb
        inv = getattr(r, f"{t}_inv_freq")
        f = pos[..., None].float() * inv.float()
        e = torch.cat([f, f], -1)
        s = getattr(r, f"{t}_attention_scaling")
        return (e.cos() * s).to(dtype), (e.sin() * s).to(dtype)

    def _layer_args(self):
        """Per layer: its weights as tensors (lean.py's linear parameters: bf16, torchao or FP8Linear) and its
        constants. The layer functions take them as arguments, so one compiled function serves every layer of a kind
        (a module argument would be guarded by identity: a compile per layer)."""
        from .lean import _params
        eps = self.cfg.rms_norm_eps
        out = []
        for i, layer in enumerate(self.layers):
            a, m = layer.self_attn, layer.mlp
            P = dict(ln_in=layer.input_layernorm.weight, q_norm=a.q_norm.weight, o=_params(a.o_proj),
                     ln_attn=layer.post_attention_layernorm.weight, ln_pre=layer.pre_feedforward_layernorm.weight,
                     ln_ff=layer.post_feedforward_layernorm.weight, down=_params(m.down_proj),
                     scalar=layer.layer_scalar)
            sizes = None
            if i in self.qkv:
                P["qkv"], sizes = _params(self.qkv[i][0]), tuple(self.qkv[i][1])
            else:
                P["q"] = _params(a.q_proj)
                if self.own[i]:
                    P["k"] = _params(a.k_proj)
                    if a.v_proj is not None:
                        P["v"] = _params(a.v_proj)
            if self.own[i]:
                P["k_norm"] = a.k_norm.weight
            if i in self.gu:
                P["gu"] = _params(self.gu[i][0])
            else:
                P["gate"], P["up"] = _params(m.gate_proj), _params(m.up_proj)
            if self.ple:
                P.update(ple_gate=_params(layer.per_layer_input_gate), ple_proj=_params(layer.per_layer_projection),
                         ln_ple=layer.post_per_layer_input_norm.weight)
            heads = a.q_proj.out_features // a.head_dim
            kv_heads = self.kv_shapes[i][0] if self.own[i] else 0
            out.append((P, LayerC(heads, kv_heads, a.head_dim, eps, sizes)))
        return out

    def _embed(self, ids):
        """Token embeddings (scaled) and, on E2B / E4B, the per-layer inputs [b, L, layers, 256]."""
        h = self.lm.embed_tokens(ids)
        return h, self.lm.project_per_layer_inputs(h, self.lm.get_per_layer_inputs(ids, None)) if self.ple else None

    # -- attention ----------------------------------------------------------------------------------------------------
    @staticmethod
    @torch.compiler.disable
    def _math(q, k, v, m):
        """SDPA's math kernel: q [b, H, Lq, d], k / v [b, Hkv, Lk, d], m bool [b, 1, Lq, Lk] (True = attend) ->
        [b, Lq, H * d]. Long rows in parts of queries (SCORES). Outside the compiled graph (the reference kernel)."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        b, H, Lq, _ = q.shape
        n = max(16, Gemma4Graph.SCORES // (b * H * k.shape[2]))
        outs = []
        with sdpa_kernel([SDPBackend.MATH]):
            for s in range(0, Lq, n):
                outs.append(F.scaled_dot_product_attention(q[:, :, s: s + n], k, v, attn_mask=m[:, :, s: s + n],
                                                           scale=1.0, enable_gqa=True))
        o = outs[0] if len(outs) == 1 else torch.cat(outs, 2)
        return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _flex(q, k, v, bm):
        """FlexAttention with a block mask. The 512-wide global heads take 32 x 32 tiles: Inductor's default tiles
        need more shared memory than an H100 has (Triton "out of resource", 2026-10-20); in fp32 one stage as well.
        With 99 KB per block (RTX PRO 6000 Blackwell; H100: 227 KB) one stage in bf16 too, and 16 x 16 tiles in fp32
        in both layer types (sm_120 probe, 2026-10-20: d512 bf16 32 x 32 x 1 stage 0.44 ms against math's 3.13;
        fp32 32 x 32 x 1 stage needed 131 KB)."""
        opts = {"FORCE_USE_FLEX_ATTENTION": True}
        small = (Gemma4Graph.SMEM or 1 << 30) < 160 * 1024
        fp32 = q.dtype == torch.float32
        if fp32 and small:
            opts.update(BLOCK_M=16, BLOCK_N=16, num_stages=1)
        elif q.shape[-1] > 256:
            opts.update(BLOCK_M=32, BLOCK_N=32, **({"num_stages": 1} if fp32 or small else {}))
        o = _flex_attention()(q, k, v, block_mask=bm, scale=1.0, enable_gqa=True, kernel_options=opts)
        return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _fa3(q, k, v, W):
        """FlashAttention 3, causal with a window of W tokens (None: none), plain rows from position 0: q [b, H, L, d],
        k / v [b, Hkv, L, d] -> [b, L, H * d]."""
        import flash_attn_interface as fa3
        o = fa3.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=1.0,
                                causal=True, window_size=(W - 1 if W else -1, 0))
        o = o[0] if isinstance(o, (tuple, list)) else o
        return o.reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _fa3_varlen(q, k, v, W, cu):
        """FlashAttention 3's varlen kernel over compact rows (prompts back to back in one row [1, T], cu: cumulative
        token counts, int32), causal with a window of W tokens within each prompt -> [1, T, H * d]."""
        import flash_attn_interface as fa3
        T = q.shape[2]
        o = fa3.flash_attn_varlen_func(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), cu, cu, T, T,
                                       softmax_scale=1.0, causal=True, window_size=(W - 1 if W else -1, 0))
        o = o[0] if isinstance(o, (tuple, list)) else o
        return o.reshape(1, T, -1)

    @staticmethod
    def _attend(q, k, v, m):
        """Attention with the kernel the mask is for: a boolean mask (math), a block mask (flex), ("fa3", window) or
        ("fa3v", window, cu) (compact rows)."""
        if isinstance(m, tuple):
            if m[0] == "fa3v":
                return Gemma4Graph._fa3_varlen(q, k, v, m[1], m[2])
            return Gemma4Graph._fa3(q, k, v, m[1])
        if isinstance(m, torch.Tensor):
            return Gemma4Graph._math(q, k, v, m)
        return Gemma4Graph._flex(q, k, v, m)

    # -- masks --------------------------------------------------------------------------------------------------------
    def _window(self, t):
        return self.window if t == "sliding_attention" else None

    MARGIN = 256  # sliding layers of a long state keep window + MARGIN tokens: a state that continues it a few tokens
    # before its end (a tokenizer merge where new text joins the old) still has a window of keys (incremental states)

    def kv_lens(self, L):
        """Root tokens per own layer that pass 2 (or a continuation of the state) can read, for a root of L tokens: a
        sliding layer's last window + MARGIN only, as later rows see window - 1 tokens back at most."""
        return [min(L, self.window + self.MARGIN) if t == "sliding_attention" else L for t in self.types[: self.n_kv]]

    def _plain_masks(self, L, qpos, dev, twl=None):
        """Boolean masks [b, 1, Lq, L] per layer type of plain causal rows (keys at positions 0..L-1) for queries at
        qpos [b, Lq] (all positions, or the readout rows). twl [b]: two-way rows (a model trained with two-way
        reading): in the global layers every query of row b also sees the keys up to twl[b], its last real token
        (-1: a row read one way); the sliding-window layers stay causal."""
        k = torch.arange(L, device=dev)[None, None, :]
        causal = k <= qpos[:, :, None]
        out = {}
        for t in set(self.types):
            W = self._window(t)
            if W:
                out[t] = (causal & (k > qpos[:, :, None] - W))[:, None]
            else:
                out[t] = (causal if twl is None else causal | (k <= twl[:, None, None]))[:, None]
        return out

    def _suffix_masks(self, mask, pos, lens):
        """Pass 2: the trie mask (additive, 0 = visible) [b, 1, S, P + S] -> boolean masks per layer type over its
        keys: the root's last lens[t] (all P, or a sliding layer's window of a long root: kv_lens) and the branches;
        sliding layers also need the key within the window of the query's position (root keys at P - lens[t]..P - 1,
        branch keys at pos)."""
        P = mask.shape[-1] - pos.shape[1]
        out = {}
        for t in set(self.types):
            o, W = P - lens[t], self._window(t)
            allow = mask[..., o:] >= 0
            if W:
                kpos = torch.cat([torch.arange(o, P, device=pos.device)[None].expand(pos.shape[0], P - o), pos], 1)
                allow = allow & (kpos[:, None, :] > pos[:, :, None] - W)[:, None]
            out[t] = allow
        return out

    def _plain_block(self, L, t, dev):
        """FlexAttention block mask of a plain causal row of length L for layer type t (no batch dimension: the same
        for every row), built from block arithmetic (create_block_mask evaluates all L x L pairs: 2.5 GB at 50k
        tokens): key block j is visible from query block i when 0 <= i - j <= dmax (the window), and full (no mask
        evaluated) when 1 <= i - j <= fmax. Built inside the forward, so a compiled forward reads L as a symbolic
        size (a cached mask made outside fixed L in its guards: a recompile for every graph length)."""
        from torch.nn.attention.flex_attention import BlockMask
        W, Bk = self._window(t), self.BLOCK
        n = -(-L // Bk)
        cols = torch.arange(n, device=dev)[None]
        d = torch.arange(n, device=dev)[:, None] - cols               # [n, n]: query block - key block
        vis = (d >= 0) & (d <= (W + Bk - 2) // Bk) if W else d >= 0
        full = (d >= 1) & (d <= (W - Bk) // Bk) if W else d >= 1
        part = vis & ~full

        def pack(sel):  # selected key blocks first, ascending
            idx = torch.where(sel, cols, n).sort(1).values.clamp(max=n - 1)
            return sel.sum(1).to(torch.int32)[None, None], idx.to(torch.int32)[None, None]

        def mod(b, h, q, kv):
            return (q >= kv) & (q - kv < W) if W else q >= kv
        return BlockMask.from_kv_blocks(*pack(part), *pack(full), BLOCK_SIZE=Bk, mask_mod=mod, seq_lengths=(L, L))

    def _dense_block(self, B, Lq, Lk, mod, dev):
        """A block mask that visits every (query block, key block) pair, each evaluated with mod: its block lists do
        not depend on the values the mask reads, so a CUDA graph keeps it and only those tensors change."""
        from torch.nn.attention.flex_attention import BlockMask
        nq, nk = -(-Lq // self.BLOCK), -(-Lk // self.BLOCK)
        num = torch.full((B, 1, nq), nk, dtype=torch.int32, device=dev)
        idx = torch.arange(nk, dtype=torch.int32, device=dev).expand(B, 1, nq, nk).contiguous()
        return BlockMask.from_kv_blocks(num, idx, BLOCK_SIZE=self.BLOCK, mask_mod=mod, seq_lengths=(Lq, Lk))

    def _pads(self, pos, mask, ro):
        """The index tensors the flex masks read, each padded by a block: flex evaluates a mask on whole blocks, so
        query (and branch key) indices run up to a block past the rows; padded copies keep every read in bounds. Made
        outside the compiled forward and passed in (a clamp of the index in the mask broke Inductor's flex template,
        "XBLOCK is not defined", and tensors made inside it cannot be read by a mask: "'function' object has no
        attribute 'graph'", 2026-10-20)."""
        from .lean import TreeMask
        if not self.flex:
            return {}
        Bk, out = self.BLOCK, {}
        if ro is not None:
            out["ro"] = F.pad(ro, (0, Bk))
        if isinstance(mask, TreeMask):  # padding tokens are block S, which sees nothing but itself
            out["s"] = F.pad(mask.s, (0, Bk), value=pos.shape[1])
            out["pos"] = F.pad(pos, (0, Bk))
        return out

    def _rows_block(self, rop, R, L, t):
        """Block mask of the readout rows (rop: their positions, padded) of plain causal rows of length L."""
        W = self._window(t)

        def mod(b, h, q, kv):
            p = rop[b, q]
            return (p >= kv) & (p - kv < W) if W else p >= kv
        return self._dense_block(rop.shape[0], R, L, mod, rop.device)

    def _tree_block(self, tm, pads, S, P, t, R=None, off=0):
        """Block mask of pass 2 (the TreeMask of basal/lean.py: root tokens per row, block of every branch token,
        ancestors per block; lean.tree_mask_mod's mask) with the window on positions in sliding layers; P keys before
        the rows, the root's from position off on (a sliding layer's window of a long root: kv_lens); R: the readout
        rows only (pads["ro"])."""
        W = self._window(t)
        sp, posp, rop = pads["s"], pads["pos"], pads.get("ro") if R is not None else None

        def mod(b, h, q, kv):
            qq = rop[b, q] if rop is not None else q
            j = kv - P                                  # the branch token a key is (negative: a root key)
            jj = j * (j >= 0)
            tree = (j >= 0) & ((tm.anc[b, sp[b, qq], sp[b, jj]] & (jj <= qq)) | (jj == qq))
            rk = kv + off if off else kv                # a root key's position
            ok = (rk < tm.pr[b]) | tree
            if W:
                ok = ok & (posp[b, qq] - torch.where(j >= 0, posp[b, jj], rk) < W)
            return ok
        return self._dense_block(sp.shape[0], S if R is None else R, P + S, mod, sp.device)

    # -- the forward --------------------------------------------------------------------------------------------------
    def _compact_masks(self, kernels, T, cu, ro, pads):
        """Masks of compact rows: each prompt attends to itself only. Sliding layers: FlashAttention 3's varlen kernel
        with the window; global layers: a flex block mask that visits, for each query block, the key blocks from the
        start of its first query's prompt on (the prompts are contiguous); the readout rows: a dense block mask over
        the row (one query per prompt)."""
        from torch.nn.attention.flex_attention import BlockMask
        Bk, dev = self.BLOCK, cu.device
        tok = torch.arange(T, device=dev)
        doc = torch.bucketize(tok, cu[1:], right=True)                 # the prompt of every token
        start = cu[doc]                                                  # its prompt's first token
        n = -(-T // Bk)
        qb = torch.arange(n, device=dev)
        lo = start[qb * Bk] // Bk                                        # first key block of each query block
        idx = torch.minimum(lo[:, None] + qb[None], qb[:, None]).to(torch.int32)
        num = (qb - lo + 1).to(torch.int32)
        docp, rop = pads["doc"], pads["ro"]

        m_all, m_ro = {}, {}
        for t in set(self.types):
            W = self._window(t)

            def same(b, h, q, kv, W=W):  # the same prompt, causal (and the window: positions advance with indices)
                ok = (docp[q] == docp[kv]) & (q >= kv)
                return ok & (q - kv < W) if W else ok
            if kernels[t] == "fa3":
                m_all[t] = ("fa3v", W, cu)
            else:
                m_all[t] = BlockMask.from_kv_blocks(num[None, None], idx[None, None].contiguous(), BLOCK_SIZE=Bk,
                                                    mask_mod=same, seq_lengths=(T, T))

            def rows(b, h, q, kv, W=W):
                p = rop[0, q]
                ok = (docp[p] == docp[kv]) & (p >= kv)
                return ok & (p - kv < W) if W else ok
            m_ro[t] = self._dense_block(1, ro.shape[1], T, rows, dev)
        return m_all, m_ro

    def _masks(self, kernels, L, pos, prefix, mask, ro, pads, twl=None):
        """The attention masks of one forward per layer type, for every query and for the readout rows ro: boolean
        masks for "math", block masks for "flex" (reading pads), ("fa3", window) for FlashAttention 3 over plain
        rows. twl: the two-way rows of plain rows (_plain_masks; the global layers then need the math kernel)."""
        from .lean import TreeMask
        dev, types = pos.device, set(self.types)
        math = [t for t in types if kernels[t] == "math"]
        flex = [t for t in types if kernels[t] != "math"]  # fa3: FlashAttention 3 for every query of plain rows
        m_all, m_ro = {}, {}
        if prefix is None:
            if twl is not None and any(kernels[t] != "math" for t in types if self._window(t) is None):
                raise ValueError("two-way rows need the math kernel in the global layers")
            if math:
                m_all = self._plain_masks(L, pos, dev, twl)
                m_ro = self._plain_masks(L, ro, dev, twl) if ro is not None else {}
            for t in flex:
                m_all[t] = ("fa3", self._window(t)) if kernels[t] == "fa3" else self._plain_block(L, t, dev)
                if ro is not None:
                    m_ro[t] = self._rows_block(pads["ro"], ro.shape[1], L, t)
            return m_all, m_ro
        lens = {self.types[i]: prefix[i][0].shape[2] for i in range(self.n_kv)}  # root keys per layer type
        P = max(lens.values())                     # the root (its global layers keep every key)
        if isinstance(mask, TreeMask):
            if math:
                raise ValueError("the trie as a TreeMask needs the flex kernel in every layer type")
        elif math:
            m_all = self._suffix_masks(mask, pos, lens)
            if ro is not None:
                b, R = ro.shape
                m_ro = {t: m.gather(2, ro[:, None, :, None].expand(b, 1, R, m.shape[-1])) for t, m in m_all.items()}
        if flex and not isinstance(mask, TreeMask):
            raise ValueError("the flex kernel takes the trie of pass 2 as a TreeMask")
        for t in flex:
            m_all[t] = self._tree_block(mask, pads, L, lens[t], t, off=P - lens[t])
            if ro is not None:
                m_ro[t] = self._tree_block(mask, pads, L, lens[t], t, ro.shape[1], P - lens[t])
        return m_all, m_ro

    def _run(self, kernels, ids, pos, kv=None, prefix=None, mask=None, ro=None, between=None, pads=None,
             compiled=False, cu=None, twl=None):
        """kernels: per layer type; ids / pos [b, L]; kv: per own layer (K, V) buffers [b, kv heads, L, d] that receive
        this pass's keys and values (pass 1; a shorter buffer receives the last ones: kv_lens); prefix: per own layer
        (K, V) [b, kv heads, P, d] (sliding layers: the last kv_lens of the root may be all there is) the rows continue
        (pass 2) with mask (the trie: additive, or a TreeMask for flex); ro [b, R]: readout positions -> final hidden states
        [b, R, D] (None: every position, [b, L, D]); between(i) is called before layer i (the long lane); pads: the
        padded index tensors of the flex masks (_pads); compiled: the compiled layer functions; twl: two-way rows (_plain_masks). The masks of every
        query and of the readout rows are built here (_masks; the kernel follows the mask). A layer is two functions
        (_pre, _post), so the rows-only layers can split after the keys and values; a shared layer's keys and values
        are those its source layer kept."""
        if cu is not None:  # compact rows (forward_compact)
            m_all, m_ro = self._compact_masks(kernels, ids.shape[1], cu, ro, pads)
        else:
            m_all, m_ro = self._masks(kernels, ids.shape[1], pos, prefix, mask, ro, pads or {}, twl)
        pre, post, embed = (self.pre, self.post, self.embed) if compiled else (_pre, _post, self._embed)
        h, pli = embed(ids)
        types = set(self.types)
        cs = {t: self._rope(pos, t, h.dtype) for t in types}
        masks, kept = m_all, {}
        for i, (P, C) in enumerate(self.args):
            if between is not None:
                between(i)
            t = self.types[i]
            buf = kv[i] if kv is not None and self.own[i] else (None, None)
            tail = buf[0] is not None and buf[0].shape[2] < ids.shape[1]  # a sliding layer's window (kv_lens)
            pre_kv = prefix[i] if prefix is not None and self.own[i] else (None, None)
            q, k, v = pre(h, P, C, *cs[t], *((None, None) if tail else buf), *pre_kv)
            if tail:
                n = buf[0].shape[2]
                buf[0].copy_(k[:, :, -n:])
                buf[1].copy_(v[:, :, -n:])
            if self.own[i]:
                if i in self.store:
                    kept[t] = (k, v)
            else:
                k, v = kept[t]
            if ro is not None and i == self.last_kv:  # from here on, the readout rows only
                g = ro[:, :, None]
                h, q = h.gather(1, g.expand(-1, -1, h.shape[-1])), q.gather(1, g.expand(-1, -1, q.shape[-1]))
                if pli is not None:
                    pli = pli.gather(1, ro[:, :, None, None].expand(-1, -1, *pli.shape[2:]))
                cs = {u: tuple(c.gather(1, g.expand(-1, -1, c.shape[-1])) for c in cs[u]) for u in types}
                masks = m_ro
            h = post(h, q, k, v, P, C, *cs[t], masks[t], pli[:, :, i] if pli is not None else None)
        return _rms(h, self.lm.norm.weight, self.cfg.rms_norm_eps)

    def _kernels(self, dense=False):
        """The kernels of a call: the configured ones, or the math kernel for pass 2 with a dense mask (flex takes the
        trie as a TreeMask)."""
        if not (dense and self.flex):
            return self.kernels
        return {t: "math" for t in self.kernels}

    @torch.no_grad()
    def forward(self, ids, kv=None, ro=None, between=None, compiled=True, twl=None):
        """Plain causal rows [b, L] from position 0 (right padding cannot reach a real token); kv: per own layer
        (K, V) buffers [b, kv heads, L, d] that receive the keys and values (pass 1); ro [b, R]: readout positions ->
        [b, R, D] (None: [b, L, D]); between(i) before layer i (the long lane: uncompiled, one layer at a time).
        compiled: through the compiled forward (when the engine compiles; shapes without a graph run uncompiled).
        twl [b]: per row its last real token when it is read two-way, else -1 (a model trained with two-way
        reading; the global layers' kernel is then math)."""
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device)[None].expand(ids.shape[0], L)
        compiled = compiled and between is None
        return self._run(self.kernels, ids, pos, kv=kv, ro=ro, between=between, pads=self._pads(pos, None, ro),
                         compiled=compiled and self.compiled, twl=twl)

    @torch.no_grad()
    def forward_compact(self, ids, pos, cu, ro, compiled=True):
        """Compact rows (--batches compact, with the fa3 kernels): single prompts packed back to back in one row
        [1, T] (positions from 0 per prompt, cu: cumulative token counts [prompts + 2], int32; the padding after the
        last prompt is one more prompt nobody reads) -> final hidden states at ro [R] (one readout per prompt): [R, D].
        No padding of rows to graph lengths or of row counts."""
        T = ids.shape[1]
        doc = torch.bucketize(torch.arange(T, device=ids.device), cu[1:], right=True)
        pads = dict(ro=F.pad(ro[None], (0, self.BLOCK)), doc=F.pad(doc, (0, self.BLOCK), value=-1))
        out = self._run(self.kernels, ids, pos, ro=ro[None], pads=pads, compiled=compiled and self.compiled, cu=cu)
        return out[0]

    @torch.no_grad()
    def forward_suffix(self, ids, mask, pos, prefix, ro=None, compiled=True, kv=None):
        """Pass 2: rows [b, S] at positions pos over per own layer prefix (K, V) [b, kv heads, P, d] (or a sliding
        layer's last kv_lens(P) of them), with the trie mask (additive [b, 1, S, P + S], 0 = visible; a TreeMask for
        flex) -> [b, R, D] at ro [b, R] (None: all)."""
        from .lean import TreeMask
        kernels = self._kernels(dense=not isinstance(mask, TreeMask))
        return self._run(kernels, ids, pos, kv=kv, prefix=prefix, mask=mask, ro=ro, pads=self._pads(pos, mask, ro),
                         compiled=compiled and self.compiled)
