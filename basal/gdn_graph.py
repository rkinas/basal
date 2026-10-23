"""Qwen3.5 (Gated DeltaNet) models in the CUDA graph engine (GraphBackend: --mode fast, fast-nocompile; basal 2.4).

A Qwen3.5 decoder (basal-omni-pico: Qwen3.5-0.8B, 18 Gated DeltaNet layers and 6 gated full-attention layers) carries a
state from token to token in its linear-attention layers: a matrix per head (16 x 128 x 128 in fp32, 1 MB per layer and
sequence) and the last three inputs of a short causal convolution. GraphBackend shares a request's state (the root of
its prefix trie) as attention keys and values only, so 2.3.2 refused these models there: 2.3.1 restarted the recurrent
layers from zero state in pass 2 and answered wrongly. GDNGraphBackend keeps GraphBackend's units, batching, graphs and
readout, and changes three things:

- **Pass 1** (plain rows from position 0: single prompts, and the roots of branched units) also writes each row's
  recurrent state and convolution tail at its last real token. Rows are padded on the right, and the padding tokens get
  beta = 0 and g = 0 (no update, no decay), so the state at the end of the padded row is the state of the real tokens.
- **Pass 2** runs one row per leaf of the trie below the root (a question, or one option order of it): the path of
  blocks from the root to that leaf, from the root's recurrent state and convolution tail (fla's chunk_gated_delta_rule
  with initial_state) and over the root's keys and values (the attention layers). A block shared by several leaves (a
  question's text under its two option orders) runs once per leaf: one packed row with a trie mask cannot continue one
  recurrent state into two branches.
- **Graphs.** Plain graphs (rows, length) as before; pass-2 graphs (leaf rows, leaf length, root bucket). Pass 1 writes
  the roots' keys and values in the layout of the root bucket (ROOT_LENS), and each pass-2 row gathers its root's keys,
  values and recurrent state inside the graph, so the pass-2 graphs do not depend on the exact root length.

The layers are transformers' own modules (projections, norms, MLP, rotary embedding); the Gated DeltaNet kernel is fla's
(flash-linear-attention), as in the eager reference. Not here (refused before loading, with what to use instead): fp8 /
nvfp4, compact rows, the state and prefix caches, template prefixes, --readout-layer on, --lean, the long lane.
"""
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine import GraphBackend, PREFILL, _copy_same, persist_compile_cache, set_attention

# the decoders this forward implements: Qwen3.5 text models (dense MLP) of these layer kinds
GDN_MODEL_TYPES = ("qwen3_5_text", "qwen3_5")
GDN_LAYERS = {"linear_attention", "full_attention"}


def _decoder_config(model_dir):
    """config.json of a model directory, the decoder part (text_config of a multimodal wrapper); None if unreadable."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return cfg.get("text_config") or cfg


def is_gdn(model_dir):
    """A Qwen3.5 decoder of Gated DeltaNet and full-attention layers (what GDNGraphBackend serves), from config.json.
    Other recurrent models (Qwen3-Next, Mamba and hybrids) stay refused by the graph engine."""
    c = _decoder_config(model_dir)
    if not c or c.get("model_type") not in GDN_MODEL_TYPES:
        return False
    kinds = set(c.get("layer_types") or ())
    return "linear_attention" in kinds and kinds <= GDN_LAYERS


def fla_rule():
    """fla's chunk_gated_delta_rule (flash-linear-attention), None when it is not installed."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception:
        return None
    return chunk_gated_delta_rule


def refused(quant=None, state_cache_gb=0, prefix=False, batches="default", lean=False, tree="dense",
            readout_layer="off", template="off", device=True):
    """Why GDNGraphBackend cannot serve a Qwen3.5 model with these settings (None: it can), before anything is loaded.
    device: also check for a CUDA GPU and fla (the server checks the settings before the download, the engine both)."""
    if quant:
        return (f"--mode {quant} is not available for Qwen3.5 (Gated DeltaNet) models: the CUDA graph engine serves "
                "them in bf16 (--mode fast); --mode vllm runs their FP8 checkpoints")
    opts = [name for name, on in (("--state-cache-gb", state_cache_gb), ("--prefix-cache on", prefix),
                                  ("--batches compact", batches == "compact"), ("--lean", lean),
                                  ("--tree flex", tree == "flex"), ("--readout-layer on", readout_layer == "on"),
                                  ("--template-prefix on", template == "on")) if on]
    if opts:
        return (f"{', '.join(opts)}: not available for Qwen3.5 (Gated DeltaNet) models in the CUDA graph engine (basal "
                "2.4 runs their plain and branched rows; the caches would have to keep recurrent states)")
    if device and not torch.cuda.is_available():
        return "the CUDA graph engine needs a CUDA GPU; serve Qwen3.5 models with --mode eager here"
    if device and fla_rule() is None:
        return ("the CUDA graph engine runs Qwen3.5's Gated DeltaNet layers with flash-linear-attention: pip install "
                "flash-linear-attention (or serve with --mode eager)")
    return None


def default_fast():
    """Whether a Qwen3.5 model gets --mode fast without --mode: on the GPU class its gate passed (Hopper, H100, 23 Oct)
    with fla installed; eager elsewhere until another GPU class passes it."""
    return (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9 and fla_rule() is not None)


def _rms(x, w, eps):
    """Qwen3_5RMSNorm: in fp32, weight stored as 1 + w."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * (1.0 + w.float())).type_as(x)


def _mlp(h, nw, gu, down, eps):
    """h + Qwen3_5MLP(post_attention_layernorm(h)), gate and up in one GEMM."""
    g, u = F.linear(_rms(h, nw, eps), gu).chunk(2, -1)
    return h + F.linear(F.silu(g) * u, down)


def _gdn_in(h, nw, w_in, A_log, dt_bias, mask, eps, sizes):
    """input_layernorm, the four input projections of Qwen3_5GatedDeltaNet in one GEMM, beta and g (padding: beta 0,
    g 0) -> (convolution input, z, beta, g)."""
    qkv, z, b, a = F.linear(_rms(h, nw, eps), w_in).split(sizes, -1)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    if mask is not None:  # padding: no update (beta 0), no decay (g 0)
        beta = beta * mask[..., None]
        g = g * mask[..., None]
    return qkv, z, beta, g


def _conv(qkv, w, c0, sizes):
    """The causal depthwise convolution and SiLU of Qwen3_5GatedDeltaNet (causal_conv1d_fn) -> q, k, v (views); c0
    [b, kernel - 1, channels]: the inputs before the rows (pass 2: the root's last ones)."""
    T = qkv.shape[1]
    if c0 is None:
        y = F.conv1d(qkv.transpose(1, 2).to(w.dtype), w, padding=w.shape[-1] - 1, groups=w.shape[0])[..., :T]
    else:
        y = F.conv1d(torch.cat([c0, qkv], 1).transpose(1, 2).to(w.dtype), w, groups=w.shape[0])
    return F.silu(y).to(qkv.dtype).transpose(1, 2).split(sizes, -1)


def _conv_fla(qkv, w, c0, sizes, conv):
    """_conv with fla's causal_conv1d kernel (token-major tiles, SiLU applied in fp32 before the rounding): one call
    per q / k / v over its channels, each output contiguous for fla's Gated DeltaNet kernel. cuDNN's generic depthwise
    kernel and the transposes of q / k / v into fla's layout took about a third of a forward's GPU time on an H100
    (kernel profile, 1.5k-token state, 22 Oct). c0 goes in as fla's initial state [b, channels, kernel] (the newest
    input last; the first slot is not read by a kernel of that size)."""
    init = None
    if c0 is not None:
        init = F.pad(c0.transpose(1, 2), (1, 0))
    out, a = [], 0
    for m in sizes:
        y, _ = conv(qkv[..., a: a + m], w[a: a + m, 0], initial_state=None if init is None else
                    init[:, a: a + m].contiguous(), activation="silu")
        out.append(y)
        a += m
    return out


def fla_conv1d():
    """fla's causal_conv1d, None when it is not there."""
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
    except Exception:
        return None
    return causal_conv1d


def _gdn_out(h, o, z, gw, out_w, eps):
    """h + out_proj(Qwen3_5RMSNormGated(o, z)) over the value heads."""
    dt, shape = o.dtype, o.shape
    x = o.reshape(-1, shape[-1]).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = gw * x.to(dt)
    x = (x * F.silu(z.reshape(-1, shape[-1]).float())).to(dt)
    return h + F.linear(x.reshape(*shape[:2], -1), out_w)


def _rotary(x, cos, sin, r):
    """apply_rotary_pos_emb of Qwen3.5 on one of q / k [b, heads, T, d]: the first r dims rotated (half split)."""
    c, s = cos[:, None], sin[:, None]
    xr = x[..., :r]
    rot = torch.cat([-xr[..., r // 2:], xr[..., : r // 2]], -1)
    return torch.cat([xr * c + rot * s, x[..., r:]], -1)


def _att_in(h, nw, w_qkv, qn, kn, cos, sin, eps, sizes, d, r):
    """input_layernorm, q (with its output gate) / k / v in one GEMM, q / k norms, rotary embedding of the first r dims
    -> q, k, v [b, heads, T, d], gate [b, T, heads * d]. (Head counts from sizes, not -1: Inductor failed on the
    inferred dimension with dynamic shapes.)"""
    b, T, _ = h.shape
    H, Hkv = sizes[0] // (2 * d), sizes[1] // d
    qg, k, v = F.linear(_rms(h, nw, eps), w_qkv).split(sizes, -1)
    q, gate = torch.chunk(qg.view(b, T, H, d * 2), 2, dim=-1)
    q = _rotary(_rms(q, qn, eps).transpose(1, 2), cos, sin, r)
    k = _rotary(_rms(k.view(b, T, Hkv, d), kn, eps).transpose(1, 2), cos, sin, r)
    return q, k, v.view(b, T, Hkv, d).transpose(1, 2), gate.reshape(b, T, H * d)


def _att_out(h, o, gate, o_w):
    """h + o_proj(attention * sigmoid(gate))."""
    b, T = gate.shape[:2]
    return h + F.linear(o.transpose(1, 2).reshape(b, T, -1) * torch.sigmoid(gate), o_w)


def _merged(mods):
    """One weight for several Linear layers of one input (no bias), the layers' weights then views of it (no copy kept
    beside them, and the transformers forward reads the same values) -> (weight, output sizes)."""
    w = torch.cat([m.weight.detach() for m in mods])
    sizes = [m.weight.shape[0] for m in mods]
    for m, part in zip(mods, w.split(sizes)):
        m.weight = torch.nn.Parameter(part, requires_grad=False)
    return w, sizes


class Qwen35Graph:
    """The forward of a Qwen3.5 decoder for GDNGraphBackend, layer by layer as transformers' Qwen3_5DecoderLayer (its
    modules' weights and maths; the projections that read one input are one GEMM):

      forward(ids, lens, kv, st)                        plain rows from position 0; kv / st: buffers that receive each
                                                        row's keys / values and recurrent state (pass 1)
      forward_suffix(ids, pos, unit, plen, kv, st)      pass 2: rows that continue root `unit` (plen tokens) from its
                                                        keys / values and recurrent state

    compile (--mode fast): the parts between the Gated DeltaNet kernels and the attention (norms, projections, gates,
    rotary embedding, MLP) as torch.compile functions of dynamic shapes, one per kind for every layer. On CUDA the
    convolution is fla's kernel in both modes.
    """

    def __init__(self, model, compile=False):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
        self.m = model.model
        cfg = self.m.config
        self.layers = list(self.m.layers[: cfg.num_hidden_layers])
        self.kinds = [cfg.layer_types[i] for i in range(len(self.layers))]
        self.eps = cfg.rms_norm_eps
        # fla's kernel on CUDA; transformers' torch version on the CPU (the tests: the same arguments; unwrapped, as
        # transformers routes that name to fla whenever fla is installed)
        cuda = next(model.parameters()).device.type == "cuda"
        self.rule = (fla_rule() if cuda else None) or inspect.unwrap(q35.torch_chunk_gated_delta_rule)
        self.args = []  # per layer: its weights and sizes for the functions above
        for lay, kind in zip(self.layers, self.kinds):
            mlp = (lay.post_attention_layernorm.weight, _merged([lay.mlp.gate_proj, lay.mlp.up_proj])[0],
                   lay.mlp.down_proj.weight)
            if kind == "linear_attention":
                a = lay.linear_attn
                if a.conv1d.bias is not None:
                    raise ValueError("a Gated DeltaNet convolution with a bias (not in Qwen3.5)")
                w_in, sizes = _merged([a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a])
                self.args.append(dict(a=a, nw=lay.input_layernorm.weight, w_in=w_in, sizes=sizes, mlp=mlp))
            else:
                a = lay.self_attn
                w_qkv, sizes = _merged([a.q_proj, a.k_proj, a.v_proj])
                self.args.append(dict(a=a, nw=lay.input_layernorm.weight, w_qkv=w_qkv, sizes=sizes, mlp=mlp))
        att = [x["a"] for x, k in zip(self.args, self.kinds) if k == "full_attention"]
        gdn = [x["a"] for x, k in zip(self.args, self.kinds) if k == "linear_attention"]
        self.kv_shapes = [(cfg.num_key_value_heads, a.head_dim) for a in att]
        self.heads = cfg.num_attention_heads
        g = gdn[0]
        # per Gated DeltaNet layer: the recurrent state [value heads, key dim, value dim] (fp32, as fla returns it) and
        # the convolution tail [kernel - 1, conv channels]: the inputs of the convolution at the last tokens
        self.state_shape = (g.num_v_heads, g.head_k_dim, g.head_v_dim)
        self.tail_shape = (g.conv_kernel_size - 1, g.conv_dim)
        self.n_gdn = len(gdn)
        # the model's constants bound here, so that a compiled function sees them as constants (dynamic=True traces
        # its float and int arguments as symbols; Inductor then failed on the norms' eps)
        eps, geps, d = self.eps, g.norm.variance_epsilon, att[0].head_dim
        gs = self.args[self.kinds.index("linear_attention")]["sizes"]
        qs = self.args[self.kinds.index("full_attention")]["sizes"]
        rd = self.m.rotary_emb.inv_freq.shape[0] * 2  # rotated dims of a head (partial_rotary_factor)
        cs = [g.key_dim, g.key_dim, g.value_dim]
        fns = dict(mlp=lambda h, nw, gu, down: _mlp(h, nw, gu, down, eps),
                   gdn_in=lambda h, nw, w, A_log, dt_bias, mask: _gdn_in(h, nw, w, A_log, dt_bias, mask, eps, gs),
                   conv=lambda qkv, w, c0: _conv(qkv, w, c0, cs),
                   gdn_out=lambda h, o, z, gw, out_w: _gdn_out(h, o, z, gw, out_w, geps),
                   att_in=lambda h, nw, w, qn, kn, cos, sin: _att_in(h, nw, w, qn, kn, cos, sin, eps, qs, d, rd),
                   att_out=_att_out, rms=lambda h, w: _rms(h, w, eps))
        conv = fla_conv1d() if cuda else None  # on CUDA fla's kernel, uncompiled (Triton), in both modes
        if compile:
            import torch._dynamo as dynamo
            import torch._inductor.config as inductor
            dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)  # (as the Llama forward)
            # the fused kernels round to bf16 where the uncompiled maths does (norms, gates, SwiGLU), so the compiled
            # forward gives the uncompiled one's numbers (tiny model, bf16, CPU: bit-identical; max diff 0.023
            # without). Without it the pico's top answers moved away from eager's (300 five-question x two-order
            # requests on an H100: 99.3% against 99.6% for fast-nocompile)
            inductor.emulate_precision_casts = True
            # Inductor's deterministic mode (TORCHINDUCTOR_DETERMINISTIC=0: without it), as basal-omni's forward since
            # 2.3.4: otherwise Inductor times some reduction kernels at their first launch and keeps the faster config,
            # and what its cache holds then depends on its history, so two starts could round differently
            det = os.environ.get("TORCHINDUCTOR_DETERMINISTIC", "1") == "1"
            fns = {k: torch.compile(f, dynamic=True, options={"deterministic": det}) for k, f in fns.items()}
        if conv is not None:
            fns["conv"] = lambda qkv, w, c0: _conv_fla(qkv, w, c0, cs, conv)
        self.f = fns

    def _rope(self, h, pos):
        return self.m.rotary_emb(h, pos[None].expand(3, *pos.shape))  # text: the same position in all three sections

    def _gdn(self, x, h, mask=None, lens=None, out=None, s0=None, c0=None):
        """h + Qwen3_5GatedDeltaNet(input_layernorm(h)) without a cache: mask [b, T] zeroes beta and g of padding
        (pass 1), out = (state [b, ...], tail [b, ...]) receives the state and convolution tail at token lens - 1; s0 /
        c0: the state and tail the rows continue (pass 2)."""
        a, f = x["a"], self.f
        b, T, _ = h.shape
        qkv, z, beta, g = f["gdn_in"](h, x["nw"], x["w_in"], a.A_log, a.dt_bias, mask)
        w = a.conv1d.weight  # [channels, 1, kernel]
        q, k, v = f["conv"](qkv, w, c0)
        q = q.reshape(b, T, -1, a.head_k_dim)
        k = k.reshape(b, T, -1, a.head_k_dim)
        v = v.reshape(b, T, -1, a.head_v_dim)
        r = a.num_v_heads // a.num_k_heads
        if r > 1:
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
        o, fs = self.rule(q, k, v, g=g, beta=beta, initial_state=s0, output_final_state=out is not None,
                          use_qk_l2norm_in_kernel=True)
        if out is not None:
            out[0].copy_(fs)
            n = w.shape[-1] - 1
            ix = lens[:, None] - n + torch.arange(n, device=h.device)[None]  # tokens lens - n .. lens - 1
            t = qkv.gather(1, ix.clamp_min(0)[..., None].expand(-1, -1, qkv.shape[-1]))
            out[1].copy_(t * (ix >= 0)[..., None])  # before position 0: zeros (a root shorter than the tail)
        return f["gdn_out"](h, o, z.reshape(b, T, -1, a.head_v_dim), a.norm.weight, a.out_proj.weight)

    def _attention(self, x, h, cos, sin, out=None, prefix=None):
        """h + Qwen3_5Attention(input_layernorm(h)) (queries with an output gate, q / k norms, partial rotary
        embedding). out = (K, V) buffers [b, kv heads, T, d] it fills (pass 1); prefix = (root K, root V [b, kv heads,
        Lk, d], allow [b, 1, groups * T, Lk + T]): the rows attend to their root and causally to themselves (pass 2)."""
        a = x["a"]
        b, T, _ = h.shape
        d = a.head_dim
        q, k, v, gate = self.f["att_in"](h, x["nw"], x["w_qkv"], a.q_norm.weight, a.k_norm.weight, cos, sin)
        if out is not None:
            out[0].copy_(k)
            out[1].copy_(v)
        G = a.num_key_value_groups
        if prefix is None:  # causal, no mask: SDPA's flash kernel (FlashAttention 3 where --attn chose it)
            kk, vv = (t[:, :, None].expand(-1, -1, G, -1, -1).reshape(b, -1, T, d) for t in (k, v))
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=a.scaling)
        else:  # the G query heads of a key / value head folded into the query length: no repeated keys
            pk, pv, allow = prefix
            kk, vv = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            o = F.scaled_dot_product_attention(q.reshape(b, kk.shape[1], G * T, d), kk, vv, attn_mask=allow,
                                               scale=a.scaling).reshape(b, -1, T, d)
        return self.f["att_out"](h, o, gate, a.o_proj.weight)

    def _layers(self, h, gdn, att):
        ia = ig = 0
        for x, kind in zip(self.args, self.kinds):
            if kind == "linear_attention":
                h = gdn(x, h, ig)
                ig += 1
            else:
                h = att(x, h, ia)
                ia += 1
            h = self.f["mlp"](h, *x["mlp"])
        return self.f["rms"](h, self.m.norm.weight)

    @torch.no_grad()
    def forward(self, ids, lens=None, kv=None, st=None):
        """Plain rows [b, T] from position 0 -> normalised hidden states [b, T, D]. lens [b]: real tokens per row (the
        rest is padding; needed with st); kv: per attention layer (K, V) [b, kv heads, T, d] to fill; st: per Gated
        DeltaNet layer (state [b, ...] fp32, tail [b, kernel - 1, channels]) to fill at token lens - 1."""
        b, T = ids.shape
        h = self.m.embed_tokens(ids)
        cos, sin = self._rope(h, torch.arange(T, device=ids.device)[None].expand(b, T))
        mask = None if lens is None else torch.arange(T, device=ids.device)[None] < lens[:, None]
        return self._layers(
            h,
            lambda x, h, i: self._gdn(x, h, mask, lens, None if st is None else st[i]),
            lambda x, h, i: self._attention(x, h, cos, sin, None if kv is None else kv[i]))

    @torch.no_grad()
    def forward_suffix(self, ids, pos, unit, plen, kv, st):
        """Pass 2: rows [b, S] (positions pos) that continue the roots unit [b] (rows of the pass-1 buffers kv / st,
        per attention layer (K, V) [roots, kv heads, Lk, d], per Gated DeltaNet layer (state, tail)), whose first
        plen [b] tokens are real -> normalised hidden states [b, S, D]."""
        b, S = ids.shape
        dev = ids.device
        h = self.m.embed_tokens(ids)
        cos, sin = self._rope(h, pos)
        Lk = kv[0][0].shape[2] if kv else 0
        G = self.heads // self.kv_shapes[0][0] if kv else 1
        root = (torch.arange(Lk, device=dev)[None] < plen[:, None])[:, None].expand(b, S, Lk)
        own = torch.ones(S, S, dtype=torch.bool, device=dev).tril()[None].expand(b, S, S)
        allow = torch.cat([root, own], -1)[:, None, None].expand(b, 1, G, S, Lk + S).reshape(b, 1, G * S, Lk + S)
        roots = [(k.index_select(0, unit), v.index_select(0, unit)) for k, v in kv]
        states = [(s.index_select(0, unit), c.index_select(0, unit)) for s, c in st]
        return self._layers(
            h,
            lambda x, h, i: self._gdn(x, h, s0=states[i][0], c0=states[i][1]),
            lambda x, h, i: self._attention(x, h, cos, sin, prefix=(*roots[i], allow)))


class GDNGraphBackend(GraphBackend):
    """GraphBackend for Qwen3.5 (Gated DeltaNet) models: GraphBackend(model_dir, ...) of such a model builds this one
    (GraphBackend.__new__). Single prompts run in the plain graphs as before; a branched unit runs its root in a plain
    graph that also keeps each row's recurrent state, then one row per leaf of its trie (see the module docstring)."""

    TREE_ROWS = [1, 2, 4, 8]  # roots per pass-1 forward (rows of the state buffers)
    ROOT_LENS = [512, 1024, 2048, 4096, 8192]  # pass-2 root buckets: the layout pass 1 writes the roots' keys in
    LEAF_ROWS = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64]  # pass-2 graph rows (leaves: questions x option orders)
    LEAF_LENS = [32, 64, 128, 256, 512, 1024]  # pass-2 graph row lengths (a leaf's path below the root)
    SUFFIX_EAGER = 128  # longer leaf rows (no graph) are padded to a multiple of this
    PASS2_KEYS = 65536  # pass-2 rows x root bucket per forward: the root keys / values each row gathers
    ONE_FORWARD = 1 << 30  # a prompt beyond the largest graph runs as one uncaptured forward
    _warmed = False  # warm() ran: the pass-2 graphs it did not capture (GPU memory) stay uncaptured

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, shared=True, warm=True, attn="auto",
                 merge="auto", lean=False, tree="dense", state_cache_gb=0, prefix=False, batches="default",
                 readout_layer="off", template="off", gemma_attn="auto"):
        if compile:
            persist_compile_cache()  # before anything compiles (GraphBackend.__init__)
        why = refused(quant, state_cache_gb, prefix, batches, lean, tree, readout_layer, template)  # before loading
        if why:
            raise ValueError(f"{model_dir}: {why}")
        cuda = torch.cuda.is_available()
        if cuda:
            torch.backends.cuda.enable_cudnn_sdp(False)  # before capture: a captured graph keeps its attention kernel
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")  # (GraphBackend.__init__)
        self.fp8_checkpoint = False
        self.attn = set_attention(attn)  # the causal plain rows; pass 2 runs SDPA with its mask
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
        self.model = self.model.to("cuda" if cuda else "cpu").eval()
        self.dev = next(self.model.parameters()).device
        self.prefill = PREFILL
        self.net = Qwen35Graph(self.model, compile=compile and cuda)
        self.fwd, self.fwd_sfx = self.net.forward, self.net.forward_suffix
        self.merged, self.lean, self.flex = len(self.net.args), None, False
        self.shared = shared
        self._wrows = {}
        if batches == "fine":
            self.BATCHES = self.FINE_BATCHES
        self._alloc_kv()
        self.template = []
        self.counts = {}  # forwards per kind: "plain" / "pass1" / "pass2", each "graph" or "eager" (gate scripts)
        self.pool = torch.cuda.graph_pool_handle() if cuda else None
        self.graphs = {}
        if warm and cuda:
            self.warm()
        self.one_forward = self.ONE_FORWARD

    # -- buffers -------------------------------------------------------------------------------------------------
    def _root_len(self, L):
        """The pass-2 root bucket of a pass-1 length."""
        return self._bucket(L, self.ROOT_LENS)

    def _alloc_kv(self):
        """Pass-1 outputs of the plain graphs of at most TREE_ROWS[-1] rows (the others write nothing): per attention
        layer one flat buffer for keys and one for values, viewed per root bucket L2 as [rows, kv heads, L2, d] (a
        graph of length L <= L2 writes [:b, :, :L]); per Gated DeltaNet layer the recurrent states [rows, heads, dk,
        dv] (fp32) and convolution tails [rows, kernel - 1, channels]. Zeros, not empty: pass 2 reads the masked tail
        of a root's bucket too, and a masked NaN still poisons a readout (0 x NaN)."""
        self.kv_shapes = self.net.kv_shapes
        self.kv_shape = self.kv_shapes[0]
        R = self.TREE_ROWS[-1]
        self._kv_n = max(b * self._root_len(L) for b, L in self.grid(self.PLAIN_LENS) if b <= R)
        dt, dev = self._dtype, self.dev
        if not self.shared:
            self._kv = self._st = None
            return
        self._kv = [tuple(torch.zeros(self._kv_n * H * D, dtype=dt, device=dev) for _ in "kv")
                    for H, D in self.kv_shapes]
        self._st = [(torch.zeros((R, *self.net.state_shape), dtype=torch.float32, device=dev),
                     torch.zeros((R, *self.net.tail_shape), dtype=dt, device=dev)) for _ in range(self.net.n_gdn)]

    def _root_views(self, L2):
        """Per attention layer (K, V) [rows, kv heads, L2, d] over the flat buffers."""
        n = self._kv_n // L2
        return [(k[: n * H * L2 * D].view(n, H, L2, D), v[: n * H * L2 * D].view(n, H, L2, D))
                for (k, v), (H, D) in zip(self._kv, self.kv_shapes)]

    def _outs(self, b, L):
        """The pass-1 outputs of b rows of length L (None, None for more than TREE_ROWS[-1] rows)."""
        if self._kv is None or b > self.TREE_ROWS[-1]:
            return None, None
        kv = [(k[:b, :, :L], v[:b, :, :L]) for k, v in self._root_views(self._root_len(L))]
        return kv, [(s[:b], c[:b]) for s, c in self._st]

    def _fresh(self, b, L):
        """Pass-1 outputs for b roots of L tokens beyond the graphs (one uncaptured forward)."""
        dt, dev = self._dtype, self.dev
        kv = [tuple(torch.zeros((b, H, L, D), dtype=dt, device=dev) for _ in "kv") for H, D in self.kv_shapes]
        st = [(torch.zeros((b, *self.net.state_shape), dtype=torch.float32, device=dev),
               torch.zeros((b, *self.net.tail_shape), dtype=dt, device=dev)) for _ in range(self.net.n_gdn)]
        return kv, st

    # -- graphs --------------------------------------------------------------------------------------------------
    def _graph(self, b, L):
        """Plain graph (b rows, length L): (graph, ids, real tokens per row, hidden states [b, L, D]); with at most
        TREE_ROWS[-1] rows it also writes the roots' keys, values and recurrent states (pass 1)."""
        key = ("p", b, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        lens = torch.full((b,), L, dtype=torch.long, device=self.dev)
        out = self._shared_out("plain", max(x * y for x, y in self.grid(self.PLAIN_LENS)) * self._hidden_size,
                               self._dtype)[: b * L * self._hidden_size].view(b, L, -1)
        kv, st = self._outs(b, L)
        fwd = self.fwd

        def plain(i, n):
            return _copy_same(out, fwd(i, n, kv, st))
        return self._capture(key, plain, ids, lens)

    def _sfx_ok(self, b, S, L2):
        return b * S <= self.TOKEN_BUDGET and b * L2 <= self.PASS2_KEYS and self._kv_n // L2 >= 1

    def _graph_sfx(self, b, S, L2):
        """Pass-2 graph (b leaf rows of S tokens over roots of bucket L2): (graph, ids, positions, root row of each
        row, root length of each row, hidden states [b, S, D])."""
        key = ("x", b, S, L2)
        if key in self.graphs:
            return self.graphs[key]
        dev = self.dev
        ids = torch.full((b, S), self.tok.pad_token_id, dtype=torch.long, device=dev)
        pos = torch.arange(L2, L2 + S, device=dev)[None].expand(b, S).contiguous()
        unit = torch.zeros(b, dtype=torch.long, device=dev)
        plen = torch.ones(b, dtype=torch.long, device=dev)
        top = max(x * y for x in self.LEAF_ROWS for y in self.LEAF_LENS for z in self.ROOT_LENS
                  if self._sfx_ok(x, y, z))
        D = self._hidden_size
        out = self._shared_out("pass2", top * D, self._dtype)[: b * S * D].view(b, S, D)
        kv, st = self._root_views(L2), self._st
        sfx = self.fwd_sfx

        def suffix(i, p, u, n):
            return _copy_same(out, sfx(i, p, u, n, kv, st))
        return self._capture(key, suffix, ids, pos, unit, plen)

    def warm(self):
        """Capture every plain graph of the grid and, with `shared`, the pass-2 graphs; warmed up in grid order and
        captured the largest first (_capture_deferred). Then the uncaptured shapes beyond the largest graph run once
        (their kernels compile at first use)."""
        self._deferred = {}
        for b, L in self.grid(self.PLAIN_LENS):
            self._graph(b, L)
        if self.shared:
            for L2 in self.ROOT_LENS:
                for S in self.LEAF_LENS:
                    for b in self.LEAF_ROWS:
                        if self._sfx_ok(b, S, L2):
                            self._graph_sfx(b, S, L2)
        self._capture_deferred()
        self._warmed = True
        pad = self.tok.pad_token_id
        for n in (self.PLAIN_LENS[-1] + 64, self.PLAIN_LENS[-1] + 65):  # Triton specialises lengths divisible by 16
            if self.model.config.max_position_embeddings < n + 8:
                break
            root = [pad] * n
            self.run_shared([([root + [5]], [[10, 11]])] + ([([root + [5, 6], root + [6, 5]], [[10, 11], [10, 11]])]
                                                            if self.shared else []))
        torch.cuda.empty_cache()

    # -- forwards ------------------------------------------------------------------------------------------------
    def _count(self, kind, graph, shape=None):
        k = f"{kind} {'graph' if graph else 'eager'}"
        self.counts[k] = self.counts.get(k, 0) + 1
        if shape is not None and not graph:
            self.counts.setdefault(f"{kind} eager shapes", set()).add(shape)

    def _rows_in(self, rows, L, b=None):
        """Token lists -> ids [b, L] padded on the right, real tokens per row [b] (padding rows: 0)."""
        b = b or len(rows)
        ids = np.full((b, L), self.tok.pad_token_id, dtype=np.int64)
        lens = np.zeros(b, dtype=np.int64)
        for r, t in enumerate(rows):
            ids[r, : len(t)] = t
            lens[r] = len(t)
        return torch.from_numpy(ids), torch.from_numpy(lens)

    def _plain_hidden(self, packs, idx):
        """Single-prompt units -> hidden states at their readout positions: a plain graph, or one uncaptured forward
        beyond the largest graph (or without CUDA graphs)."""
        n = max(len(packs[k][0]) for k in idx)
        rows = [packs[k][0] for k in idx]
        self._count("plain", n <= self.PLAIN_LENS[-1] and self.pool is not None)
        if n > self.PLAIN_LENS[-1] or self.pool is None:
            ids, lens = self._rows_in(rows, n)
            h = self.fwd(self._to_dev(ids), self._to_dev(lens))
        else:
            L, b = self._bucket(n, self.PLAIN_LENS), self._bucket(len(idx), self.BATCHES)
            g, s_ids, s_lens, h = self._graph(b, L)
            ids, lens = self._rows_in(rows, L, b)
            s_ids.copy_(self._to_dev(ids), non_blocking=True)
            s_lens.copy_(self._to_dev(lens), non_blocking=True)
            g.replay()
        r = [j for j, k in enumerate(idx) for _ in packs[k][3]]
        c = [i for k in idx for i in packs[k][3]]
        return h[self._to_dev(torch.tensor(r)), self._to_dev(torch.tensor(c))]

    @staticmethod
    def _leaves(pack, P):
        """The trie of a packed unit below its root (P tokens) as one row per leaf block: the token indices of the
        blocks on the path from the root to the leaf -> (rows, readouts: per prompt (row, column), row None for a
        readout inside the root, column then its index)."""
        t, pp, sg, last, par = pack
        start, end = {}, {}
        for i, s in enumerate(sg):
            start.setdefault(s, i)
            end[s] = i + 1
        kids = [[] for _ in par]
        for blk, p in enumerate(par):
            if p >= 0:
                kids[p].append(blk)
        rows, first = [], {}  # first: block -> (the first row through it, its offset there)
        for leaf in (x for x in range(1, len(par)) if not kids[x]):  # depth-first order: parents before children
            path, p = [], leaf
            while p > 0:
                path.append(p)
                p = par[p]
            ix = []
            for blk in reversed(path):
                first.setdefault(blk, (len(rows), len(ix)))
                ix += range(start[blk], end[blk])
            rows.append(ix)
        reads = [(None, i) if i < P else (first[sg[i]][0], first[sg[i]][1] + i - start[sg[i]]) for i in last]
        return rows, reads

    @torch.no_grad()
    def _tree_hidden(self, packs, idx, keys=None, hits=None):
        """Branched units -> hidden states at their readout positions. Pass 1: the roots in a plain graph that also
        writes their keys, values and recurrent states (beyond the largest graph: one uncaptured forward into fresh
        buffers). Pass 2: one row per leaf of each unit's trie, over its root, in pass-2 graphs of up to LEAF_ROWS[-1]
        rows (uncaptured beyond them)."""
        P = [packs[k][2].index(1) for k in idx]  # block 0 (the root) comes first and is contiguous
        b = len(idx)
        roots = [packs[k][0][:p] for k, p in zip(idx, P)]
        captured = self.pool is not None and max(P) <= self.PLAIN_LENS[-1] and b <= self.TREE_ROWS[-1]
        self._count("pass1", captured)
        if captured:
            L, B = self._bucket(max(P), self.PLAIN_LENS), self._bucket(b, self.BATCHES)
            L2 = self._root_len(L)
            g, s_ids, s_lens, h1 = self._graph(B, L)
            ids, lens = self._rows_in(roots, L, B)
            s_ids.copy_(self._to_dev(ids), non_blocking=True)
            s_lens.copy_(self._to_dev(lens), non_blocking=True)
            g.replay()
            kv, st = self._root_views(L2), self._st
        else:
            L2 = max(P)
            kv, st = self._fresh(b, L2)
            ids, lens = self._rows_in(roots, L2)
            h1 = self.fwd(self._to_dev(ids), self._to_dev(lens), kv, st)
        rows, sel = [], []  # rows: (unit, token indices); sel: per readout (pass, row, column), in unit order
        for r, (k, p) in enumerate(zip(idx, P)):
            leaf, reads = self._leaves(packs[k], p)
            sel += [(1, r, c) if q is None else (2, len(rows) + q, c) for q, c in reads]
            rows += [(r, ix) for ix in leaf]
        out = torch.empty((len(sel), h1.shape[-1]), dtype=h1.dtype, device=self.dev)
        j1 = [n for n, x in enumerate(sel) if x[0] == 1]
        if j1:
            out[self._to_dev(torch.tensor(j1))] = h1[self._to_dev(torch.tensor([sel[n][1] for n in j1])),
                                                     self._to_dev(torch.tensor([sel[n][2] for n in j1]))]
        for part in self._parts([len(ix) for _, ix in rows], L2):  # leaf rows by length, within the budgets
            n = len(rows[part[-1]][1])
            S = next((x for x in self.LEAF_LENS if n <= x), None)
            Bp = self._bucket(len(part), self.LEAF_ROWS)
            graph = None  # captured at start-up while memory allowed (_capture_deferred); without warm(), as first used
            if captured and S is not None and self._sfx_ok(Bp, S, L2):
                graph = self.graphs.get(("x", Bp, S, L2)) if self._warmed else self._graph_sfx(Bp, S, L2)
            self._count("pass2", graph is not None, (len(part), n, L2))
            if graph is None:  # bucketed rows and lengths: the shapes fla's kernels were tuned and compiled for
                S = -(-n // self.SUFFIX_EAGER) * self.SUFFIX_EAGER
            ids2 = np.full((Bp, S), self.tok.pad_token_id, dtype=np.int64)
            pos2 = np.tile(np.arange(L2, L2 + S, dtype=np.int64), (Bp, 1))
            unit2 = np.zeros(Bp, dtype=np.int64)
            plen2 = np.ones(Bp, dtype=np.int64)
            for j, q in enumerate(part):
                r, ix = rows[q]
                t, pp = packs[idx[r]][0], packs[idx[r]][1]
                m = len(ix)
                ids2[j, :m] = [t[i] for i in ix]
                pos2[j, :m] = [pp[i] for i in ix]
                pos2[j, m:] = pp[ix[-1]] + 1 + np.arange(S - m)
                unit2[j], plen2[j] = r, P[r]
            if graph is not None:
                g2, s_ids, s_pos, s_unit, s_plen, h2 = graph
                for dst, src in ((s_ids, ids2), (s_pos, pos2), (s_unit, unit2), (s_plen, plen2)):
                    dst.copy_(self._to_dev(src), non_blocking=True)
                g2.replay()
            else:
                h2 = self.fwd_sfx(self._to_dev(ids2), self._to_dev(pos2), self._to_dev(unit2), self._to_dev(plen2),
                                  kv, st)
            at = {q: j for j, q in enumerate(part)}
            j2 = [n for n, x in enumerate(sel) if x[0] == 2 and x[1] in at]
            if j2:  # read out before the next replay overwrites the shared output
                out[self._to_dev(torch.tensor(j2))] = h2[self._to_dev(torch.tensor([at[sel[n][1]] for n in j2])),
                                                         self._to_dev(torch.tensor([sel[n][2] for n in j2]))]
        return out

    def _parts(self, lens, L2):
        """Leaf rows (their lengths) -> pass-2 forwards: sorted by length, each part at most LEAF_ROWS[-1] rows, rows x
        root bucket within PASS2_KEYS, and (bucketed) rows x length within TOKEN_BUDGET unless a row is alone."""
        per = min(self.LEAF_ROWS[-1], max(1, self.PASS2_KEYS // L2))
        out, cur = [], []
        for q in sorted(range(len(lens)), key=lens.__getitem__):
            S = next((x for x in self.LEAF_LENS if lens[q] <= x), -(-lens[q] // self.SUFFIX_EAGER) * self.SUFFIX_EAGER)
            if cur and (len(cur) == per or self._bucket(len(cur) + 1, self.LEAF_ROWS) * S > self.TOKEN_BUDGET):
                out.append(cur)
                cur = []
            cur.append(q)
        return out + ([cur] if cur else [])

    def run_long(self, groups, policy=None, between=None):
        raise RuntimeError("--long-lane is not available for Qwen3.5 (Gated DeltaNet) models")
