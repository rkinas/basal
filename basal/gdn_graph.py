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
nvfp4.
- **State cache** (--state-cache-gb, 2.5): a request cut at its state boundary keeps its root as a full-state checkpoint:
  the attention layers' keys and values of the state and, per Gated DeltaNet layer, the recurrent state and convolution
  tail after its last token. A later request with the same state copies them where pass 1 would have written them and
  runs pass 2 only. A state that begins with a whole cached one (a growing agent trace) continues it
  (forward_continue) and replaces it; one that differs inside the cached tokens runs again (a recurrent state cannot
  be cut there).
"""
import copy
import functools
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import adapter as adapter_module
from .engine import (GraphBackend, MPSBackend, PREFILL, StateCache, _copy_same, _varlen_attention, compact_refused,
                     graph_memory_bytes, persist_compile_cache, set_attention, torch_varlen_available)

# the decoders this forward implements: Qwen3.5 text models (dense MLP) of these layer kinds
GDN_MODEL_TYPES = ("qwen3_5_text", "qwen3_5")
GDN_LAYERS = {"linear_attention", "full_attention"}


def _decoder_config(model_dir, strict=False):
    """config.json of a model directory, the decoder part (text_config of a multimodal wrapper); None if unreadable
    (strict: one that exists but cannot be read or parsed raises instead)."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as e:
        if strict and not isinstance(e, FileNotFoundError):
            raise
        return None
    return cfg.get("text_config") or cfg


def is_gdn(model_dir, strict=False):
    """A Qwen3.5 decoder of Gated DeltaNet and full-attention layers (what GDNGraphBackend serves), from config.json.
    Other recurrent models (Qwen3-Next, Mamba and hybrids) stay refused by the graph engine."""
    c = _decoder_config(model_dir, strict)
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
            readout_layer="off", template="off", device=True, disk=False):
    """Why GDNGraphBackend cannot serve a Qwen3.5 model with these settings (None: it can), before anything is loaded.
    Only quantisation is refused: the state cache, the prefix cache, compact rows, template prefixes and the long lane
    run since 2.5 (their arguments stay for the callers). device: also check for a CUDA GPU and fla (the server checks
    the settings before the download, the engine both).
    --lean / --tree flex change nothing here, as for basal-omni's Gemma 4 forward: Qwen35Graph is lean already (one
    compiled function per layer kind, called with each layer's weights), and its pass 2 has no trie mask (one row per
    leaf). disk (--state-cache-dir): its cached states carry recurrent states, not the files' keys / values layout."""
    if disk:
        return ("--state-cache-dir is not available for Qwen3.5 (Gated DeltaNet) models: their cached states carry "
                "recurrent states, not only attention keys and values; serve them without it")
    if quant:
        return (f"--mode {quant} is not available for Qwen3.5 (Gated DeltaNet) models: the CUDA graph engine serves "
                "them in bf16 (--mode fast); --mode vllm runs their FP8 checkpoints")
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


def conv_kernel(cuda):
    """The Gated DeltaNet convolution's kernel -> (fla's causal_conv1d or None, its name): on CUDA fla's where this fla
    has it, else torch's conv1d (_conv; cuDNN on CUDA). torch's rounds differently (SiLU on the bf16 output, not in fp32
    before the rounding) from fla's, the one --mode fast was checked with: on CUDA that fallback is said on stderr."""
    conv = fla_conv1d() if cuda else None
    if cuda and conv is None:
        print("basal: warning: this flash-linear-attention has no causal_conv1d (fla.modules.conv.causal_conv1d): the "
              "Gated DeltaNet convolution runs on torch's conv1d (cuDNN), which rounds differently from fla's kernel "
              "--mode fast was checked with (flash-linear-attention 0.5.2)", file=sys.stderr, flush=True)
    return conv, "fla causal_conv1d" if conv is not None else "torch conv1d"


def fla_varlen():
    """fla's pieces for compact rows (prompts packed back to back in one row): the forward of its Gated DeltaNet rule
    and its l2 norm, which take the chunk indices of the packed prompts as an argument; None when this fla has no such
    entry points (compact rows are refused then)."""
    try:
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    except Exception:
        return None
    if "chunk_indices" not in inspect.signature(chunk_gated_delta_rule_fwd).parameters:
        return None
    return chunk_gated_delta_rule_fwd, l2norm_fwd


def _rule_varlen(q, k, v, g, beta, s0, cu, ci, fns):
    """fla's chunk_gated_delta_rule (l2-normalised q / k, as use_qk_l2norm_in_kernel) over compact rows [1, T, heads,
    d]: cu [prompts + 1] int32 cumulative token counts, every prompt's start a multiple of 64, ci [T / 64, 2] int32 the
    (prompt, chunk) of every 64-token chunk, s0 [prompts, heads, dk, dv] fp32 the states the prompts continue (None:
    zero). The forward of fla's autograd function with the chunk indices given: fla would compute them from cu with a
    host sync (no CUDA graph), and keep them per tensor identity (a graph's static cu would get stale ones)."""
    fwd, l2norm = fns
    q, _ = l2norm(q.contiguous())
    k, _ = l2norm(k.contiguous())
    o = fwd(q, k, v.contiguous(), g.contiguous(), beta.contiguous(), k.shape[-1] ** -0.5, s0, False, cu_seqlens=cu,
            chunk_indices=ci)[1]
    return o.to(q.dtype)


def _conv_varlen(qkv, w, c0, sizes, conv, cu, ci):
    """_conv_fla over compact rows (cu, ci as in _rule_varlen); c0 [prompts, kernel - 1, channels]: each prompt's
    convolution tail (None: zeros)."""
    init = None if c0 is None else F.pad(c0.transpose(1, 2), (1, 0))
    out, a = [], 0
    for m in sizes:
        y, _ = conv(qkv[..., a: a + m], w[a: a + m, 0], initial_state=None if init is None else
                    init[:, a: a + m].contiguous(), activation="silu", cu_seqlens=cu, chunk_indices=ci)
        out.append(y)
        a += m
    return out


def _spans(cu):
    """Compact rows' cumulative counts -> (prompt, start, end) of every non-empty prompt."""
    c = [int(x) for x in cu]
    return [(s, a, b) for s, (a, b) in enumerate(zip(c, c[1:])) if b > a]


def _varlen_sdpa(q, k, v, cu, scale, layer=0, impl=None):
    """engine._varlen_attention with one SDPA call per packed prompt (its queries attend causally to its keys,
    bottom-right: a template prefix's keys first, as gathered): fp32 compact rows (the FlashAttention kernels take
    bf16 / fp16) and the CPU (tests). Reads cu on the host, so no CUDA graph."""
    T, G = q.shape[2], q.shape[1] // k.shape[1]
    k, v = k[0].transpose(0, 1), v[0].transpose(0, 1)
    if isinstance(cu, tuple):
        cu, cu_k, gather, prefix = cu
        pk, pv = prefix[layer]
        k, v = torch.cat([pk, k]).index_select(0, gather), torch.cat([pv, v]).index_select(0, gather)
    else:
        cu_k = cu
    out = torch.zeros((T, q.shape[1], q.shape[3]), dtype=q.dtype, device=q.device)
    for s, a, b in _spans(cu):
        ka, kb = int(cu_k[s]), int(cu_k[s + 1])
        n, m = b - a, kb - ka
        kk, vv = (t[ka:kb].transpose(0, 1).repeat_interleave(G, 0) for t in (k, v))
        allow = torch.arange(m, device=q.device)[None] <= (m - n + torch.arange(n, device=q.device))[:, None]
        out[a:b] = F.scaled_dot_product_attention(q[0, :, a:b], kk, vv, attn_mask=allow, scale=scale).transpose(0, 1)
    return out[None]


# fla's Triton kernels pick their launch config by timing the candidates at the first launch of every shape key
# (triton.autotune), so two starts of the server could pick different configs, round differently and give different
# probabilities (and a new shape stalled a request while it was timed). FLA_CONFIGS holds the configs an H100 picked for
# the keys basal's pass shapes use (fla_record): with it every recorded fla kernel takes the recorded config of its
# key, and its most common one for any other key, without timing (a kernel not in the table autotunes as before). Only
# on the GPU and fla version it was recorded with.
FLA_CONFIGS = Path(__file__).parent / "fla_configs.json"


def _fla_autotuners():
    """fla's Triton autotuners: {"module.kernel": Autotuner}, after importing the modules basal's passes use (the
    convolution's are imported only at its first call)."""
    import gc
    import importlib
    from triton.runtime.autotuner import Autotuner
    for name in ("fla.ops.gated_delta_rule", "fla.modules.conv.triton", "fla.modules.l2norm"):
        try:
            importlib.import_module(name)
        except Exception:
            pass
    out = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, Autotuner) and str(getattr(o.base_fn, "__module__", "")).startswith("fla."):
                out[f"{o.base_fn.__module__}.{o.base_fn.__qualname__}"] = o
        except ReferenceError:  # a weakref.proxy whose object is gone: isinstance() reads its __class__
            continue
    return out


def _config_dict(c):
    return dict(kwargs=c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages, num_ctas=c.num_ctas,
                maxnreg=c.maxnreg)


def fla_record(path):
    """Write the configs fla's autotuners picked so far (after a warm-up of every pass shape) as FLA_CONFIGS."""
    import fla
    from collections import Counter
    kernels = {}
    for name, t in sorted(_fla_autotuners().items()):
        # (a key value JSON cannot hold is written as its str: that key then gets the kernel's default when pinned)
        entries = [json.loads(json.dumps([list(k), _config_dict(c)], default=str)) for k, c in t.cache.items()]
        if entries:
            common = Counter(json.dumps(c, sort_keys=True) for _, c in entries).most_common(1)[0][0]
            kernels[name] = dict(default=json.loads(common), keys=entries)
    Path(path).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), fla=fla.__version__, kernels=kernels),
                                     indent=1, sort_keys=True))
    return len(kernels)


class _PinnedCache(dict):
    """An autotuner's cache that has every key: the recorded config, else the kernel's default (nothing is timed)."""

    def __init__(self, entries, default):
        super().__init__(entries)
        self.default = default

    def __contains__(self, key):
        return True

    def __missing__(self, key):
        return self.default


def fla_pin(path=None):
    """Give fla's autotuners the recorded configs (see FLA_CONFIGS; a prepared start-up's own table replaces it,
    prepared.load) -> the number of kernels pinned, 0 when the table is for another GPU or fla version (they autotune
    as before), or missing. Only the kernels in the table are pinned: a kernel the recording never launched has no
    measured config (its first candidate may not even fit the GPU), so it autotunes as without a table.
    BASAL_FLA_PIN=0 turns it off."""
    path = path or FLA_CONFIGS
    if os.environ.get("BASAL_FLA_PIN", "1") != "1" or not Path(path).exists() or not torch.cuda.is_available():
        return 0
    import fla
    from triton.runtime.autotuner import Config
    table = json.loads(Path(path).read_text())
    if table.get("gpu") != torch.cuda.get_device_name() or table.get("fla") != fla.__version__:
        return 0
    n = 0
    def cfg(d):
        return Config(d["kwargs"], num_warps=d["num_warps"], num_stages=d["num_stages"], num_ctas=d["num_ctas"],
                      maxnreg=d["maxnreg"])
    for name, t in _fla_autotuners().items():
        e = table["kernels"].get(name)
        if e is not None:
            t.cache = _PinnedCache({tuple(k): cfg(c) for k, c in e["keys"]}, cfg(e["default"]))
            n += 1
    return n


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
        conv, self.conv_kernel = conv_kernel(cuda)  # on CUDA fla's kernel, uncompiled (Triton), in both modes
        self.conv, self.conv_sizes, self.cuda = conv, cs, cuda
        # compact rows (GDNGraphBackend sets them): the attention kernel, fla's varlen pieces, the template prefixes
        self.varlen = self.fla_v = self.template_kv = self.template_st = None
        self.f_eager = dict(fns)  # uncompiled (the long lane: run beside the graphs, from another thread)
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
            fns["conv"] = self.f_eager["conv"] = lambda qkv, w, c0: _conv_fla(qkv, w, c0, cs, conv)
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
            # tokens lens - n .. lens - 1; a row that continues a tail (c0) reads its first ones there
            src = qkv if c0 is None else torch.cat([c0.to(qkv.dtype), qkv], 1)
            ix = lens[:, None] - n + torch.arange(n, device=h.device)[None] + (0 if c0 is None else n)
            t = src.gather(1, ix.clamp_min(0)[..., None].expand(-1, -1, src.shape[-1]))
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

    between = None  # the long lane's copy (GDNGraphBackend._long_net): called with i before layer i

    def _layers(self, h, gdn, att):
        ia = ig = 0
        for j, (x, kind) in enumerate(zip(self.args, self.kinds)):
            if self.between is not None:
                self.between(j)
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


    @torch.no_grad()
    def forward_continue(self, ids, n, kv0, st0, kv, st):
        """Rows [b, T] that continue a cached state of n tokens (positions n ..): kv0 per attention layer (K, V)
        [b, kv heads, n, d], st0 per Gated DeltaNet layer (state, tail); fills kv (K, V) [b, kv heads, T, d] with the
        rows' own keys and values and st (state, tail) after their last token -> normalised hidden states [b, T, D]."""
        b, T = ids.shape
        dev = ids.device
        h = self.m.embed_tokens(ids)
        cos, sin = self._rope(h, torch.arange(n, n + T, device=dev)[None].expand(b, T))
        G = self.heads // self.kv_shapes[0][0] if kv else 1
        own = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        allow = torch.cat([torch.ones(T, n, dtype=torch.bool, device=dev), own], -1)
        allow = allow[None, None, None].expand(b, 1, G, T, n + T).reshape(b, 1, G * T, n + T)
        lens = torch.full((b,), T, dtype=torch.long, device=dev)
        return self._layers(
            h,
            lambda x, h, i: self._gdn(x, h, None, lens, st[i], s0=st0[i][0], c0=st0[i][1]),
            lambda x, h, i: self._attention(x, h, cos, sin, kv[i], prefix=(*kv0[i], allow)))


    def _gdn_varlen(self, x, h, cu, ci, st):
        """_gdn over compact rows [1, T] (cu, ci: _rule_varlen's; ci None: no CUDA, each prompt on its own with
        transformers' kernel, as _gdn): every prompt from its own recurrent state and convolution tail, st = (s0
        [prompts, ...], c0 [prompts, ...]), or from zero (st None)."""
        a, f = x["a"], self.f
        T = h.shape[1]
        qkv, z, beta, g = f["gdn_in"](h, x["nw"], x["w_in"], a.A_log, a.dt_bias, None)
        w = a.conv1d.weight
        s0, c0 = st if st is not None else (None, None)
        r = a.num_v_heads // a.num_k_heads

        def heads(q, k, v, n):
            q, k, v = (t.reshape(1, n, -1, d) for t, d in ((q, a.head_k_dim), (k, a.head_k_dim), (v, a.head_v_dim)))
            return (q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2), v) if r > 1 else (q, k, v)
        if ci is not None:
            q, k, v = heads(*_conv_varlen(qkv, w, c0, self.conv_sizes, self.conv, cu, ci), T)
            o = _rule_varlen(q, k, v, g, beta, s0, cu, ci, self.fla_v)
        else:
            parts = []
            for i, lo, hi in _spans(cu):
                q, k, v = heads(*f["conv"](qkv[:, lo:hi], w, None if c0 is None else c0[i: i + 1]), hi - lo)
                parts.append(self.rule(q, k, v, g=g[:, lo:hi], beta=beta[:, lo:hi], output_final_state=False,
                                       initial_state=None if s0 is None else s0[i: i + 1],
                                       use_qk_l2norm_in_kernel=True)[0])
            o = torch.cat(parts, 1)
        return f["gdn_out"](h, o, z.reshape(1, T, -1, a.head_v_dim), a.norm.weight, a.out_proj.weight)

    def _attention_varlen(self, x, h, cos, sin, cu, i, ro=None):
        """_attention over compact rows: each prompt attends causally to itself (and to its template prefix's keys and
        values, cu = (cu, cu_k, gather, template keys / values) as for engine._varlen_attention); ro: the readout
        positions, where alone the output gate, o_proj and the rest of the layer run (--readout-layer on)."""
        a = x["a"]
        q, k, v, gate = self.f["att_in"](h, x["nw"], x["w_qkv"], a.q_norm.weight, a.k_norm.weight, cos, sin)
        o = self.varlen(q, k, v, cu, a.scaling, i)  # [1, T, heads, d]
        if ro is not None:
            h, o, gate = h[:, ro], o[:, ro], gate[:, ro]
        return self.f["att_out"](h, o.transpose(1, 2), gate, a.o_proj.weight)

    @torch.no_grad()
    def forward_compact(self, ids, pos, cu, ci=None, tix=None, cu_k=None, gather=None, ro=None):
        """Compact rows (--batches compact): prompts packed back to back in one row [1, T] (positions from 0, or from
        the end of their template prefix; cu: cumulative token counts, int32; on CUDA every prompt starts on a
        64-token chunk and ci [T / 64, 2] gives fla the chunk indices). Each prompt attends to itself only and its Gated
        DeltaNet layers start from zero, or with tix [prompts] (--template-prefix on) from template tix's checkpoint:
        its keys / values (template_kv, gathered by cu_k / gather) and states / tails (template_st; the last row is
        zeros). ro (--readout-layer on, a model whose last layer is attention): the readout positions [R], where alone
        the last layer runs after its attention -> [1, R, D]; else the normalised hidden states [1, T, D]."""
        h = self.m.embed_tokens(ids)
        cos, sin = self._rope(h, pos)
        st, cu_a = [None] * self.n_gdn, cu
        if tix is not None:
            st = [(s.index_select(0, tix), c.index_select(0, tix)) for s, c in self.template_st]
            cu_a = (cu, cu_k, gather, self.template_kv)
        n_att = len(self.kv_shapes)
        return self._layers(
            h,
            lambda x, h, i: self._gdn_varlen(x, h, cu, ci, st[i]),
            lambda x, h, i: self._attention_varlen(x, h, cos, sin, cu_a, i, ro if i == n_att - 1 else None))


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
    row_per_leaf = True  # pass 2 runs one row per leaf (basal-serve cuts only single prompts at a template prefix)
    _warmed = False  # warm() ran: the pass-2 graphs it did not capture (GPU memory) stay uncaptured

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, shared=True, warm=True, attn="auto",
                 merge="auto", lean=False, tree="dense", state_cache_gb=0, prefix=False, batches="default",
                 readout_layer="off", template="off", gemma_attn="auto", fp8_scaling="auto", fp8_bf16_layers=2,
                 nvfp4_amax=None, nvfp4_scope="all", quant_load="gpu", state_cache_bits=16, graph_memory_gb=None,
                 device=None, global_attn="flex", adapter=None, state_cache_dir=None, state_cache_disk_gb=20.0,
                 state_cache_signature=None):
        # (fp8_scaling .. quant_load: GraphBackend's quantisation options, which basal-serve passes to every graph
        # engine; quantisation is refused here. graph_memory_gb: the pass-2 graphs captured, _pin_memory). device:
        # "mps" / "cpu" (GDNMPSBackend: no CUDA graphs, uncompiled). state_cache_dir ..: refused (refused(disk=))
        if compile == "layer":
            raise ValueError(f"{model_dir}: --compile-scope layer: the bf16 Llama forward only (Qwen3.5's compiled "
                             "code is per layer kind already)")
        if compile:
            persist_compile_cache()  # before anything compiles (GraphBackend.__init__)
        why = refused(quant, state_cache_gb, prefix, batches, lean, tree, readout_layer, template,
                      device is None, disk=state_cache_dir is not None)  # before loading
        if state_cache_bits != 16 and not why:
            why = "--state-cache-bits 8 is for attention keys and values; recurrent states are cached as computed"
        if global_attn != "flex" and not why:
            why = f"--global-attn {global_attn}: basal-omni (Gemma 4) only"
        if why:
            raise ValueError(f"{model_dir}: {why}")
        cuda = torch.cuda.is_available() if device is None else device == "cuda"
        if cuda:
            torch.backends.cuda.enable_cudnn_sdp(False)  # before capture: a captured graph keeps its attention kernel
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")  # (GraphBackend.__init__)
        self.fp8_checkpoint = False
        self.graph_memory = graph_memory_bytes(graph_memory_gb)
        self.attn = set_attention(attn)  # the causal plain rows; pass 2 runs SDPA with its mask
        # --batches compact (opt-in; auto keeps padded rows here): single prompts packed into one row, each starting on
        # a 64-token chunk (fla's), with FlashAttention 3's varlen kernel or torch's; without CUDA (tests) per prompt
        self.compact = batches == "compact"
        if self.compact and cuda:
            why = None if self.attn == "fa3" else torch_varlen_available(getattr(torch, dtype))
            if why and dtype != "float32":
                raise ValueError(f"{model_dir}: {compact_refused(why)}")
            if fla_varlen() is None:
                raise ValueError(f"{model_dir}: --batches compact needs flash-linear-attention's "
                                 "chunk_gated_delta_rule forward with chunk indices (flash-linear-attention 0.5)")
            if fla_conv1d() is None:  # (compact rows call it with the chunk indices; there is no torch fallback)
                raise ValueError(f"{model_dir}: --batches compact needs flash-linear-attention's causal_conv1d "
                                 "(fla.modules.conv.causal_conv1d, flash-linear-attention 0.5.2)")
            # (fp32, which the FlashAttention kernels do not take: one SDPA call per prompt, no graph; the gates' fp32
            # rows)
            self.compact_kernel = ("sdpa per prompt" if dtype == "float32" else "fa3" if self.attn == "fa3"
                                   else "torch varlen")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
        self.adapter_modules = adapter_module.merge_torch(self.model, adapter) if adapter else 0  # --adapter
        self.model = self.model.to("cuda" if cuda else device or "cpu").eval()
        self.dev = next(self.model.parameters()).device
        self.prefill = PREFILL
        self.net = Qwen35Graph(self.model, compile=compile and cuda)
        self.conv_kernel = self.net.conv_kernel  # (reported at start-up)
        self.fla_pinned = fla_pin() if cuda else 0  # (after Qwen35Graph imported fla's kernels)
        self.fwd, self.fwd_sfx = self.net.forward, self.net.forward_suffix
        if self.compact:
            if cuda:
                import fla.utils  # fla keeps index tensors per argument identity: a graph's static cu would get stale
                fla.utils.FLA_DISABLE_TENSOR_CACHE = True  # ones (only compact rows pass cu_seqlens to fla)
                self.net.fla_v = fla_varlen()
                self.net.varlen = (_varlen_sdpa if self.compact_kernel == "sdpa per prompt" else
                                   functools.partial(_varlen_attention, impl=self.compact_kernel))
            else:
                self.compact_kernel, self.net.varlen = "sdpa per prompt", _varlen_sdpa
            # --readout-layer on: the last layer after its attention at the readout positions only (an attention
            # layer: Qwen3.5's last is)
            self.readout_layer = readout_layer in ("on", "auto") and self.net.kinds[-1] == "full_attention"
            self.fwd_c = self.net.forward_compact
        self.merged, self.lean, self.flex = len(self.net.args), None, False
        self.shared = shared
        self._wrows = {}
        if batches == "fine":
            self.BATCHES = self.FINE_BATCHES
        self._alloc_kv()
        # --state-cache-gb: roots cut at the state boundary, kept as full-state checkpoints (keys / values + states);
        # --prefix-cache: also the template prefixes of short single-prompt requests (GraphBackend's budget)
        self.prefix = bool(prefix) and shared
        gb = max(state_cache_gb, self.PREFIX_GB if self.prefix else 0)
        self.state_cache = StateCache(gb * 2 ** 30) if gb and shared else None
        self.template = []  # --template-prefix (with compact rows): [(start in template_kv, prefix ids, its row)]
        if self.compact and template in ("on", "auto"):
            self._template_prefixes(model_dir)
        self.counts = {}  # forwards per kind: "plain" / "pass1" / "pass2", each "graph" or "eager" (gate scripts)
        self.pool = torch.cuda.graph_pool_handle() if cuda else None
        self.graphs = {}
        if warm and cuda:
            self.warm()
        self.one_forward = self.ONE_FORWARD
        self._shape_line()

    # -- buffers -------------------------------------------------------------------------------------------------
    def _root_len(self, L):
        """The pass-2 root bucket of a pass-1 length."""
        return self._bucket(L, self.ROOT_LENS)

    def _alloc_kv(self):
        """Pass-1 outputs of the plain graphs of at most TREE_ROWS[-1] rows (the others write nothing): per attention
        layer one flat buffer for keys and one for values, viewed per root bucket L2 as [rows, kv heads, L2, d] (a
        graph of length L <= L2 writes [:b, :, :L]); per Gated DeltaNet layer the recurrent states [rows, heads, dk,
        dv] (fp32) and convolution tails [rows, kernel - 1, channels]. Zeros, not empty: pass 2 reads the masked tail
        of a root's bucket too, and a masked NaN still poisons a readout (0 x NaN). With --graph-memory-gb the pass-2
        graphs captured follow it (_pin_memory)."""
        self._pin_memory()
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

    # -- compact rows (--batches compact) ------------------------------------------------------------------------
    CHUNK = 64  # fla's chunk: every prompt of a compact row starts on a multiple of it, so a row of T tokens has T / 64
    # chunks whatever its prompts' lengths (fixed shapes for the graphs); GraphBackend's steps, in multiples of 64
    COMPACT_LENS = (list(range(64, 2048, 64)) + list(range(2048, 4096, 128)) + list(range(4096, 8192, 256))
                    + list(range(8192, 12289, 512)))
    n_templates = 0

    def _compact_graphs(self):
        """Whether compact rows run in CUDA graphs (a FlashAttention varlen kernel, which reads cu on the GPU)."""
        return self.pool is not None and self.compact_kernel in ("fa3", "torch varlen")

    def _template_prefixes(self, model_dir):
        """--template-prefix: the tokens every prompt of a language starts with (GraphBackend._template_prefixes: chat
        template, system prompt and the user header up to the state, one token less) as full-state checkpoints,
        computed once: their keys / values in every attention layer (net.template_kv[layer] = (K, V) [prefix tokens
        of all languages, kv heads, d]) and their recurrent states and convolution tails (net.template_st[layer] =
        (states, tails) [languages + 1, ...], the last row zeros: a prompt without a template prefix). A compact
        prompt that starts with one runs only its own tokens from that checkpoint."""
        from .prompt import FORMATS, model_format, render
        fmt = model_format(model_dir)
        prefixes = []
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
            if n >= 2:
                prefixes.append(full[:n])
        self._set_templates(prefixes)

    def _set_templates(self, prefixes):
        """The checkpoints of these template prefixes (token id lists), as _template_prefixes describes."""
        kvs, sts, a = [], [], 0
        for p in prefixes:
            kv, st = self._fresh(1, len(p))
            self.net.forward(self._to_dev(torch.tensor([list(p)])), self._to_dev(torch.tensor([len(p)])), kv, st)
            self.template.append((a, tuple(p), len(sts)))
            kvs.append(kv)
            sts.append(st)
            a += len(p)
        if not self.template:
            return
        self.template.sort(key=lambda x: -len(x[1]))  # the longest match first
        self.template_tokens, self.template_max, self.n_templates = a, max(len(x[1]) for x in self.template), len(sts)
        self.net.template_kv = [tuple(torch.cat([kv[i][j][0].transpose(0, 1) for kv in kvs]).contiguous()
                                      for j in (0, 1)) for i in range(len(self.kv_shapes))]
        self.net.template_st = [tuple(torch.cat([st[i][j] for st in sts] + [torch.zeros_like(sts[0][i][j])])
                                      for j in (0, 1)) for i in range(self.net.n_gdn)]

    def _graph_c(self, T):
        """Compact graph of T tokens (a multiple of CHUNK): (graph, ids, positions, cumulative counts, chunk indices,
        [template rows, key counts, key gather,] [readout positions,] hidden states)."""
        key = ("c", T)
        if key in self.graphs:
            return self.graphs[key]
        dev, NS = self.dev, self.COMPACT_SEQS + 1  # (the padding after the last prompt is one more prompt)
        ids = torch.full((1, T), self.tok.pad_token_id, dtype=torch.long, device=dev)
        pos = torch.arange(T, device=dev)[None].contiguous()
        cu = torch.full((NS + 1,), T, dtype=torch.int32, device=dev)
        cu[0] = 0
        ci = torch.stack([torch.zeros(T // self.CHUNK, dtype=torch.int32, device=dev),
                          torch.arange(T // self.CHUNK, dtype=torch.int32, device=dev)], 1).contiguous()
        static = [ids, pos, cu, ci]
        if self.template:
            gather = torch.zeros(T + self.COMPACT_SEQS * self.template_max, dtype=torch.long, device=dev)
            gather[:T] = self.template_tokens + torch.arange(T, device=dev)
            static += [torch.full((NS,), self.n_templates, dtype=torch.long, device=dev), cu.clone(), gather]
        ro = None
        if self.readout_layer:
            ro = torch.zeros(self.COMPACT_SEQS, dtype=torch.long, device=dev)
            static.append(ro)
        D = self._hidden_size
        R = self.COMPACT_SEQS if ro is not None else T
        out = self._shared_out("compact", max(self.COMPACT_LENS) * D, self._dtype)[: R * D].view(1, R, D)
        fwd, tmpl = self.fwd_c, bool(self.template)

        def compact(*a):
            tm = a[4:7] if tmpl else (None,) * 3
            return _copy_same(out, fwd(*a[:4], *tm, ro=a[-1] if ro is not None else None))
        return self._capture(key, compact, *static)

    def warm(self):
        """Capture every plain graph of the grid and, with `shared`, the pass-2 graphs; warmed up in grid order and
        captured the largest first (_capture_deferred). Then the uncaptured shapes beyond the largest graph run once
        (their kernels compile at first use)."""
        self._deferred = {}
        if self.compact and self._compact_graphs():  # plain graphs then only for pass 1 of branched units
            for T in self.COMPACT_LENS:
                self._graph_c(T)
        for b, L in self.grid(self.PLAIN_LENS):
            if not self.compact or not self._compact_graphs() or self.shared and b <= self.TREE_ROWS[-1]:
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

    def _forwards(self, units, cached=()):
        """GraphBackend._forwards; with compact rows the single prompts go into compact rows (shortest first, up to
        COMPACT_SEQS prompts and COMPACT_LENS[-1] tokens, each prompt counted to its next CHUNK), and only a prompt
        beyond the largest compact graph runs alone."""
        if not self.compact:
            return GraphBackend._forwards(self, units, cached)
        packs = [u[0] for u in units]
        out, cur, tot = [], [], 0
        for k in sorted((k for k, pk in enumerate(packs) if len(pk[4]) == 1), key=lambda k: len(packs[k][0])):
            n = -(-len(packs[k][0]) // self.CHUNK) * self.CHUNK
            if n > self.COMPACT_LENS[-1]:
                out.append(([k], "plain"))
                continue
            if cur and (len(cur) == self.COMPACT_SEQS or tot + n > self.COMPACT_LENS[-1]):
                out.append((cur, "compact"))
                cur, tot = [], 0
            cur.append(k)
            tot += n
        out += [(cur, "compact")] if cur else []
        for kind in ("tree", "cached"):  # (GraphBackend._forwards)
            tree = [k for k, pk in enumerate(packs) if len(pk[4]) > 1 and (k in cached) == (kind == "cached")]
            roots = [packs[k][2].index(1) for k in tree]
            rest = [len(packs[k][0]) - r for k, r in zip(tree, roots)]
            out += [([tree[j] for j in idx], kind) for idx in self._chunks(roots, self.PLAIN_LENS, rest,
                                                                           self.TREE_ROWS[-1])]
        return out

    def _compact_hidden(self, packs, idx):
        """Single-prompt units packed into one compact row, each from a multiple of CHUNK (the rest of its chunk and the
        tail of the row: padding after its tokens, read by nobody), and from its template prefix's checkpoint when it
        starts with one (only its own tokens run) -> hidden states at their readout positions."""
        C, NS, pad = self.CHUNK, self.COMPACT_SEQS + 1, self.tok.pad_token_id
        rows = []  # per prompt: own tokens, positions, readout columns, template row (n_templates: none), prefix start
        for k in idx:
            t, pp, _, last, _ = packs[k]
            hit = next(((a, len(p), r) for a, p, r in self.template if len(t) > len(p) and tuple(t[: len(p)]) == p),
                       (0, 0, self.n_templates))
            P = hit[1]
            rows.append((t[P:], pp[P:], [i - P for i in last], hit))
        n = [-(-len(r[0]) // C) * C for r in rows]
        T = self._bucket(sum(n), self.COMPACT_LENS)
        h_ids = np.full(T, pad, dtype=np.int64)
        h_pos = np.zeros(T, dtype=np.int64)
        h_cu = np.full(NS + 1, T, dtype=np.int32)
        h_tix = np.full(NS, self.n_templates, dtype=np.int64)
        h_ro = np.zeros(self.COMPACT_SEQS, dtype=np.int64)
        h_cuk = np.zeros(NS + 1, dtype=np.int32)
        h_g = np.zeros(T + self.COMPACT_SEQS * max(1, getattr(self, "template_max", 0)), dtype=np.int64)
        cols, sel, o, ok = [], [], 0, 0  # (ok: keys so far, with template prefixes)
        for j, ((t, pp, last, (a, P, r)), m) in enumerate(zip(rows, n)):
            h_ids[o: o + len(t)] = t
            h_pos[o: o + len(t)] = pp
            h_pos[o + len(t): o + m] = pp[-1] + 1 + np.arange(m - len(t))
            h_cu[j], h_tix[j], h_ro[j] = o, r, o + len(t) - 1
            cols += [o + i for i in last]
            sel += [j] * len(last)
            h_cuk[j] = ok
            h_g[ok: ok + P] = np.arange(a, a + P)
            h_g[ok + P: ok + P + m] = getattr(self, "template_tokens", 0) + np.arange(o, o + m)
            ok += P + m
            o += m
        h_cu[len(idx)], h_cuk[len(idx)] = o, ok  # the padding: one more prompt from position 0, without a template
        h_pos[o:] = np.arange(T - o)
        h_g[ok: ok + T - o] = getattr(self, "template_tokens", 0) + np.arange(o, T)
        h_cuk[len(idx) + 1:] = ok + T - o
        spans = np.diff(h_cu) // C
        h_ci = np.stack([np.repeat(np.arange(NS, dtype=np.int32), spans),
                         np.concatenate([np.arange(x, dtype=np.int32) for x in spans])], 1)
        host = [h_ids[None], h_pos[None], h_cu, h_ci] + ([h_tix, h_cuk, h_g] if self.template else [])
        host += [h_ro] if self.readout_layer else []
        graph = self._compact_graphs()
        self._count("compact", graph)
        if graph:
            g, *static, out = self._graph_c(T)
            for dst, src in zip(static, host):
                dst.copy_(self._to_dev(src), non_blocking=True)
            g.replay()
        else:  # no graph (fp32, the tests): the forward itself; without CUDA no chunk indices (one prompt at a time)
            dev = [self._to_dev(torch.from_numpy(np.ascontiguousarray(x))) for x in host]
            tm = dev[4:7] if self.template else [None] * 3
            out = self.fwd_c(*dev[:3], dev[3] if self.dev.type == "cuda" else None, *tm,
                             ro=dev[-1] if self.readout_layer else None)
        if self.readout_layer:
            return out[0, self._to_dev(torch.tensor(sel))]
        return out[0, self._to_dev(torch.tensor(cols))]

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

    CONTINUE_MAX = 8192  # most new tokens a cached state is continued by (beyond: the whole state runs again)

    def _extend_states(self, units, keys, hits):
        """A request state that begins with a whole cached one (a growing agent trace or chat history): its new tokens
        continue the cached checkpoint (Qwen35Graph.forward_continue: the attention layers over the cached keys and
        values, the Gated DeltaNet layers from the cached states and tails), and the extended checkpoint replaces the
        old entry and serves as a hit. A state that differs inside the cached tokens runs again: a recurrent state
        cannot be cut there."""
        if not keys or self.state_cache is None:
            return
        na = len(self.kv_shapes)
        for k, key in enumerate(keys):
            if key is None or k in hits:
                continue
            pk = units[k][0]
            root = list(pk[0][: pk[2].index(1)])
            found = self.state_cache.longest(root)
            if found is None:
                continue
            old, e, m, n = found
            if old == key:  # cached by an earlier unit of this call
                hits[k] = e
                continue
            if m != n or n >= len(root) or len(root) - n > self.CONTINUE_MAX:
                continue
            T, dt, dev = len(root) - n, self._dtype, self.dev
            kv = [tuple(torch.empty((1, H, T, D), dtype=dt, device=dev) for _ in "kv") for H, D in self.kv_shapes]
            st = [(torch.empty((1, *self.net.state_shape), dtype=torch.float32, device=dev),
                   torch.empty((1, *self.net.tail_shape), dtype=dt, device=dev)) for _ in range(self.net.n_gdn)]
            with self._launch:
                self.net.forward_continue(self._to_dev(torch.tensor([root[n:]])), n,
                                          [(ke[None], ve[None]) for ke, ve in e[:na]],
                                          [(se[None], ce[None]) for se, ce in e[na:]], kv, st)
            new = ([(torch.cat([ke, kn[0]], 1), torch.cat([ve, vn[0]], 1)) for (ke, ve), (kn, vn) in zip(e[:na], kv)]
                   + [(s[0], c[0]) for s, c in st])
            self.state_cache.put(key, new, tokens=root, replaces=old)
            hits[k] = new

    @staticmethod
    def _entry(kv, st, r, p):
        """The state-cache entry of root row r (p tokens): per attention layer (K, V) [kv heads, p, d], then per Gated
        DeltaNet layer (recurrent state, convolution tail) after its last token."""
        return ([(k[r, :, :p].clone(), v[r, :, :p].clone()) for k, v in kv]
                + [(s[r].clone(), c[r].clone()) for s, c in st])

    @staticmethod
    def _load(kv, st, hits, P):
        """Cached entries into root rows 0.. of the pass-1 outputs, where pass 1 would have written them."""
        n = len(kv)
        for r, (e, p) in enumerate(zip(hits, P)):
            for (kb, vb), (ke, ve) in zip(kv, e[:n]):
                kb[r, :, :p].copy_(ke, non_blocking=True)
                vb[r, :, :p].copy_(ve, non_blocking=True)
            for (sb, cb), (se, ce) in zip(st, e[n:]):
                sb[r].copy_(se, non_blocking=True)
                cb[r].copy_(ce, non_blocking=True)

    @torch.no_grad()
    def _tree_hidden(self, packs, idx, keys=None, hits=None):
        """Branched units -> hidden states at their readout positions. Pass 1: the roots in a plain graph that also
        writes their keys, values and recurrent states (beyond the largest graph: one uncaptured forward into fresh
        buffers). Pass 2: one row per leaf of each unit's trie, over its root, in pass-2 graphs of up to LEAF_ROWS[-1]
        rows (uncaptured beyond them). keys / hits (--state-cache-gb): per unit, the state-cache key of its root and its
        cached entry; when every root is cached and no prompt ends inside one, pass 1 is skipped (_load); roots computed
        here are stored (_entry)."""
        P = [packs[k][2].index(1) for k in idx]  # block 0 (the root) comes first and is contiguous
        b = len(idx)
        roots = [packs[k][0][:p] for k, p in zip(idx, P)]
        keys, hits = keys or [None] * b, hits or [None] * b
        leaves = [self._leaves(packs[k], p) for k, p in zip(idx, P)]
        # a prompt that is a root itself is read out in pass 1, so its unit needs pass 1
        cached = all(e is not None for e in hits) and all(q is not None for _, reads in leaves for q, _ in reads)
        row = list(range(b))  # each unit's root row; cached: units over the same root (a template prefix) share one
        if cached:
            first = {}
            row = [first.setdefault(key, len(first)) for key in keys]
        captured = self.pool is not None and max(P) <= self.PLAIN_LENS[-1] and max(row) < self.TREE_ROWS[-1]
        h1 = None
        if cached:
            self._count("pass1 cached", captured)
            if captured:
                L2 = self._root_len(self._bucket(max(P), self.PLAIN_LENS))
                kv, st = self._root_views(L2), self._st
            else:
                L2 = max(P)
                kv, st = self._fresh(max(row) + 1, L2)
            one = [row.index(r) for r in range(max(row) + 1)]  # a unit of each root row
            self._load(kv, st, [hits[u] for u in one], [P[u] for u in one])
        elif captured:
            self._count("pass1", captured)
            L, B = self._bucket(max(P), self.PLAIN_LENS), self._bucket(b, self.BATCHES)
            L2 = self._root_len(L)
            g, s_ids, s_lens, h1 = self._graph(B, L)
            ids, lens = self._rows_in(roots, L, B)
            s_ids.copy_(self._to_dev(ids), non_blocking=True)
            s_lens.copy_(self._to_dev(lens), non_blocking=True)
            g.replay()
            kv, st = self._root_views(L2), self._st
        else:
            self._count("pass1", captured)
            L2 = max(P)
            kv, st = self._fresh(b, L2)
            ids, lens = self._rows_in(roots, L2)
            h1 = self.fwd(self._to_dev(ids), self._to_dev(lens), kv, st)
        if self.state_cache is not None and not cached:  # the roots computed here, as full-state checkpoints
            for r, key in enumerate(keys):
                if key is not None and hits[r] is None:
                    self.state_cache.put(key, self._entry(kv, st, r, P[r]), tokens=roots[r])
        rows, sel = [], []  # rows: (unit, token indices); sel: per readout (pass, row, column), in unit order
        for r, (leaf, reads) in enumerate(leaves):
            sel += [(1, r, c) if q is None else (2, len(rows) + q, c) for q, c in reads]
            rows += [(r, ix) for ix in leaf]
        width = h1.shape[-1] if h1 is not None else self._hidden_size
        out = torch.empty((len(sel), width), dtype=self._dtype if h1 is None else h1.dtype, device=self.dev)
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
                unit2[j], plen2[j] = row[r], P[r]
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

    def _long_net(self, between):
        """The forward of the long lane: the same layers and weights, uncompiled (the long lane's thread runs it while
        the main lane's graphs replay), between(i) called before layer i."""
        net = copy.copy(self.net)
        net.f, net.between = self.net.f_eager, between
        return net

    @torch.no_grad()
    def _long_hidden(self, pack, key=None, hit=None, between=None):
        """One unit of the long lane (GraphBackend.run_long) -> hidden states at its readout positions, uncompiled and
        one layer at a time over fresh buffers: a single prompt as one row; a branched unit's root as one row that
        writes its keys, values and recurrent states (or its state-cache checkpoint), stored when keyed, then one row
        per leaf from them. The main lane's static buffers and graphs are never touched."""
        net, dev = self._long_net(between), self._to_dev
        t, pp, sg, last, par = pack
        if len(par) == 1:
            return net.forward(dev(torch.tensor([t])))[0, last]
        P = sg.index(1)  # block 0 (the root) comes first and is contiguous
        rows, reads = self._leaves(pack, P)
        h1, na = None, len(self.kv_shapes)
        if hit is not None and all(q is not None for q, _ in reads):  # (a prompt that is the root needs pass 1)
            kv = [(k[None], v[None]) for k, v in hit[:na]]
            st = [(s[None], c[None]) for s, c in hit[na:]]
        else:
            kv, st = self._fresh(1, P)
            h1 = net.forward(dev(torch.tensor([t[:P]])), dev(torch.tensor([P])), kv, st)[0]
            if key is not None and self.state_cache is not None:
                self.state_cache.put(key, self._entry(kv, st, 0, P), tokens=t[:P])
        S = max(map(len, rows))
        ids = np.full((len(rows), S), self.tok.pad_token_id, dtype=np.int64)
        pos = np.zeros((len(rows), S), dtype=np.int64)
        for j, ix in enumerate(rows):
            ids[j, : len(ix)] = [t[i] for i in ix]
            pos[j, : len(ix)] = [pp[i] for i in ix]
            pos[j, len(ix):] = pp[ix[-1]] + 1 + np.arange(S - len(ix))
        unit, plen = torch.zeros(len(rows), dtype=torch.long), torch.full((len(rows),), P, dtype=torch.long)
        h2 = net.forward_suffix(dev(torch.from_numpy(ids)), dev(torch.from_numpy(pos)), dev(unit), dev(plen), kv, st)
        return torch.stack([h1[c] if q is None else h2[q, c] for q, c in reads])


class GDNMPSBackend(GDNGraphBackend, MPSBackend):
    """--mode mps for Qwen3.5 models, refused in 2.5 (engine.gdn_mps_refused: too slow without a Metal kernel);
    with BASAL_EXPERIMENTAL_GDN_MPS=1, MPSBackend(model_dir) of such a model builds this one: GDNGraphBackend's two
    passes (the recurrent state carried from each root into one row per leaf) on the Apple GPU through PyTorch MPS, or
    on the CPU, without CUDA graphs or compilation; the Gated DeltaNet layers run transformers' torch kernels, as
    --mode eager does there."""

    streams, inflight = False, 1  # (MPSBackend's: one call at a time)

    def __init__(self, model_dir, dtype="bfloat16", shared=True, device=None, adapter=None):
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        GDNGraphBackend.__init__(self, model_dir, dtype, shared=shared, warm=False, device=device or (
            "mps" if mps else "cpu"), adapter=adapter)

    run = GraphBackend.run
    run_shared = GraphBackend.run_shared
    _oversize = GraphBackend._oversize
