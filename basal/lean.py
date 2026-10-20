"""A lean Llama forward for the CUDA graph engine (--lean on): the loaded model's weights run through ONE layer
function, compiled once (dynamic shapes) and called for every layer with that layer's weights as arguments, instead of
transformers' module tree compiled as a whole. The arithmetic is transformers' Llama layer (RMSNorm in fp32, rotary
embeddings from fp32 tables, SDPA attention, SiLU MLP); the residual add is fused into the next normalisation's
kernel, the projections are merged when the model was (merge_projections), and FP8 checkpoints keep FP8Linear's GEMM.

  forward(ids, kv=None)                    plain causal rows (pass 1); kv: per-layer (K, V) buffers it fills
  forward_suffix(ids, mask, pos, prefix)   pass 2: rows over the root keys / values, with an additive mask, or with
                                           a TreeMask (FlexAttention over the KV heads as they are, no dense mask)
"""
from collections import namedtuple

import torch
import torch.nn.functional as F

from . import fp8block
from .engine import FP8_FAST_ACCUM, FP8Linear, _ATTN

E4M3 = torch.float8_e4m3fn


# The trie mask of pass 2 for FlexAttention: pr [b] root tokens of each row; s [b, S] block of each branch token (S:
# padding); anc [b, S + 1, S + 1] anc[r, x, y] = block y is block x or one of its ancestors (row / column S: none).
TreeMask = namedtuple("TreeMask", "pr s anc")


def tree_mask_mod(t, L):
    """A token of a branch sees the root tokens of its row, earlier tokens of its own block and every token of its
    ancestor blocks, and itself (padding included): the additive mask of GraphBackend._mask_from_seg plus the root."""
    def mod(b, h, q, kv):
        j = (kv - L).clamp(min=0)
        tree = (kv >= L) & ((t.anc[b, t.s[b, q], t.s[b, j]] & (j <= q)) | (j == q))
        return (kv < t.pr[b]) | tree
    return mod


def block_mask(t, L, S):
    """A FlexAttention block mask of every (query block, key block) pair, each evaluated with tree_mask_mod: the same
    blocks for any trie of this shape, so a CUDA graph keeps it and only the TreeMask tensors change."""
    from torch.nn.attention.flex_attention import BlockMask
    B = t.pr.shape[0]
    blk = 128
    nq, nk = -(-S // blk), -(-(L + S) // blk)
    num = torch.full((B, 1, nq), nk, dtype=torch.int32, device=t.pr.device)
    idx = torch.arange(nk, dtype=torch.int32, device=t.pr.device).expand(B, 1, nq, nk).contiguous()
    return BlockMask.from_kv_blocks(num, idx, BLOCK_SIZE=blk, mask_mod=tree_mask_mod(t, L), seq_lengths=(S, L + S))


def supports(model):
    """Only plain Llama decoders (every layer LlamaDecoderLayer with LlamaAttention / LlamaMLP)."""
    return all(type(layer).__name__ == "LlamaDecoderLayer" and type(layer.self_attn).__name__ == "LlamaAttention"
               and type(layer.mlp).__name__ == "LlamaMLP" for layer in model.model.layers) \
        and getattr(model.config, "hidden_act", "silu") == "silu"


def _params(m):
    """A linear module -> its weights as arguments: (weight, bias), (weight, weight_scale, input_scale, bias) of an
    FP8Linear or (weight, block scales, bias) of a fp8block.BlockFP8Linear."""
    if isinstance(m, FP8Linear):
        return (m.weight, m.weight_scale, m.input_scale, m.bias)
    if isinstance(m, fp8block.BlockFP8Linear):
        return (m.weight, m.weight_scale, m.bias)
    return (m.weight, m.bias)


def linear(x, p):
    """F.linear, FP8Linear's arithmetic (static input scale, rows padded to 16, one scaled GEMM, bias fused) or the
    block-scaled FP8 GEMM (fp8block.linear)."""
    if len(p) == 2:
        return F.linear(x, p[0], p[1])
    if len(p) == 3:
        return fp8block.linear(x, *p)
    w, ws, xs, b = p
    shape, dt = x.shape, x.dtype
    x = x.reshape(-1, shape[-1])
    q = (x.float() / xs).clamp(-448.0, 448.0).to(E4M3)
    if x.is_cuda:
        m = q.shape[0]
        if m % 16:
            q = F.pad(q.view(torch.uint8), (0, 0, 0, -m % 16)).view(E4M3)
        y = torch._scaled_mm(q, w.t(), scale_a=xs, scale_b=ws, bias=None if b is None else b.to(dt), out_dtype=dt,
                             use_fast_accum=FP8_FAST_ACCUM[0])[:m]
    else:
        y = (q.float() * xs) @ (w.float() * ws).t()
        y = (y if b is None else y + b.float()).to(dt)
    return y.reshape(*shape[:-1], w.shape[0])


class Lean:
    def __init__(self, model, compile=True):
        mm, cfg = model.model, model.config
        self.H, self.Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps, self.scale = cfg.rms_norm_eps, self.d ** -0.5
        self.emb, self.norm = mm.embed_tokens.weight, mm.norm.weight
        dt, dev = self.emb.dtype, self.emb.device
        rope = mm.rotary_emb
        n = getattr(cfg, "max_position_embeddings", 32768)
        f = torch.outer(torch.arange(n, device=dev, dtype=torch.float32), rope.inv_freq.float().to(dev))
        e = torch.cat([f, f], -1)
        s = getattr(rope, "attention_scaling", 1.0)
        self.cos, self.sin = (e.cos() * s).to(dt), (e.sin() * s).to(dt)
        self.layers = []
        for layer in mm.layers:
            a, m = layer.self_attn, layer.mlp
            qkv = [a.qkv_proj] if hasattr(a, "qkv_proj") else [a.q_proj, a.k_proj, a.v_proj]
            gu = [m.gate_up_proj] if hasattr(m, "gate_up_proj") else [m.gate_proj, m.up_proj]
            self.layers.append((layer.input_layernorm.weight, layer.post_attention_layernorm.weight,
                                tuple(_params(x) for x in qkv), _params(a.o_proj), tuple(_params(x) for x in gu),
                                _params(m.down_proj)))
        self.layer = torch.compile(self._layer, dynamic=True) if compile else self._layer

    def rms(self, x, w):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

    @staticmethod
    def rot(x, cos, sin):
        h = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin

    def attend(self, q, k, v, mask=None):
        """q [b, L, H, d], k / v [b, Hkv, Lk, d] -> [b, L, H, d]. Without a mask: causal, with the kernel of --attn;
        with a FlexAttention block mask: flex_attention over the KV heads as they are; with an additive mask
        [b, 1, L, Lk]: SDPA over the KV heads repeated, as transformers does."""
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
        if isinstance(mask, BlockMask):  # the prefill kernel: Inductor's decoding kernel has no dynamic-shape choice
            return flex_attention(q.transpose(1, 2), k, v, block_mask=mask, scale=self.scale, enable_gqa=True,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True}).transpose(1, 2)
        direct = _ATTN["direct"]
        if mask is None:
            if direct is not None and q.dtype in (torch.bfloat16, torch.float16):
                return direct(q.transpose(1, 2), k, v, self.scale)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, is_causal=True, scale=self.scale,
                                               enable_gqa=True)
            return o.transpose(1, 2)
        r = self.H // self.Hkv
        b, _, Lk, d = k.shape
        # contiguous: with one key / value head the reshape is a stride-0 view, from which SDPA's memory-efficient
        # kernel (the dispatcher's pick for a masked call) returns a wrong last query row on torch < 2.14.1
        # (pytorch/pytorch#191937); with more heads the reshape copies anyway
        k = k[:, :, None].expand(b, self.Hkv, r, Lk, d).reshape(b, self.H, Lk, d).contiguous()
        v = v[:, :, None].expand(b, self.Hkv, r, Lk, d).reshape(b, self.H, Lk, d).contiguous()
        return F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask, scale=self.scale).transpose(1, 2)

    def _layer(self, h, res, P, cos, sin, kb=None, vb=None, pk=None, pv=None, mask=None):
        """One decoder layer: h + res (the previous layer's MLP output) is the residual stream. kb / vb: buffers that
        receive this layer's keys and values (pass 1); pk / pv: prefix keys and values the rows continue (pass 2)."""
        ln1, ln2, qkv, o, gu, down = P
        b, L, _ = h.shape
        h = h + res
        x = self.rms(h, ln1)
        if len(qkv) == 1:
            q, k, v = linear(x, qkv[0]).split([self.H * self.d, self.Hkv * self.d, self.Hkv * self.d], -1)
        else:
            q, k, v = (linear(x, p) for p in qkv)
        q = self.rot(q.view(b, L, self.H, self.d), cos, sin)
        k = self.rot(k.view(b, L, self.Hkv, self.d), cos, sin).transpose(1, 2)
        v = v.view(b, L, self.Hkv, self.d).transpose(1, 2)
        if kb is not None:
            kb.copy_(k)
            vb.copy_(v)
        if pk is not None:
            k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
        a = self.attend(q, k, v, mask)
        h = h + linear(a.reshape(b, L, self.H * self.d), o)
        x = self.rms(h, ln2)
        if len(gu) == 1:
            g, u = linear(x, gu[0]).chunk(2, -1)
        else:
            g, u = linear(x, gu[0]), linear(x, gu[1])
        return h, linear(F.silu(g) * u, down)

    def _run(self, ids, cos, sin, kv=None, prefix=None, mask=None):
        h = F.embedding(ids, self.emb)
        res = torch.zeros_like(h)
        for i, P in enumerate(self.layers):
            kb, vb = kv[i] if kv is not None else (None, None)
            pk, pv = prefix[i] if prefix is not None else (None, None)
            h, res = self.layer(h, res, P, cos, sin, kb, vb, pk, pv, mask)
        return self.rms(h + res, self.norm)

    @torch.no_grad()
    def forward(self, ids, kv=None):
        """Plain causal rows [b, L] from position 0 -> normalised hidden states [b, L, D]; kv: per-layer (K, V) buffers
        [b, kv heads, L, head dim] that receive the keys and values (the root pass)."""
        L = ids.shape[1]
        return self._run(ids, self.cos[:L, None], self.sin[:L, None], kv=kv)

    @torch.no_grad()
    def forward_suffix(self, ids, mask, pos, prefix):
        """Pass 2: rows [b, S] at positions pos that continue per-layer prefix (K, V) [b, kv heads, P, head dim], with
        an additive mask [b, 1, S, P + S] or a TreeMask -> normalised hidden states [b, S, D]."""
        if isinstance(mask, TreeMask):
            mask = block_mask(mask, prefix[0][0].shape[2], ids.shape[1])
        return self._run(ids, self.cos[pos][:, :, None], self.sin[pos][:, :, None], prefix=prefix, mask=mask)
