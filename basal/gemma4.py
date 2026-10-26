"""The Gemma 4 (basal-omni) decoder forward of the eager backend (basal 2.3). It runs the layers of the transformers
model (the same modules and weights) in its own loop, for three things the transformers forward cannot do:

- **SOAM (state once, ask many):** the prompts of a group (the option orders of a question; with SOAM every question
  and option order of a request) share a token prefix: the chat template, the state and, with media, the image /
  audio soft tokens. The prefix runs once into keys and values; the rest of every prompt (its suffix) then runs as one
  right-padded batch over them, positions continuing from the prefix. Each suffix sees the prefix and its own earlier
  tokens only, so its readout equals the forward of its whole prompt (up to float rounding). Prefixes of several groups
  are batched together, a long one runs alone in CHUNK-token parts; a single prompt longer than CHUNK runs its parts
  the same way (the transformers path's chunked prefill). Sliding-window layers keep the last window - 1 keys of a
  prefix only (the most a later token can see), global layers all of them. A state over 16k tokens (ATTN_PAIRS //
  CHUNK) writes its global layers' keys and values into buffers of the whole state, and their attention runs in calls
  of at most SCORES scores, each KV head's query heads as one block of queries (_split); up to 16k tokens every call
  is as before.
- **Answer rows only (G1):** the layers from the last one with its own keys and values onwards matter only at the
  readout positions: Gemma 4 E2B / E4B reuse the keys and values of earlier layers in their last layers (E4B: 18 of
  42), so those layers produce nothing a later token reads. Such layers, and the attention and MLP of the last layer
  with its own keys and values (on 12B / 31B the last layer), run at the answer rows only: one row per prompt, its
  query attending the stored keys. A prefix stops after the keys and values of that layer. Same maths.
- **Media once per request:** the image and audio towers run once for all prompts of a request (the prompts carry the
  same media), and the soft tokens are part of the shared prefix. On the larger Gemma 4 models
  (use_bidirectional_attention "vision") the soft tokens of one image attend to each other as transformers builds the
  masks: Gemma 4 31B in its sliding-window layers within the window, the unified models (basal-omni max, 1.1-max) in
  every layer, past the window as well; a prefix part never ends inside an image.

Attention runs per layer type (E1, --eager-attn): torch SDPA with an explicit boolean mask (the math kernel by default;
EagerBackend turns the others off for Gemma 4, so the memory-efficient kernel runs only when chosen), or a
FlashAttention varlen kernel over the real queries and keys of every row. Measured on real activations of mini and
max (H100, 2026-10-20: every attention call of a 1,857-token and three 12.6k-token prompts against float64): in the
sliding-window layers (head size 256) FlashAttention 3 is within bf16 rounding of the math kernel at 11x its speed,
the memory-efficient kernel at 4.4x; in the global layers (512) only the math and memory-efficient kernels run on
Hopper (FA3 / FA4 take at most 256, cuDNN has no kernel, FlexAttention runs out of shared memory), the
memory-efficient one 3x faster. The 7 Oct wrong attention was the memory-efficient kernel on keys and values with a
zero-stride head dimension (pytorch/pytorch#191937, torch <= 2.13: query lengths = 1 mod 32 at head size 256 / 512):
transformers repeats max's one global KV head to 16 query heads as an expanded view, and on the 1,857-token prompt the
kernel returned errors of 10-28 on outputs of 14-22 in every global layer (fp32 and bf16). Here the keys and values are
copied per head before that kernel.
"""
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# attention kernels per layer type (E1): "sdpa" = torch SDPA with the boolean mask on the kernels the process allows
# (the math kernel: EagerBackend turns the others off for Gemma 4), "efficient" / "cudnn" = that SDPA kernel with the
# mask, "fa3" / "fa4" = a FlashAttention varlen kernel over the rows' real queries and keys (causal, aligned at each
# row's end, the sliding window as its left window)
SDPA_KERNELS = {"sdpa": None, "efficient": "EFFICIENT_ATTENTION", "cudnn": "CUDNN_ATTENTION"}


def _fa3(q, k, v, cu_q, cu_k, max_q, max_k, scale, window):
    import flash_attn_interface as fa
    o = fa.flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=scale, causal=True,
                                  window_size=(window - 1, 0) if window else (-1, -1))
    return o[0] if isinstance(o, (tuple, list)) else o


def _fa4(q, k, v, cu_q, cu_k, max_q, max_k, scale, window):
    from flash_attn.cute import interface as fa
    o = fa.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q,
                                  max_seqlen_k=max_k, softmax_scale=scale, causal=True,
                                  window_size=(window - 1, 0) if window else (None, None))
    return o[0] if isinstance(o, (tuple, list)) else o


VARLEN_KERNELS = {"fa3": _fa3, "fa4": _fa4}
KERNELS = tuple(SDPA_KERNELS) + tuple(VARLEN_KERNELS)
TYPES = {"sliding": "sliding_attention", "full": "full_attention"}


def attention_plan(spec="auto"):
    """--eager-attn -> {layer type: kernel} of Gemma4Forward. math: the SDPA math kernel in every layer (basal 2.2);
    auto: on Hopper (sm90, where E1 measured them) FlashAttention 3 in the sliding-window layers (the memory-efficient
    kernel without the flash_attn_3 package) and the memory-efficient kernel in the global layers (head size 512: no
    flash kernel runs it on Hopper), the math kernel elsewhere; "sliding=K,full=K" with K one of KERNELS."""
    if spec == "math":
        return {}
    if spec == "auto":
        from .engine import attention_available
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return {}
        sliding = "fa3" if attention_available("fa3") is None else "efficient"
        return {"sliding_attention": sliding, "full_attention": "efficient"}
    plan = {}
    for part in spec.split(","):
        t, _, k = part.partition("=")
        if t not in TYPES or k not in KERNELS:
            raise ValueError(f"--eager-attn {spec!r}: math, auto, or sliding=K,full=K with K one of {', '.join(KERNELS)}")
        plan[TYPES[t]] = k
    return plan


def text_model(model):
    """The decoder of a Gemma 4 model: Gemma4(Unified)ForConditionalGeneration -> model.language_model,
    Gemma4ForCausalLM -> model."""
    return getattr(model.model, "language_model", model.model)


def is_gemma4(model):
    """A loaded Gemma 4 model (basal-omni): model types gemma4, gemma4_text, gemma4_unified, ..."""
    return str(getattr(model.config, "model_type", "")).startswith("gemma4")


class HostRows(torch.nn.Module):
    """Gemma 4's per-layer embedding table in host memory (G3, --ple-offload on): E2B / E4B look up 256 extra numbers
    per layer for every token in a table of 262,144 rows (E4B: x 42 layers = 10,752 columns, 5.6 GB in bf16, a third
    of mini's weights). Only the rows of a forward's tokens are copied to the GPU; the values are those of
    Gemma4TextScaledWordEmbedding (the rows times the scale in the table's dtype). Not a parameter, so model.to(device)
    leaves the table where it is; the transformers forward reads it the same way."""

    def __init__(self, emb):
        super().__init__()
        w = emb.weight.data.to("cpu")
        self.table = w.pin_memory() if torch.cuda.is_available() else w
        self.scale = emb.scalar_embed_scale
        self.embedding_dim = w.shape[1]

    def forward(self, ids):
        flat = ids.reshape(-1).cpu()
        rows = torch.empty((flat.numel(), self.embedding_dim), dtype=self.table.dtype,
                           pin_memory=self.table.is_pinned())
        torch.index_select(self.table, 0, flat, out=rows)
        out = rows.to(ids.device, non_blocking=True).view(*ids.shape, -1)
        return out * torch.tensor(self.scale, dtype=out.dtype, device=out.device)


def offload_ple(model):
    """Move a Gemma 4 model's per-layer embedding table to host memory (before model.to(device)); -> bytes moved
    (0: the model has no per-layer inputs)."""
    lm = text_model(model)
    emb = getattr(lm, "embed_tokens_per_layer", None)
    if emb is None or isinstance(emb, HostRows):
        return 0
    lm.embed_tokens_per_layer = HostRows(emb)
    return lm.embed_tokens_per_layer.table.numel() * lm.embed_tokens_per_layer.table.element_size()


class Gemma4Forward:
    # memory bounds as in EagerBackend: prefix parts of at most CHUNK tokens; at most MAX_BATCH_TOKENS padded tokens and
    # ATTN_PAIRS query-key pairs (rows x query tokens x keys: the attention scores of the math kernel) per forward,
    # the pairs of one CHUNK-token part over a 16k-token context
    CHUNK = 2048
    MAX_BATCH_TOKENS = 16384
    ATTN_PAIRS = 2048 * 16384
    # keys and values repeated per query head (the math kernel's GQA, the memory-efficient kernel's input) over the
    # rows of a suffix forward: each row repeats the prefix keys, so 31B (32 heads x 512 in its global layers, 64 KB a
    # row and key) with five questions over a 10k-token state would hold 3.2 GB per layer; at most KV_BYTES
    KV_BYTES = 2 << 30
    # an attention call past those budgets (a state over 16k tokens) runs as calls of at most SCORES scores (query heads
    # x queries x keys; fp32 in the math kernel: 256 MB), as the graph engine's math kernel (Gemma4Graph.SCORES)
    SCORES = 2 ** 26

    def __init__(self, model, answer_rows=True, attn=None, two_way=None):
        self.model = model
        # two-way reading (basal.json "two_way", prompt.model_two_way): {"max_prompt", "max_record"} or None. The
        # global layers read a prompt's tokens both ways, the sliding-window layers stay causal; a prompt over
        # max_prompt tokens, or every prompt of a request over max_record tokens, is read one way (two_way_rows)
        self.two_way = dict(two_way) if two_way else None
        lm = self.lm = text_model(model)
        cfg = self.cfg = lm.config
        self.layers = list(lm.layers[: cfg.num_hidden_layers])
        self.types = list(cfg.layer_types)
        self.window = cfg.sliding_window
        self.own = [not layer.self_attn.is_kv_shared_layer for layer in self.layers]
        # the last layer with its own keys and values; a shared layer reads the last such layer of its type (the
        # layer transformers marks store_full_length_kv)
        self.last_kv = max(i for i, o in enumerate(self.own) if o)
        self.src = {i: max(j for j in range(len(self.layers)) if self.own[j] and self.types[j] == t)
                    for i, t in enumerate(self.types) if not self.own[i]}
        self.store = set(self.src.values())
        self.ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        self.bidir = getattr(cfg, "use_bidirectional_attention", None) == "vision"
        # where an image's soft tokens see each other: transformers' Gemma 4 (31B) masks the sliding-window layers only,
        # the window still applying (create_masks_for_vision_model); Gemma 4 unified (max, 1.1-max) passes the image
        # blocks to every layer's mask, OR-ed after the window (create_masks_for_generate). Taking the 31B rule for max
        # moved image answers by up to 0.48 against the transformers forward (33 of 150 images over 0.05; H100,
        # 2026-10-20); text and audio were unaffected
        self.unified_blocks = str(getattr(cfg, "model_type", "")).startswith("gemma4_unified")
        self.block_types = set(self.types) if self.unified_blocks else {"sliding_attention"}
        self.rope = sys.modules[type(self.layers[0].self_attn).__module__].apply_rotary_pos_emb
        self.answer_rows = answer_rows
        self.attn = dict(attn or {})  # layer type -> kernel (SDPA_KERNELS, VARLEN_KERNELS); "sdpa" when absent
        top = model.config
        self.media_ids = {k: getattr(top, f"{k}_token_id", None) for k in ("image", "video", "audio")}
        self.pad = cfg.pad_token_id
        w = lm.embed_tokens.weight
        self.kv_row_bytes = 2 * w.element_size() * max(cfg.num_attention_heads * layer.self_attn.head_dim
                                                       for layer in self.layers)

    # -- embeddings ---------------------------------------------------------------------------------------------------
    def _embed(self, ids, media=None, offsets=None):
        """Token ids [B, S] -> (embeddings [B, S, D], per-layer inputs [B, S, L, P] or None). media: {kind: soft
        tokens [n, D]} of the request; a row's media tokens take them from offsets[b][kind] on (the number of that
        kind's tokens before the row in its prompt). As transformers: media positions read the pad token's rows."""
        lm = self.lm
        kinds = [(k, self.media_ids[k]) for k in (media or {}) if self.media_ids.get(k) is not None]
        llm = ids
        if kinds:
            mm = torch.zeros_like(ids, dtype=torch.bool)
            for _, t in kinds:
                mm |= ids == t
            llm = torch.where(mm, self.pad, ids)
        h = lm.embed_tokens(llm)
        for k, t in kinds:
            m = ids == t
            counts = m.sum(1).tolist()
            if not sum(counts):
                continue
            feats = torch.cat([media[k][offsets[b][k]: offsets[b][k] + c] for b, c in enumerate(counts) if c])
            h = h.masked_scatter(m[..., None], feats.to(h.dtype))
        pli = None
        if self.ple:
            pli = lm.project_per_layer_inputs(h, lm.get_per_layer_inputs(llm, None))
        return h, pli

    def _blocks(self, ids):
        """Vision block ids [B, S] (-1: not an image / video soft token): runs of soft tokens, as transformers'
        get_block_sequence_ids_for_mask; None when the model attends causally only."""
        if not self.bidir:
            return None
        vis = torch.zeros_like(ids, dtype=torch.bool)
        for k in ("image", "video"):
            if self.media_ids.get(k) is not None:
                vis |= ids == self.media_ids[k]
        if not vis.any():
            return None
        start = vis & ~F.pad(vis, (1, 0))[:, :-1]
        return torch.where(vis, start.long().cumsum(1) - 1, -1)

    # -- one forward --------------------------------------------------------------------------------------------------
    def _mask(self, t, qpos, kc, pos, valid, qblk, two_way=False):
        """Boolean attention mask [B, 1, Sq, Lc + S] of layer type t: keys of earlier passes (positions kc [B, Lc],
        -1 = none) then this pass's tokens (positions pos [B, S]; valid = real tokens, a padding query sees itself so
        that no row is empty). two_way: whole prompts (no earlier keys) read both ways in the global layers."""
        sliding = t == "sliding_attention"
        d = qpos[:, :, None] - pos[:, None, :]
        m = d >= 0
        block = None
        if qblk is not None and t in self.block_types:
            block = (qblk[:, :, None] == qblk[:, None, :]) & (qblk[:, :, None] >= 0)
        if block is not None and not self.unified_blocks:
            m = m | block  # 31B: within the window
        if sliding:
            m = m & (d < self.window)
        if block is not None and self.unified_blocks:
            m = m | block  # unified: the whole image, past the window too
        m = m & (valid[:, None, :] | (d == 0))
        if two_way and not sliding:
            m = valid[:, None, :] | (d == 0)
        if kc is not None:  # (comparisons, not distances: over an 82k-token state those were 1.3 GB of int64)
            kq = kc[:, None, :]
            mc = (kq >= 0) & (kq <= qpos[:, :, None])
            if sliding:
                mc = mc & (qpos[:, :, None] - kq < self.window)
            m = torch.cat([mc, m], -1)
        return m[:, None]

    def _q(self, a, x, cs):
        q = a.q_norm(a.q_proj(x).view(*x.shape[:2], -1, a.head_dim))
        return self.rope(q, *cs, unsqueeze_dim=2).transpose(1, 2)

    def _kv(self, a, x, cs):
        shape = (*x.shape[:2], -1, a.head_dim)
        k = a.k_proj(x).view(shape)
        v = a.v_proj(x).view(shape) if a.v_proj is not None else k
        return self.rope(a.k_norm(k), *cs, unsqueeze_dim=2).transpose(1, 2), a.v_norm(v).transpose(1, 2)

    def _kernel(self, t, dtype, blk):
        """The kernel of layer type t in a forward: the masked SDPA math kernel for fp32 and, in the layers where an
        image's soft tokens attend to each other (block_types), for rows with images (a flash kernel's mask is causal
        only)."""
        k = self.attn.get(t, "sdpa")
        if k in VARLEN_KERNELS and (dtype not in (torch.bfloat16, torch.float16)
                                    or blk is not None and t in self.block_types):
            return "sdpa"
        return k

    def _split(self, a, q, k, v, mask):
        """An SDPA call past the budgets of a forward over a 16k-token context (ATTN_PAIRS query-key pairs, KV_BYTES of
        keys and values repeated per query head; only states over 16k tokens reach them) as calls of SCORES scores:
        per KV head, its query heads folded into one block of queries over its keys and values (nothing repeated per
        query head), in parts of queries. Without it the 2,048 queries of a part of ultra over a 76k-token state were
        one math-kernel call of 32 heads x 2,048 x 76k scores (fp32: 20 GB) over keys and values repeated 8x."""
        B, H, Sq, d = q.shape
        G, Lk = k.shape[1], k.shape[2]
        r = H // G
        n = max(1, self.SCORES // (B * r * Lk))  # queries per call
        out = q.new_empty(B, H, Sq, d)
        for j in range(G):
            for s0 in range(0, Sq, n):
                qs = q[:, j * r: (j + 1) * r, s0: s0 + n]
                m = qs.shape[2]
                o = F.scaled_dot_product_attention(qs.reshape(B, 1, r * m, d), k[:, j: j + 1], v[:, j: j + 1],
                                                   attn_mask=mask[:, :, s0: s0 + m].repeat(1, 1, r, 1), scale=a.scaling)
                out[:, j * r: (j + 1) * r, s0: s0 + m] = o.view(B, r, m, d)
        return out

    def _attend(self, kind, a, q, k, v, mask, kval, qval, window, split=False):
        """Attention of one layer: q [B, H, Sq, d], k / v [B, kv heads, Lk, d] -> [B, H, Sq, d]. SDPA kernels take the
        boolean mask; varlen kernels the real queries (qval [B, Sq]) and keys (kval [B, Lk]) of each row packed back to
        back: a row's real keys are its earlier tokens' and its own, in position order, ending at its last real query,
        so causal attention aligned at the row's end (with the window) is the mask's attention. split: an SDPA call in
        _split's parts even within the budgets (a state over 16k tokens)."""
        if kind in SDPA_KERNELS:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            name = SDPA_KERNELS[kind]
            B, H, Sq, d = q.shape
            G, Lk = k.shape[1], k.shape[2]
            if split or B * Sq * Lk > self.ATTN_PAIRS or B * Lk * H * d * k.element_size() * 2 > self.KV_BYTES:
                with sdpa_kernel([getattr(SDPBackend, name)]) if name else nullcontext():
                    return self._split(a, q, k, v, mask)
            gqa = name is None
            if not gqa:  # the memory-efficient and cuDNN kernels take no enable_gqa: keys and values per query head,
                # copied (an expanded, zero-stride view gives the memory-efficient kernel wrong attention at query
                # lengths = 1 mod 32: pytorch/pytorch#191937, the 7 Oct case)
                k, v = (x.repeat_interleave(q.shape[1] // k.shape[1], 1) for x in (k, v))
            with sdpa_kernel([getattr(SDPBackend, name)]) if name else nullcontext():
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a.scaling, enable_gqa=gqa)
        B, H, Sq, d = q.shape
        nq, nk = qval.sum(1), kval.sum(1)
        cu_q = F.pad(nq.cumsum(0), (1, 0)).to(torch.int32)
        cu_k = F.pad(nk.cumsum(0), (1, 0)).to(torch.int32)
        o = VARLEN_KERNELS[kind](q.transpose(1, 2)[qval], k.transpose(1, 2)[kval], v.transpose(1, 2)[kval], cu_q, cu_k,
                                 int(nq.max()), int(nk.max()), a.scaling, window)
        out = q.new_zeros(B, Sq, H, d)
        out[qval] = o.to(q.dtype)
        return out.transpose(1, 2)

    def _rest(self, layer, h, o, pli):
        """The decoder layer after attention (transformers' Gemma 4 decoder layer: MLP, MoE block, per-layer input)."""
        o = layer.self_attn.o_proj(o.transpose(1, 2).reshape(*h.shape[:2], -1))
        h = h + layer.post_attention_layernorm(o)
        y = layer.mlp(layer.pre_feedforward_layernorm(h))
        if getattr(layer, "enable_moe_block", False):
            flat = h.reshape(-1, h.shape[-1])
            _, w, idx = layer.router(flat)
            y2 = layer.experts(layer.pre_feedforward_layernorm_2(flat), idx, w).reshape(h.shape)
            y = layer.post_feedforward_layernorm_1(y) + layer.post_feedforward_layernorm_2(y2)
        h = h + layer.post_feedforward_layernorm(y)
        if pli is not None:
            g = layer.act_fn(layer.per_layer_input_gate(h)) * pli
            h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
        return h * layer.layer_scalar

    def _pass(self, h, pli, pos, valid, ctx=None, grp=None, read=None, blk=None, split=False, two_way=False):
        """One forward over token rows [B, S]: embeddings h [B, S, D], per-layer inputs pli [B, S, L, P] or None,
        positions pos [B, S], valid [B, S] (right padding), blk vision block ids or None. ctx: (kv, kpos) of earlier
        tokens, kv {own layer: (K, V) [G, kv heads, Lc, d]}, kpos {layer type: positions [G, Lc]}; row b continues
        ctx row grp[b]. read [B]: the index of each row's readout token, or None for a prefix part (it stops after the
        keys and values of the last layer that has its own). split: the global layers' attention in _split's parts.
        -> ({own layer: (K, V) of this pass's tokens}, final hidden state at `read` [B, D] or None). A readout pass
        keeps no keys and values ({}): nothing reads them, and ultra's of a 16k-token batch were 14 GB. two_way: rows
        of whole prompts read both ways in the global layers (_mask)."""
        assert not (two_way and ctx is not None), "a two-way prompt is read whole, without earlier keys"
        B, S = valid.shape
        kv, kpos = ctx if ctx is not None else ({}, {})
        if ctx is not None and grp is None:
            grp = torch.zeros(B, dtype=torch.long, device=pos.device)
        types = set(self.types)
        rope = {t: self.lm.rotary_emb(h, pos, t) for t in types}
        # a two-way forward's global layers take the masked SDPA kernel (a flash kernel's mask is causal only)
        kinds = {t: "sdpa" if two_way and t != "sliding_attention" else self._kernel(t, h.dtype, blk) for t in types}
        masks = {t: self._mask(t, pos, kpos[t][grp] if t in kpos else None, pos, valid, blk, two_way)
                 for t in types if kinds[t] in SDPA_KERNELS}
        kval = {t: torch.cat([kpos[t][grp] >= 0, valid], 1) if t in kpos else valid
                for t in types if kinds[t] in VARLEN_KERNELS}
        qval = valid
        new, kept, sliced = {}, {}, False
        rows = None if read is None else (torch.arange(B, device=pos.device), read)
        # G1: the layers from the last one with its own keys and values on run at the answer rows only
        cut = self.last_kv if self.answer_rows and read is not None and S > 1 else None
        for i, layer in enumerate(self.layers):
            a, t = layer.self_attn, self.types[i]
            x = layer.input_layernorm(h)
            if self.own[i]:  # every own layer comes before the cut: its keys and values at every position
                k, v = self._kv(a, x, rope[t])
                if read is None:
                    new[i] = (k, v)
                if i in kv:
                    K, V = kv[i]
                    K = K.expand(B, -1, -1, -1) if K.shape[0] == 1 else K.index_select(0, grp)
                    V = V.expand(B, -1, -1, -1) if V.shape[0] == 1 else V.index_select(0, grp)
                    k, v = torch.cat([K, k], 2), torch.cat([V, v], 2)
                if i in self.store:
                    kept[t] = (k, v)
            else:
                k, v = kept[t]
            if read is None and i == self.last_kv:
                break
            if i == cut:
                h, x = h[rows][:, None], x[rows][:, None]
                pli = pli[rows][:, None] if pli is not None else None
                rope = {t2: (c[rows][:, None], s[rows][:, None]) for t2, (c, s) in rope.items()}
                masks = {t2: m[rows[0], :, rows[1]][:, :, None] for t2, m in masks.items()}
                qval = torch.ones(B, 1, dtype=torch.bool, device=pos.device)
                sliced = True
            q = self._q(a, x, rope[t])
            o = self._attend(kinds[t], a, q, k, v, masks.get(t), kval.get(t), qval,
                             self.window if t == "sliding_attention" else None, split and t != "sliding_attention")
            h = self._rest(layer, h, o, pli[:, :, i] if pli is not None else None)
        if read is None:
            return new, None
        return new, self.lm.norm(h[:, 0] if sliced else h[rows])

    # -- scheduling ---------------------------------------------------------------------------------------------------
    def _vision(self, tok):
        return self.bidir and tok in (self.media_ids.get("image"), self.media_ids.get("video"))

    def _run_start(self, ids, n):
        """n moved back to the start of the image its cut would split (a prefix never ends inside an image)."""
        while 0 < n < len(ids) and self._vision(ids[n - 1]) and self._vision(ids[n]):
            n -= 1
        return n

    def _parts(self, ids, n):
        """CHUNK-token parts [(start, end)] of ids[:n], none ending inside an image (a longer image is one part)."""
        out, s = [], 0
        while s < n:
            e = min(s + self.CHUNK, n)
            if e < n:
                e = self._run_start(ids, e)
                while e <= s or (e < n and self._vision(ids[e - 1]) and self._vision(ids[e])):
                    e += 1
            out.append((s, e))
            s = e
        return out

    def _rows(self, seqs, starts, media=None, prompts=None):
        """Token rows (right-padded) -> (embeddings, per-layer inputs, positions, valid, block ids, lengths); a row
        starts at position starts[b] of its prompt (prompts[b], for the media offsets)."""
        dev = self.lm.embed_tokens.weight.device
        S = max(map(len, seqs))
        ids = torch.tensor([list(s) + [self.pad] * (S - len(s)) for s in seqs], device=dev)
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        ar = torch.arange(S, device=dev)
        valid = ar[None] < lens[:, None]
        pos = torch.tensor(starts, device=dev)[:, None] + ar[None]
        offsets = None
        if media:
            offsets = [{k: sum(1 for x in p[:s0] if x == self.media_ids[k]) for k in media}
                       for p, s0 in zip(prompts, starts)]
        h, pli = self._embed(ids, media, offsets)
        return h, pli, pos, valid, self._blocks(ids), lens

    def _trim(self, kv, kpos):
        """Keys of sliding-window layers: the last window - 1 of each row (a later token sees no earlier ones); a row's
        keys are its first ones (right padding)."""
        t, keep = "sliding_attention", self.window - 1
        if t not in kpos or kpos[t].shape[1] <= keep:
            return kv, kpos
        n = (kpos[t] >= 0).sum(1)
        idx = (n - keep).clamp_min(0)[:, None] + torch.arange(keep, device=n.device)
        ok = idx < n[:, None]
        idx = idx.clamp_max(kpos[t].shape[1] - 1)
        kpos = dict(kpos, **{t: torch.where(ok, kpos[t].gather(1, idx), -1)})
        out = {}
        for i, (K, V) in kv.items():
            if self.types[i] == t:
                g = idx[:, None, :, None].expand(-1, K.shape[1], -1, K.shape[3])
                K, V = K.gather(2, g), V.gather(2, g)
            out[i] = (K, V)
        return out, kpos

    def _prefix(self, seqs, media=None):
        """Keys and values of prefixes: short ones in one forward, a long one (alone) in parts through the keys of the
        parts before it: global layers write theirs into buffers of the whole prefix (no copy of the earlier ones per
        part; an 82k-token state of ultra holds 6.7 GB of them), sliding-window layers keep their last window - 1
        after each part, layer by layer (_trim's keys)."""
        if len(seqs) == 1 and len(seqs[0]) > self.CHUNK:
            seq, keep = seqs[0], self.window - 1
            # a state over 16k tokens: every global-layer call in parts, from its first part on (its buffers leave
            # no room for the scores of a part over 14k keys in one call: 3.5 GB on ultra at 76k)
            split = len(seq) > self.ATTN_PAIRS // self.CHUNK
            kv, bufs, kpos = {}, {}, None  # kpos: set with kv's first keys, at the end of the first part
            at = torch.arange(len(seq), device=self.lm.embed_tokens.weight.device)[None]
            for s0, s1 in self._parts(seq, len(seq)):
                h, pli, pos, valid, blk, _ = self._rows([seq[s0:s1]], [s0], media, seqs)
                new, _ = self._pass(h, pli, pos, valid, ctx=(kv, kpos) if kv else None, blk=blk, split=split)
                for i in list(new):
                    k, v = new.pop(i)
                    if self.types[i] == "sliding_attention":
                        if i in kv:
                            k, v = torch.cat([kv[i][0], k], 2), torch.cat([kv[i][1], v], 2)
                        kv[i] = (k[:, :, -keep:].clone(), v[:, :, -keep:].clone())
                    else:
                        if i not in bufs:
                            bufs[i] = tuple(x.new_empty(*x.shape[:2], len(seq), x.shape[3]) for x in (k, v))
                        bufs[i][0][:, :, s0:s1], bufs[i][1][:, :, s0:s1] = k, v
                        kv[i] = (bufs[i][0][:, :, :s1], bufs[i][1][:, :, :s1])
                kpos = {t: at[:, max(0, s1 - keep): s1] if t == "sliding_attention" else at[:, :s1]
                        for t in set(self.types)}
            return kv, kpos
        h, pli, pos, valid, blk, _ = self._rows(seqs, [0] * len(seqs), media, seqs)
        new, _ = self._pass(h, pli, pos, valid, blk=blk)
        return self._trim(new, {t: torch.where(valid, pos, -1) for t in set(self.types)})

    def _shared(self, jobs, groups, out, media=None):
        """A batch of jobs (group, prompt indices, prefix length): their prefixes once, then every suffix over them,
        in forwards within MAX_BATCH_TOKENS tokens, ATTN_PAIRS query-key pairs and KV_BYTES of repeated keys."""
        ctx = self._prefix([groups[g][js[0]][:P] for g, js, P in jobs], media)
        Lc = max(p.shape[1] for p in ctx[1].values())
        rows = sorted(((b, g, j, P) for b, (g, js, P) in enumerate(jobs) for j in js),
                      key=lambda r: len(groups[r[1]][r[2]]) - r[3])
        cur = []
        for r in rows + [None]:
            if cur and r is not None:
                n, S = len(cur) + 1, len(groups[r[1]][r[2]]) - r[3]
                if (n * S <= self.MAX_BATCH_TOKENS and n * S * (Lc + S) <= self.ATTN_PAIRS
                        and n * (Lc + S) * self.kv_row_bytes <= self.KV_BYTES):
                    cur.append(r)
                    continue
            if cur:
                prompts = [groups[g][j] for _, g, j, _ in cur]
                h, pli, pos, valid, blk, lens = self._rows([p[P:] for p, (_, _, _, P) in zip(prompts, cur)],
                                                           [P for *_, P in cur], media, prompts)
                grp = torch.tensor([b for b, *_ in cur], device=pos.device)
                _, hid = self._pass(h, pli, pos, valid, ctx=ctx, grp=grp, read=lens - 1, blk=blk,
                                    split=Lc > self.ATTN_PAIRS // self.CHUNK)
                for (_, g, j, _), x in zip(cur, hid):
                    out[g][j] = x
            cur = [r] if r is not None else []

    def _singles(self, items, groups, out, media=None, two_way=False):
        """Prompts without a shared prefix (each at most CHUNK tokens; two-way ones at most max_prompt): batched by
        length, one forward each batch."""
        items = sorted(items, key=lambda x: len(groups[x[0]][x[1]]))
        cur = []
        for it in items + [None]:
            if it is not None and (not cur or (len(cur) + 1) * len(groups[it[0]][it[1]]) <= self.MAX_BATCH_TOKENS):
                cur.append(it)
                continue
            if cur:
                prompts = [groups[g][j] for g, j in cur]
                h, pli, pos, valid, blk, lens = self._rows(prompts, [0] * len(cur), media, prompts)
                _, hid = self._pass(h, pli, pos, valid, read=lens - 1, blk=blk, two_way=two_way)
                for (g, j), x in zip(cur, hid):
                    out[g][j] = x
            cur = [it] if it is not None else []

    def two_way_rows(self, toks):
        """Which prompts of one request (token ids) are read two-way: those of at most max_prompt tokens, when the
        request's record (its shared prefix once, then every prompt's own tokens) has at most max_record."""
        if not self.two_way:
            return [False] * len(toks)
        P = len(os.path.commonprefix(toks))
        record = P + sum(len(t) - P for t in toks)
        return [record <= self.two_way["max_record"] and len(t) <= self.two_way["max_prompt"] for t in toks]

    @torch.no_grad()
    def hidden(self, groups, media=None):
        """groups: [[token ids of each prompt]] -> [[final hidden state of each prompt]] ([D] each). The prompts of a
        group share a prefix (the option orders of a question, or every question of a request); media ({kind: soft
        tokens}, media_features) are those of every prompt (one request). A two-way model: each group is a request;
        its two-way prompts are read whole, one by one (nothing shared: their state depends on the question), the
        others one way as below."""
        if not self.two_way:
            return self._one_way(groups, media)
        out = [[None] * len(g) for g in groups]
        both, rest = [], []
        for gi, toks in enumerate(groups):
            ok = self.two_way_rows(toks)
            both += [(gi, j) for j, o in enumerate(ok) if o]
            if not all(ok):
                rest.append((gi, [j for j, o in enumerate(ok) if not o]))
        if both:
            self._singles(both, groups, out, media, two_way=True)
        if rest:
            for (gi, js), hs in zip(rest, self._one_way([[groups[gi][j] for j in js] for gi, js in rest], media)):
                for j, h in zip(js, hs):
                    out[gi][j] = h
        return out

    def _one_way(self, groups, media=None):
        out = [[None] * len(g) for g in groups]
        jobs, singles = [], []
        for gi, toks in enumerate(groups):
            if len(toks) > 1:
                P = self._run_start(toks[0], min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1))
                if P > 0 and max(map(len, toks)) - P <= self.CHUNK:
                    jobs.append((gi, list(range(len(toks))), P))
                    continue
            for j, t in enumerate(toks):
                if len(t) <= self.CHUNK:
                    singles.append((gi, j))
                    continue
                # a long prompt alone: its CHUNK-token parts through the keys of the earlier ones, the last part reads
                P = self._run_start(t, (len(t) - 1) // self.CHUNK * self.CHUNK)
                if P > 0:
                    jobs.append((gi, [j], P))
                else:
                    singles.append((gi, j))
        if singles:
            self._singles(singles, groups, out, media)
        jobs.sort(key=lambda jb: jb[2])
        cur = []
        for jb in jobs + [None]:
            if jb is not None and jb[2] <= self.CHUNK and (not cur or (len(cur) + 1) * jb[2] <= self.MAX_BATCH_TOKENS):
                cur.append(jb)
                continue
            if cur:
                self._shared(cur, groups, out, media)
            cur = []
            if jb is not None and jb[2] > self.CHUNK:
                self._shared([jb], groups, out, media)
            elif jb is not None:
                cur = [jb]
        if jobs and jobs[-1][2] > self.ATTN_PAIRS // self.CHUNK and self.lm.embed_tokens.weight.is_cuda:
            torch.cuda.empty_cache()  # a state over 16k tokens: its parts' cached blocks back to the GPU (as the fast
            # engine's long roots: after one large request the next ones ran out of memory sooner, ultra, 20 Oct)
        return out

    @torch.no_grad()
    def media_features(self, enc):
        """The soft tokens of a request's media from one prompt's processor output ({kind: [n, D]}): the image and audio
        towers run once for all prompts of the request."""
        mm, w = self.model.model, self.lm.embed_tokens.weight
        out = {}
        if enc.get("pixel_values") is not None:
            f = mm.get_image_features(enc["pixel_values"].to(w.device, w.dtype), enc["image_position_ids"].to(w.device),
                                      return_dict=True).pooler_output
            out["image"] = torch.cat(f, 0).to(w.dtype)
        if enc.get("input_features") is not None and enc.get("input_features_mask") is not None:
            a = mm.get_audio_features(enc["input_features"].to(w.device, w.dtype),
                                      enc["input_features_mask"].to(w.device), return_dict=True)
            out["audio"] = a.pooler_output[a.attention_mask.to(a.pooler_output.device)].to(w.dtype)
        return out
