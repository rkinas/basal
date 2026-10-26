"""The opt-in decoder forward of the MLX backend (basal 2.5, --mlx-forward basal): the layers of the mlx-lm model (the
same modules and weights) in basal's own loop, for the Llama decoders (basal-1.x, 2.x) and Gemma 4 (basal-omni). What
mlx-lm's forward does not do:

- **Readout rows only.** From the last layer with its own keys and values on, a layer's output matters only at the
  readout positions (the last token of each prompt). That layer computes its keys and values at every position, and its
  queries, attention, output projection and MLP at the readout rows only; Gemma 4 E2B / E4B reuse the keys and values of
  earlier layers in the layers after it (E4B: the last 18 of 42), which run at the readout rows only as well. A shared
  prefix stops after the keys and values of that layer. A readout row's query sees the same keys through the same mask
  row, as a row of its own, so its probabilities equal the full forward's up to float rounding.
- **Early start.** mx.async_eval after the first layer and then every ASYNC_EVERY layers: the GPU runs the first layers
  while Python builds the graph of the rest.
- **Merged projections, checked at load.** The gate and up projections (and q / k / v) become one matrix each, made at
  load; the separate modules keep views of it, so the weights take no extra memory. Every merged product is compared
  with the separate ones bit for bit, on the real weights of one layer of each shape and on CHECK_ROWS (1 to 4,096
  rows); a merge whose bits differ anywhere is undone (MLX picks its matmul kernel by shape, so a wider matrix can sum
  in another order).

The layers before that last own-key layer compute what mlx-lm's do, op for op (bit-identical with no merge or a merge
that passed its check). Other architectures, and Llama models with sliding windows or a RoPE whose frequencies depend on
the sequence length, keep mlx-lm's forward (build returns None).
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text, llama
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import Llama3RoPE, ProportionalRoPE

STATIC_ROPES = (nn.RoPE, Llama3RoPE, ProportionalRoPE)  # frequencies independent of the sequence length


def bits_equal(a, b):
    """Same shape, dtype and bits in every element."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}.get(a.dtype.size)
    if view is None:
        return bool(mx.array_equal(a, b).item())
    return bool(mx.all(mx.contiguous(a).view(view) == mx.contiguous(b).view(view)).item())


def _stacked(mods):
    """One linear layer of the class of mods[0] whose arrays (weight; scales and biases when quantised; bias) are those
    of mods stacked along the output dimension; None if they do not stack."""
    m0 = mods[0]
    if any(type(m) is not type(m0) or set(m.keys()) != set(m0.keys()) for m in mods):
        return None
    if any(getattr(m, k, None) != getattr(m0, k, None) for m in mods for k in ("group_size", "bits", "mode")):
        return None
    new = type(m0).__new__(type(m0))
    nn.Module.__init__(new)
    for k in ("group_size", "bits", "mode"):
        if hasattr(m0, k):
            setattr(new, k, getattr(m0, k))
    for k, v in m0.items():
        if not isinstance(v, mx.array) or v.ndim == 0 or any(m[k].shape[1:] != v.shape[1:] for m in mods):
            return None
        new[k] = mx.concatenate([m[k] for m in mods], axis=0)
    mx.eval(new.parameters())
    return new


def _signature(mods):
    """What decides the matmul kernels of a group: classes, quantisation and array shapes / dtypes."""
    return tuple((type(m).__name__, getattr(m, "bits", None), getattr(m, "group_size", None), getattr(m, "mode", None),
                  tuple((k, v.shape, str(v.dtype)) for k, v in sorted(m.items()) if isinstance(v, mx.array)))
                 for m in mods)


class _Group:
    """Linear layers that read the same input: one product of the merged layer (split), or one product each."""

    def __init__(self, label, mods):
        self.label, self.mods, self.cat = label, list(mods), None
        sizes = [m["weight"].shape[0] for m in self.mods]
        self.cuts = [sum(sizes[: k + 1]) for k in range(len(sizes) - 1)]

    def __call__(self, x):
        if self.cat is None:
            return [m(x) for m in self.mods]
        return mx.split(self.cat(x), self.cuts, axis=-1)

    def merge(self, cat):
        """Use the merged layer; the separate layers keep views of its arrays (the old arrays are released)."""
        self.cat, start = cat, 0
        for m, end in zip(self.mods, self.cuts + [None]):
            for k in list(m.keys()):
                if isinstance(m[k], mx.array):
                    m[k] = cat[k][start:end]
            start = end
        mx.eval([m.parameters() for m in self.mods])


class MLXForward:
    """The forward over the layers of an mlx-lm Llama or Gemma 4 text model (see the module docstring)."""

    ASYNC_EVERY = 4
    # (rows, tokens) of the self-check inputs: the one-row and batched calls of the backend, 1 to 4,096 rows per product
    CHECK_ROWS = ((1, 1), (3, 1), (8, 1), (16, 1), (1, 9), (2, 16), (1, 64), (16, 24), (1, 255), (4, 128), (1, 1000),
                  (8, 256), (1, 2048), (1, 4096))

    @classmethod
    def build(cls, lm, merge=True):
        """The forward of this mlx-lm model, or None when it keeps mlx-lm's (another architecture, sliding windows on
        Llama, a RoPE that depends on the sequence length)."""
        if isinstance(lm, llama.Model):
            if lm.model.swa_idx is not None:
                return None
        elif not isinstance(lm, gemma4_text.Model):
            return None
        if not all(type(layer.self_attn.rope) in STATIC_ROPES for layer in lm.model.layers):
            return None
        return cls(lm, merge)

    def __init__(self, lm, merge=True):
        self.m = lm.model
        self.layers = list(self.m.layers)
        n = len(self.layers)
        self.gemma = isinstance(lm, gemma4_text.Model)
        if self.gemma:
            self.own = [layer.self_attn.has_kv for layer in self.layers]
            self.prev = list(self.m.previous_kvs)
            self.types = [layer.layer_type for layer in self.layers]
        else:
            self.own, self.prev, self.types = [True] * n, list(range(n)), ["full_attention"] * n
        self.last = max(i for i in range(n) if self.own[i])  # from this layer on: the readout rows only
        # the projections that read the same input: q / k / v of an own-key layer (k / v in the last one, whose
        # queries are the readout rows'; k alone with Gemma's attention_k_eq_v), gate / up of every MLP
        self.qkv, self.gu = [], []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            names = ["q_proj", "k_proj", "v_proj"]
            if self.gemma and a.use_k_eq_v:
                names.remove("v_proj")
            if i == self.last:
                names.remove("q_proj")
            self.qkv.append(_Group("k/v" if i == self.last else "q/k/v", [getattr(a, k) for k in names])
                            if self.own[i] else None)
            self.gu.append(_Group("gate/up", [layer.mlp.gate_proj, layer.mlp.up_proj]))
        self.checks = {}  # group label -> "merged" or why it was kept
        if merge:
            self._merge()

    # ---- merged projections and their self-check -------------------------------------------------------------------

    def _check(self, mods, cat, cuts, dim, dtype):
        """"merged" if the merged product equals the separate ones bit for bit on every CHECK_ROWS input."""
        key = mx.random.key(0)
        for B, L in self.CHECK_ROWS:
            key, sub = mx.random.split(key)
            x = mx.random.normal((B, L, dim), key=sub).astype(dtype)
            got = mx.split(cat(x), cuts, axis=-1)
            if not all(bits_equal(g, m(x)) for g, m in zip(got, mods)):
                return f"kept (bits differ at {B} x {L})"
        return "merged"

    def _merge(self):
        """Merges every group of two or more projections whose shape signature passed the self-check (run once per
        signature, on the first layer that has it)."""
        verdict = {}
        dtype = self.layers[0].input_layernorm.weight.dtype
        for i, g in [(i, g) for groups in (self.qkv, self.gu) for i, g in enumerate(groups)]:
            if g is None or len(g.mods) < 2:
                continue
            sig = _signature(g.mods)
            cat = _stacked(g.mods) if verdict.get(sig, "merged") == "merged" else None
            if sig not in verdict:
                verdict[sig] = "kept (does not stack)" if cat is None else \
                    self._check(g.mods, cat, g.cuts, self.layers[i].input_layernorm.weight.shape[0], dtype)
            if cat is not None and verdict[sig] == "merged":
                g.merge(cat)
            if self.checks.get(g.label, "merged") == "merged":
                self.checks[g.label] = verdict[sig]

    def describe(self):
        """One line for the start-up message."""
        merged = ", ".join(f"{k} {v}" for k, v in self.checks.items())
        return f"readout rows from layer {self.last + 1} of {len(self.layers)}" + (f"; {merged}" if merged else "")

    # ---- the forward ---------------------------------------------------------------------------------------------

    def prefix(self, ids, past=None):
        """ids [1, P] -> per own-key layer up to the last one, (keys, values) [1, H, P, D] of every position; nothing
        after the last one's keys and values runs. past: those of the tokens before ids (M of them, a cached state),
        which the forward continues (positions from M) -> keys / values of all M + P positions."""
        out = []
        self._run(ids, None, past, None, None, out)
        return out

    def hidden(self, ids, ro, past=None, pos=None, mask=None, full=False):
        """ids [B, L] right-padded rows -> final-norm hidden states [B, R, D] at the token indices ro [B, R] of each
        row. past: per own-key layer (keys, values) [1, H, P, D] of a prefix every row continues (positions from P).
        pos / mask: packed rows (no past): positions [B, L] and the boolean block mask [B, 1, L, L]. full: every
        layer at every position and the rows taken after the final norm, as mlx-lm's forward (for checks)."""
        return self._run(ids, ro, past, pos, mask, None, full)

    def _run(self, ids, ro, past, pos, mask, kv_out, full=False):
        m = self.m
        B, L = ids.shape
        cached = past is not None or kv_out is not None  # mlx-lm with a KV cache: its masks and RoPE offsets
        P = past[0][0].shape[2] if past is not None else 0
        h = m.embed_tokens(ids)
        pli = [None] * len(self.layers)
        if self.gemma:
            emb, h = h, h * m.embed_scale
            if m.hidden_size_per_layer_input:
                p = m._project_per_layer_inputs(h, m._get_per_layer_inputs(ids, emb))
                pli = [p[:, :, i, :] for i in range(len(self.layers))]
            off = mx.array(P) if cached else 0
        else:
            off = P
        masks = {t: self._mask(t, L, P, cached, mask) for t in set(self.types)}
        kv = [None] * len(self.layers)  # (keys, values, RoPE offset) of the own-key layers
        for i, layer in enumerate(self.layers[: self.last] if not full else self.layers):
            h = self._layer(i, layer, h, masks[self.types[i]], past, off, pos, pli[i], kv)
            if i == 0 or (i + 1) % self.ASYNC_EVERY == 0:
                mx.async_eval(h)
        if full:
            return mx.take_along_axis(m.norm(h), ro[..., None], axis=1)
        i = self.last
        if kv_out is not None:  # a prefix: the last own-key layer's keys and values, then stop
            self._keys(i, self.layers[i].self_attn, self.layers[i].input_layernorm(h), past, off, pos, kv)
            kv_out += [x[:2] for x in kv[: i + 1]]
            return None
        # the readout rows: their positions, mask rows and states
        rows = mx.take_along_axis(pos, ro, axis=1) if pos is not None else ro + P
        rmask = {t: self._row_mask(t, rows, P + L, mask, ro) for t in set(self.types)}
        x = h
        h = mx.take_along_axis(h, ro[..., None], axis=1)
        for i in range(self.last, len(self.layers)):
            layer = self.layers[i]
            p = mx.take_along_axis(pli[i], ro[..., None], axis=1) if pli[i] is not None else None
            if i == self.last:
                self._keys(i, layer.self_attn, layer.input_layernorm(x), past, off, pos, kv)
            h = self._layer(i, layer, h, rmask[self.types[i]], past, off, pos, p, kv, rows)
        return m.norm(h)

    def _mask(self, t, L, P, cached, mask):
        """The mask of a layer type for every position, as mlx-lm builds it (with or without a KV cache)."""
        if mask is not None:
            return mask
        window = self.m.window_size if self.gemma and t == "sliding_attention" else None
        if window is not None and (cached or L > window):
            return create_causal_mask(L, P, window_size=window)
        return "causal" if L > 1 else None

    def _row_mask(self, t, rows, n, mask, ro):
        """Boolean mask [B, 1, R, n] of the readout rows (absolute positions rows [B, R]) over n keys: the rows of the
        packed block mask, else causal (within the window on a sliding-window layer)."""
        if mask is not None:
            return mx.take_along_axis(mask, ro[:, None, :, None], axis=2)
        k, q = mx.arange(n)[None, None], rows[..., None]
        out = k <= q
        if self.gemma and t == "sliding_attention":
            out = out & (q < k + self.m.window_size)
        return out[:, None]

    @staticmethod
    def _rope(rope, x, off, pos):
        """RoPE of x [B, H, L, D] at offset off, or at the positions pos [B, L]: each token as its own row of the same
        kernel (bit-identical to a contiguous run; MLXBackend's packed rows do the same)."""
        if pos is None:
            return rope(x, offset=off)
        B, H, L, D = x.shape
        y = rope(x.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D), offset=pos.reshape(-1))
        return y.reshape(B, L, H, D).transpose(0, 2, 1, 3)

    def _keys(self, i, a, x, past, off, pos, kv, outs=None):
        """Keys and values of own-key layer i from its normed input x [B, L, D] (every position), after the prefix's
        when there is one: kv[i] = (keys, values, RoPE offset). outs: its k / v projections when already made."""
        B, L, _ = x.shape
        k, *v = outs if outs is not None else self.qkv[i](x)
        if self.gemma:
            k = k.reshape(B, L, a.n_kv_heads, a.head_dim)
            v = v[0].reshape(B, L, a.n_kv_heads, a.head_dim) if v else k  # attention_k_eq_v: the key projection
            k = self._rope(a.rope, a.k_norm(k).transpose(0, 2, 1, 3), off, pos)
            v = a.v_norm(v).transpose(0, 2, 1, 3)
        else:
            k = self._rope(a.rope, k.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3), off, pos)
            v = v[0].reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if past is not None:
            pk, pv = past[i]
            k = mx.concatenate([mx.repeat(pk, B, axis=0), k], axis=2)
            v = mx.concatenate([mx.repeat(pv, B, axis=0), v], axis=2)
        kv[i] = (k, v, off)

    def _layer(self, i, layer, h, mask, past, off, pos, pli, kv, rows=None):
        """Layer i on h [B, L, D] (every position), or with rows (their absolute positions [B, R]) on the readout
        rows h [B, R, D], over the keys and values in kv."""
        a = layer.self_attn
        B, L, _ = h.shape
        x = layer.input_layernorm(h)
        if rows is None and self.own[i]:
            q, *outs = self.qkv[i](x) if i != self.last else [a.q_proj(x)] + self.qkv[i](x)
            self._keys(i, a, x, past, off, pos, kv, outs)
        else:
            q = a.q_proj(x)
        k, v, qoff = kv[self.prev[i]]
        q = q.reshape(B, L, a.n_heads, -1)
        if self.gemma:
            q = a.q_norm(q)
        q = q.transpose(0, 2, 1, 3)
        q = self._rope(a.rope, q, qoff, pos) if rows is None else self._rope(a.rope, q, None, rows)
        o = scaled_dot_product_attention(q, k, v, cache=None, scale=a.scale, mask=mask)
        r = a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
        if not self.gemma:
            h = h + r
            g, u = self.gu[i](layer.post_attention_layernorm(h))
            return h + layer.mlp.down_proj(swiglu(g, u))
        h = h + layer.post_attention_layernorm(r)
        res = h
        g, u = self.gu[i](layer.pre_feedforward_layernorm(h))
        h = layer.mlp.down_proj(gemma4_text.geglu(g, u))
        if layer.enable_moe:
            h1 = layer.post_feedforward_layernorm_1(h)
            idx, w = layer.router(res)
            h2 = layer.post_feedforward_layernorm_2(layer.experts(layer.pre_feedforward_layernorm_2(res), idx, w))
            h = h1 + h2
        h = res + layer.post_feedforward_layernorm(h)
        if layer.per_layer_input_gate is not None and pli is not None:
            gate = nn.gelu_approx(layer.per_layer_input_gate(h))
            h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(mx.multiply(gate, pli)))
        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar
        return h
