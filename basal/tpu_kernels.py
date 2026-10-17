"""Plain-JAX kernels of the TPU backend
"""
import math
import jax
import jax.numpy as jnp


def dot_precision(dtype):
    return jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None


def rms_norm(x, w, eps=1e-6):
    """x (..., H), w (H,): normalised in fp32, returned in x.dtype."""
    x32 = x.astype(jnp.float32)
    x32 = x32 / jnp.sqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * w).astype(x.dtype)


def linear(x, w, b=None):
    """x (M, K) @ w (K, N) + b (N,), fp32 accumulation."""
    out = jnp.dot(x, w, preferred_element_type=jnp.float32, precision=dot_precision(x.dtype))
    if b is not None:
        out = out + b.astype(jnp.float32)
    return out.astype(x.dtype)


def swiglu(x, wg, bg, wu, bu):
    """silu(x @ wg + bg) * (x @ wu + bu); biases are added before the activation (Llama mlp_bias)."""
    prec = dot_precision(x.dtype)
    g = jnp.dot(x, wg, preferred_element_type=jnp.float32, precision=prec) + bg.astype(jnp.float32)
    u = jnp.dot(x, wu, preferred_element_type=jnp.float32, precision=prec) + bu.astype(jnp.float32)
    return (g * jax.nn.sigmoid(g) * u).astype(x.dtype)


def rope_cache(n, head_dim, theta):
    """cos / sin tables (n, head_dim // 2), fp32."""
    inv = jnp.exp(jnp.arange(head_dim // 2, dtype=jnp.float32) * -(math.log(theta) * 2.0 / head_dim))
    ang = jnp.arange(n, dtype=jnp.float32)[:, None] * inv[None]
    return jnp.cos(ang), jnp.sin(ang)


def rope(x, pos, cos, sin):
    """x (b, L, heads, d), pos (b, L): HF "rotate half" RoPE at explicit positions."""
    half = x.shape[-1] // 2
    c, s = cos[pos][:, :, None], sin[pos][:, :, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1).astype(x.dtype)


def tree_mask(end):
    """end (b, L): for every token, the end (exclusive) of the subtree of its block in the depth-first packed prefix
    trie (padding: its own index + 1) -> bool (b, L, L). allowed(i, j) = j <= i and i < end[j]: in depth-first order
    the queries inside the subtree of key j's block are exactly those of its own block and its descendants, so every
    token sees its ancestor blocks and itself only, as in a separate forward of its own prompt. Padding sits after
    every real token, so causality hides it from every row that is read out."""
    L = end.shape[1]
    i = jnp.arange(L, dtype=end.dtype)
    return (i[None, None, :] <= i[None, :, None]) & (i[None, :, None] < end[:, None, :])


def _attend(qg, k, v, mask, scale):
    """qg (b, nkv, g, q, d) query heads grouped per KV head, k / v (b, nkv, L, d), mask (b, q, L) -> (b, nkv, g, q, d)."""
    prec = dot_precision(qg.dtype)
    s = jnp.einsum("bkgqd,bkld->bkgql", qg, k, precision=prec, preferred_element_type=jnp.float32) * scale
    s = jnp.where(mask[:, None, None], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bkgql,bkld->bkgqd", p.astype(v.dtype), v, precision=prec)


def attention(q, k, v, end, scale, q_block=None, kv_block=None):
    """q (b, nh, L, d), k / v (b, nkv, L, d) with GQA, end (b, L) (see tree_mask) -> (b, nh, L, d). The query heads
    of one KV head are grouped instead of repeating K / V, so K / V are never copied per query head.

    q_block / kv_block (long prompts; L a multiple of both): blocks of q_block queries, one after the other, each over
    blocks of kv_block keys with a running max and sum (online softmax, as in flash attention). The same softmax up to
    rounding, but no tensor spans L x L (16 heads x 8192^2 x 4 bytes = 4.3 GB at once) and no reduction spans all L
    keys: on a v5e, XLA's full-row softmax hits a cliff at some lengths (6144 / 8192 keys: 128 / 291 ms per layer for
    1.5B shapes, against 2.1 / 3.6 ms blocked)."""
    b, nh, L, d = q.shape
    nkv, g = k.shape[1], nh // k.shape[1]
    qg = q.reshape(b, nkv, g, L, d)
    if q_block is None or L <= q_block:
        o = _attend(qg, k, v, tree_mask(end), scale)
        return o.reshape(b, nh, L, d).astype(q.dtype)
    kv_block = kv_block or q_block
    nq, nk = L // q_block, L // kv_block
    assert nq * q_block == L and nk * kv_block == L, (L, q_block, kv_block)
    prec = dot_precision(q.dtype)
    kb = k.reshape(b, nkv, nk, kv_block, d).transpose(2, 0, 1, 3, 4)
    vb = v.reshape(b, nkv, nk, kv_block, d).transpose(2, 0, 1, 3, 4)
    eb = end.reshape(b, nk, kv_block).transpose(1, 0, 2)

    def q_step(args):
        qi, q0 = args  # (b, nkv, g, q_block, d), first query index of the block
        i = q0 + jnp.arange(q_block, dtype=end.dtype)

        def kv_step(carry, blk):
            m, l, acc = carry
            kj, vj, ej, k0 = blk
            j = k0 + jnp.arange(kv_block, dtype=end.dtype)
            mask = (j[None, None, :] <= i[None, :, None]) & (i[None, :, None] < ej[:, None, :])
            s = jnp.einsum("bkgqd,bkld->bkgql", qi, kj, precision=prec, preferred_element_type=jnp.float32) * scale
            s = jnp.where(mask[:, None, None], s, -jnp.inf)
            m_new = jnp.maximum(m, s.max(-1))
            m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)  # nothing visible yet: p and the rescale are 0
            p = jnp.exp(s - m_safe[..., None])
            a = jnp.exp(m - m_safe)
            pv = jnp.einsum("bkgql,bkld->bkgqd", p.astype(vj.dtype), vj, precision=prec,
                            preferred_element_type=jnp.float32)
            return (m_new, l * a + p.sum(-1), acc * a[..., None] + pv), None

        init = (jnp.full((b, nkv, g, q_block), -jnp.inf, jnp.float32), jnp.zeros((b, nkv, g, q_block), jnp.float32),
                jnp.zeros((b, nkv, g, q_block, d), jnp.float32))
        (_, l, acc), _ = jax.lax.scan(kv_step, init, (kb, vb, eb, jnp.arange(nk, dtype=end.dtype) * kv_block))
        return acc / l[..., None]  # every real row sees itself, so l > 0

    blocks = qg.reshape(b, nkv, g, nq, q_block, d).transpose(3, 0, 1, 2, 4, 5)
    o = jax.lax.map(q_step, (blocks, jnp.arange(nq, dtype=end.dtype) * q_block))  # (nq, b, nkv, g, q_block, d)
    return o.transpose(1, 2, 3, 0, 4, 5).reshape(b, nh, L, d).astype(q.dtype)
