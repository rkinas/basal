"""Causal attention for 512-wide heads (the global layers of basal-omni, Gemma 4) in Triton, for plain rows: every row
from position 0, queries and keys the same tokens. FlexAttention takes these heads only with small tiles (32 x 32: its
default tiles need more shared memory than a GPU has), so the global layers of a long state spent most of their time
there.

The kernel: one program per (128 query rows, query head, half of the output dim). S = Q K^T over key tiles of 64 (the
full 512-wide dot), an online softmax in exp2, O[:, half] += P V[:, half]; each half recomputes S (two 256-wide
accumulators fit where one 512-wide does not). Key tiles below the diagonal run unmasked, the diagonal tiles masked; the
longest rows of the causal triangle start first. Grouped key / value heads (one key head serves query heads h // group).
On an H100, against flex with 32 x 32 tiles (bf16, causal, 16k / 32k tokens): 1.92 / 2.22x for mini's global layers (8
query / 2 key heads), 1.62 / 1.58x for 1.1-max's (16 / 1), 1.60 / 1.55x for ultra's (32 / 4); the error against exact
fp32 attention the same as flex's. Its tiles need 200+ KB of shared memory per block: Hopper (sm_90) only."""
import torch

CONFIG = dict(BLOCK_M=128, BLOCK_N=64, DSPLIT=2, num_warps=8, num_stages=1)
_KERNEL = []


def available():
    """None if the kernel runs here, else why not."""
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability() != (9, 0):
        return "needs a Hopper GPU (sm_90; its tiles need more shared memory than other GPU classes have per block)"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "needs Triton"
    return None


def reference(q, k, v, scale=1.0):
    """The kernel's maths without the kernel (CPU, tests): causal softmax attention in fp32 (or wider), grouped heads."""
    g, dt = q.shape[1] // k.shape[1], torch.promote_types(q.dtype, torch.float32)
    k, v = k.to(dt).repeat_interleave(g, 1), v.to(dt).repeat_interleave(g, 1)
    s = (q.to(dt) @ k.transpose(-1, -2)) * scale
    L = q.shape[2]
    s = s.masked_fill(torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return (torch.softmax(s, -1) @ v).to(q.dtype)


def _build():
    """The kernel as the torch op basal::attn512_causal (triton_op: Inductor keeps it inside a compiled layer; a CUDA
    graph captures it like any kernel)."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def kernel(Q, K, V, O, sq_b, sq_h, sq_l, sk_b, sk_h, sk_l, sv_b, sv_h, sv_l, so_b, so_h, so_l, L, n_m, qk_scale,
               HQ: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
               DV: tl.constexpr):
        pid_m = n_m - 1 - tl.program_id(0)  # the longest rows of the causal triangle first
        bh, part = tl.program_id(1), tl.program_id(2)
        b, h = bh // HQ, bh % HQ
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        offs_v = part * DV + tl.arange(0, DV)
        rows = offs_m.to(tl.int64)
        q = tl.load(Q + b.to(tl.int64) * sq_b + h * sq_h + rows[:, None] * sq_l + offs_d[None, :],
                    mask=offs_m[:, None] < L, other=0.0)
        k_base = K + b.to(tl.int64) * sk_b + (h // GROUP) * sk_h
        v_base = V + b.to(tl.int64) * sv_b + (h // GROUP) * sv_h
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DV], tl.float32)
        lo_diag = pid_m * BLOCK_M // BLOCK_N * BLOCK_N  # the first key tile that reaches the diagonal
        hi = tl.minimum((pid_m + 1) * BLOCK_M, L)
        for start in tl.range(0, lo_diag, BLOCK_N):  # below the diagonal: no mask
            keys = (start + offs_n).to(tl.int64)
            k = tl.load(k_base + keys[None, :] * sk_l + offs_d[:, None])
            s = tl.dot(q, k) * qk_scale
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + keys[:, None] * sv_l + offs_v[None, :])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        for start in tl.range(lo_diag, hi, BLOCK_N):  # the diagonal tiles: the causal mask and the end of the row
            offs_k = start + offs_n
            keys = offs_k.to(tl.int64)
            k = tl.load(k_base + keys[None, :] * sk_l + offs_d[:, None], mask=offs_k[None, :] < L, other=0.0)
            s = tl.dot(q, k) * qk_scale
            s = tl.where((offs_m[:, None] >= offs_k[None, :]) & (offs_k[None, :] < L), s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + keys[:, None] * sv_l + offs_v[None, :], mask=offs_k[:, None] < L, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(O + b.to(tl.int64) * so_b + h * so_h + rows[:, None] * so_l + offs_v[None, :],
                 acc.to(O.dtype.element_ty), mask=offs_m[:, None] < L)

    @triton_op("basal::attn512_causal", mutates_args=())
    def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        B, Hq, L, D = q.shape
        o = torch.empty_like(q)
        cf = CONFIG
        n_m = (L + cf["BLOCK_M"] - 1) // cf["BLOCK_M"]
        if L:
            wrap_triton(kernel)[(n_m, B * Hq, cf["DSPLIT"])](
                q, k, v, o, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), L, n_m,
                scale * 1.4426950408889634, HQ=Hq, GROUP=Hq // k.shape[1], BLOCK_M=cf["BLOCK_M"],
                BLOCK_N=cf["BLOCK_N"], D=D, DV=D // cf["DSPLIT"], num_warps=cf["num_warps"],
                num_stages=cf["num_stages"])
        return o

    return op


def prepare():
    """Build the op before any layer is compiled (a compiled layer must find it registered)."""
    if not _KERNEL:
        _KERNEL.append(_build())


def attention(q, k, v, scale=1.0):
    """Causal attention of plain rows: q [b, Hq, L, 512], k / v [b, Hkv, L, 512] (Hq a multiple of Hkv; v may be k)
    -> [b, Hq, L, 512] in q's dtype."""
    if not q.is_cuda:
        return reference(q, k, v, scale)
    prepare()
    same = v is k  # (keys = values: 1.1-max's and ultra's global layers)
    q, k = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k))
    v = k if same else (v if v.stride(-1) == 1 else v.contiguous())
    return torch.ops.basal.attn512_causal(q, k, v, scale)
