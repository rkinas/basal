"""NVFP4 linears for the CUDA graph engine (--mode nvfp4; basal-omni, Gemma 4; Blackwell): e2m1 values, one e4m3
scale per 16 consecutive input features, one fp32 global scale per tensor; the GEMM is torch._scaled_mm on cuBLAS's
block-scaled FP4 kernel (sm_100 / sm_120), the scales in its 128 x 4 tile swizzle.

Weights: the global scale is amax / (6 * 448); each block of 16 takes the scale (of 8 candidates, amax / 6 times
1.00, 0.96, ..., 0.72, rounded to e4m3) with the smallest squared error. Activations: a fixed global scale per linear
(calibrated: 2 x the largest input |x| over calibration prompts in bf16, / (6 * 448)), the block scales per token
from the block's own amax, so a row's codes do not depend on the other rows. Values round to the nearest e2m1 (ties
to even).

  quantize(w)               weight [N, K] -> (codes [N, K / 2] uint8, swizzled e4m3 scales, global scale)
  act_quant(x, inv_g)       activations [M, K] -> (codes [Mp, K / 2], swizzled scales); Mp = M rounded up to 128
  linear(x, p)              x [..., K] -> [..., N] through (codes, scales, alpha, inv_g, bias)
  NVFP4Linear               a linear layer holding them; lean.linear runs its parameters in compiled layers
"""
import torch

BLOCK = 16
FP4_MAX, E4M3_MAX = 6.0, 448.0
E4M3 = torch.float8_e4m3fn
MARGIN = 2.0   # activations: the global scale covers 2 x the calibrated amax (values past it are clipped)
CANDIDATES = tuple(1.0 - 0.04 * j for j in range(8))
_KERNEL = []


def fits(lin):
    """True for a weight this module can hold: K a multiple of 64 (four scale blocks), N a multiple of 16."""
    n, k = lin.weight.shape
    return k % 64 == 0 and n % 16 == 0


def _codes(u):
    """fp32 values in units of their block scale -> e2m1 codes (sign bit 8), round to nearest, ties to even."""
    a = u.abs()
    c = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
         + (a >= 3.5).int() + (a > 5.0).int())
    return (c | ((u < 0).int() << 3)).to(torch.uint8)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _values(c):
    v = torch.tensor(_E2M1, device=c.device)[(c & 7).long()]
    return torch.where(c >= 8, -v, v)


def swizzle(s):
    """[rows, K / 16] e4m3 scales -> cuBLAS's 128 x 4 tiles, flattened (rows padded to 128, columns to 4 with 0)."""
    r, c = s.shape
    R, C = -(-r // 128), -(-c // 4)
    p = torch.zeros(R * 128, C * 4, dtype=torch.uint8, device=s.device)
    p[:r, :c] = s.view(torch.uint8)
    b = p.view(R, 128, C, 4).permute(0, 2, 1, 3)
    return b.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).view(E4M3)


def unswizzle(f, rows, cols):
    """The inverse of swizzle -> [rows, cols] e4m3."""
    R, C = -(-rows // 128), -(-cols // 4)
    b = f.view(torch.uint8).view(R * C, 32, 4, 4).transpose(1, 2).reshape(R, C, 128, 4).permute(0, 2, 1, 3)
    return b.reshape(R * 128, C * 4)[:rows, :cols].contiguous().view(E4M3)


def pack(c):
    """[M, K] codes -> [M, K / 2] bytes (even feature in the low nibble)."""
    return c[:, 0::2] | (c[:, 1::2] << 4)


def unpack(p):
    return torch.stack([p & 15, p >> 4], -1).view(p.shape[0], -1)


@torch.no_grad()
def quantize(w, rows=4096):
    """[N, K] -> (codes [N, K / 2] uint8, swizzled e4m3 block scales, fp32 global scale [1]), in parts of rows."""
    N, K = w.shape
    g = (w.float().abs().amax() / (FP4_MAX * E4M3_MAX)).clamp(min=1e-12)
    codes, scales = [], []
    for o in range(0, N, rows):
        b = w[o: o + rows].float().view(-1, K // BLOCK, BLOCK) / g
        amax = b.abs().amax(-1, keepdim=True)
        best = err = None
        for f in CANDIDATES:
            s = (amax * f / FP4_MAX).clamp(max=E4M3_MAX).to(E4M3).float()
            u = torch.where(s > 0, b / s, torch.zeros_like(b))
            c = _codes(u)
            e = ((_values(c) * s - b) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, err, sc = c, e, s
            else:
                m = e < err
                best, err, sc = torch.where(m, c, best), torch.where(m, e, err), torch.where(m, s, sc)
        codes.append(pack(best.view(best.shape[0], K)))
        scales.append(sc[..., 0].to(E4M3))
    return torch.cat(codes), swizzle(torch.cat(scales)), g.reshape(1)


def dequantize(codes, scales, g, N, K):
    """The weight's values as fp32 [N, K] (tests, the reference maths)."""
    s = unswizzle(scales, N, K // BLOCK).float()
    return (_values(unpack(codes)).view(N, K // BLOCK, BLOCK) * s[..., None]).view(N, K) * g


def _act_ops(x, inv_g):
    """act_quant's maths in torch ops (CPU, tests): -> (codes [M, K / 2], e4m3 scales [M, K / 16], unswizzled)."""
    M, K = x.shape
    b = x.float().view(M, K // BLOCK, BLOCK) * inv_g
    s = (b.abs().amax(-1, keepdim=True) / FP4_MAX).clamp(max=E4M3_MAX).to(E4M3).float()
    u = torch.where(s > 0, b / s, torch.zeros_like(b))
    return pack(_codes(u).view(M, K)), s[..., 0].to(E4M3)


def _build():
    """The activation quantiser as the torch op basal::nvfp4_act_quant: one Triton program per 128 rows x 64
    features writes the packed codes and the four block scales of each row straight into cuBLAS's swizzle."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def kernel(x, q, s, inv_g, M, K, C4, BM: tl.constexpr):
        pm, pk = tl.program_id(0), tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rk = pk * 64 + tl.arange(0, 64)
        v = tl.load(x + rm.to(tl.int64)[:, None] * K + rk[None, :], mask=(rm < M)[:, None], other=0.0).to(tl.float32)
        v = tl.reshape(v * tl.load(inv_g), (BM, 4, 16))  # IEEE division below (Triton's / is approximate)
        amax = tl.max(tl.abs(v), 2)
        sc = tl.minimum(tl.math.div_rn(amax, 6.0), 448.0).to(tl.float8e4nv)
        scf = sc.to(tl.float32)
        u = tl.where(scf[:, :, None] > 0, tl.math.div_rn(v, tl.broadcast_to(scf[:, :, None], (BM, 4, 16))), 0.0)
        a = tl.abs(u)
        c = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
             + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
        c = c | ((u < 0).to(tl.int32) << 3)
        lo, hi = tl.split(tl.reshape(c, (BM, 32, 2)))
        byte = (lo | (hi << 4)).to(tl.uint8)
        tl.store(q + rm.to(tl.int64)[:, None] * (K // 2) + pk * 32 + tl.arange(0, 32)[None, :], byte)
        # scale (row r, block 4 pk + j): tile (r // 128, pk) of 512 bytes, at (r % 32) * 16 + (r % 128 // 32) * 4 + j
        r = rm % 128
        off = ((rm // 128).to(tl.int64) * C4 + pk) * 512 + (r % 32) * 16 + (r // 32) * 4
        tl.store(s + off[:, None] + tl.arange(0, 4)[None, :], sc.to(tl.uint8, bitcast=True))

    @triton_op("basal::nvfp4_act_quant", mutates_args=())
    def op(x: torch.Tensor, inv_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        M, K = x.shape
        Mp = (M + 127) // 128 * 128  # no max() / branch on M: a traced size must stay one expression of M
        q = torch.empty((Mp, K // 2), dtype=torch.uint8, device=x.device)
        s = torch.empty((Mp * (K // BLOCK),), dtype=torch.uint8, device=x.device)
        if M:
            wrap_triton(kernel)[(Mp // 128, K // 64)](x, q, s, inv_g, M, K, K // 64, BM=128)
        return q, s

    return op


def act_quant(x, inv_g):
    """[M, K] bf16 -> (codes [Mp, K / 2] uint8, swizzled scales uint8 [Mp * K / 16]), Mp = M rounded up to 128; rows
    past M are zeros."""
    if not _KERNEL:
        _KERNEL.append(_build())
    return torch.ops.basal.nvfp4_act_quant(x, inv_g)


def build():
    """The Triton op, before any layer is compiled (no graph break while tracing)."""
    if torch.cuda.is_available() and not _KERNEL:
        _KERNEL.append(_build())


@torch.library.custom_op("basal::nvfp4_gemm", mutates_args=())
def _gemm(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor, m: int,
          dtype: torch.dtype) -> torch.Tensor:
    """cuBLAS's block-scaled FP4 GEMM, the global scales (alpha) applied in its fp32 epilogue (F.scaled_mm, torch 2.10+;
    the same bits as torch._scaled_mm in fp32 times alpha and 1.1-2.1x faster on sm_120, 24 Oct)."""
    a, b = q.view(torch.float4_e2m1fn_x2), w.view(torch.float4_e2m1fn_x2).t()
    F = torch.nn.functional
    if hasattr(F, "scaled_mm"):
        st, sw = F.ScalingType, F.SwizzleType
        y = F.scaled_mm(a, b, [s.view(E4M3), alpha], [st.BlockWise1x16, st.TensorWise],
                        [ws.view(E4M3), torch.ones_like(alpha)], [st.BlockWise1x16, st.TensorWise],
                        swizzle_a=[sw.SWIZZLE_32_4_4, sw.NO_SWIZZLE], swizzle_b=[sw.SWIZZLE_32_4_4, sw.NO_SWIZZLE],
                        output_dtype=dtype)
        return y[:m].clone() if m < y.shape[0] else y
    y = torch._scaled_mm(a, b, s.view(E4M3), ws.view(E4M3), out_dtype=torch.float32)
    return (y[:m] * alpha).to(dtype)


@_gemm.register_fake
def _(q, s, w, ws, alpha, m, dtype):
    return q.new_empty((m, w.shape[0]), dtype=dtype)


def linear(x, w, ws, alpha, inv_g, bias=None):
    """x [..., K] (bf16) through an NVFP4 weight (codes w [N, K / 2], swizzled scales ws, alpha = the two global
    scales' product, inv_g = 1 / the input's global scale) -> [..., N] in x's dtype."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    N = w.shape[0]
    if x.is_cuda:
        q, s = act_quant(x2.contiguous(), inv_g)
        y = _gemm(q, s, w, ws, alpha, x2.shape[0], x.dtype)
    else:  # the reference maths: both operands dequantised, fp32 products
        K = x2.shape[1]
        qc, sc = _act_ops(x2, inv_g)
        xd = (_values(unpack(qc)).view(-1, K // BLOCK, BLOCK) * sc.float()[..., None]).view(-1, K)
        wd = _values(unpack(w)).view(N, K // BLOCK, BLOCK) * unswizzle(ws, N, K // BLOCK).float()[..., None]
        y = ((xd @ wd.view(N, K).t()) * alpha).to(x.dtype)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*shape[:-1], N)


class NVFP4Linear(torch.nn.Module):
    """A linear layer with an NVFP4 weight (quantize) and a fixed input global scale; part(o, n) is a view of output
    rows o..o + n (multiples of 128: whole scale tiles), so the projections of a merged weight stay modules."""

    def __init__(self, weight, scale, wg, xg, bias=None, out_features=None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.alpha = torch.nn.Parameter((wg * xg).float().reshape(1), requires_grad=False)
        self.inv_g = torch.nn.Parameter((1.0 / xg).float().reshape(1), requires_grad=False)
        self.wg, self.xg = wg, xg
        self.bias = bias
        self.out_features, self.in_features = weight.shape[0], weight.shape[1] * 2

    @classmethod
    def of(cls, lin, amax):
        """The NVFP4 version of a torch.nn.Linear whose inputs reached |x| <= amax in calibration (bf16 not kept)."""
        q, s, wg = quantize(lin.weight.detach())
        xg = torch.tensor([max(float(amax), 1e-6) * MARGIN / (FP4_MAX * E4M3_MAX)], device=q.device)
        return cls(q, s, wg, xg, None if lin.bias is None else lin.bias.detach())

    def part(self, o, n):
        assert o % 128 == 0 and n % 128 == 0, "parts of whole scale tiles"
        cols = self.in_features // BLOCK
        C = -(-cols // 4) * 4
        return NVFP4Linear(self.weight[o: o + n], self.weight_scale[o * C: (o + n) * C], self.wg, self.xg,
                           None if self.bias is None else self.bias[o: o + n])

    def forward(self, x):
        return linear(x, self.weight, self.weight_scale, self.alpha, self.inv_g, self.bias)
