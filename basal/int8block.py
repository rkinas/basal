"""Block-scaled int8 linears for the CUDA graph engine (--mode int8; basal-omni, Gemma 4): the layout of
basal/fp8block.py with int8 values. Weights in 128 x 128 blocks and activations in 1 x 128 tiles (a token's 128
consecutive input features), each block with its own fp32 scale (amax / 127, values rounded to the nearest integer);
the GEMM multiplies int8 by int8 into int32 for every 128-deep slice of the reduction (exact), applies both scales and
adds the slices in fp32. For GPUs without FP8 (Ampere, sm80 / sm86): their int8 tensor cores, half the memory of bf16.

  quantize(w)            weight [N, K] -> (int8 [N, K], scales [N / 128, K / 128])
  act_quant(x)           activations [M, K] -> (int8 [M, K], scales [M, K / 128])
  gemm(q, s, w, ws, dt)  the Triton GEMM: [M, N] in dt (CPU: the same maths dequantised)
  linear(x, w, ws, b)    x [..., K] -> [..., N]: act_quant + gemm; the bias
  Int8BlockLinear        a linear layer holding (w, ws, bias), a BlockFP8Linear whose weight is int8 (lean.linear
                         picks the GEMM by the weight's dtype)
"""
import torch

from .fp8block import BLOCK, BlockFP8Linear, fits, grid  # noqa: F401  (fits: the same shapes)

INT8_MAX = 127.0
# the GEMM's tiles and launch, fixed for every shape as fp8block's (a row's output must not depend on how many rows
# there are); pipeline stages from the GPU's shared memory (32 KB a stage: 4 on A100 / H100, 2 on sm86 / sm89)
CONFIG = dict(BM=128, BN=128, GROUP_M=8, num_warps=8)
_KERNEL, _ACT = [], []


def int8_refused():
    """Why --mode int8 cannot run here, None if it can: its GEMM is a Triton kernel on int8 tensor cores (compute
    capability 8.0 or newer: Ampere, Ada, Hopper, Blackwell)."""
    if not torch.cuda.is_available():
        return "--mode int8 needs a CUDA GPU of compute capability 8.0 or newer (Ampere and newer)"
    cc = torch.cuda.get_device_capability()
    if tuple(cc) < (8, 0):
        return (f"--mode int8 needs a GPU of compute capability 8.0 or newer (Ampere and newer); "
                f"{torch.cuda.get_device_name()} is sm_{cc[0]}{cc[1]}. Serve with --mode fast (bf16) on this GPU")
    return None


@torch.no_grad()
def quantize(w):
    """[N, K] (bf16 / fp32) -> (int8 [N, K], fp32 scales [N / 128, K / 128]): each 128 x 128 block divided by its
    amax / 127 and rounded (ties to even)."""
    N, K = w.shape
    b = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    s = b.abs().amax((1, 3)).clamp(min=1e-12) / INT8_MAX
    q = (b / s[:, None, :, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8).view(N, K)
    return q, s.contiguous()


def _act_quant_ops(x):
    M, K = x.shape
    b = x.float().view(M, K // BLOCK, BLOCK)
    s = b.abs().amax(-1).clamp(min=1e-12) / INT8_MAX
    return (b / s[..., None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8).view(M, K), s


def act_quant(x):
    """[M, K] -> (int8 [M, K], fp32 scales [M, K / 128]): each token's 128-feature tile by its own amax. Inside a
    compiled forward Inductor fuses it with what precedes it; elsewhere on CUDA one compiled kernel (any shape)."""
    if torch.compiler.is_compiling() or not x.is_cuda:
        return _act_quant_ops(x)
    if not _ACT:
        _ACT.append(torch.compile(_act_quant_ops, dynamic=True))
    return _ACT[0](x)


def _reference(q, s, w, ws, dtype):
    """The GEMM's maths without the kernel (CPU, tests): both operands dequantised, fp32 products."""
    M, K = q.shape
    N = w.shape[0]
    x = (q.float().view(M, K // BLOCK, BLOCK) * s[..., None]).view(M, K)
    y = (w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK) * ws[:, None, :, None]).view(N, K)
    return (x @ y.t()).to(dtype)


def _stages():
    smem = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
    return max(2, min(4, (smem - 8 * 1024) // (32 * 1024)))


def _build():
    """The Triton kernel as the torch op basal::int8_block_gemm (triton_op: Inductor keeps it inside the compiled
    layer, no graph break; a CUDA graph captures it like any kernel)."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def kernel(a, a_s, b, b_s, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
        # a [M, K] int8, a_s [M, K / BK]; b [N, K] int8, b_s [N / BK, K / BK]; c [M, N]. As fp8block's kernel: groups
        # of GROUP_M row tiles, rows past M read row (r mod M) and are not stored; each BK slice exact in int32
        pid = tl.program_id(0)
        nm, nn = tl.cdiv(M, BM), N // BN
        width = GROUP_M * nn
        first = (pid // width) * GROUP_M
        gm = tl.minimum(nm - first, GROUP_M)
        pm = first + (pid % width) % gm
        pn = (pid % width) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        ra = (rm % M).to(tl.int64)
        rk = tl.arange(0, BK)
        nk = K // BK
        ap = a + ra[:, None] * K + rk[None, :]
        bp = b + rn.to(tl.int64)[None, :] * K + rk[:, None]
        asp = a_s + ra * nk
        bsp = b_s + (rn // BK) * nk
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(nk):
            x = tl.load(ap)
            y = tl.load(bp)
            acc += tl.dot(x, y, out_dtype=tl.int32).to(tl.float32) * tl.load(asp)[:, None] * tl.load(bsp)[None, :]
            ap += BK
            bp += BK
            asp += 1
            bsp += 1
        cp = c + rm.to(tl.int64)[:, None] * N + rn[None, :]
        tl.store(cp, acc.to(c.dtype.element_ty), mask=(rm < M)[:, None])

    stages = _stages()

    @triton_op("basal::int8_block_gemm", mutates_args=())
    def op(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        M, K = q.shape
        N = w.shape[0]
        c = torch.empty((M, N), dtype=dtype, device=q.device)
        if M:
            wrap_triton(kernel)[grid(M, N, CONFIG)](q, s, w, ws, c, M, N, K, BK=BLOCK, num_stages=stages, **CONFIG)
        return c

    return op


def build():
    """The Triton kernel, built once (before any layer is compiled)."""
    if torch.cuda.is_available() and not _KERNEL:
        _KERNEL.append(_build())


def gemm(q, s, w, ws, dtype):
    """(q, s) [M, K] x (w, ws) [N, K]^T -> [M, N] in dtype."""
    if not q.is_cuda:
        return _reference(q, s, w, ws, dtype)
    build()
    return torch.ops.basal.int8_block_gemm(q, s.contiguous(), w, ws, dtype)


def linear(x, w, ws, bias=None):
    """x [..., K] through a block-scaled int8 weight -> [..., N] in x's dtype."""
    shape = x.shape
    q, s = act_quant(x.reshape(-1, shape[-1]))
    y = gemm(q, s, w, ws, x.dtype)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*shape[:-1], w.shape[0])


class Int8BlockLinear(BlockFP8Linear):
    """A linear layer with a block-scaled int8 weight (quantize) and an optional bias; part(o, n) as
    BlockFP8Linear's."""

    @classmethod
    def of(cls, lin):
        """The int8 version of a torch.nn.Linear (the bf16 weight is not kept)."""
        q, s = quantize(lin.weight.detach())
        return cls(q, s, None if lin.bias is None else lin.bias.detach())

    def part(self, o, n):
        return Int8BlockLinear(self.weight[o: o + n], self.weight_scale[o // BLOCK: (o + n) // BLOCK],
                               None if self.bias is None else self.bias[o: o + n])

    def forward(self, x):
        return linear(x, self.weight, self.weight_scale, self.bias)
