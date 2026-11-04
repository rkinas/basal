"""Block-scaled FP8 linears for the CUDA graph engine (--mode fp8 --fp8-scaling block; basal-omni, Gemma 4): weights
in 128 x 128 blocks and activations in 1 x 128 tiles (a token's 128 consecutive input features), each block with its
own fp32 scale (amax / 448, e4m3); the GEMM applies both scales to every 128-deep slice of the reduction and adds the
slices in fp32 (the tensor cores' FP8 accumulator never sums more than 128 products). Finer than torchao's per-row
scales (one scale per output row of the weight and per token of the input), which failed the port gate on
basal-omni-1.1-max by 0.0001 (2-10 options at margin >= 0.2, 20 Oct).

The GEMM: on Hopper flashinfer's SM90 block-scaled kernel (DeepGEMM's, compiled on first use of a shape; it quantises
the bf16 activations itself) when flashinfer is installed: 1.0-1.45x cuBLAS bf16 on 1.1-max / ultra shapes (H100, 20
Oct). Elsewhere one Triton kernel with fixed tiles (128 x 128 x 128, no split of the reduction: a row's output does
not depend on the other rows), which is correct but not faster than bf16 (0.6-0.9x on the H100).

  quantize(w)            weight [N, K] -> (FP8 [N, K], scales [N / 128, K / 128])
  act_quant(x)           activations [M, K] -> (FP8 [M, K], scales [M, K / 128])
  gemm(q, s, w, ws, dt)  the Triton GEMM: [M, N] in dt (CPU: the same maths dequantised)
  linear(x, w, ws, b)    x [..., K] -> [..., N]: flashinfer's kernel, or act_quant + gemm; the bias
  BlockFP8Linear         a linear layer holding (w, ws, bias); lean.linear runs its parameters in compiled layers
"""
import torch

E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0
BLOCK = 128
# the GEMM's tiles and launch: rows x outputs per program (BN = BLOCK: every N here is a multiple of 128, fits), row
# tiles per group, warps, pipeline stages. Fixed for every shape (no autotuning: a CUDA graph replays what warm-up
# chose, and a row's output must not depend on how many rows there are)
CONFIG = dict(BM=128, BN=128, GROUP_M=8, num_warps=8, num_stages=4)
_KERNEL, _ACT = [], []


def fits(lin):
    """True for a weight this module can hold: both dimensions multiples of 128 (every Gemma 4 projection)."""
    n, k = lin.weight.shape
    return n % BLOCK == 0 and k % BLOCK == 0


@torch.no_grad()
def quantize(w):
    """[N, K] (bf16 / fp32) -> (FP8 [N, K], fp32 scales [N / 128, K / 128]): each 128 x 128 block divided by its amax /
    448, saturated."""
    N, K = w.shape
    b = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    s = b.abs().amax((1, 3)).clamp(min=1e-12) / FP8_MAX
    q = (b / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(E4M3).view(N, K)
    return q, s.contiguous()


def _act_quant_ops(x):
    M, K = x.shape
    b = x.float().view(M, K // BLOCK, BLOCK)
    s = b.abs().amax(-1).clamp(min=1e-12) / FP8_MAX
    return (b / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(E4M3).view(M, K), s


def act_quant(x):
    """[M, K] -> (FP8 [M, K], fp32 scales [M, K / 128]): each token's 128-feature tile by its own amax. Inside a compiled
    forward Inductor fuses it with what precedes it; elsewhere on CUDA one compiled kernel (any shape)."""
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


def grid(M, N, config=CONFIG):
    """The launch grid of the block GEMM (fp8block's and int8block's) for M rows and N outputs: one program per BM x BN
    tile. N must be a multiple of BN: the kernel stores whole BN-wide tiles only, so the outputs past the last one would
    stay unwritten (fits() keeps every weight to multiples of 128)."""
    if N % config["BN"]:
        raise ValueError(f"block GEMM: {N} outputs, not a multiple of {config['BN']}")
    return ((M + config["BM"] - 1) // config["BM"] * (N // config["BN"]),)


def _build():
    """The Triton kernel as the torch op basal::fp8_block_gemm (triton_op: Inductor keeps it inside the compiled
    layer, no graph break; a CUDA graph captures it like any kernel)."""
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def kernel(a, a_s, b, b_s, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
        # a [M, K] FP8, a_s [M, K / BK]; b [N, K] FP8 (both K-major, as Hopper's FP8 MMA takes them), b_s [N / BK,
        # K / BK]; c [M, N]. Programs in groups of GROUP_M row tiles (weight tiles reused from L2); rows past M read
        # row (r mod M) and are not stored.
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
            acc += tl.dot(x, y) * tl.load(asp)[:, None] * tl.load(bsp)[None, :]
            ap += BK
            bp += BK
            asp += 1
            bsp += 1
        cp = c + rm.to(tl.int64)[:, None] * N + rn[None, :]
        tl.store(cp, acc.to(c.dtype.element_ty), mask=(rm < M)[:, None])

    @triton_op("basal::fp8_block_gemm", mutates_args=())
    def op(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        M, K = q.shape
        N = w.shape[0]
        c = torch.empty((M, N), dtype=dtype, device=q.device)
        if M:
            wrap_triton(kernel)[grid(M, N)](q, s, w, ws, c, M, N, K, BK=BLOCK, **CONFIG)
        return c

    return op


def gemm(q, s, w, ws, dtype):
    """(q, s) [M, K] x (w, ws) [N, K]^T -> [M, N] in dtype."""
    if not q.is_cuda:
        return _reference(q, s, w, ws, dtype)
    if not _KERNEL:
        _KERNEL.append(_build())
    return torch.ops.basal.fp8_block_gemm(q, s.contiguous(), w, ws, dtype)


_SM90 = []


def sm90():
    """True where flashinfer's SM90 block-scaled GEMM runs: a Hopper GPU and flashinfer installed. Its kernels are
    compiled at first use with nvcc: when nvcc is not on PATH, CUDA_HOME's or /usr/local/cuda's is added."""
    if not _SM90:
        ok = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
        if ok:
            try:
                import flashinfer.gemm  # noqa: F401
            except ImportError:
                ok = False
        if ok:
            import os
            import shutil
            if shutil.which("nvcc") is None:
                for d in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
                    if d and os.path.exists(os.path.join(d, "bin", "nvcc")):
                        os.environ["PATH"] = os.path.join(d, "bin") + os.pathsep + os.environ.get("PATH", "")
                        break
        _SM90.append(ok)
        if not ok and torch.cuda.is_available() and not _KERNEL:  # the Triton kernel, before any layer is compiled
            _KERNEL.append(_build())
    return _SM90[0]


@torch.library.custom_op("basal::fp8_block_linear_sm90", mutates_args=())
def _linear_sm90(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    from flashinfer.gemm import fp8_blockscale_gemm_sm90
    if not x.shape[0]:
        return x.new_empty(0, w.shape[0])
    return fp8_blockscale_gemm_sm90(x, w, None, ws, out_dtype=x.dtype)


@_linear_sm90.register_fake
def _(x, w, ws):
    return x.new_empty(x.shape[0], w.shape[0])


def linear(x, w, ws, bias=None):
    """x [..., K] (bf16) through a block-scaled FP8 weight -> [..., N] in x's dtype."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x.is_cuda and x.dtype == torch.bfloat16 and sm90():
        y = torch.ops.basal.fp8_block_linear_sm90(x2.contiguous(), w, ws)
    else:
        q, s = act_quant(x2)
        y = gemm(q, s, w, ws, x.dtype)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*shape[:-1], w.shape[0])


class BlockFP8Linear(torch.nn.Module):
    """A linear layer with a block-scaled FP8 weight (quantize) and an optional bias; part(o, n) is a view of output
    rows o..o + n (multiples of 128): the projections of a merged weight stay modules of their own."""

    def __init__(self, weight, scale, bias=None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.bias = bias
        self.out_features, self.in_features = weight.shape

    @classmethod
    def of(cls, lin):
        """The FP8 version of a torch.nn.Linear (the bf16 weight is not kept)."""
        q, s = quantize(lin.weight.detach())
        return cls(q, s, None if lin.bias is None else lin.bias.detach())

    def part(self, o, n):
        return BlockFP8Linear(self.weight[o: o + n], self.weight_scale[o // BLOCK: (o + n) // BLOCK],
                              None if self.bias is None else self.bias[o: o + n])

    def forward(self, x):
        return linear(x, self.weight, self.weight_scale, self.bias)
