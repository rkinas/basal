"""The core of the block-scaled linears of the CUDA graph engine: --mode fp8 --fp8-scaling block (basal/fp8block.py,
e4m3 values) and --mode int8 (basal/int8block.py, int8 values) are one format each of this module. Weights in 128 x 128
blocks and activations in 1 x 128 tiles (a token's 128 consecutive input features), each block with its own fp32 scale
(amax / the format's largest value); the GEMM takes the product of every 128-deep slice of the reduction (FP8: the
tensor cores' FP8 product; int8: exact in int32), applies both scales to it and adds the slices in fp32.

  BlockFormat(op, dtype, qmax, rounded, stages)   one format: quantize, act_quant, gemm, linear, build
  grid(M, N, config)                              the GEMM's launch grid (whole BN-wide output tiles)
  BlockLinear                                     a linear layer holding (w, block scales, bias) of a format
  fuse(mods)                                      one weight for the linears of one input, theirs views of it (the
                                                  merged GEMMs of the graph engines; a merged weight's 128 x 128
                                                  blocks are its parts' blocks when every part is a multiple of 128)

The GEMM is one Triton kernel for both formats (the product of a slice is the only line that differs), with fixed
tiles (128 x 128 x 128, no split of the reduction: a row's output does not depend on the other rows), registered as the
torch op basal::<op> (triton_op: Inductor keeps it inside a compiled layer, a CUDA graph captures it like any kernel).
On the CPU the GEMM is the same maths dequantised (tests).
"""
import torch

BLOCK = 128
# the GEMM's tiles and launch: rows x outputs per program (BN = BLOCK: every N here is a multiple of 128, fits), row
# tiles per group, warps; pipeline stages per format. Fixed for every shape (no autotuning: a CUDA graph replays what
# warm-up chose, and a row's output must not depend on how many rows there are)
TILES = dict(BM=128, BN=128, GROUP_M=8, num_warps=8)


def fits(lin):
    """True for a weight a block format can hold: both dimensions multiples of 128 (every Gemma 4 projection)."""
    n, k = lin.weight.shape
    return n % BLOCK == 0 and k % BLOCK == 0


def grid(M, N, config=TILES):
    """The launch grid of the block GEMM for M rows and N outputs: one program per BM x BN tile. N must be a multiple
    of BN: the kernel stores whole BN-wide tiles only, so the outputs past the last one would stay unwritten (fits()
    keeps every weight to multiples of 128)."""
    if N % config["BN"]:
        raise ValueError(f"block GEMM: {N} outputs, not a multiple of {config['BN']}")
    return ((M + config["BM"] - 1) // config["BM"] * (N // config["BN"]),)


def fuse(mods):
    """One weight for several linear layers that read the same input (no bias): their weights concatenated into a new
    tensor, and the layers' own weights become views of it (transformers' forward of the same model still runs, and no
    second copy is kept) -> (weight, output sizes). The Qwen3.5 and Gemma 4 graph engines' merged projections."""
    w = torch.cat([m.weight.detach() for m in mods])
    sizes = [m.weight.shape[0] for m in mods]
    for m, part in zip(mods, w.split(sizes)):
        m.weight = torch.nn.Parameter(part, requires_grad=False)
    return w, sizes


def reference(q, s, w, ws, dtype):
    """The GEMM's maths without the kernel (CPU, tests): both operands dequantised, fp32 products."""
    M, K = q.shape
    N = w.shape[0]
    x = (q.float().view(M, K // BLOCK, BLOCK) * s[..., None]).view(M, K)
    y = (w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK) * ws[:, None, :, None]).view(N, K)
    return (x @ y.t()).to(dtype)


def _kernel():
    """The block GEMM's Triton kernel (one for every format; INT: int8 operands, each slice's product exact in
    int32)."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(a, a_s, b, b_s, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr, INT: tl.constexpr):
        # a [M, K], a_s [M, K / BK]; b [N, K] (both K-major, as Hopper's FP8 MMA takes them), b_s [N / BK, K / BK];
        # c [M, N]. Programs in groups of GROUP_M row tiles (weight tiles reused from L2); rows past M read row
        # (r mod M) and are not stored.
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
            if INT:
                acc += tl.dot(x, y, out_dtype=tl.int32).to(tl.float32) * tl.load(asp)[:, None] * tl.load(bsp)[None, :]
            else:
                acc += tl.dot(x, y) * tl.load(asp)[:, None] * tl.load(bsp)[None, :]
            ap += BK
            bp += BK
            asp += 1
            bsp += 1
        cp = c + rm.to(tl.int64)[:, None] * N + rn[None, :]
        tl.store(cp, acc.to(c.dtype.element_ty), mask=(rm < M)[:, None])

    return kernel


class BlockFormat:
    """One block-scaled format: values of `dtype` within +-qmax (rounded to the nearest integer, ties to even, when
    `rounded`; else saturated and cast), its GEMM the torch op basal::<op> with `stages` pipeline stages (an int, or a
    function of the GPU called when the kernel is built)."""

    def __init__(self, op, dtype, qmax, rounded, stages):
        self.op, self.dtype, self.qmax, self.rounded, self.stages = op, dtype, qmax, rounded, stages
        self.kernel, self._act = [], []

        def act_quant_ops(x):
            M, K = x.shape
            b = x.float().view(M, K // BLOCK, BLOCK)
            s = b.abs().amax(-1).clamp(min=1e-12) / qmax
            v = b / s[..., None]
            return (v.round() if rounded else v).clamp(-qmax, qmax).to(dtype).view(M, K), s
        self.act_quant_ops = act_quant_ops

    @torch.no_grad()
    def quantize(self, w):
        """[N, K] (bf16 / fp32) -> (values [N, K], fp32 scales [N / 128, K / 128]): each 128 x 128 block divided by its
        amax / qmax."""
        N, K = w.shape
        b = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
        s = b.abs().amax((1, 3)).clamp(min=1e-12) / self.qmax
        v = b / s[:, None, :, None]
        q = (v.round() if self.rounded else v).clamp(-self.qmax, self.qmax).to(self.dtype).view(N, K)
        return q, s.contiguous()

    def act_quant(self, x):
        """[M, K] -> (values [M, K], fp32 scales [M, K / 128]): each token's 128-feature tile by its own amax. Inside a
        compiled forward Inductor fuses it with what precedes it; elsewhere on CUDA one compiled kernel (any shape)."""
        if torch.compiler.is_compiling() or not x.is_cuda:
            return self.act_quant_ops(x)
        if not self._act:
            self._act.append(torch.compile(self.act_quant_ops, dynamic=True))
        return self._act[0](x)

    def _build(self):
        from torch.library import triton_op, wrap_triton
        kernel, integer = _kernel(), not self.dtype.is_floating_point
        stages = self.stages() if callable(self.stages) else self.stages

        @triton_op(f"basal::{self.op}", mutates_args=())
        def op(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            M, K = q.shape
            N = w.shape[0]
            c = torch.empty((M, N), dtype=dtype, device=q.device)
            if M:
                wrap_triton(kernel)[grid(M, N)](q, s, w, ws, c, M, N, K, BK=BLOCK, INT=integer, num_stages=stages,
                                                **TILES)
            return c

        return op

    def build(self):
        """The Triton kernel as the op basal::<op>, built once (before any layer is compiled; CUDA only)."""
        if torch.cuda.is_available() and not self.kernel:
            self.kernel.append(self._build())

    def gemm(self, q, s, w, ws, dtype):
        """(q, s) [M, K] x (w, ws) [N, K]^T -> [M, N] in dtype."""
        if not q.is_cuda:
            return reference(q, s, w, ws, dtype)
        self.build()
        return getattr(torch.ops.basal, self.op)(q, s.contiguous(), w, ws, dtype)

    def linear(self, x, w, ws, bias=None):
        """x [..., K] through a block-scaled weight of this format -> [..., N] in x's dtype: act_quant + gemm, the
        bias."""
        shape = x.shape
        q, s = self.act_quant(x.reshape(-1, shape[-1]))
        y = self.gemm(q, s, w, ws, x.dtype)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.reshape(*shape[:-1], w.shape[0])


class BlockLinear(torch.nn.Module):
    """A linear layer with a block-scaled weight of a format (the subclass's FORMAT) and an optional bias; part(o, n)
    is a view of output rows o..o + n (multiples of 128): the projections of a merged weight stay modules of their own.
    forward: the subclass's module-level linear (lean.linear runs the same parameters in compiled layers)."""

    FORMAT = None

    def __init__(self, weight, scale, bias=None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.bias = bias
        self.out_features, self.in_features = weight.shape

    @classmethod
    def of(cls, lin):
        """The block-scaled version of a torch.nn.Linear (the bf16 weight is not kept)."""
        q, s = cls.FORMAT.quantize(lin.weight.detach())
        return cls(q, s, None if lin.bias is None else lin.bias.detach())

    def part(self, o, n):
        return type(self)(self.weight[o: o + n], self.weight_scale[o // BLOCK: (o + n) // BLOCK],
                          None if self.bias is None else self.bias[o: o + n])

    def forward(self, x):
        return self.FORMAT.linear(x, self.weight, self.weight_scale, self.bias)
