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

The blocks, the Triton GEMM and the layer are basal/blockquant.py's, shared with --mode int8 (FP8: the format
FORMAT, values saturated to +-448 and cast to e4m3).

  quantize(w)            weight [N, K] -> (FP8 [N, K], scales [N / 128, K / 128])
  act_quant(x)           activations [M, K] -> (FP8 [M, K], scales [M, K / 128])
  gemm(q, s, w, ws, dt)  the Triton GEMM: [M, N] in dt (CPU: the same maths dequantised)
  linear(x, w, ws, b)    x [..., K] -> [..., N]: flashinfer's kernel, or act_quant + gemm; the bias
  BlockFP8Linear         a linear layer holding (w, ws, bias); lean.linear runs its parameters in compiled layers
"""
import torch

from .blockquant import BLOCK, TILES, BlockFormat, BlockLinear, fits, grid  # noqa: F401  (the shared core)
from .blockquant import reference as _reference  # noqa: F401  (tests)

E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0
CONFIG = dict(TILES, num_stages=4)  # the GEMM's tiles and launch (blockquant.TILES), 4 pipeline stages
FORMAT = BlockFormat("fp8_block_gemm", E4M3, FP8_MAX, rounded=False, stages=CONFIG["num_stages"])
quantize, act_quant, gemm = FORMAT.quantize, FORMAT.act_quant, FORMAT.gemm


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
        if not ok:  # the Triton kernel, before any layer is compiled
            FORMAT.build()
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


class BlockFP8Linear(BlockLinear):
    """A linear layer with a block-scaled FP8 weight (quantize) and an optional bias (blockquant.BlockLinear: part(o, n)
    a view of output rows o..o + n); forward: linear (flashinfer's kernel on Hopper)."""

    FORMAT = FORMAT

    def forward(self, x):
        return linear(x, self.weight, self.weight_scale, self.bias)
