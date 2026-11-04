"""Block-scaled int8 linears for the CUDA graph engine (--mode int8; basal-omni, Gemma 4): the layout of
basal/fp8block.py with int8 values. Weights in 128 x 128 blocks and activations in 1 x 128 tiles (a token's 128
consecutive input features), each block with its own fp32 scale (amax / 127, values rounded to the nearest integer);
the GEMM multiplies int8 by int8 into int32 for every 128-deep slice of the reduction (exact), applies both scales and
adds the slices in fp32. For GPUs without FP8 (Ampere, sm80 / sm86): their int8 tensor cores, half the memory of bf16.

  quantize(w)            weight [N, K] -> (int8 [N, K], scales [N / 128, K / 128])
  act_quant(x)           activations [M, K] -> (int8 [M, K], scales [M, K / 128])
  gemm(q, s, w, ws, dt)  the Triton GEMM: [M, N] in dt (CPU: the same maths dequantised)
  linear(x, w, ws, b)    x [..., K] -> [..., N]: act_quant + gemm; the bias
  Int8BlockLinear        a linear layer holding (w, ws, bias) (lean.linear picks the GEMM by the weight's dtype)

The blocks, the Triton GEMM and the layer are basal/blockquant.py's, shared with --mode fp8 (int8: the format FORMAT,
values rounded to the nearest integer within +-127, each 128-deep slice's product in int32).
"""
import torch

from .blockquant import BLOCK, TILES, BlockFormat, BlockLinear, fits, grid  # noqa: F401  (the shared core)
from .blockquant import reference as _reference  # noqa: F401  (tests)

INT8_MAX = 127.0
CONFIG = TILES  # the GEMM's tiles and launch, fixed for every shape as fp8block's; pipeline stages: _stages()


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


def _stages():
    """Pipeline stages from the GPU's shared memory (32 KB a stage: 4 on A100 / H100, 2 on sm86 / sm89)."""
    smem = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
    return max(2, min(4, (smem - 8 * 1024) // (32 * 1024)))


FORMAT = BlockFormat("int8_block_gemm", torch.int8, INT8_MAX, rounded=True, stages=_stages)
quantize, act_quant, gemm, build, linear = FORMAT.quantize, FORMAT.act_quant, FORMAT.gemm, FORMAT.build, FORMAT.linear


class Int8BlockLinear(BlockLinear):
    """A linear layer with a block-scaled int8 weight (quantize) and an optional bias (blockquant.BlockLinear)."""

    FORMAT = FORMAT
