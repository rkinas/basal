"""MLX on the GPU of an M5 (NAX) runs float32 matmuls in TF32 unless MLX_ENABLE_TF32=0: about 1e-3 relative error,
above the 2e-4 tolerance with which the float32 tiny models of tests/test_mlx.py are compared to PyTorch. MLX reads the
variable at its first matmul, so it is set here, before any test module imports mlx, and over an inherited value: the
tolerances assume float32. Served bf16 models are not affected: their weights are bf16, which TF32 holds exactly, and
the readout accumulates in float32; the float32 compute of Qwen3.5 (basal-omni-pico) sets the variable at load
(MLXBackend) unless it is set."""
import os

import pytest
import torch

os.environ["MLX_ENABLE_TF32"] = "0"

_SDPA = ("flash", "mem_efficient", "math", "cudnn")


@pytest.fixture(autouse=True)
def _restore_sdpa_backends():
    """EagerBackend turns SDPA backends off process-wide (cuDNN on CUDA; flash and memory-efficient too for Gemma 4),
    and the switches also choose the CPU kernel. A test that builds one would otherwise change the attention kernel,
    and so the bf16 rounding, of every later test in the same process: under pytest-xdist, which tests follow it
    depends on how the files are split between the workers."""
    before = {k: getattr(torch.backends.cuda, f"{k}_sdp_enabled")() for k in _SDPA}
    yield
    for k, on in before.items():
        getattr(torch.backends.cuda, f"enable_{k}_sdp")(on)
