"""MLX on the GPU of an M5 (NAX) runs float32 matmuls in TF32 unless MLX_ENABLE_TF32=0: about 1e-3 relative error,
above the 2e-4 tolerance with which the float32 tiny models of tests/test_mlx.py are compared to PyTorch. MLX reads the
variable at its first matmul, so it is set here, before any test module imports mlx, and over an inherited value: the
tolerances assume float32. Served bf16 models are not affected: their weights are bf16, which TF32 holds exactly, and
the readout accumulates in float32; the float32 compute of Qwen3.5 (basal-omni-pico) sets the variable at load
(MLXBackend) unless it is set."""
import os

os.environ["MLX_ENABLE_TF32"] = "0"
