- **`basal-bench --identity --quant nvfp4` on basal-omni** (`basal-bench --identity`): the identity gate built the
  engine without the calibrated input amax, so basal-omni in NVFP4 was refused before the check started. It reads
  the model folder's `NVFP4_AMAX.json` as basal-serve does, or `--nvfp4-amax FILE`.
