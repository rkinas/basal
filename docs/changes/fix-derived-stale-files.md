- **Fused and interpolated models leave out what was fitted to the source weights** (`basal-fuse-adapter`,
  `basal-interpolate`): a fused checkpoint kept the base model's evidence head, trained on the base model's hidden
  states (`--adapter` already answered evidence requests with HTTP 422 for that reason), and an interpolated one kept
  `--meta-from`'s `NVFP4_AMAX.json`, whose activation scales `--mode nvfp4` then read for other weights (values above
  a stale scale are clipped). Neither is copied now: evidence and `--mode nvfp4` need the model's own files.
