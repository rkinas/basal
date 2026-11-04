- **One rule for what a derived model keeps of its source** (`basal-finetune`, `basal-fuse-adapter`,
  `basal-interpolate`, `basal-export-gguf --systemone`): the tools that write a model with other weights now leave
  out the same files, those fitted to the source weights (NVFP4 activation scales, evidence head, early-exit heads,
  thinking folder) and those describing it (model card, manifest, checksums). `basal-fuse-adapter` no longer copies
  the base model's card and manifest; `basal-interpolate` no longer copies the manifest or half of `exit_heads/`
  (its thresholds without the heads, which were trained on the source's hidden states). Calibrations as before:
  dropped by fine-tuning and fusing, kept by interpolation as a starting point.
