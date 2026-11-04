- **One launch rule for the block-scaled FP8 and int8 GEMMs** (`--mode fp8 --fp8-scaling block`, `--mode int8`): both
  kernels carried a tile-width override that could never apply, under a comment describing a rule they did not follow;
  one grid helper now serves both and refuses an output size that is not a multiple of 128 (whose last outputs would
  have stayed unwritten) instead of launching. Every basal-omni layer is such a multiple: answers are unchanged.
