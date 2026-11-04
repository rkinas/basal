- **fla kernels outside the recorded table autotune again** (`--mode fast`, Qwen3.5 models): with the recorded fla
  configs in use, a kernel the recording never launched was pinned to its first candidate config without timing, a
  config never measured on that GPU (it may not fit there); only the recorded kernels are pinned now, the others pick
  their config as fla does without a table. The recorded kernels keep their configs.
