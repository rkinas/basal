- **The Gated DeltaNet convolution's kernel is said and checked** (`--mode fast`, Qwen3.5 models): without
  flash-linear-attention's `causal_conv1d` the convolution silently ran on torch's `conv1d` (cuDNN), which rounds
  differently from the fla kernel `--mode fast` was checked with; start-up now warns, and the ready line and `/ui`
  name the kernel in use. `--batches compact`, which needs fla's kernel, is refused at start-up without it instead of
  failing at the first request.
