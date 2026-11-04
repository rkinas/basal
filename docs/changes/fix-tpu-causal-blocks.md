- **Less attention work on long TPU prompts** (`--mode tpu`, experimental): the blocked attention of prompts longer
  than the largest bucket scanned every key block for every query block, also the blocks after the query block's last
  token, which causality hides from all its queries (about half of the attention work of a long prompt). Each query
  block now stops at the key block of its last query; the result is bit-identical (JAX on the CPU, fp32 and bf16).
