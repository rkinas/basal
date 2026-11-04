- **`basal-bench`: valid JSON without a memory reading, a message for too few items, the `--identity` status with
  `python -m`** (`basal-bench`): `--out` wrote `NaN` for `mem_gb` (GGUF, Ollama, llama.cpp and the CPU have no
  memory tracking), which is not JSON; it is now `null`. Fewer than 6 usable items (5 warm up the latency runs), or
  `--n` / `--lat-n` below 1, crashed in the timing code; they now exit with a message before an engine is loaded.
  `python -m basal.bench --identity` (as `make bench` runs it) exited 0 when the gate failed; it now exits 1, as
  `basal-bench --identity` does.
