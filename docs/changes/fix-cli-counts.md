- **`--concurrency 0` is refused** (`basal-run`, `basal-calibrate`): it made a semaphore that admits nothing, so the
  run waited for ever.
