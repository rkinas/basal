- **`basal-selfcheck` sees bucket temperatures** (`basal-selfcheck`): a server was taken as uncalibrated when it
  reported no per-type temperatures, even with format-2 option-count bucket temperatures in use, so
  `--make-reference` could write tempered probabilities as the reference. It now checks every temperature, as
  `basal-calibrate --url` does (all 1.0, or none, is raw).
