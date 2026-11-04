- **`basal-router --blend` with a weight of 0 on `--replicas`** (`basal-router --blend`): when only one model with a
  weight above 0 answered, the answer was the first replica's even if its weight was 0 (`--blend-weights 0 1` gave the
  first model's choice under the second model's name); it is now that model's own answer, and a blend of several
  models leaves the zero-weight ones out of the answer as well as out of the mean.
