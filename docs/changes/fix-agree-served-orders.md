- **`basal-agree` scores decisions in the orders the server answers with** (`basal-agree`, `basal-agree --identity`):
  the decision agreement and the accuracy averaged both option orders for every model, while `basal-serve` answers
  prompt-format-2 models in the given order alone by default. Format-2 models are now scored on that order; format-1
  models keep the average of both. The readout rows and the gate are unchanged.
