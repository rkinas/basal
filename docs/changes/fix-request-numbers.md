- **More bad numbers answer 422 before the run** (`basal-serve`): an `act` cost that is NaN, infinite or an integer
  beyond the largest float (also as text, `"NaN"`, `"1e400"`) gave a 500 after the engine's run, or answers with
  non-finite expected costs; a `multi` question with more than 256 labels under `objective` `iou` or `f1` gave a 500
  after the run. Both are now refused with a 422 before it; numbers given as text (`"9"`) are read as before.
