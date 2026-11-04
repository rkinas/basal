- **A thought answer keeps the request's contracts** (`basal-serve --think auto`, experimental): readout 2's answer
  dropped the `abstain` / `answer_set` a question asked for with `max_error` / `coverage` (a confident answer without
  its safety fields when readout 2 has no calibration); it now carries them under readout 2's calibration (without
  one: `abstain: true`, the set of every option). `ask_if` now reads the answers after thinking, so a question is
  skipped by the answer the request gets, not the one before; and a request with images or audio no longer thinks
  (readout 2 read its text only).
