- **A calibration file of another model no longer certifies decisions** (`basal-serve --calibration`, all modes): a
  file fitted on another model, revision or adapter only warned at start-up and kept its thresholds, so `max_error`,
  tasks and answer sets acted on certificates of that model while `GET /v1/models` showed none. It is now kept for its
  temperatures only, as a file of other option orders: `max_error` defers, tasks report `automated: null`, answer sets
  hold every option and every answer is marked `"calibration": "unvalidated"` (`--strict-calibration` still refuses
  it). Two Hugging Face snapshots are compared by their whole repository ID, so `OrgA/basal-x` no longer matches
  `OrgB/basal-x`; a local copy still matches by its directory name. `basal-calibrate --from-rows` now applies the
  same orders and layout rule as basal-serve to the file it compares with (it ignored the layout).
