- **`guard()` and `route()` fail closed on incomplete or malformed answers** (Python `basal.presets` and the
  TypeScript client): both started from pass and looked only at the answers present, so an empty object or a battery
  without its hazardous checks passed, and a NaN probability passed every threshold comparison. Every question of the
  battery must now be answered (`expect`: `"input"`, `"output"`, a list of names for a battery of your own, or the
  built-in battery of the answers by default); a missing, null or malformed answer (a probability that is not a finite
  number in [0, 1], a severity without level probabilities, with `max_error` an `abstain` that is not true / false) is a
  review, and `route()` sends such answers to a human. The shared test fixture holds the new cases for both languages.
  A call that passes a battery of its own without `expect` now gets a review.
