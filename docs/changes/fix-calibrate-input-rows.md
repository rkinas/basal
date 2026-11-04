- **`basal-calibrate` input and rows are checked before the fit** (`basal-calibrate`): a labelled question with
  `ask_if` / `depends_on` could be answered null and crash the run after the whole inference; such conditions are now
  dropped (every labelled question is asked) and a missing answer is reported as a failed row. A `"group": null` or
  `"id": null` no longer puts every such line into one group "None" (one huge FIT or CERT cluster). An integer `gold` of
  a score question is always the level index, and is refused when a numeric criteria key names another level (it
  silently picked the key before). Rows with non-finite probabilities or a gold out of range, from the engine or a
  `--from-rows` file, are left out of the fit and reported; one NaN row made every temperature 0.25 before.
