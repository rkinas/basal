- **`basal-loadtest` exits on a question file without simple items** (`basal-loadtest`): a `--questions` file of full
  requests only, or an empty one, gave no request bodies and the tool spun one CPU at 100% for ever (also with
  `--determinism`); it now exits with a message before any request. `--n-conc` and `--determinism` below 0 are
  refused.
