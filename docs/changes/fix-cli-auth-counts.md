- **`basal-run`, `basal-loadtest` and `basal-calibrate --url` work with a server started with `--api-keys`**
  (`basal-run`, `basal-loadtest`, `basal-calibrate`): they had no way to send a key; `--api-key KEY` (or
  `BASAL_API_KEY`) is sent as `Authorization: Bearer`, and a missing or refused key stops the tool with a message
  that says so (it wrote an error per line, ended in a traceback, or reported that no server answers).
