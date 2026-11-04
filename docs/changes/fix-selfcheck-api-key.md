- **`basal-selfcheck --api-key`** (`basal-selfcheck`): the check sent no API key, so against a basal-serve started
  with `--api-keys` it stopped at its first request with HTTP 401. `--api-key` (default `$BASAL_API_KEY`) is sent as
  `Authorization: Bearer` with every request, also with `--make-reference`.
