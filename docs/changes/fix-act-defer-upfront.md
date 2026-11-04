- **`act` with `max_error` and no defer action is refused before the run** (`basal-serve`): API.md asks for a `defer`
  / `human` action (or `defer_action`) with `max_error`, but the request was refused (422, the whole response lost)
  only when a decision was below its threshold and answered 200 otherwise; it is now refused before the engine runs,
  whatever the confidence, and so is a `defer_action` that names no row of `costs` (it was returned as the action).
