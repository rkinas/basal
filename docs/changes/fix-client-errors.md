- **The Python client tells a key error and a starting server from the rest** (`basal.client`): a 401 / 403 was a
  `ServerError`, the 503 of a server still loading its model an `OutOfMemory`, and an OpenAI-style error body (`--error-format
  openai`, `/v1/decisions`) came back as the text of a dict. They are now `AuthError` (a `ServerError`) and
  `NotReady` (an `OutOfMemory`, retried the same way), so existing `except` clauses still catch them, and the message
  is the error's own text (`"<type>: <message>"`).
