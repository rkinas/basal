- **Cancelled requests cost the engine nothing** (`basal-serve`): a request cancelled while its work waited in the queue
  (an MCP call cancelled, a caller's timeout) left its queue entries, which the engine still ran; they are now dropped
  before the engine call (the main lane, `--scheduler` and `--long-lane`), and a request whose entry failed drops its
  other queued entries too. Work already in the engine finishes as before; answers are unchanged.
