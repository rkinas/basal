- **TypeScript client: errors while the body is read are `BasalError`s; Node >= 18.17** (`basal-client`): a body
  that stalled after the headers ended in a raw `DOMException` instead of a `timeout`, and a body that broke off in a
  generic `Error` instead of a `network` error; the read is now inside the same error boundary (the caller's own
  abort still passes through). A `timeout` that is not a number of seconds > 0 is refused when the client is made,
  and a fraction of a millisecond no longer fails in `AbortSignal.timeout`. `engines` says Node >= 18.17, the first
  18.x with `AbortSignal.any`.
