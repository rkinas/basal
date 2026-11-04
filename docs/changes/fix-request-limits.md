- **`basal-serve --max-body-mb` and `--max-branches`: a request's size is bounded before any work** (`basal-serve`):
  a request body of any size was read whole and every question and label rendered before a limit applied; a body
  over `--max-body-mb` (default 256 MiB: the largest media request, 8 items of 20 MB, is about 214 MB in base64) now
  answers 413 (`PAYLOAD_TOO_LARGE`) before it is read whole, also when it is sent without a length, and a request with
  more than `--max-branches` branches (default 4,096; a question counts 1, a `multi` question or a tool set one per
  label, as `usage.branches` counts them) answers 422 before it is rendered. Requests within the defaults are answered
  as before.
