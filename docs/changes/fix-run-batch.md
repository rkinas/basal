- **`basal-run` survives a bad line and streams its output** (`basal-run`): a line that is not JSON, a simple item
  without options or an answer that cannot be read gets its own `error` instead of ending the run without output;
  lines are read and answers written as they go (at most a few times `--concurrency` lines held), through
  `<output>.part`, which an interrupted run leaves with its answers so far; `--output` equal to `--input` (also
  through a link) is refused instead of replacing the requests.
