- **Think mode answers every request it takes** (`basal-serve --think auto`, experimental): a request with a question
  that `ask_if` skipped (a `null` answer) failed with a 500, also with `"think": "off"`, and a request with a task
  (`classify`, `verify`, `compress`, `extract`) or `rank` question was refused with a 422 after the first readout; such
  questions now keep their answers (tasks and rank never think, as multi questions do not), and the vLLM log file the
  think server writes is closed when it stops.
