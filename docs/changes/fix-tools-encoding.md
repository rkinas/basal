- **UTF-8 text in the tools, and the format-2 vectors replaced only when complete** (`basal-interpolate`, tools):
  the interpolation panel builders and run comparison, `basal-interpolate` (tokenizer, config and basal.json) and the
  figure script opened text without an encoding, which on Windows is the locale's (cp1252: Polish text cannot be
  written, a tokenizer vocabulary is misread); they open it as UTF-8, and a test checks every text open in these
  files. `docs/format2_vectors.py` wrote its output in place, so a run that failed left a truncated vectors file; it
  writes `<out>.part` and replaces the file at the end.
