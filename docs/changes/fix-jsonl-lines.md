- **JSONL lines with U+2028, U+2029 or U+0085 in a string are read whole** (`basal-run`, `basal-bench`,
  `basal-calibrate`, `basal-agree`, `basal-finetune`, `basal-viz`): the readers split lines with `str.splitlines`, which
  also splits at these separators, so a valid line written with `ensure_ascii=False` (text pasted from a word processor
  or a PDF) was cut in two; they now split at `\n` only, as `scripts/check_jsonl.py` does.
