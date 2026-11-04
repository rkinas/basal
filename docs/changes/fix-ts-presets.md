- **TypeScript client: `langOf` as the Python presets' language guess** (`basal-client`): the Python fix for short
  Polish lines typed without diacritics (#97) was not in the TypeScript client, so `preset("nie dziala logowanie")`
  was worded in English from TypeScript and in Polish from Python. `langOf` now counts the same signs as
  `basal.presets.lang_of`; `scripts/ts_presets.py` writes `src/presets.ts` and the shared test fixture (371 language
  cases, every counted word among them) from `basal/presets.py`, and a Python test fails when they drift.
