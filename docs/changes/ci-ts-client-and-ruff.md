- **CI runs the TypeScript client's tests and one Ruff over the whole repository** (CI): a job builds and tests
  `clients/typescript` (`npm ci`, `npm test` against `basal-serve` on the stub engine), so an API or preset change
  that breaks the client no longer merges green. pre-commit and CI use the same Ruff (0.15.16) with the rules of
  `pyproject.toml`, over every Python file and notebook (pre-commit checked 4 rule groups with another version, CI
  `basal` and `tests` only).
