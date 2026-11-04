- **`basal-router` keeps `ask_if` consistent through the cascade, the blend and the fallback** (`basal-router
  --escalate`, `--blend`, `--fallback`): a question asked again kept its `ask_if` / `depends_on`, so a replica refused
  the sub-request when the question it names was not asked again (a 422, and nothing was escalated or blended), and a
  replaced answer left its gated questions as the first replica had decided them (a child of a parent that changed
  from `a` to `b` stayed answered under `ask_if: {"parent": ["a"]}`). The questions asked again now go without their
  gates, and the router applies `ask_if` to the final answers: a question whose gate now fails is `null` and in
  `usage.skipped`, one whose gate now passes is asked of the first replica and goes through the same levels.
