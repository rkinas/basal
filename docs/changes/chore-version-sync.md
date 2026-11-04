- **The plugin and the TypeScript client carry the release version** (packaging): `.claude-plugin/plugin.json` and
  `clients/typescript/package.json` still said 2.5.0 in 2.5.3; they say 2.5.3, and a test fails when a release bumps
  `basal/__init__.py` and `pyproject.toml` without them.
