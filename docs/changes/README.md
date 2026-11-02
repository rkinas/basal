# Changelog fragments

Each pull request adds its entry for [WHATS_NEW.md](../WHATS_NEW.md) here as **`<PR number>.md`** instead of editing
WHATS_NEW.md itself, so open pull requests do not conflict on its first lines.

- One or more list items, written as they read in WHATS_NEW.md (bold summary, the mode in backticks, numbers, credit
  and the PR link), for example `docs/changes/44.md`:

  ```markdown
  - **Less host work in the CUDA cascade** (`--mode fast`, basal-2.0): ... Contributed by Miłosz Kucharski
    (@Butterski), [#44](https://github.com/rkinas/basal/pull/44).
  ```

- The PR number is known once the pull request is opened: open it, then add the file (or rename it) in a commit.
- A change too small for the release notes needs no fragment.

At release, `python scripts/changelog.py` prints the next section from the fragments, and
`python scripts/changelog.py 2.4.2` puts it at the top of WHATS_NEW.md as "What is new in basal 2.4.2" (entries in
PR order) and removes the fragments. The section can then be edited by hand like any other.
