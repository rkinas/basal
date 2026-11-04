- **`basal-attach` refuses an `--out` that overlaps the base model** (`basal-attach`): with `--force`, an `--out` equal
  to the base's own directory (also through a symlink) replaced the base's files with links to themselves; such an
  `--out`, one inside the base's directory or one containing it is now refused before anything is written.
