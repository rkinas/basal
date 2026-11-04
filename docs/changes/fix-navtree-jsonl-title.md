- **`basal-navigate build` checks the title of a JSONL document** (`basal-navigate`): a `"title"` that was not a
  string (a number, a list) ended the build with a `TypeError` from the parser; the line is now refused with its file
  and line number, as a document without text is.
