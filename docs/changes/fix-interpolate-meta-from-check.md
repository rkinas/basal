- **`basal-interpolate --meta-from` another checkpoint is checked** (`basal-interpolate`): the merged model takes its
  config.json and tokenizer.json from `--meta-from`, but only the two interpolated checkpoints were compared, so a third
  checkpoint with another vocabulary or architecture gave a merged model served with the wrong tokenizer or config. It
  is now refused before anything is written unless its architecture and vocabulary are the checkpoints'.
