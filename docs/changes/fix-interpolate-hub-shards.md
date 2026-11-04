- **`basal-interpolate` takes sharded checkpoints from the Hugging Face cache** (`basal-interpolate`): every shard of a
  downloaded repository is a link into the cache's `blobs/` folder, and the check that the index names no file outside
  the model folder followed those links, so any sharded checkpoint given as a repository id (or a snapshot folder) was
  refused as "outside the model folder". The index's paths are now checked as written (still no absolute path and no
  `..`).
