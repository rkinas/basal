- **`basal-navigate build`: Polish k-means topics, and inputs whose files have the same name** (`basal-navigate`):
  `--clusters K --lang pl` failed in scikit-learn (the Polish stop words were passed as a tuple); two inputs with a
  file of the same name (`docs1/index.md`, `docs2/index.md`) gave the same page id and the build failed, and now
  their page ids get the input's folder in front (ids of a build without such a collision are unchanged); the tree
  written to `-o` inside an input folder is no longer read back as a document by the next build.
