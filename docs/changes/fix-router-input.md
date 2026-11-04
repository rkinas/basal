- **`basal-router --route` and `--fallback` read every request as the replicas do** (`basal-router`): `--route`
  answered a 500 for a state that is a JSON number, list or object (it now reads the state's JSON text, as
  basal-serve shows it to the model, so such a state routes like the same text would), and `--fallback` sent a
  question of a request with `images` (the alias of `media`) to the text-only LLM, which never saw the images; such
  requests now keep the basal answer, as requests with `media` did.
