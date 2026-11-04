- **`basal-router`'s `/health` and `/v1/models` follow the pool of ordinary requests** (`basal-router`): `/health`
  answered 200 while only an `--escalate` or `--blend` replica was ready, though every ordinary request then got a 503,
  and `/v1/models` gave the first ready replica in URL order, which could be the escalation model; both now read the
  replicas of `--replicas` / `--gpus` only, and `/health` also gives the ready replicas of every pool (`"pools"`).
