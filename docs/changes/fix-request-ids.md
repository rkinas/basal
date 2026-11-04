- **One request, one id and one count** (`basal-serve`, `basal-router`): a body that is not JSON is counted in
  `basal_requests_total` and `basal_requests_failed_total` (and the `--ui` traffic totals) on `/v1/systemone` and
  `/v1/basal` too, as it was on `/v1/decisions`; basal-router answers with `x-request-id` and passes the client's
  `X-Request-Id` (or one it makes) to every replica the request reaches, so the replicas' logs name the request the
  client knows.
