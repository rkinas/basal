- **`basal-router` serves `POST /v1/decisions`** (`basal-router`): OpenAI SDK clients of the Decisions API got a 404
  from the router; the route now goes to a replica, routed by the request's `input` as a state is (with the client's
  key), and its response comes back as the replica gave it. The cascade, the blend and the fallback, which read basal
  answers, apply to `/v1/systemone` and `/v1/basal` only.
