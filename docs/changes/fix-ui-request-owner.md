- **`--ui full` with `--api-keys` shows a request only to its own key** (`basal-serve --ui full`): any valid key could
  read every kept request at `GET /ui/request/<id>` (the ids count up), the text and answer of another client's
  request included; it now opens for the key that sent the request or an admin key, and another key gets 403. Without
  `--api-keys` the page stays open, as documented.
