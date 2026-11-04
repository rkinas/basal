- **`basal-router --ui on` no longer shows the credentials of `--route` URLs** (`basal-router`): the page's option
  list redacts URLs (no user, password, query or fragment), but a `--route SCRIPT=URL[,URL]` value did not parse as a
  URL, so `user:password@` of its first URL was shown on `/ui/state` (and its other URLs were cut off). Each URL of the
  value is now redacted on its own.
