- **`basal-router --replica-timeout S`** (`basal-router`): a replica that accepted a request and never answered held
  the client and the router's in-flight count forever; with `--replica-timeout` such a call ends after S seconds with a
  504 (an escalation level or a blend model that times out leaves the previous answers, as one that is down). Off by
  default, as before; connecting to a replica now fails after 10 s (a 504) instead of waiting for the system's TCP
  timeout.
