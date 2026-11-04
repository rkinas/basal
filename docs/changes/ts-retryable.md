- **TypeScript client: 429, 502 and 504 are retryable** (`basal-client`): `retryable` is also true for HTTP 429, 502
  and 504, which a proxy or a router in front of `basal-serve` answers.
