- **TypeScript client: types of rank questions, router fallback answers and the usage block** (`basal-client`):
  `RankQuestion` / `RankAnswer` (a rank question did not compile); a choice answered by basal-router's `--fallback`
  is a `FallbackChoiceAnswer` (`answered_by: "fallback"`, `probabilities` and `confidence` may be `null`), so
  `AnswerOf["choice"]` and `choice()` return `ChoiceAnswer | FallbackChoiceAnswer`; `calibration` may be `"text"`
  (media); answers may carry the router's `level` and `model`; `usage` is always there (`full.usage.skipped`
  compiles), `usage.truncated` is the object the server sends (it was declared a boolean), with `usage.media`; the
  response may carry `fallback`. Code that reads `confidence` of a choice answer as a `number` must now narrow it
  first (`answered_by !== "fallback"`).
