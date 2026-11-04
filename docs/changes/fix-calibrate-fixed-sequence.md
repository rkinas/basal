- **Certified thresholds come from a true fixed-sequence test** (`basal-calibrate`, decision types and tasks): the scan
  went on past failing confidence levels until one passed, so a type whose error was just above the target was
  certified far more often than 5% of the time (seeded simulation, 800 decisions at 5.1% error and a 5% target: 19.6%;
  at 1.05% and 1%: 9.5%). The levels are now tested from the most confident down and the test stops at the first
  level that fails (4.3% and 4.5% in the same simulation). The first level tested is set by the counts alone: enough
  decisions to meet the bound with one error (93 at 5%, 473 at 1%) when the type has at least twice that many, else
  without an error (59 / 299). A type with errors among its most confident decisions can now get no threshold where it
  got one before; the reason names the first level tested. The docs no longer say that a temperature keeps the order
  of choice and score confidences (it does for yes/no only). The bound is a binomial tail instead of a bisection: a
  task with 50,000 claims and no threshold fits in 0.2 s instead of 18 s. Shipped calibration files are unchanged; a
  refit can certify less.
