- **Decisions whose errors go together count as fewer independent ones in the bound** (`basal-calibrate`, decision
  types and tasks): the Clopper-Pearson bound counted every claim of a document, or every field of a form, as an
  independent decision, so a task with 70 FIT documents × 30 claims at a mean error just above the target was
  certified in 15-20% of runs instead of at most 5%. The correlation of the errors within groups (ICC) is now
  estimated on FIT and the accepted decisions and errors are divided by the design effect 1 + (m − 1) ICC (5-7% in
  the same simulations: an approximation, since the ICC is estimated from the same decisions).
  One decision per group, or errors that do not go together, change nothing; the report gives `icc` and
  `design_effect` per type and task, and the CERT figures and the CERT refutation use the same discount.
