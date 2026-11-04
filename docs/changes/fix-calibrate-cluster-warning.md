- **`basal-calibrate` warns when decisions share a group** (`basal-calibrate`): the Clopper-Pearson bound of a threshold
  counts every decision as independent, so when a type or task has more FIT decisions than groups (the claims of one
  document, the fields of one form, several questions about one state) and they share their errors, the certified
  threshold is optimistic. The report now names the decisions per group of each type and task (`fit_groups` in the
  report and in `task_gates`), and the docs state the assumption. No threshold changes.
