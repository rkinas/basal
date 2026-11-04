- **`basal-finetune` keeps a shared group on one side of the split** (`basal-finetune`): a line went with the group of
  its first question only, so a per-question `group` shared by lines with other first groups (one customer asked about
  in several records) could land in both the training part and the held-out part, and the calibration fitted on
  `heldout.jsonl` then saw decisions the model trained on. Lines that share a group on any question now go to the same
  part; a line that shares no group keeps the side it had.
