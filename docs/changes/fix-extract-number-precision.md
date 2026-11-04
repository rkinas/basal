- **`extract` keeps large integers exact** (task type `extract`, all modes): every number went through a float, so
  9007199254740993 came back as 9007199254740992. Integers are now parsed exactly; a number beyond a float's range is
  still "not a number" (the field reports the error and is never automated). `basal-calibrate` compares integer fields
  exactly.
