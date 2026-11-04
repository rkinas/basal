- **"N dni temu" helper facts are deduplicated and capped** (`facts: "auto"`): every occurrence added a fact line, so
  a state repeating "2 dni temu" two hundred times got two hundred identical lines in its prompt. Each N now gives one
  line, and at most three (`MAX_DURATIONS`, as for durations). A state with more than three such values, or repeated
  ones, gets a shorter prompt, which can move its probabilities.
