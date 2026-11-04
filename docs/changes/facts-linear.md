- **`"facts": "auto"` reads a long state in linear time** (`basal-serve`): the fact extraction rescanned the text
  per match (a long run of space-separated digit groups, many zł amounts, many percentages without VAT, many exchange
  rates), so a 40 kB state could take 14 s and a 1 MB state minutes, with the server answering nothing else meanwhile;
  it now takes about 2 µs per character at worst, and a state over 1,000,000 characters with `"facts": "auto"`
  answers 422 (send `"facts": "off"` or a shorter state). The facts themselves are unchanged.
