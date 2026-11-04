- **`extract` reads a Polish thousands dot** (task type `extract`, all modes): "2.500 zł" came back as 2.5 while the
  helper facts read it as 2500, and the field could still be automated. In a Polish question a single dot before
  exactly three digits is now a thousands dot (an English question keeps 2.5), and a comma or dot followed by a space
  ends the number ("1, 2" is 1, not 1.2; "Rok 2024, 10 osób" is 2024, not 2024.1). `basal-calibrate` reads a number
  given as text in the gold of a field the same way.
