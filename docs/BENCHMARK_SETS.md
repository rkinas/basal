# Preparing a benchmark set

basal-ui's **Benchmark** (Models → Loaded) and `basal-loadtest --questions` read simple JSONL items: one JSON object per
line, each a state, one question and its options.

```json
{"id": "inv-01", "state": "Faktura z 3 marca, termin płatności 14 dni, zapłacono 20 marca.", "question": "Czy zapłacono w terminie?", "options": ["Tak", "Nie"], "gold": 1}
{"id": "ticket-07", "state": "Customer: the parcel has not arrived for two weeks, I want my money back.", "question": "What does the customer want?", "options": ["A refund", "Delivery status", "Something else"], "gold": 0}
{"state": "Umowa najmu na czas nieokreślony, wypowiedzenie złożone 31 stycznia.", "question": "Jaki okres wypowiedzenia obowiązuje?", "options": ["Miesięczny", "Trzymiesięczny", "Nie da się ustalić"]}
```

| Field | Required | Meaning |
|---|---|---|
| `state` | yes | the text (or a JSON object) the decision is based on, with every fact it needs |
| `question` | yes | one question, phrased for the state |
| `options` | yes | 2 or more answers, as text; each is sent as a `choice` option (`option_1`, `option_2`, … with hidden keys) |
| `gold` | no | the 0-based index of the correct option; items without it count for speed only |
| `id` | no | your own label; ignored by the benchmark |

- One file may mix Polish and English items; ask each question in the language of its state.
- At least 10 items (basal-ui refuses a smaller set); a few hundred give stable accuracy. A small set is reused to fill
  the 155 requests of a run, but each item is scored once.
- Scoring: an item is correct when the option with the highest probability is the `gold` one. Accuracy is reported
  overall and per file (by file name).
- Lines that are not JSON stop the run with the file and line number; objects without `state`, `question` or
  `options` are skipped. Full-request lines (with `questions`) are skipped as well.

**Where to put it.** Anywhere readable: in basal-ui choose **Custom path…** in the set selector and enter the path of a
`.jsonl` file, or of a directory (every `*.jsonl` in it is read, in name order). `~` is expanded. The bundled set is
`basal/examples/{questions,choice,noul,score}.jsonl` (44 items, all with `gold`), a good template.

**From the command line** against a `basal-serve` (`basal-loadtest` sends no `model`, so basal-ui's API server, which
requires one, answers 400; run the benchmark from the UI there):

```bash
basal-loadtest --url http://127.0.0.1:8000/v1/systemone --questions my-set.jsonl
```

`basal-loadtest` reports latency and throughput; the accuracy on `gold` is reported by basal-ui's benchmark (and by
`basal-bench` offline).
