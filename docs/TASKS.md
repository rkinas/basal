# Tasks: classify, verify, compress, extract (experimental)

basal 2.1 adds four task types on `POST /v1/basal` (and `/v1/systemone`). They run on the basal models you already serve,
with no new model: each task question is expanded into the decision types (`choice`, `noul`, `multi`) over the
request's state, so it shares the state with the request's other questions (one forward pass, [SOAM](SOAM.md)) and uses
the model's calibration. The answer is assembled from those decisions.

*Experimental* (see [FEATURES.md](FEATURES.md)): the API may change in a minor release, and the models were not trained on
these task formats. Measured quality is in [Results](#results); check it on your own data.

| task | what you send | what you get |
|---|---|---|
| [`classify`](#classify) | labels with a rubric each | the label, a calibrated probability per label, whether it may be automated |
| [`verify`](#verify) | statements about the state | per statement: supported / contradicted / not stated, with probabilities |
| [`compress`](#compress) | a question (optional) | the state's sentences, lines or paragraphs needed for it, as shorter text |
| [`extract`](#extract) | a JSON schema | the values, copied exactly from the state with character offsets, or chosen from an enum |

Inputs may be structured: a state, instructions, a claim or a label description that is a JSON object or array is
written into the prompt as compact JSON (this has been true for every question type since 1.5).

A task question is expanded into decision questions named `<name>::<part>`, so in a request with task questions no
question's name may contain `::` (HTTP 422).

## classify

```json
{"state": "I was charged twice for my card payment yesterday.",
 "questions": {"intent": {"type": "classify", "none": true, "labels": [
   {"name": "card_payment_duplicate", "rubric": "a card payment was taken more than once"},
   {"name": "card_lost", "rubric": "the card is lost or stolen"},
   {"name": "refund_status", "rubric": "asks about a refund in progress"}]}}}
```
```json
{"type": "classify", "label": "card_payment_duplicate", "top_label": "card_payment_duplicate",
 "scores": {"card_payment_duplicate": 0.9996, "card_lost": 0.0002, "refund_status": 0.0000, "none": 0.0002},
 "confidence": 0.9996, "automated": {"0.01": null, "0.05": null}, "experimental": true}
```
(basal-omni-1.0-max, eager bf16, rounded to 4 places. basal-omni-1.1-max ships a `classify` threshold for 5%, so the
same answer there reads `"automated": {"0.01": null, "0.05": true}`.)

- `labels`: `{key: rubric}`, a list of keys, or a list of `{"name", "rubric"}`. Up to 255 labels (format-2 models).
- `none: true` adds the label `none` ("none of the labels above").
- `multi_label: true`: an independent probability per label and the selected set (`labels`), as `multi`
  (`threshold`, `min`, `max`, `objective` apply).
- `automated[max_error]`: the confidence is at or above a threshold certified **for this task** at that error rate, so
  the answer may be used without review; `null` when the calibration file has none (`"task_thresholds": {"classify":
  {"0.01": 0.97}}`; `basal-calibrate` fits them from your labelled task questions, see
  [CALIBRATION.md](CALIBRATION.md#tasks-classify-verify-extract)). The thresholds of the decision types are not used: they do
  not hold on new label sets (see [Results](#results)). `max_error` sets the rates (default 0.01 and 0.05).
- `classify` is `choice` (one label, probabilities sum to 1) or `multi` (independent labels) with a classification
  wording; use those types directly for full control.

## verify

```json
{"state": "Invoice FV/12/2026 ... Total due: 1 230,00 PLN, payable within 14 days.",
 "questions": {"check": {"type": "verify", "claims": {"total": "The total is 1,230 PLN.",
                                                     "terms": "Payment is due in 30 days."}}}}
```
```json
{"type": "verify", "all_supported": false, "not_supported": ["terms"],
 "claims": {"total": {"verdict": "supported", "probabilities": {"supported": 0.98, "contradicted": 0.01, "not_stated": 0.01},
                      "confidence": 0.98, "automated": {"0.01": null, "0.05": null}},
            "terms": {"verdict": "contradicted", "probabilities": {"supported": 0.02, "contradicted": 0.95, "not_stated": 0.03},
                      "confidence": 0.95, "automated": {"0.01": null, "0.05": null}}},
 "experimental": true}
```

- Each claim is a three-way decision over the shared state; `binary: true` asks only "supported or not".
- `automated[max_error]` per claim, with a threshold certified for `verify` (as for `classify`).
- Typical uses: check fields that another system (or `extract`) produced, check an LLM's answer or summary sentence by
  sentence against its source before it is used, check a document against stated facts.

## compress

```json
{"state": "<a long document>",
 "questions": {"ctx": {"type": "compress", "instructions": "What are the payment terms?", "unit": "sentence",
                       "keep_ratio": 0.3}}}
```
```json
{"type": "compress", "compressed": "<the kept sentences, in document order>", "kept": [3, 4, 17],
 "units": [{"start": 0, "end": 84, "probability": 0.03}, ...], "unit": "sentence",
 "original_chars": 9120, "compressed_chars": 2650, "ratio": 0.29, "experimental": true}
```

- Every unit (`sentence`, `line` or `paragraph`) gets one yes/no question, "does it hold information needed to answer
  the question?", all over one shared state. Without `instructions`, "does it hold important information?".
- `keep_ratio`: keep the most probable units up to that share of the characters; otherwise `threshold` (default 0.5).
  `min_keep` (default 1). `compressed` is a drop-in replacement for the state in a call to another model.
- At most 600 units per request; split longer documents or use a larger unit. The models were trained on states of up
  to about 6,000 tokens.
- Cost grows with the number of units: every unit is a question that repeats its text. A 5,000-word paper split into
  paragraphs took about 27 s on one H100 (eager, bf16); prefer `paragraph` or `line` units for long documents.

## extract

```json
{"state": "FAKTURA VAT nr FV/309/02/2026\nData wystawienia: 03.02.2026\n...",
 "questions": {"inv": {"type": "extract", "schema": {"type": "object", "properties": {
   "invoice_number": {"type": "string", "description": "the invoice number"},
   "issue_date": {"type": "string", "description": "the date of issue"},
   "gross_total": {"type": "number"},
   "payment_method": {"enum": ["transfer", "card", "cash"]},
   "seller": {"type": "object", "properties": {"name": {"type": "string"}, "tax_id": {"type": "string"}}}}}}}}}
```
```json
{"type": "extract",
 "data": {"invoice_number": "FV/309/02/2026", "issue_date": "03.02.2026", "gross_total": 1230.0,
          "payment_method": "transfer", "seller": {"name": "Med Energia Sp. z o.o.", "tax_id": "5251234567"}},
 "fields": {"invoice_number": {"value": "FV/309/02/2026", "text": "FV/309/02/2026", "start": 15, "end": 29,
                               "present_probability": 0.99, "line_probability": 0.99, "span_probability": 0.97,
                               "confidence": 0.96, "automated": {"0.01": null, "0.05": null}, "source": "span",
                               "calibration": "per decision"}, ...},
 "experimental": true}
```

- `enum` / `oneOf` (with `const`), `boolean` and arrays of enum values are decisions (`choice`, `multi`), with a "not
  stated" option and calibrated probabilities.
- `string`, `number` and `integer` fields are two decisions, with no text generation: **which line** of the state
  gives the value (a `choice` over the lines, plus "not stated"), then **which span of that line** is exactly the value
  (a `choice` over the line's candidate spans: the parts between separators, the part after a colon, and runs of
  words). A value is therefore always exact text of the document, with `start` / `end` character offsets. Numbers are
  parsed from the span (`1 234,56` and `1,234.56`). All fields of a request are answered in two packed rounds.
- `confidence` of a text field = P(line) × P(span); each factor is a calibrated decision.
- `automated[max_error]` per field (2.2), with a threshold certified for `extract` (fitted on the fields of your
  labelled extract questions, see [CALIBRATION.md](CALIBRATION.md#tasks-classify-verify-extract)); `null` without one
  and for arrays of enum values; `false` for a number that does not parse. `max_error` sets the rates, as for
  `classify`.
- Not supported yet: arrays of strings or objects (line items), text fields from images (enum and boolean fields work
  with media).

## Summaries with another model

basal does not write text. For summaries, put `compress` in front of a generative model (for example Bielik or Gemma on
vLLM or Ollama) and `verify` behind it:

1. `compress` the document with the reader's question (or none, for "the important parts"), `keep_ratio` 0.3–0.5.
2. Send `compressed` to the generative model with your summary prompt.
3. Split the summary into sentences and send them as `claims` of one `verify` question over the **original** document;
   sentences that are `contradicted` or `not_stated` are flagged for review or removed.

Steps 1 and 3 are one request each to basal-serve, whatever the number of sentences.

## Results

Measured with basal-omni-1.0-max (Gemma 4 12B), basal-serve eager bf16, on sets not used in training:

| task | set (licence) | n | result |
|---|---|---|---|
| classify | CLINC150 "plus", test, 151 labels incl. out-of-scope (CC BY 3.0) | 1,000 | accuracy 85.4% (in-scope 88.8%; out-of-scope found 69.3%) |
| classify | BANKING77, test, 77 labels (CC BY 4.0) | 1,000 | accuracy 74.6% |
| verify | VitaminC, real revisions, 3-way (MIT) | 1,000 | accuracy 80.3%, macro-F1 0.77 |
| verify | RAGTruth, response vs source, binary (MIT) | 1,000 | accuracy 80.3%, F1 of "not supported" 0.65 |
| compress | QASPER, paragraphs, keep 30% of characters (CC BY 4.0) | 248 | needed paragraphs kept: 90.1% (all of them for 84.3% of questions); at 10% / 20% / 50% of the text: 77.2% / 86.3% / 96.3% |
| extract | synthetic PL / EN invoices, 12 fields (ours) | 400 | fields correct: 90.1% (invoice number 96%, tax id 100%, amounts 96.5–99.5%, dates 86–94%, seller / buyer name 81% / 86%, IBAN 81%, currency 62%, payment method 100%) |

**basal-omni-1.1-max** (experimental, on Hugging Face from Sat 17 Oct, 09:00 CEST) is basal-omni-1.0-max with extra
training on the `verify`, `compress` and `extract` formats (the train splits of VitaminC and QASPER, and synthetic
documents; not CLINC150, BANKING77 or RAGTruth). Same sets, same engine:

| set | 1.0-max | 1.1-max |
|---|---|---|
| CLINC150 (accuracy) | 85.4% | 86.4% |
| BANKING77 (accuracy) | 74.6% | 74.0% |
| VitaminC (accuracy) | 80.3% | 84.3% |
| RAGTruth (accuracy) | 80.3% | 81.6% |
| QASPER (needed paragraphs kept at 30%) | 90.1% | 93.1% |
| invoices (fields correct) | 90.1% | 91.4% |
| documents in layouts never used in training (fields correct) | 84.8% | 89.3% |

An overlap check found 27 test items (12 VitaminC, 15 QASPER) that share a 13-word span with its training rows; without
them the gains are +4.2 and +2.7 points (paired 95% intervals +2.1 to +6.2 and +0.8 to +4.9), and the mean of the six
sets is still 85.1 (1.0-max: 83.5).

It is slightly behind 1.0-max on general decisions (Werdykt 0.821 vs 0.823; ContractNLI −13.7 points, where it leans
to "not stated"), so 1.0-max stays the default; its card has the full comparison.

The `automated` flag uses thresholds certified on the models' own decision tasks. On these new label sets they do
not hold (at a 1% target, the automated answers were wrong 2–9% of the time): so task answers report `automated` only with
thresholds fitted for the task (`task_thresholds` in the calibration file).
