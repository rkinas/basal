# External OCR-generated criteria and image context

Basal can add OCR matches to the candidates of an ordinary **choice** or **multi**
decision. The OCR engine runs separately; Basal uses its HTTP response to
construct option lists, then lets the decision model evaluate those options.
The model may receive the original image, recognized text, or both. Requests
without `generated_criteria` or non-default `media.context` are unchanged.

## Start the OCR service and Basal

Install PaddleOCR and the selected PaddlePaddle or Transformers inference
backend in the OCR environment, following the
[PaddleOCR serving guide](https://www.paddleocr.ai/main/en/version3.x/inference_deployment/serving/serving.html).
Start the general PaddleX OCR service:

```sh
paddlex --install serving
paddlex --serve --pipeline OCR   # default HTTP port: 8080
```

Start Basal in another process:

```sh
basal-serve \
  --model Remek/basal-omni-1.0-mini \
  --mode eager --media on \
  --ocr-api paddlex --ocr-url http://127.0.0.1:8080 \
  --ocr-timeout 30 \
  --host 127.0.0.1 --port 8000
```

`--host` and `--port` configure **Basal**, not the external OCR service.
If they run in different Docker containers, use a reachable OCR container
hostname rather than `127.0.0.1`. `--ocr-api` is optional and disabled
by default. OCR runs only if a question references the image through
`generated_criteria`, or the image requests `context: "text"` or `"both"`.
The first supported adapter is `paddlex`.

## Request

```json
{
  "state": "Extract values from this invoice.",
  "lang": "en",
  "media": [{
    "type": "image",
    "name": "invoice.png",
    "ocr": "on",
    "data": "<base64-encoded-PNG-JPEG-or-WebP>"
  }],
  "questions": {
    "amount": {
      "type": "choice",
      "instructions": "What is the invoice total including tax?",
      "criteria": ["1250,00"],
      "generated_criteria": {
        "preprocessor": "ocr",
        "source": "invoice.png",
        "filter": "\\d+[,.]\\d{2}"
      },
      "__other__": "None of these amounts"
    },
    "dates": {
      "type": "multi",
      "instructions": "Which dates refer to payment deadlines?",
      "generated_criteria": {
        "preprocessor": "ocr",
        "source": "invoice.png",
        "filter": "\\d{2}[./-]\\d{2}[./-]\\d{4}"
      }
    }
  }
}
```

POST the JSON to Basal's `/v1/systemone` or `/v1/basal` endpoint.
With the default `context: "image"`, the same original image is passed to
the multimodal model (with `--media on`) for evaluating the candidates.
OCR is executed only once per referenced image per request.

## Per-image OCR context

Use the optional field **media[].context** to choose what the decision model receives from a named image:

| Value | Model receives | Calls PaddleX without generated_criteria? |
| --- | --- | --- |
| `"image"` (default) | Original image only | No |
| `"text"` | OCR text appended to `state`; original image removed | Yes |
| `"both"` | Both OCR text in `state` and original image | Yes |
| `"none"` | Neither image nor OCR text; only generated option candidates (when requested) | No |

For `"text"` and `"both"`, the image **must** have `ocr: "on"`, a
unique `name`, and a configured external OCR service (`--ocr-api paddlex`).
OCR is executed once per image across both context injection and
`generated_criteria` extraction, regardless of question count. The
default `"image"` mode never adds OCR text to `state`; it runs OCR only
when a question requests generated candidates.

For example, use this in a **CPU-only text-model** request:

```json
{
  "state": "Przeanalizuj fakturę.",
  "lang": "pl",
  "media": [{
    "type": "image",
    "name": "faktura.png",
    "data": "<base64>",
    "ocr": "on",
    "context": "text"
  }],
  "questions": {
    "kwota": {
      "type": "choice",
      "instructions": "Jaka jest kwota brutto faktury?",
      "criteria": ["999,00"],
      "generated_criteria": {
        "preprocessor": "ocr",
        "source": "faktura.png",
        "filter": "\\d+[,.]\\d{2}"
      }
    }
  }
}
```

If PaddleX recognizes `FAKTURA VAT` and `Kwota brutto: 1250,00`, the
decision model receives the following **text state**:

```text
Przeanalizuj fakturę.

[OCR: "faktura.png"]
FAKTURA VAT
Kwota brutto: 1250,00
[/OCR]
```

The chosen candidate list still comes from static `criteria` and the
`generated_criteria` regex. OCR text injection is **independent** of
candidate extraction: with `context: "text"` or `"both"`, the service
recognizes the image even if the question has only static `criteria`.
However, merely setting `ocr: "on"` with default `context: "image"`
does not call OCR unless `generated_criteria` uses that image.

For `"text"` and `"none"`, Basal removes that image from the request sent to the
model. If no other media remains, it removes the `media` field entirely
and uses the ordinary text-inference path (suitable for CPU-only Basal).
Other images and audio are preserved in their original order, so a
request with remaining media still needs a model that supports them.
A non-string `state` is converted to JSON text before appending OCR.
Several OCR blocks are appended in the original `media` order.
`"both"` keeps the original image and therefore requires a multimodal
model with `--media on`.

Invalid `context` values, `context` on audio, and `"text"`/`"both"`
without `ocr: "on"` return HTTP 422. An unavailable PaddleX service
returns HTTP 502 **only if OCR is actually called**, including when OCR
is called solely for text context. Requests that do not need OCR are
unaffected by an OCR outage.


### Candidate-only OCR: `context: "none"`

Use `"none"` when you want the image to be processed **only by the external OCR
preprocessor**. The original image and the full OCR text are **not** forwarded
to the Basal model. If the question has `generated_criteria`, PaddleX still
reads the image, the regex extracts matching option values, and the decision
model sees only the original `state`, the question instructions and the
resulting option list (plus `__other__` for dynamic `choice`).

```json
{
  "state": "Wybierz właściwą kwotę brutto spośród znalezionych wartości.",
  "media": [{
    "type": "image",
    "name": "faktura.png",
    "data": "<base64>",
    "ocr": "on",
    "context": "none"
  }],
  "questions": {
    "kwota": {
      "type": "choice",
      "instructions": "Jaka kwota odpowiada kwocie brutto?",
      "generated_criteria": {
        "preprocessor": "ocr",
        "source": "faktura.png",
        "filter": "[0-9]+[,.][0-9]{2}"
      }
    }
  }
}
```

Omitting static `criteria` makes the candidate list come exclusively from
OCR. The **decision model is still invoked** if a generated candidate exists;
`context: "none"` means *no document context*, not *bypass Basal inference*.
If the OCR matches nothing and there are no static options, the resolver skips
inference for that question and returns the existing empty-answer shape.

With `context: "none"` **without** `generated_criteria`, the image is
simply omitted from model input without calling OCR. In this case `ocr: "on"`
and a configured OCR service are not required. If `generated_criteria`
references the image, `ocr: "on"`, a unique image name, and a configured
OCR service remain mandatory.

This mode is compatible with CPU-only, text-capable Basal models. Since the
model cannot see OCR text, surrounding labels or image layout, decisions
between ambiguous matches may have lower accuracy than with `"text"` or
`"both"`.

## Reserved option identifier: `__other__`

The special OCR choice option uses one reserved identifier everywhere:

- Request: `"__other__": "None of the above"` sets its label.
- Probability: `probabilities.__other__` is the probability of this option.
- Selection: `choice: null`, `__other__: true`, `source: ["__other__"]`.

Basal shows the label to the model. Static `criteria` cannot contain
`__other__` as an ordinary option. This is a **breaking change** for
OCR requests: the former `other` request field now returns HTTP 422,
and clients must use the new probability and selection keys.

## Candidate rules

- `criteria` (array or object) supplies static candidates; its original
  keys are preserved.
- `generated_criteria` supports `preprocessor: "ocr"`, `source` matching
  a named `media` image with `ocr: "on"`, and `filter` as a regular
  expression. Full, nonempty matches (group 0) from each recognized text
  fragment are candidates, in first-occurrence OCR order.
- Static candidates come first. Duplicate OCR values merge into the
  static candidate if they match its key or description; remaining
  duplicates are removed. The answer can carry `source: ["static", "ocr"]`.
- Dynamic **choice** automatically adds `__other__` as an extra option.
  A custom `__other__` string changes its visible label; the default is
  "None of the above" / "Żadna z powyższych". If selected, Basal returns
  `choice: null`, `__other__: true` and probabilities containing `__other__`.
- A choice without `generated_criteria` keeps its original options:
  `__other__` is added only when explicitly provided.
- Dynamic **multi** does not add `__other__`: `selected: []` means no values
  were selected. `multi` uses the same merge and deduplication.
- If there is **one candidate**, the decision model always evaluates it
  against `__other__` (choice), or as a multi label. If there are **zero
  combined candidates**, inference for that question is skipped:
  `choice: null` or `selected: []`, with `decision_source:
  "ocr_preprocessor"`, `reason: "no_candidates"`, and no fabricated
  probability or confidence. Other questions still run.
- Zero candidates in a request that uses conditional `depends_on` or
  `ask_if` questions currently returns HTTP 422 (not a guessed answer).

Output `usage.ocr` reports OCR image calls (from context and/or generated criteria)
and raw regex match counts;
`usage.preprocessor_only` counts questions answered without the decision
model. An unreachable service, HTTP error, or malformed PaddleX response
returns HTTP **502** (not a null match). Invalid inputs and unsupported
question types return HTTP **422**.

Only raster **images** are supported by this integration. PDF and audio
are not routed to OCR. Existing Base64 image size and decompression limits
apply. The choice candidates must fit the served model's option limit
(including `__other__`); too many unique matches return HTTP 422.
Dynamic multi accepts at most 255 unique labels.
Regexes have bounded execution time; avoid ambiguous broad patterns.

PaddleX request: `POST <ocr-url>/ocr`, JSON
`{"file": "<base64>", "fileType": 1, "visualize": false}`.
The adapter reads `result.ocrResults[0].prunedResult.rec_texts`.
See the [official PaddleOCR API reference](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html).
