# basal-serve as the local backend of your decision code

Code that today sends "pick one of these options" or "is this true?" to a hosted LLM and parses the text it gets back can
send the same decision to a local basal server instead. You change one transport function; the rest of your code keeps
its own types.

```bash
basal-serve --model <model> --port 8000          # on your own GPU, Mac or CPU: docs/INSTALL.md
```

## The transport

A decision is one `POST /v1/systemone` with `{"state": ..., "questions": {...}}`; the response has an answer per question
(`choice`, `noul`, `score`), each with `probabilities` and a `confidence`. `basal/examples/local_backend.py` is a
complete asker function: it takes a state and a question, returns the option and its confidence, and needs only `httpx`.

```python
from local_backend import ask_choice  # copy the file next to your code

option, confidence = ask_choice("http://127.0.0.1:8000", state, "What does the customer want?",
                                {"refund": "money back", "replace": "a new item", "info": "information only"})
```

Three things change compared with a text-generating backend:

- **No parsing.** The answer is one of the keys you gave; there is no free text that fails to parse.
- **A real confidence.** `confidence` is calibrated ([CALIBRATION.md](CALIBRATION.md)). Send low-confidence decisions to
  a person, or to a larger model, instead of acting on them: `ask_choice(..., max_error=0.05)` returns `None` for the
  option when the server marks the answer `abstain`.
- **Same request for every client.** A Python, TypeScript or plain-curl client differs only in the HTTP call
  ([clients/typescript](../clients/typescript), `basal.client.Basal`).

## What to check before you switch

1. Put 200 or more of your own decisions, with the correct answer, through `basal-calibrate`; compare its accuracy with
   what your current backend gets on the same decisions.
2. Look at the abstain rate at the error you can accept. If it is high, adapt the model ([FINETUNE.md](FINETUNE.md)).
3. Keep the old backend as the fallback for what abstains.

Not tested here: any particular third-party project as the caller. What is tested is the request and response shape
(`tests/test_local_backend.py`, against a stub server).
