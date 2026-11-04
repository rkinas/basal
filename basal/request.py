"""What a request body is checked for before any work, shared by basal-serve's routes (POST /v1/systemone, /v1/basal,
/v1/decisions and the MCP tools); no torch, so basal-router can use it too.

A body is JSON that a client may have built by hand: it can nest deeper than the code that reads it recurses, carry an
escape such as "\\udead" (a lone surrogate: not Unicode text, which the tokenizer and the UTF-8 response refuse) or,
in a question, a NaN or an infinity (Python's json reads them, and an answer keyed or computed by one cannot be
written). Each is the request's fault, answered 422 (400 on /v1/decisions) before anything is rendered or run."""
import math

from .prompt import RequestError

MAX_DEPTH = 256  # objects and arrays nested in a request body (JSON states nest a few levels; Python's parser and
#                  the recursive readers after it stop near 1,000, with a 500)


def _text_ok(s, where):
    if not s.isascii():
        try:
            s.encode("utf-8")
        except UnicodeEncodeError as e:
            bad = s[e.start: e.end].encode("utf-8", "backslashreplace").decode("ascii")
            raise RequestError(f"{where}: {bad} is a lone surrogate (half of a UTF-16 pair), not text") from None


def check_json(body, max_depth=MAX_DEPTH, finite=("questions",)):
    """A parsed request body -> RequestError when its objects / arrays nest deeper than max_depth, a string (a key or a
    value) holds a lone surrogate, or a number under one of the top-level fields `finite` is NaN or infinite. Iterative:
    a deep body cannot exhaust the stack here."""
    stack = [(body, 1, "the request body", False)]
    while stack:
        x, depth, where, fin = stack.pop()
        if isinstance(x, str):
            _text_ok(x, where)
        elif isinstance(x, float):
            if fin and not math.isfinite(x):
                raise RequestError(f"{where}: {x} is not a finite number")
        elif isinstance(x, (dict, list)):
            if depth > max_depth:
                raise RequestError(f"the request body nests objects and arrays more than {max_depth} levels deep")
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(k, str):
                        _text_ok(k, where)
                    at = repr(k) if depth > 1 else k
                    stack.append((v, depth + 1, at if depth == 1 else f"{where}[{at}]",
                                  fin or depth == 1 and k in finite))
            else:
                stack.extend((v, depth + 1, f"{where}[{i}]", fin) for i, v in enumerate(x))
