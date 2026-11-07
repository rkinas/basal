"""What a request body is checked for before any work, shared by basal-serve's routes (POST /v1/systemone, /v1/basal,
/v1/decisions and the MCP tools); no torch, so basal-router can use it too.

A body is JSON that a client may have built by hand: it can nest deeper than the code that reads it recurses, carry an
escape such as "\\udead" (a lone surrogate: not Unicode text, which the tokenizer and the UTF-8 response refuse) or,
in a question, a NaN or an infinity (Python's json reads them, and an answer keyed or computed by one cannot be
written). Each is the request's fault, answered 422 (400 on /v1/decisions) before anything is rendered or run.

Also the request helpers both read alike: a value as text (text_of), the options as the model sees them
(named_options) and a body read once (read_json for basal-serve, parse_body for the router)."""
import json
import math
from collections import Counter

from .prompt import RequestError

MAX_BODY_BYTES = 256 * 2**20  # --max-body-mb: a request body (8 media items of 20 MB are ~214 MB in base64)
MAX_DEPTH = 256  # objects and arrays nested in a request body (JSON states nest a few levels; Python's parser and
#                  the recursive readers after it stop near 1,000, with a 500)


def _text_ok(s, where):
    if not s.isascii():
        try:
            s.encode("utf-8")
        except UnicodeEncodeError as e:
            bad = s[e.start: e.end].encode("utf-8", "backslashreplace").decode("ascii")
            raise RequestError(f"{where}: {bad} is a lone surrogate (half of a UTF-16 pair), not text") from None


def _not_text(s):
    """A string that holds a lone surrogate (not UTF-8 text)."""
    if s.isascii():
        return False
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _refused(body, max_depth, finite):
    """Whether check_json refuses the body, found without naming where: no location text or stack entry per element
    (the walk that names one took 5.5 s and 45 times the body's size in memory on 32 MiB of [{}, ...], on the event
    loop). A stack of iterators, one per open container: its memory follows the nesting, not the breadth."""
    if not isinstance(body, (dict, list)):
        return isinstance(body, str) and _not_text(body)
    is_dict = isinstance(body, dict)
    stack = [(iter(body.items()) if is_dict else iter(body), 1, False, is_dict)]
    while stack:
        it, depth, fin, is_dict = stack[-1]
        for e in it:
            if is_dict:
                k, v = e
                if isinstance(k, str) and _not_text(k):
                    return True
                f = fin or depth == 1 and k in finite
            else:
                v, f = e, fin
            if isinstance(v, str):
                if _not_text(v):
                    return True
            elif isinstance(v, float):
                if f and not math.isfinite(v):
                    return True
            elif isinstance(v, (dict, list)):
                if depth + 1 > max_depth:
                    return True
                if v:  # (descend; this container's iterator goes on after it)
                    d = isinstance(v, dict)
                    stack.append((iter(v.items()) if d else iter(v), depth + 1, f, d))
                    break
        else:
            stack.pop()
    return False


def check_json(body, max_depth=MAX_DEPTH, finite=("questions",)):
    """A parsed request body -> RequestError when its objects / arrays nest deeper than max_depth, a string (a key or a
    value) holds a lone surrogate, or a number under one of the top-level fields `finite` is NaN or infinite. Iterative:
    a deep body cannot exhaust the stack here. A body it takes is walked once, fast (_refused); only a refused one is
    walked again, to name where the problem is (_locate)."""
    if max_depth < 1 or _refused(body, max_depth, finite):
        _locate(body, max_depth, finite)


def _locate(body, max_depth, finite):
    """check_json's refusal of a body, with where it is (the walk check_json was before _refused)."""
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


def read_json(raw):
    """A request body (bytes) -> its JSON value; RequestError for one that is not JSON, not UTF-8 or nested beyond what
    the parser reads (RecursionError)."""
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as e:  # (UnicodeDecodeError is a ValueError)
        raise RequestError(f"the request body is not valid JSON: {e}") from None


async def request_json(request):
    """A Starlette request's body as read_json reads it, kept on the request as request.json() keeps it (--ui reads
    it there)."""
    try:
        return await request.json()
    except (ValueError, RecursionError) as e:
        raise RequestError(f"the request body is not valid JSON: {e}") from None


def parse_body(raw):
    """A request body read once for basal-router (routing, the cascade, the blend, the fallback): the JSON object, or
    None when it is not one basal-serve would take (not JSON, not an object, nested too deeply, a lone surrogate); the
    replica it goes to then answers it with its own error."""
    try:
        body = read_json(raw)
        if not isinstance(body, dict):
            return None
        check_json(body, finite=())
    except RequestError:
        return None
    return body


def text_of(x):
    """Strings as they are; structured values (objects, lists, numbers) as compact JSON, as the model sees them."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def named_options(crit, option_keys="show"):
    """{key: description} -> (keys, option texts shown to the model).

    option_keys="show" (default): every described option is shown as "key: description", because only the caller knows
    whether a key is meaningful (a service code "B", a decision name "approve"); the server never guesses. A key without
    a description is shown by itself.
    option_keys="hide": the caller states that the keys are placeholders and the descriptions alone define the options
    (the format of the training data and of our PL/EN evaluation clients); descriptions that are not unique within the
    question still get their key, otherwise the options could not be told apart."""
    if option_keys not in ("show", "hide"):
        raise RequestError(f'option_keys must be "show" or "hide", got {option_keys!r}')
    texts = [str(k) if v is None else text_of(v) for k, v in crit.items()]
    dup = {t for t, n in Counter(texts).items() if n > 1}  # (once: texts.count per option took 4.5 s for 32,000)
    shown = [t if v is None or (option_keys == "hide" and t not in dup) else f"{k}: {t}"
             for (k, v), t in zip(crit.items(), texts)]
    return list(crit), shown


def limit_body(app, max_bytes):
    """ASGI: a request body over max_bytes answers 413 before the app reads it (its Content-Length), or as soon as more
    of it arrived (a body sent without one: the app sees the client gone, and its answer is replaced by the 413)."""
    async def too_large(scope, send):
        msg = f"the request body is larger than this server's limit of {max_bytes} bytes (--max-body-mb)"
        body = {"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": "PAYLOAD_TOO_LARGE"}} \
            if scope["path"] in ("/v1/decisions", "/decisions") else {"error": msg, "code": "PAYLOAD_TOO_LARGE"}
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    async def limited(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        length = dict(scope.get("headers", ())).get(b"content-length", b"")
        if length.isdigit() and int(length) > max_bytes:
            await too_large(scope, send)
            return
        seen, over = 0, False

        async def counted():
            nonlocal seen, over
            if over:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > max_bytes:
                    over = True
                    return {"type": "http.disconnect"}
            return message

        async def answer(message):
            if not over:
                await send(message)
        try:
            await app(scope, counted, answer)
        except Exception:  # noqa: BLE001  (the app's error over a body it never got whole)
            if not over:
                raise
        if over:
            await too_large(scope, send)
    return limited
