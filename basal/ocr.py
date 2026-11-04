"""External image OCR: generated candidates and optional text context (PaddleX HTTP)."""
import asyncio
import base64
import json
import time
from urllib.parse import urlsplit

import httpx
import regex

from . import media as media_module
from .prompt import RequestError, lang_of

OTHER_KEY = "__other__"
MAX_TEXT_LINES = 4096
MAX_TEXT_CHARS = 200_000
MAX_OCR_RESPONSE_BYTES = 8 * 1024 * 1024  # bound untrusted upstream HTTP bodies before JSON parsing
MAX_MATCHES = 4096


class OCRServiceError(RuntimeError):
    """Upstream OCR failure; this is not an empty recognition result."""


class OCRClient:
    def __init__(self, api, url, timeout=30.0):
        if api != "paddlex":
            raise ValueError(f"unsupported OCR API {api!r}")
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("--ocr-url requires an HTTP(S) base URL without credentials, query or fragment")
        if not 0 < timeout <= 300:
            raise ValueError("--ocr-timeout must be within (0, 300] seconds")
        self.url = url.rstrip("/") + "/ocr"
        self.timeout = timeout

    async def _request(self, data):
        """Read a bounded, uncompressed PaddleX body before decoding any JSON."""
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            async with client.stream("POST", self.url,
                                     json={"file": data, "fileType": 1, "visualize": False},
                                     headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                # aiter_raw avoids decompressing a large upstream body before the size check.
                if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                    raise OCRServiceError("PaddleX OCR response must not be compressed")
                declared = response.headers.get("content-length")
                if declared and declared.isascii() and declared.isdigit():
                    if len(declared) > len(str(MAX_OCR_RESPONSE_BYTES)) or int(declared) > MAX_OCR_RESPONSE_BYTES:
                        raise OCRServiceError(f"PaddleX OCR response exceeds {MAX_OCR_RESPONSE_BYTES} bytes")
                body = bytearray()
                if response.is_stream_consumed:
                    # Some in-memory HTTPX mock responses are already buffered.
                    # Real streamed upstream responses use the bounded branch below.
                    buffered = response.content
                    if len(buffered) > MAX_OCR_RESPONSE_BYTES:
                        raise OCRServiceError(f"PaddleX OCR response exceeds {MAX_OCR_RESPONSE_BYTES} bytes")
                    body.extend(buffered)
                else:
                    async for chunk in response.aiter_raw(chunk_size=64 * 1024):
                        if len(body) + len(chunk) > MAX_OCR_RESPONSE_BYTES:
                            raise OCRServiceError(f"PaddleX OCR response exceeds {MAX_OCR_RESPONSE_BYTES} bytes")
                        body.extend(chunk)
        return json.loads(body)

    async def read(self, image_base64, index):
        try:
            raw = media_module._raw(image_base64, index)
            media_module._image(raw, index)  # enforce all image limits before contacting PaddleX
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        data = base64.b64encode(raw).decode("ascii")
        started = time.monotonic()
        try:
            # httpx timeout is per read/write/connect inactivity, not a total wall-clock limit.
            # wait_for also cancels a slow streaming response (supported on Python 3.10+).
            payload = await asyncio.wait_for(self._request(data), timeout=self.timeout)
            if time.monotonic() - started >= self.timeout:
                # The bounded synchronous JSON parse may overrun the deadline.
                raise asyncio.TimeoutError()
        except (httpx.HTTPError, ValueError, RecursionError, TimeoutError) as exc:
            raise OCRServiceError(f"PaddleX OCR request failed ({type(exc).__name__})") from exc
        if not isinstance(payload, dict) or payload.get("errorCode") != 0:
            raise OCRServiceError("PaddleX OCR returned an unsuccessful response")
        result = payload.get("result")
        pages = result.get("ocrResults") if isinstance(result, dict) else None
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            raise OCRServiceError("PaddleX OCR response has no single-image ocrResults")
        pruned = pages[0].get("prunedResult")
        texts = pruned.get("rec_texts") if isinstance(pruned, dict) else None
        if not isinstance(texts, list) or any(not isinstance(s, str) for s in texts):
            raise OCRServiceError("PaddleX OCR response is missing rec_texts")
        if len(texts) > MAX_TEXT_LINES or sum(map(len, texts)) > MAX_TEXT_CHARS:
            raise OCRServiceError("PaddleX OCR returned too much text")
        return texts


def _enabled(value, index):
    if value is True or value == "on":
        return True
    if value is None or value is False or value == "off":
        return False
    raise RequestError(f'media {index}: ocr must be "on" or "off"')


def _static(name, q):
    raw = q.get("criteria") or {}
    if isinstance(raw, dict):
        c = dict(raw)
    elif isinstance(raw, list):
        try:
            c = dict.fromkeys(raw)
        except TypeError as exc:
            raise RequestError(f"question {name!r}: unhashable criteria") from exc
    else:
        raise RequestError(f"question {name!r}: criteria must be an object or list")
    if any(not isinstance(k, str) for k in c):
        raise RequestError(f"question {name!r}: generated criteria require string option keys")
    if OTHER_KEY in c:
        raise RequestError(f"question {name!r}: {OTHER_KEY!r} is reserved for special OCR option")
    return c


def _matches(name, pattern, texts):
    if not isinstance(pattern, str) or not 0 < len(pattern) <= 512:
        raise RequestError(f"question {name!r}: filter must be a regex of 1..512 characters")
    try:
        pat = regex.compile(pattern)
    except regex.error as exc:
        raise RequestError(f"question {name!r}: invalid OCR regex: {exc}") from None
    out = []
    deadline = time.monotonic() + 2.0  # total bound for all OCR fragments of this question
    for text in texts:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestError(f"question {name!r}: OCR regex timed out")
        try:
            for match in pat.finditer(text, timeout=min(0.05, remaining)):
                value = match.group(0).strip()
                if value:
                    out.append(value)
                    if len(out) > MAX_MATCHES:
                        raise RequestError(f"question {name!r}: too many OCR matches")
        except TimeoutError:
            raise RequestError(f"question {name!r}: OCR regex timed out") from None
    return out


def _label(q, lang):
    value = q.get("__other__")
    if value is None:
        return "Żadna z powyższych" if lang == "pl" else "None of the above"
    if not isinstance(value, str) or not value.strip():
        raise RequestError("__other__ must be a non-empty label or null")
    return value.strip()


def _empty(q):
    base = {"decision_source": "ocr_preprocessor", "reason": "no_candidates"}
    if q.get("type") == "multi":
        if q.get("objective", "threshold") == "threshold":
            return dict(type="multi", selected=[], probabilities={}, threshold=float(q.get("threshold", 0.5)),
                        set_confidence=None, **base)
        return dict(type="multi", selected=[], probabilities={}, objective=q["objective"],
                    expected_score=None, set_confidence=None, **base)
    return dict(type="choice", choice=None, confidence=None, probabilities={}, **base)


async def prepare(body, client, max_options):
    """Resolve generated criteria and media.context without changing the caller's request.

    Image (default): keep the image; OCR only if generated_criteria references it.
    Text: read OCR and replace that image with OCR text in the state.
    Both: read OCR, append the text to the state, and keep the original image.
    None: keep the image available only to the OCR candidate resolver, never to the model.
    Each image needed for either purpose is recognized at most once.
    """
    questions = body["questions"]
    for name, q in questions.items():
        if "other" in q:
            raise RequestError(f"question {name!r}: use __other__ instead of other")
    entries = body.get("media")
    if isinstance(entries, list) and len(entries) > media_module.MAX_ITEMS:
        raise RequestError(f"{len(entries)} media items, at most {media_module.MAX_ITEMS} per request")
    has_criteria = any("generated_criteria" in q or "__other__" in q for q in questions.values())
    has_context = isinstance(entries, list) and any(isinstance(m, dict) and "context" in m for m in entries)
    if not has_criteria and not has_context:
        return body, {}, None
    if entries is not None and body.get("images") is not None:
        raise RequestError("give media or images, not both")
    entries = entries or []
    if not isinstance(entries, list):
        raise RequestError("media must be a list")

    by_name, contexts = {}, {}
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise RequestError(f"media {i}: expected an object")
        context = item.get("context", "image")
        if "context" in item:
            if item.get("type") != "image":
                raise RequestError(f"media {i}: context is supported only for images")
            if context not in ("image", "text", "both", "none"):
                raise RequestError(f'media {i}: context must be "image", "text", "both" or "none"')
        enabled = _enabled(item.get("ocr"), i)
        if context in ("text", "both") and not enabled:
            raise RequestError(f'media {i}: context={context!r} requires ocr:"on"')
        if enabled:
            if item.get("type") != "image":
                raise RequestError(f"media {i}: OCR is supported only for images")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RequestError(f"media {i}: ocr:on requires a name")
            if name in by_name:
                raise RequestError(f"media {i}: duplicate OCR image name {name!r}")
            by_name[name] = (i, item)
            contexts[name] = context

    specs = {}
    for name, q in questions.items():
        if "generated_criteria" not in q:
            continue
        spec = q["generated_criteria"]
        if q.get("type", "choice") not in ("choice", "multi"):
            raise RequestError(f"question {name!r}: generated_criteria supports choice and multi only")
        if not isinstance(spec, dict) or spec.get("preprocessor") != "ocr":
            raise RequestError(f"question {name!r}: generated_criteria.preprocessor must be 'ocr'")
        src = spec.get("source")
        if not isinstance(src, str) or src not in by_name:
            raise RequestError(f"question {name!r}: source must name a media image with ocr:on")
        if not isinstance(spec.get("filter"), str):
            raise RequestError(f"question {name!r}: filter must be a regex string")
        specs[name] = spec

    context_names = [name for name in by_name if contexts[name] in ("text", "both")]
    drop = {i for i, item in enumerate(entries) if item.get("context") in ("text", "none")}
    requested = set(context_names) | {spec["source"] for spec in specs.values()}
    if requested and client is None:
        raise RequestError('OCR context or generated_criteria requires --ocr-api paddlex')
    names = [name for name in by_name if name in requested]
    if not names and not has_criteria and not drop:
        return body, {}, None
    tasks = [asyncio.create_task(client.read(by_name[n][1].get("data"), by_name[n][0])) for n in names]
    try:
        read = await asyncio.gather(*tasks)
    except BaseException:
        # gather does not cancel sibling tasks when one raises. Stop and await them
        # before the request returns; also clean up if our own request is cancelled.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    texts = dict(zip(names, read))

    prepared, empty, sources, counts = {}, {}, {}, {}
    for name, q in questions.items():
        if name not in specs and "__other__" not in q:
            prepared[name] = q
            continue
        kind = q.get("type", "choice")
        if "__other__" in q and kind != "choice":
            raise RequestError(f"question {name!r}: __other__ only applies to choice")
        criteria = _static(name, q)
        origin = {k: ["static"] for k in criteria}
        seen = {}
        for key, description in criteria.items():
            seen.setdefault(key, key)
            if isinstance(description, str):
                seen.setdefault(description, key)
        matches = _matches(name, specs[name]["filter"], texts[specs[name]["source"]]) if name in specs else []
        for candidate in matches:
            if candidate in seen:
                key = seen[candidate]
                if "ocr" not in origin[key]:
                    origin[key].append("ocr")
            elif candidate != OTHER_KEY:
                criteria[candidate] = None
                seen[candidate] = candidate
                origin[candidate] = ["ocr"]
        counts[name] = len(matches)
        limit = max_options - 1 if kind == "choice" else 255
        if len(criteria) > limit:
            raise RequestError(f"question {name!r}: {len(criteria)} options exceed the {limit} candidate limit")
        if not criteria:
            empty[name] = _empty(q)
            sources[name] = origin
            continue
        item = dict(q, criteria=criteria)
        item.pop("generated_criteria", None)
        if kind == "choice":
            lang = body.get("lang") or lang_of(str(q.get("instructions", "")))
            item["_basal_other_label"] = _label(q, lang)
            criteria[OTHER_KEY] = None
            origin[OTHER_KEY] = ["__other__"]
        prepared[name] = item
        sources[name] = origin

    if empty and any(q.get("ask_if") is not None or q.get("depends_on") is not None for q in questions.values()):
        raise RequestError("conditional questions with zero OCR candidates are not supported")

    adjusted = dict(body, questions=prepared)
    if context_names:
        state = body["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        blocks = []
        for name in context_names:
            label = json.dumps(name, ensure_ascii=False)
            blocks.append(f"[OCR: {label}]\n" + "\n".join(texts[name]) + "\n[/OCR]")
        adjusted["state"] = "\n\n".join(part for part in [state, *blocks] if part)
    if drop:
        remaining = [item for i, item in enumerate(entries) if i not in drop]
        if remaining:
            adjusted["media"] = remaining
        else:
            adjusted.pop("media", None)

    meta = {"sources": sources, "matched": counts, "images": len(names)}
    return adjusted, empty, meta if names or has_criteria else None


def apply_answers(output, original, empty, meta, seconds):
    answers = dict(empty)
    answers.update(output["answers"])
    for name, origin in meta["sources"].items():
        answer = answers[name]
        if answer.get("type") == "choice":
            key = answer["choice"]
            if key == OTHER_KEY:
                answer["choice"] = None
                answer["__other__"] = True
                answer["source"] = ["__other__"]
            elif key in origin:
                answer["source"] = origin[key][:]
        elif answer.get("type") == "multi" and answer.get("selected"):
            answer["sources"] = {k: origin.get(k, [])[:] for k in answer["selected"]}
    output["answers"] = {name: answers[name] for name in original["questions"]}
    output["usage"]["questions"] = len(original["questions"])
    output["usage"]["preprocessor_only"] = len(empty)
    if meta["images"]:
        output["usage"]["ocr"] = {"images": meta["images"], "matches": meta["matched"]}
    output["usage"]["latency_ms"] = round(seconds * 1000, 2)
    return output
