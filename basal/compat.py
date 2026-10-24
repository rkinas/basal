"""OpenAI Decisions compatibility translation for basal-serve."""
import base64
import re
import time
import uuid

from starlette.responses import JSONResponse


def openai_error(status, message, code, param=None):
    typ = "authentication_error" if status == 401 else "server_error" if status >= 500 else "invalid_request_error"
    return JSONResponse({"error": {"message": message, "type": typ, "param": param, "code": code}}, status_code=status)


MAX_MEDIA_BYTES = 20 * 1024 * 1024  # basal/media.py MAX_BYTES


def _decoded_size(encoded, status, what):
    """Bytes a base64 string decodes to; the length bound is checked first so an oversized item never gets decoded."""
    if len(encoded) * 3 // 4 > MAX_MEDIA_BYTES:
        raise ValueError((status, "media item exceeds 20 MiB", "INVALID_REQUEST"))
    try:
        size = len(base64.b64decode(encoded, validate=True))
    except Exception:
        raise ValueError((status, f"{what} must be valid base64", "INVALID_REQUEST")) from None
    if size > MAX_MEDIA_BYTES:
        raise ValueError((status, "media item exceeds 20 MiB", "INVALID_REQUEST"))
    return size


def _data_url(value, status=400):
    """(mime, base64 payload, decoded size) of a data URL; remote URLs are refused."""
    if not isinstance(value, str):
        raise ValueError((status, "media URL must be a data URL", "INVALID_REQUEST"))
    if value.startswith(("http://", "https://")):
        raise ValueError((status, "only data: URLs are accepted; basal does not fetch remote media", "INVALID_REQUEST"))
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value, re.DOTALL)
    if not match:
        raise ValueError((status, "media must be a base64 data URL", "INVALID_REQUEST"))
    return match.group(1), match.group(2), _decoded_size(match.group(2), status, "media data URL")


def systemone_media(body):
    """The vLLM-style `images` / `audio` data-URL arrays folded into basal's `media` items, every item validated."""
    out = dict(body)
    media = list(out.get("media") or [])
    image_values, audio_values = out.get("images", []), out.get("audio", [])
    if not isinstance(image_values, list):
        raise ValueError((422, "images must be an array", "INVALID_REQUEST"))
    if not isinstance(audio_values, list):
        raise ValueError((422, "audio must be an array", "INVALID_REQUEST"))
    if len(media) + len(image_values) + len(audio_values) > 8:
        raise ValueError((422, "at most 8 media items", "INVALID_REQUEST"))
    for item in media:
        if not isinstance(item, dict) or item.get("type") not in ("image", "audio") or not isinstance(item.get("data"), str):
            raise ValueError((422, "invalid media item", "INVALID_REQUEST"))
        value = item["data"]
        if value.startswith("data:"):
            _data_url(value, 422)
        else:
            _decoded_size(value, 422, "media data")
    for field, kind, values in (("images", "image", image_values), ("audio", "audio", audio_values)):
        out.pop(field, None)
        for value in values:
            mime, data, _ = _data_url(value, 422)
            if not mime.startswith(f"{kind}/"):
                raise ValueError((422, f"{field} entries must have {kind} MIME types", "INVALID_REQUEST"))
            media.append({"type": kind, "data": data})
    if media:
        out["media"] = media
    return out


def validate_systemone(body):
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {name!r} must be an object")
        if "type" in question and not isinstance(question["type"], str):
            raise ValueError(f"question {name!r}: type must be a string")
        for field in ("criteria", "levels"):
            value = question.get(field)
            if value is not None and not isinstance(value, (dict, list)):
                raise ValueError(f"question {name!r}: {field} must be an object or array")


def decision_body(body):
    allowed = {"model", "input", "questions", "safety_identifier", "metadata", "store"}
    extra = set(body) - allowed
    if extra:
        raise ValueError((400, f"unsupported parameter: {sorted(extra)[0]}", "unsupported_parameter", sorted(extra)[0]))
    source = body.get("input")
    state, media = "", []
    if isinstance(source, str):
        state = source
    elif isinstance(source, list):
        texts = []
        for msg in source:
            if not isinstance(msg, dict) or msg.get("role", "user") != "user":
                raise ValueError((400, "input messages must have role user", "INVALID_REQUEST"))
            content = msg.get("content", "")
            parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
            if not isinstance(parts, list):
                raise ValueError((400, "message content must be text or a list of parts", "INVALID_REQUEST"))
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError((400, "invalid input part", "INVALID_REQUEST"))
                if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif part.get("type") == "input_image":
                    _, data, size = _data_url(part.get("image_url"), 400)
                    if size > 20 * 1024 * 1024:
                        raise ValueError((400, "media item exceeds 20 MiB", "INVALID_REQUEST"))
                    media.append({"type": "image", "data": data})
                else:
                    raise ValueError((400, "unsupported input part", "INVALID_REQUEST"))
        state = "\n\n".join(texts)
    else:
        raise ValueError((400, "input must be a string or a list of user messages", "INVALID_REQUEST"))
    questions = body.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 200:
        raise ValueError((400, "questions must contain 1 to 200 entries", "INVALID_REQUEST"))
    converted, names, values, spec = {}, set(), {}, []
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError((400, "each question must be an object", "INVALID_REQUEST"))
        permitted = {"type", "name", "instructions", "choices", "levels"}
        if set(question) - permitted:
            field = sorted(set(question) - permitted)[0]
            raise ValueError((400, f"unsupported question field: {field}", "unsupported_parameter", f"questions[{i}].{field}"))
        kind, instructions, name = question.get("type"), question.get("instructions"), question.get("name")
        if not isinstance(instructions, str) or not instructions.strip() or kind not in ("predicate", "choice", "score"):
            raise ValueError((400, "question needs a supported type and non-empty instructions", "INVALID_REQUEST"))
        if name is not None and (not isinstance(name, str) or name in names):
            raise ValueError((400, "question names must be unique strings", "INVALID_REQUEST"))
        if name is not None:
            names.add(name)
        key = name or f"q{i}"
        while key in converted:
            key = "_" + key
        if kind == "predicate":
            converted[key] = {"type": "noul", "instructions": instructions}
        elif kind == "choice":
            choices = question.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise ValueError((400, "choices must contain 2 to 255 entries", "INVALID_REQUEST"))
            crit, vals = {}, []
            for choice in choices:
                if not isinstance(choice, dict) or not isinstance(choice.get("value"), (str, bool)):
                    raise ValueError((400, "choice value must be a string or boolean", "INVALID_REQUEST"))
                value = choice["value"]
                k = str(value).lower() if isinstance(value, bool) else value
                if k in crit:
                    raise ValueError((400, "choice values must be unique after string conversion", "INVALID_REQUEST"))
                crit[k] = choice.get("description")
                vals.append(value)
            converted[key] = {"type": "choice", "instructions": instructions, "criteria": crit}
            values[key] = vals
        else:
            levels = question.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10 or any(not isinstance(x, dict) or not isinstance(x.get("label"), str) for x in levels):
                raise ValueError((400, "levels must contain 2 to 10 labeled entries", "INVALID_REQUEST"))
            labels = [x["label"] for x in levels]
            if len(set(labels)) != len(labels):
                raise ValueError((400, "score level labels must be unique", "INVALID_REQUEST"))
            criteria = {x["label"]: x.get("description") for x in levels} if any(x.get("description") is not None for x in levels) else labels
            score_keys = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(criteria))]
            converted[key] = {"type": "score", "instructions": instructions, "criteria": criteria}
            values[key] = {"labels": [x["label"] for x in levels], "keys": score_keys}
        spec.append((name, kind, key))
    internal = {"state": state, "questions": converted}
    if media:
        if len(media) > 8:
            raise ValueError((400, "at most 8 media items", "INVALID_REQUEST"))
        internal["media"] = media
    return internal, spec, values


def decision_response(result, spec, values):
    answers = []
    for name, typ, key in spec:
        ans = result["answers"][key]
        if typ == "predicate":
            answers.append({"type": typ, "name": name, "probability": ans["noul"]})
        elif typ == "choice":
            probs = [{"value": v, "probability": ans["probabilities"][str(v).lower() if isinstance(v, bool) else str(v)]}
                     for v in values[key]]
            chosen = max(probs, key=lambda p: p["probability"])["value"]
            answers.append({"type": typ, "name": name, "choice": chosen, "probabilities": probs, "confidence": ans["confidence"]})
        else:
            probs = [{"value": i, "label": label, "probability": ans["probabilities"][values[key]["keys"][i]]}
                     for i, label in enumerate(values[key]["labels"])]
            answers.append({"type": typ, "name": name, "score": ans["score"], "probabilities": probs, "confidence": ans["confidence"]})
    usage = result.get("usage", {})
    total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return {"id": "dec_" + uuid.uuid4().hex, "object": "decision", "created": int(time.time()), "model": result["model"],
            "answers": answers, "usage": {"input_tokens": usage.get("input_tokens", 0), "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("output_tokens", 0), "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": total}}
