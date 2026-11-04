"""POST /v1/decisions: OpenAI's Decisions API over basal-serve's own decisions (the OpenAI Python SDK's
client.decisions.create works against it).

A request becomes the System One request it stands for and is decided like POST /v1/systemone, so both endpoints
give the same probabilities for the same questions:

  input         the state: the text, or the list of messages / items as JSON (images are refused)
  predicate     noul: the probability that the statement is true
  choice        choice with the options {value: description} (a boolean value is shown as true / false); the answer's
                choice and probabilities carry the typed values back, in the order of the choices
  score         score with the levels "label" or "label: description", lowest first; probabilities per level index
  name          kept on the answer (null when the question has none); answers come in question order

Errors have OpenAI's shape {"error": {"message", "type", "param", "code"}}: HTTP 400 invalid_request_error for a
request to change, 5xx server_error otherwise.
"""
import json

from .prompt import RequestError

MAX_QUESTIONS = 400  # the Decisions API's limit


class DecisionsError(RequestError):
    """An invalid /v1/decisions request; `param` names the field (OpenAI's error "param")."""

    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


def _value_key(value):
    """A choice value as a System One key: the string itself, a boolean as "true" / "false"."""
    return json.dumps(value) if isinstance(value, bool) else value


def _has_image(x):
    """An input_image part anywhere in x (a walk, not a recursion: an input nested a few hundred levels deep, valid
    JSON, raised RecursionError, an HTTP 500)."""
    todo = [x]
    while todo:
        x = todo.pop()
        if isinstance(x, dict):
            if x.get("type") == "input_image":
                return True
            todo += x.values()
        elif isinstance(x, list):
            todo += x
    return False


def _parts(x, where):
    """Instructions or input: text, or a list of parts / items; image parts are refused."""
    if isinstance(x, str):
        return x
    if not isinstance(x, list) or not x:
        raise DecisionsError("must be a string or a non-empty array", where)
    if _has_image(x):
        raise DecisionsError("images are not supported by this server", where)
    return x


def to_systemone(body):
    """A /v1/decisions body -> (the System One body, [(question name, kind, typed values or level labels)])."""
    if not isinstance(body, dict):
        raise DecisionsError("the request body must be a JSON object")
    if "input" not in body:
        raise DecisionsError("missing required parameter: 'input'", "input")
    if "model" in body and not isinstance(body["model"], str):
        raise DecisionsError("'model' must be a string", "model")
    state = _parts(body["input"], "input")
    qs = body.get("questions")
    if not isinstance(qs, list) or not 1 <= len(qs) <= MAX_QUESTIONS:
        raise DecisionsError(f"'questions' must be an array of 1 to {MAX_QUESTIONS} questions", "questions")
    questions, specs = {}, []
    for i, q in enumerate(qs):
        where = f"questions[{i}]"
        if not isinstance(q, dict):
            raise DecisionsError("a question must be an object", where)
        name = q.get("name")
        if name is not None and not isinstance(name, str):
            raise DecisionsError("'name' must be a string", f"{where}.name")
        if "instructions" not in q:
            raise DecisionsError("missing required parameter: 'instructions'", f"{where}.instructions")
        instr, kind = _parts(q["instructions"], f"{where}.instructions"), q.get("type")
        if kind == "predicate":
            questions[where] = {"type": "noul", "instructions": instr}
            specs.append((name, kind, None))
        elif kind == "choice":
            choices = q.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise DecisionsError("'choices' must be an array of 2 to 255 choices", f"{where}.choices")
            crit, values = {}, []
            for j, c in enumerate(choices):
                at = f"{where}.choices[{j}]"
                if not isinstance(c, dict) or not isinstance(c.get("value"), (str, bool)):
                    raise DecisionsError("a choice needs a string or boolean 'value'", at)
                if not isinstance(c.get("description", ""), str):
                    raise DecisionsError("'description' must be a string", f"{at}.description")
                key = _value_key(c["value"])
                if key in crit:
                    raise DecisionsError(f"choice values must be unique (as text): {key!r}", at)
                crit[key] = c.get("description")
                values.append(c["value"])
            questions[where] = {"type": "choice", "instructions": instr, "criteria": crit}
            specs.append((name, kind, values))
        elif kind == "score":
            levels = q.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 255:
                raise DecisionsError("'levels' must be an array of 2 to 255 levels", f"{where}.levels")
            for j, lv in enumerate(levels):
                if not isinstance(lv, dict) or not isinstance(lv.get("label"), str) \
                        or not isinstance(lv.get("description", ""), str):
                    raise DecisionsError("a level needs a string 'label' (and an optional string 'description')",
                                         f"{where}.levels[{j}]")
            texts = [lv["label"] if lv.get("description") is None else f"{lv['label']}: {lv['description']}"
                     for lv in levels]
            questions[where] = {"type": "score", "instructions": instr, "criteria": texts}
            specs.append((name, kind, [lv["label"] for lv in levels]))
        else:
            raise DecisionsError("'type' must be one of: predicate, choice, score", f"{where}.type")
    return {"state": state, "questions": questions}, specs


def to_decision(out, specs):
    """basal-serve's System One response -> the Decision object (answers in question order)."""
    answers = []
    for i, (name, kind, extra) in enumerate(specs):
        a = out["answers"][f"questions[{i}]"]
        if kind == "predicate":
            answers.append({"type": "predicate", "name": name, "probability": a["noul"]})
        elif kind == "choice":
            p = a["probabilities"]
            by_key = {_value_key(v): v for v in extra}
            answers.append({"type": "choice", "name": name, "choice": by_key[a["choice"]],
                            "confidence": a["confidence"],
                            "probabilities": [{"value": v, "probability": p[_value_key(v)]} for v in extra]})
        else:
            p = a["probabilities"]
            answers.append({"type": "score", "name": name, "score": a["score"], "confidence": a["confidence"],
                            "probabilities": [{"label": lab, "probability": p[str(k)], "value": k}
                                              for k, lab in enumerate(extra)]})
    n = int(out.get("usage", {}).get("input_tokens", 0))
    return {"model": out.get("model"), "answers": answers,
            "usage": {"input_tokens": n, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                      "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": n}}


def error_body(e, status):
    """OpenAI's error object for a failed /v1/decisions request with this HTTP status."""
    kind = "invalid_request_error" if status < 500 else "server_error"
    return {"error": {"message": str(e), "type": kind, "param": getattr(e, "param", None), "code": None}}
