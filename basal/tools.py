"""The `tool` question type: which tool an agent calls next, from tool definitions in the OpenAI or MCP form.

  {"type": "tool", "instructions": "..." (optional, a default question per language),
   "tools": [{"type": "function", "function": {"name", "description", "parameters"}}   OpenAI Chat Completions
             {"type": "function", "name", "description", "parameters"}                 OpenAI Responses
             {"name", "description", "inputSchema"}                                    MCP tools/list
             {"name", "description", "input_schema"}, ...],                            Anthropic
   "extra": ["answer_directly", "ask_user", "none"]   options besides the tools (default: all three; [] = tools only)
   "rank": k                                          also the k most probable options, in order
   "description_chars": n                             cap of each description (default DESCRIPTION_CHARS; 0 = none)
   "multi": true                                      the set of tools to call (the `multi` objectives and constraints)
   "max_error": 0.01}                                 defer to a person below the certified confidence threshold

Each tool is one option, `name(a, b, c?) — description` (the top-level parameters, optional ones marked "?"; the
description on one line, cut on a word boundary after DESCRIPTION_CHARS characters with "…"), followed by the extra
options in the request's order; the keys of the answer are the tool names and the extra names. The model
reads the options as a choice question (as a yes/no question per tool with "multi"), so the tool count is bounded by the
model's prompt format: up to 255 options with prompt-format-2 models, 10 with format 1.
"""
import re
from collections import Counter

EXTRA = {
    "answer_directly": {"en": "Answer directly without calling a tool",
                        "pl": "Odpowiedz bezpośrednio, bez wywoływania narzędzia"},
    "ask_user": {"en": "Ask the user for the missing information",
                 "pl": "Dopytaj użytkownika o brakujące informacje"},
    "none": {"en": "No suitable tool — tell the user",
             "pl": "Brak odpowiedniego narzędzia — poinformuj użytkownika"},
}
QUESTION = {"en": "What should the assistant do next: which tool should it call?",
            "pl": "Co asystent powinien teraz zrobić: które narzędzie wywołać?"}
QUESTION_SET = {"en": "Which tools should the assistant call now?",
                "pl": "Które narzędzia asystent powinien teraz wywołać?"}
SCHEMA_KEYS = ("parameters", "inputSchema", "input_schema")
DESCRIPTION_CHARS = 300  # default cap of a tool description in its option (MCP descriptions can run to many hundreds)


def _definition(name, i, t):
    """One tool definition in any of the accepted forms -> (name, description, parameter schema or None)."""
    where = f"question {name!r}: tools[{i}]"
    if not isinstance(t, dict):
        raise ValueError(f"{where} must be an object (a tool definition), got {type(t).__name__}")
    if "type" in t and t["type"] != "function":
        raise ValueError(f"{where}: type {t['type']!r} is not supported (function tools only)")
    f = t.get("function", t)
    if not isinstance(f, dict):
        raise ValueError(f"{where}.function must be an object")
    tool = f.get("name")
    if not isinstance(tool, str) or not tool or re.search(r"\s", tool):
        raise ValueError(f"{where}: name must be a non-empty string without spaces, got {tool!r}")
    desc = f.get("description")
    if desc is not None and not isinstance(desc, str):
        raise ValueError(f"{where} ({tool}): description must be a string")
    key = next((k for k in SCHEMA_KEYS if k in f), None)
    return tool, desc, _check_schema(f.get(key), f"{where} ({tool}).{key}")


def _check_schema(s, where):
    """The parameter schema must be a JSON-schema object: type "object" (when given), properties an object of schemas,
    required a list of property names."""
    if s is None:
        return None
    if not isinstance(s, dict):
        raise ValueError(f"{where} must be a JSON schema object, got {type(s).__name__}")
    if "type" in s and s["type"] != "object":
        raise ValueError(f"{where}: type must be \"object\", got {s['type']!r}")
    props = s.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError(f"{where}.properties must be an object of parameter schemas")
    bad = [p for p, v in props.items() if not isinstance(v, (dict, bool))]
    if bad:
        raise ValueError(f"{where}.properties: {bad} are not schemas (objects)")
    req = s.get("required", [])
    if not isinstance(req, list) or not all(isinstance(r, str) for r in req) or len(set(req)) != len(req):
        raise ValueError(f"{where}.required must be a list of distinct parameter names")
    unknown = [r for r in req if r not in props]
    if unknown:
        raise ValueError(f"{where}.required: {unknown} are not among the properties")
    return s


def option_text(tool, desc, schema, cap=DESCRIPTION_CHARS):
    """`name(a, b, c?) — description`: top-level parameters in schema order, optional ones marked "?"; the description
    on one line, cut after `cap` characters (0: no cap) at the last word boundary, with "…"."""
    s = schema or {}
    req = set(s.get("required", []))
    params = ", ".join(p if p in req else p + "?" for p in s.get("properties", {}))
    text = f"{tool}({params})"
    desc = " ".join((desc or "").split())
    if cap and len(desc) > cap:
        head = desc[:cap]
        if desc[cap] != " " and " " in head:  # inside a word: back to the last space
            head = head[: head.rindex(" ")]
        desc = head.rstrip(" ,;:") + "…"
    return f"{text} — {desc}" if desc else text


def parse(name, q, lang):
    """A `tool` question -> (keys, option texts, tool names). Duplicate or clashing names, malformed definitions and
    unknown extras are refused (HTTP 422)."""
    if "criteria" in q:
        raise ValueError(f"question {name!r}: tool takes \"tools\" (tool definitions), not criteria")
    tools = q.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"question {name!r}: tool needs \"tools\", a non-empty list of tool definitions")
    defs = [_definition(name, i, t) for i, t in enumerate(tools)]
    names = [d[0] for d in defs]
    dup = sorted(n for n, k in Counter(names).items() if k > 1)  # (once: names.count per tool was quadratic)
    if dup:
        raise ValueError(f"question {name!r}: duplicate tool names {dup}")
    extra = q.get("extra", [] if q.get("multi") else list(EXTRA))
    if not isinstance(extra, list) or not all(isinstance(e, str) for e in extra) or len(set(extra)) != len(extra):
        raise ValueError(f"question {name!r}: extra must be a list of distinct names from {list(EXTRA)}")
    unknown = [e for e in extra if e not in EXTRA]
    if unknown:
        raise ValueError(f"question {name!r}: unknown extra options {unknown} (supported: {list(EXTRA)})")
    clash = [e for e in extra if e in names]
    if clash:
        raise ValueError(f"question {name!r}: tool names {clash} clash with the extra options; rename the tools or "
                         "leave those names out of extra")
    if not isinstance(q.get("multi", False), bool):
        raise ValueError(f"question {name!r}: multi must be true or false")
    n = len(names) + len(extra)
    rank = q.get("rank")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= n):
        raise ValueError(f"question {name!r}: rank must be an integer in 1..{n} (the number of options)")
    cap = q.get("description_chars", DESCRIPTION_CHARS)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError(f"question {name!r}: description_chars must be a non-negative integer (0 = no cap)")
    me = q.get("max_error")
    if me is not None:
        if q.get("multi"):
            raise ValueError(f"question {name!r}: max_error is not supported with multi (the certified thresholds are "
                             "for single decisions)")
        if isinstance(me, bool) or not isinstance(me, (int, float)) or not 0 < me < 1:
            raise ValueError(f"question {name!r}: max_error must be a number between 0 and 1")
    return names + extra, [option_text(*d, cap=cap) for d in defs] + [EXTRA[e][lang] for e in extra], names


def ranking(probs, k):
    """The k most probable options, ties in option order."""
    return [{"tool": t, "probability": probs[t]} for t in sorted(probs, key=probs.get, reverse=True)[:k]]


def answer(q, probs, conf, cal):
    from .server import NO_CERTIFICATE, certified  # the server imports this module
    """The chosen option of a single-tool question. With max_error, a decision below the certified confidence
    threshold of choice questions (or with no such threshold for the target) is deferred: `tool` is null, `deferred`
    true and `refused` names the option the model would have chosen."""
    top = max(probs, key=probs.get)
    ans = {"type": "tool", "tool": top, "call": top in q["tools"], "probabilities": probs, "confidence": conf}
    if q["rank"]:
        ans["ranking"] = ranking(probs, q["rank"])
    if q["max_error"] is not None:
        thr = certified(cal, q["max_error"], "choice")
        if thr is None:
            ans["max_error"] = NO_CERTIFICATE
        ans["deferred"] = thr is None or conf < thr
        if ans["deferred"]:
            ans.update(tool=None, call=False, refused=top)
    return ans


def set_answer(q, ans):
    """A `multi` answer over the tools -> the `tool` answer: the selected set, and whether it calls any tool."""
    out = {"type": "tool", "multi": True, "tools": ans.pop("selected"), **{k: v for k, v in ans.items() if k != "type"}}
    out["call"] = any(t in q["tools"] for t in out["tools"])
    if q["rank"]:
        out["ranking"] = ranking(out["probabilities"], q["rank"])
    return out
