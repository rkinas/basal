"""Task types (experimental, 2.1): classify, verify, compress, extract.

A task question is expanded into the decision types (choice, noul, multi) over the request's state, so it shares the
state (SOAM) with the request's other questions and uses their calibration; its answer is assembled from theirs. Text
fields of `extract` are two decisions: which line of the state holds the value, then which of that line's candidate
spans is the value (second_round), so a value is always exact source text, with its character offsets.

  classify  {"labels": {key: rubric} | [keys], "multi_label": false, "none": false, "max_error": [0.01, 0.05]}
  verify    {"claims": {key: claim} | [claims], "binary": false}
  compress  {"instructions": "<query>", "unit": "sentence" | "line" | "paragraph", "keep_ratio": 0.4 | "threshold": 0.5}
  extract   {"schema": <JSON schema: object properties, enum / oneOf const, boolean, string, number, integer, arrays of enum>,
             "max_error": [0.01, 0.05]}

`automated` (classify, each verify claim, each extract field) needs a threshold certified for the task in the
calibration file ("task_thresholds"); basal-calibrate fits them on labelled task questions.
"""
import json
import re

from .prompt import lang_of, reads_request

TASKS = ("classify", "verify", "compress", "extract")
SEP = "::"  # expanded question names: <task question>::<part>
RISKS = (0.01, 0.05)
MAX_UNITS = 600

W = {
    "pl": dict(
        classify="Która kategoria najlepiej opisuje ten tekst?", none="Żadna z powyższych kategorii",
        classify_multi="Które kategorie opisują ten tekst?",
        verify="Twierdzenie: {claim}\nCo źródło mówi o tym twierdzeniu?", supported="Źródło to potwierdza",
        contradicted="Źródło temu przeczy", not_stated="Źródło tego nie mówi",
        verify_binary="Twierdzenie: {claim}\nCzy źródło potwierdza to twierdzenie?",
        compress="Fragment: „{unit}”\nCzy ten fragment zawiera informację potrzebną do odpowiedzi na pytanie: {q}",
        compress_generic="Fragment: „{unit}”\nCzy ten fragment zawiera ważną informację z dokumentu?",
        line="W której linii dokumentu jest wartość pola: {field}?", enum="Jaka jest wartość pola: {field}?",
        span="Linia: „{line}”\nKtóry fragment tej linii to dokładnie wartość pola: {field}?",
        enum_multi="Które wartości ma pole: {field}?", missing="Nie podano w dokumencie", yes="Tak", no="Nie",
),
    "en": dict(
        classify="Which label best describes this text?", none="None of the labels above",
        classify_multi="Which labels describe this text?",
        verify="Claim: {claim}\nWhat does the source say about this claim?", supported="The source supports it",
        contradicted="The source contradicts it", not_stated="The source does not say",
        verify_binary="Claim: {claim}\nDoes the source support this claim?",
        compress="Passage: \"{unit}\"\nDoes this passage hold information needed to answer this question: {q}",
        compress_generic="Passage: \"{unit}\"\nDoes this passage hold important information of the document?",
        line="Which line of the document gives the value of the field: {field}?",
        enum="What is the value of the field: {field}?",
        span="Line: \"{line}\"\nWhich part of this line is exactly the value of the field: {field}?",
        enum_multi="Which values does the field have: {field}?", missing="Not stated in the document", yes="Yes",
        no="No"),
}


def is_task(q):
    return isinstance(q, dict) and q.get("type") in TASKS


def _risks(q):
    r = q.get("max_error", list(RISKS))
    return [float(x) for x in (r if isinstance(r, list) else [r])]


def _labels(spec, name):
    """{key: rubric}, [keys] or [{"name": key, "rubric": ...}] (the ScaleDown-style list) -> {key: rubric or None}."""
    if isinstance(spec, list) and spec and all(isinstance(x, dict) and "name" in x for x in spec):
        return {str(x["name"]): x.get("rubric") or x.get("description") for x in spec}
    if isinstance(spec, list) and spec:
        return {str(k): None for k in spec}
    if isinstance(spec, dict) and spec:
        return {str(k): v for k, v in spec.items()}
    raise ValueError(f"question {name!r}: labels must be a non-empty object {{key: rubric}} or a list of keys")


# --- compress units -------------------------------------------------------------------------------------------------

ABBREV = {"np", "itd", "itp", "tzw", "ul", "nr", "sp", "o.o", "tj", "ok", "godz", "pkt", "art", "ust", "zł", "e.g",
          "i.e", "etc", "mr", "mrs", "dr", "prof", "inc", "ltd", "co", "no", "vs", "st", "jr", "fig", "al"}
SENT_END = re.compile(r"[.!?…]+[\"”»)\]]*\s+(?=[\"„«(\[]?[A-ZĄĆĘŁŃÓŚŹŻ0-9])")
WORD_BEFORE = re.compile(r"[.!?…]*(\w[\w.]*)")  # matched on the reversed text before a sentence end


def units(text, unit="sentence"):
    """The text's units as (start, end) character spans, whitespace trimmed: lines, paragraphs (blank-line separated) or
    sentences (sentence ends, blank lines and line breaks before a capital, digit or bullet; common abbreviations and
    single letters do not end a sentence)."""
    if unit not in ("sentence", "line", "paragraph"):
        raise ValueError('unit must be "sentence", "line" or "paragraph"')
    if unit == "line":
        cuts = [m.end() for m in re.finditer(r"\n", text)]
    elif unit == "paragraph":
        cuts = [m.end() for m in re.finditer(r"\n[ \t]*\n", text)]
    else:
        cuts, prev = [], 0
        for m in SENT_END.finditer(text):
            # The word before the sentence end: the window starts after the previous match (no word spans its
            # whitespace) and is read backwards from the end, so every character is scanned once (O(n) per document).
            word = WORD_BEFORE.match(text[prev:m.start() + 1][::-1])
            prev = m.end()
            w = word.group(1)[::-1].lower() if word else ""
            if m.group(0)[0] == "." and (w in ABBREV or len(w) == 1 and w.isalpha()):
                continue
            cuts.append(m.end())
        cuts += [m.end() for m in re.finditer(r"\n[ \t]*\n|\n(?=[ \t]*[A-ZĄĆĘŁŃÓŚŹŻ0-9•\-*–])", text)]
    out, s = [], 0
    for c in sorted(set(cuts)) + [len(text)]:
        a, b = s, c
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
        s = c
    return out


def join_units(text, spans):
    """Kept units in source order; neighbours in the source keep their separator, gaps become a line break or space."""
    parts, prev = [], None
    for k, (a, b) in spans:
        if prev is not None:
            pk, pb = prev
            parts.append(text[pb:a] if k == pk + 1 else ("\n" if "\n" in text[pb:a] else " "))
        parts.append(text[a:b])
        prev = (k, b)
    return "".join(parts)


# --- extract schema -------------------------------------------------------------------------------------------------

def _type(s):
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def _consts(s):
    for key in ("oneOf", "anyOf"):
        alts = s.get(key)
        if alts and all(isinstance(a, dict) and "const" in a for a in alts):
            return [(a["const"], a.get("description") or a.get("title")) for a in alts]
    if "enum" in s:
        return [(v, None) for v in s["enum"] if v is not None]
    return None


def leaves(schema, path=""):
    """A JSON schema -> (path, kind, spec): kind is enum, enum_list, boolean, string, number or integer."""
    if not isinstance(schema, dict):
        raise ValueError(f"extract: schema at {path or 'root'!r} must be an object")
    t = _type(schema)
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        if not props:
            raise ValueError(f"extract: object {path or 'root'!r} has no properties")
        for k, sub in props.items():
            yield from leaves(sub, f"{path}.{k}" if path else str(k))
        return
    if _consts(schema):
        yield path, "enum", schema
    elif t == "boolean":
        yield path, "boolean", schema
    elif t == "array":
        items = schema.get("items") or {}
        if _consts(items):
            yield path, "enum_list", schema
        else:
            raise ValueError(f"extract: field {path!r}: arrays are supported only with enum items (experimental)")
    elif t in ("string", "number", "integer", None):
        yield path, t or "string", schema
    else:
        raise ValueError(f"extract: field {path!r}: type {t!r} is not supported")


MAX_CODES = 254  # option codes left for candidates (one is "not stated")
TRIM = " \t,;:()[]{}\"'„”«»|*"


def _shown(t, limit=400):
    t = " ".join(t.split())
    return t if len(t) <= limit else t[:limit] + " …"


def line_regions(text, cap=None):
    """The text's non-empty lines as (start, end); more than `cap` lines are grouped into consecutive blocks."""
    cap = cap or MAX_CODES
    lines = units(text, "line")
    if len(lines) <= cap:
        return lines
    k = -(-len(lines) // cap)
    return [(lines[i][0], lines[min(i + k, len(lines)) - 1][1]) for i in range(0, len(lines), k)]


def _trim(text, a, b):
    while a < b and text[a] in TRIM:
        a += 1
    while b > a and text[b - 1] in TRIM:
        b -= 1
    return a, b


def span_candidates(text, a, b, cap=None):
    """Candidate values inside text[a:b] as (start, end): the parts between separators (, ; | tab, and after a colon),
    then runs of up to L words (L as large as the cap allows, at most 8), each trimmed of punctuation, deduplicated."""
    cap = cap or MAX_CODES
    out, seen = [], set()

    def add(s, e):
        s, e = _trim(text, s, e)
        if e > s and text[s:e] not in seen and len(out) < cap and any(ch.isalnum() for ch in text[s:e]):
            seen.add(text[s:e])
            out.append((s, e))
    for m in re.finditer(r"[^,;|\t\n]+", text[a:b]):
        s, e = a + m.start(), a + m.end()
        add(s, e)
        c = text.find(":", s, e)
        if c >= 0:
            add(c + 1, e)
    words = [(a + m.start(), a + m.end()) for m in re.finditer(r"\S+", text[a:b])]
    L = max(1, min(8, cap // max(1, len(words))))
    for i in range(len(words)):
        for j in range(i, min(len(words), i + L)):
            add(words[i][0], words[j][1])
    return out


def second_round(out, body, plans, max_options=255):
    """Extract's text fields after the line decisions -> the request of span decisions (one choice per field over
    the candidates of its chosen line), or None. The candidates are kept in the plans for finish()."""
    state = body.get("state") if isinstance(body.get("state"), str) else ""
    qs = {}
    for name, p in plans.items():
        if p["type"] != "extract":
            continue
        w = W[p["lang"]]
        p["cands"] = {}
        for path, kind, spec in p["fields"]:
            if kind not in ("string", "number", "integer"):
                continue
            c = out["answers"][f"{name}{SEP}f{SEP}{path}"]["choice"]
            if c == "__missing__":
                continue
            a, b = p["regions"][int(c[1:])]
            cands = span_candidates(state, a, b, max_options - 1)
            p["cands"][path] = cands
            if len(cands) <= 1:  # nothing to choose: the one candidate is the value, none means "not stated"
                continue
            crit = {f"S{j}": state[s:e] for j, (s, e) in enumerate(cands)}
            crit["__missing__"] = w["missing"]
            qs[f"{name}{SEP}s{SEP}{path}"] = {"type": "choice", "criteria": crit, "option_keys": "hide",
                                              "instructions": p["pre"] + w["span"].format(
                                                  line=_shown(state[a:b]), field=_field_text(path, spec))}
    if not qs:
        return None
    return {k: v for k, v in body.items() if k in ("state", "lang", "facts")} | {"questions": qs}


def _field_text(path, spec):
    name = path.replace("_", " ").replace(".", " / ")
    d = spec.get("description") or spec.get("title")
    return f"{name} ({d})" if d else name


def _set_path(obj, path, value):
    *head, last = path.split(".")
    for k in head:
        obj = obj.setdefault(k, {})
    obj[last] = value


NUM = re.compile(r"[-+]?\d[\d\s .,']*")


def parse_number(text, integer=False):
    m = NUM.search(text or "")
    if not m:
        return None
    s = re.sub(r"[\s ']", "", m.group(0)).rstrip(".,")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) <= 2 and s.count(",") == 1 else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if integer:
        return int(v) if v == int(v) else None
    return int(v) if v == int(v) and "." not in s else v


# --- expansion ------------------------------------------------------------------------------------------------------

@reads_request
def expand(body, max_options=255):
    """A request with task questions -> (the request with only decision questions, plans to assemble the answers)."""
    state = body.get("state")
    text = state if isinstance(state, str) else ""
    questions, plans = {}, {}
    text_pl = not body.get("lang") and lang_of(text) == "pl"  # once per request, not once per task question
    for name, q in body["questions"].items():
        if not is_task(q):
            questions[name] = q
            continue
        if SEP in name:
            raise ValueError(f"question {name!r}: a task question's name may not contain {SEP!r}")
        t = q["type"]
        lang = body.get("lang") or ("pl" if text_pl else lang_of(str(q.get("instructions") or "")))
        w = W[lang]
        sub = {}
        if t == "classify":
            labels = _labels(q.get("labels") or q.get("criteria"), name)
            multi = bool(q.get("multi_label"))
            if q.get("none") and not multi:
                labels["none"] = w["none"]
            spec = {"type": "multi" if multi else "choice", "criteria": labels,
                    "instructions": q.get("instructions") or w["classify_multi" if multi else "classify"],
                    "option_keys": q.get("option_keys", "show")}
            if multi:
                spec.update({k: q[k] for k in ("threshold", "min", "max", "objective") if k in q})
            sub["labels"] = spec
            plans[name] = dict(type=t, multi=multi, risks=_risks(q))
        elif t == "verify":
            claims = q.get("claims")
            claims = {str(i): c for i, c in enumerate(claims)} if isinstance(claims, list) else claims
            if not isinstance(claims, dict) or not claims:
                raise ValueError(f"question {name!r}: claims must be a non-empty object or list")
            binary = bool(q.get("binary"))
            pre = (str(q["instructions"]) + "\n") if q.get("instructions") else ""
            for k, c in claims.items():
                c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                if binary:
                    sub[f"c{SEP}{k}"] = {"type": "noul", "instructions": pre + w["verify_binary"].format(claim=c)}
                else:
                    sub[f"c{SEP}{k}"] = {"type": "choice", "instructions": pre + w["verify"].format(claim=c),
                                         "criteria": {"supported": w["supported"], "contradicted": w["contradicted"],
                                                      "not_stated": w["not_stated"]}, "option_keys": "hide"}
            plans[name] = dict(type=t, keys=list(claims), binary=binary, risks=_risks(q))
        elif t == "compress":
            if not text:
                raise ValueError(f"question {name!r}: compress needs a text state")
            unit = q.get("unit", "sentence")
            spans = units(text, unit)
            if not spans:
                raise ValueError(f"question {name!r}: the state has no text to compress")
            if len(spans) > MAX_UNITS:
                raise ValueError(f"question {name!r}: {len(spans)} {unit}s (at most {MAX_UNITS}); use a larger unit "
                                 "or split the document")
            query = str(q.get("instructions") or "").strip()
            for i, (a, b) in enumerate(spans):
                u = text[a:b]
                sub[f"u{SEP}{i}"] = {"type": "noul", "instructions": w["compress"].format(q=query, unit=u) if query
                                     else w["compress_generic"].format(unit=u)}
            keep_ratio, thr = q.get("keep_ratio"), float(q.get("threshold", 0.5))
            if keep_ratio is not None and not 0 < float(keep_ratio) <= 1:
                raise ValueError(f"question {name!r}: keep_ratio must be in (0, 1]")
            plans[name] = dict(type=t, spans=spans, unit=unit, keep_ratio=keep_ratio, threshold=thr,
                               min_keep=int(q.get("min_keep", 1)))
        elif t == "extract":
            fields, regions = list(leaves(q.get("schema") or {})), None
            pre = (str(q["instructions"]) + "\n") if q.get("instructions") else ""
            for path, kind, spec in fields:
                f = _field_text(path, spec)
                if kind in ("enum", "enum_list"):
                    vals = _consts(spec if kind == "enum" else spec["items"])
                    crit = {str(v): f"{v} ({d})" if d else str(v) for v, d in vals}  # shown: the value (+ its description)
                    if kind == "enum":
                        crit["__missing__"] = w["missing"]
                        sub[f"f{SEP}{path}"] = {"type": "choice", "criteria": crit, "option_keys": "hide",
                                                "instructions": pre + w["enum"].format(field=f)}
                    else:
                        sub[f"f{SEP}{path}"] = {"type": "multi", "criteria": crit,
                                                "instructions": pre + w["enum_multi"].format(field=f)}
                elif kind == "boolean":
                    sub[f"f{SEP}{path}"] = {"type": "choice", "instructions": pre + w["enum"].format(field=f),
                                            "criteria": {"true": w["yes"], "false": w["no"], "__missing__": w["missing"]},
                                            "option_keys": "hide"}
                else:
                    if not text:
                        raise ValueError(f"question {name!r}: field {path!r} is taken from the state's text, which is "
                                         "empty (fields from media: enum and boolean only)")
                    regions = regions or line_regions(text, max_options - 1)
                    crit = {f"L{i}": _shown(text[a:b]) for i, (a, b) in enumerate(regions)}
                    crit["__missing__"] = w["missing"]
                    sub[f"f{SEP}{path}"] = {"type": "choice", "criteria": crit, "option_keys": "hide",
                                            "instructions": pre + w["line"].format(field=f)}
            plans[name] = dict(type=t, fields=fields, lang=lang, pre=pre, regions=regions, risks=_risks(q))
        for k, v in sub.items():
            questions[f"{name}{SEP}{k}"] = v
    return dict(body, questions=questions), plans


# --- assembly -------------------------------------------------------------------------------------------------------

def _automated(cal, task, conf, risks):
    """Whether a task answer may be used without review at each max_error: only with a threshold certified for that task
    (calibration file: "task_thresholds": {task: {"0.01": confidence, ...}}, fitted on the user's own labelled
    decisions); the thresholds of the decision types do not hold on task label sets (measured 2026-10-15), so without
    one the answer is null."""
    own = (cal.get("task_thresholds") or {}).get(task) or {}
    out = {}
    for r in risks:
        thr = own.get(str(float(r)))
        out[str(r)] = None if thr is None else bool(conf >= float(thr))
    return out


def finish(out, body, plans, cal, second=None):
    """The expanded answers of a response (and of second_round's request, `second`) -> one answer per task question."""
    ans = out["answers"]
    state = body.get("state") if isinstance(body.get("state"), str) else ""
    for name, p in plans.items():
        part = {k[len(name) + len(SEP):]: ans.pop(k) for k in list(ans) if k.startswith(name + SEP)}
        t = p["type"]
        if t == "classify":
            a = part["labels"]
            if p["multi"]:
                res = {"type": "classify", "labels": a["selected"], "scores": a["probabilities"],
                       "set_confidence": a["set_confidence"]}
            else:
                res = {"type": "classify", "label": a["choice"], "top_label": a["choice"],
                       "scores": a["probabilities"], "confidence": a["confidence"],
                       "automated": _automated(cal, "classify", a["confidence"], p["risks"])}
        elif t == "verify":
            claims, flagged = {}, []
            for k in p["keys"]:
                a = part[f"c{SEP}{k}"]
                if p["binary"]:
                    pr = {"supported": a["noul"], "not_supported": 1 - a["noul"]}
                    verdict = "supported" if a["noul"] >= 0.5 else "not_supported"
                else:
                    pr, verdict = a["probabilities"], a["choice"]
                conf = max(pr.values())
                claims[k] = {"verdict": verdict, "probabilities": pr, "confidence": conf,
                             "automated": _automated(cal, "verify", conf, p["risks"])}
                if verdict != "supported":
                    flagged.append(k)
            res = {"type": "verify", "claims": claims, "all_supported": not flagged, "not_supported": flagged}
        elif t == "compress":
            spans = p["spans"]
            pr = [part[f"u{SEP}{i}"]["noul"] for i in range(len(spans))]
            order = sorted(range(len(spans)), key=lambda i: -pr[i])
            total = sum(b - a for a, b in spans)
            if p["keep_ratio"] is not None:
                keep, n = set(), 0
                for i in order:
                    if n >= float(p["keep_ratio"]) * total:
                        break
                    keep.add(i)
                    n += spans[i][1] - spans[i][0]
            else:
                keep = {i for i in range(len(spans)) if pr[i] >= p["threshold"]}
            keep |= set(order[:p["min_keep"]])
            kept = sorted(keep)
            text = join_units(state, [(i, spans[i]) for i in kept])
            res = {"type": "compress", "compressed": text, "kept": kept, "unit": p["unit"],
                   "units": [{"start": a, "end": b, "probability": round(x, 6)} for (a, b), x in zip(spans, pr)],
                   "original_chars": len(state), "compressed_chars": len(text),
                   "ratio": round(len(text) / max(1, len(state)), 4)}
        else:  # extract
            data, fields = {}, {}
            for path, kind, spec in p["fields"]:
                a = part[f"f{SEP}{path}"]
                if kind == "enum":
                    vals = {str(v): v for v, _ in _consts(spec)}
                    c = a["choice"]
                    f = {"value": None if c == "__missing__" else vals[c], "probabilities": a["probabilities"],
                         "confidence": a["confidence"], "source": "choice"}
                elif kind == "enum_list":
                    vals = {str(v): v for v, _ in _consts(spec["items"])}
                    f = {"value": [vals[k] for k in a["selected"]], "probabilities": a["probabilities"],
                         "confidence": a["set_confidence"], "source": "choice"}
                elif kind == "boolean":
                    c = a["choice"]
                    f = {"value": None if c == "__missing__" else c == "true", "probabilities": a["probabilities"],
                         "confidence": a["confidence"], "source": "choice"}
                else:
                    pl = a["probabilities"]
                    p_line = 1 - pl["__missing__"]
                    f = {"value": None, "present_probability": p_line, "source": "span"}
                    cands = (p.get("cands") or {}).get(path)
                    if a["choice"] == "__missing__" or not cands:
                        f["confidence"] = pl["__missing__"]
                    else:
                        b2 = (second or {}).get(f"{name}{SEP}s{SEP}{path}")
                        if b2 is None:  # one candidate: the line itself
                            j, p_span = 0, 1.0
                        elif b2["choice"] == "__missing__":
                            j, p_span = None, b2["probabilities"]["__missing__"]
                        else:
                            j, p_span = int(b2["choice"][1:]), b2["confidence"]
                        if j is None:
                            f["confidence"] = p_span
                        else:
                            s0, e0 = cands[j]
                            txt = state[s0:e0]
                            f.update(text=txt, start=s0, end=e0, line_probability=a["confidence"],
                                     span_probability=p_span, confidence=a["confidence"] * p_span)
                            if kind in ("number", "integer"):
                                f["value"] = parse_number(txt, kind == "integer")
                                if f["value"] is None:
                                    f["error"] = f"not a {kind}"
                            else:
                                f["value"] = txt
                if "error" in f:  # a number that does not parse is never used without review
                    f["automated"] = {str(r): False for r in p["risks"]}
                elif kind == "enum_list":  # a label set has no certified threshold
                    f["automated"] = {str(r): None for r in p["risks"]}
                else:
                    f["automated"] = _automated(cal, "extract", f["confidence"], p["risks"])
                f["calibration"] = "validated" if f["source"] == "choice" else "per decision"
                fields[path] = f
                _set_path(data, path, f["value"])
            res = {"type": "extract", "data": data, "fields": fields}
        res["experimental"] = True
        ans[name] = res
    out["usage"]["questions"] = len(body["questions"])
    return out
