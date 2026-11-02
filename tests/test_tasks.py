"""Task types (basal/tasks.py) without a model: units, numbers, schema leaves, the four tasks through Server.decide with
a stub backend, and copy-only decoding with a character-level stand-in model."""
import asyncio
import re

import pytest

from basal import tasks
from basal.tasks import join_units, leaves, parse_number, span_candidates, units

from test_server import FakeServer


def by_text(pref):
    """Position probabilities from option texts: every option line "A. text" of the prompt gets pref(text) (default
    1), normalised over the shown options, so both option orders agree on the canonical probabilities."""
    def f(prompt, ids):
        s = "".join(map(chr, prompt))
        opts = re.findall(r"^[A-J]\. (.*)$", s, re.M)[-len(ids):]
        w = [pref(s, o) for o in opts]
        return [x / sum(w) for x in w]
    return f


def run(srv, body):
    return asyncio.run(srv._run(body))


def test_units_sentences_lines_paragraphs():
    t = "Faktura nr 12 z dnia 3.10.2026. Sprzedawca: ACME Sp. z o.o., ul. Długa 5. Termin płatności 14 dni!\n\nUwagi: brak."
    s = [t[a:b] for a, b in units(t)]
    assert s == ["Faktura nr 12 z dnia 3.10.2026.", "Sprzedawca: ACME Sp. z o.o., ul. Długa 5.",
                 "Termin płatności 14 dni!", "Uwagi: brak."]
    assert [t[a:b] for a, b in units(t, "paragraph")] == [t.split("\n\n")[0], "Uwagi: brak."]
    u = "a\n\n b \nc"
    assert [u[a:b] for a, b in units(u, "line")] == ["a", "b", "c"]
    sp = units(t)
    assert join_units(t, [(0, sp[0]), (1, sp[1]), (3, sp[3])]) == "Faktura nr 12 z dnia 3.10.2026. Sprzedawca: ACME Sp. " \
                                                                   "z o.o., ul. Długa 5.\nUwagi: brak."


def test_units_abbreviation_right_after_a_sentence_end():
    # the word before each sentence end is searched only from the previous sentence end on; skipped abbreviations and
    # single letters right after one must still not end a sentence
    t = 'Koniec. Dr. Nowak i prof. Kowalski. „Tak.” A. Mickiewicz napisał. Np. to. Ul. Długa 5! Nr. 7 stoi.'
    assert [t[a:b] for a, b in units(t)] == ["Koniec.", "Dr. Nowak i prof. Kowalski.", "„Tak.”",
                                             "A. Mickiewicz napisał.", "Np. to.", "Ul. Długa 5!", "Nr. 7 stoi."]


@pytest.mark.parametrize("text,integer,value", [
    ("1 234,56 zł", False, 1234.56), ("1,234.56", False, 1234.56), ("12,5", False, 12.5), ("1.234.567", False, 1234567),
    ("PLN 99", False, 99), ("7 szt.", True, 7), ("2,5", True, None), ("brak", False, None), ("1 000,00", False, 1000)])
def test_parse_number(text, integer, value):
    assert parse_number(text, integer) == value


def test_schema_leaves():
    schema = {"type": "object", "properties": {
        "number": {"type": "string", "description": "invoice number"},
        "seller": {"type": "object", "properties": {"name": {"type": "string"}, "vat_id": {"type": ["string", "null"]}}},
        "method": {"enum": ["transfer", "card", "cash"]},
        "risk": {"oneOf": [{"const": "low", "description": "routine"}, {"const": "high", "description": "check"}]},
        "paid": {"type": "boolean"}, "total": {"type": "number"},
        "tags": {"type": "array", "items": {"enum": ["urgent", "foreign"]}}}}
    assert [(p, k) for p, k, _ in leaves(schema)] == [
        ("number", "string"), ("seller.name", "string"), ("seller.vat_id", "string"), ("method", "enum"),
        ("risk", "enum"), ("paid", "boolean"), ("total", "number"), ("tags", "enum_list")]
    with pytest.raises(ValueError, match="arrays are supported only with enum items"):
        list(leaves({"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "object"}}}}))


def test_classify_single_and_multi():
    srv = FakeServer(by_text(lambda s, o: 8 if "refund" in o else 1))
    a = run(srv, {"state": "I want my money back for the broken kettle.", "questions": {
        "intent": {"type": "classify", "labels": [{"name": "refund", "rubric": "asks for a refund"},
                                                  {"name": "track", "rubric": "asks where the parcel is"}], "none": True},
        "other": {"type": "noul", "instructions": "Is it polite?"}}})
    c = a["answers"]["intent"]
    assert c["label"] == "refund" and c["top_label"] == "refund" and set(c["scores"]) == {"refund", "track", "none"}
    assert abs(c["scores"]["refund"] - 0.8) < 1e-6 and c["experimental"] and "other" in a["answers"]
    assert c["automated"] == {"0.01": None, "0.05": None}  # no task threshold: no certificate
    srv = FakeServer(by_text(lambda s, o: 8 if "refund" in o else 1),
                     cal={"thresholds": {"0.01": {"per_type": {"choice": {"confidence": 0.5}}}},
                          "task_thresholds": {"classify": {"0.05": 0.75}}})
    c = run(srv, {"state": "Money back please.", "questions": {"i": {"type": "classify", "labels": ["refund", "track"]}}})
    assert c["answers"]["i"]["automated"] == {"0.01": None, "0.05": True}  # the decision type's threshold is not used
    assert a["usage"]["questions"] == 2
    srv = FakeServer(by_text(lambda s, o: (9 if "refund" in s.split("Does this apply")[-1] else 1) if o == "Yes" else 3))
    c = run(srv, {"state": "Refund the kettle.", "questions": {
        "tags": {"type": "classify", "labels": ["refund", "damage"], "multi_label": True}}})["answers"]["tags"]
    assert c["labels"] == ["refund"] and abs(c["scores"]["refund"] - 0.75) < 1e-6


def test_verify_three_way_and_binary():
    def pref(s, o):
        claim = re.search(r"Claim: (.*)", s).group(1)
        good = {"The total is 120 EUR.": "supports", "Payment is due in 30 days.": "does not say"}[claim]
        return 9 if good in o else 1
    srv = FakeServer(by_text(pref))
    v = run(srv, {"state": "Invoice 7: total 120 EUR.", "questions": {
        "check": {"type": "verify", "claims": ["The total is 120 EUR.", "Payment is due in 30 days."]}}})["answers"]["check"]
    assert v["claims"]["0"]["verdict"] == "supported" and v["claims"]["1"]["verdict"] == "not_stated"
    assert v["all_supported"] is False and v["not_supported"] == ["1"]
    srv = FakeServer(by_text(lambda s, o: 9 if o == "Yes" else 1))
    v = run(srv, {"state": "Invoice 7.", "questions": {"c": {"type": "verify", "claims": {"t": "Total 120."},
                                                            "binary": True}}})["answers"]["c"]
    assert v["claims"]["t"]["verdict"] == "supported" and abs(v["claims"]["t"]["probabilities"]["supported"] - 0.9) < 1e-6


def test_compress_threshold_and_keep_ratio():
    state = "The order was placed on Monday. The weather was sunny. The total is 120 EUR. Our office has a cat."
    srv = FakeServer(by_text(lambda s, o: (9 if any(w in s.split("Passage:")[-1].split("answer")[0]
                                                     for w in ("total", "placed")) else 1) if o == "Yes" else 3))
    q = {"type": "compress", "instructions": "When and for how much was the order placed?"}
    c = run(srv, {"state": state, "questions": {"ctx": q}})["answers"]["ctx"]
    assert c["kept"] == [0, 2] and c["compressed"] == "The order was placed on Monday. The total is 120 EUR."
    assert c["original_chars"] == len(state) and len(c["units"]) == 4 and abs(c["units"][0]["probability"] - 0.75) < 1e-6
    c = run(srv, {"state": state, "questions": {"ctx": dict(q, keep_ratio=0.2)}})["answers"]["ctx"]
    assert len(c["kept"]) == 1


def test_extract_choice_fields():
    schema = {"type": "object", "properties": {"method": {"enum": ["transfer", "card"]}, "paid": {"type": "boolean"},
                                               "number": {"type": "string", "description": "invoice number"}}}

    def pref(s, o):
        q = s.split("<user>")[-1]
        if "method" in q:
            assert "__missing__" not in s
            return 9 if o.startswith("transfer") else 1
        if "paid" in q:
            return 9 if o == "No" else 1
        return 9 if o.startswith("Not stated") else 1  # number: not in the document
    srv = FakeServer(by_text(pref))
    e = run(srv, {"state": "Invoice. Pay by transfer.", "questions": {"x": {"type": "extract", "schema": schema}}})
    x = e["answers"]["x"]
    assert x["data"] == {"method": "transfer", "paid": False, "number": None}
    assert x["fields"]["number"]["present_probability"] < 0.5 and x["fields"]["method"]["source"] == "choice"


def test_span_candidates():
    t = "Seller: ACME Sp. z o.o., NIP 525-123-45-67 | Total: 1 230,00 PLN"
    c = [t[a:b] for a, b in span_candidates(t, 0, len(t))]
    for want in ("ACME Sp. z o.o.", "NIP 525-123-45-67", "525-123-45-67", "1 230,00 PLN", "1 230,00", "PLN"):
        assert want in c
    assert len(c) == len(set(c)) and all(x == x.strip(" ,:|") for x in c)
    assert len(span_candidates(" ".join(["w%d" % i for i in range(400)]), 0, 2289, cap=50)) == 50


def test_extract_text_fields_line_then_span():
    state = "Invoice no: FV/12\nSeller: ACME Ltd\nTotal: 120.50 EUR"
    want = {"number": ("FV/12", "FV/12"), "seller": ("ACME", "ACME Ltd"), "total": ("120.50", "120.50"),
            "iban": (None, None)}

    def pref(s, o):
        q = s.split("<user>")[-1]
        field = re.search(r"field: (\w+)", q).group(1)
        line_kw, value = want[field]
        if "Which line" in q:
            return 9 if (line_kw and line_kw in o) or (line_kw is None and o.startswith("Not stated")) else 1
        return 9 if o == value else 1
    srv = FakeServer(by_text(pref))
    schema = {"type": "object", "properties": {"number": {"type": "string"}, "seller": {"type": "string"},
                                               "total": {"type": "number"}, "iban": {"type": "string"}}}
    x = run(srv, {"state": state, "questions": {"x": {"type": "extract", "schema": schema}}})["answers"]["x"]
    assert x["data"] == {"number": "FV/12", "seller": "ACME Ltd", "total": 120.5, "iban": None}
    f = x["fields"]["seller"]
    assert state[f["start"]:f["end"]] == "ACME Ltd" and f["source"] == "span"
    assert abs(f["confidence"] - f["line_probability"] * f["span_probability"]) < 1e-9
    assert x["fields"]["iban"]["present_probability"] < 0.5


def test_extract_line_without_candidates_is_not_stated():
    state = "Invoice no: FV/12\n---\nTotal: 5"
    schema = {"type": "object", "properties": {"number": {"type": "string"}}}

    def pref(s, o):
        q = s.split("<user>")[-1]
        return 9 if "Which line" in q and o == "---" else 1
    x = run(FakeServer(by_text(pref)), {"state": state, "questions": {"x": {"type": "extract", "schema": schema}}})
    assert x["answers"]["x"]["data"] == {"number": None}


def test_task_name_separator_refused():
    with pytest.raises(ValueError, match="may not contain"):
        tasks.expand({"state": "x", "questions": {"a::b": {"type": "classify", "labels": ["p", "q"]}}})
    # a decision question named like one of a task's expanded questions took its answer: classify "task" read the
    # choice "task::labels" (its label "red") and the choice's own answer was dropped
    with pytest.raises(ValueError, match="may not contain"):
        tasks.expand({"state": "x", "questions": {
            "task": {"type": "classify", "labels": ["invoice", "letter"]},
            "task::labels": {"type": "choice", "instructions": "Which colour?", "criteria": ["red", "blue"]}}})
    # without task questions nothing is expanded, and "::" stays allowed in names
    out = run(FakeServer([0.6, 0.4]), {"state": "x", "questions": {"a::b": {"type": "noul", "instructions": "?"}}})
    assert list(out["answers"]) == ["a::b"]


def test_task_language_from_state_then_instructions():
    # a Polish state decides every task question; an English one leaves it to each question's instructions;
    # the request's lang overrides both
    qs = {"a": {"type": "classify", "labels": ["p", "q"], "none": True, "instructions": "Which one?"},
          "b": {"type": "classify", "labels": ["p", "q"], "none": True, "instructions": "Która pasuje?"}}
    none = lambda body: [body["questions"][f"{k}::labels"]["criteria"]["none"] for k in "ab"]
    pl, en = tasks.W["pl"]["none"], tasks.W["en"]["none"]
    assert none(tasks.expand({"state": "Kot śpi.", "questions": qs})[0]) == [pl, pl]
    assert none(tasks.expand({"state": "The cat sleeps.", "questions": qs})[0]) == [en, pl]
    assert none(tasks.expand({"state": "Kot śpi.", "questions": qs, "lang": "en"})[0]) == [en, en]
