"""Ready question sets for common jobs (2.5): `basal-run --preset triage`, `Basal.preset(state, "triage")`, the
framework adapters (basal/adapters.py). Each preset has Polish and English wording; the language follows the state
unless given. The questions are plain HTTP-API questions (docs/API.md), so a preset is only a starting point: copy
and edit it for your own rules.

  basal-run --list-presets
  basal-run --preset triage --input tickets.jsonl --output out.jsonl     (input lines: {"state": "..."})"""
import copy
import re

LEVELS = {"pl": ["niska", "średnia", "wysoka"], "en": ["low", "medium", "high"]}  # score levels, lowest first

PRESETS = {
    "triage": {
        "pl": {
            "urgency": {"type": "score", "instructions": "Jak pilna jest sprawa?", "criteria": LEVELS["pl"]},
            "category": {"type": "choice", "instructions": "Czego dotyczy zgłoszenie?",
                         "criteria": {"complaint": "reklamacja lub skarga", "billing": "płatności lub faktury",
                                      "technical": "problem techniczny", "account": "konto lub dane klienta",
                                      "other": "coś innego"}},
            "needs_human": {"type": "noul", "instructions": "Czy sprawa wymaga decyzji człowieka?"},
        },
        "en": {
            "urgency": {"type": "score", "instructions": "How urgent is this case?", "criteria": LEVELS["en"]},
            "category": {"type": "choice", "instructions": "What is the request about?",
                         "criteria": {"complaint": "a complaint", "billing": "payments or invoices",
                                      "technical": "a technical problem", "account": "the account or customer data",
                                      "other": "something else"}},
            "needs_human": {"type": "noul", "instructions": "Does the case need a human decision?"},
        },
    },
    "moderation": {
        "pl": {
            "harmful": {"type": "noul", "instructions": "Czy tekst zawiera treści obraźliwe, groźby lub nękanie?"},
            "spam": {"type": "noul", "instructions": "Czy tekst jest spamem lub reklamą?"},
            "personal_data": {"type": "noul", "instructions": "Czy tekst ujawnia dane osobowe innej osoby?"},
        },
        "en": {
            "harmful": {"type": "noul", "instructions": "Does the text contain insults, threats or harassment?"},
            "spam": {"type": "noul", "instructions": "Is the text spam or advertising?"},
            "personal_data": {"type": "noul", "instructions": "Does the text reveal someone else's personal data?"},
        },
    },
    "compliance": {
        "pl": {
            "personal_data": {"type": "noul", "instructions": "Czy dokument zawiera dane osobowe?"},
            "deadline": {"type": "noul", "instructions": "Czy dokument wyznacza termin do dotrzymania?"},
            "legal_review": {"type": "noul", "instructions": "Czy dokument powinien przejrzeć prawnik?"},
        },
        "en": {
            "personal_data": {"type": "noul", "instructions": "Does the document contain personal data?"},
            "deadline": {"type": "noul", "instructions": "Does the document set a deadline to meet?"},
            "legal_review": {"type": "noul", "instructions": "Should a lawyer review the document?"},
        },
    },
}

_PL = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]|\b(i|w|z|nie|się|jest|na|do|że|od)\b")


def names():
    return sorted(PRESETS)


def lang_of(text):
    """pl when the text looks Polish (diacritics, common words), else en."""
    return "pl" if len(_PL.findall(text or "")) >= 2 else "en"


def questions(name, lang=None, state=""):
    """The questions of preset `name` in `lang` (pl / en; None: the language of `state`), as a fresh dict."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (presets: {', '.join(names())})")
    lang = lang or lang_of(state)
    if lang not in PRESETS[name]:
        raise ValueError(f"preset {name!r} has no {lang!r} wording (pl, en)")
    return copy.deepcopy(PRESETS[name][lang])
