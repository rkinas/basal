"""The TypeScript client's presets from basal/presets.py: clients/typescript/src/presets.ts (the question sets, the
guard policies) and test/presets_fixture.json (the same data, the language guess on shared texts, and guard() / route()
on shared answers, with the answers of the Python functions), which the client's tests compare it with.

  python scripts/ts_presets.py            write both files from basal/presets.py
  python scripts/ts_presets.py --check    exit 1, naming the file, when one of them is not what basal/presets.py gives

The guard and route cases are kept from the fixture (their answers are recomputed); the language cases are the texts
below plus every word that basal.presets.lang_of counts, alone, twice, in capitals and before "/".
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "clients" / "typescript" / "src" / "presets.ts"
FIXTURE = ROOT / "clients" / "typescript" / "test" / "presets_fixture.json"

TEXTS = [  # short lines as people type them: Polish often without diacritics, English with Polish names
    "Faktura nie została zapłacona w terminie, proszę o kontakt.", "The invoice was not paid on time, please call.",
    "Zażółć gęślą jaźń", "ZAŻÓŁĆ GĘŚLĄ JAŹŃ", "Hello", "Ala ma kota i psa w domu",
    "Dostałem to w poniedziałek i nie wiem co z tym zrobić", "Reply to w z", "", None, "Sprawa jest pilna",
    "Aplikacja na iPhone wylogowuje mnie co kilka minut.", "nie dziala logowanie", "NIE DZIALA LOGOWANIE",
    "Czy moge zmienic adres dostawy?", "Mam problem z faktura", "Prosze o kontakt", "to jest pilne",
    "Czego dotyczy zgłoszenie?", "Klient zgłasza, że nie działa logowanie do konta",
    "The customer cannot log in to the account", "App logs me out every few minutes on iPhone.", "i cant login",
    "Please do", "Why do I have to do this?", "ACME Co. invoice", "Order nr 123 w/o invoice", "PO 12345 not received",
    "Meeting in Kraków", "The customer from Łódź cannot log in", "Login broken", "niejest jestnie", "nie_dziala",
    "nie2 sie3", "w_z na-co", "Ćma i ŻUBR", "I I", "I/O na/po",
]


def words(pattern):
    """The words of the alternations (?:...) and (?-i:...) of a lang_of pattern."""
    return [w for group in re.findall(r"\(\?(?:-i)?:([^()]*)\)", pattern) for w in group.split("|")]


def lang_cases():
    from basal.presets import _EN, _PL, lang_of
    texts = list(TEXTS)
    for w in words(_PL.pattern) + words(_EN.pattern):
        texts += [w, f"{w} {w}", f"{w.upper()} {w.upper()}", f"{w}/x {w}/x", f"x{w} {w}x"]
    texts = list(dict.fromkeys(texts))
    return [[t, lang_of(t)] for t in texts]


def ts_source():
    from basal.presets import POLICIES, PRECEDENCE, PRESETS
    return ("// Generated from basal/presets.py by scripts/ts_presets.py (test/presets_fixture.json checks that they "
            "agree). Edit the Python file, not this one.\n"
            'import type { GuardPolicy, Questions } from "./types.js";\n\n'
            f"export const PRESETS = {json.dumps(PRESETS, indent=2, ensure_ascii=False)}"
            ' as unknown as Record<string, Record<"pl" | "en", Questions>>;\n\n'
            "/** Named guard policies (as POLICIES in basal/presets.py). */\n"
            f"export const POLICIES: Record<string, GuardPolicy> = {json.dumps(POLICIES, indent=2, ensure_ascii=False)}"
            " as unknown as Record<string, GuardPolicy>;\n\n"
            f"export const PRECEDENCE = {json.dumps(PRECEDENCE)} as const;\n")


def fixture_source(old):
    from basal.presets import POLICIES, PRESETS, guard, route
    out = {"presets": PRESETS, "langs": lang_cases(), "policies": POLICIES,
           "guard": [dict(c, expect=guard(c["answers"], c["policy"], c.get("max_error"), c.get("battery")))
                     for c in old["guard"]],
           "route": [dict(c, expect=route(c["answers"], 0.6, c.get("max_error"))) for c in old["route"]]}
    return json.dumps(out, indent=1, ensure_ascii=False)


def generated():
    """{path: the text basal/presets.py gives}."""
    return {TS: ts_source(), FIXTURE: fixture_source(json.loads(FIXTURE.read_text(encoding="utf-8")))}


def drift():
    """The files that differ from what basal/presets.py gives."""
    return [p for p, text in generated().items() if p.read_text(encoding="utf-8") != text]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv not in ([], ["--check"]):
        raise SystemExit(__doc__)
    if argv:
        bad = drift()
        for p in bad:
            print(f"{p.relative_to(ROOT)}: not what basal/presets.py gives; run python scripts/ts_presets.py")
        return int(bool(bad))
    for p, text in generated().items():
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
