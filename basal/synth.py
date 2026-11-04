"""Synthetic decision items with long states, for speed and identity measurements (no real data).

A state is a generated case log (Polish or English) of about `tokens` tokens, the question asks which unit should
handle the latest entry, and the options are 2..k department names (or `options` of them, for many-option tests).
The texts are meaningless as decisions; only their shape (length, language, option count) matters.

  python -m basal.synth --n 500 --tokens 1500 --out synth.jsonl
"""
import argparse
import json
import random

CHARS_PER_TOKEN = {"pl": 2.1, "en": 1.85}  # Bielik v3 tokenizers on these logs (dates and amounts; measured)

WORDS = {
    "pl": dict(
        names=["Anna Nowak", "Piotr Wiśniewski", "Katarzyna Wójcik", "Tomasz Kamiński", "Magdalena Lewandowska",
               "Michał Zieliński", "Agnieszka Szymańska", "Paweł Woźniak", "Ewa Dąbrowska", "Krzysztof Kozłowski"],
        issues=["uszkodzona przesyłka", "brak zwrotu pieniędzy", "błędna faktura", "opóźniona dostawa",
                "nie działa logowanie", "podwójne obciążenie karty", "zmiana adresu dostawy", "reklamacja towaru",
                "pytanie o gwarancję", "wymiana na inny rozmiar"],
        status=["przyjęte", "w toku", "czeka na klienta", "przekazane", "zamknięte"],
        line="[{d:02d}.{mo:02d}.2026 {h:02d}:{mi:02d}] Zgłoszenie {id}: {name} — {issue}; kwota {amt},{gr:02d} zł, "
             "status: {st}.",
        head="Historia zgłoszeń klienta (system obsługi, eksport automatyczny):",
        question="Który dział powinien obsłużyć ostatnie zgłoszenie?",
        units=["Reklamacje", "Płatności", "Logistyka", "Obsługa techniczna", "Faktury", "Sprzedaż", "Dział prawny",
               "Kadry", "Magazyn", "Zwroty"],
        unit="Dział {k}"),
    "en": dict(
        names=["Alice Moore", "John Carter", "Emily Brooks", "David Turner", "Sarah Hughes", "Mark Bennett",
               "Laura Price", "James Foster", "Olivia Reed", "Daniel Ward"],
        issues=["damaged parcel", "refund not received", "wrong invoice", "late delivery", "cannot log in",
                "card charged twice", "change of delivery address", "product complaint", "warranty question",
                "exchange for another size"],
        status=["received", "in progress", "waiting for customer", "forwarded", "closed"],
        line="[{d:02d}/{mo:02d}/2026 {h:02d}:{mi:02d}] Ticket {id}: {name} - {issue}; amount {amt}.{gr:02d} EUR, "
             "status: {st}.",
        head="Customer ticket history (support system, automatic export):",
        question="Which department should handle the latest ticket?",
        units=["Complaints", "Payments", "Logistics", "Technical support", "Invoicing", "Sales", "Legal", "HR",
               "Warehouse", "Returns"],
        unit="Department {k}"),
}


def state(rnd, lang, tokens):
    w = WORDS[lang]
    lines, n = [w["head"]], len(w["head"])
    while n < tokens * CHARS_PER_TOKEN[lang]:
        x = w["line"].format(d=rnd.randint(1, 28), mo=rnd.randint(1, 12), h=rnd.randint(7, 20), mi=rnd.randint(0, 59),
                             id=rnd.randint(10000, 99999), name=rnd.choice(w["names"]), issue=rnd.choice(w["issues"]),
                             amt=rnd.randint(10, 9999), gr=rnd.randint(0, 99), st=rnd.choice(w["status"]))
        lines.append(x); n += len(x) + 1
    return "\n".join(lines)


def items(n, tokens=1500, seed=0, options=None, spread=0.33, langs=("pl", "en"), long_frac=0.0, long_tokens=(3000, 10000)):
    """n items; state lengths uniform in tokens * (1 +- spread), except a share long_frac of long states (uniform in
    long_tokens; a long-document tail); options: fixed count, else 2..6 at random."""
    if options is not None and options < 2:
        raise ValueError(f"options must be at least 2, not {options}")
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        lang = langs[i % len(langs)]
        w = WORDS[lang]
        k = options or rnd.randint(2, 6)
        units = w["units"][:] if k <= len(w["units"]) else w["units"] + [w["unit"].format(k=j) for j in range(11, k + 1)]
        units = units[:k]
        rnd.shuffle(units)
        t = int(tokens * (1 + rnd.uniform(-spread, spread)))
        if long_frac and rnd.random() < long_frac:
            t = rnd.randint(*long_tokens)
        out.append(dict(id=f"synth-{seed}-{i}", state=state(rnd, lang, t), question=w["question"], options=units))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tokens", type=int, default=1500, help="mean state length in tokens")
    ap.add_argument("--options", type=int, default=None, help="options per item (default: 2..6 at random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long-frac", dest="long_frac", type=float, default=0.0, help="share of long states (e.g. 0.1)")
    ap.add_argument("--long-tokens", dest="long_tokens", type=int, nargs=2, default=(3000, 10000))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items(a.n, a.tokens, a.seed, a.options, long_frac=a.long_frac, long_tokens=tuple(a.long_tokens)):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
