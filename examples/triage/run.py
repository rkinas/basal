"""Helpdesk triage on labelled tickets: the `triage` preset (urgency, category, needs_human) against a running
basal-serve, scored against the gold labels, with a certified error target that sends uncertain answers to a person.

  basal-serve --model Remek/basal-2.0-mini-MLX-8bit --mode mlx --port 8000     # or any model and mode
  python examples/triage/run.py --url http://127.0.0.1:8000 --max-error 0.05
  python examples/triage/run.py --url http://127.0.0.1:8000 --rules             # the preset with RULES in the options
  python examples/triage/run.py --calibration-input labelled.jsonl             # requests for basal-calibrate
  python examples/triage/run.py --rules --lang both --calibration-input labelled.jsonl   # Polish and English wording

Input lines: {"id", "state", "gold": {"category": key, "urgency": 0-2, "needs_human": "yes" | "no"}}.
"""
import argparse
import copy
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from basal.client import AsyncBasal
from basal.presets import lang_of, questions

HERE = Path(__file__).parent

# --rules: the preset's questions with the rules of one helpdesk written into the options. The preset alone says
# "category: the account or customer data", "needs a human decision?"; the model answers better when the options say
# what they cover here.
RULES = {
    "pl": {
        "category": {"complaint": "reklamacja lub skarga: wadliwy, uszkodzony lub niezgodny towar, opóźniona, zgubiona "
                                  "lub niedostarczona przesyłka, status zamówienia lub dostawy, zwrot towaru, odwołana "
                                  "lub niezrealizowana usługa (wyjazd, transfer, wynajem, wdrożenie, wizyta technika), "
                                  "szkoda, niedotrzymana obietnica lub umowa, zła obsługa",
                     "billing": "pieniądze: płatności, faktury, ceny, plany i abonamenty, raty, metody płatności (karta, "
                                "BLIK, przelew), opłaty, kaucje i blokady na karcie, zwroty pieniędzy, windykacja",
                     "technical": "technika: błąd, awaria lub pytanie techniczne o aplikację, stronę, API, integrację, "
                                  "internet, sieć, telefon, telewizję, router, dekoder lub inne urządzenie, także "
                                  "konfiguracja, zgodność i limity",
                     "account": "konto klienta: logowanie, hasło, kody weryfikacyjne, dane, adres, użytkownicy, "
                                "uprawnienia, przejęcie konta lub numeru, zmiana właściciela konta lub umowy, wnioski "
                                "RODO o dane, usunięcie konta",
                     "other": "tylko gdy żadna z powyższych nie pasuje: oferta handlowa, współpraca, prasa, "
                              "rekrutacja, godziny otwarcia, lokalizacja, ogólne pytanie o firmę, podziękowanie"},
        "urgency": ["niska: pytanie lub prośba bez terminu, nic nie przestało działać",
                    "średnia: coś nie działa albo klient traci pieniądze, ale może poczekać dzień",
                    "wysoka: awaria blokująca pracę lub sprzedaż, włamanie, klient potrzebuje pomocy teraz"],
        "needs_human": {"true": "pracownik musi podjąć decyzję: zwrot pieniędzy, korekta, reklamacja, wyjątek, "
                                "sprawa prawna, bezpieczeństwo konta, awaria",
                        "false": "wystarczy gotowa odpowiedź, instrukcja lub zwykła zmiana danych"},
    },
    "en": {
        "category": {"complaint": "a complaint: a faulty, damaged or wrong product, a late, lost or undelivered "
                                  "parcel, the status of an order or delivery, returning goods, a cancelled or "
                                  "undelivered service (a trip, a transfer, a rental, an implementation, a technician "
                                  "visit), damage, a broken promise or contract, bad service",
                     "billing": "money: payments, invoices, prices, plans and subscriptions, instalments, payment "
                                "methods (card, BLIK, transfer), fees, deposits and holds on a card, refunds, debt "
                                "collection",
                     "technical": "technical: an error, an outage or a technical question about the app, the website, "
                                  "the API, an integration, the internet, the network, a phone, TV, a router, a "
                                  "set-top box or another device, also setup, compatibility and limits",
                     "account": "the customer account: login, password, verification codes, data, address, users, "
                                "permissions, a taken-over account or number, a change of the account or contract "
                                "holder, GDPR data requests, closing the account",
                     "other": "only when none of the above fits: a sales offer, a partnership, press, recruitment, "
                              "opening hours, a location, a general question about the company, thanks"},
        "urgency": ["low: a question or request with no deadline, nothing stopped working",
                    "medium: something does not work or the customer loses money, but it can wait a day",
                    "high: an outage that blocks work or sales, a break-in, the customer needs help now"],
        "needs_human": {"true": "a person must decide: a refund, a correction, a complaint, an exception, a legal "
                                "matter, account security, an outage",
                        "false": "a ready answer, an instruction or a plain change of data is enough"},
    },
}


# --rules also asks the safety question, before any routing (app.py sends a yes to the "alarm" queue). It is a filter
# tuned to miss nothing, not an automated decision: a ticket goes to the alarm queue from SAFETY_AT, far below 0.5,
# and it carries no max_error, so no calibration can turn it off.
SAFETY_AT = 0.2
SAFETY = {
    "pl": {"type": "noul", "instructions": "Czy w zgłoszeniu jest zagrożenie zdrowia, życia lub bezpieczeństwa ludzi?",
           "criteria": {"true": "tak: dym, ogień, iskrzenie, porażenie prądem, zapach gazu lub chemii, zatrucie lub "
                                "reakcja alergiczna, uraz, nagły stan zdrowia, problem z lekiem, sprzęt medyczny lub "
                                "osoba bez potrzebnego zasilania, niebezpieczny produkt lub pojazd, ktoś uwięziony, "
                                "groźba przemocy, także wspomniane mimochodem",
                        "false": "nie: pieniądze, dostawa, konto, dane, awaria albo pilna sprawa bez ryzyka dla "
                                 "zdrowia i bezpieczeństwa ludzi"}},
    "en": {"type": "noul", "instructions": "Is there a risk to people's health, life or safety in the ticket?",
           "criteria": {"true": "yes: smoke, fire, sparks, an electric shock, a smell of gas or chemicals, poisoning "
                                "or an allergic reaction, an injury, a sudden illness, a medicine problem, medical "
                                "equipment or a person without the power they need, an unsafe product or vehicle, "
                                "someone trapped, a threat of violence, also when mentioned in passing",
                        "false": "no: money, a delivery, the account, data, an outage or an urgent matter with no "
                                 "risk to people's health or safety"}},
}


def triage_questions(state, lang=None, max_error=None, rules=False):
    """The triage preset's questions about state; with rules, RULES in the options and the SAFETY question."""
    qs = questions("triage", lang, state, max_error)
    if rules:
        lang = lang or lang_of(state)
        for name, criteria in RULES[lang].items():
            qs[name]["criteria"] = criteria
        qs["danger"] = copy.deepcopy(SAFETY[lang])
    return qs


def predicted(answer):
    """The answer of a choice / score / noul question as a gold label: a key, a level index or "yes" / "no"."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "score":
        return int(max(answer["probabilities"], key=answer["probabilities"].get))
    return "yes" if answer["noul"] >= 0.5 else "no"


def report(rows, max_error):
    names = [n for n in rows[0]["gold"] if n in rows[0]["answers"] and n != "danger"]
    print(f"{len(rows)} tickets\n")
    print(f"{'question':<12} {'accuracy':>9}" + (f" {'automated':>10} {'error when automated':>21}" if max_error else ""))
    for name in names:
        hits = [predicted(r["answers"][name]) == r["gold"][name] for r in rows]
        line = f"{name:<12} {sum(hits) / len(hits):>9.0%}"
        if max_error:
            auto = [h for h, r in zip(hits, rows) if not r["answers"][name].get("abstain")]
            err = f"{1 - sum(auto) / len(auto):.0%} of {len(auto)}" if auto else "-"
            line += f" {len(auto) / len(rows):>10.0%} {err:>21}"
        print(line)
    whole = [r for r in rows if not any(r["answers"][n].get("abstain") for n in names)] if max_error else rows
    if max_error:
        print(f"\nautomated end to end (no question abstained): {len(whole)} of {len(rows)}; "
              f"the rest go to the human queue")
        reasons = Counter(r["answers"][n].get("max_error", "below the certified threshold")
                          for r in rows for n in names if r["answers"][n].get("abstain"))
        for why, n in reasons.most_common():
            print(f"  abstained {n}x: {why}")
    if "danger" in rows[0]["answers"] and "danger" in rows[0]["gold"]:
        risky = [r for r in rows if r["gold"]["danger"] == "yes"]
        alarm = [r for r in rows if r["answers"]["danger"]["noul"] >= SAFETY_AT]
        caught = [r for r in risky if r in alarm]
        print(f"\ndanger (alarm from p >= {SAFETY_AT}): caught {len(caught)} of {len(risky)}, false alarms "
              f"{len(alarm) - len(caught)} of {len(rows) - len(risky)}")
        for r in risky:
            if r not in alarm:
                print(f"  missed {r['id']} (p {r['answers']['danger']['noul']:.2f}): {r['state'][:100]}")
    wrong = Counter((r["gold"]["category"], predicted(r["answers"]["category"])) for r in rows
                    if predicted(r["answers"]["category"]) != r["gold"]["category"])
    if wrong:
        print("\ncategory mistakes (gold -> predicted): "
              + ", ".join(f"{g} -> {p} x{n}" for (g, p), n in wrong.most_common()))


async def answer_all(url, tickets, max_error, lang, concurrency, rules):
    sem = asyncio.Semaphore(concurrency)
    async with AsyncBasal(url, timeout=300) as b:
        async def one(t):
            async with sem:
                return {**t, "answers": await b.decide(t["state"], triage_questions(t["state"], lang, max_error, rules))}
        return await asyncio.gather(*(one(t) for t in tickets))


def calibration_lines(tickets, lang, rules):
    """The labelled /v1/systemone requests of basal-calibrate (docs/CALIBRATION.md): one per ticket, or with lang
    "both" two, the Polish and the English wording, in one group (they stay on one side of the FIT / CERT split) and
    with a family per language (the report then gives each wording its own calibration error)."""
    for t in tickets:
        for wording in (("pl", "en") if lang == "both" else (lang,)):
            qs = triage_questions(t["state"], wording, rules=rules)
            qs.pop("danger", None)  # a filter at SAFETY_AT, not a decision to certify
            for name, gold in t["gold"].items():
                if name not in qs:
                    continue
                qs[name]["gold"] = gold
                if lang == "both":
                    qs[name]["family"] = f"{name}/{wording}"
            line = {"id": t["id"], "state": t["state"], "questions": qs}
            yield {**line, "id": f"{t['id']}/{wording}", "group": t["id"]} if lang == "both" else line


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", default=str(HERE / "tickets.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--max-error", type=float, help="certified error target: answers below its threshold abstain")
    p.add_argument("--lang", choices=["pl", "en", "both"],
                   help="preset wording (default: the language of each ticket); both: the Polish and the English "
                        "wording of every ticket, with --calibration-input only")
    p.add_argument("--rules", action="store_true", help="the preset with this helpdesk's rules in the options (RULES)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", help="also write every ticket with its answers (JSON lines)")
    p.add_argument("--calibration-input", metavar="FILE",
                   help="write the tickets as labelled requests for basal-calibrate and exit (no server needed)")
    a = p.parse_args(argv)
    tickets = [json.loads(line) for line in Path(a.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.lang == "both" and not a.calibration_input:
        p.error("--lang both goes with --calibration-input: a run asks each ticket in one wording")
    if a.calibration_input:
        with open(a.calibration_input, "w", encoding="utf-8") as f:
            n = 0
            for n, line in enumerate(calibration_lines(tickets, a.lang, a.rules), 1):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{n} labelled requests in {a.calibration_input}", file=sys.stderr)
        return
    rows = asyncio.run(answer_all(a.url, tickets, a.max_error, a.lang, a.concurrency, a.rules))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows, a.max_error)


if __name__ == "__main__":
    main()
