"""Helpdesk triage on labelled tickets: the `triage` preset (urgency, category, needs_human) against a running
basal-serve, scored against the gold labels, with a certified error target that sends uncertain answers to a person.

  basal-serve --model Remek/basal-2.0-mini-MLX-8bit --mode mlx --port 8000     # or any model and mode
  python examples/triage/run.py --url http://127.0.0.1:8000 --max-error 0.05
  python examples/triage/run.py --url http://127.0.0.1:8000 --rules             # the preset with RULES in the options
  python examples/triage/run.py --calibration-input labelled.jsonl             # requests for basal-calibrate

Input lines: {"id", "state", "gold": {"category": key, "urgency": 0-2, "needs_human": "yes" | "no"}}.
"""
import argparse
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
        "category": {"complaint": "reklamacja, skarga, wadliwy lub niezgodny towar, problem z dostawą lub obsługą",
                     "billing": "płatności, faktury, ceny, zwroty pieniędzy, karty płatnicze",
                     "technical": "błąd lub awaria aplikacji, strony, API albo integracji",
                     "account": "konto klienta: logowanie, hasło, kody weryfikacyjne, dane, adres, użytkownicy, "
                                "usunięcie konta",
                     "other": "coś innego: oferta, współpraca, pytanie ogólne, podziękowanie"},
        "urgency": ["niska: pytanie lub prośba bez terminu, nic nie przestało działać",
                    "średnia: coś nie działa albo klient traci pieniądze, ale może poczekać dzień",
                    "wysoka: awaria blokująca pracę lub sprzedaż, włamanie, klient potrzebuje pomocy teraz"],
        "needs_human": {"true": "pracownik musi podjąć decyzję: zwrot pieniędzy, korekta, reklamacja, wyjątek, "
                                "sprawa prawna, bezpieczeństwo konta, awaria",
                        "false": "wystarczy gotowa odpowiedź, instrukcja lub zwykła zmiana danych"},
    },
    "en": {
        "category": {"complaint": "a complaint, a faulty or wrong product, a delivery or service problem",
                     "billing": "payments, invoices, prices, refunds, payment cards",
                     "technical": "an error or outage of the app, the website, the API or an integration",
                     "account": "the customer account: login, password, verification codes, data, address, users, "
                                "closing the account",
                     "other": "something else: an offer, a partnership, a general question, thanks"},
        "urgency": ["low: a question or request with no deadline, nothing stopped working",
                    "medium: something does not work or the customer loses money, but it can wait a day",
                    "high: an outage that blocks work or sales, a break-in, the customer needs help now"],
        "needs_human": {"true": "a person must decide: a refund, a correction, a complaint, an exception, a legal "
                                "matter, account security, an outage",
                        "false": "a ready answer, an instruction or a plain change of data is enough"},
    },
}


def triage_questions(state, lang=None, max_error=None, rules=False):
    """The triage preset's questions about state, with RULES in the options when rules is set."""
    qs = questions("triage", lang, state, max_error)
    if rules:
        for name, criteria in RULES[lang or lang_of(state)].items():
            qs[name]["criteria"] = criteria
    return qs


def predicted(answer):
    """The answer of a choice / score / noul question as a gold label: a key, a level index or "yes" / "no"."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "score":
        return int(max(answer["probabilities"], key=answer["probabilities"].get))
    return "yes" if answer["noul"] >= 0.5 else "no"


def report(rows, max_error):
    names = list(rows[0]["gold"])
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
    """One labelled /v1/systemone request per ticket, the input of basal-calibrate (docs/CALIBRATION.md)."""
    for t in tickets:
        qs = triage_questions(t["state"], lang, rules=rules)
        for name, gold in t["gold"].items():
            qs[name]["gold"] = gold
        yield {"id": t["id"], "state": t["state"], "questions": qs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", default=str(HERE / "tickets.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--max-error", type=float, help="certified error target: answers below its threshold abstain")
    p.add_argument("--lang", choices=["pl", "en"], help="preset wording (default: the language of each ticket)")
    p.add_argument("--rules", action="store_true", help="the preset with this helpdesk's rules in the options (RULES)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", help="also write every ticket with its answers (JSON lines)")
    p.add_argument("--calibration-input", metavar="FILE",
                   help="write the tickets as labelled requests for basal-calibrate and exit (no server needed)")
    a = p.parse_args(argv)
    tickets = [json.loads(line) for line in Path(a.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.calibration_input:
        with open(a.calibration_input, "w", encoding="utf-8") as f:
            for line in calibration_lines(tickets, a.lang, a.rules):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{len(tickets)} labelled requests in {a.calibration_input}", file=sys.stderr)
        return
    rows = asyncio.run(answer_all(a.url, tickets, a.max_error, a.lang, a.concurrency, a.rules))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows, a.max_error)


if __name__ == "__main__":
    main()
