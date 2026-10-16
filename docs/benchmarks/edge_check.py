"""Edge-case requests against a running basal-serve: sequential answers, the same requests all at once (batched by the
server), and in-request invariants. Saves the answers for cross-engine comparison.

  python docs/benchmarks/edge_check.py --url http://127.0.0.1:8000/v1/systemone --label gguf
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

PL_STATE = ("Klient Jan Kowalski zamówił 12.03.2026 r. pralkę za 2 499 zł. Dostawa miała nastąpić do 20.03.2026 r., "
            "ale kurier przyjechał 27.03.2026 r. Klient żąda obniżenia ceny o 300 zł i zwrotu kosztów dojazdu.")
EN_STATE = ("Order #5521: the customer paid 89 EUR for headphones on 2 May. The parcel was marked delivered on 5 May, "
            "but the customer says it never arrived and asks for a refund. No photo of the doorstep was taken.")
LONG_OPT = ("Sprawa wymaga przekazania do działu prawnego, ponieważ dotyczy roszczenia odszkodowawczego, którego "
            "wysokość przekracza próg samodzielnej decyzji konsultanta, a dokumentacja jest niekompletna")


def q_choice(instr, opts, keys="hide"):
    return {"type": "choice", "instructions": instr, "option_keys": keys,
            "criteria": {f"o{k}": o for k, o in enumerate(opts)}}


def requests():
    late = q_choice("Czy dostawa była opóźniona?", ["Tak, o tydzień", "Nie, na czas", "Brak danych"])
    rq = {
        "mixed-lang": {"state": PL_STATE, "questions": {
            "pl": late,
            "en": q_choice("Was the delivery late?", ["Yes, by a week", "No, on time", "Unknown"])}},
        "ten-options": {"state": PL_STATE, "questions": {"amount": q_choice(
            "Jaką obniżkę ceny żąda klient?",
            ["100 zł", "150 zł", "200 zł", "250 zł", "300 zł", "350 zł", "400 zł", "450 zł", "500 zł", "Nie żąda"])}},
        "duplicates": {"state": PL_STATE, "questions": {"a": late, "b": dict(late), "c": dict(late)}},
        "json-state": {"state": {"ticket": {"id": 88412, "text": "Czajnik przyszedł pęknięty, proszę o zwrot.",
                                            "channel": "email"}, "customer": {"segment": "VIP", "orders": [1, 2, 3]}},
                       "questions": {"refund": {"type": "noul", "instructions": "Czy klient prosi o zwrot pieniędzy?"},
                                     "vip": {"type": "noul", "instructions": "Czy klient jest VIP?"}}},
        "facts-auto": {"state": PL_STATE, "facts": "auto", "questions": {"days": q_choice(
            "Ile dni wyniosło opóźnienie dostawy?", ["3 dni", "7 dni", "14 dni"])}},
        "long-options": {"state": PL_STATE, "questions": {"route": q_choice(
            "Jak należy dalej poprowadzić sprawę?",
            [LONG_OPT, "Konsultant może sam przyznać obniżkę ceny o 300 zł zgodnie z regulaminem sklepu",
             "Należy odrzucić żądanie, ponieważ opóźnienie nie uzasadnia obniżki ceny ani zwrotu kosztów",
             "Należy poprosić klienta o dodatkowe dokumenty przed podjęciem jakiejkolwiek decyzji"])}},
        "score-5": {"state": PL_STATE, "questions": {"anger": {"type": "score", "instructions": "Jak bardzo klient jest niezadowolony?",
                                                              "criteria": ["wcale", "lekko", "średnio", "bardzo", "skrajnie"]}}},
        "multi-8": {"state": EN_STATE, "questions": {"tags": {
            "type": "multi", "instructions": "Which topics does the message cover?", "min": 1, "max": 3,
            "criteria": {"refund": "Refund", "delivery": "Delivery problem", "damage": "Damaged item", "price": "Price",
                         "invoice": "Invoice", "account": "Account access", "fraud": "Possible fraud", "warranty": "Warranty"}}}},
        "act-matrix": {"state": EN_STATE, "questions": {"decide": {
            "type": "act", "instructions": "Should the refund be approved?", "criteria": {"true": "Yes", "false": "No"},
            "costs": {"approve": {"true": 0, "false": 89}, "reject": {"true": 40, "false": 0},
                      "human": {"true": 5, "false": 5}}}}},
        "en-only": {"state": EN_STATE, "questions": {
            "lost": {"type": "noul", "instructions": "Does the customer claim the parcel never arrived?"},
            "photo": {"type": "noul", "instructions": "Is there photo evidence of delivery?"},
            "urgency": {"type": "score", "instructions": "How urgent is this case?", "criteria": ["low", "medium", "high"]}}},
        "tiny-state": {"state": "ok", "questions": {"q": {"type": "noul", "instructions": "Czy stan zawiera skargę?"}}},
        "keys-show-hide": {"state": PL_STATE, "questions": {
            "hide": q_choice("Czego żąda klient?", ["Obniżki ceny", "Wymiany towaru", "Odstąpienia od umowy"]),
            "show": {"type": "choice", "instructions": "Czego żąda klient?", "option_keys": "show",
                     "criteria": {"discount": "Obniżki ceny", "exchange": "Wymiany towaru", "withdraw": "Odstąpienia od umowy"}}}},
    }
    return rq


def probs(r):
    return {k: v.get("probabilities") for k, v in r["answers"].items()}


async def main(a):
    rq = requests()
    out = {"label": a.label, "sequential": {}, "concurrent": {}, "checks": {}}
    async with httpx.AsyncClient(timeout=1200) as c:
        async def post(b):
            r = await c.post(a.url, json=b)
            if r.status_code != 200:
                return {"error": r.text[:300]}
            return r.json()
        await post(rq["tiny-state"])
        for name, b in rq.items():
            out["sequential"][name] = await post(b)
        res = await asyncio.gather(*[post(b) for b in rq.values()])
        out["concurrent"] = dict(zip(rq, res))
    seq, con = out["sequential"], out["concurrent"]
    errors = {n: r["error"] for n, r in seq.items() if "error" in r}
    errors.update({f"{n} (concurrent)": r["error"] for n, r in con.items() if "error" in r})
    dmax = 0.0
    for n in rq:
        if "error" in seq[n] or "error" in con[n]:
            continue
        for q, p in probs(seq[n]).items():
            dmax = max(dmax, max(abs(p[x] - probs(con[n])[q][x]) for x in p))
    dup = probs(seq["duplicates"]) if "error" not in seq["duplicates"] else None
    dup_d = max(abs(dup["a"][x] - dup[k][x]) for k in ("b", "c") for x in dup["a"]) if dup else None
    tags = seq["multi-8"]["answers"]["tags"] if "error" not in seq["multi-8"] else None
    out["checks"] = {
        "errors": errors,
        "concurrent_vs_sequential_max_dp": dmax,
        "duplicate_questions_max_dp": dup_d,
        "multi_selected_within_min_max": None if tags is None else 1 <= len(tags["selected"]) <= 3,
        "act_action": seq["act-matrix"]["answers"]["decide"].get("action") if "error" not in seq["act-matrix"] else None,
        "branches": {n: r["usage"]["branches"] for n, r in seq.items() if "usage" in r},
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.label}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(a.label, json.dumps(out["checks"], ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results/edge")
    asyncio.run(main(ap.parse_args()))
