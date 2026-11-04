"""Test vectors of prompt format 2 (docs/format2_vectors.jsonl, described in docs/FORMAT2.md): one line per request,
with every prompt the engine renders for it, its token ids, where the state ends and the option code ids. All states
and questions are synthetic.

    python docs/format2_vectors.py TOKENIZER_DIR docs/format2_vectors.jsonl
    python docs/format2_vectors.py MODEL_DIR docs/format2_vectors.jsonl --model     # + fp32 probabilities per prompt
    python docs/format2_vectors.py MODEL_DIR docs/format2_vectors.jsonl --url http://localhost:8000   # + answers

TOKENIZER_DIR: any basal model directory (the format-2 models use the tokenizer and chat template of the published
basal models). Without --model / --url, `expected` is left empty. --model fills expected.probabilities_fp32 (the raw
readout of one fp32 forward per prompt, before any temperature); --url fills expected.answers of the requests whose
`orders` is the server's --orders (run it once per value)."""
import argparse
import json
import os
import random
from types import SimpleNamespace

from transformers import AutoTokenizer

from basal.prompt import CODES, PREFILL, letter_ids, render
from basal.server import Server, to_items

PL_STATE = ("Klient: Zamówiłem laptopa trzy tygodnie temu, przyszedł z pękniętym ekranem. Kurier zostawił paczkę pod "
            "drzwiami. Chcę zwrotu pieniędzy, nie naprawy.")
EN_STATE = ("Customer: I ordered a laptop three weeks ago and it arrived with a cracked screen. The courier left the "
            "parcel at the door. I want my money back, not a repair.")
TOOLS = [
    {"type": "function", "function": {"name": "get_order", "description": "Look up an order by its number: status, "
                                      "items, payment.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                                                      "fields": {"type": "array"}},
                                                     "required": ["order_id"]}}},
    {"name": "refund", "description": "Refund a paid order, fully or partially.",
     "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"},
                                                      "reason": {"type": "string"}}, "required": ["order_id", "amount"]}},
    {"name": "send_message", "description": "Send a message to the customer."},
]
FORMS = [  # one tool in each accepted definition form
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                      "units": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "name": "get_time", "description": "Local time in a time zone.",
     "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}},
    {"name": "search_docs", "description": "Search the product documentation.\n\n  Returns   the top passages.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                     "required": ["query"]}},
    {"name": "create_ticket", "description": "Open a support ticket.",
     "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}}},
    {"name": "ping"},
]
LONG_DESC = ("Create a return label for a parcel. " + "The label is valid for 14 days and can be printed or shown as a "
             "QR code at any drop-off point; it covers parcels up to 30 kg and 120 cm on the longest side. " * 4)


def many(n, lang):
    word = "Kategoria" if lang == "pl" else "Category"
    return {f"c{i}": f"{word} {i + 1}" for i in range(n)}


def long_state(tok, target):
    """A synthetic Polish order-service log of about `target` tokens (deterministic)."""
    rnd = random.Random(7)
    events = ["klient zgłosił opóźnienie dostawy zamówienia {o}", "kurier oznaczył przesyłkę {o} jako doręczoną",
              "magazyn potwierdził wysyłkę zamówienia {o} ({n} szt.)", "płatność {k} zł za zamówienie {o} zaksięgowana",
              "klient zapytał o fakturę do zamówienia {o}", "zwrot {k} zł do zamówienia {o} zlecony",
              "konsultant dodał notatkę: klient prosi o kontakt telefoniczny", "system wysłał przypomnienie o "
              "płatności za zamówienie {o}", "klient zmienił adres dostawy zamówienia {o}",
              "reklamacja zamówienia {o} przyjęta do rozpatrzenia"]
    lines = []
    while True:
        d, h, m = rnd.randrange(1, 29), rnd.randrange(7, 22), rnd.randrange(60)
        e = rnd.choice(events).format(o=rnd.randrange(10000, 99999), n=rnd.randrange(1, 9), k=rnd.randrange(20, 4000))
        lines.append(f"2026-03-{d:02d} {h:02d}:{m:02d} — {e}")
        text = "Historia obsługi klienta (konto 48213):\n" + "\n".join(lines) + (
            "\nOstatnia wiadomość klienta: Laptop z zamówienia 51120 przyszedł z pękniętym ekranem. Chcę zwrotu "
            "pieniędzy.")
        if len(tok(text, add_special_tokens=False).input_ids) >= target:
            return text


def cases(tok):
    q = lambda **kw: kw  # noqa: E731
    one = lambda name, state, question, covers, orders=1: (name, covers, {"state": state, "questions": {"q": question}},
                                                          orders)
    longs = long_state(tok, 4000)
    return [
        one("pl-noul", PL_STATE, q(type="noul", instructions="Czy towar dotarł uszkodzony?"), "yes/no, Polish"),
        one("en-noul", EN_STATE, q(type="noul", instructions="Did the goods arrive damaged?"), "yes/no, English"),
        one("en-noul-2-orders", EN_STATE, q(type="noul", instructions="Did the goods arrive damaged?"),
            "yes/no in two option orders", 2),
        one("pl-noul-criteria", PL_STATE, q(type="noul", instructions="Czy klient żąda zwrotu pieniędzy?",
                                            criteria={"true": "Tak — żąda zwrotu", "false": "Nie — chce naprawy"}),
            "yes/no with the request's option texts"),
        one("en-noul-structured", EN_STATE, q(type="noul", instructions={"task": "Is a refund requested?"},
                                              criteria={"true": {"means": "refund"}, "false": "anything else"}),
            "structured question and option text (JSON)"),
        one("pl-choice-2", PL_STATE, q(type="choice", instructions="Czego chce klient?",
                                       criteria={"zwrot": "Zwrotu pieniędzy", "naprawa": "Naprawy"},
                                       option_keys="hide"), "choice, 2 options, keys hidden"),
        one("pl-choice-3-show", PL_STATE, q(type="choice", instructions="Który dział obsłuży zgłoszenie?",
                                            criteria={"zwroty": "Zwroty i reklamacje", "serwis": "Serwis", "it": "IT"}),
            "choice, keys shown (`key: description`)"),
        one("pl-choice-3-2-orders", PL_STATE, q(type="choice", instructions="Który dział obsłuży zgłoszenie?",
                                                criteria={"zwroty": "Zwroty i reklamacje", "serwis": "Serwis",
                                                          "it": "IT"}), "choice in two option orders", 2),
        one("en-choice-3-hide", EN_STATE, q(type="choice", instructions="Which department should handle the ticket?",
                                            criteria={"returns": "Returns and complaints", "service": "Repairs",
                                                      "it": "IT"}, option_keys="hide"), "choice, keys hidden"),
        one("en-choice-hide-duplicates", EN_STATE, q(type="choice", instructions="Which queue?",
                                                     criteria={"q1": "Returns", "q2": "Returns", "q3": "Repairs"},
                                                     option_keys="hide"),
            "keys hidden, duplicate descriptions keep their keys"),
        one("en-choice-keys", EN_STATE, q(type="choice", instructions="What does the customer want?",
                                          criteria=["refund", "repair", "replacement"]), "choice, bare keys (a list)"),
        one("en-choice-default-type", EN_STATE, q(instructions="What does the customer want?",
                                                  criteria={"refund": None, "repair": "A repair of the laptop"}),
            "no type (choice), a key without a description"),
        one("en-choice-10", EN_STATE, q(type="choice", instructions="Which category does the ticket belong to?",
                                        criteria=many(10, "en"), option_keys="hide"), "choice, 10 options"),
        one("pl-choice-26", PL_STATE, q(type="choice", instructions="Do której kategorii należy zgłoszenie?",
                                        criteria=many(26, "pl"), option_keys="hide"), "choice, 26 options (A-Z)"),
        one("en-choice-27", EN_STATE, q(type="choice", instructions="Which category does the ticket belong to?",
                                        criteria=many(27, "en"), option_keys="hide"),
            "choice, 27 options (the first two-character code)"),
        one("pl-choice-151", PL_STATE, q(type="choice", instructions="Do której kategorii należy zgłoszenie?",
                                         criteria=many(151, "pl"), option_keys="hide"), "choice, 151 options"),
        one("en-choice-255", EN_STATE, q(type="choice", instructions="Which category does the ticket belong to?",
                                         criteria=many(255, "en"), option_keys="hide"), "choice, 255 options (all codes)"),
        one("en-choice-255-2-orders", EN_STATE, q(type="choice", instructions="Which category does the ticket belong "
                                                  "to?", criteria=many(255, "en"), option_keys="hide"),
            "choice, 255 options in two option orders", 2),
        one("pl-score", PL_STATE, q(type="score", instructions="Jak pilne jest zgłoszenie?",
                                    criteria=["niska", "średnia", "wysoka", "krytyczna"]), "score, listed levels"),
        one("en-score-named", EN_STATE, q(type="score", instructions="How angry is the customer?",
                                          criteria={"calm": "Calm", "annoyed": "Annoyed", "angry": "Angry"}),
            "score, named levels (keys shown)"),
        one("en-score-levels", EN_STATE, q(type="score", instructions="How complete is the customer's report?",
                                           levels=["nothing useful", "some details", "most details",
                                                   "all details", "all details and evidence"]),
            "score, `levels`, 5 levels"),
        one("en-multi", EN_STATE, q(type="multi", instructions="Which issues does the message raise?",
                                    criteria={"damage": "Damaged goods", "delivery": "Delivery problem",
                                              "refund": "Refund request"}), "multi, threshold"),
        one("en-multi-2-orders", EN_STATE, q(type="multi", instructions="Which issues does the message raise?",
                                             criteria={"damage": "Damaged goods", "delivery": "Delivery problem"}),
            "multi in two option orders", 2),
        one("pl-multi-fallback", PL_STATE, q(type="multi", instructions="Które kategorie dotyczą zgłoszenia?",
                                             criteria={"uszk": "Uszkodzenie towaru", "dost": "Problem z dostawą",
                                                       "ogolne": "Sprawa ogólna"}, fallback="ogolne"),
            "multi, fallback label (stored or default threshold)"),
        one("en-multi-fallback-threshold", EN_STATE, q(type="multi", instructions="Which teams must act?",
                                                       criteria={"logistics": "Logistics", "finance": "Finance",
                                                                 "none": "No team"}, fallback="none",
                                                       fallback_threshold=0.3), "multi, fallback with its threshold"),
        one("en-multi-iou-exclusive", EN_STATE, q(type="multi", instructions="Which tags apply?",
                                                  criteria={"hardware": "Hardware fault", "courier": "Courier issue",
                                                            "refund": "Refund", "spam": "Spam or abuse"},
                                                  objective="iou", exclusive=["spam"], max=2),
            "multi, expected IoU, an exclusive label, max"),
        one("en-multi-f1-keys", EN_STATE, q(type="multi", instructions="Which apply?",
                                            criteria=["refund", "repair", "replacement"], objective="f1", min=1),
            "multi, expected F1, bare keys as labels, min"),
        one("en-tool-3-extras", EN_STATE, q(type="tool", tools=TOOLS, rank=3),
            "tool, three tools (OpenAI and MCP forms) and the three extra options"),
        one("pl-tool", "Użytkownik: Gdzie jest moja paczka? Numer zamówienia 5512.",
            q(type="tool", tools=TOOLS, instructions="Które narzędzie wywołać?"), "tool, Polish question"),
        one("pl-tool-default-question", PL_STATE, q(type="tool", tools=TOOLS, extra=["ask_user", "none"]),
            "tool, Polish default question, two extras in the request's order"),
        one("en-tool-forms", "User: what's the weather in Gdansk right now?",
            q(type="tool", tools=FORMS, extra=[]),
            "tool, every definition form, no schema, whitespace in a description, no extras"),
        one("en-tool-long-description", EN_STATE, q(type="tool", extra=["ask_user"],
                                                    tools=TOOLS + [{"name": "create_return_label",
                                                                    "description": LONG_DESC}]),
            "tool, a description cut at 300 characters"),
        one("en-tool-description-chars", EN_STATE, q(type="tool", description_chars=40, extra=["answer_directly"],
                                                     tools=TOOLS + [{"name": "create_return_label",
                                                                     "description": LONG_DESC}]),
            "tool, description_chars 40"),
        one("en-tool-multi", EN_STATE, q(type="tool", tools=TOOLS, multi=True), "tool set (multi), no extras"),
        one("pl-tool-multi-extras", PL_STATE, q(type="tool", tools=TOOLS, multi=True, extra=["ask_user"]),
            "tool set, Polish, an extra (selected only alone)"),
        one("pl-act", PL_STATE, q(type="act", instructions="Czy zwrot przysługuje?", costs={"wrong": 50, "defer": 2}),
            "act on a yes/no question, short cost form"),
        one("en-act-choice", EN_STATE, q(type="act", instructions="What should we do with the claim?",
                                         criteria={"approve": "Approve the refund", "reject": "Reject the claim"},
                                         costs={"approve": {"approve": 0, "reject": 20},
                                                "reject": {"approve": 10, "reject": 0},
                                                "human": {"approve": 1, "reject": 1}}, max_error=0.01),
            "act on named outcomes, full cost matrix, max_error"),
        one("en-state-json", {"order": 5512, "status": "delivered", "notes": ["screen cracked", "left at the door"]},
            q(type="noul", instructions="Was the order delivered?"), "structured state (JSON)"),
        one("en-state-braces", 'Log: {"error": "E42", "msg": "Payment {id} failed"}\n\nUser: "why?"',
            q(type="noul", instructions="Did a payment fail?"), "a state with braces, quotes and blank lines"),
        one("lang-from-question", "Order 5512: delivered, screen cracked.",
            q(type="noul", instructions="Czy ekran jest pęknięty?"), "English state, Polish question: Polish prompt"),
        ("en-soam-3", "three questions over one state (state once, ask many)",
         {"state": EN_STATE, "questions": {
             "damaged": q(type="noul", instructions="Did the goods arrive damaged?"),
             "wants": q(type="choice", instructions="What does the customer want?",
                        criteria=["refund", "repair", "replacement"]),
             "urgency": q(type="score", instructions="How urgent is the ticket?", criteria=["low", "medium", "high"])}},
         1),
        ("soam-mixed-language", "two questions over one state in two languages (different prompt prefixes)",
         {"state": "Order 5512: delivered, screen cracked.", "questions": {
             "en": q(type="noul", instructions="Is the screen cracked?"),
             "pl": q(type="noul", instructions="Czy ekran jest pęknięty?")}}, 1),
        ("pl-long-state-5q", "a state of about 4,000 tokens with five questions",
         {"state": longs, "questions": {
             "zwrot": q(type="noul", instructions="Czy klient żąda zwrotu pieniędzy?"),
             "dzial": q(type="choice", instructions="Który dział obsłuży zgłoszenie?",
                        criteria={"zwroty": "Zwroty i reklamacje", "serwis": "Serwis", "ksiegowosc": "Księgowość"}),
             "pilnosc": q(type="score", instructions="Jak pilne jest zgłoszenie?",
                          criteria=["niska", "średnia", "wysoka"]),
             "tematy": q(type="multi", instructions="Które tematy występują w historii?",
                         criteria={"dostawa": "Dostawa", "platnosc": "Płatności", "reklamacja": "Reklamacje"}),
             "narzedzie": q(type="tool", tools=TOOLS)}}, 1),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--model", action="store_true", help="fill expected.probabilities_fp32 (needs the weights, CUDA)")
    ap.add_argument("--url", help="fill expected.answers from this basal-serve")
    ap.add_argument("--orders", type=int, default=1, help="--url: the server's --orders (only those requests get answers)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = None
    if a.model:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.tokenizer, dtype=torch.float32).cuda().eval()
    srv = SimpleNamespace(tok=tok, fmt=2)  # Server.state_cut needs the tokenizer and the format
    table = None
    n = 0
    part = f"{a.out}.part"  # the vectors file is replaced only when complete (a failed run kept a truncated one)
    with open(part, "w", encoding="utf-8") as f:
        for name, covers, request, orders in cases(tok):
            prompts = []
            for item in to_items(request["state"], request["questions"], len(CODES)):
                multi = item["type"] == "multi" or item.get("multi")
                for bi, (question, options) in enumerate(item["branches"]):
                    k = len(options)
                    for perm in [list(range(k)), list(range(k))[::-1]][:orders]:
                        opts = [options[c] for c in perm]
                        prompt = render(tok, item["state"], question, opts, item["lang"], fmt=2)
                        ids = tok(prompt, add_special_tokens=False).input_ids
                        codes = letter_ids(tok, prompt, k, fmt=2)
                        if table is None:
                            table = letter_ids(tok, prompt, len(CODES), fmt=2)
                        assert prompt.endswith(PREFILL) and codes == table[:k]  # the code ids depend only on k
                        cut = Server.state_cut(srv, dict(item, question=question, options=opts))
                        rec = {"question": item["name"], "label": item["keys"][bi] if multi else None,
                               "lang": item["lang"], "option_order": perm, "options": opts, "prompt": prompt,
                               "input_ids": ids, "state_end": cut, "code_ids": codes}
                        if model is not None:
                            with torch.no_grad():
                                logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()
                            rec["probabilities_fp32"] = torch.softmax(logits[codes], -1).tolist()
                        prompts.append(rec)
            expected = {}
            if model is not None:
                expected["probabilities_fp32"] = [p.pop("probabilities_fp32") for p in prompts]
            if a.url and orders == a.orders:
                import urllib.request
                req = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", json.dumps(request).encode(),
                                             {"content-type": "application/json"})
                expected["answers"] = json.loads(urllib.request.urlopen(req).read())["answers"]
            f.write(json.dumps({"id": name, "covers": covers, "request": request, "orders": orders,
                                "prefill": PREFILL, "prompts": prompts, "expected": expected},
                               ensure_ascii=False) + "\n")
            n += 1
            print(name, len(prompts), "prompts", max(len(p["input_ids"]) for p in prompts), "tokens", flush=True)
    os.replace(part, a.out)
    print(n, "requests ->", a.out)


if __name__ == "__main__":
    main()
