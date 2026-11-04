"""Fact injection: helper facts computed from a Polish state and appended to it, so the
model reads the arithmetic instead of doing it. The same function runs in training (on 50% of items) and in the
engine (`"facts": "auto"`), so the model works with or without them. Only arithmetic and calendar facts are added —
never a rule or an answer:
  - every date in the text: weekday, and whether it is a Saturday or a statutory day off (with the holiday name);
  - day gaps between consecutive dates (in order of appearance);
  - "today minus N days" for "N dni temu" when the text states today's date;
  - date + duration for the durations the text mentions ("14 dni", "2 tygodnie", "6 miesięcy", "5 lat"), for the
    first dates (calendar arithmetic only: no shifting for days off — that is a rule);
  - net/gross amounts for the VAT rates mentioned (23%, 8%, 5%), for the first amounts in zł; an amount
    without a thousands separator ("2026,00 zł") is read whole, an exchange rate ("kurs … 4,31 zł") is skipped, and a
    percentage is a VAT rate only with "VAT" / "podatek od towarów" in its sentence (not "5% wartości sporu").
Money arithmetic, appended after the facts above, at most MAX_TOTAL lines in all:
  - an amount in euro × the exchange rate the text states ("2 000 000 euro × 4,3770 zł = 8 754 000,00 zł"); with
    several different rates only where the sentence of the euro amount names exactly one;
  - budget minus what was spent ("budżet 5 000,00 zł − wykorzystano 3 750,00 zł = 1 250,00 zł");
  - quantity × unit price ("3 × 465,50 zł = 1 396,50 zł") and the total of the lines of one list;
  - a quantity-tiered price list: the tier the ordered quantity falls in and its value ("20 ≤ 98 ≤ 99; 98 × 156,60 zł
    = 15 346,80 zł"), plus the delivery fee the text names ("15 346,80 zł + 39,00 zł = 15 385,80 zł");
  - net/gross of the amounts the text labels "netto"/"brutto" (and of computed totals), for the one VAT rate stated,
    where the VAT line above does not already give it ("26 016,26 zł netto × 1,23 = 32 000,00 zł brutto");
  - comparisons of those amounts with each other and with the other zł amounts of the text (at most MAX_COMPARE;
    unit prices, fees, rates and the operands of a remainder are left out): "9 654 410,27 zł > 8 754 000,00 zł".
    Symbols only (>, <, =): never "przekracza", "mieści się", "w terminie" — which amount matters is the model's call.

  from basal.facts import inject
  state2 = inject(state)              # the same function adds the facts in training
"""
import re
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext

MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
          "października", "listopada", "grudnia"]
WEEKDAYS = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
HEADER = "Fakty pomocnicze (wyliczone automatycznie, bez oceny prawnej):"
MAX_DATES, MAX_DURATIONS, MAX_AMOUNTS = 6, 3, 3
MAX_CHARS = 1_000_000  # basal-serve's "facts": "auto" reads a state of at most this many characters (HTTP 422 beyond)

DATE_TXT = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})(?: r\.)?")
DATE_NUM = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DURATION = re.compile(r"\b(\d{1,3}) (dni|dnia|dzień|tygodni|tygodnie|tydzień|miesięcy|miesiące|miesiąc|lat|lata|rok)\b")
DUR_WORD = re.compile(r"\b(tydzień|tygodnia|miesiąc|miesiąca|rok|roku)\b")
YEAR_BEFORE = re.compile(r"\d{4}(?:-\d{2}-\d{2})?\s+$")  # "2025 roku", "2025-03-03 roku": part of a date
AGO = re.compile(r"\b(\d{1,3}) (dni|dzień) temu\b")
TODAY = re.compile(r"\b(?:[Dd]ziś|[Dd]zisiaj)(?: jest| mamy|:)? ")
# Not the start of an amount: a digit group whose group before it is itself a group of the same run ("… 111 111").
# The run's earlier group reads the same groups to the same end, so the amount was found there or does not exist;
# without this, every group of a long run with no "zł" after it rescanned the run to its end (quadratic: 40 kB of
# "111 " took 14 s).
IN_RUN = r"(?!(?<=[ \u00a0]\d{3}[ \u00a0])\d{3})"
IN_RUN_SP = r"(?!(?<=[  ]\d{3}[  ])\d{3})"  # the same for AMOUNT, whose groups are separated by a space only
AMOUNT = re.compile(r"(?<![\d,.])\b(" + IN_RUN_SP + r"\d{1,3}(?:[  ]\d{3})+(?:,\d{2})?|\d{1,3}(?:\.\d{3})+(?:,\d{2})?"
                    r"|\d+(?:,\d{2})?) zł\b")
VAT = re.compile(r"(?<![\d,.])\b(23|8|5) ?%")
VAT_CTX = re.compile(r"\bVAT\b|\bpodat\w* od towarów")

# money arithmetic
MAX_TOTAL, MAX_COMPARE, MAX_PLAIN, MAX_EURO = 12, 6, 3, 3
AMT = IN_RUN + r"\d{1,3}(?:[ \u00a0]\d{3})*(?:,\d{2})?"
STATED = re.compile(r"(?<![\d,.])\b(" + IN_RUN + r"\d{1,3}(?:[ \u00a0]\d{3})+(?:,\d{2})?|\d{1,3}(?:\.\d{3})+(?:,\d{2})?"
                    r"|\d+(?:,\d{2})?)"
                    r"[ \u00a0]?(?:zł|PLN)\b")
EURO_AMT = r"(?<![\d,])\b(" + IN_RUN + r"\d{1,3}(?:[ \u00a0]\d{3})*(?:,\d{1,2})?)[ \u00a0]?(?:euro\b|EUR\b|€)"
EURO = re.compile(EURO_AMT)
RATE = r"(\d{1,2},\d{2,4})[ \u00a0]?(?:zł|PLN)\b"
RATE_ANY = re.compile(r"(?<![\d,])\b" + RATE)
RATE_PER = re.compile(r"(?<![\d,])\b" + RATE + r"[ \u00a0]?(?:za|/)[ \u00a0]?(?:1[ \u00a0])?(?:euro\b|EUR\b|€)")
RATE_EQ = re.compile(r"\b1[ \u00a0]?(?:euro\b|EUR\b|€)[ \u00a0]?=[ \u00a0]?" + RATE)
EURO_X_RATE = re.compile(EURO_AMT + r"[ \u00a0]?[×*][ \u00a0]?" + RATE)
KURS = re.compile(r"\b[Kk]urs\w*")
BOUND = re.compile(r"[.;!?](?=\s+[A-ZĄĆĘŁŃÓŚŹŻ„\"(])|\n")
LABEL = re.compile(r"[ \u00a0]?\(?(netto|brutto)\b")
BUDGET = re.compile(r"\b[Bb]udżet\w*")
USED = re.compile(r"\b(wykorzystano|wydano|wydatkowano|zużyto|zaangażowano|rozdysponowano)\b")
UNIT = r"(?:szt\.|sztuk[ia]?|usł\.|godz\.|kg|op\.|opak\.|kpl\.|os\.)"
PRODUCT = re.compile(r"(?<![\d,])(?<!\d[ \u00a0])\b(\d{1,4})[ \u00a0]?(?:" + UNIT + r"[ \u00a0]?)?×[ \u00a0]?(?=\d)")
TIER = re.compile(r"(?<![\d,])\b(" + AMT + r") zł(?: netto| brutto)?(?: za (?:sztukę|szt\.))?(?: przy (?:zamówieniu|zakupie))?"
                  r" (poniżej|do|od|powyżej|co najmniej|min\.) (\d{1,5})(?: do (\d{1,5})|[ \u00a0]?[–-][ \u00a0]?(\d{1,5}))?"
                  r" (?:szt\.|sztuk)")
QTY = re.compile(r"(?<![\d,])(?<!\d[ \u00a0])\b(\d{1,5})[ \u00a0](?:szt\.|sztuk[ia]?\b)")
QTY_PRE = ("poniżej ", "od ", "do ", "powyżej ", "najmniej ", "min. ", "–", "-")
EQ_NEXT = re.compile(r"\s*=")  # "1 EUR =": the euro of a rate
TIMES_NEXT = re.compile(r"[ \u00a0]?×")  # "5 szt. × 12,00 zł": a product line, not the ordered quantity
FEE = re.compile(r"\b(?:[Dd]ostawa|[Ww]ysyłka|[Tt]ransport|[Pp]rzesyłka|[Kk]oszty? (?:dostawy|wysyłki|transportu|przesyłki))\b")


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    return date(y, mo, (h + l - 7 * m + 114) % 31 + 1)


def holidays(y):
    """Statutory days off in Poland (ustawa z 18 stycznia 1951 r.; 24 December from 2025)."""
    e = _easter(y)
    h = {date(y, 1, 1): "Nowy Rok", date(y, 1, 6): "Trzech Króli", e: "Wielkanoc",
         e + timedelta(days=1): "Poniedziałek Wielkanocny", date(y, 5, 1): "Święto Pracy",
         date(y, 5, 3): "Święto Konstytucji 3 Maja", e + timedelta(days=49): "Zielone Świątki",
         e + timedelta(days=60): "Boże Ciało", date(y, 8, 15): "Wniebowzięcie NMP",
         date(y, 11, 1): "Wszystkich Świętych", date(y, 11, 11): "Święto Niepodległości",
         date(y, 12, 25): "Boże Narodzenie", date(y, 12, 26): "drugi dzień Bożego Narodzenia"}
    if y >= 2025:
        h[date(y, 12, 24)] = "Wigilia Bożego Narodzenia"
    return h


def fmt(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year} r."


def add_months(d, k):
    y, m = divmod(d.year * 12 + d.month - 1 + k, 12)
    m += 1
    last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day
    return date(y, m, min(d.day, last))


def _dates(text):
    """Dates in order of first appearance, deduplicated."""
    found = []
    for rx, conv in ((DATE_TXT, lambda g: (int(g[2]), MONTHS.index(g[1]) + 1, int(g[0]))),
                     (DATE_NUM, lambda g: (int(g[2]), int(g[1]), int(g[0]))),
                     (DATE_ISO, lambda g: (int(g[0]), int(g[1]), int(g[2])))):
        for m in rx.finditer(text):
            try:
                found.append((m.start(), date(*conv(m.groups()))))
            except ValueError:
                pass
    return list(dict.fromkeys(d for _, d in sorted(found)))


def _durations(text):
    out = {}  # (n, unit) in order of first appearance
    for m in DURATION.finditer(text):
        n, u = int(m.group(1)), m.group(2)
        unit = "d" if u.startswith("d") else "w" if u.startswith("ty") else "m" if u.startswith("mie") else "y"
        if not text[m.end():m.end() + 5].startswith(" temu"):
            out.setdefault((n, unit))
    for m in DUR_WORD.finditer(text):  # "w terminie tygodnia", "miesiąca", "roku" = 1 unit
        if YEAR_BEFORE.search(text[max(0, m.start() - 20):m.start()]):
            continue
        out.setdefault((1, {"ty": "w", "mi": "m", "ro": "y"}[m.group(1)[:2]]))
    return list(out)[:MAX_DURATIONS]


def _plus(d, n, unit):
    """d + n units, or None when the result is outside date.min..date.max (year 1..9999)."""
    try:
        if unit == "d":
            return d + timedelta(days=n)
        if unit == "w":
            return d + timedelta(weeks=n)
        if unit == "m":
            return add_months(d, n)
        try:
            return d.replace(year=d.year + n)
        except ValueError:
            return d.replace(year=d.year + n, day=28)
    except (OverflowError, ValueError):
        return None


UNIT_PL = {"d": ("dzień", "dni"), "w": ("tydzień", "tygodnie"), "m": ("miesiąc", "miesiące"), "y": ("rok", "lata")}


def _pl_amount(x):
    with localcontext() as c:  # long amounts (> 26 digits) would exceed the default precision of quantize
        c.prec = 60
        q = x.quantize(Decimal("0.01"), ROUND_HALF_UP)
    s = f"{q:,.2f}".replace(",", " ").replace(".", ",")
    return f"{s} zł"


def _dec(raw):
    return Decimal(raw.replace(" ", "").replace("\u00a0", "").replace(".", "").replace(",", "."))


def _q(x):
    with localcontext() as c:
        c.prec = 60
        return x.quantize(Decimal("0.01"), ROUND_HALF_UP)


def _pl_num(x):
    """A number in Polish notation without a unit: 2 000 000, 139 000,50."""
    s = f"{x:,.0f}" if x == x.to_integral_value() else f"{_q(x):,.2f}"
    return s.replace(",", " ").replace(".", ",")


def _cut(text, a, n):
    """Window of n characters from a, up to the end of the sentence."""
    w = text[a:a + n]
    b = BOUND.search(w)
    return w[:b.start()] if b else w


def _rates(text):
    """Exchange rates the text states, as [(position, "4,3770")] (1 < rate < 10): "kurs … 4,3770 zł", "4,2586 zł za
    1 euro", "1 EUR = 4,31 zł", "216 000 euro × 4,31 zł"; and the explicit euro × rate pairs {euro position: rate}."""
    rates = []
    for m in KURS.finditer(text):
        rates += [(m.end() + r.start(1), r.group(1)) for r in RATE_ANY.finditer(_cut(text, m.end(), 160))]
    for rx in (RATE_PER, RATE_EQ):
        rates += [(m.start(1), m.group(1)) for m in rx.finditer(text)]
    explicit = {m.start(1): (m.start(2), m.group(2)) for m in EURO_X_RATE.finditer(text)}
    rates += list(explicit.values())
    rates = [(p, r) for p, r in rates if 1 < _dec(r) < 10]
    kept = set(rates)
    return rates, {k: v for k, v in explicit.items() if v in kept}


def _vat_rates(text):
    """The VAT rates a text states: 23, 8 or 5% (not a decimal like 13,5%) with "VAT" or "podatek od towarów (i usług)"
    in the same sentence; "5% wartości przedmiotu sporu" or "stopa referencyjna 5%" is not a VAT rate."""
    bounds = [m.start() for m in BOUND.finditer(text)]
    ctx = [(m.start(), m.end()) for m in VAT_CTX.finditer(text)]  # (they cannot overlap: these are all of them)
    has = {}  # sentence -> it names VAT (a context match lies inside it)
    out = set()
    for m in VAT.finditer(text):
        i = bisect_left(bounds, m.start())
        if i not in has:
            a, b = bounds[i - 1] + 1 if i else 0, bounds[i] if i < len(bounds) else len(text)
            j = bisect_left(ctx, (a,))
            has[i] = j < len(ctx) and ctx[j][1] <= b
        if has[i]:
            out.add(int(m.group(1)))
    return out


def _money(text, covered):
    """Money arithmetic of a state: (computation lines, comparison lines). `covered`: amounts whose net/gross the VAT
    line of facts() already gives."""
    bounds = [m.start() for m in BOUND.finditer(text)]

    def sent(pos):
        return bisect_left(bounds, pos)

    # stated zł amounts; use: None (free), "rate", "alias" (a stated equivalent of a conversion), "op" (an operand:
    # unit price, fee, budget, spending), "subj" (a labelled amount or a lone line price, compared via the pool)
    st = [dict(s=m.start(), v=_dec(m.group(1)), use=None, lab=(LABEL.match(text, m.end()) or [None, None])[1])
          for m in STATED.finditer(text)]
    at = {a["s"]: a for a in st}  # (in text order: lookups by position, not scans of every amount per lookup)
    starts = [a["s"] for a in st]

    def stated_at(pos):
        return at.get(pos)

    def first_free(a, n):
        lim = a + len(_cut(text, a, n))
        for i in range(bisect_left(starts, a), bisect_left(starts, lim)):
            if st[i]["use"] is None:
                return st[i]
        return None

    calc, refs, subjects = [], [], []

    # 1. euro × the stated exchange rate
    rates, explicit = _rates(text)
    for p, _ in rates:
        a = stated_at(p)
        if a:
            a["use"] = "rate"
    distinct = {}
    for _, r in rates:
        distinct.setdefault(_dec(r), r)
    eur, seen = [], set()
    for m in EURO.finditer(text):
        if text[max(0, m.start() - 3):m.start()] == "za " or EQ_NEXT.match(text, m.end()):
            continue  # the "za 1 euro" / "1 EUR =" of a rate
        if _dec(m.group(1)) not in seen:
            seen.add(_dec(m.group(1)))
            eur.append((m.start(), _dec(m.group(1))))
    for p, v in eur[:MAX_EURO]:
        if p in explicit:
            r = explicit[p][1]
        elif len(distinct) == 1:
            r = next(iter(distinct.values()))
        else:
            near = {_dec(r): r for pos, r in rates if sent(pos) == sent(p)}
            if len(near) != 1:
                continue
            r = next(iter(near.values()))
        c = _q(v * _dec(r))
        calc.append(f"{_pl_num(v)} euro × {r} zł = {_pl_amount(c)}")
        for a in st:
            if a["use"] is None and a["v"] == c and sent(a["s"]) == sent(p):
                a["use"] = "alias"
        refs.append(c)

    # 2. budget minus spending
    m = BUDGET.search(text)
    b = first_free(m.end(), 80) if m else None
    if b:
        used, taken = [], set()
        for u in USED.finditer(text):
            x = first_free(u.end(), 40) if sent(u.start()) in (sent(b["s"]), sent(b["s"]) + 1) else None
            if x and x is not b and x["s"] not in taken:
                used.append((u.group(1), x))
                taken.add(x["s"])
        if used:
            r = b["v"] - sum(x["v"] for _, x in used)
            calc.append(f"budżet {_pl_amount(b['v'])} − " + " − ".join(f"{w} {_pl_amount(x['v'])}" for w, x in used)
                        + f" = {_pl_amount(r)}")
            for x in [b] + [x for _, x in used]:
                x["use"] = "op"
            refs.append(_q(r))

    # 3. quantity × unit price, and the total of the lines of one list (one sentence)
    lists = {}
    for m in PRODUCT.finditer(text):
        a = stated_at(m.end())
        if a and a["use"] is None and int(m.group(1)) > 0:
            lists.setdefault(sent(m.start()), []).append((int(m.group(1)), a))
    for items in list(lists.values())[:2]:
        items = items[:5]
        labs = {a["lab"] for _, a in items}
        lab = labs.pop() if len(labs) == 1 else None
        if len(items) == 1 and items[0][0] == 1:  # "1 szt. × P": the price is the total
            items[0][1]["use"] = "subj"
            subjects.append((items[0][1]["v"], lab, items[0][1]))
            continue
        tots = []
        for n, a in items:
            if n > 1:
                calc.append(f"{n} × {_pl_amount(a['v'])} = {_pl_amount(n * a['v'])}")
            tots.append(n * a["v"])
            a["use"] = "op"
        if len(tots) > 1:
            calc.append(" + ".join(_pl_amount(t) for t in tots) + f" = {_pl_amount(sum(tots))}")
        subjects.append((sum(tots), lab, None))

    # 4. a quantity-tiered price list and the ordered quantity
    tiers = list(TIER.finditer(text))
    if len(tiers) >= 2:
        spans = [(m.start(), m.end()) for m in tiers]

        def in_tier(pos):
            i = bisect_right(spans, (pos, float("inf"))) - 1
            return i >= 0 and pos < spans[i][1]
        qs = {int(m.group(1)) for m in QTY.finditer(text)
              if not in_tier(m.start()) and not TIMES_NEXT.match(text, m.end())
              and not text.endswith(QTY_PRE, 0, m.start())}
        if len(qs) == 1:
            q, hits = qs.pop(), []
            for m in tiers:
                kind, x, y = m.group(2), int(m.group(3)), m.group(4) or m.group(5)
                if kind == "poniżej":
                    ok, cond = q < x, f"{q} < {x}"
                elif kind == "do":
                    ok, cond = q <= x, f"{q} ≤ {x}"
                elif kind == "powyżej":
                    ok, cond = q > x, f"{q} > {x}"
                elif y:
                    ok, cond = x <= q <= int(y), f"{x} ≤ {q} ≤ {y}"
                else:
                    ok, cond = q >= x, f"{q} ≥ {x}"
                if ok:
                    hits.append((cond, _dec(m.group(1))))
            if len(hits) == 1:
                cond, p = hits[0]
                calc.append(f"{cond}; {q} × {_pl_amount(p)} = {_pl_amount(q * p)}")
                subjects.append((q * p, None, None))
        for m in tiers:
            a = stated_at(m.start(1))
            if a and a["use"] is None:
                a["use"] = "op"

    # 5. the delivery fee added to the one total
    if len(subjects) == 1:
        for m in FEE.finditer(text):
            f = first_free(m.end(), 30)
            if f:
                t = subjects[0][0]
                calc.append(f"{_pl_amount(t)} + {_pl_amount(f['v'])} = {_pl_amount(t + f['v'])}")
                f["use"] = "op"
                break

    # 6. net/gross of the labelled amounts and totals, for the one VAT rate stated
    for a in st:
        if a["use"] is None and a["lab"]:
            a["use"] = "subj"
            subjects.append((a["v"], a["lab"], a))
    vr = _vat_rates(text)
    k = 1 + Decimal(vr.pop()) / 100 if len(vr) == 1 else None
    pool = []  # (value, kind 0 subject / 1 other stated amount / 2 reference, group: no comparison inside a group)
    for g, (v, lab, src) in enumerate(subjects):
        pool.append((_q(v), 0, g))
        if k and lab:
            kk = str(k).replace(".", ",")
            d = v * k if lab == "netto" else v / k
            if src is None or src["v"] not in covered:
                calc.append(f"{_pl_amount(v)} {lab} {'×' if lab == 'netto' else '÷'} {kk} = {_pl_amount(d)} "
                            f"{'brutto' if lab == 'netto' else 'netto'}")
            pool.append((_q(d), 0, g))
    pool += [(v, 2, -1 - i) for i, v in enumerate(refs)]
    pool += [(a["v"], 1, -100 - i) for i, a in enumerate(st) if a["use"] is None]

    # 7. comparisons (symbols only)
    first = {}
    for x in pool:
        first.setdefault((x[0], x[1]), x)  # the first of each (value, kind), in pool order
    uniq = list(first.values())
    comps = []
    for _, i, j in _first_pairs(uniq, MAX_COMPARE):
        x, y = (uniq[i], uniq[j]) if uniq[i][1] <= uniq[j][1] else (uniq[j], uniq[i])
        op = ">" if x[0] > y[0] else "<" if x[0] < y[0] else "="
        comps.append(f"{_pl_amount(x[0])} {op} {_pl_amount(y[0])}")
    return calc, comps


def _first_pairs(uniq, k):
    """The first k of the sorted (n, i, j), i < j, n = how many of the two are other stated amounts (kind 1), of the
    pairs of different groups (pairs of two kind-1 amounts only when there are at most MAX_PLAIN of them), without
    building every pair: a text with thousands of amounts made all of their pairs (quadratic time and memory)."""
    plain = [i for i, x in enumerate(uniq) if x[1] == 1]
    other = [i for i, x in enumerate(uniq) if x[1] != 1]
    out = []
    for n in (0, 1, 2):
        if n == 2 and len(plain) > MAX_PLAIN:
            break
        for i, x in enumerate(uniq):
            need = n - (x[1] == 1)  # how many kind-1 amounts j adds
            if need not in (0, 1):
                continue
            js = plain if need else other
            for t in range(bisect_right(js, i), len(js)):
                j = js[t]
                if x[2] != uniq[j][2]:
                    out.append((n, i, j))
                    if len(out) == k:
                        return out
    return out


def facts(text):
    """The helper facts for a state, as a list of lines (empty if nothing to compute)."""
    lines = []
    ds = _dates(text)[:MAX_DATES]
    for d in ds:
        tag = WEEKDAYS[d.weekday()]
        h = holidays(d.year).get(d)
        if h:
            tag += f"; dzień ustawowo wolny od pracy ({h})"
        lines.append(f"{fmt(d)} to {tag}")
    for a, b in zip(ds, ds[1:]):
        n = (b - a).days
        lines.append(f"Od {fmt(a)} do {fmt(b)}: {abs(n)} dni ({'później' if n >= 0 else 'wcześniej'})")
    m = TODAY.search(text)
    if m:
        today = _dates(text[m.end():m.end() + 40])
        if today:
            for g in AGO.finditer(text):
                n = int(g.group(1))
                try:
                    before = today[0] - timedelta(days=n)
                except OverflowError:  # before year 1: no fact rather than an HTTP 500
                    continue
                lines.append(f"{n} dni przed {fmt(today[0])} to {fmt(before)}")
    durs = _durations(text)
    for d in ds[:3]:
        for n, unit in durs:
            one, many = UNIT_PL[unit]
            end = _plus(d, n, unit)
            if end is None:  # outside year 1..9999: no fact rather than an HTTP 500
                continue
            lines.append(f"{fmt(d)} + {n} {one if n == 1 else many} (kalendarzowo) = {fmt(end)} "
                         f"({WEEKDAYS[end.weekday()]})")
    rates = sorted(_vat_rates(text), reverse=True)
    covered = set()
    rate_at = {p for p, _ in _rates(text)[0]}  # an exchange rate is not an amount to put VAT on
    for raw in [m.group(1) for m in AMOUNT.finditer(text) if m.start() not in rate_at][:MAX_AMOUNTS]:
        x = _dec(raw)
        for r in rates:
            k = 1 + Decimal(r) / 100
            kk = str(k).replace(".", ",")
            lines.append(f"{_pl_amount(x)} × {kk} = {_pl_amount(x * k)}; {_pl_amount(x)} ÷ {kk} = {_pl_amount(x / k)}")
            covered.add(x)
    calc, comps = _money(text, covered)
    room = max(0, MAX_TOTAL - len(lines))
    calc = calc[:room]
    return lines + calc + comps[:min(MAX_COMPARE, room - len(calc))]


def inject(state):
    """State with the helper facts appended (unchanged if there is nothing to compute)."""
    ls = facts(state)
    return state if not ls else state + "\n\n" + HEADER + "\n" + "\n".join(f"- {x}" for x in ls)
