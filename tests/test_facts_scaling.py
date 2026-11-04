"""facts="auto" reads a long state in linear time, its amount patterns match what they matched before, and the server
bounds the state it reads."""
import asyncio
import time

import pytest

from basal import facts as f
from basal.facts import MAX_CHARS, facts
from test_server import FakeServer

# Texts that made facts() quadratic: one unit repeated to N characters (each unit, the i-th of the run).
UNITS = {
    "digit groups without a unit": lambda i: "111 ",
    "zł amounts": lambda i: f"{i} zł ",
    "netto amounts and a VAT rate": lambda i: f"{i} zł netto " if i else "Stawka VAT 23%. ",
    "percentages without VAT": lambda i: "23% ",
    "exchange rates": lambda i: "kurs 4,31 zł ",
    "budget and spending": lambda i: "Budżet 1 000 000 zł, " if i == 0 else f"wydano {i},00 zł, ",
    "product lines": lambda i: f"{i % 9999 + 1} szt. × {i},50 zł ",
    "euro × rate": lambda i: f"{i % 999 + 1},{i // 999 % 99:02d} euro × 4,{i % 90 + 10} zł ",
    "dates": lambda i: f"{i % 28 + 1}.{i // 28 % 12 + 1}.{2000 + i // 336 % 7000} ",
    "durations": lambda i: f"{i % 999 + 1} dni {i % 999 + 1} lat ",
    "tiers and quantities": lambda i: ("10,00 zł za sztukę od 1 do 99 szt. 9,00 zł za sztukę od 100 szt. " if i == 0
                                       else f"{i % 99999 + 1} szt. "),
}


def _text(unit, n):
    out, size, i = [], 0, 0
    while size < n:
        out.append(unit(i))
        size += len(out[-1])
        i += 1
    return "".join(out)[:n]


def _best(text, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        facts(text)
        best = min(best, time.perf_counter() - t)
    return best


@pytest.mark.parametrize("name", list(UNITS))
def test_facts_reads_a_long_text_in_linear_time(name):
    """16 times the text takes far less than 16 x 16 times as long: the ratio of the best of a few runs on 40 000 and
    2 500 characters (linear scans: about 16; the quadratic ones took 0.1 to 14 s on 40 000 characters, a ratio near
    256). A ratio, not a wall time, so a loaded machine slows both sizes alike."""
    small, large = _text(UNITS[name], 2_500), _text(UNITS[name], 40_000)
    _best(small, 1)  # (warm: compiled patterns, caches)
    ratio = _best(large, 2) / max(_best(small, 5), 1e-4)
    assert ratio < 64, f"{name}: 16 x the text took {ratio:.0f} x as long"


@pytest.mark.parametrize("text,expected", [
    ("Kwoty: 1111 111 111 zł, 111 11 111 zł, 1 111 111 zł i 111 111 111 111.",
     {"AMOUNT": ["111 111", "11 111", "1 111 111"], "STATED": ["111 111", "11 111", "1 111 111"]}),
    ("Raty 100 200 300 400 500 zł oraz 600 700 PLN, a 12.345.678,90 zł i 5 000 000,50 zł.",
     {"AMOUNT": ["100 200 300 400 500", "12.345.678,90", "5 000 000,50"],
      "STATED": ["100 200 300 400 500", "600 700", "12.345.678,90", "5 000 000,50"]}),
    ("Limit 1 111 111 euro, 111 1111 111 €, 2 000 000 EUR × 4,3770 zł.",
     {"EURO": ["1 111 111", "111", "2 000 000"], "EURO_X_RATE": ["2 000 000"]}),
    ("Cennik: 1 000 000,00 zł za sztukę od 1 000 1 do 99 szt.; 111 222 333 zł od 100 szt.",
     {"STATED": ["1 000 000,00", "111 222 333"], "TIER": ["111 222 333"]}),
])
def test_amounts_in_digit_group_runs_keep_their_matches(text, expected):
    for name, want in expected.items():
        assert [m.group(1) for m in getattr(f, name).finditer(text)] == want, name


def test_facts_auto_bounds_the_state():
    body = {"state": "Doręczono 20 marca 2027 r. " * (MAX_CHARS // 27 + 1), "facts": "auto",
            "questions": {"q": {"type": "noul", "instructions": "Czy w terminie?"}}}
    with pytest.raises(ValueError, match="at most"):
        asyncio.run(FakeServer([0.6, 0.4])._run(body))
    asyncio.run(FakeServer([0.6, 0.4])._run(dict(body, state="Doręczono 20 marca 2027 r.")))
