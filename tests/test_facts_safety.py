"""facts=auto on dates outside year 1..9999 gives no fact instead of an exception."""
import pytest

from basal.facts import facts, inject


@pytest.mark.parametrize("text", ["Umowa z 31.12.9999 na 30 dni.", "Umowa z 31.12.9999 na 1 miesiąc.",
                                  "Umowa z 29.02.9996 na 5 lat.", "Umowa z 15.06.9999 na 2 lata.",
                                  "Dziś 1.01.0001, 5 dni temu byliśmy tam.", "Termin 15.12.9999 plus 3 miesiące."])
def test_a_date_out_of_range_gives_no_fact_instead_of_an_exception(text):
    assert isinstance(inject(text), str)


@pytest.mark.parametrize("text, fact", [
    ("Umowa z 31.12.2025 na 30 dni.", "30 dni (kalendarzowo) = 30 stycznia 2026"),
    ("Umowa z 29.02.2024 na 1 rok.", "1 rok (kalendarzowo) = 28 lutego 2025"),
    ("Dziś 10.03.2025, 5 dni temu.", "5 dni przed 10 marca 2025 r. to 5 marca 2025")])
def test_dates_in_range_keep_their_facts(text, fact):
    assert fact in inject(text)
    assert facts("Dnia 31.12.9999 r.")
