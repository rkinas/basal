"""Regression tests for fix/facts-amounts (Z-1082)."""
import re
import unicodedata

import pytest

from basal import facts


@pytest.mark.parametrize("sp", ["\u00a0", "\u202f", "\u2009"], ids=["nbsp", "narrow-nbsp", "thin-space"])
def test_i08_amount_with_a_non_breaking_or_thin_space_thousands_separator(sp):
    text = f"Wartość 1{sp}234,56 zł netto, VAT 23%."
    out = facts.inject(text)
    assert "1 518,51" in out  # 1 234,56 x 1,23
    assert "234,56 zł × 1,23 = 288,51" not in out  # not the tail "234,56" as an amount of its own


def test_i08_plain_space_and_small_amount_are_unchanged():
    assert "1 518,51" in facts.inject("Wartość 1 234,56 zł netto, VAT 23%.")
    assert "99,50 zł × 1,08" in facts.inject("Cena 99,50 zł, VAT 8%.")


ZS = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Zs"]


def test_i08_the_space_class_is_exactly_unicode_zs():
    assert {c for c in map(chr, range(0x3100)) if re.fullmatch(facts.ZS, c)} == set(ZS)


@pytest.mark.parametrize("sp", ZS, ids=lambda c: f"U+{ord(c):04X}")
def test_i08_every_space_separator_works_between_the_digits(sp):
    out = facts.facts(f"Razem 12{sp}500 zł plus 3{sp}000 zł")
    assert out == ["12 500,00 zł > 3 000,00 zł"], out


@pytest.mark.parametrize("sp", ZS, ids=lambda c: f"U+{ord(c):04X}")
def test_i08_every_space_separator_works_between_the_number_and_the_currency(sp):
    assert facts.facts(f"Razem 12 500{sp}zł plus 3 000{sp}PLN") == ["12 500,00 zł > 3 000,00 zł"]
    assert facts.facts(f"Razem 12{sp}500{sp}zł plus 3{sp}000{sp}zł") == ["12 500,00 zł > 3 000,00 zł"]
    assert "1 518,51" in facts.inject(f"Wartość 1{sp}234,56{sp}zł netto, VAT 23%.")


def test_i08_a_zero_width_or_line_separator_is_not_a_space():
    for c in ("\u200b", "\u2028", "\t"):
        assert facts.facts(f"Razem 12{c}500 zł plus 3{c}000 zł") != ["12 500,00 zł > 3 000,00 zł"]
