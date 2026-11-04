"""The helper facts of "N dni temu": each N once, at most MAX_DURATIONS lines, as the durations are."""
from basal.facts import MAX_DURATIONS, facts


def ago_lines(text):
    return [x for x in facts(text) if " dni przed " in x]


def test_days_ago_are_deduplicated_and_capped():
    text = "Dziś jest 10.03.2026. " + " ".join(f"Zgłoszenie {i}: {n} dni temu." for i, n in
                                                enumerate([5, 5, 5, 7, 9, 11, 13, 5, 7]))
    lines = ago_lines(text)
    assert len(lines) == MAX_DURATIONS == 3
    assert [x.split(" ")[0] for x in lines] == ["5", "7", "9"]                # first seen, each once
    assert lines[0] == "5 dni przed 10 marca 2026 r. to 5 marca 2026 r."
    assert len(ago_lines("Dziś jest 10.03.2026. " + "Było to 2 dni temu. " * 200)) == 1
