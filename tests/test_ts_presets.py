"""The TypeScript client's presets (clients/typescript/src/presets.ts) and the fixture its tests compare it with
(test/presets_fixture.json) are what scripts/ts_presets.py makes from basal/presets.py: a change of a preset, a policy,
guard() / route() or lang_of fails here until they are regenerated (the client's own tests then check its langOf,
guard and route on the new cases)."""
from scripts import ts_presets


def test_typescript_presets_and_fixture_follow_basal_presets():
    assert [p.name for p in ts_presets.drift()] == [], "run python scripts/ts_presets.py"


def test_a_changed_preset_is_drift(monkeypatch):
    from basal import presets
    monkeypatch.setitem(presets.PRESETS["triage"]["en"]["urgency"], "instructions", "How urgent?")
    assert {p.name for p in ts_presets.drift()} == {"presets.ts", "presets_fixture.json"}


def test_the_language_cases_cover_every_word_lang_of_counts():
    from basal.presets import _EN, _PL
    texts = {t for t, _ in ts_presets.lang_cases()}
    ws = ts_presets.words(_PL.pattern) + ts_presets.words(_EN.pattern)
    assert len(ws) > 50 and {"nie", "się", "na", "I", "the"} <= set(ws) and all(f"{w} {w}" in texts for w in ws)
