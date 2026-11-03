"""examples/triage/run.py offline: the questions with and without RULES, the labelled requests for basal-calibrate, and
the answers read as gold labels."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triage_run", ROOT / "examples/triage/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_rules_replace_the_options_of_every_preset_question():
    for lang in ("pl", "en"):
        plain = run.triage_questions("x", lang)
        ruled = run.triage_questions("x", lang, max_error=0.05, rules=True)
        assert set(ruled) == set(plain) == set(run.RULES[lang])
        for name, q in ruled.items():
            assert q["criteria"] == run.RULES[lang][name] and q["max_error"] == 0.05
            assert q["type"] == plain[name]["type"]
        assert list(ruled["category"]["criteria"]) == list(plain["category"]["criteria"])  # same keys as the preset
        assert len(ruled["urgency"]["criteria"]) == len(plain["urgency"]["criteria"])


def test_calibration_input_carries_the_gold_labels(tmp_path):
    out = tmp_path / "labelled.jsonl"
    run.main(["--calibration-input", str(out), "--rules"])
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    tickets = [json.loads(line) for line in (ROOT / "examples/triage/tickets.jsonl").read_text().splitlines()]
    assert len(lines) == len(tickets) == 100
    forced = tmp_path / "pl.jsonl"
    run.main(["--calibration-input", str(forced), "--rules", "--lang", "pl"])
    assert all(json.loads(line)["questions"]["category"]["criteria"] == run.RULES["pl"]["category"]
               for line in forced.read_text(encoding="utf-8").splitlines())
    for line, t in zip(lines, tickets):
        assert line["id"] == t["id"] and line["state"] == t["state"]
        assert {n: q["gold"] for n, q in line["questions"].items()} == t["gold"]
        assert line["questions"]["category"]["criteria"] == run.RULES[run.lang_of(t["state"])]["category"]
        assert t["gold"]["category"] in line["questions"]["category"]["criteria"]


def test_answers_read_as_gold_labels():
    assert run.predicted({"type": "choice", "choice": "billing"}) == "billing"
    assert run.predicted({"type": "score", "score": 1.1, "probabilities": {"0": 0.3, "1": 0.3, "2": 0.4}}) == 2
    assert run.predicted({"type": "noul", "noul": 0.5}) == "yes"
    assert run.predicted({"type": "noul", "noul": 0.49}) == "no"
