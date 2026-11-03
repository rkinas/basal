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


spec_app = importlib.util.spec_from_file_location("triage_app", ROOT / "examples/triage/app.py")
app_mod = importlib.util.module_from_spec(spec_app)
spec_app.loader.exec_module(app_mod)


def answers(cat_abstain=False, urgency=(0.1, 0.2, 0.7), human=0.8, human_abstain=False):
    return {"category": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.95, "other": 0.05},
                         "confidence": 0.95, "abstain": cat_abstain},
            "urgency": {"type": "score", "score": 1.6, "legend": {"0": "niska", "1": "średnia", "2": "wysoka"},
                        "probabilities": {str(i): p for i, p in enumerate(urgency)}, "abstain": True},
            "needs_human": {"type": "noul", "noul": human, "probabilities": {"true": human, "false": 1 - human},
                            "confidence": max(human, 1 - human), "abstain": human_abstain}}


def test_route_sends_abstained_categories_to_people():
    r = app_mod.route(answers())
    assert r == {"queue": "billing", "priority": "high", "category": "billing", "urgency": "wysoka",
                 "needs_human": True, "automated": False}
    assert app_mod.route(answers(cat_abstain=True))["queue"] == "human"
    assert app_mod.route(answers(urgency=(0.6, 0.3, 0.1)))["priority"] == "normal"
    assert app_mod.route(answers(human_abstain=True))["needs_human"] is None


def test_webhook_asks_basal_with_the_rules_and_returns_the_queue():
    import asyncio

    import httpx

    from basal.client import AsyncBasal
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "usage": {}, "answers": answers()})
    basal = AsyncBasal("http://basal", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    web = app_mod.make_app(basal, max_error=0.05, lang="pl")

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url="http://app") as c:
            return (await c.post("/ticket", json={"id": "7", "text": "Pobrano dwa razy 129 zł"}),
                    await c.post("/ticket", json={"id": "8", "text": " "}))
    ok, empty = asyncio.run(go())
    assert ok.status_code == 200 and ok.json()["queue"] == "billing" and ok.json()["id"] == "7"
    assert empty.status_code == 422 and len(sent) == 1
    q = sent[0]["questions"]
    assert q["category"]["criteria"] == run.RULES["pl"]["category"] and q["category"]["max_error"] == 0.05
