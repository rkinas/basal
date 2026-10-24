"""Decision safety: the certified thresholds and context policy a model reports, deferral under max_error, a state
truncated to the context, and the History flags of both."""
from tests.e2e.support import long_state

NO_CERTIFICATE = "no certified threshold for this type at this max_error"
AMBIGUOUS = {"state": "Klient pisze: coś mi nie działa, nie wiem co.",
             "questions": {"refund": {"type": "act", "instructions": "Czy zwrócić klientowi pieniądze?",
                                      "criteria": {"refund": "należy się zwrot", "keep": "zwrot się nie należy"},
                                      "costs": {"wrong": 10, "defer": 1}, "max_error": 0.01}}}


def test_the_models_entry_carries_its_certified_thresholds_and_context_policy(ui, text_model):
    entry = ui.load(text_model["id"])["entry"]
    assert entry is not None, "basal-ui did not read the child's GET /v1/models entry"
    context = entry["context"]
    assert isinstance(context["limit"], int) and context["limit"] >= 512
    assert context["overflow"] == "truncate" and 0 <= context["truncate_head"] <= 1
    for target, per_type in (entry.get("certified") or {}).items():
        assert float(target) in (0.01, 0.05) and per_type
        assert all(t in ("choice", "noul", "score") and 0 < c < 1 for t, c in per_type.items())


def test_act_below_the_certified_threshold_is_deferred_to_a_person(ui, text_model):
    entry = ui.load(text_model["id"])["entry"]
    answer = ui.decide(text_model["id"], AMBIGUOUS)["answers"]["refund"]
    threshold = (entry.get("certified") or {}).get("0.01", {}).get("choice")
    assert set(answer["expected_costs"]) == {"refund", "keep", "defer"}
    if threshold is None:  # no certificate at 1%: every automatic action is refused
        assert answer["max_error"] == NO_CERTIFICATE and answer["deferred"] is True and answer["action"] == "defer"
    elif answer["confidence"] < threshold:
        assert answer["deferred"] is True and answer["action"] == "defer"
    if answer.get("refused") is not None:
        assert answer["refused"] in ("refund", "keep")
    record = ui.get("/api/history", params={"model": text_model["id"], "source": "playground"})["items"][0]
    assert record["deferred"] is answer["deferred"]


def test_a_state_over_the_context_is_truncated_and_flagged(ui, text_model):
    limit = ui.load(text_model["id"])["entry"]["context"]["limit"]
    body = {"state": long_state(limit), "questions": {"q": {"type": "noul", "instructions": "Czy to umowa najmu?"}}}
    usage = ui.decide(text_model["id"], body)["usage"]
    t = usage["truncated"]
    assert t["dropped_tokens"] > 0 and t["head_tokens"] + t["tail_tokens"] + t["dropped_tokens"] == t["state_tokens"]
    assert usage["longest_prompt_tokens"] <= limit
    record = ui.get("/api/history", params={"model": text_model["id"], "source": "playground"})["items"][0]
    assert record["truncated"] is True
