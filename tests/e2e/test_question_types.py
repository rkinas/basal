"""Every question type end to end on a real model: the bundled advanced examples (one request per type), facts,
evidence, and the statistics per type."""
import pytest

from tests.e2e.support import CHOICE, advanced_requests

REQUESTS = {r["id"]: r for r in advanced_requests()}


def check_multi(a, q):
    assert set(a["probabilities"]) == set(q["criteria"]) and set(a["selected"]) <= set(q["criteria"])
    assert len(a["selected"]) <= q.get("max", len(q["criteria"])) and 0 <= a["set_confidence"] <= 1


def check_act(a, q):
    outcomes = set(q["criteria"])
    assert set(a["probabilities"]) == outcomes and a["answer"] in outcomes
    assert set(a["expected_costs"]) == outcomes | {"defer"} and a["action"] in a["expected_costs"]
    cheapest = min(a["expected_costs"], key=a["expected_costs"].get)
    assert a["action"] == cheapest or a.get("refused") == cheapest  # refused under max_error: deferred instead


def check_tool(a, q):
    names = {t.get("name") or t["function"]["name"] for t in q["tools"]}
    options = names | {"answer_directly", "ask_user", "none"}
    assert set(a["probabilities"]) == options and (a["tool"] in options or a.get("deferred"))
    assert a["call"] == (a["tool"] in names) and len(a["ranking"]) == q["rank"]
    assert all(r["tool"] in options for r in a["ranking"])


def check_classify(a, q):
    labels = set(q["labels"]) | ({"none"} if q.get("none") else set())
    assert a["label"] in labels and set(a["scores"]) == labels and set(a["automated"]) == {"0.01", "0.05"}


def check_verify(a, q):
    assert set(a["claims"]) == set(q["claims"])
    for claim in a["claims"].values():
        assert claim["verdict"] in ("supported", "contradicted", "not_stated") and 0 < claim["confidence"] <= 1
    assert a["all_supported"] == (not a["not_supported"])


def check_compress(a, q, state):
    assert a["unit"] == q.get("unit", "sentence") and a["kept"] and set(a["kept"]) <= set(range(len(a["units"])))
    assert a["compressed_chars"] <= a["original_chars"] == len(state)
    for i in a["kept"]:  # the kept units, in order, are the compressed text
        u = a["units"][i]
        assert state[u["start"]:u["end"]].strip() in a["compressed"]


def check_extract(a, q, state):
    props = q["schema"]["properties"]
    assert set(a["fields"]) == set(props)
    for path, f in a["fields"].items():
        if f["value"] is None:
            continue
        kind = props[path].get("type")
        if "enum" in props[path]:
            assert f["value"] in props[path]["enum"]
        elif kind == "boolean":
            assert isinstance(f["value"], bool)
        elif kind == "number":
            assert isinstance(f["value"], (int, float))
        if "start" in f:  # a value taken from the state: its offsets point at it
            assert state[f["start"]:f["end"]] == f["text"]


@pytest.mark.parametrize("example", sorted(REQUESTS))
def test_each_question_type_answers_in_its_own_shape(ui, text_model, example):
    ui.load(text_model["id"])
    request = REQUESTS[example]
    name, q = next(iter(request["questions"].items()))
    a = ui.decide(text_model["id"], request)["answers"][name]
    assert a["type"] == q["type"]
    check = globals()[f"check_{q['type']}"]
    if q["type"] in ("compress", "extract"):
        check(a, q, request["state"])
    else:
        check(a, q)
    if q["type"] in ("classify", "verify", "compress", "extract"):
        assert a["experimental"] is True


def test_every_type_reaches_the_statistics(ui, text_model):
    ui.load(text_model["id"])
    for request in REQUESTS.values():
        ui.decide(text_model["id"], request)
    by_type = ui.get("/api/stats")["models"][text_model["id"]]["by_type"]
    assert {"multi", "act", "tool", "classify", "verify", "compress", "extract"} <= set(by_type)


def test_facts_and_language_are_accepted_and_evidence_needs_its_head(ui, text_model):
    ui.load(text_model["id"])
    body = {**CHOICE, "state": CHOICE["state"] + " Blokada nastąpiła 3 marca 2025 o 14:05.", "facts": "auto", "lang": "pl"}
    assert ui.decide(text_model["id"], body)["answers"]["dept"]["choice"] in ("cards", "loans", "online")
    q = {**CHOICE["questions"]["dept"], "evidence": True}
    r = ui.call("POST", "/api/decide", expect=(200, 422), json={"model": text_model["id"], **CHOICE,
                                                                  "questions": {"dept": q}})
    if isinstance(r, dict) and "answers" in r:  # the model ships evidence_head.pt: spans of the state
        for span in r["answers"]["dept"]["evidence"]:
            assert CHOICE["state"][span["start"]:span["end"]] == span["text"]
    else:
        assert "evidence is not available for this model" in r["error"]
