"""basal run (batch): the fields of a full request survive to_request (B-097). Fails on 8e80d3e plus the earlier packages."""
from basal import run as run_module


def test_basal_run_keeps_the_request_fields_of_a_full_request():
    item = {"id": "x", "state": "s", "questions": {"q": {"type": "choice", "criteria": ["A", "B"]}}, "lang": "pl",
            "facts": "auto", "think": "off", "media": [{"type": "image", "data": "AAAA"}], "detail": "high"}
    sent = run_module.to_request(item)
    assert {k: v for k, v in item.items() if k != "id"} == sent
    assert "lang" not in run_module.to_request({"state": "s", "questions": {}})            # control: absent stays absent
