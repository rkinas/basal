import json
from pathlib import Path

from basal.compat import validate_systemone
from basal.server import to_items
from basal.tasks import expand, is_task

ADVANCED = Path(__file__).resolve().parent.parent / "basal/examples/advanced.jsonl"
TYPES = {"multi", "act", "tool", "classify", "verify", "compress", "extract"}


def test_advanced_examples_cover_every_type_and_parse_as_the_server_parses_them():
    requests = [json.loads(line) for line in ADVANCED.read_text().splitlines() if line.strip()]
    assert {q["type"] for r in requests for q in r["questions"].values()} == TYPES
    for r in requests:
        validate_systemone(r)
        decisions, plans = expand(r)
        assert set(plans) == {n for n, q in r["questions"].items() if is_task(q)}
        items = to_items(decisions["state"], decisions["questions"])
        assert items and all(item["branches"] for item in items)
