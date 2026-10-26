"""basal-finetune: the split by group, the examples (the prompt basal-serve builds, both option orders), and the
readout loss over the option codes. The training itself needs peft and a model: tests run it on a toy module."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from basal import finetune
from basal.prompt import render
from tests.test_server import Tok

DATA = Path(finetune.__file__).parent / "examples" / "labelled_tickets.jsonl"


def test_split_keeps_groups_together_and_trains_only_on_the_rest():
    train, held, skipped = finetune.split(DATA, holdout=0.25, seed=0)
    lines = DATA.read_text(encoding="utf-8").splitlines()
    assert skipped == {} and len(train) + 2 * len(held) == 2 * len(lines)  # two decisions per line
    held_groups = {json.loads(l)["group"] for l in held}
    train_groups = {d["group"] for _, d in train}
    assert held_groups and train_groups and not held_groups & train_groups
    assert 0.15 < len(held) / len(lines) < 0.35
    again = finetune.split(DATA, holdout=0.25, seed=0)[1]
    assert again == held                                                        # deterministic
    assert finetune.split(DATA, holdout=0.25, seed=1)[1] != held                # the seed moves it


def test_example_is_the_served_prompt_in_both_orders_and_the_gold_follows_the_order():
    body = {"state": "Anna writes: the app crashes.", "questions": {"dept": {
        "type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "ship": "parcels"}}}}
    d = {"name": "dept", "prim": "choice", "gold": 1}                            # tech
    tok = Tok()
    orders, why = finetune.example(tok, body, d, 1, "canonical", 100000)
    assert why is None and len(orders) == 2
    (ids0, codes0, gold0), (ids1, codes1, gold1) = orders
    assert (gold0, gold1) == (1, 1)                                              # the middle option, either order
    assert "".join(map(chr, ids0)) == render(tok, body["state"], "Which team?",
                                             ["billing: money", "tech: bugs", "ship: parcels"], "en", 1)
    assert "".join(map(chr, ids1)) == render(tok, body["state"], "Which team?",
                                             ["ship: parcels", "tech: bugs", "billing: money"], "en", 1)
    d = dict(d, gold=0)
    assert [g for _, _, g in finetune.example(tok, body, d, 1, "canonical", 100000)[0]] == [0, 2]
    assert finetune.example(tok, body, d, 1, "canonical", 10)[1].startswith("longer than --max-len")


def test_code_logits_read_the_option_codes_at_the_last_position_with_the_softcap():
    class Toy(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep=None):
            assert logits_to_keep == 1
            v = torch.zeros(1, 1, 8)
            v[0, 0, 3], v[0, 0, 5] = 4.0, 8.0
            return SimpleNamespace(logits=v)
    got = finetune.code_logits(Toy(), [1, 2, 3], [5, 3, 0], None, "cpu")
    assert got.tolist() == [8.0, 4.0, 0.0]
    capped = finetune.code_logits(Toy(), [1], [5, 3], 2.0, "cpu")
    assert capped.tolist() == pytest.approx([2 * torch.tanh(torch.tensor(4.0)).item(), 2 * torch.tanh(torch.tensor(2.0)).item()])
    ex = [[([1, 2], [5, 3, 0], 0), ([1, 2], [5, 3, 0], 2)]]
    assert finetune.accuracy(Toy(), ex, None, "cpu") == 1.0
