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


def test_a_group_shared_by_lines_with_other_first_groups_stays_on_one_side(tmp_path):
    """A per-question group joins the lines that share it, whatever their first group: no group is on both sides."""
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "Alpha", "b": "Beta"}, "gold": "a"}
    train_g = next(str(n) for n in range(100) if not finetune.in_holdout(str(n), 0.25, 0))
    held_g = next(str(n) for n in range(100) if finetune.in_holdout(str(n), 0.25, 0))
    lines = [{"state": "s", "questions": {"first": dict(q, group=g), "second": dict(q, group="shared-customer")}}
             for g in (train_g, held_g, train_g + "x", held_g + "y")]
    lines += [{"state": "s", "group": f"alone{i}", "questions": {"q": q}} for i in range(40)]   # one group per line
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
    train, held, _ = finetune.split(p, 0.25, 0)
    held_groups = {x.get("group") for line in held for x in json.loads(line)["questions"].values()} | \
        {json.loads(line).get("group") for line in held}
    train_groups = {d["group"] for _, d in train}
    assert not (held_groups - {None}) & train_groups
    assert {"shared-customer", train_g, held_g} <= train_groups    # the unit goes with its first line's first group
    for i in range(40):  # a line that shares no group keeps the side of its own group, as before
        assert (f"alone{i}" in train_groups) != finetune.in_holdout(f"alone{i}", 0.25, 0)


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
    class Toy(torch.nn.Module):  # the head's input at the last position: [1, 0, 0, 0]; code rows 3 -> 4.0, 5 -> 8.0
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 8, bias=False)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.weight[3, 0], self.head.weight[5, 0] = 4.0, 8.0

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, logits_to_keep=None):
            assert logits_to_keep == 1
            return SimpleNamespace(logits=self.head(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])))
    got = finetune.code_logits(Toy(), [1, 2, 3], [5, 3, 0], None, "cpu")
    assert got.tolist() == [8.0, 4.0, 0.0]
    capped = finetune.code_logits(Toy(), [1], [5, 3], 2.0, "cpu")
    assert capped.tolist() == pytest.approx([2 * torch.tanh(torch.tensor(4.0)).item(), 2 * torch.tanh(torch.tensor(2.0)).item()])
    ex = [[([1, 2], [5, 3, 0], 0), ([1, 2], [5, 3, 0], 2)]]
    assert finetune.accuracy(Toy(), ex, None, "cpu") == 1.0


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_code_logits_stay_fp32_under_the_training_autocast(softcap):
    """The training steps run the forward under autocast (bf16 / fp16); the code readout stays the served one, in fp32
    (under autocast the head's input times the code rows was a bf16 product: logits of 16-32 step by 0.125)."""
    h = torch.tensor([[[1.2345, -0.777, 2.5, 0.111]]])

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 8, bias=True)
            with torch.no_grad():
                self.head.weight.copy_(torch.linspace(-3, 3, 32).view(8, 4) * 7.123)
                self.head.bias.copy_(torch.linspace(-1, 1, 8) / 3)

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, logits_to_keep=None):
            return SimpleNamespace(logits=self.head(h))
    m, codes = Toy(), [5, 3, 0]
    want = h[0, -1] @ m.head.weight[codes].T + m.head.bias[codes]
    want = torch.tanh(want / softcap) * softcap if softcap else want
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = finetune.code_logits(m, [1], codes, softcap, "cpu")
    assert got.dtype == torch.float32 and torch.allclose(got, want, rtol=0, atol=1e-5)


@pytest.mark.parametrize("kind", ["gemma", "llama"])
def test_code_logits_are_the_served_readout_capped_once(kind):
    """On real model classes: a Gemma 4 forward caps its logits itself (final_logit_softcapping), a Llama one has no
    cap; the trained code logits are the served ones (the softmax over the codes of the model's own logits), and their
    gradients agree."""
    from test_engine import tiny as tiny_llama
    from test_gemma4_graph import tiny as tiny_gemma
    m = tiny_gemma("e4b", torch.float32) if kind == "gemma" else tiny_llama()
    cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    cap = getattr(cfg, "final_logit_softcapping", None)
    assert (cap is not None) == (kind == "gemma")
    ids, codes = [3, 9, 4, 17, 22, 5, 30, 8], [7, 11, 2]
    z = finetune.code_logits(m, ids, codes, cap, "cpu")
    ref = m(input_ids=torch.tensor([ids])).logits[0, -1].float()[codes]
    assert torch.allclose(torch.softmax(z, -1), torch.softmax(ref, -1), atol=1e-6)
    w = next(p for n, p in m.named_parameters() if "layers.0" in n and p.dim() == 2)
    g1 = torch.autograd.grad(torch.nn.functional.cross_entropy(z[None], torch.tensor([1])), w)[0]
    g2 = torch.autograd.grad(torch.nn.functional.cross_entropy(ref[None], torch.tensor([1])), w)[0]
    assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-5)

def test_refuses_models_read_with_other_masks(tmp_path):
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2}))
    assert finetune.refused(tmp_path) is None
    for key in ("two_way", "option_isolation"):
        (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, key: True}))
        assert key in finetune.refused(tmp_path)


def test_lora_targets_the_decoder_layers_only():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.ModuleDict({"encoder": torch.nn.ModuleDict(
                {"layers": torch.nn.ModuleList([torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)})])})})
            self.language_model = torch.nn.ModuleDict({"layers": torch.nn.ModuleList(
                [torch.nn.ModuleDict({"self_attn": torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)}),
                                      "input_layernorm": torch.nn.LayerNorm(2)})])})
            self.lm_head = torch.nn.Linear(2, 3)
    m = torch.nn.Module()
    m.model = Mod()
    assert finetune.lora_targets(m) == ["model.language_model.layers.0.self_attn.q_proj"]


def test_merged_model_drops_the_base_models_calibration_and_nvfp4_scales(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(), out.mkdir()
    for name in ("tokenizer.json", "chat_template.jinja", "CALIBRATION.json", "CALIBRATION.nvfp4.json",
                 "NVFP4_AMAX.json", "evidence_head.pt", "model.safetensors", "basal.json"):
        (src / name).write_text("x")
    finetune.copy_side_files(src, out)
    assert sorted(f.name for f in out.iterdir()) == ["chat_template.jinja", "tokenizer.json"]
