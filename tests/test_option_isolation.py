"""Isolated options (basal.json "option_isolation"): the prompt's pieces are tokenized one by one, each option block
reads the prefix and itself (never another option) at shared positions, and the suffix reads all of them. Every
readout of the engine's own eager Gemma 4 forward equals the transformers forward with the same masks and positions
(fp32, the tiny random models of test_gemma4); the blocks are read as a set; the server's refusals."""
import argparse
import json

import pytest
import torch

from basal import engine
from basal.engine import EagerBackend
from basal.gemma4 import Gemma4Forward
from basal.prompt import (OptionIds, OptionPrompt, check_option_tokenizer, model_option_isolation, option_pieces,
                          option_tokens, render)
from basal.server import add_engine_args, models_payload, option_isolation_refused
from test_gemma4 import CONFIGS, last_hidden, tiny


def ids_of(blocks, seed=0):
    """An OptionIds of random tokens with these block lengths (prefix, options..., suffix)."""
    g = torch.Generator().manual_seed(seed)
    out = OptionIds(torch.randint(1, 56, (sum(blocks),), generator=g).tolist())
    out.blocks = tuple(blocks)
    return out


def ref_inputs(m, p):
    """Masks per layer type and positions of one OptionIds prompt, as the trainer builds them."""
    bl = p.blocks
    P, n, L = bl[0], len(bl) - 2, max(bl[1:-1])
    seg = torch.tensor([0] * P + [k + 1 for k in range(n) for _ in range(bl[k + 1])] + [n + 1] * bl[-1])
    pos = torch.tensor(list(range(P)) + [P + i for b in bl[1:-1] for i in range(b)] + [P + L + i for i in range(bl[-1])])
    i = torch.arange(len(p))
    opt = (seg > 0) & (seg <= n)
    m_ = (i[:, None] >= i[None]) & ~(opt[:, None] & opt[None] & (seg[:, None] != seg[None]))
    sl = m_ & (pos[:, None] - pos[None] < m.config.sliding_window)
    return {"full_attention": m_[None, None], "sliding_attention": sl[None, None]}, pos[None]


def ref_hidden(m, p):
    masks, pos = ref_inputs(m, p)
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([list(p)]), attention_mask=masks, position_ids=pos).last_hidden_state[0, -1]


def close(h, r):
    return (h - r).abs().max() <= 1e-4 * r.abs().max()


GROUPS = [[(12, 4, 6, 3, 5), (12, 6, 4, 3, 5)], [(30, 9, 2, 7)], [(3, 5, 5, 5, 5, 5, 4)], [(40, 2, 30, 6)]]


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("answer_rows", [True, False])
def test_isolated_equals_transformers(name, answer_rows):
    m = tiny(name)
    g = Gemma4Forward(m, answer_rows=answer_rows, option_isolation=True)
    groups = [[ids_of(b, 10 * gi + j) for j, b in enumerate(grp)] for gi, grp in enumerate(GROUPS)]
    for grp, hs in zip(groups, g.hidden(groups)):
        for p, h in zip(grp, hs):
            assert close(h, ref_hidden(m, p)), (name, p.blocks)


def test_blocks_are_read_as_a_set():
    """The same blocks in another order (each block keeps its tokens) give the same readout; a causal read does not."""
    m = tiny("e4b")
    p = ids_of((10, 4, 7, 5, 6))
    bl, cuts = p.blocks, [0]
    for b in bl:
        cuts.append(cuts[-1] + b)
    piece = [list(p[cuts[k]: cuts[k + 1]]) for k in range(len(bl))]
    order = [0, 3, 1, 2, 4]
    q = OptionIds(x for k in order for x in piece[k])
    q.blocks = tuple(bl[k] for k in order)
    g = Gemma4Forward(m, option_isolation=True)
    a, b = (g.hidden([[x]])[0][0] for x in (p, q))
    assert close(a, b)
    assert not close(last_hidden(m, list(p)), last_hidden(m, list(q)))
    assert not close(a, last_hidden(m, list(p)))  # isolation reads differently from the causal read


def test_one_way_unchanged():
    m = tiny("31b")
    groups = [[list(ids_of(b)) for b in grp] for grp in GROUPS]
    a = Gemma4Forward(m).hidden(groups)
    b = Gemma4Forward(m, option_isolation=False).hidden(groups)
    assert all(torch.equal(x, y) for ga, gb in zip(a, b) for x, y in zip(ga, gb))
    with pytest.raises(AssertionError, match="blocks"):
        Gemma4Forward(m, option_isolation=True).hidden([[[1, 2, 3, 4, 5]]])


def test_model_option_isolation(tmp_path):
    assert model_option_isolation(tmp_path) is False
    for v, want in [({}, False), ({"option_isolation": False}, False), ({"option_isolation": True}, True)]:
        (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, **v}))
        assert model_option_isolation(tmp_path) is want
    (tmp_path / "basal.json").write_text(json.dumps({"option_isolation": "yes"}))
    with pytest.raises(ValueError):
        model_option_isolation(tmp_path)


@pytest.fixture(scope="module")
def gtok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    except Exception as e:  # no network and no cache
        pytest.skip(f"tokenizer unavailable: {e}")


def test_option_pieces_and_tokens(gtok):
    opts = ["zwrot pieniędzy w ciągu 14 dni", "naprawa\ngwarancyjna", "inne"]
    p = render(gtok, "Stan: klient zgłasza wadę.", "Jaka decyzja?", opts, fmt=2)
    pre, blocks, suf = option_pieces(p, opts)
    assert pre + "".join(blocks) + suf == p and all(b.startswith("\n") for b in blocks)
    ids = option_tokens(gtok, p, opts)
    assert isinstance(ids, OptionIds) and len(ids.blocks) == 5 and sum(ids.blocks) == len(ids)
    assert list(ids) == gtok(p, add_special_tokens=False).input_ids  # Gemma 4: the pieces tokenize as the whole
    check_option_tokenizer(gtok)
    op = OptionPrompt(p)
    op.options = opts
    assert op == p and isinstance(op, str) and op + "A" == p + "A"


def test_eager_backend_and_server_refusals(tmp_path, monkeypatch):
    tiny("e4b").save_pretrained(tmp_path)
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())
    monkeypatch.setattr(engine, "check_option_tokenizer", lambda t: None)
    assert EagerBackend(tmp_path, "float32", device="cpu").g4.iso is False
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "option_isolation": True}))
    assert EagerBackend(tmp_path, "float32", device="cpu").g4.iso is True
    with pytest.raises(SystemExit, match="isolated options"):
        EagerBackend(tmp_path, "float32", device="cpu", forward="hf")
    ap = add_engine_args(argparse.ArgumentParser())
    assert option_isolation_refused("eager", ap.parse_args([])) is None
    assert "--soam on" in option_isolation_refused("eager", ap.parse_args(["--soam", "on"]))
    assert "--state-cache-gb" in option_isolation_refused("eager", ap.parse_args(["--state-cache-gb", "4"]))
    assert "--mode eager" in option_isolation_refused("graph", ap.parse_args([]))
    assert "forward" in option_isolation_refused("eager", ap.parse_args(["--eager-forward", "hf"]))
    assert "--reread" in option_isolation_refused("eager", ap.parse_args(["--reread", "0.7"]))
    assert models_payload("m", "eager", {}, option_isolation=True)["models"][0]["option_isolation"] is True
    assert "option_isolation" not in models_payload("m", "eager", {})["models"][0]


def test_server_prompts_carry_their_pieces(gtok):
    """Server.jobs_for / encode of a model with isolated options: OptionPrompt jobs, then OptionIds with the blocks of
    each option order (the ids of the whole prompt for Gemma 4), the code ids of the whole prompt."""
    from types import SimpleNamespace
    from basal.server import Server
    fake = SimpleNamespace(tok=gtok, fmt=2, layout="canonical", iso=True, orders="both", letters={}, max_options=255,
                           backend=SimpleNamespace())
    fake.code_ids = lambda prompt, k: Server.code_ids(fake, prompt, k)
    q = dict(state="Stan: klient zgłasza wadę towaru.", question="Jaka decyzja?", lang="pl",
             options=["zwrot", "naprawa gwarancyjna", "odmowa", "inne"])
    jobs = Server.jobs_for(fake, q, perms=[[0, 1, 2, 3], [3, 2, 1, 0]])
    assert all(isinstance(p, OptionPrompt) for _, p, _ in jobs) and jobs[1][1].options == q["options"][::-1]
    br = [(0, 0, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs])]
    n, longest = Server.encode(fake, br)
    ids = br[0][3]
    assert all(isinstance(x, OptionIds) and len(x.blocks) == 6 for x in ids)
    assert [list(x) for x in ids] == gtok([j[1] for j in jobs], add_special_tokens=False).input_ids
    assert n == sum(map(len, ids)) and longest == [max(map(len, ids))]
