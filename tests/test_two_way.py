"""Two-way reading (basal.json "two_way"): the global layers of the engine's own Gemma 4 forward read a whole prompt
both ways, the sliding-window layers stay causal, and a prompt over max_prompt tokens or a request over max_record
tokens is read one way. Every readout equals the transformers forward of the whole prompt with the same masks (fp32,
the tiny random models of test_gemma4)."""
import argparse
import json

import pytest
import torch

from basal import engine, gemma4
from basal.engine import EagerBackend
from basal.gemma4 import Gemma4Forward
from basal.prompt import TWO_WAY, model_two_way
from basal.server import add_engine_args, models_payload, two_way_refused
from test_gemma4 import CONFIGS, last_hidden, tiny, varlen_reference

CAPS = {"max_prompt": 40, "max_record": 120}


def ref_masks(m, n, two_way):
    """The masks of one prompt of n tokens, as transformers takes them per layer type (True = attends)."""
    i = torch.arange(n)
    causal = i[None] <= i[:, None]
    sliding = causal & (i[:, None] - i[None] < m.config.sliding_window)
    full = torch.ones(n, n, dtype=torch.bool) if two_way else causal
    return {"full_attention": full[None, None], "sliding_attention": sliding[None, None]}


def two_way_hidden(m, t):
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([t]), attention_mask=ref_masks(m, len(t), True)).last_hidden_state[0, -1]


def close(h, r):
    return (h - r).abs().max() <= 1e-4 * r.abs().max()


def groups_of():
    """A request of three questions over one state (record 69 tokens), a short single, a prompt over max_prompt, a
    group of prefixes, and a request over max_record whose prompts each fit max_prompt."""
    g = torch.Generator().manual_seed(1)
    r = lambda n: torch.randint(1, 56, (n,), generator=g).tolist()  # noqa: E731
    state, long_state = r(30), r(36)
    return [[state + [3, 4, 5], state + [3, 9], state + [7] * 7 + [1]], [r(10)], [r(50)], [[1, 2, 3, 4], [1, 2, 3, 5, 6]],
            [long_state + r(3) for _ in range(30)]]


def test_model_two_way(tmp_path):
    assert model_two_way(tmp_path) is None  # no basal.json
    for tw, want in [({}, None), ({"two_way": False}, None), ({"two_way": True}, TWO_WAY),
                     ({"two_way": {"max_prompt": 2048}}, {"max_prompt": 2048, "max_record": 8192})]:
        (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, **tw}))
        assert model_two_way(tmp_path) == want
    assert TWO_WAY == {"max_prompt": 4096, "max_record": 8192}
    for bad in ["yes", {"max_tokens": 10}, {"max_prompt": 0}, {"max_prompt": True}, {"max_record": 1.5}]:
        (tmp_path / "basal.json").write_text(json.dumps({"two_way": bad}))
        with pytest.raises(ValueError):
            model_two_way(tmp_path)


def test_two_way_rows():
    g = Gemma4Forward(tiny("e4b"), two_way=CAPS)
    assert g.two_way_rows([[1] * 40]) == [True] and g.two_way_rows([[1] * 41]) == [False]
    state = [5] * 30
    assert g.two_way_rows([state + [1] * 10, state + [2] * 11]) == [True, False]  # per prompt: max_prompt
    assert g.two_way_rows([state + [k] * 9 for k in range(10)]) == [True] * 10  # record 30 + 10 x 9 = 120
    assert g.two_way_rows([state + [k] * 9 for k in range(11)]) == [False] * 11  # 129 > max_record: all one way
    assert Gemma4Forward(tiny("e4b")).two_way_rows([[1] * 4]) == [False]


def test_mask_per_layer_type():
    g = Gemma4Forward(tiny("e4b"), two_way=CAPS)
    pos = torch.arange(6)[None].repeat(2, 1)
    valid = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    for two_way in (False, True):
        for t in ("full_attention", "sliding_attention"):
            m = g._mask(t, pos, None, pos, valid, None, two_way)
            assert m.shape == (2, 1, 6, 6)
            if two_way and t == "full_attention":  # every real token, a padding query itself only
                want = (valid[:, None, :] | torch.eye(6, dtype=torch.bool)[None])[:, None]
            else:
                want = g._mask(t, pos, None, pos, valid, None)
            assert torch.equal(m, want)
    causal = g._mask("full_attention", pos, None, pos, valid, None)
    assert torch.equal(causal[0, 0], torch.ones(6, 6, dtype=torch.bool).tril())


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("answer_rows", [True, False])
def test_two_way_equals_transformers(name, answer_rows):
    m = tiny(name)
    g = Gemma4Forward(m, answer_rows=answer_rows, two_way=CAPS)
    g.CHUNK = 24  # one-way prompts run in parts; two-way prompts are read whole
    groups = groups_of()
    for grp, hs in zip(groups, g.hidden(groups)):
        for t, h, tw in zip(grp, hs, g.two_way_rows(grp)):
            assert close(h, two_way_hidden(m, t) if tw else last_hidden(m, t)), (name, len(t), tw)
    assert [any(g.two_way_rows(grp)) for grp in groups] == [True, True, False, True, False]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_two_way_with_varlen_kernels(name, monkeypatch):
    """Varlen (flash) kernels are causal only: a two-way forward's global layers take the masked SDPA kernel, its
    sliding-window layers (and every one-way forward) the varlen kernel asked for."""
    monkeypatch.setitem(gemma4.VARLEN_KERNELS, "ref", varlen_reference)
    monkeypatch.setattr(Gemma4Forward, "_kernel", lambda self, t, dtype, blk: self.attn[t])
    m = tiny(name)
    g = Gemma4Forward(m, two_way=CAPS, attn={"sliding_attention": "ref", "full_attention": "ref"})
    g.CHUNK = 24
    groups = groups_of()
    for grp, hs in zip(groups, g.hidden(groups)):
        for t, h, tw in zip(grp, hs, g.two_way_rows(grp)):
            assert close(h, two_way_hidden(m, t) if tw else last_hidden(m, t)), (name, len(t), tw)


def test_two_way_reads_differently():
    m = tiny("e4b")
    t = groups_of()[0][0]
    one = Gemma4Forward(m).hidden([[t]])[0][0]
    two = Gemma4Forward(m, two_way=CAPS).hidden([[t]])[0][0]
    assert close(two, two_way_hidden(m, t)) and not close(two, one)
    # the reference's sliding layers are causal: reading them two-way as well gives another answer
    with torch.no_grad():
        both = m.model(input_ids=torch.tensor([t]), attention_mask={
            k: torch.ones(1, 1, len(t), len(t), dtype=torch.bool) for k in ("full_attention", "sliding_attention")})
    assert not close(two, both.last_hidden_state[0, -1])


def test_one_way_unchanged():
    m = tiny("31b")
    groups = groups_of()
    a = Gemma4Forward(m).hidden(groups)
    b = Gemma4Forward(m, two_way=None).hidden(groups)
    assert all(torch.equal(x, y) for ga, gb in zip(a, b) for x, y in zip(ga, gb))


def test_eager_backend_reads_basal_json(tmp_path, monkeypatch):
    tiny("e4b").save_pretrained(tmp_path)
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())
    assert EagerBackend(tmp_path, "float32", device="cpu").g4.two_way is None
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "two_way": True}))
    assert EagerBackend(tmp_path, "float32", device="cpu").g4.two_way == TWO_WAY
    with pytest.raises(SystemExit, match="two-way"):
        EagerBackend(tmp_path, "float32", device="cpu", forward="hf")


def test_server_refusals():
    ap = add_engine_args(argparse.ArgumentParser())
    a = ap.parse_args([])
    assert a.soam is None  # resolved by the server: on, off for a two-way model
    assert two_way_refused("eager", a) is None
    assert "--soam on" in two_way_refused("eager", ap.parse_args(["--soam", "on"]))
    assert "state" in two_way_refused("graph", ap.parse_args(["--state-cache-gb", "4"]))
    assert "--mode eager" in two_way_refused("graph", a)
    assert "--mode eager" in two_way_refused("vllm", a)
    assert "forward" in two_way_refused("eager", ap.parse_args(["--eager-forward", "hf"]))
    entry = models_payload("m", "eager", {}, two_way=TWO_WAY)["models"][0]
    assert entry["two_way"] == TWO_WAY
    assert "two_way" not in models_payload("m", "eager", {})["models"][0]
