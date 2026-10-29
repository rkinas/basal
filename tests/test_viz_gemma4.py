"""basal-viz on Gemma 4 basal-omni: the logit lens over the text decoder matches the eager engine's served code
probabilities (head bias, final logit softcap), and media requests are refused."""
from types import SimpleNamespace

import pytest
import torch

from basal import viz
from basal.engine import EagerBackend
from basal.gemma4 import Gemma4Forward
from tests.test_engine import tiny as tiny_llama
from tests.test_gemma4 import tiny


class Tokenizer:
    def decode(self, ids):
        return str(ids[0])


def explainer_for(model):
    ex = viz.Explainer.__new__(viz.Explainer)
    ex.gemma4 = viz.is_gemma4(model)
    ex.model = model
    ex.text_model = model.model
    ex.head = model.get_output_embeddings()
    ex.softcap = getattr(model.config, "final_logit_softcapping", None)
    ex.dev = "cpu"
    ex.tok = Tokenizer()
    return ex


@pytest.mark.parametrize("codes", [[1, 2], [1, 2, 3, 4], list(range(1, 6)), list(range(1, 13))],
                         ids=["noul", "choice", "score", "many-options"])
def test_gemma4_lens_matches_eager_served_probabilities_at_every_layer(codes):
    model = tiny("e4b")
    ex = explainer_for(model)
    ids = [4, 5, 8, 3, 12, 6]

    layers, _ = ex.lens(ids, codes)

    assert len(layers) == model.config.num_hidden_layers
    assert all(sum(p) == pytest.approx(1.0, abs=1e-6) for p in layers)
    eager = object.__new__(EagerBackend)
    eager.head = model.get_output_embeddings()
    eager.softcap = model.config.final_logit_softcapping
    hidden = Gemma4Forward(model).hidden([[ids]])[0][0]
    served = eager._codes(hidden[None].float(), [codes])[0]
    assert layers[-1] == pytest.approx(served, abs=1e-6)
    with torch.no_grad():
        states = model.model(input_ids=torch.tensor([ids]), use_cache=False, output_hidden_states=True).hidden_states
        hidden = torch.stack([model.model.norm(h[0, -1]) for h in states[1:-1]] + [states[-1][0, -1]])
        reference = eager._codes(hidden.float(), [codes] * len(layers))
    for actual, expected in zip(layers, reference):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_gemma4_lens_applies_final_logit_softcap():
    model = tiny("e4b")
    ex = explainer_for(model)
    ids, codes = [4, 5, 8, 3, 12, 6], list(range(1, 6))

    layers, _ = ex.lens(ids, codes)
    hidden = Gemma4Forward(model).hidden([[ids]])[0][0]
    raw = torch.softmax(hidden.float() @ model.get_output_embeddings().weight[codes].float().T, -1).tolist()

    assert max(abs(a - b) for a, b in zip(layers[-1], raw)) > 1e-3


def test_llama_lens_keeps_its_original_probabilities():
    model = tiny_llama()
    ex = explainer_for(model)
    ids, codes = [4, 5, 8, 3, 12, 6], [1, 2, 3, 4]
    layers, _ = ex.lens(ids, codes)
    with torch.no_grad():
        expected = torch.softmax(model(input_ids=torch.tensor([ids])).logits[0, -1, codes], -1).tolist()
    assert len(layers) == model.config.num_hidden_layers
    assert layers[-1] == pytest.approx(expected, abs=1e-6)


def test_gemma4_lens_does_not_retain_full_sequence_hidden_states(monkeypatch):
    model = tiny("e4b")
    ex = explainer_for(model)
    forward = ex.text_model.forward
    norm = ex.text_model.norm.forward
    seen, storage = {}, []

    def check_forward(*args, **kwargs):
        seen["output_hidden_states"] = kwargs.get("output_hidden_states")
        return forward(*args, **kwargs)

    monkeypatch.setattr(ex.text_model, "forward", check_forward)

    def check_norm(hidden):
        if hidden.shape == (1, model.config.hidden_size):
            storage.append((hidden.untyped_storage().nbytes(), hidden.numel() * hidden.element_size()))
        return norm(hidden)

    monkeypatch.setattr(ex.text_model.norm, "forward", check_norm)
    layers, _ = ex.lens([i % 55 + 1 for i in range(512)], [1, 2, 3, 4])

    assert seen["output_hidden_states"] is False
    assert len(layers) == model.config.num_hidden_layers
    assert len(storage) == model.config.num_hidden_layers - 1
    assert all(allocated == needed for allocated, needed in storage)


def test_long_gemma4_inspection_is_rejected_before_a_forward(monkeypatch):
    ex = explainer_for(tiny("e4b"))

    def no_forward(*_, **__):
        pytest.fail("long inspections must be rejected before allocating full-sequence attention")

    monkeypatch.setattr(ex.text_model, "forward", no_forward)
    with pytest.raises(ValueError, match="at most 2048 prompt tokens") as error:
        ex.lens([1] * (EagerBackend.CHUNK + 1), [1, 2])
    assert viz.http_status(error.value) == 422
    assert all(not layer._forward_hooks for layer in ex.text_model.layers)


def test_gemma4_explainer_accepts_text_model_and_refuses_other_architectures(monkeypatch):
    model = tiny("e4b")

    class Server:
        def __init__(self, _args):
            self.backend = SimpleNamespace(tok=Tokenizer(), model=model, dev="cpu")
            self.fmt, self.orders, self.cal, self.name = 1, 1, {}, "tiny"

    monkeypatch.setattr(viz, "Server", Server)
    args = SimpleNamespace(model="tiny", device="cpu", dtype="bfloat16", orders=None, calibration=None)
    assert viz.Explainer(args).info["layers"] == model.config.num_hidden_layers

    unsupported = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    def unsupported_server(self, _args):
        self.backend = SimpleNamespace(tok=Tokenizer(), model=unsupported, dev="cpu")
        self.fmt, self.orders, self.cal, self.name = 1, 1, {}, "tiny"

    monkeypatch.setattr(Server, "__init__", unsupported_server)
    with pytest.raises(SystemExit, match="Llama-architecture models"):
        viz.Explainer(args)


def test_gemma4_explainer_refuses_media_requests():
    ex = viz.Explainer.__new__(viz.Explainer)
    with pytest.raises(ValueError, match="basal-viz reads text only: remove the media"):
        ex.explain({"media": [{"type": "image"}]})
