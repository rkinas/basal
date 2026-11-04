"""One code readout (basal/readout.py) for every torch engine and for basal-finetune: the softmax over the codes of the
model's own logits (bias, final soft cap once), fp32, one copy to the host; the engines and the fine-tune loss read the
same numbers."""
import torch

from basal import finetune
from basal.engine import EagerBackend, GraphBackend
from basal.readout import CodeReadout, capped
from test_engine import tiny as tiny_llama
from test_gemma4_graph import tiny as tiny_gemma


def own(m, h, ids):
    """The softmax over the codes of the model's own logits (the head and its final soft cap) of hidden state h."""
    cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    z = capped(m.get_output_embeddings()(h).float(), getattr(cfg, "final_logit_softcapping", None))
    return torch.softmax(z[ids], -1)


def test_readout_equals_the_models_own_logits_with_a_bias_and_a_cap():
    head = torch.nn.Linear(8, 20)
    with torch.no_grad():
        head.bias.normal_()
    h = torch.randn(3, 8)
    lids = [[3, 4], [5, 6, 7], [3, 4]]
    got = CodeReadout(head, softcap=2.0).probs(h, lids)
    for row, ids in zip(range(3), lids):
        want = torch.softmax(capped(head(h[row]).float(), 2.0)[ids], -1)
        assert torch.allclose(torch.tensor(got[row]), want, atol=1e-6)


def test_engines_and_finetune_read_the_same_numbers():
    """Eager, the graph engine's readout and basal-finetune's code logits: one function, the same probabilities, for a
    Gemma 4 model (soft cap) and a Llama one (none)."""
    for m in (tiny_gemma("e4b", torch.float32), tiny_llama()):
        cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
        cap = getattr(cfg, "final_logit_softcapping", None)
        h = torch.randn(4, cfg.hidden_size, generator=torch.Generator().manual_seed(1))
        lids = [[7, 11, 2], [7, 11], [5, 9], [7, 11, 2]]
        eager = EagerBackend.__new__(EagerBackend)
        eager.head, eager.softcap = m.get_output_embeddings(), cap
        graph = GraphBackend.__new__(GraphBackend)
        graph.model, graph.softcap, graph._wrows, graph.dev = m, cap, {}, torch.device("cpu")
        a, b = eager._codes(h, lids), graph._readout(h, lids)
        for row, ids in enumerate(lids):
            want = own(m, h[row], ids)
            assert torch.allclose(torch.tensor(a[row]), want, atol=1e-6)
            assert a[row] == b[row]  # one implementation: the same numbers
        ids = [3, 9, 4, 17, 22, 5, 30, 8]
        z = finetune.code_logits(m, ids, [7, 11, 2], cap, "cpu")
        with torch.no_grad():
            hl = m.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        assert torch.allclose(torch.softmax(z, -1), own(m, hl, [7, 11, 2]), atol=1e-6)
