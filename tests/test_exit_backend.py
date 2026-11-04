"""--mode fast-exit start-up (ExitGraphBackend, built on the CPU with the graph engine's loading stubbed): an unknown
--early-exit default level, or thresholds for a layer without a head, are refused at start-up (not a KeyError in every
request), and the pass-1 key / value buffer, which its segments never read, is released."""
import json

import pytest
import torch


from basal.engine import ExitGraphBackend, ExitHead, GraphBackend
from test_engine import tiny


def exit_dir(tmp_path, layers=(1,), levels=None):
    m = tiny()
    H = m.config.hidden_size
    heads = torch.nn.ModuleDict({str(L): ExitHead(m.model.norm, H, 4) for L in layers})
    d = tmp_path / "exit_heads"
    d.mkdir()
    torch.save({"rank": 4, "layers": list(layers), "state": heads.state_dict()}, d / "heads.pt")
    levels = levels or {"0.99": {"taus": {"1": 0.9}}}
    (d / "thresholds.json").write_text(json.dumps({"levels": levels}))
    return m, d


def build(monkeypatch, m, d, policy):
    def stub_init(self, *a, **kw):
        self.model, self.dev, self.TOKEN_BUDGET = m, torch.device("cpu"), 64
        self._kv = [(torch.zeros(8), torch.zeros(8))]
    monkeypatch.setattr(GraphBackend, "__init__", stub_init)
    monkeypatch.setattr(ExitGraphBackend, "_graph_exit", lambda self, b, L: None)
    return ExitGraphBackend("unused", heads_dir=d, default_policy=policy)


def test_known_default_level_starts_and_drops_the_kv_buffer(tmp_path, monkeypatch):
    m, d = exit_dir(tmp_path)
    be = build(monkeypatch, m, d, "0.99")
    assert be.default_policy == "0.99" and be.exits == [1] and be._kv is None


def test_unknown_default_level_is_refused(tmp_path, monkeypatch):
    m, d = exit_dir(tmp_path)
    with pytest.raises(ValueError, match="--early-exit 0.97: not a level"):
        build(monkeypatch, m, d, "0.97")


def test_thresholds_without_a_head_are_refused(tmp_path, monkeypatch):
    m, d = exit_dir(tmp_path, layers=(1,), levels={"0.99": {"taus": {"1": 0.9, "0": 0.95}}})
    with pytest.raises(ValueError, match=r"without an exit head in heads.pt: \[0\]"):
        build(monkeypatch, m, d, "off")
