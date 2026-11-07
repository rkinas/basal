"""The lean layer function reproduces transformers' Llama forward (fp32, CPU): plain rows with the keys / values
written for pass 2, and pass 2 over a prefix with an additive mask; merged projections and FP8 linears included."""
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import FP8Linear, causal_sdpa, merge_projections
from basal.lean import Lean, linear, supports


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, attention_bias=True, mlp_bias=True, max_position_embeddings=128,
                      rope_theta=10000.0)
    return causal_sdpa(LlamaForCausalLM(cfg).eval())


def kv_buffers(m, b, L):
    return [(torch.zeros(b, 2, L, 8), torch.zeros(b, 2, L, 8)) for _ in m.model.layers]


def check(m):
    lean = Lean(m, compile=False)
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        kv_ref, kv = kv_buffers(m, 2, 10), kv_buffers(m, 2, 10)
        ref = m.model(input_ids=ids, use_cache=False, basal_kv=kv_ref).last_hidden_state
        got = lean.forward(ids, kv)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()
        assert all(torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
                   for a, b in zip(kv_ref, kv))
        # pass 2: 4 tokens per row over the 10-token prefix, a causal mask inside the suffix
        sfx = torch.randint(0, 64, (2, 4))
        pos = torch.arange(10, 14)[None].expand(2, 4)
        allow = torch.cat([torch.ones(4, 10, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool).tril()], -1)
        mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None].expand(2, 1, 4, 14).contiguous()
        ref2 = m.model(input_ids=sfx, attention_mask=mask, position_ids=pos, use_cache=False,
                       basal_prefix=kv_ref).last_hidden_state
        got2 = lean.forward_suffix(sfx, mask, pos, kv)
        assert torch.allclose(ref2, got2, atol=1e-5), (ref2 - got2).abs().max()
        # the same continuation computed in one forward over prefix + suffix
        full = m.model(input_ids=torch.cat([ids, sfx], 1), use_cache=False).last_hidden_state[:, 10:]
        assert torch.allclose(full, got2, atol=1e-4)


def test_lean_matches_transformers():
    m = tiny()
    assert supports(m)
    check(m)


def test_lean_with_merged_projections():
    m = tiny()
    merge_projections(m)
    check(m)


def test_fp8_linear_arithmetic_is_fp8linear():
    torch.manual_seed(2)
    w = torch.randn(24, 32)
    s = w.abs().amax() / 448
    f = FP8Linear((w / s).to(torch.float8_e4m3fn), s, torch.tensor(0.02), torch.randn(24))
    x = torch.randn(3, 5, 32)
    assert torch.equal(linear(x, (f.weight, f.weight_scale, f.input_scale, f.bias)), f(x))


def test_flex_tree_mask_is_the_dense_mask():
    """Pass 2 of a two-order question and of a three-branch trie: FlexAttention with the TreeMask gives the same rows
    as the additive dense mask (GraphBackend._mask_from_seg plus the root part)."""
    import numpy as np

    from basal.engine import GraphBackend
    from basal.lean import TreeMask, tree_mask_mod
    m = tiny()
    lean = Lean(m, compile=False)
    L, S, B = 10, 6, 2
    roots = [10, 7]                                       # row 1's root is shorter than the bucket
    seg = np.array([[0, 0, 1, 1, 1, -1], [0, 1, 1, 2, 2, 2]])
    parents = [[-1, -1], [-1, 0, 0]]                      # row 1: blocks 1 and 2 continue block 0
    stub = type("Stub", (), {"_to_dev": staticmethod(torch.as_tensor), "model": m,
                             "_ancestors": staticmethod(GraphBackend._ancestors)})()
    tm = GraphBackend._tree_mask(stub, seg, parents, roots, S)
    m2 = GraphBackend._mask_from_seg(stub, torch.from_numpy(seg), parents)
    root = torch.tensor(roots)[:, None] > torch.arange(L)[None]
    pre = torch.where(root, 0.0, torch.finfo(torch.float32).min)[:, None, None].expand(B, 1, S, L)
    dense = torch.cat([pre, m2], -1)
    mod = tree_mask_mod(tm, L)
    allowed = torch.tensor([[[bool(mod(torch.tensor(b), 0, torch.tensor(q), torch.tensor(kv)))
                              for kv in range(L + S)] for q in range(S)] for b in range(B)])
    assert torch.equal(allowed, dense[:, 0] == 0)
    ids = torch.randint(0, 64, (B, L))
    kv = kv_buffers(m, B, L)
    with torch.no_grad():
        lean.forward(ids, kv)
        sfx, pos = torch.randint(0, 64, (B, S)), torch.arange(L, L + S)[None].expand(B, S)
        ref = lean.forward_suffix(sfx, dense, pos, kv)
        got = lean.forward_suffix(sfx, tm, pos, kv)
    real = torch.from_numpy(seg >= 0)
    assert torch.allclose(ref[real], got[real], atol=1e-5), (ref[real] - got[real]).abs().max()
    assert isinstance(tm, TreeMask) and tm.anc.shape == (B, S + 1, S + 1)


@pytest.mark.parametrize("entry", ["serve", "calibrate", "think"])
def test_flex_with_lean_on_refused_before_loading(monkeypatch, tmp_path, entry):
    """--lean on with --tree flex fails at CUDA graph capture (FlexAttention falls back to its uncompiled kernel). The
    refusal comes before the model is resolved, also on a CUDA GPU without FlashAttention 3 (simulated here), where
    choosing the default mode resolves the model to see whether it is Gemma 4: from basal-serve, basal-calibrate with
    the engine in process, and a ThinkServer built directly (it resolves the model and starts its sidecar first)."""
    import sys

    import basal.calibrate as calibrate_module
    import basal.server as server_module
    import basal.think as think_module
    for module in (server_module, calibrate_module, think_module):
        monkeypatch.setattr(module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # e.g. an RTX 4060 Ti: CUDA, no FlashAttention 3
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: None)  # and not sm_120: Gemma 4 has no fast
    args = ["--lean", "on", "--tree", "flex"]
    with pytest.raises(SystemExit, match="--tree flex runs with --lean pass2"):
        if entry == "serve":
            monkeypatch.setattr(sys, "argv", ["basal-serve", *args])
            server_module.main()
        elif entry == "calibrate":
            (tmp_path / "l.jsonl").write_text(json.dumps({"state": "x", "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"], "gold": "a"}}}) + "\n", encoding="utf-8")
            calibrate_module.main([str(tmp_path / "l.jsonl"), *args, "--out", str(tmp_path / "c.json")])
        else:
            think_module.ThinkServer(server_module.parser().parse_args([*args, "--think", "auto"]))


def test_lean_tree_auto():
    """--lean / --tree auto: the FlexAttention pass 2 on the compiled CUDA graph engine without torchao, else dense."""
    from basal.server import lean_tree
    assert lean_tree("auto", "auto", "graph") == ("pass2", "flex")
    assert lean_tree("auto", "auto", "graph", compiled=False) == ("off", "dense")
    assert lean_tree("auto", "auto", "graph", quant="fp8") == ("off", "dense")
    assert lean_tree("auto", "auto", "mlx") == ("off", "dense")
    assert lean_tree("auto", "dense", "graph") == ("off", "dense")
    assert lean_tree("auto", "flex", "graph") == ("pass2", "flex")
    assert lean_tree("on", "auto", "graph") == ("on", "dense")
    assert lean_tree("pass2", "dense", "graph") == ("pass2", "dense")


@pytest.mark.parametrize("env, want", [(None, {"options": {"triton.autotune_pointwise": False}}), ("1", {})])
def test_compiled_llama_functions_time_no_pointwise_kernels(monkeypatch, env, want):
    """The compiled Llama code (the lean layer here; the forwards, the early-exit segments and the cascade's merge
    through the same compile_llama) has Inductor's pointwise autotuning off: a pointwise kernel's block shapes are
    timed within noise of each other and some give other last bits, so each cold start of basal-2.0-mini could keep
    other shapes and give other answers (3 sets in 4 starts on an H100). BASAL_POINTWISE_AUTOTUNE=1 times them."""
    import basal.engine as E
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: seen.append(kw) or fn)
    if env is None:
        monkeypatch.delenv("BASAL_POINTWISE_AUTOTUNE", raising=False)
    else:
        monkeypatch.setenv("BASAL_POINTWISE_AUTOTUNE", env)
    Lean(tiny(), compile=True)
    E._merge_fused.__wrapped__()  # (functools.cache: one compile per process)
    assert seen == [{**want, "dynamic": True}, {**want, "dynamic": True, "fullgraph": True}]
