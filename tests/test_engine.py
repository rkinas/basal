"""Shared-prefix packing is exactly equivalent to separate forward passes (tiny random model, CPU)."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import GraphBackend, MPSBackend, _block_plan, cascade_ok


class Stub(GraphBackend):
    def __init__(self, model):  # no loading, no CUDA graphs
        self.model, self.dev, self._wrows = model, torch.device("cpu"), {}


class MPSStub(MPSBackend):
    def __init__(self, model):  # the real __init__ minus loading, on the CPU (--mode mps --device cpu)
        self.model, self.dev, self._wrows, self.shared = model, torch.device("cpu"), {}, True
        self.tok = type("Tok", (), {"pad_token_id": 0})()


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


def separate(m, toks, lids):
    """Reference: one forward per prompt, full log-softmax over the vocabulary, softmax over the letters."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


def test_pack():
    ids, pos, seg, last, par = GraphBackend._pack([[1, 2, 3, 4, 5], [1, 2, 3, 7, 8, 9]])
    assert ids == [1, 2, 3, 4, 5, 7, 8, 9] and pos == [0, 1, 2, 3, 4, 3, 4, 5]
    assert seg == [0, 0, 0, 1, 1, 2, 2, 2] and last == [4, 7] and par == [-1, 0, 0]


def test_pack_trie_shares_nested_prefixes():
    """state | question 1 (two orders) | question 2 (two orders): the question text is shared by its orders."""
    s, q1, q2 = [1, 2, 3], [10, 11], [20, 21]
    toks = [s + q1 + [4, 5], s + q1 + [5, 4], s + q2 + [4, 5], s + q2 + [5, 4]]
    ids, pos, seg, last, par = GraphBackend._pack(toks)
    assert len(ids) == 3 + 2 * (2 + 2 * 2) and par == [-1, 0, 1, 1, 0, 4, 4]
    for m, tk in enumerate(toks):  # each readout's visible tokens (ancestors + own block) are exactly its prompt
        chain, b = [], seg[last[m]]
        while b >= 0:
            chain.append(b); b = par[b]
        seen = [ids[i] for i in range(last[m] + 1) if seg[i] in chain]
        assert seen == tk and pos[last[m]] == len(tk) - 1


def test_shared_equals_separate():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    be = Stub(m)
    a, b = [1, 5, 9, 11, 3, 4, 6], [1, 5, 9, 11, 3, 8, 2, 7]
    ids, pos, seg, last, par = GraphBackend._pack([a, b])
    h = be._forward_masked(torch.tensor([ids + [0] * 3]), be._mask_from_seg(torch.tensor([seg + [-1] * 3]), [par]),
                           torch.tensor([pos + [0] * 3]))
    with torch.no_grad():
        ha = m.model(input_ids=torch.tensor([a])).last_hidden_state[0, -1]
        hb = m.model(input_ids=torch.tensor([b])).last_hidden_state[0, -1]
    assert torch.allclose(h[0, last[0]], ha, atol=1e-4) and torch.allclose(h[0, last[1]], hb, atol=1e-4)


def test_soam_many_branches_equal_separate():
    """Every question x option order of a request packed into one row over the state (SOAM) equals separate prompts,
    with the row-only readout, including a row bisected because it exceeds the largest length."""
    m = tiny()
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14]
    toks = [state + [20 + q, 21, 30 + o, 31 - o, 40 + q, 41] for q in range(5) for o in range(2)]
    toks.append(list(toks[0]))  # a duplicate prompt reads out at the same position
    lids = [[7, 8, 9] if q % 2 else [7, 8] for q in range(5) for o in range(2)] + [[7, 8]]
    ref = separate(m, toks, lids)
    for lens in ([64], [24]):  # 24 < packed length: the two passes over the root's KV cache
        be.LENS = lens
        units = be._units([(toks, lids)])
        assert len(units) == 1
        probs = [be._eager_shared(pk, ul) if lens == [64] else be._long(pk, ul) for pk, ul, k, a in units]
        res = GraphBackend._assemble([(toks, lids)], units, probs)[0]
        for r, e in zip(res, ref):
            assert torch.allclose(torch.tensor(r), e, atol=1e-5)
    # prompts without a common first token have no root to share: a long group is bisected
    be.LENS = [24]
    other = [[2] + t[1:] for t in toks[:4]]
    units = be._units([(toks[:4] + other, lids[:4] * 2)])
    assert len(units) > 1 and all(pk[4].count(-1) == 1 or len(pk[0]) <= 24 for pk, *_ in units)
    res = GraphBackend._assemble([(toks[:4] + other, lids[:4] * 2)], units, [be._long(pk, ul) for pk, ul, *_ in units])[0]
    for r, e in zip(res, separate(m, toks[:4] + other, lids[:4] * 2)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-5)


def test_mps_backend_on_cpu_equals_separate():
    """The packed engine without CUDA graphs (MPSBackend) run on the CPU: several requests in one call, rows padded on
    the right to the longest, each request's state shared by its questions and option orders, and a request over the
    largest length through the long path; every readout equals its prompt run alone."""
    m = tiny()
    be = MPSStub(m)
    s1, s2 = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14], [2, 7, 8, 10, 15, 16]
    g1 = ([s1 + [20 + q, 21, 30 + o, 31 - o, 40 + q] for q in range(3) for o in range(2)], [[7, 8]] * 6)
    g2 = ([s2 + [22, 23, 24], s2 + [25, 26]], [[7, 8, 9], [7, 8]])
    g3 = ([[3, 4, 5]], [[7, 8]])
    groups = [g1, g2, g3]
    for lens in ([64], [12]):  # 12: g1's packed row (34 tokens) is over the largest length and takes the long path
        be.LENS = lens
        out = be.run_shared(groups)
        assert [len(r) for r in out] == [len(t) for t, _ in groups], lens  # every request, every readout
        for (toks, lids), res in zip(groups, out):
            for r, e, ids in zip(res, separate(m, toks, lids), lids):
                assert len(r) == len(ids) and torch.allclose(torch.tensor(r), e, atol=1e-5), lens


def test_mps_cpu_branch_pass_cascade_root_once():
    """On the CPU the long path's branch pass attends to the root and to each block's ancestors separately, merged by
    log-sum-exp (_cascade): a request with more tokens below its root than BRANCH_TOKENS stays one unit (the root runs
    once), its branch pass runs in parts of whole questions, and every readout equals its prompt run alone: 6 query
    heads per key / value head, questions in two option orders (nested blocks), a duplicate prompt and a prompt that
    ends at the root."""
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=48, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=12, num_key_value_heads=2, head_dim=8,
                                     max_position_embeddings=256)).eval()
    m.config._attn_implementation = "sdpa"
    be = MPSStub(m)
    be.LENS, be.BRANCH_TOKENS = [16], 24  # every unit through the long path; parts of at most 24 tokens below the root
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 2, 7]
    toks = [state + [20 + q, 21, 22 + q, 30 + o, 31 - o] for q in range(4) for o in range(2)]  # 4 x 7 tokens below
    toks += [list(toks[0]), list(state)]  # a duplicate, and a prompt that ends at the root (read out in pass 1)
    lids = [[7, 8, 9] if k % 3 else [7, 8] for k in range(len(toks))]
    assert len(be._units([(toks, lids)])) == 1  # 28 tokens below the root > BRANCH_TOKENS: not split
    out = be.run_shared([(toks, lids)])[0]
    assert len(out) == len(toks)
    for r, e, ids in zip(out, separate(m, toks, lids), lids):
        assert len(r) == len(ids) and torch.allclose(torch.tensor(r), e, atol=1e-5)


def test_mps_cpu_cascade_large_question_split_and_padding_bound():
    """A question whose own block (with its option orders) holds more than BRANCH_TOKENS tokens is split from the
    others, so its shared block becomes a root and nothing is quadratic in a long branch; and _block_plan's batches,
    padded to their longest block, stay within `waste` x the real query-key pairs (many one-token blocks next to one
    long one)."""
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=60, hidden_size=48, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=12, num_key_value_heads=2, head_dim=8,
                                     max_position_embeddings=256)).eval()
    m.config._attn_implementation = "sdpa"
    be = MPSStub(m)
    be.LENS, be.BRANCH_TOKENS = [16], 24
    state = [1, 5, 9, 11, 3, 4, 6, 12]
    long_q = list(range(30, 58))  # 28 tokens: more than BRANCH_TOKENS in one block
    toks = [state + long_q + [20 + o, 21 - o] for o in range(2)] + [state + [40 + q, 41] for q in range(3)]
    lids = [[7, 8]] * len(toks)
    units = be._units([(toks, lids)])
    assert len(units) > 1 and all(pk[4].count(-1) == 1 for pk, *_ in units)
    for r, e in zip(be.run_shared([(toks, lids)])[0], separate(m, toks, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-5)
    seg = [0] * 20 + list(range(1, 41))  # one 20-token block, then 40 one-token blocks under it
    parents = [-1] + [0] * 40
    plan = _block_plan(seg, parents)
    real = 20 * 20 + 40 * 21  # query-key pairs: the long block causally, each short block over 20 ancestors + itself
    padded = sum(qi.shape[0] * qi.shape[1] * ki.shape[1] for qi, ki, _ in plan)
    assert sorted(i for qi, _, _ in plan for i in qi.flatten().tolist() if i < len(seg)) == list(range(len(seg)))
    assert padded <= 1.5 * real


def test_cascade_only_for_attention_only_models():
    """Sliding-window and linear-attention layers need more than the root's keys / values and the block's ancestors:
    those models keep the dense branch pass over the transformers cache."""
    from transformers import Gemma4TextConfig, MistralConfig, Qwen3_5TextConfig
    assert cascade_ok(LlamaConfig()) and cascade_ok(MistralConfig(sliding_window=None))
    assert not cascade_ok(MistralConfig()) and not cascade_ok(Gemma4TextConfig())
    assert not cascade_ok(Qwen3_5TextConfig())  # Gated DeltaNet (linear attention) and attention layers


def test_cascade_needs_the_cpu_flash_kernel(monkeypatch):
    """A torch without the private CPU flash op that _attention_lse calls keeps the dense branch pass."""
    import basal.engine as engine_module

    class NoFlash:
        def __getattr__(self, name):
            raise AttributeError(name)
    monkeypatch.setattr(engine_module.torch.ops, "aten", NoFlash())
    assert not cascade_ok(LlamaConfig())


def test_row_only_readout_equals_full_softmax():
    m = tiny()
    be = Stub(m)
    h = torch.randn(3, 32)
    lids = [[3, 4], [4, 5, 6], [10, 3]]
    got = be._readout(h, lids)
    lp = torch.log_softmax(m.lm_head(h).float(), -1)
    for g, ids, row in zip(got, lids, lp):
        assert len(g) == len(ids) and torch.allclose(torch.tensor(g), torch.softmax(row[ids], -1), atol=1e-6)


def test_default_device_env_override(monkeypatch):
    from basal.engine import default_device
    monkeypatch.setenv("BASAL_DEVICE", "cpu")
    assert default_device() == "cpu"
    from basal.server import parser
    assert parser().parse_args(["--mode", "eager", "--device", "mps"]).device == "mps"


def test_sglang_letter_readout_parsing_and_equivalence():
    """1.5: the SGLang backend reads the option letters' log-probabilities; their softmax equals the letter readout."""
    import torch
    from basal.engine import SGLangBackend
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.0])
    lsm = torch.log_softmax(logits, -1)
    ids = [3, 0, 2]
    meta = {"output_token_ids_logprobs": [[(float(lsm[i]), i, None) for i in (0, 2, 3)]]}  # any order
    got = torch.softmax(torch.tensor(SGLangBackend._letters(meta, ids)), -1)
    ref = torch.softmax(logits[ids], -1)
    assert torch.allclose(got, ref, atol=1e-6)
    meta = {"output_token_ids_logprobs": [[(float(lsm[3]), 3, None)]]}  # a missing letter gets no mass
    assert SGLangBackend._letters(meta, [3, 1])[1] == -1e9


def test_graph_engine_sets_the_small_cublas_workspace_unless_set(monkeypatch, tmp_path):
    """Every model in the graph engine gets 8 MB cuBLAS workspaces per captured graph (CUBLAS_WORKSPACE_CONFIG,
    :4096:2) before any GEMM; a value already set (the user's, or --deterministic's :4096:8) is kept."""
    import os

    from basal.engine import GraphBackend
    for preset, want in ((None, ":4096:2"), (":4096:8", ":4096:8")):
        if preset is None:
            monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        else:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", preset)
        try:
            GraphBackend(str(tmp_path))  # no model there: it stops after the environment is set
        except Exception:
            pass
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == want


def test_graph_engine_keeps_the_compile_cache_across_reboots_unless_set(monkeypatch, tmp_path):
    """A compiling graph engine moves torch.compile's cache to ~/.cache/basal/torchinductor (Inductor's default under
    /tmp is emptied by a reboot) before anything compiles, Inductor's memoised paths with it: also when
    TORCHINDUCTOR_CACHE_DIR holds Inductor's default, which Inductor writes there itself on import (basal-serve's case).
    Kept: a value given before basal was imported (Inductor's default too), another directory set since, everything
    without compile, and the old value when the cache directory is not writable."""
    import os

    from torch._inductor.codecache import CacheBase
    from torch._inductor.runtime.cache_dir_utils import cache_dir, default_cache_dir
    from torch._inductor.utils import clear_caches

    import basal
    from basal.engine import GraphBackend
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    mine, tmp_default = tmp_path / "mine", default_cache_dir()
    default = tmp_path / "home" / ".cache" / "basal" / "torchinductor"
    for given, preset, compile, want in ((False, None, True, default), (False, tmp_default, True, default),
                                         (True, tmp_default, True, tmp_default), (False, mine, True, mine),
                                         (False, tmp_default, False, tmp_default)):
        monkeypatch.setattr(basal, "INDUCTOR_CACHE_DIR_GIVEN", given)
        if preset is None:
            monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        else:
            monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(preset))
        clear_caches()  # as in a new process: Inductor's paths memoised under the value before
        CacheBase.get_local_cache_path()
        try:
            GraphBackend(str(tmp_path), compile=compile)  # no model there: it stops after the environment is set
        except Exception:
            pass
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(want) and cache_dir() == str(want)
        assert CacheBase.get_local_cache_path().is_relative_to(want)
    if os.geteuid() != 0:  # root writes anywhere
        monkeypatch.setenv("HOME", str(tmp_path / "ro"))
        (tmp_path / "ro" / ".cache" / "basal" / "torchinductor").mkdir(parents=True, mode=0o500)
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", tmp_default)
        try:
            GraphBackend(str(tmp_path), compile=True)
        except Exception:
            pass
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == tmp_default
