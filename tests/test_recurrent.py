"""Models with recurrent-state layers (Qwen3.5: Gated DeltaNet / linear attention beside full attention). Since 2.4
the CUDA graph engine (--mode fast, fast-nocompile) serves Qwen3.5 decoders through basal/gdn_graph.py, which carries
the recurrent state from the root into every branch; fast-exit, fp8, nvfp4, mps and every other recurrent model stay
refused from config.json, before any weights are read. --mode fp8 stops before loading on GPUs below compute capability
8.9. No GPU needed (the graph test at the end runs where CUDA and fla are installed)."""
import inspect
import json

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

import basal.engine as engine_module
import basal.gdn_graph as gdn_module
import basal.server as server_module
from basal.engine import (EagerBackend, ExitGraphBackend, GraphBackend, MPSBackend, fp8_refused, recurrent_layers)
from basal.gdn_graph import GDNGraphBackend, is_gdn
from basal.server import Server, default_mode, parser

PICO_LAYERS = (["linear_attention"] * 3 + ["full_attention"]) * 6  # Qwen3.5-0.8B: 18 Gated DeltaNet, 6 attention


def write_config(d, cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


@pytest.fixture
def qwen35(tmp_path):
    return write_config(tmp_path / "pico", {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
                                              "layer_types": PICO_LAYERS, "num_hidden_layers": 24})


@pytest.fixture
def llama(tmp_path):
    return write_config(tmp_path / "llama", {"model_type": "llama", "num_hidden_layers": 2})


@pytest.fixture
def qwen3_next(tmp_path):
    """Gated DeltaNet too, but another layer code (merged projections, MoE): not ported, stays refused."""
    return write_config(tmp_path / "next", {"model_type": "qwen3_next", "layer_types": PICO_LAYERS})


def test_recurrent_layers_from_config(tmp_path, qwen35, llama):
    assert recurrent_layers(qwen35) == "18 linear_attention layers"
    assert recurrent_layers(llama) is None
    # Gemma 4 (basal-omni): sliding and full attention in text_config, no recurrent state
    omni = write_config(tmp_path / "omni", {"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}})
    assert recurrent_layers(omni) is None
    # the multimodal Qwen3.5 wrapper: its decoder in text_config; a config without layer_types: by model_type
    vl = write_config(tmp_path / "vl", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text",
                                                                                 "layer_types": PICO_LAYERS}})
    assert recurrent_layers(vl) == "18 linear_attention layers"
    bare = write_config(tmp_path / "bare", {"model_type": "qwen3_5_text"})
    assert recurrent_layers(bare) == "model_type qwen3_5_text"
    assert recurrent_layers(tmp_path / "missing") is None


def test_is_gdn(tmp_path, qwen35, llama, qwen3_next):
    assert is_gdn(qwen35)
    assert not is_gdn(llama) and not is_gdn(qwen3_next) and not is_gdn(tmp_path / "missing")
    vl = write_config(tmp_path / "vl", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text",
                                                                                 "layer_types": PICO_LAYERS}})
    assert is_gdn(vl)
    moe = write_config(tmp_path / "moe", {"model_type": "qwen3_5_moe_text", "layer_types": PICO_LAYERS})
    assert not is_gdn(moe)


def no_load(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the refusal must come before the tokenizer and the weights are loaded")
    for mod in (engine_module, gdn_module):
        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", fail)
        monkeypatch.setattr(mod.AutoModelForCausalLM, "from_pretrained", fail)


@pytest.mark.parametrize("cls, kw", [(GraphBackend, {}), (GraphBackend, {"quant": "fp8"}), (ExitGraphBackend, {}),
                                     (MPSBackend, {})])
def test_other_recurrent_models_refused_before_loading(monkeypatch, qwen3_next, cls, kw):
    no_load(monkeypatch)
    with pytest.raises(ValueError, match="--mode eager") as e:
        cls(str(qwen3_next), **kw)
    assert "18 linear_attention layers" in str(e.value) and "--mode vllm" in str(e.value)


@pytest.mark.parametrize("cls", [ExitGraphBackend, MPSBackend])
def test_fast_exit_and_mps_refuse_qwen35_naming_fast(monkeypatch, qwen35, cls):
    no_load(monkeypatch)
    with pytest.raises(ValueError, match="--mode fast") as e:
        cls(str(qwen35))
    assert "18 linear_attention layers" in str(e.value) and "--mode eager" in str(e.value)


@pytest.mark.parametrize("kw, words", [({"quant": "fp8"}, "--mode fp8"), ({"quant": "nvfp4"}, "--mode nvfp4"),
                                       ({"batches": "compact"}, "--batches compact"),
                                       ({"prefix": True, "state_cache_gb": 1}, "--prefix-cache on"),
                                       ({"lean": "pass2", "tree": "flex"}, "--lean, --tree flex"),
                                       ({"readout_layer": "on", "template": "on"},
                                        "--readout-layer on, --template-prefix on")])
def test_gdn_engine_refuses_settings_before_loading(monkeypatch, qwen35, kw, words):
    """GraphBackend(qwen35) builds the port, which refuses what it has not been gated with before loading."""
    no_load(monkeypatch)
    with pytest.raises(ValueError, match="Qwen3.5") as e:
        GraphBackend(str(qwen35), **kw)
    assert words in str(e.value)


def test_gdn_engine_needs_cuda_and_fla(monkeypatch, qwen35):
    no_load(monkeypatch)
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="needs a CUDA GPU"):
        GraphBackend(str(qwen35))
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module, "fla_rule", lambda: None)
    with pytest.raises(ValueError, match="flash-linear-attention"):
        GraphBackend(str(qwen35))


@pytest.mark.parametrize("args, words", [(["--mode", "fast-exit"], "--mode fast-exit and --mode mps"),
                                         (["--mode", "fp8"], "--mode fp8"), (["--mode", "nvfp4"], "--mode nvfp4"),
                                         (["--mode", "fast", "--prefix-cache", "on"], "--prefix-cache on"),
                                         (["--mode", "fast", "--long-lane", "on"], "--long-lane"),
                                         (["--mode", "fast-nocompile", "--batches", "compact"], "--batches compact")])
def test_server_refuses_qwen35_settings_before_download(monkeypatch, qwen35, args, words):
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module, "fla_rule", lambda: object())

    def resolve(name, revision=None, metadata_only=False):
        assert metadata_only, "the weights must not be downloaded for a refused model"
        return str(qwen35)
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="Qwen3.5|--mode fast") as e:
        Server(parser().parse_args(["--model", "Remek/basal-omni-pico"] + args))
    assert words in str(e.value)


@pytest.mark.parametrize("mode", ["fast", "fast-nocompile"])
def test_server_refuses_fast_qwen35_without_cuda_before_download(monkeypatch, qwen35, mode):
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: False)

    def resolve(name, revision=None, metadata_only=False):
        assert metadata_only, "the weights must not be downloaded for a refused model"
        return str(qwen35)
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="--mode eager"):
        Server(parser().parse_args(["--model", "Remek/basal-omni-pico", "--mode", mode]))


@pytest.mark.parametrize("mode", ["fast", "fast-nocompile", "fast-exit", "fp8", "nvfp4"])
def test_server_refuses_other_recurrent_models_before_download(monkeypatch, qwen3_next, mode):
    def resolve(name, revision=None, metadata_only=False):
        assert metadata_only, "the weights must not be downloaded for a refused model"
        return str(qwen3_next)
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="--mode eager"):
        Server(parser().parse_args(["--model", "Qwen/Qwen3-Next", "--mode", mode]))


@pytest.mark.parametrize("cc, fla, mode", [((9, 0), True, "fast"), ((9, 0), False, "eager"), ((8, 0), True, "eager"),
                                           ((12, 0), True, "eager")])
def test_default_mode_of_qwen35_on_cuda(monkeypatch, qwen35, cc, fla, mode):
    """Without --mode: fast where the port passed its gate (Hopper, with fla), eager on other GPU classes."""
    monkeypatch.setattr(server_module, "resolve", lambda name, revision=None, metadata_only=False: name)
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module.torch.cuda, "get_device_capability", lambda *a: cc)
    monkeypatch.setattr(gdn_module, "fla_rule", lambda: object() if fla else None)
    assert default_mode(str(qwen35)) == mode


def test_default_mode_of_other_recurrent_models(monkeypatch, qwen35, qwen3_next, llama):
    """Without --mode: eager for other recurrent models on CUDA and for every one on Apple Silicon (mps refuses them,
    mlx is not checked on them); attention-only models keep their defaults."""
    monkeypatch.setattr(server_module, "resolve", lambda name, revision=None, metadata_only=False: name)
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: "fa3")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module.torch.cuda, "get_device_capability", lambda *a: (9, 0))
    monkeypatch.setattr(gdn_module, "fla_rule", lambda: object())
    assert default_mode(str(qwen3_next)) == "eager"
    assert default_mode(str(llama)) == default_mode() == "fast"
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    assert default_mode(str(qwen35)) == "eager"
    assert default_mode(str(llama)) in ("mlx", "mps")


def test_default_mode_keeps_fast_when_the_config_cannot_be_read(monkeypatch):
    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(server_module, "resolve", offline)
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: "fa3")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    assert default_mode("Remek/basal-omni-pico") == "fast"  # the model load then reports it


def tiny_qwen35(d, v_heads=2):
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=v_heads,
        linear_key_head_dim=8, linear_value_head_dim=8, max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})).eval()
    m.save_pretrained(d)
    return m


class Tok:
    pad_token = eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, add_special_tokens=False, **kw):
        return type("E", (), {"input_ids": texts})()


def gdn_backend(monkeypatch, d, dtype="float32", cpu=True, **kw):
    """GraphBackend of a Qwen3.5 checkpoint; on the CPU (also where CUDA is there) the port's two passes run without
    CUDA graphs, with transformers' torch kernel for the Gated DeltaNet layers (in the eager reference too: transformers
    routes it to fla whenever fla is installed, and fla needs CUDA tensors)."""
    for mod in (engine_module, gdn_module):
        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    if cpu:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            monkeypatch.setattr(q35, name, inspect.unwrap(getattr(q35, name)))
        monkeypatch.setattr(gdn_module, "refused", lambda *a, **k: None)
        with monkeypatch.context() as m:
            m.setattr(gdn_module.torch.cuda, "is_available", lambda: False)
            return GraphBackend(str(d), dtype, **kw)
    return GraphBackend(str(d), dtype, **kw)


ROOT = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 7 + [13]  # 71 tokens: a state of more than one 64-token chunk
GROUPS = [  # (prompts, code ids)
    # five questions over one state
    ([ROOT + [20 + q, 30, 31, 32 + q] for q in range(5)], [[40, 41, 42]] * 5),
    # two option orders under one question text (a block shared by two leaves), and a prompt inside a leaf's path
    ([ROOT + [20, 21, 22, 23, 24], ROOT + [20, 21, 22, 25, 26], ROOT + [20, 21, 22]], [[40, 41], [42, 43], [44, 45]]),
    ([[5, 6, 7]], [[40, 41]]),  # a single prompt
    # a shorter root, padded beside the others in pass 1, and a prompt that is the shared prefix itself
    ([ROOT[:30] + [20], ROOT[:30] + [21, 22], ROOT[:30]], [[40, 41], [42, 43], [44, 45]]),
]


def max_dp(a, b):
    return max(abs(x - y) for ga, gb in zip(a, b) for pa, pb in zip(ga, gb) for x, y in zip(pa, pb))


@pytest.mark.parametrize("v_heads", [2, 4])
def test_fast_equals_eager_on_a_tiny_qwen35(monkeypatch, tmp_path, v_heads):
    """The port's two passes (roots of different lengths padded together; the state carried into one row per leaf:
    five questions, two option orders under one question text, a prompt inside a path, a prompt that is the shared
    prefix itself) answer as eager does, which answers as each prompt's own fp32 forward."""
    m = tiny_qwen35(tmp_path / "tiny", v_heads)
    be = gdn_backend(monkeypatch, tmp_path / "tiny")
    assert isinstance(be, GDNGraphBackend)
    ref = EagerBackend(str(tmp_path / "tiny"), "float32", device="cpu")
    got = be.run_shared(GROUPS)
    assert max_dp(got, ref.run_shared(GROUPS)) < 1e-5
    with torch.no_grad():
        for g, (prompts, lids) in zip(got, GROUPS):
            for p, t, ids in zip(g, prompts, lids):
                want = torch.softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float()[ids], -1)
                assert torch.allclose(torch.tensor(p), want, atol=1e-5)
    alone = be.run([p for g in GROUPS for p in g[0]], [x for g in GROUPS for x in g[1]])  # single-prompt units
    assert max_dp([[p] for p in alone], [[p] for g in got for p in g]) < 1e-5


def test_state_cache_keeps_full_state_checkpoints(monkeypatch, tmp_path):
    """--state-cache-gb: a request cut at its state stores its root as a full-state checkpoint (attention keys / values,
    then per Gated DeltaNet layer the recurrent state and convolution tail); the same state again skips pass 1 and
    answers as before and as without the cache. A prompt that is the cut root itself still runs pass 1."""
    tiny_qwen35(tmp_path / "tiny")
    ref = gdn_backend(monkeypatch, tmp_path / "tiny").run_shared(GROUPS)
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    cut = [len(ROOT), len(ROOT), None, 30]
    groups = [(p, ids, c) if c else (p, ids) for (p, ids), c in zip(GROUPS, cut)]
    first = be.run_shared(groups)
    assert max_dp(first, ref) < 1e-5
    st = be.state_cache.stats()
    assert st["entries"] == 2 and be.counts.get("pass1 cached eager") is None
    n_att, n_gdn = len(be.kv_shapes), be.net.n_gdn
    e = be.state_cache.get(be.state_cache.key(ROOT))
    assert len(e) == n_att + n_gdn and e[0][0].shape[1] == len(ROOT)  # keys / values of the 71 state tokens
    assert e[n_att][0].shape == be.net.state_shape and e[n_att][1].shape == be.net.tail_shape
    again = be.run_shared(groups[:2])
    assert be.counts.get("pass1 cached eager") == 1  # both units of the call from the cache, no pass 1
    assert max_dp(again, first[:2]) < 1e-6
    be.run_shared(groups[3:])  # its root is a prompt (read out in pass 1): computed again
    assert be.counts.get("pass1 cached eager") == 1


def test_state_cache_hit_needs_the_cached_state(monkeypatch, tmp_path):
    """A cached checkpoint whose recurrent states are zeroed changes the answers: pass 2 reads the cached states."""
    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    groups = [(p, ids, len(ROOT)) for p, ids in GROUPS[:2]]
    good = be.run_shared(groups)
    for s, c in be.state_cache.get(be.state_cache.key(ROOT))[len(be.kv_shapes):]:
        s.zero_()
        c.zero_()
    assert max_dp(good, be.run_shared(groups)) > 1e-4  # (untouched: < 1e-6, test above)


def test_branches_continue_the_root_state(monkeypatch, tmp_path):
    """Pass 2 from zero state (2.3.1's error) moves the answers; the port carries the root's state and tail."""
    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny")
    good = be.run_shared(GROUPS[:2])
    sfx = be.fwd_sfx
    be.fwd_sfx = lambda ids, pos, unit, plen, kv, st: sfx(ids, pos, unit, plen, kv,
                                                          [(torch.zeros_like(s), torch.zeros_like(c)) for s, c in st])
    assert max_dp(good, be.run_shared(GROUPS[:2])) > 1e-3


def test_leaves_of_a_trie():
    """One pass-2 row per leaf: the path of blocks below the root; each readout in the first row through its block."""
    pack = GraphBackend._pack([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 7], [1, 2, 3]])
    rows, reads = GDNGraphBackend._leaves(pack, pack[2].index(1))
    assert [[pack[0][i] for i in r] for r in rows] == [[4, 5], [4, 6], [7]]
    assert reads == [(0, 1), (1, 1), (2, 0), (None, 2)]


@pytest.mark.parametrize("L2", [512, 8192])
def test_pass2_parts_keep_the_budgets(L2):
    """Leaf rows are run shortest first, at most LEAF_ROWS[-1] (and PASS2_KEYS // root bucket) per forward, rows x
    length within TOKEN_BUDGET unless a row is alone; every row once."""
    be = GDNGraphBackend.__new__(GDNGraphBackend)
    lens = [30] * 70 + [300] * 20 + [700] * 13 + [2000]
    parts = be._parts(lens, L2)
    assert sorted(q for p in parts for q in p) == list(range(len(lens)))
    for p in parts:
        n = max(lens[q] for q in p)
        S = next((x for x in be.LEAF_LENS if n <= x), -(-n // be.SUFFIX_EAGER) * be.SUFFIX_EAGER)
        assert len(p) <= min(be.LEAF_ROWS[-1], be.PASS2_KEYS // L2)
        assert len(p) == 1 or be._bucket(len(p), be.LEAF_ROWS) * S <= be.TOKEN_BUDGET
    assert [lens[q] for p in parts for q in p] == sorted(lens)


@pytest.mark.skipif(not torch.cuda.is_available() or gdn_module.fla_rule() is None, reason="needs CUDA and fla")
@pytest.mark.parametrize("dtype, tol", [("float32", 1e-4), ("bfloat16", 0.03)])
def test_graphs_equal_eager_on_cuda(monkeypatch, tmp_path, dtype, tol):
    """The CUDA graphs of the port (captured as they are first used) against eager with fla, and a group batched with
    others against the same group alone."""
    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", dtype, cpu=False, warm=False)
    ref = EagerBackend(str(tmp_path / "tiny"), dtype, device="cuda")
    got = be.run_shared(GROUPS)
    assert be.graphs and any(k[0] == "x" for k in be.graphs)
    assert max_dp(got, ref.run_shared(GROUPS)) < tol
    assert max_dp(got[:1], be.run_shared(GROUPS[:1])) < tol


@pytest.mark.parametrize("cc, refused", [((8, 6), True), ((8, 0), True), ((8, 9), False), ((9, 0), False),
                                         ((12, 0), False)])
def test_fp8_needs_compute_capability_8_9(monkeypatch, cc, refused):
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: cc)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    why = fp8_refused()
    assert bool(why) == refused
    if refused:
        assert f"sm_{cc[0]}{cc[1]}" in why and "8.9" in why and "--mode fast" in why


@pytest.mark.parametrize("args", [["--mode", "fp8"], ["--mode", "fast", "--quant", "fp8"],
                                  ["--mode", "fast-exit", "--quant", "fp8"]])
def test_server_refuses_fp8_on_ampere_before_loading(monkeypatch, args):
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(server_module, "_recurrent", lambda *a: None)

    def resolve(name, revision=None, metadata_only=False):
        raise AssertionError("refused before the model is resolved")
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="compute capability 8.9"):
        Server(parser().parse_args(["--model", "Remek/basal-2.0"] + args))


def test_fp8_refused_in_the_engine_too(monkeypatch, llama):
    """basal-bench and basal-calibrate build the engine without the server's checks."""
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(engine_module.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("refused before loading"))
    with pytest.raises(ValueError, match="sm_86"):
        GraphBackend(str(llama), quant="fp8")
