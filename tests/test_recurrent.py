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


def test_fast_exit_refuses_qwen35_naming_fast(monkeypatch, qwen35):
    no_load(monkeypatch)
    with pytest.raises(ValueError, match="--mode fast") as e:
        ExitGraphBackend(str(qwen35))
    assert "18 linear_attention layers" in str(e.value) and "--mode eager" in str(e.value)
    assert "--mode mps" not in str(e.value)


def test_mps_refuses_qwen35_naming_the_mac_paths(monkeypatch, qwen35):
    """--mode mps of a Qwen3.5 model is refused in 2.5 (seconds per prompt without fla's kernels), before loading; the
    message names the Mac paths: the GGUF file on llama.cpp's Metal backend and MLX."""
    no_load(monkeypatch)
    monkeypatch.delenv("BASAL_EXPERIMENTAL_GDN_MPS", raising=False)
    with pytest.raises(ValueError, match="--mode mps is not served") as e:
        MPSBackend(str(qwen35))
    assert all(w in str(e.value) for w in ("--mode gguf", "Metal", "--mode mlx", "MLX", "--mode eager"))


@pytest.mark.parametrize("kw, words", [({"quant": "fp8"}, "--mode fp8"), ({"quant": "nvfp4"}, "--mode nvfp4")])
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


@pytest.mark.parametrize("args, words", [(["--mode", "fast-exit"], "--mode fast-exit keeps"),
                                         (["--mode", "fp8"], "--mode fp8"), (["--mode", "nvfp4"], "--mode nvfp4"),
                                         (["--mode", "fast", "--long-lane", "on"], "--long-lane")])
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


def test_qwen35_long_lane_is_opt_in(monkeypatch, qwen35):
    """--long-lane on a Qwen3.5 model is refused unless --long-lane-gdn on (off by default) is given, and the refusal
    names that option; with it the server goes on to download the weights. --long-lane-gdn on alone is refused."""
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module, "fla_rule", lambda: object())

    class Downloaded(Exception):
        pass

    def resolve(name, revision=None, metadata_only=False):
        if not metadata_only:
            raise Downloaded
        return str(qwen35)
    monkeypatch.setattr(server_module, "resolve", resolve)
    base = ["--model", "Remek/basal-omni-pico", "--mode", "fast", "--long-lane", "on"]
    assert parser().parse_args(base).long_lane_gdn == "off"
    with pytest.raises(SystemExit, match="--long-lane is not available for Qwen3.5") as e:
        Server(parser().parse_args(base))
    assert "--long-lane-gdn on" in str(e.value)
    with pytest.raises(Downloaded):
        Server(parser().parse_args(base + ["--long-lane-gdn", "on"]))
    with pytest.raises(SystemExit, match="--long-lane-gdn on needs --long-lane on"):
        Server(parser().parse_args(["--model", "Remek/basal-omni-pico", "--mode", "fast", "--long-lane-gdn", "on"]))


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
    """Without --mode: eager for other recurrent models on CUDA and for every one on Apple Silicon (mps refuses them;
    mlx serves Qwen3.5 only when asked); attention-only models keep their defaults."""
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


def tiny_qwen35(d, v_heads=2, layers=("linear_attention", "full_attention", "linear_attention"), head_dim=16):
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=len(layers),
        layer_types=list(layers), num_attention_heads=2,
        num_key_value_heads=1, head_dim=head_dim, linear_num_key_heads=2, linear_num_value_heads=v_heads,
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


@pytest.mark.parametrize("grow", [1, 2, 40])
def test_state_cache_continues_a_growing_state(monkeypatch, tmp_path, grow):
    """A state that begins with a whole cached one (an agent trace that grew by `grow` tokens, fewer than the
    convolution tail included) continues the cached checkpoint and replaces it: the answers equal an uncached engine's,
    and the extended entry is then an exact hit. A state that differs inside the cached tokens runs again."""
    tiny_qwen35(tmp_path / "tiny")
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    be.state_cache.BLOCK = 16  # (256-token blocks in serving; the tiny model's states are short)
    qs = lambda root: ([root + [20 + q, 30, 31] for q in range(3)], [[40, 41, 42]] * 3, len(root))  # noqa: E731
    be.run_shared([qs(ROOT)])
    longer = ROOT + [14, 15, 16, 17, 18] * 8
    longer = ROOT + longer[len(ROOT): len(ROOT) + grow]
    got = be.run_shared([qs(longer)])
    assert max_dp(got, plain.run_shared([qs(longer)[:2]])) < 1e-5
    stats = be.state_cache.stats()
    assert stats["prefix_hits"] == 1 and stats["entries"] == 1  # the extended entry replaced the old one
    assert be.state_cache.get(be.state_cache.key(longer))[0][0].shape[1] == len(longer)
    edited = ROOT[:40] + [19] + ROOT[41:] + [14]
    assert max_dp(be.run_shared([qs(edited)]), plain.run_shared([qs(edited)[:2]])) < 1e-5
    assert be.state_cache.stats()["entries"] == 2


TEMPLATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3]  # 11 tokens: chat template, system prompt and the user header


def test_prefix_cache_continues_the_template_state(monkeypatch, tmp_path):
    """--prefix-cache: requests cut at their template prefix store it once as a full-state checkpoint (the keys and
    values of the attention layers, the recurrent state and convolution tail of each Gated DeltaNet layer), and later
    ones run pass 2 only, every unit over one shared root row; the answers equal an engine without the cache. A
    request with several prompts (one row per leaf below the cut) answers as exactly."""
    tiny_qwen35(tmp_path / "tiny")
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", prefix=True)
    assert be.prefix and be.state_cache is not None and be.state_cache.budget == be.PREFIX_GB * 2 ** 30
    single = [([TEMPLATE + [20 + i, 30 + i, 31, 32] * (1 + i % 3)], [[40, 41, 42]], len(TEMPLATE)) for i in range(6)]
    many = [([TEMPLATE + [21, 22] + [30 + q, 31] for q in range(3)], [[40, 41]] * 3, len(TEMPLATE)),
            ([TEMPLATE + ROOT[:20] + [20, 21], TEMPLATE + ROOT[:20] + [21, 20]], [[40, 41], [41, 40]], len(TEMPLATE))]
    first = be.run_shared(single[:2])
    assert be.state_cache.stats()["entries"] == 1 and be.counts.get("pass1 cached eager") is None
    loads = []
    load = be._load
    be._load = lambda kv, st, hits, P: loads.append(len(hits)) or load(kv, st, hits, P)
    got = be.run_shared(single + many)
    assert loads and all(n == 1 for n in loads)  # the template's checkpoint into one root row per forward
    want = plain.run_shared([g[:2] for g in single + many])
    assert max_dp(got, want) < 1e-5 and max_dp(first, want[:2]) < 1e-5
    assert be.state_cache.stats()["entries"] == 1 and be.counts.get("pass1 cached eager") == 1  # one forward


def test_server_cuts_single_prompts_only_for_a_row_per_leaf_engine():
    """basal-serve cuts a short request at its template prefix; for a Qwen3.5 engine (row_per_leaf) only when the
    request is one prompt (below the cut it would read the state once per prompt, slower than pass 1 + pass 2)."""
    from basal.engine import StateCache
    from basal.server import to_items
    from tests.test_server import FakeServer
    srv = FakeServer([0.5, 0.5])
    srv.backend = type("Be", (), dict(PREFIX_SFX=[10 ** 6], prefix=True, state_cache=StateCache(1e6),
                                       row_per_leaf=True))()
    srv.states_cached = False  # (with --state-cache-gb the state cut is the fallback)
    q = to_items("Customer: the parcel arrived damaged.", {"q": {"type": "noul", "instructions": "Damaged?"}})[0]
    two = [srv.branch(0, 0, q)]  # two option orders
    srv.encode(two)
    assert len(two[0][3]) == 2 and srv.cut_of(q, two) is None
    one = [two[0][:3] + (two[0][3][:1],) + tuple(two[0][4:])]
    t = srv.template_cut(q, one[0][3][0])
    assert t and srv.cut_of(q, one) == t
    srv.backend.row_per_leaf = False
    assert srv.cut_of(q, two) == t


SINGLES = [ROOT[:3], ROOT[:64], ROOT[:65], ROOT[:30] * 4 + [13], [5, 6, 7, 8, 9, 10, 11], ROOT * 2 + [14]]
LAST_ATTENTION = ("linear_attention", "full_attention", "linear_attention", "full_attention")


@pytest.mark.parametrize("v_heads", [2, 4])
def test_compact_rows_equal_padded_rows(monkeypatch, tmp_path, v_heads):
    """--batches compact: single prompts packed into one row (each from a multiple of 64 tokens, padding after its own
    tokens and at the end of the row) answer as padded rows do; branched units still run their two passes."""
    tiny_qwen35(tmp_path / "tiny", v_heads)
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", batches="compact")
    assert be.compact and not be.template and not be.readout_layer
    lids = [[40, 41, 42 + i % 3] for i in range(len(SINGLES))]
    assert max_dp([be.run(SINGLES, lids)], [plain.run(SINGLES, lids)]) < 1e-5
    assert be.counts.get("compact eager") == 1 and "plain eager" not in be.counts
    assert max_dp(be.run_shared(GROUPS), plain.run_shared(GROUPS)) < 1e-5


def test_compact_rows_chunk_layout(monkeypatch, tmp_path):
    """What a compact row gives fla: every prompt (and the padding after the last) starts on a multiple of 64 tokens,
    the row's 64-token chunks are listed once each as (prompt, chunk), positions restart per prompt."""
    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", batches="compact")
    seen = []
    fwd = be.fwd_c
    be.fwd_c = lambda ids, pos, cu, ci, *a, **k: seen.append((ids, pos, cu)) or fwd(ids, pos, cu, ci, *a, **k)
    be.run(SINGLES, [[40, 41]] * len(SINGLES))
    ids, pos, cu = seen[0]
    T = ids.shape[1]
    assert T in be.COMPACT_LENS and T % 64 == 0 and len(cu) == be.COMPACT_SEQS + 2
    starts = cu.tolist()
    assert all(x % 64 == 0 for x in starts) and starts[0] == 0 and starts[-1] == T
    assert all(pos[0, a].item() == 0 for a, b in zip(starts, starts[1:]) if b > a)
    assert sum(b - a for a, b in zip(starts, starts[1:])) == T


def test_compact_rows_continue_a_template_prefix(monkeypatch, tmp_path):
    """--template-prefix on with compact rows: a prompt that starts with a template prefix runs only its own tokens,
    from the prefix's keys / values and its recurrent states and convolution tails (computed once); others from zero;
    the same answers as whole prompts in padded rows. Zeroed template states move the answers."""
    tiny_qwen35(tmp_path / "tiny")
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", batches="compact", template="off")
    be._set_templates([TEMPLATE, TEMPLATE[:5]])
    assert [len(p) for _, p, _ in be.template] == [len(TEMPLATE), 5] and be.n_templates == 2
    prompts = [TEMPLATE + t for t in SINGLES] + [TEMPLATE[:5] + [20, 21, 22]] + SINGLES[:2]
    lids = [[40, 41, 42]] * len(prompts)
    seen = []
    fwd = be.fwd_c
    be.fwd_c = lambda *a, **k: seen.append(a[1:3]) or fwd(*a, **k)
    got = be.run(prompts, lids)
    pos, cu = seen[0]
    starts = sorted(pos[0, a].item() for a, b in zip(cu.tolist(), cu.tolist()[1:]) if b > a)
    assert [x for x in starts if x] == [5] + [len(TEMPLATE)] * len(SINGLES) and starts.count(0) in (2, 3)
    assert max_dp([got], [plain.run(prompts, lids)]) < 1e-5
    for s, c in be.net.template_st:
        s.zero_()
        c.zero_()
    assert max_dp([be.run(prompts, lids)], [got]) > 1e-4


def test_compact_rows_readout_layer(monkeypatch, tmp_path):
    """--readout-layer on (compact rows, a model whose last layer is attention): the last layer's projections after
    its attention and MLP at the readout positions only; the same answers. A model ending in a Gated DeltaNet layer
    keeps the whole last layer."""
    tiny_qwen35(tmp_path / "tiny", layers=LAST_ATTENTION)
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", batches="compact", readout_layer="on")
    assert be.readout_layer
    lids = [[40, 41, 42]] * len(SINGLES)
    assert max_dp([be.run(SINGLES, lids)], [plain.run(SINGLES, lids)]) < 1e-5
    tiny_qwen35(tmp_path / "gdn_last")
    assert not gdn_backend(monkeypatch, tmp_path / "gdn_last", batches="compact", readout_layer="on").readout_layer


@pytest.mark.parametrize("device, tol", [("cpu", 1e-5), ("mps", 1e-4)])
def test_mps_mode_carries_the_state(monkeypatch, tmp_path, device, tol):
    """--mode mps with BASAL_EXPERIMENTAL_GDN_MPS=1: MPSBackend of a Qwen3.5 model builds GDNMPSBackend (the two passes,
    no graphs, uncompiled) on the Apple GPU or the CPU; the answers equal eager's. Without prefix sharing it stays
    MPSBackend (eager forwards)."""
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        pytest.skip("no Apple GPU")
    monkeypatch.setenv("BASAL_EXPERIMENTAL_GDN_MPS", "1")
    tiny_qwen35(tmp_path / "tiny")
    for mod in (engine_module, gdn_module):
        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(q35, name, inspect.unwrap(getattr(q35, name)))
    be = MPSBackend(str(tmp_path / "tiny"), "float32", device=device)
    assert isinstance(be, gdn_module.GDNMPSBackend) and be.dev.type == device and be.pool is None
    ref = EagerBackend(str(tmp_path / "tiny"), "float32", device="cpu")
    assert max_dp(be.run_shared(GROUPS), ref.run_shared(GROUPS)) < tol
    assert type(MPSBackend(str(tmp_path / "tiny"), "float32", shared=False, device="cpu")) is MPSBackend


def test_lean_and_flex_change_nothing(monkeypatch, tmp_path):
    """--lean / --tree flex are taken and change nothing (as for basal-omni): the Qwen3.5 forward is lean already and
    its pass 2 has no trie mask."""
    tiny_qwen35(tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", lean="pass2", tree="flex")
    assert be.lean is None and not be.flex
    assert max_dp(be.run_shared(GROUPS), gdn_backend(monkeypatch, tmp_path / "tiny").run_shared(GROUPS)) == 0


def test_long_lane_answers_as_the_main_lane(monkeypatch, tmp_path):
    """--long-lane: run_long computes every unit on its own, uncompiled, one layer at a time (between(i) before each
    layer) over fresh buffers: the main lane's answers; with the state cache its roots are stored and then read."""
    tiny_qwen35(tmp_path / "tiny")
    plain = gdn_backend(monkeypatch, tmp_path / "tiny")
    be = gdn_backend(monkeypatch, tmp_path / "tiny", state_cache_gb=0.01)
    calls = []
    groups = [g + (len(ROOT),) if len(g[0][0]) > len(ROOT) else g for g in GROUPS]
    got = be.run_long(groups, None, between=calls.append)
    assert max_dp(got, plain.run_shared(GROUPS)) < 1e-5
    assert calls[:3] == [0, 1, 2] and len(calls) % 3 == 0 and be.net.between is None
    assert be.state_cache.stats()["entries"] == 1
    assert max_dp(be.run_long(groups[:2], None, between=calls.append), got[:2]) < 1e-6
    assert be.state_cache.stats()["hits"] >= 2


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


def test_conv_fallback_is_said_and_reported(monkeypatch, tmp_path, capsys):
    """Without fla's causal_conv1d the Gated DeltaNet convolution runs on torch's conv1d, which rounds differently
    from the kernel --mode fast was checked with: on CUDA that is said at start-up, and the backend names its kernel
    (the start-up line and /ui show it). The CPU runs torch's, the eager reference's, without a warning."""
    sentinel = object()
    monkeypatch.setattr(gdn_module, "fla_conv1d", lambda: None)
    assert gdn_module.conv_kernel(True) == (None, "torch conv1d")
    assert "torch's conv1d" in capsys.readouterr().err
    assert gdn_module.conv_kernel(False) == (None, "torch conv1d") and capsys.readouterr().err == ""
    monkeypatch.setattr(gdn_module, "fla_conv1d", lambda: sentinel)
    assert gdn_module.conv_kernel(True) == (sentinel, "fla causal_conv1d") and capsys.readouterr().err == ""
    tiny_qwen35(tmp_path / "tiny")
    assert gdn_backend(monkeypatch, tmp_path / "tiny").conv_kernel == "torch conv1d"


def test_compact_rows_need_fla_conv(monkeypatch, qwen35):
    """Compact rows call fla's causal_conv1d with the chunk indices (there is no torch fallback for them): without it
    they are refused before loading, not failed at the first request."""
    no_load(monkeypatch)
    monkeypatch.setattr(gdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gdn_module, "fla_varlen", lambda: (None, None))
    monkeypatch.setattr(gdn_module, "fla_conv1d", lambda: None)
    monkeypatch.setattr(gdn_module, "set_attention", lambda *a, **k: "sdpa")
    monkeypatch.setattr(gdn_module, "torch_varlen_available", lambda *a: None)
    with pytest.raises(ValueError, match="causal_conv1d"):
        GraphBackend(str(qwen35), "float32", batches="compact", device="cuda")


def test_pinned_fla_cache_never_times():
    """A pinned autotuner cache has every key (so Triton never times candidates): the recorded config for a recorded
    key, the kernel's default for any other; without CUDA nothing is pinned."""
    c = gdn_module._PinnedCache({(64, "torch.bfloat16"): "recorded"}, "default")
    assert (64, "torch.bfloat16") in c and (128, "torch.float32") in c
    assert c[(64, "torch.bfloat16")] == "recorded" and c[(128, "torch.float32")] == "default"
    if not torch.cuda.is_available():
        assert gdn_module.fla_pin() == 0


def test_fla_pin_pins_only_the_recorded_kernels(monkeypatch, tmp_path):
    """The table's kernels get their recorded configs (untimed); a kernel the recording never launched keeps its own
    autotuner (it was pinned to its first candidate: a config never measured, which may not fit the GPU)."""
    import sys
    import types

    class Config:
        def __init__(self, kwargs, **launch):
            self.kwargs, self.launch = kwargs, launch

    tuner = types.SimpleNamespace
    recorded, other = tuner(cache={}, configs=["first"]), tuner(cache={}, configs=["first"])
    triton = types.ModuleType("triton")
    triton.runtime = types.ModuleType("triton.runtime")
    triton.runtime.autotuner = types.ModuleType("triton.runtime.autotuner")
    triton.runtime.autotuner.Config = Config
    for name, mod in (("triton", triton), ("triton.runtime", triton.runtime),
                      ("triton.runtime.autotuner", triton.runtime.autotuner)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "fla", types.SimpleNamespace(__version__="0.5.2"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "GPU X")
    monkeypatch.setattr(gdn_module, "_fla_autotuners", lambda: {"fla.ops.k": recorded, "fla.ops.backward_k": other})
    launch = dict(num_warps=4, num_stages=2, num_ctas=1, maxnreg=None)
    table = dict(gpu="GPU X", fla="0.5.2", kernels={"fla.ops.k": dict(
        default=dict(kwargs={"BT": 64}, **launch), keys=[[[64, "torch.bfloat16"], dict(kwargs={"BT": 32}, **launch)]])})
    (tmp_path / "fla.json").write_text(json.dumps(table), encoding="utf-8")
    assert gdn_module.fla_pin(tmp_path / "fla.json") == 1
    assert recorded.cache[(64, "torch.bfloat16")].kwargs == {"BT": 32}
    assert recorded.cache[(128, "torch.float32")].kwargs == {"BT": 64}  # (any other key: the kernel's default)
    assert other.cache == {} and (64, "torch.bfloat16") not in other.cache  # timed at its first launch, as fla does


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


@pytest.mark.skipif(not torch.cuda.is_available() or gdn_module.fla_varlen() is None, reason="needs CUDA and fla")
@pytest.mark.parametrize("dtype, attn, tol", [("float32", "auto", 1e-4), ("bfloat16", "auto", 0.03),
                                              ("bfloat16", "fa2", 0.03)])
def test_compact_graphs_equal_eager_on_cuda(monkeypatch, tmp_path, dtype, attn, tol):
    """Compact rows on CUDA: fla's kernels over the packed prompts with the chunk indices given (bf16: in CUDA graphs,
    with FlashAttention 3's varlen kernel or torch's; fp32: SDPA per prompt, no graph), with and without template
    prefixes, against eager; a replayed graph with other prompts answers as a fresh capture."""
    if attn == "auto" and dtype == "bfloat16" and engine_module.attention_available("fa3"):
        pytest.skip("FlashAttention 3 is not here")
    if attn == "fa2" and engine_module.torch_varlen_available(torch.bfloat16):  # the engine refuses compact rows then
        pytest.skip(f"torch's varlen attention: {engine_module.torch_varlen_available(torch.bfloat16)}")
    tiny_qwen35(tmp_path / "tiny", layers=LAST_ATTENTION, head_dim=64)
    ref = EagerBackend(str(tmp_path / "tiny"), dtype, device="cuda")
    for kw in ({}, {"readout_layer": "on"}):
        be = gdn_backend(monkeypatch, tmp_path / "tiny", dtype, cpu=False, warm=False, batches="compact", attn=attn,
                         template="off", **kw)
        be._set_templates([TEMPLATE])
        prompts = [TEMPLATE + t for t in SINGLES] + SINGLES
        lids = [[40, 41, 42]] * len(prompts)
        got = be.run(prompts, lids)
        assert (be.counts.get("compact graph") == 1) == (dtype == "bfloat16")
        assert max_dp([got], [ref.run(prompts, lids)]) < tol
        assert max_dp([be.run(prompts[3:], lids[3:])], [got[3:]]) < tol


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


def test_gdn_engine_takes_every_graph_engine_option():
    """GraphBackend(...) of a Qwen3.5 model builds GDNGraphBackend with the same arguments (basal-serve passes the
    quantisation options to every graph engine), so its __init__ takes every parameter GraphBackend's takes."""
    graph = set(inspect.signature(GraphBackend.__init__).parameters)
    assert graph <= set(inspect.signature(GDNGraphBackend.__init__).parameters)


def test_fla_autotuners_skip_dead_weak_proxies():
    """gc.get_objects() can hold a weakref.proxy whose object is gone; isinstance() on it raises ReferenceError, which
    stopped basal-prepare's fla table on an H100 (26 Oct). The scan skips it."""
    pytest.importorskip("triton")
    import weakref

    class Gone:
        pass

    o = Gone()
    dead = weakref.proxy(o)
    del o
    holder = [dead]  # keeps the dead proxy in gc's lists
    assert isinstance(gdn_module._fla_autotuners(), dict) and holder


def test_vllm_deterministic_preset_refused_for_qwen35_before_download(monkeypatch, qwen35):
    """vLLM's batch-invariant mode has no Gated DeltaNet kernel (vLLM 0.31 raises in its engine core): basal-serve refuses
    --vllm-preset deterministic for a recurrent model before the weights load, and keeps the default preset."""
    def resolve(name, revision=None, metadata_only=False):
        if not metadata_only:
            raise RuntimeError("loading the weights")
        return str(qwen35)
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="batch-invariant kernels do not cover 18 linear_attention layers"):
        Server(parser().parse_args(["--model", "Remek/basal-omni-pico", "--mode", "vllm", "--vllm-preset",
                                    "deterministic"]))
    with pytest.raises(RuntimeError, match="loading the weights"):  # the default preset goes on to load
        Server(parser().parse_args(["--model", "Remek/basal-omni-pico", "--mode", "vllm"]))
