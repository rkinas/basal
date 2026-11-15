"""ModelOpt checkpoints (the -FP8 / -NVFP4 repositories) in the engines that cannot read them: transformers has no
ModelOpt loader, so eager (and mps, mlx, strix, tpu) loaded the FP8 weights without their scales and answered wrongly
(basal-omni-1.0-mini-FP8 in --mode eager). They are refused from config.json before any weights load, and without
--mode such an engine is never picked for them."""
import json
from pathlib import Path

import pytest
import torch

import basal.server as server_module
from basal import engine, gdn_graph
from basal.engine import EagerBackend, MLXBackend, MPSBackend
from basal.server import Server, default_mode, parser


def modelopt_algo(model_dir):  # (looked up at the call: on a tree without the check, these tests fail, not collection)
    return engine.modelopt_algo(model_dir)


def modelopt_refused(*a, **k):
    return engine.modelopt_refused(*a, **k)

CONFIGS = {  # config.json of a metadata-only model folder (the refusals read nothing else)
    "gemma4": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
               "text_config": {"model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2},
               "vision_config": {"model_type": "gemma4_vision"}},
    "qwen3": {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 4096},
    "qwen35": {"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"],
               "layer_types": ["linear_attention", "full_attention"]}}


def folder(path, family="gemma4", quant=None):
    path.mkdir(parents=True)
    cfg = dict(CONFIGS[family])
    if quant:
        cfg["quantization_config"] = quant
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return path

MINI_FP8_QUANT = {  # basal-omni-1.0-mini-FP8's config.json "quantization_config" (abridged)
    "config_groups": {"group_0": {"input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                                  "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                                  "targets": ["Linear"]}},
    "ignore": ["lm_head", "model.vision_tower*"], "quant_algo": "FP8", "quant_method": "modelopt",
    "producer": {"name": "modelopt", "version": "0.47.0"}}


def quantised(path, family="gemma4", algo="FP8"):
    """A metadata-only model folder whose config.json says ModelOpt `algo`."""
    return folder(path, family, {**MINI_FP8_QUANT, "quant_algo": algo})


@pytest.fixture
def fp8(tmp_path):
    return quantised(tmp_path / "mini-FP8")


def no_weights(monkeypatch):
    """Any weight load fails the test: the refusals come first."""
    def load(*a, **k):
        pytest.fail("the weights were loaded")
    for cls in (engine.AutoModelForCausalLM, gdn_graph.AutoModelForCausalLM):
        monkeypatch.setattr(cls, "from_pretrained", load)


def test_modelopt_algo_reads_the_quantization_config(tmp_path, fp8):
    assert modelopt_algo(fp8) == "FP8"
    assert modelopt_algo(quantised(tmp_path / "nvfp4", algo="NVFP4")) == "NVFP4"
    plain = folder(tmp_path / "plain")
    assert modelopt_algo(plain) is None
    (tmp_path / "torchao").mkdir()
    (tmp_path / "torchao" / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "torchao"}}))
    assert modelopt_algo(tmp_path / "torchao") is None
    assert modelopt_algo(tmp_path / "missing") is None  # (the model load reports it)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{")
    assert modelopt_algo(tmp_path / "broken") is None
    assert engine.modelopt_fp8(fp8)  # the CUDA graph engine's own test (load_modelopt_fp8) agrees


@pytest.mark.parametrize("kind", ["eager", "mps", "mlx", "strix", "tpu"])
def test_engines_without_a_modelopt_loader_refuse_it(tmp_path, fp8, kind):
    why = modelopt_refused(fp8, kind)
    assert why == (f"--mode {kind} cannot read a ModelOpt FP8 checkpoint (config.json \"quantization_config\"): it "
                   "would load the FP8 weights without their scales, and every answer would be wrong. Serve it with "
                   "--mode fast (the CUDA graph engine reads its FP8 layers) or --mode vllm on a CUDA GPU, or serve "
                   "its bf16 model with this mode")
    assert "--mode vllm on a CUDA GPU" in modelopt_refused(quantised(tmp_path / "nvfp4", algo="NVFP4"), kind)
    assert modelopt_refused(folder(tmp_path / "plain"), kind) is None


def test_engines_that_read_it_serve_it(tmp_path, fp8):
    """The CUDA graph engine reads the FP8 layers (load_modelopt_fp8), vLLM and SGLang every ModelOpt algorithm; the
    GGUF engines read only the folder's tokenizer and calibration. The graph engine refuses another algorithm, and
    Qwen3.5 (its engine loads with transformers)."""
    for kind in ("graph", "exit", "vllm", "sglang", "gguf", "ollama", "llamacpp"):
        assert modelopt_refused(fp8, kind, "fast") is None
    nvfp4 = quantised(tmp_path / "nvfp4", algo="NVFP4")
    assert modelopt_refused(nvfp4, "vllm") is None
    assert modelopt_refused(nvfp4, "graph", "fast").startswith("--mode fast cannot read a ModelOpt NVFP4 checkpoint")
    assert "Serve it with --mode vllm on a CUDA GPU" in modelopt_refused(nvfp4, "graph", "fast")
    qwen = quantised(tmp_path / "pico-FP8", "qwen35")
    why = modelopt_refused(qwen, "graph", "fast", gdn=True)
    assert why.startswith("--mode fast cannot read a ModelOpt FP8 checkpoint (config.json \"quantization_config\", "
                          "a Qwen3.5 model)") and "Serve it with --mode vllm on a CUDA GPU" in why


def test_eager_and_mps_backends_refuse_before_loading(monkeypatch, fp8):
    no_weights(monkeypatch)
    with pytest.raises(ValueError, match="--mode eager cannot read a ModelOpt FP8 checkpoint"):
        EagerBackend(str(fp8), device="cpu")
    with pytest.raises(ValueError, match="--mode eager cannot read a ModelOpt FP8 checkpoint"):
        EagerBackend(str(fp8), device="cpu", media=True)
    with pytest.raises(ValueError, match="--mode mps cannot read a ModelOpt FP8 checkpoint"):
        MPSBackend(str(fp8), device="cpu")


def test_qwen35_graph_and_mlx_refuse_before_loading(monkeypatch, tmp_path, fp8):
    no_weights(monkeypatch)
    with pytest.raises(ValueError, match="a Qwen3.5 model.*Serve it with --mode vllm"):
        gdn_graph.GDNGraphBackend(str(quantised(tmp_path / "pico-FP8", "qwen35")))
    with pytest.raises(ValueError, match="--mode mlx cannot read a ModelOpt FP8 checkpoint"):
        MLXBackend.__new__(MLXBackend)._load(Path(fp8))  # (before mlx is imported)


def test_strix_refuses_before_loading(monkeypatch, fp8):
    pytest.importorskip("triton")  # (basal.strix imports it)
    from basal import strix
    no_weights(monkeypatch)
    monkeypatch.setattr(strix.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("loaded"))
    monkeypatch.setattr(torch.version, "hip", "6.4")  # (past its ROCm check)
    with pytest.raises(ValueError, match="--mode strix cannot read a ModelOpt FP8 checkpoint"):
        strix.StrixHaloFastBackend(str(fp8))


def test_tpu_refuses_before_loading(fp8):
    pytest.importorskip("jax")
    from basal.tpu import TPUBackend
    with pytest.raises(ValueError, match="--mode tpu cannot read a ModelOpt FP8 checkpoint"):
        TPUBackend(str(fp8))


class Downloaded(Exception):
    pass


def metadata_resolve(monkeypatch, md):
    def resolve(name, revision=None, metadata_only=False):
        if not metadata_only:
            raise Downloaded(name)
        return str(md)
    monkeypatch.setattr(server_module, "resolve", resolve)


@pytest.mark.parametrize("mode", ["eager", "mps", "mlx", "strix", "tpu"])
def test_basal_serve_refuses_it_before_the_download(monkeypatch, fp8, mode):
    metadata_resolve(monkeypatch, fp8)
    with pytest.raises(SystemExit) as e:
        Server(parser().parse_args(["--model", "Remek/basal-omni-1.0-mini-FP8", "--mode", mode]))
    assert str(e.value).startswith(f"Remek/basal-omni-1.0-mini-FP8: --mode {mode} cannot read a ModelOpt FP8 "
                                   "checkpoint")


def test_basal_serve_refuses_eager_with_media_and_nvfp4_in_fast(monkeypatch, tmp_path, fp8):
    metadata_resolve(monkeypatch, fp8)
    with pytest.raises(SystemExit, match="--mode eager cannot read a ModelOpt FP8 checkpoint"):
        Server(parser().parse_args(["--model", "Remek/basal-omni-1.0-mini-FP8", "--mode", "eager", "--media", "on"]))
    metadata_resolve(monkeypatch, quantised(tmp_path / "nvfp4", "qwen3", algo="NVFP4"))
    with pytest.raises(SystemExit, match="--mode fast cannot read a ModelOpt NVFP4 checkpoint"):
        Server(parser().parse_args(["--model", "Org/model-NVFP4", "--mode", "fast"]))


def test_basal_serve_lets_vllm_through(monkeypatch, fp8):
    """vLLM reads ModelOpt checkpoints: the start goes past the check, to the download or to what this machine lacks
    (without vLLM, the mode to serve with here: none picked here reads it, default_mode says so)."""
    metadata_resolve(monkeypatch, fp8)
    with pytest.raises((Downloaded, SystemExit)) as e:
        Server(parser().parse_args(["--model", "Remek/basal-omni-1.0-mini-FP8", "--mode", "vllm"]))
    assert "--mode vllm cannot read" not in str(e.value)


def test_default_mode_never_picks_an_engine_that_cannot_read_it(monkeypatch, tmp_path, fp8):
    """Without --mode: where basal-omni's fast configuration passed its gate the CUDA graph engine serves the -FP8
    checkpoint as before; where the pick is eager (B200's sm_100, Turing), mps, mlx or tpu, the start is refused with
    the modes that serve it. Other models keep their defaults."""
    plain = folder(tmp_path / "plain")
    monkeypatch.setattr(server_module, "resolve", lambda name, revision=None, metadata_only=False: name)
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: "triton")
    assert default_mode(str(fp8)) == default_mode(str(plain)) == "fast"
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: None)
    assert default_mode(str(plain)) == "eager"
    with pytest.raises(SystemExit) as e:
        default_mode(str(fp8))
    assert str(e.value) == (
        f"{fp8}: without --mode this machine serves it with --mode eager, which cannot read a ModelOpt FP8 checkpoint "
        "(config.json \"quantization_config\"): it would load the FP8 weights without their scales, and every answer "
        "would be wrong. Serve it with --mode fast (the CUDA graph engine reads its FP8 layers) or --mode vllm on a "
        "CUDA GPU, or serve its bf16 model")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="serves it with --mode (mlx|mps), which cannot read a ModelOpt FP8"):
        default_mode(str(fp8))
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="serves it with --mode (eager|tpu), which cannot read a ModelOpt FP8"):
        default_mode(str(fp8))
    assert default_mode(str(plain)) in ("eager", "tpu")

