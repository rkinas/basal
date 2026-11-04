"""basal-serve's options are checked in one function that needs no model (serve_options): every refused combination
exits before the model is resolved (downloaded), with the reason, and is testable without weights or a GPU."""
import pytest

from basal import server as S
from basal.server import parser, serve_options

REFUSED = [
    (["--mode", "fast", "--media", "on"], "--media on needs --mode eager or vllm"),
    (["--mode", "fast", "--ple-offload", "on"], "--ple-offload on needs --mode eager"),
    (["--mode", "fast", "--eager-attn", "sdpa"], "--eager-attn needs --mode eager"),
    (["--mode", "eager", "--attn", "fa3"], "--attn fa3 needs the CUDA graph engine"),
    (["--mode", "eager", "--mlx-packed"], "--mlx-packed needs --mode mlx"),
    (["--mode", "mps", "--device", "cuda"], "--mode mps runs on Apple Silicon"),
    (["--mode", "eager", "--batches", "compact"], "--batches needs the CUDA graph engine"),
    (["--mode", "fast", "--fp8-scaling", "block"], "--fp8-scaling block needs --mode fp8"),
    (["--mode", "eager", "--state-cache-gb", "1"], "--state-cache-gb needs the CUDA graph engine"),
    (["--mode", "eager", "--prefix-cache", "on"], "--prefix-cache needs the CUDA graph engine"),
    (["--mode", "eager", "--compile-scope", "layer"], "--compile-scope layer"),
    (["--mode", "gguf"], "--mode gguf needs --gguf"),
    (["--mode", "eager", "--long-lane-gdn", "on"], "--long-lane-gdn on needs --long-lane on"),
    (["--mode", "eager", "--vllm-preset", "deterministic"], "--vllm-preset needs --mode vllm"),
    (["--mode", "eager", "--calibration", "/no/such/CALIBRATION.json"], "no such file"),
]


@pytest.mark.parametrize("args,said", REFUSED)
def test_a_refused_option_exits_before_the_model_is_read(args, said, monkeypatch):
    def resolve(*_, **__):
        raise AssertionError("the model was resolved before the options were checked")
    monkeypatch.setattr(S, "resolve", resolve)
    with pytest.raises(SystemExit, match=said.replace("-", r"\-")):
        S.Server(parser().parse_args(args))


@pytest.mark.parametrize("args,said", REFUSED)
def test_serve_options_refuses_without_a_model(args, said):
    with pytest.raises(SystemExit, match=said.replace("-", r"\-")):
        serve_options(parser().parse_args(args))


@pytest.mark.parametrize("mode", ["eager", "fast", "mps", "mlx", "vllm"])
def test_the_defaults_pass_and_name_the_engine(mode):
    a = parser().parse_args(["--mode", mode])
    kind, quant, comp = serve_options(a)
    assert (kind, quant, comp) == S.MODES[mode] and a.lean is not None and a.tree is not None


def test_the_graph_engines_own_option_checks_run_before_the_weights_are_downloaded(tmp_path, monkeypatch):
    """engine.graph_options (GraphBackend's checks) on the model's metadata, before the full download: --compile-scope
    layer is the bf16 Llama forward's only, which basal-serve's own checks cannot know without the model's config."""
    import json
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": {}}), encoding="utf-8")
    calls = []

    def resolve(name, revision=None, metadata_only=False):
        calls.append(metadata_only)
        if not metadata_only:
            raise AssertionError("the weights were downloaded before the options were checked")
        return tmp_path
    monkeypatch.setattr(S, "resolve", resolve)
    a = parser().parse_args(["--mode", "fast", "--compile-scope", "layer", "--model", "x/gemma"])
    with pytest.raises(SystemExit, match="not Gemma 4"):
        S.Server(a)
    assert calls and all(calls)  # (metadata only)
