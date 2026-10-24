import time

import pytest
from fastapi.testclient import TestClient

from basal.ui import catalog, options, settings
from basal.ui.app import Fleet, create_app
from basal.ui.catalog import Variant
from basal.ui.hardware import Hardware
from basal.ui.history import History
from basal.ui.stats import Stats

CPU = Hardware("cpu", "cpu", 1024**4, None, None, frozenset())
CUDA = Hardware("cuda", "NVIDIA H100", 80 * 1024**3, None, (9, 0), frozenset({"cuda", "vllm", "sglang"}))
BF16 = Variant("mini", "Remek/basal-2.0-mini", "basal-2.0", "mini", "bf16", "bf16", None, 100, False)
GGUF = Variant("mini-gguf", "Remek/basal-2.0-mini-GGUF", "basal-2.0", "mini", "GGUF", "GGUF Q8_0", "m.gguf", 100, False)


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("BASAL_UI_CONFIG", str(tmp_path / "settings.json"))


def wait_for(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def available(variant, hw):
    return {m["mode"] for m in options.available_modes(variant, hw, catalog.assess(variant, hw)) if m["available"]}


def test_defaults_emit_nothing_and_every_non_default_its_flag():
    assert options.args({}) == [] and options.args(dict(options.DEFAULTS)) == []
    flags = options.args({"mode": "fast", "quant": "fp8", "attn": "fa3", "deterministic": True, "orders": "2le4",
                          "auto_band": [0.1, 0.9], "soam": False, "max_batch": 16, "wait_ms": 2, "prefix_cache": True,
                          "state_cache_gb": 1.5, "scheduler": "on", "long_lane": True, "long_tokens": 4096,
                          "context_overflow": "refuse", "truncate_head": 0.5, "max_len": 8192,
                          "vllm_preset": "deterministic", "ollama_model": "basal:mini", "device": "cpu",
                          "think": True, "think_tokens": 512, "early_exit": "0.99"})
    assert flags == ["--quant", "fp8", "--attn", "fa3", "--deterministic", "--orders", "2le4",
                     "--auto-band", "0.1", "0.9", "--soam", "off", "--max-batch", "16", "--wait-ms", "2",
                     "--state-cache-gb", "1.5", "--prefix-cache", "on", "--scheduler", "on", "--long-lane", "on",
                     "--long-tokens", "4096", "--context-overflow", "refuse", "--truncate-head", "0.5",
                     "--max-len", "8192", "--vllm-preset", "deterministic", "--ollama-model", "basal:mini",
                     "--device", "cpu", "--think", "auto", "--think-tokens", "512", "--early-exit", "0.99"]


def test_layout_is_passed_and_refused_with_the_caches():
    assert options.args({"layout": "qfirst"}) == ["--layout", "qfirst"]
    with pytest.raises(ValueError, match="layout"):
        options.validate({"layout": "statefirst"})
    for cache in ({"prefix_cache": True}, {"state_cache_gb": 1.0}):
        with pytest.raises(ValueError, match="need the canonical layout"):
            options.check_combination(BF16, CUDA, options.validate({"layout": "state_twice", **cache}), "fast")
        options.check_combination(BF16, CUDA, options.validate({"layout": "canonical", **cache}), "fast")
    options.check_combination(BF16, CUDA, options.validate({"layout": "qfirst"}), "fast")


@pytest.mark.parametrize("orders", ["2le00", "2le01"])
def test_orders_limit_must_be_at_least_two(orders):
    with pytest.raises(ValueError, match="orders"):
        options.validate({"orders": orders})

@pytest.mark.parametrize("bad, message", [
    ({"colour": 1}, "unknown load option: colour"), ({"orders": "3"}, "orders"), ({"max_batch": 0}, "max_batch"),
    ({"gpu_memory": 0}, "gpu_memory"), ({"truncate_head": 1.5}, "truncate_head"), ({"early_exit": "0.9"}, "early_exit"),
    ({"deterministic": "yes"}, "deterministic"), ({"max_len": 100}, "max_len"), ({"mode": "warp"}, "mode"),
    ({"mode": "mlx", "quant": "fp8"}, "--quant fp8 is not supported with --mode mlx"),
    ({"eager_attn": "sliding=fa9,full=math"}, "eager_attn"), ({"lean": "yes"}, "lean"),
    ({"think_memory": 0}, "think_memory"), ({"ple_offload": "on"}, "ple_offload"),
])
def test_invalid_options_name_the_option(bad, message):
    with pytest.raises(ValueError, match=message):
        options.validate(bad)


def test_every_engine_switch_reaches_basal_serve_as_it_reads_it():
    from basal.server import parser
    chosen = {"gemma_attn": "flex", "merge": "off", "lean": "pass2", "tree": "flex", "batches": "fine",
              "readout_layer": "on", "template_prefix": "off", "fp8_accum": "full", "eager_attn": "sliding=fa3,full=efficient",
              "eager_forward": "hf", "vllm_fast_prefill": "off", "dtype": "float16", "exit_heads": "/heads",
              "think_calibration": "off", "think_model": "/think", "think_adapter": "/adapter", "think_memory": 0.5,
              "think_max_len": 4096, "think_log": "t.log"}
    a = parser().parse_args(["--model", "m", "--ple-offload", "on", "--strict-calibration", *options.args(chosen)])
    assert {k: getattr(a, k) for k in chosen} == chosen
    assert options.args({"ple_offload": True, "strict_calibration": True}) == [
        "--ple-offload", "on", "--strict-calibration"]
    defaults = parser().parse_args(["--model", "m"])
    assert {k: getattr(defaults, k) for k in chosen} == {k: options.DEFAULTS[k] for k in chosen}


def test_media_loads_are_eager_or_vllm_where_cuda_has_vllm():
    omni = Variant("omni", "Remek/basal-omni-1.0-mini", "basal-omni-1.0", "mini", "bf16", "bf16", None, 100, True,
                   media=("image", "audio"))
    media_cuda = Hardware("cuda", "NVIDIA H100", 80 * 1024**3, None, (9, 0), frozenset({"cuda", "vllm", "media"}))
    no_vllm = Hardware("cuda", "NVIDIA H100", 80 * 1024**3, None, (9, 0), frozenset({"cuda", "media"}))
    modes = lambda hw: {m["mode"]: m["available"] for m in options.available_media_modes(catalog.assess(omni, hw), hw)}
    assert modes(media_cuda) == {"eager": True, "vllm": True}
    assert modes(no_vllm) == {"eager": True, "vllm": False}
    assert modes(CUDA) == {"eager": False, "vllm": False}  # no basal[media]
    assert catalog.serve_args(omni, "/m", "vllm", "cuda")[-4:] == ["--mode", "vllm", "--media", "on"]
    assert catalog.serve_args(omni, "/m", "eager", "apple")[-4:] == ["--media", "on", "--device", "mps"]


def test_available_modes_follow_the_hardware_and_the_variant():
    assert available(BF16, CPU) == {"eager", "mps", "ollama", "llamacpp"}
    gguf_cpu = Hardware("cpu", "cpu", 1024**4, None, None, frozenset({"gguf"}))
    assert available(GGUF, gguf_cpu) == {"gguf", "ollama", "llamacpp"}
    cuda = available(BF16, CUDA)
    assert {"fast", "fast-nocompile", "fp8", "eager", "vllm", "sglang"} <= cuda
    assert not cuda & {"tpu", "strix", "mlx", "nvfp4", "fast-exit"}
    reasons = {m["mode"]: m["reason"] for m in options.available_modes(BF16, CUDA, catalog.assess(BF16, CUDA))}
    assert reasons["fast-exit"] == "exit heads ship with basal-1.0" and "TPU" in reasons["tpu"]
    assert [m["mode"] for m in options.available_modes(BF16, CUDA, catalog.assess(BF16, CUDA)) if m["default"]] == ["fast"]


def test_cuda_graph_modes_require_ampere_or_newer():
    sm75 = Hardware("cuda", "old NVIDIA", 8 * 1024**3, None, (7, 5), frozenset({"cuda"}))
    modes = {m["mode"]: m for m in options.available_modes(BF16, sm75, catalog.assess(BF16, sm75))}
    for mode in ("fast", "fast-nocompile", "fast-exit", "fp8", "nvfp4"):
        assert not modes[mode]["available"]
        assert "8.0+" in modes[mode]["reason"]
    sm86 = Hardware("cuda", "RTX 3090", 24 * 1024**3, None, (8, 6), frozenset({"cuda"}))
    modes = {m["mode"]: m for m in options.available_modes(BF16, sm86, catalog.assess(BF16, sm86))}
    assert modes["fast"]["available"] and not modes["fp8"]["available"]
    assert modes["fp8"]["reason"] == options.fp8_unavailable(sm86) and "8.9+" in modes["fp8"]["reason"]
    with pytest.raises(ValueError, match=r"8\.9\+"):
        options.check_combination(BF16, sm86, options.validate({"quant": "fp8"}, "fast"), "fast")
    sm89 = Hardware("cuda", "RTX 4090", 24 * 1024**3, None, (8, 9), frozenset({"cuda"}))
    options.check_combination(BF16, sm89, options.validate({"quant": "fp8"}, "fast"), "fast")


def test_pico_modes_follow_basal_serve_2_4_refusals():
    pico = Variant("pico", "Remek/basal-omni-pico", "basal-omni", "pico", "bf16", "bf16", None, 100, True,
                   recurrent=True)
    fla = Hardware("cuda", "NVIDIA H100", 80 * 1024**3, None, (9, 0), frozenset({"cuda", "vllm", "fla"}))
    modes = {m["mode"]: m for m in options.available_modes(pico, fla, catalog.assess(pico, fla))}
    assert {m for m, row in modes.items() if row["available"]} >= {"fast", "fast-nocompile", "eager", "vllm"}
    assert [m for m, row in modes.items() if row["default"]] == ["fast"]
    for mode in ("fast-exit", "mps"):
        assert modes[mode]["reason"] == options.RECURRENT_REASON
    for mode in ("fp8", "nvfp4"):
        assert not modes[mode]["available"] and "Qwen3.5" in modes[mode]["reason"]
    assert modes["mlx"]["reason"] == modes["mlx-q8"]["reason"] == options.MLX_RECURRENT_REASON
    no_fla = {m["mode"]: m for m in options.available_modes(pico, CUDA, catalog.assess(pico, CUDA))}
    assert no_fla["fast"]["reason"] == no_fla["fast-nocompile"]["reason"] == options.FLA_REASON
    assert [m for m, row in no_fla.items() if row["default"]] == ["eager"]
    options.check_combination(pico, fla, options.validate({}, "fast"), "fast")
    for refused in ({"batches": "compact"}, {"prefix_cache": True}, {"state_cache_gb": 1.0}, {"long_lane": True},
                    {"lean": "on"}, {"quant": "fp8"}):
        with pytest.raises(ValueError, match=r"Qwen3\.5"):
            options.check_combination(pico, fla, options.validate(refused), "fast")
    options.check_combination(pico, fla, options.validate({"prefix_cache": True}), "eager")


def test_thinking_needs_basal_2_0_4_5b_with_vllm_and_early_exit_needs_fast_exit():
    big = Variant("big", "Remek/basal-2.0-4.5B", "basal-2.0", "4.5B", "bf16", "bf16", None, 100, False)
    assert options.args({"think": True, "think_tokens": 512, "early_exit": "0.99", "mode": "fast-exit"}) == [
        "--think", "auto", "--think-tokens", "512", "--early-exit", "0.99"]
    merged = options.validate({"think": True})
    assert options.think_unavailable(BF16, CUDA) == options.THINK_REASON  # the mini has no thinking model
    assert options.think_unavailable(big, CUDA) is None
    assert "Python of an environment with vLLM" in options.think_unavailable(big, CPU)
    assert options.think_unavailable(big, CPU, {"think_url": "http://gpu:8100"}) is None
    with pytest.raises(ValueError, match="basal-2.0 4.5B"):
        options.check_combination(BF16, CUDA, merged, "fast")
    with pytest.raises(ValueError, match="early_exit needs --mode fast-exit"):
        options.check_combination(big, CUDA, options.validate({"early_exit": "0.99"}), "fast")
    options.check_combination(big, CUDA, options.validate({"early_exit": "0.99", "mode": "fast-exit"}), "fast")


def test_load_options_reach_the_command_are_remembered_and_reused_by_restart(tmp_path, monkeypatch):
    fleet = Fleet(models_dir_fn=lambda: tmp_path, platform_fn=lambda: "cpu", settings_mod=settings,
                  hardware_fn=lambda: CPU, stats=Stats(tmp_path / "s.json"), history=History(tmp_path / "h.jsonl"))
    commands = []

    def start(self, assessment, gen, media=False, cache_dir=None, serve=None):
        commands.append(self._build_command(BF16, tmp_path, assessment, media, 1, settings.serve_settings()))
        with self.lock:
            self.state = "ready"
        settings.remember_loaded(assessment.variant.id, media, self.options)

    monkeypatch.setattr("basal.ui.app.Runner._start", start)
    app = create_app(catalog_fn=lambda **_: ([BF16], "test"), hardware_fn=lambda: CPU, fleet=fleet, stats=fleet.stats,
                     history=fleet.history)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        bad = client.post("/api/load", json={"id": "mini", "options": {"orders": "7"}})
        assert bad.status_code == 400 and "orders" in bad.json()["detail"]
        unavailable = client.post("/api/load", json={"id": "mini", "options": {"mode": "mlx"}})
        assert unavailable.status_code == 400 and unavailable.json()["detail"].startswith("--mode mlx:")
        chosen = {"deterministic": True, "orders": "2", "context_overflow": "refuse"}
        assert client.post("/api/load", json={"id": "mini", "options": chosen}).status_code == 202
        wait_for(lambda: len(commands) == 1)
        assert commands[-1][-5:] == ["--deterministic", "--orders", "2", "--context-overflow", "refuse"]
        assert client.get("/api/status").json()["models"][0]["options"] == chosen
        assert settings.last_models() == [{"id": "mini", "media": False, "options": chosen}]
        assert client.post("/api/restart", json={"id": "mini"}).status_code == 202
        wait_for(lambda: len(commands) == 2)
        assert commands[-1][-5:] == ["--deterministic", "--orders", "2", "--context-overflow", "refuse"]
        client.post("/api/restart", json={"id": "mini", "options": {}})
        wait_for(lambda: len(commands) == 3)
        assert "--deterministic" not in commands[-1]
        saved = client.put("/api/models/options", json={"id": "mini", "options": {"max_batch": 8}}).json()
        assert saved["options"]["max_batch"] == 8 and saved["saved"] is True
        assert settings.load_options("mini") == {"max_batch": 8}
        listed = client.get("/api/models/options", params={"id": "mini"}).json()
        assert listed["defaults"]["max_batch"] == 64 and {m["mode"] for m in listed["modes"]} >= {"eager", "tpu"}


def test_revision_is_validated_but_never_emitted_as_a_serve_flag():
    assert options.validate({"revision": "refs/pr/4"})["revision"] == "refs/pr/4"
    assert options.args({"revision": "abc123"}) == []
    with pytest.raises(ValueError, match="revision"):
        options.validate({"revision": "bad revision"})
    assert options.validate({"revision": ""})["revision"] is None  # a cleared field: the default branch
    assert options.validate({"revision": " abc "})["revision"] == "abc"
