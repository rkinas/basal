import math
from pathlib import Path

import pytest

from basal.ui.catalog import (
    Variant,
    _label,
    _media_mode,
    _variants,
    assess,
    delete,
    download_kwargs,
    parse_repo,
    serve_args,
    suggest,
)
from basal.ui import catalog
from basal.ui.catalog import base_dir, download, is_downloaded
from basal.ui.hardware import Hardware


def hw(platform="cuda", name="H100", cap=(9, 0), engines=frozenset({"cuda"}), memory=80 * 1024**3,
       available=None):
    return Hardware(platform, name, memory, memory, cap, engines, available)


def variant(repo="Remek/basal-2.0-4.5B", port=None, size=None, gguf=None):
    family, tier, parsed_port = parse_repo(repo)
    port = port or parsed_port
    size = size if size is not None else int({"mini": 2, "4.5B": 8, "max": 12, "ultra": 16}.get(tier, 8) * 1024**3)
    return Variant(f"{repo}:{gguf}" if gguf else repo, repo, family, tier, port, _label(port, gguf), gguf, size,
                   family.startswith("basal-omni"), recurrent=(family, tier) in catalog.RECURRENT_MODELS)


def test_hardware_detects_media_engine(monkeypatch):
    from basal.ui import hardware

    find_spec = hardware.importlib.util.find_spec
    packages = {"PIL", "soundfile", "scipy", "torchvision"}
    monkeypatch.setattr(hardware.importlib.util, "find_spec",
                        lambda name: object() if name in packages else find_spec(name))
    hardware._static.cache_clear()
    assert "media" in hardware.detect().engines
    monkeypatch.setattr(hardware.importlib.util, "find_spec",
                        lambda name: None if name == "torchvision" else find_spec(name))
    hardware._static.cache_clear()
    assert "media" not in hardware.detect().engines
    hardware._static.cache_clear()


def test_media_kinds_and_mode_eligibility():
    for tier, kinds in (("mini", ("image", "audio")), ("max", ("image", "audio")), ("ultra", ("image",))):
        (v,) = _variants(f"Remek/basal-omni-1.0-{tier}", {})
        assert v.media == kinds
        (mlx,) = _variants(f"Remek/basal-omni-1.0-{tier}-MLX-8bit", {})
        assert mlx.media == ()
    for repo in ("Remek/basal-omni-1.1-max", "Remek/basal-omni-1.0-mini-MLX"):
        (v,) = _variants(repo, {})
        assert v.media == ()

    v = _variants("Remek/basal-omni-1.0-mini", {})[0]
    for platform in ("cuda", "apple", "cpu"):
        assert _media_mode(v, hw(platform, engines=frozenset({"media"}))) == ("eager", None)
        mode, reason = _media_mode(v, hw(platform, engines=frozenset()))
        assert mode is None and "install basal[media]" in reason
    assert _media_mode(_variants("Remek/basal-2.0-mini", {})[0], hw())[1] == \
        "this checkpoint has no image or audio encoder"


def test_media_serve_args_devices():
    v = _variants("Remek/basal-omni-1.0-mini", {})[0]
    for platform, device in (("cuda", []), ("apple", ["--device", "mps"]), ("cpu", ["--device", "cpu"])):
        args = serve_args(v, Path("/model"), "mlx", media_platform=platform)
        assert args == ["--model", "/model", "--mode", "eager", "--media", "on", *device]


def test_parse_repo_contract_names():
    assert parse_repo("Remek/basal-omni-1.0-max-GGUF") == ("basal-omni-1.0", "max", "GGUF")
    assert parse_repo("Remek/basal-2.0-4.5B-MLX-8bit") == ("basal-2.0", "4.5B", "MLX-8bit")
    assert parse_repo("Remek/basalt-1.0-4.5B") is None
    assert parse_repo("Remek/basal-1.0-4.5B-GGUF-extra") is None
    assert parse_repo("Remek/basal-omni-pico") == ("basal-omni", "pico", "bf16")
    assert parse_repo("Remek/basal-omni-pico-GGUF") == ("basal-omni", "pico", "GGUF")
    assert parse_repo("Remek/basal-pico") is None
    assert parse_repo("Remek/basal-omni-1.0-pico") is None


def test_quantisation_reads_the_repo_markers():
    mlx = variant("Remek/basal-2.0-mini-MLX-8bit")
    config = {"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    assert catalog.quantisation(mlx, None, config)["scheme"] == "MLX affine 8-bit, group size 64"
    gguf = variant("Remek/basal-omni-pico-GGUF", "GGUF", gguf="basal-omni-pico-Q8_0.gguf")
    assert catalog.quantisation(gguf, None, None)["scheme"] == "GGUF Q8_0"
    fp8 = variant("Remek/basal-omni-1.0-mini-FP8")
    basal_json = {"quantization": "ModelOpt FP8 of the language model", "engine": "basal-serve --mode vllm"}
    assert catalog.quantisation(fp8, basal_json, {"quantization_config": {"quant_algo": "FP8"}}) == {
        "scheme": "FP8", "description": basal_json["quantization"], "engine": basal_json["engine"]}
    assert catalog.quantisation(fp8, None, None) == {"scheme": None, "description": None, "engine": None}


def test_assessment_runtime_engines_and_fit_boundaries():
    fp8 = variant("Remek/basal-2.0-4.5B-FP8", "FP8")
    assert assess(fp8, hw(cap=(8, 9))).runnable
    assert not assess(fp8, hw(cap=(8, 6))).runnable
    mlx = variant("Remek/basal-2.0-4.5B-MLX-8bit", "MLX-8bit")
    assert not assess(mlx, hw(platform="cuda", engines=frozenset({"cuda"}))).runnable
    assert assess(mlx, hw(platform="apple", cap=None, engines=frozenset({"mlx"}))).runnable
    gguf = variant("Remek/basal-2.0-4.5B-GGUF", "GGUF", gguf="model-Q8_0.gguf")
    a = assess(gguf, hw(engines=frozenset()))
    assert not a.runnable and "basal[gguf]" in a.reason
    assert not assess(variant("Remek/basal-omni-1.0-mini", "bf16"), hw(platform="cpu", cap=None, engines=frozenset())).runnable
    size = 10 * 1024**3
    v = variant(size=size)
    budget = size * 1.2 + 4 * 1024**3  # the H100 default is fast: the CUDA graph engine's overhead
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 64 * 1024**3)
    assert assess(v, apple).est_memory_bytes == int(size * 1.2) + 1024**3
    assert assess(v, hw(memory=math.ceil(budget / .85))).fit == "fits"
    assert assess(v, hw(memory=int(budget))).fit == "tight"
    assert assess(v, hw(memory=int(budget), available=math.ceil(budget / .85))).fit_now == "fits"
    assert assess(v, hw(memory=int(budget), available=int(budget))).fit_now == "tight"
    assert assess(v, hw(memory=int(budget), available=0)).fit_now == "too_large"
    unknown = Variant("unknown", "Remek/basal-2.0-4.5B", "basal-2.0", "4.5B", "bf16", "bf16", None, None, False)
    assert assess(unknown, hw()).fit == "unknown"
    assert assess(v, hw()).fit_now == "unknown"
    weights_4_5b = variant(size=9_510_000_000)  # basal-2.0-4.5B in bf16, --mode fast measured 14.4-15.2 GB
    assert assess(weights_4_5b, Hardware("cuda", "RTX 4060 Ti", 17_179_869_184, None, (8, 9),
                                         frozenset({"cuda"}))).fit == "tight"
    assert assess(weights_4_5b, Hardware("cuda", "RTX 4090", 25_769_803_776, None, (8, 9),
                                         frozenset({"cuda"}))).fit == "fits"


def test_suggestions_by_hardware_and_available_memory():
    def choices(hw_, repos):  # built the way the catalog builds them, so basal-omni bf16 carries its media kinds
        return suggest([assess(_variants(repo, {"model.safetensors": variant(repo).weights_bytes})[0], hw_)
                        for repo in repos], hw_)

    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 24 * 1024**3, 6 * 1024**3)
    repos = ["Remek/basal-2.0-4.5B-MLX-8bit", "Remek/basal-2.0-max-MLX-8bit",
             "Remek/basal-2.0-mini-MLX-8bit"]
    cards = choices(apple, repos)
    assert cards[0]["slot"] == "recommended"
    assert cards[0]["id"] == "Remek/basal-2.0-mini-MLX-8bit"
    assert cards[0]["why"].endswith("; about 1.5x faster with --mlx-packed (Load options)")

    dc = hw("cuda", "H100", (9, 0), frozenset({"cuda"}), 80 * 1024**3, 70 * 1024**3)
    cards = choices(dc, ["Remek/basal-2.0-mini", "Remek/basal-2.0-4.5B", "Remek/basal-2.0-max"])
    assert [(card["slot"], card["id"]) for card in cards[:3]] == [
        ("recommended", "Remek/basal-2.0-4.5B"),
        ("best_quality", "Remek/basal-2.0-max"),
        ("fastest", "Remek/basal-2.0-mini"),
    ]
    assert cards[0]["why"] == "4.5B in bf16: needs about 14.6 GB of 75.2 GB available"
    assert "--mlx-packed" not in cards[0]["why"]

    omni = "Remek/basal-omni-1.0-mini"
    without_media = choices(dc, [omni])
    assert all(card["slot"] != "multimodal" for card in without_media)
    media_hw = hw("cuda", "H100", (9, 0), frozenset({"cuda", "media"}), 80 * 1024**3, 70 * 1024**3)
    media_cards = choices(media_hw, [omni])
    assert media_cards == [{"slot": "multimodal", "id": omni,
                            "why": "images and audio, served with --mode eager"}]


def test_pico_is_recurrent_and_follows_basal_serve_default_modes():
    pico = variant("Remek/basal-omni-pico", size=1_700_000_000)
    assert pico.recurrent and pico.experimental and pico.media == ()
    (built,) = _variants("Remek/basal-omni-pico", {"model.safetensors": 1})
    assert built.recurrent and built.media == ()
    # basal-serve 2.4: fast on Hopper with flash-linear-attention, eager on other GPUs, Apple Silicon and the CPU
    assert assess(pico, hw(engines=frozenset({"cuda", "fla"}))).mode == "fast"
    for machine in (hw(), hw(cap=(8, 9), engines=frozenset({"cuda", "fla"})),
                    hw("apple", "M4 Max", None, frozenset({"mlx"}), 64 * 1024**3),
                    hw("cpu", "cpu", None, frozenset())):
        a = assess(pico, machine)
        assert a.runnable and a.mode == "eager"
    assert serve_args(pico, Path("/m"), "eager", platform="cpu")[-2:] == ["--device", "cpu"]
    gguf = variant("Remek/basal-omni-pico-GGUF", "GGUF", 900_000_000, "basal-omni-pico-Q8_0.gguf")
    assert assess(gguf, hw("cpu", "cpu", None, frozenset({"gguf"}))).mode == "gguf"


def test_lightest_suggestion_is_pico_in_the_preferred_format():
    pico = variant("Remek/basal-omni-pico", size=1_700_000_000)
    gguf = variant("Remek/basal-omni-pico-GGUF", "GGUF", 900_000_000, "basal-omni-pico-Q8_0.gguf")
    cpu = hw("cpu", "CPU", None, frozenset({"gguf"}), 64 * 1024**3, 64 * 1024**3)
    cards = suggest([assess(v, cpu) for v in (pico, gguf, variant())], cpu)
    (light,) = [card for card in cards if card["slot"] == "lightest"]
    assert light["id"] == gguf.id and light["why"].endswith("served with --mode gguf")
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 64 * 1024**3)
    cards = suggest([assess(v, apple) for v in (pico, gguf, variant("Remek/basal-2.0-mini-MLX-8bit"))], apple)
    assert [card["slot"] for card in cards] == ["recommended", "lightest"]
    assert cards[1]["id"] == pico.id and cards[1]["why"].endswith("served with --mode eager")


def test_available_memory_parses_macos_vm_stat(monkeypatch):
    from basal.ui import hardware

    output = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
              "Pages free: 10.\nPages inactive: 20.\nPages speculative: 3.\n")
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.subprocess, "check_output", lambda *args, **kwargs: output)
    assert hardware._available_bytes() == 33 * 16384


def test_memory_on_windows_comes_from_global_memory_status(monkeypatch):
    from basal.ui import hardware

    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_windows_memory", lambda: (16 * 1024**3, 6 * 1024**3))
    assert hardware._ram() == 16 * 1024**3 and hardware._available_bytes() == 6 * 1024**3
    monkeypatch.setattr(hardware, "_windows_memory", lambda: None)
    assert hardware._ram() is None and hardware._available_bytes() is None


def test_gguf_launch_and_download_patterns():
    v = variant("Remek/basal-2.0-4.5B-GGUF", "GGUF", gguf="basal-2.0-4.5B-Q8_0.gguf")
    assert serve_args(v, Path("/models/repo"), "gguf")[-2:] == ["--gguf", "/models/repo/basal-2.0-4.5B-Q8_0.gguf"]
    assert download_kwargs(v)["allow_patterns"][0] == v.gguf_file


def test_thinking_model_is_neither_counted_nor_downloaded():
    (v,) = _variants("Remek/basal-2.0-4.5B", {"model.safetensors": 9, "think/model.safetensors": 9, "basal.json": 1})
    assert v.weights_bytes == 9
    assert download_kwargs(v) == {"ignore_patterns": ["think/*"]}


def test_serving_mode_per_family_and_port():
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 64 * 1024**3)
    assert assess(variant("Remek/basal-2.0-4.5B-MLX-8bit", "MLX-8bit"), apple).mode == "mlx"  # the port as stored
    h100 = hw()
    assert assess(variant("Remek/basal-omni-1.0-max", "bf16"), h100).mode == "eager"
    older_fp8 = assess(variant("Remek/basal-1.5-4.5B-FP8", "FP8"), h100)
    assert not older_fp8.runnable and "basal[vllm]" in older_fp8.reason
    assert assess(variant("Remek/basal-1.5-4.5B-FP8", "FP8"), hw(engines=frozenset({"cuda", "vllm"}))).mode == "vllm"
    assert "--device" not in serve_args(variant("Remek/basal-omni-1.0-max", "bf16"), Path("/m"), "eager")


def test_eager_cpu_device_flag_is_only_set_for_cpu_platform():
    v = variant("Remek/basal-2.0-mini")
    assert "--device" not in serve_args(v, Path("/m"), "eager")
    assert serve_args(v, Path("/m"), "eager", platform="cpu")[-2:] == ["--device", "cpu"]


def test_tpu_is_default_for_bf16_text_models_when_available():
    tpu = hw("cpu", "TPU VM", None, frozenset({"tpu"}))
    assert assess(variant("Remek/basal-2.0-mini"), tpu).mode == "tpu"


def test_cpu_with_llama_cpp_prefers_gguf_q8():
    cpu = hw("cpu", "CPU", None, frozenset({"gguf"}), 64 * 1024**3, 64 * 1024**3)
    repo = "Remek/basal-2.0-4.5B-GGUF"
    items = [assess(Variant(f"{repo}:{f}", repo, "basal-2.0", "4.5B", "GGUF", label, f, 5 * 1024**3, False), cpu)
             for f, label in (("basal-2.0-4.5B-Q5_K_M.gguf", "GGUF Q5_K_M"), ("basal-2.0-4.5B-Q8_0.gguf", "GGUF Q8_0"))]
    items.append(assess(variant(), cpu))
    assert suggest(items, cpu)[0]["id"] == f"{repo}:basal-2.0-4.5B-Q8_0.gguf"


def test_delete_gguf_variants_removes_only_selected_file_then_repo(tmp_path):
    repo_id = "Remek/basal-2.0-mini-GGUF"
    repo = tmp_path / "models--Remek--basal-2.0-mini-GGUF"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    names = ("basal-2.0-mini-Q4_K_M.gguf", "basal-2.0-mini-Q8_0.gguf")
    for index, name in enumerate(names, 1):
        blob = blobs / (str(index) * 64)
        blob.write_bytes(bytes([index]) * index)
        (snapshot / name).symlink_to(blob)
    first, second = _variants(repo_id, {name: index for index, name in enumerate(names, 1)})

    assert delete(first, tmp_path) == 1
    assert not (snapshot / names[0]).exists()
    assert (snapshot / names[1]).exists()
    assert repo.is_dir()
    assert delete(second, tmp_path) == 2
    assert not repo.exists()


def test_old_cuda_bf16_requires_ampere_but_omni_remains_eager():
    sm75 = hw(cap=(7, 5))
    text = assess(variant("Remek/basal-2.0-mini"), sm75)
    assert not text.runnable
    assert text.reason == ("The basal engine needs an NVIDIA GPU with compute capability 8.0+ (Ampere or newer); "
                           "on this GPU use a GGUF file with basal[gguf].")
    omni = assess(variant("Remek/basal-omni-1.0-mini"), sm75)
    assert omni.runnable and omni.mode == "eager"


def test_suggestion_prefers_fit_before_preferred_format():
    apple = hw("apple", "M4 Max", None, frozenset({"mlx"}), 24 * 1024**3, 4_200_000_000)
    preferred = variant("Remek/basal-2.0-mini-MLX-8bit", "MLX-8bit", 3_000_000_000)
    compact = variant("Remek/basal-2.0-mini-MLX-fp4", "MLX-fp4", 2_000_000_000)
    cards = suggest([assess(preferred, apple), assess(compact, apple)], apple)
    assert cards[0]["id"] == compact.id


def _gguf_cache(tmp_path):
    repo = tmp_path / "models--Remek--basal-2.0-mini-GGUF"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / ("b" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("b" * 40)
    return blobs, snapshot


def test_delete_gguf_in_a_cache_of_copied_files_counts_the_freed_file(tmp_path):
    blobs, snapshot = _gguf_cache(tmp_path)
    selected, other = "basal-2.0-mini-Q4_K_M.gguf", "basal-2.0-mini-Q8_0.gguf"
    (snapshot / selected).write_bytes(b"12345")  # no symlinks (Windows default): the snapshot holds the file
    (blobs / ("c" * 64)).write_bytes(b"67")
    (snapshot / other).symlink_to(blobs / ("c" * 64))
    first, _ = _variants("Remek/basal-2.0-mini-GGUF", {selected: 5, other: 2})
    assert delete(first, tmp_path) == 5
    assert not (snapshot / selected).exists()
    assert (snapshot / other).read_bytes() == b"67"


def test_delete_gguf_keeps_a_blob_another_file_name_still_uses(tmp_path):
    blobs, snapshot = _gguf_cache(tmp_path)
    selected, other = "basal-2.0-mini-Q4_K_M.gguf", "basal-2.0-mini-Q8_0.gguf"
    blob = blobs / ("c" * 64)
    blob.write_bytes(b"12345")
    (snapshot / selected).symlink_to(blob)
    (snapshot / other).symlink_to(blob)
    first, _ = _variants("Remek/basal-2.0-mini-GGUF", {selected: 5, other: 5})
    assert delete(first, tmp_path) == 0
    assert not (snapshot / selected).exists()
    assert (snapshot / other).read_bytes() == b"12345"
def test_new_community_ports_and_weights_only_gguf(monkeypatch):
    class Api:
        def list_models(self, *, author, search):
            return [type("Model", (), {"id": f"{author}/basal-1.0-4.5B-GGUF"})(),
                    type("Model", (), {"id": "other/basal-1.0-4.5B-GGUF"})()]

        def model_info(self, repo, *, files_metadata):
            return type("Info", (), {
                "siblings": [type("File", (), {"rfilename": "model-Q8_0.gguf", "size": 12})()],
                "card_data": type("Card", (), {"base_model": "Remek/basal-1.0-4.5B"})(),
            })()

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    catalog._hub_variants.cache_clear()
    variants = catalog._hub_variants()
    assert {v.author for v in variants} == set(catalog.AUTHORS)
    assert all(v.community == (v.author != "Remek") and v.base_repo == "Remek/basal-1.0-4.5B" for v in variants)
    assert parse_repo("agentGreg/basal-1.5-mini-MLX-6bit")[2] == "MLX-6bit"
    assert parse_repo("agentGreg/basal-1.5-mini-MLX-bf16")[2] == "MLX-bf16"
    assert parse_repo("pawelkiszczak/basal-1.0-4.5B-oQ6e")[2] == "MLX-oQ6e"
    assert _label("MLX-6bit") == "MLX 6-bit"
    assert _label("MLX-bf16") == "MLX bf16"
    assert _label("MLX-oQ6e") == "MLX oQ6e (oMLX 6-bit)"
    assert not suggest([assess(v, hw()) for v in variants], hw())
    community_media = Variant("community-media", "agentGreg/basal-omni-1.0-mini", "basal-omni-1.0", "mini",
                              "bf16", "bf16", None, 10, True, media=("image",), author="agentGreg")
    media_hw = hw(engines=frozenset({"cuda", "media"}))
    assert suggest([assess(community_media, media_hw)], media_hw) == []
    catalog._hub_variants.cache_clear()


def test_community_cache_reads_weights_only_gguf_base_model(monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nbase_model:\n- Remek/basal-1.0-4.5B\nbase_model_revision: abc123\n---\n")
    readmefile = type("File", (), {"file_name": "README.md", "size_on_disk": readme.stat().st_size,
                                   "file_path": readme})()
    file = type("File", (), {"file_name": "model-Q8_0.gguf", "size_on_disk": 12,
                             "file_path": readme})()
    repo = type("Repo", (), {
        "repo_type": "model", "repo_id": "pawelkiszczak/basal-1.0-4.5B-GGUF",
        "revisions": [type("Revision", (), {"files": [file, readmefile]})()],
    })()
    outsider = type("Repo", (), {
        "repo_type": "model", "repo_id": "other/basal-1.0-4.5B-GGUF",
        "revisions": [type("Revision", (), {"files": [file, readmefile]})()],
    })()
    monkeypatch.setattr("huggingface_hub.scan_cache_dir",
                        lambda _: type("Cache", (), {"repos": [repo, outsider]})())
    (v,) = catalog._cache_variants(tmp_path)
    assert v.community and v.author == "pawelkiszczak"
    assert v.base_repo == "Remek/basal-1.0-4.5B" and v.base_revision == "abc123" and v.weights_only


def test_weights_only_gguf_download_and_serve_require_base_metadata(monkeypatch, tmp_path):
    v = Variant("repo:model-Q8_0.gguf", "pawelkiszczak/basal-1.0-4.5B-GGUF", "basal-1.0", "4.5B",
                "GGUF", "GGUF Q8_0", "model-Q8_0.gguf", 12, False,
                author="pawelkiszczak", base_repo="Remek/basal-1.0-4.5B", base_revision="b" * 40, weights_only=True)
    variant_snapshot = tmp_path / "models--pawelkiszczak--basal-1.0-4.5B-GGUF" / "snapshots" / ("a" * 40)
    metadata_snapshot = tmp_path / "models--Remek--basal-1.0-4.5B" / "snapshots" / ("b" * 40)
    variant_snapshot.mkdir(parents=True)
    (variant_snapshot / v.gguf_file).write_text("gguf")
    metadata_snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "CALIBRATION.json"):
        (metadata_snapshot / name).write_text("{}")
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(metadata_snapshot if repo == v.base_repo else variant_snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    assert download_kwargs(v, "deadbeef")["revision"] == "deadbeef"
    assert download(v, tmp_path, revision="deadbeef") == str(variant_snapshot)
    assert [(repo_name, kwargs.get("revision")) for repo_name, kwargs in calls] == [
        (v.repo, "deadbeef"), (v.base_repo, "b" * 40)  # the base snapshot pinned to the card's base_model_revision
    ]
    calls.clear()
    assert is_downloaded(v, tmp_path, "deadbeef") == variant_snapshot
    assert calls[0][1]["revision"] == "deadbeef"
    assert base_dir(v, tmp_path) == metadata_snapshot and calls[-1][1]["revision"] == "b" * 40
    local = tmp_path / "models--pawelkiszczak--basal-1.0-4.5B-GGUF" / "snapshots" / ("a" * 40)
    assert serve_args(v, local, "gguf") == [
        "--model", str(metadata_snapshot), "--mode", "gguf", "--gguf", str(local / v.gguf_file)
    ]
    (metadata_snapshot / "CALIBRATION.json").unlink()  # config.json and tokenizer only: not the served metadata
    assert base_dir(v, tmp_path) is None
    assert is_downloaded(v, tmp_path, "deadbeef") is None


def test_weights_only_gguf_without_a_base_model_is_not_runnable():
    gguf_hw = hw(engines=frozenset({"gguf"}))
    (orphan,) = _variants("pawelkiszczak/basal-1.0-4.5B-GGUF", {"model-Q8_0.gguf": 12}, "pawelkiszczak")
    assert orphan.weights_only and orphan.base_repo is None
    assessment = assess(orphan, gguf_hw)
    assert not assessment.runnable and "names no base model" in assessment.reason
    (based,) = _variants("pawelkiszczak/basal-1.0-4.5B-GGUF", {"model-Q8_0.gguf": 12}, "pawelkiszczak",
                         "Remek/basal-1.0-4.5B")
    assert assess(based, gguf_hw).mode == "gguf"
    (shipped,) = _variants("Remek/basal-2.0-mini-GGUF", {"m-Q8_0.gguf": 7, "tokenizer.json": 1})
    assert not shipped.weights_only and assess(shipped, gguf_hw).mode == "gguf"


@pytest.mark.parametrize("text, expected", [
    ("---\nbase_model: Remek/basal-2.0-mini\n---\n", "Remek/basal-2.0-mini"),
    ("---\nbase_model: 'Remek/basal-2.0-mini'\n---\n", "Remek/basal-2.0-mini"),
    ("---\nbase_model:\n  - Remek/basal-2.0-mini\n  - other/x\n---\n", "Remek/basal-2.0-mini"),
    ("---\nbase_model: [Remek/basal-2.0-mini, other/x]\n---\n", "Remek/basal-2.0-mini"),
    ("---\nbase_model_relation: quantized\n---\n", None),
])
def test_readme_base_model_scalar_and_list_forms(text, expected):
    assert catalog._front_matter(text, "base_model") == expected


def test_hub_base_model_list_takes_the_first_entry(monkeypatch):
    class Api:
        def list_models(self, *, author, search):
            return [type("Model", (), {"id": f"{author}/basal-1.0-4.5B-GGUF"})()] if author == "pawelkiszczak" else []

        def model_info(self, repo, *, files_metadata):
            return type("Info", (), {
                "siblings": [type("File", (), {"rfilename": "model-Q8_0.gguf", "size": 12})()],
                "card_data": type("Card", (), {"base_model": ["Remek/basal-1.0-4.5B", "other/x"],
                                               "base_model_revision": "abc"})(),
            })()

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    catalog._hub_variants.cache_clear()
    try:
        (v,) = catalog._hub_variants()
    finally:
        catalog._hub_variants.cache_clear()
    assert v.base_repo == "Remek/basal-1.0-4.5B" and v.base_revision == "abc"


def test_base_model_gguf_delete_keeps_its_metadata(tmp_path):
    blobs, snapshot = _gguf_cache(tmp_path)
    name = "basal-2.0-mini-Q8_0.gguf"
    (snapshot / name).write_bytes(b"weights")
    repo = "Remek/basal-2.0-mini-GGUF"
    (snapshot.parent.parent / "refs" / "main").write_text("b" * 40)
    base = tmp_path / "models--Remek--basal-2.0-mini"
    base_snapshot = base / "snapshots" / ("c" * 40)
    base_snapshot.mkdir(parents=True)
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("c" * 40)
    (base_snapshot / "config.json").write_text("{}")
    v = Variant(f"{repo}:{name}", repo, "basal-2.0", "mini", "GGUF", "GGUF Q8_0", name, 7, False,
                base_repo="Remek/basal-2.0-mini")
    assert delete(v, tmp_path) == 7
    assert (base_snapshot / "config.json").exists()
