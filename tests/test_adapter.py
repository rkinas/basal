"""--adapter: the mlx-lm and PEFT formats read to the same deltas, the merge equals PEFT's merge_and_unload on every
framework (torch, MLX bf16 and 8-bit, the TPU backend's numpy weights), basal-serve's calibration rules under an
adapter, and basal-fuse-adapter's merged checkpoint."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file
from starlette.testclient import TestClient
from transformers import AutoModelForCausalLM

import basal.server as server_module
from basal import adapter as adapter_module
from basal.server import Server, http_app, parser
from tests.test_calibrate import SHIPPED, StubBackend
from tests.test_mlx import mlx_stub, tiny_llama

Q, DOWN = "layers.1.self_attn.q_proj", "layers.1.mlp.down_proj"


def deltas(seed=0, rank=2):
    """Random LoRA factors for tiny_llama's q_proj [64, 64] and down_proj [64, 128]: (a [r, in], b [out, r])."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.normal(0, 0.05, s).astype(np.float32)
    return {Q: (f(rank, 64), f(64, rank)), DOWN: (f(rank, 128), f(64, rank))}


def write_adapter(path, fmt, ds, scale=2.0, **cfg):
    path.mkdir(parents=True, exist_ok=True)
    rank = next(iter(ds.values()))[0].shape[0]
    if fmt == "mlx":
        conf = {"fine_tune_type": "lora", "num_layers": 2,
                "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0,
                                    "keys": sorted({k.split(".", 2)[2] for k in ds})}}
        tensors = {f"model.{k}.lora_{s}": np.ascontiguousarray(x.T) for k, (a, b) in ds.items()
                   for s, x in (("a", a), ("b", b))}
        name = "adapters.safetensors"
    else:
        conf = {"peft_type": "LORA", "r": rank, "lora_alpha": scale * rank, "task_type": "CAUSAL_LM"}
        tensors = {f"base_model.model.model.{k}.lora_{s}.weight": x for k, (a, b) in ds.items()
                   for s, x in (("A", a), ("B", b))}
        name = "adapter_model.safetensors"
    (path / "adapter_config.json").write_text(json.dumps({**conf, **cfg}), encoding="utf-8")
    save_file(tensors, str(path / name))
    return path


def test_both_formats_read_to_the_same_deltas(tmp_path):
    ds = deltas()
    mlx = adapter_module.read(write_adapter(tmp_path / "m", "mlx", ds))
    peft = adapter_module.read(write_adapter(tmp_path / "p", "peft", ds))
    assert mlx.deltas.keys() == peft.deltas.keys() == {Q, DOWN}
    for k in ds:
        for x, y, z in zip(mlx.deltas[k], peft.deltas[k], ds[k]):
            assert np.array_equal(x, z) and np.array_equal(y, z)
    assert (mlx.scale, peft.scale, mlx.rank, mlx.fmt, peft.fmt) == (2.0, 2.0, 2, "mlx", "peft")
    rs = adapter_module.read(write_adapter(tmp_path / "rs", "peft", ds, use_rslora=True))
    assert rs.scale == pytest.approx(4 / math.sqrt(2))
    assert mlx.identity == adapter_module.identity(tmp_path / "m") and mlx.identity.startswith("m@")
    assert len(mlx.identity) == len("m@") + 12 and mlx.identity != adapter_module.identity(tmp_path / "p")
    rescaled = write_adapter(tmp_path / "x" / "m", "mlx", ds, scale=4.0)  # same name and weights, another scale
    assert adapter_module.identity(rescaled) != mlx.identity


def bf16_llama(path, **save):
    """tiny_llama saved in bfloat16, as the basal models are (fuse_checkpoint keeps each tensor's stored dtype)."""
    tiny_llama(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    for f in path.glob("*.safetensors"):
        f.unlink()
    m.save_pretrained(path, **save)
    return m


@pytest.mark.parametrize("shards", [1, 3])
def test_fuse_checkpoint_writes_the_weights_the_load_time_merge_makes(tmp_path, shards):
    md = tmp_path / "model"
    m = bf16_llama(md, **({"max_shard_size": "60KB"} if shards > 1 else {}))
    files = sorted(md.glob("*.safetensors"))
    assert (len(files) > 1) == (shards > 1)
    for name in ("tokenizer_config.json", "basal.json", "CALIBRATION.json", "CALIBRATION.o2.json", "NVFP4_AMAX.json",
                 "SHA256SUMS", "evidence_head.pt"):
        (md / name).write_text("{}", encoding="utf-8")
    (md / "prepared").mkdir()
    ad = adapter_module.read(write_adapter(tmp_path / "edu", "peft", deltas(), use_rslora=True))
    out = tmp_path / "fused"
    assert adapter_module.fuse_checkpoint(md, ad, out) == 2
    adapter_module.merge_torch(m, ad)  # what --adapter merges at load
    fused = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    for k, v in m.state_dict().items():
        assert torch.equal(fused.state_dict()[k], v), k
    # the same tensor names, dtypes and metadata in the same files; a file without the adapter's modules is copied
    from safetensors import safe_open
    for f in files:
        with safe_open(f, "pt") as a, safe_open(out / f.name, "pt") as b:
            assert list(a.keys()) == list(b.keys()) and a.metadata() == b.metadata()
            assert all(a.get_tensor(k).dtype == b.get_tensor(k).dtype for k in a.keys())
            touched = any(k.removesuffix(".weight").endswith((Q, DOWN)) for k in a.keys())
        assert (f.read_bytes() == (out / f.name).read_bytes()) == (not touched)
    kept = sorted(x.name for x in out.iterdir() if x.suffix != ".safetensors")
    assert "basal.json" in kept and "tokenizer_config.json" in kept and "config.json" in kept
    assert not {"CALIBRATION.json", "CALIBRATION.o2.json", "NVFP4_AMAX.json", "SHA256SUMS", "evidence_head.pt",
                "prepared"} & set(kept)
    assert ("model.safetensors.index.json" in kept) == (shards > 1)


def test_fuse_checkpoint_refuses_missing_modules_and_fp8_weights(tmp_path):
    md = tmp_path / "model"
    bf16_llama(md)
    ds = deltas()
    ds["layers.7.self_attn.q_proj"] = ds[Q]
    with pytest.raises(SystemExit, match="layers.7.self_attn.q_proj: no such module"):
        adapter_module.fuse_checkpoint(md, adapter_module.read(write_adapter(tmp_path / "x", "peft", ds)), tmp_path / "o")
    from safetensors.torch import load_file as load_pt
    from safetensors.torch import save_file as save_pt
    w = load_pt(str(md / "model.safetensors"))
    w = {k: (v.to(torch.float8_e4m3fn) if k.endswith("q_proj.weight") else v) for k, v in w.items()}
    save_pt(w, str(md / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(SystemExit, match="cannot take a merge"):
        adapter_module.fuse_checkpoint(md, adapter_module.read(write_adapter(tmp_path / "e", "peft", deltas())),
                                       tmp_path / "o2")


def test_vllm_refuses_an_adapter_for_recurrent_models(tmp_path):
    from basal.engine import VLLMBackend
    md = tmp_path / "pico"
    md.mkdir()
    (md / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text", "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention"]}), encoding="utf-8")
    ad = adapter_module.read(write_adapter(tmp_path / "edu", "peft", deltas()))
    with pytest.raises(SystemExit, match="vllm does not take an adapter for a model with 3 linear_attention layers"):
        VLLMBackend(md, adapter=ad)
    assert not list(tmp_path.glob("basal-adapter-*"))  # refused before any copy


def test_the_merged_copy_needs_room_for_the_weights(tmp_path, monkeypatch):
    from collections import namedtuple
    md = tmp_path / "model"
    bf16_llama(md)
    ad = adapter_module.read(write_adapter(tmp_path / "edu", "peft", deltas()))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(adapter_module.shutil, "disk_usage", lambda p: usage(1 << 40, 0, 1 << 29))  # 0.5 GB free
    with pytest.raises(SystemExit, match=r"edu: vLLM / SGLang serve a merged copy.*GB needed in .*, 0.5 GB free"):
        adapter_module.fused_copy(md, ad)
    monkeypatch.setattr(sys, "argv", ["basal-fuse-adapter", "--model", str(md), "--adapter", str(ad.path),
                                      "--out", str(tmp_path / "new" / "fused")])
    with pytest.raises(SystemExit, match="the merged checkpoint has the size of the model's weights"):
        adapter_module.main()
    assert not (tmp_path / "new").exists()  # refused before anything is written


def test_vllm_and_sglang_serve_a_fused_copy(tmp_path, monkeypatch, capsys):
    from types import ModuleType

    from basal.engine import SGLangBackend, VLLMBackend
    md = tmp_path / "model"
    bf16_llama(md)
    (md / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    ad = adapter_module.read(write_adapter(tmp_path / "edu", "mlx", deltas()))
    seen = {}
    vllm, sglang = ModuleType("vllm"), ModuleType("sglang")
    vllm.LLM = lambda **kw: seen.setdefault("vllm", kw)
    sglang.Engine = lambda **kw: seen.setdefault("sglang", kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda d: type("T", (), {"eos_token": "x"})())
    VLLMBackend(md, adapter=ad)
    SGLangBackend(md, adapter=ad)
    m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16)
    adapter_module.merge_torch(m, ad)
    for engine, kw in seen.items():
        served = kw.get("model") or kw.get("model_path")
        assert Path(served).name.startswith("basal-adapter-") and Path(served) != md, engine
        fused = AutoModelForCausalLM.from_pretrained(served, dtype=torch.bfloat16)
        assert all(torch.equal(fused.state_dict()[k], v) for k, v in m.state_dict().items()), engine
        assert "enable_lora" not in kw, engine
    assert capsys.readouterr().out.count("2 layers merged into a copy of the checkpoint") == 2


def test_framework_prefixes_are_canonical():
    for key in ("model.layers.16.self_attn.q_proj", "base_model.model.model.layers.16.self_attn.q_proj",
                "language_model.model.layers.16.self_attn.q_proj"):
        assert adapter_module.canonical(key) == "layers.16.self_attn.q_proj"
    with pytest.raises(ValueError, match="lm_head"):
        adapter_module.canonical("lm_head")


def test_tensors_outside_the_decoder_layers_are_refused(tmp_path):
    ad = write_adapter(tmp_path / "head", "peft", {"layers.0.self_attn.q_proj": deltas()[Q]})
    t = load_file(str(ad / "adapter_model.safetensors"))
    t = {k.replace("model.layers.0.self_attn.q_proj", "lm_head"): v for k, v in t.items()}
    save_file(t, str(ad / "adapter_model.safetensors"))
    with pytest.raises(SystemExit, match=r"--adapter head: .*lm_head: not a module of the text decoder's layers"):
        adapter_module.read(ad)


def test_what_cannot_be_merged_is_refused_naming_the_directory(tmp_path):
    ds = deltas()
    with pytest.raises(SystemExit, match=r"--adapter .*dora: fine_tune_type dora"):
        adapter_module.read(write_adapter(tmp_path / "dora", "mlx", ds, fine_tune_type="dora"))
    with pytest.raises(SystemExit, match="DoRA"):
        adapter_module.read(write_adapter(tmp_path / "pdora", "peft", ds, use_dora=True))
    write_adapter(tmp_path / "gone", "mlx", ds)
    (tmp_path / "gone" / "adapters.safetensors").unlink()
    with pytest.raises(SystemExit, match=r"--adapter .*gone: no adapters.safetensors"):
        adapter_module.read(tmp_path / "gone")
    half = write_adapter(tmp_path / "half", "peft", ds)
    t = load_file(str(half / "adapter_model.safetensors"))
    del t[f"base_model.model.model.{Q}.lora_B.weight"]
    save_file(t, str(half / "adapter_model.safetensors"))
    with pytest.raises(SystemExit, match=f"{Q}: lora_b is missing"):
        adapter_module.read(half)


def test_torch_merge_adds_the_scaled_delta_in_float32_and_rounds_once(tmp_path):
    tiny_llama(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "peft", deltas()))
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.bfloat16)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert adapter_module.merge_torch(m, ad) == 2
    after = m.state_dict()
    for key in (Q, DOWN):
        a, b = map(torch.from_numpy, ad.deltas[key])
        name = f"model.{key}.weight"
        assert torch.equal(after[name], (before[name].float() + (b @ a) * 2.0).to(torch.bfloat16))
    assert all(torch.equal(after[k], v) for k, v in before.items() if k not in (f"model.{Q}.weight",
                                                                               f"model.{DOWN}.weight"))


def test_torch_merge_finds_the_decoder_of_a_multimodal_wrapper(tmp_path):
    m = tiny_llama(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "peft", deltas()))
    wrapper = torch.nn.Module()  # Gemma 4 style: model.model.language_model.layers
    wrapper.model = torch.nn.Module()
    wrapper.model.language_model = m.model
    w0 = m.model.layers[1].self_attn.q_proj.weight.detach().clone()
    assert adapter_module.merge_torch(wrapper, ad) == 2
    assert not torch.equal(m.model.layers[1].self_attn.q_proj.weight, w0)
    with pytest.raises(SystemExit, match="--adapter ad: the model has no decoder layers"):
        adapter_module.merge_torch(torch.nn.Linear(2, 2), ad)


def test_shape_mismatch_and_unknown_modules_are_refused(tmp_path):
    tiny_llama(tmp_path / "model")
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model")
    rng = np.random.default_rng(1)
    bad = {Q: (rng.normal(size=(2, 32)).astype(np.float32), rng.normal(size=(64, 2)).astype(np.float32))}
    with pytest.raises(SystemExit, match=rf"{Q}: the model's weight is \(64, 64\), the adapter's delta \(64, 32\)"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "bad", "peft", bad)))
    nope = {"layers.1.self_attn.nope": deltas()[Q]}
    with pytest.raises(SystemExit, match="layers.1.self_attn.nope: no such module"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "nope", "peft", nope)))
    deep = {"layers.7.self_attn.q_proj": deltas()[Q]}  # the tiny model has two layers
    with pytest.raises(SystemExit, match="layers.7.self_attn.q_proj: no such module"):
        adapter_module.merge_torch(m, adapter_module.read(write_adapter(tmp_path / "deep", "peft", deep)))


def test_numpy_merge_equals_the_torch_merge(tmp_path):
    tiny_llama(tmp_path / "model")  # float32 weights
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "mlx", deltas()))
    w = load_file(str(tmp_path / "model" / "model.safetensors"))
    assert adapter_module.merge_numpy(w, ad) == 2
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.float32)
    adapter_module.merge_torch(m, ad)
    for key in (Q, DOWN):  # the same CPU delta: identical
        assert np.array_equal(w[f"model.{key}.weight"], m.state_dict()[f"model.{key}.weight"].numpy())


def test_every_framework_and_device_merges_to_the_same_bf16_weights(tmp_path):
    """--adapter on any engine equals the basal-fuse-adapter checkpoint bit for bit: the delta is computed once on the
    CPU, whatever GEMM the device (CUDA, MPS, Metal through MLX) would use."""
    tiny_llama(tmp_path / "f32")
    AutoModelForCausalLM.from_pretrained(tmp_path / "f32", dtype=torch.bfloat16).save_pretrained(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "mlx", deltas()))
    load = lambda: AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.bfloat16)  # noqa: E731
    cpu = load()  # basal-fuse-adapter's merge
    adapter_module.merge_torch(cpu, ad)
    want = {k: cpu.state_dict()[f"model.{k}.weight"] for k in (Q, DOWN)}
    if torch.backends.mps.is_available():  # the eager / mps engines (CUDA: the same code path)
        mps = load().to("mps")
        adapter_module.merge_torch(mps, ad)
        assert all(torch.equal(mps.state_dict()[f"model.{k}.weight"].cpu(), v) for k, v in want.items())
    try:
        import mlx.core as mx
    except ImportError:
        return
    be = mlx_stub(tmp_path / "model")
    adapter_module.merge_mlx(be.model, ad)
    for k, v in want.items():
        mine = torch.from_numpy(np.array(_module(be.model, k).weight.astype(mx.float32)))
        assert torch.equal(mine.to(torch.bfloat16), v), k


def _mlx_fused(path, adapter_dir, quantize=False):
    """mlx-lm's own fuse (python -m mlx_lm fuse) of a fresh load."""
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load_adapters, load_model

    from basal.engine import mlx_config_overrides
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model, _ = load_model(path, model_config=mlx_config_overrides(cfg))
    if quantize:
        nn.quantize(model, group_size=64, bits=8)
    load_adapters(model, str(adapter_dir))
    model.update_modules(tree_unflatten([(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]))
    return model


def _module(model, key):
    _, n, rest = key.split(".", 2)
    mod = model.model.layers[int(n)]
    for attr in rest.split("."):
        mod = getattr(mod, attr)
    return mod


@pytest.mark.parametrize("quantize", [False, True], ids=["full", "8bit"])
def test_mlx_merge_rounds_once_and_stays_within_mlx_lm_fuse(tmp_path, quantize):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    tiny_llama(tmp_path / "model")
    ad_dir = write_adapter(tmp_path / "ad", "mlx", deltas())
    be = mlx_stub(tmp_path / "model")
    if quantize:  # an -MLX-8bit export: dequantised, merged, re-quantised
        nn.quantize(be.model, group_size=64, bits=8)
    w0 = {key: _module(be.model, key).weight for key in (Q, DOWN)}
    ad = adapter_module.read(ad_dir)
    assert adapter_module.merge_mlx(be.model, ad) == 2
    ref = _mlx_fused(tmp_path / "model", ad_dir, quantize)
    for key in (Q, DOWN):
        mine, theirs = _module(be.model, key), _module(ref, key)
        assert type(mine) is type(theirs) is (nn.QuantizedLinear if quantize else nn.Linear)
        d = adapter_module.delta(ad, key).numpy()
        if quantize:  # dequantised, merged with one rounding, re-quantised with the layer's group size and bits
            deq = lambda m: np.array(mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size,  # noqa: E731
                                                   bits=m.bits, mode=m.mode).astype(mx.float32))
            np.testing.assert_allclose(deq(mine), deq(theirs), atol=0.02)  # within a quantisation step of mlx-lm's
        else:  # the formula itself: w + (b @ a) * scale in float32, rounded once (float32 weights here: no rounding)
            assert np.array_equal(np.array(mine.weight), (np.array(w0[key]) + d).astype(np.float32))
            np.testing.assert_allclose(np.array(mine.weight), np.array(theirs.weight), rtol=1e-5, atol=1e-6)


def test_mlx_q8_quantises_the_merged_weights(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    tiny_llama(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "mlx", deltas()))
    q8 = mlx_stub(tmp_path / "model")  # --mode mlx-q8: a lazy load, the merge, then q8
    q8.pool.submit(q8._load, tmp_path / "model", "q8", False, adapter=ad).result()
    ref = mlx_stub(tmp_path / "model")  # the eager load merged, then quantised as MLXBackend's q8
    adapter_module.merge_mlx(ref.model, ad)
    nn.quantize(ref.model, group_size=64, bits=8, class_predicate=lambda path, m: isinstance(m, nn.Linear)
                and path.startswith("model.layers."))
    assert q8.adapter_modules == 2
    for key in (Q, DOWN):
        mine, theirs = _module(q8.model, key), _module(ref.model, key)
        assert isinstance(mine, nn.QuantizedLinear)
        for p in ("weight", "scales", "biases"):
            assert mx.array_equal(getattr(mine, p), getattr(theirs, p))


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A model directory with the shipped CALIBRATION.json behind the stub engine (--mode eager), and an adapter."""
    md = tmp_path / "model"
    md.mkdir()
    (md / "CALIBRATION.json").write_text(json.dumps(SHIPPED), encoding="utf-8")
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: md)
    monkeypatch.setattr(server_module, "EagerBackend", StubBackend)
    return md, write_adapter(tmp_path / "edu", "mlx", deltas())


def serve(md, *args):
    a = parser().parse_args(["--mode", "eager", "--model", str(md), *args])
    return Server(a), a


def test_a_model_given_as_the_current_directory_keeps_a_name(served, monkeypatch):
    """--model . (or ./): the served name is the directory's; Path(".").name is empty."""
    md, _ = served
    monkeypatch.chdir(md)
    for given in (".", "./", str(md) + "/"):
        assert serve(given)[0].name == "model"


def test_the_adapters_calibration_file_is_served_with_its_name(served):
    md, ad = served
    ident = adapter_module.identity(ad)
    own = dict(SHIPPED, source_tag="adapter", fitted_on={"model": "model", "adapter": ident})
    (ad / "CALIBRATION.json").write_text(json.dumps(own), encoding="utf-8")
    srv, _ = serve(md, "--adapter", str(ad))
    assert srv.cal["source_tag"] == "adapter" and srv.cal_unvalidated is None and srv.cal_mismatch is None
    assert srv.name == "model+edu"
    entry = TestClient(http_app(srv, "eager")).get("/v1/models").json()["models"][0]
    assert entry["name"] == "model+edu" and entry["adapter"] == ident
    assert "adapter" not in TestClient(http_app(serve(md)[0], "eager")).get("/v1/models").json()["models"][0]


def test_the_base_models_file_is_unvalidated_under_an_adapter(served):
    md, ad = served
    srv, _ = serve(md, "--adapter", str(ad))
    assert srv.cal["temperature_per_prim"] == SHIPPED["temperature_per_prim"] and "thresholds" not in srv.cal
    assert srv.cal_unvalidated == "CALIBRATION.json belongs to the base model, served with --adapter edu"
    assert serve(md)[0].cal["thresholds"] == SHIPPED["thresholds"]  # without the adapter: the model's own


def test_a_file_fitted_with_another_adapter_warns_or_is_refused(served, tmp_path, capsys):
    md, ad = served
    f = tmp_path / "C.json"
    f.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "model", "adapter": "other@000000000000"})),
                 encoding="utf-8")
    srv, _ = serve(md, "--adapter", str(ad), "--calibration", str(f))
    want = f"it was fitted with adapter other@000000000000, the served adapter is {adapter_module.identity(ad)}"
    assert srv.cal_mismatch == want and "warning: C.json" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--strict-calibration"):
        serve(md, "--adapter", str(ad), "--calibration", str(f), "--strict-calibration")
    fitted = tmp_path / "F.json"  # fitted with the adapter, served without it
    fitted.write_text(json.dumps(dict(SHIPPED, fitted_on={"model": "model", "adapter": "edu@0123456789ab"})),
                      encoding="utf-8")
    assert serve(md, "--calibration", str(fitted))[0].cal_mismatch == ("it was fitted with adapter edu@0123456789ab, "
                                                                       "the served adapter is none")
    (ad / "CALIBRATION.json").write_text(fitted.read_text(encoding="utf-8"), encoding="utf-8")  # an older adapter's
    assert "edu@0123456789ab" in serve(md, "--adapter", str(ad))[0].cal_mismatch


@pytest.mark.parametrize("mode, why", [("gguf", "basal-fuse-adapter"), ("ollama", "basal-fuse-adapter"),
                                       ("llamacpp", "basal-fuse-adapter"), ("fast-exit", "exit heads trained on the base"),
                                       ("nvfp4", "calibrate the fused checkpoint"),
                                       ("int8", "calibrate the fused checkpoint")])
def test_engines_that_cannot_take_an_adapter_refuse_it_before_loading(served, mode, why):
    md, ad = served
    with pytest.raises(SystemExit, match=why):
        serve(md, "--adapter", str(ad), "--mode", mode, "--gguf", "x.gguf")


def test_evidence_is_refused_under_an_adapter_and_the_warning_says_how_to_fit(served, capsys):
    import asyncio

    from basal.prompt import RequestError
    md, ad = served
    srv, _ = serve(md, "--adapter", str(ad))  # the base model's CALIBRATION.json: unvalidated
    assert "basal-calibrate --adapter" in capsys.readouterr().out

    async def decide(body):
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(body)
        finally:
            w.cancel()
    q = {"type": "noul", "criteria": {"true": "Yes ~0.9", "false": "No ~0.1"}, "evidence": True}
    with pytest.raises(RequestError, match="evidence is not available with --adapter"):
        asyncio.run(decide({"state": "x", "questions": {"r": q}}))
    q.pop("evidence")
    assert asyncio.run(decide({"state": "x", "questions": {"r": q}}))["answers"]["r"]["calibration"] == "unvalidated"


def test_fuse_adapter_writes_a_merged_checkpoint(tmp_path, monkeypatch, capsys):
    md = tmp_path / "model"
    m = bf16_llama(md)
    (md / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (md / "CALIBRATION.json").write_text("{}", encoding="utf-8")  # the base model's: not the merged model's
    ad = write_adapter(tmp_path / "edu", "peft", deltas())
    ident = adapter_module.identity(ad)
    fitted = {"model": "Remek/basal", "revision": "0123456789abcdef", "adapter": ident, "data": "cal.jsonl"}
    (ad / "CALIBRATION.json").write_text(json.dumps(dict(SHIPPED, fitted_on=fitted)), encoding="utf-8")
    stale = dict(fitted, adapter="edu@000000000000")  # fitted with an older version of the adapter
    (ad / "CALIBRATION.mlx.json").write_text(json.dumps(dict(SHIPPED, fitted_on=stale)), encoding="utf-8")
    out = tmp_path / "fused"
    monkeypatch.setattr(sys, "argv", ["basal-fuse-adapter", "--model", str(md), "--adapter", str(ad), "--out", str(out)])
    adapter_module.main()
    printed = capsys.readouterr().out
    assert f"fused 2 layers of edu into {out}" in printed and "CALIBRATION.mlx.json was fitted with another" in printed
    adapter_module.merge_torch(m, adapter_module.read(ad))
    fused = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    for k, v in m.state_dict().items():
        assert torch.equal(fused.state_dict()[k], v), k
    assert (out / "tokenizer_config.json").exists() and not (out / "CALIBRATION.mlx.json").exists()
    cal = json.loads((out / "CALIBRATION.json").read_text(encoding="utf-8"))
    assert cal["fitted_on"] == {"model": "fused", "data": "cal.jsonl", "fused_from": f"Remek/basal@0123456789ab+{ident}"}
    assert cal["thresholds"] == SHIPPED["thresholds"]
    # served as its own model, the copied file belongs to it
    assert server_module.calibration_mismatch(cal, str(out), out) is None
    with pytest.raises(SystemExit, match="not empty"):
        adapter_module.main()
    monkeypatch.setattr(sys, "argv", ["basal-fuse-adapter", "--model", str(md), "--adapter", str(ad), "--out",
                                      str(md / "config.json")])
    with pytest.raises(SystemExit, match="not a directory"):
        adapter_module.main()


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def peft_merged(model, path, targets, rank=4, alpha=8):
    """A PEFT rsLoRA of `model` with random factors (PEFT starts lora_B at zero), saved to `path` -> the model with it
    merged by PEFT's merge_and_unload (the reference). `model` itself is left as it was."""
    import copy
    peft = pytest.importorskip("peft")
    pm = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(r=rank, lora_alpha=alpha, use_rslora=True,
                                                                     target_modules=targets))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    pm.save_pretrained(path)
    return pm.merge_and_unload()


def assert_same_weights(mine, ref, atol=0.0):
    """Bit for bit by default: the merge is PEFT's arithmetic on the CPU."""
    ref = ref.state_dict()
    changed = 0
    for k, v in mine.state_dict().items():
        torch.testing.assert_close(v, ref[k], atol=atol, rtol=0.0 if atol == 0 else 1e-5, msg=k)
        changed += k.endswith("proj.weight")
    return changed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_merge_equals_peft_merge_and_unload_on_llama(tmp_path, dtype):
    m = tiny_llama(tmp_path / "model").to(dtype)
    ref = peft_merged(m, tmp_path / "ad", TARGETS)
    ad = adapter_module.read(tmp_path / "ad")
    assert ad.scale == pytest.approx(8 / math.sqrt(4))  # rsLoRA
    assert adapter_module.merge_torch(m, ad) == 2 * len(TARGETS)
    assert_same_weights(m, ref)


@pytest.mark.parametrize("name", ["e4b", "31b"])
def test_merge_equals_peft_merge_and_unload_on_gemma4(tmp_path, name):
    from tests.test_gemma4 import tiny
    m = tiny(name)  # E4B-like (KV-shared layers, per-layer inputs) and 31B-like (keys = values: no v_proj there)
    m.save_pretrained(tmp_path / "model")
    ref = peft_merged(m, tmp_path / "ad", TARGETS)
    ad = adapter_module.read(tmp_path / "ad")
    n = adapter_module.merge_torch(m, ad)
    assert n > 0
    assert_same_weights(m, ref)
    try:  # the same adapter through mlx-lm's Gemma 4 (--mode mlx)
        import mlx.core as mx
        from mlx_lm.utils import load_model
    except ImportError:
        return
    from basal.engine import mlx_config_overrides
    cfg = json.loads((tmp_path / "model" / "config.json").read_text(encoding="utf-8"))
    mm, _ = load_model(tmp_path / "model", model_config=mlx_config_overrides(cfg))
    assert adapter_module.merge_mlx(mm, ad) == n
    ref = ref.state_dict()
    for key in ad.deltas:
        mine = np.array(_module(getattr(mm, "language_model", mm), key).weight.astype(mx.float32))
        np.testing.assert_allclose(mine, ref[f"model.{key}.weight"].numpy(), atol=1e-6, err_msg=key)


def test_multimodal_gemma4_merges_the_text_decoder_only(tmp_path):
    from tests.test_gemma4 import tiny_mm
    m = tiny_mm(True, 2, 8)  # vision and audio towers with their own layers.N.self_attn.q_proj
    text = r".*language_model\.layers\.\d+\.(self_attn|mlp)\.(" + "|".join(TARGETS) + ")"
    ref = peft_merged(m, tmp_path / "ad", text)
    vision = {k: v.clone() for k, v in m.state_dict().items() if "vision_tower" in k or "audio_tower" in k}
    assert adapter_module.merge_torch(m, adapter_module.read(tmp_path / "ad")) > 0
    assert_same_weights(m, ref)
    assert all(torch.equal(m.state_dict()[k], v) for k, v in vision.items())
    # an adapter that also covers the vision tower's layer 0 (its q_proj wraps a Linear): refused, never merged into the
    # decoder's layer 0
    peft_merged(m, tmp_path / "both", r".*(language_model\.layers\.0\.self_attn\.q_proj|vision_tower\.encoder\.layers\.0"
                                      r"\.self_attn\.q_proj\.linear)")
    with pytest.raises(SystemExit, match="vision_tower.*not a module of the text decoder"):
        adapter_module.read(tmp_path / "both")


def test_a_bf16_adapter_reads_as_its_float32_values(tmp_path):
    """PEFT / Unsloth save adapters in bf16: read as their values in float32 (the numpy safetensors reader has no
    bf16)."""
    from safetensors.torch import save_file as save_torch
    ds = deltas()
    p = write_adapter(tmp_path / "p", "peft", ds)
    tensors = {f"base_model.model.model.{k}.lora_{s}.weight": torch.from_numpy(x).to(torch.bfloat16)
               for k, (a, b) in ds.items() for s, x in (("A", a), ("B", b))}
    save_torch(tensors, str(p / "adapter_model.safetensors"))
    ad = adapter_module.read(p)
    for k, (a, b) in ds.items():
        for got, want in zip(ad.deltas[k], (a, b)):
            assert got.dtype == np.float32
            assert np.array_equal(got, torch.from_numpy(want).to(torch.bfloat16).float().numpy())


def test_torch_merge_refuses_integer_weights(tmp_path):
    """Weights that are not floating point (an int8 quantised linear) cannot take a merge: refused, not corrupted."""
    tiny_llama(tmp_path / "model")
    ad = adapter_module.read(write_adapter(tmp_path / "ad", "peft", deltas()))
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.float32)
    q = m.model.layers[1].self_attn.q_proj
    q.weight = torch.nn.Parameter(torch.zeros_like(q.weight, dtype=torch.int8), requires_grad=False)
    with pytest.raises(SystemExit, match="int8 weights cannot take a merge"):
        adapter_module.merge_torch(m, ad)
