"""--mode vllm (2.2.1): the vLLM engine arguments of each --vllm-preset, the calibration file each one loads, the
checks made before a model is loaded, and media through vLLM. No GPU and no vLLM needed (vLLM is stubbed)."""
import json
import os
import sys

import pytest

import basal.server as server_module
from basal.calibrate import engine_name
from basal.engine import VLLM_PRESETS, VLLMBackend, vllm_args
from basal.prompt import RequestError
from basal.server import calibration_file, calibration_name, load_calibration, models_payload, parser

BF16 = {"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {"hidden_size": 1152},
        "text_config": {"num_hidden_layers": 60}}
FP8_PORT = {**BF16, "quantization_config": {"quant_method": "modelopt"}}  # a -FP8 checkpoint (ModelOpt)
E4B = {**BF16, "audio_config": {"hidden_size": 1536},
       "text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 18}}
TEXT = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32}


def test_presets():
    assert list(VLLM_PRESETS) == ["default", "deterministic"]   # no load-time FP8 weights, no FP8 KV cache (not gated)
    assert vllm_args(TEXT) == ({}, {})                            # default: as in 2.2
    assert vllm_args(TEXT, "deterministic") == ({"enable_prefix_caching": False, "enable_chunked_prefill": False,
                                                 "attention_backend": "TRITON_ATTN"}, {"VLLM_BATCH_INVARIANT": "1"})
    for name in ("nvfp4", "fp8", "fp8-kv"):
        with pytest.raises(ValueError, match=f"unknown --vllm-preset '{name}'"):
            vllm_args(TEXT, name)


def test_text_only_and_kv_shared_layers():
    assert vllm_args(BF16)[0] == {}                                         # the encoders load (attention backend)
    assert vllm_args(E4B)[0] == {"kv_sharing_fast_prefill": True}                  # E2B / E4B: on by default
    assert vllm_args(E4B, fast_prefill=False)[0] == {}                              # --vllm-fast-prefill off
    assert vllm_args(BF16)[0] == {}                                                 # no KV-shared layers: unaffected
    assert vllm_args(E4B, "deterministic", fast_prefill=True)[0]["kv_sharing_fast_prefill"]


@pytest.fixture
def fake_vllm(monkeypatch):
    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen.update(kw, env=os.environ.get("VLLM_BATCH_INVARIANT"))
    monkeypatch.setitem(sys.modules, "vllm", type(sys)("vllm"))
    sys.modules["vllm"].LLM = LLM
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda *_: None)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    return seen


def test_backend_passes_the_preset_to_vllm(tmp_path, fake_vllm):
    (tmp_path / "config.json").write_text(json.dumps(E4B), encoding="utf-8")
    be = VLLMBackend(tmp_path, max_len=16384, fast_prefill=True)
    assert be.preset == "default" and "quantization" not in fake_vllm and "kv_cache_dtype" not in fake_vllm
    assert fake_vllm["kv_sharing_fast_prefill"] and "language_model_only" not in fake_vllm
    assert fake_vllm["enable_prefix_caching"] and fake_vllm["max_model_len"] == 16384 and fake_vllm["env"] is None
    fake_vllm.clear()
    VLLMBackend(tmp_path, preset="deterministic", fast_prefill=False)
    assert fake_vllm["env"] == "1" and "kv_sharing_fast_prefill" not in fake_vllm and "quantization" not in fake_vllm
    assert fake_vllm["enable_prefix_caching"] is False  # the preset's value wins over the backend's default


def test_bench_mode_with_a_preset(tmp_path, fake_vllm):
    from basal.bench import build
    (tmp_path / "config.json").write_text(json.dumps(TEXT), encoding="utf-8")
    assert build("vllm@deterministic", tmp_path, fmt=2).preset == "deterministic"
    assert fake_vllm["enable_prefix_caching"] is False
    assert fake_vllm["max_logprobs"] == 255


def test_flags():
    a = parser().parse_args([])
    assert a.vllm_preset == "default" and a.vllm_fast_prefill == "auto"   # on for E2B / E4B (gated on mini, 2.3)
    a = parser().parse_args(["--mode", "vllm", "--vllm-preset", "deterministic", "--vllm-fast-prefill", "off"])
    assert a.vllm_preset == "deterministic" and a.vllm_fast_prefill == "off"
    for name in ("nvfp4", "fp8", "fp8-kv"):  # no NVFP4; FP8 weights at load failed, the FP8 KV cache is not gated
        with pytest.raises(SystemExit):
            parser().parse_args(["--mode", "vllm", "--vllm-preset", name])


@pytest.mark.parametrize("mode", ["fast", "eager", "sglang"])
def test_preset_needs_vllm(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--vllm-preset", "deterministic"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--vllm-preset needs --mode vllm"):
        server_module.main()


def test_calibration_file_per_preset(tmp_path):
    assert calibration_name("vllm", None) == "vllm" and calibration_name("vllm", None, "default") == "vllm"
    assert calibration_name("vllm", None, "deterministic") == "vllm.deterministic"
    assert calibration_name("graph", "fp8", "deterministic") == "fp8" and calibration_name("eager", None, "fp8") is None
    (tmp_path / "CALIBRATION.json").write_text("{}", encoding="utf-8")
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.json"
    (tmp_path / "CALIBRATION.vllm.json").write_text("{}", encoding="utf-8")
    # a preset without a file of its own: the vLLM file (a -FP8 checkpoint's own fit, or the bf16 vLLM fit)
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.vllm.json"
    (tmp_path / "CALIBRATION.vllm.deterministic.json").write_text("{}", encoding="utf-8")
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.vllm.deterministic.json"
    assert calibration_file(tmp_path, "vllm") == tmp_path / "CALIBRATION.vllm.json"
    assert calibration_file(tmp_path, "eager", preset="deterministic") == tmp_path / "CALIBRATION.json"
    assert calibration_file(tmp_path, "vllm", path="X.json", preset="deterministic").name == "X.json"  # --calibration wins


def test_preset_file_for_the_served_orders(tmp_path):
    (tmp_path / "CALIBRATION.vllm.deterministic.json").write_text(json.dumps({"orders": 1, "thresholds": {"0.01": {}}}), encoding="utf-8")
    (tmp_path / "CALIBRATION.vllm.deterministic.o2.json").write_text(json.dumps({"orders": 2, "thresholds": {"0.01": {}}}), encoding="utf-8")
    cal, path, why = load_calibration(tmp_path, "vllm", None, None, 2, preset="deterministic")
    assert path.name == "CALIBRATION.vllm.deterministic.o2.json" and why is None and cal["orders"] == 2
    cal, path, why = load_calibration(tmp_path, "vllm", None, None, 1, preset="deterministic")
    assert path.name == "CALIBRATION.vllm.deterministic.json" and why is None


def test_preset_reported_and_named():
    assert models_payload("m", "vllm", {}, vllm_preset="deterministic")["models"][0]["vllm_preset"] == "deterministic"
    assert "vllm_preset" not in models_payload("m", "fast", {})["models"][0]
    assert engine_name("vllm", None, "deterministic") == "vllm.deterministic" and engine_name("vllm", None) == "vllm"
    assert engine_name("graph", None, "deterministic") == "fast"


def test_media_engine_args():
    assert vllm_args(E4B, media=True, fast_prefill=False)[0] == {"limit_mm_per_prompt": {"image": 8, "audio": 8,
                                                                                         "video": 0}}
    assert vllm_args(BF16, media=True)[0] == {"limit_mm_per_prompt": {"image": 8, "audio": 0, "video": 0}}  # ultra
    unified = {**E4B, "architectures": ["Gemma4UnifiedForConditionalGeneration"]}  # basal-omni-1.1-max
    kw, env = vllm_args(unified, media=True, fast_prefill=False)
    assert kw["worker_extension_cls"] == "basal.vllm_unified.UnifiedWorker" and env == {"BASAL_VLLM_UNIFIED": "1"}
    assert vllm_args(unified, fast_prefill=False) == ({}, {})  # text only: vLLM's own path


def test_media_needs_an_encoder(tmp_path, fake_vllm, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(TEXT), encoding="utf-8")
    with pytest.raises(SystemExit, match="no image or audio encoder"):
        VLLMBackend(tmp_path, media=True)
    (tmp_path / "config.json").write_text(json.dumps(E4B), encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(json.dumps({"audio_seq_length": 750, "audio_ms_per_token": 40}), encoding="utf-8")
    be = VLLMBackend(tmp_path, media=True)
    assert be.media_kinds == ("image", "audio") and be.max_audio_s == 30 and fake_vllm["limit_mm_per_prompt"]
    (tmp_path / "config.json").write_text(json.dumps({**E4B, "architectures": ["Gemma4UnifiedForConditionalGeneration"]}), encoding="utf-8")
    import basal.vllm_unified as vu
    calls = []
    monkeypatch.setattr(vu, "patch", lambda: calls.append("patch"))
    monkeypatch.setattr(vu, "check", lambda llm: calls.append(("check", type(llm).__name__)))
    monkeypatch.delenv("BASAL_VLLM_UNIFIED", raising=False)
    fake_vllm.clear()
    VLLMBackend(tmp_path, media=True)  # 1.1-max: patched here, in vLLM's worker, then checked
    assert calls == ["patch", ("check", "LLM")] and os.environ["BASAL_VLLM_UNIFIED"] == "1"
    assert fake_vllm["worker_extension_cls"] == "basal.vllm_unified.UnifiedWorker"


class Out:
    def __init__(self, prompt_token_ids, logprobs):
        Logprob = type("Logprob", (), {})
        lp = {}
        for t, v in logprobs.items():
            lp[t] = Logprob()
            lp[t].logprob = v
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [type("C", (), {"logprobs": [lp]})()]


def media_backend(monkeypatch, generate):
    sps = type(sys)("vllm")
    sps.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", sps)
    be = VLLMBackend.__new__(VLLMBackend)
    be.prefill = "<prefill>"
    template = lambda self, m, **kw: f"[{m[0]['content']}|{kw['enable_thinking']}]"  # noqa: E731
    be.tok = type("Tok", (), {"image_token_id": 7, "audio_token_id": 9, "apply_chat_template": template})()
    be.llm = type("L", (), {"generate": generate})()
    return be


def test_unified_media_refuses_detail_high(monkeypatch):
    calls = []

    def generate(self, reqs, sps, use_tqdm):
        calls.append(reqs)
        return [Out([1, 7, 7, 2], {10: -0.1, 11: -2.4})]
    be = media_backend(monkeypatch, generate)
    be.unified = True  # basal-omni-1.1-max: detail "default" passed the media gate, "high" did not
    msgs = [[{"role": "user", "content": "a"}]]
    with pytest.raises(RequestError, match='detail "high" is not served with --mode vllm'):
        be.run_media(msgs, [[10, 11]], ["IMG"], [], 1120)
    assert not calls
    assert len(be.run_media(msgs, [[10, 11]], ["IMG"], [], 280)) == 1
    assert len(be.run_media(msgs, [[10, 11]], [], [[0.0, 0.1]], 1120)) == 1  # audio only: no image detail


def test_media_prompts_share_one_vllm_call(monkeypatch):
    calls = []

    def generate(self, reqs, sps, use_tqdm):
        calls.append((reqs, sps))
        return [Out([1, 7, 7, 9, 2, 3], {10: -0.1, 11: -2.4}), Out([1, 7, 7, 9, 2, 4, 5], {10: -3.0, 11: -0.05})]
    be = media_backend(monkeypatch, generate)
    msgs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    out = be.run_media(msgs, [[10, 11], [10, 11]], ["IMG"], [[0.0, 0.1]], 1120)
    (reqs, sps), = calls
    assert [r["prompt"] for r in reqs] == ["[a|False]<prefill>", "[b|False]<prefill>"]
    assert reqs[0]["multi_modal_data"] == {"image": ["IMG"], "audio": [([0.0, 0.1], 16000)]}
    assert reqs[0]["mm_processor_kwargs"] == {"max_soft_tokens": 1120}
    assert sps[0]["allowed_token_ids"] == [10, 11] and sps[0]["max_tokens"] == 1
    assert [(n, nm) for _, n, nm in out] == [(6, 3), (7, 3)]  # prompt tokens, media tokens (image + audio)
    assert out[0][0][0] > 0.9 and out[1][0][1] > 0.9
    text_only = media_backend(monkeypatch, generate)
    text_only.run_media(msgs, [[10, 11], [10, 11]], [], [[0.0]], 280)
    assert calls[-1][0][0]["multi_modal_data"] == {"audio": [([0.0], 16000)]}  # no empty image list


def test_media_prompt_over_the_context(monkeypatch):
    def generate(self, reqs, sps, use_tqdm):
        raise ValueError("The decoder prompt (length 20000) is longer than the maximum model length of 16383. ...")
    be = media_backend(monkeypatch, generate)
    with pytest.raises(ValueError, match="the prompt with its media is 20000 tokens, more than the 16383 tokens"):
        be.run_media([[{"role": "user", "content": "a"}]], [[10, 11]], ["IMG"], [], 280, 16383)


@pytest.fixture
def fake_async_vllm(monkeypatch):
    """vLLM's AsyncLLM stubbed: generate() answers a prompt only once `wait` prompts are in the engine at the same
    time, so a backend that sends one call after another never gets an answer (the test then times out)."""
    import asyncio
    import types
    seen = dict(prompts=0, peak=0, wait=1)

    class AsyncEngineArgs:
        def __init__(self, **kw):
            seen["args"] = kw

    class Engine:
        cond = None

        async def generate(self, prompt, sp, request_id):
            if Engine.cond is None:
                Engine.cond = asyncio.Condition()
            async with Engine.cond:
                seen["prompts"] += 1
                seen["peak"] = max(seen["peak"], seen["prompts"])
                Engine.cond.notify_all()
                await Engine.cond.wait_for(lambda: seen["peak"] >= seen["wait"])
            seen["prompts"] -= 1
            seen.setdefault("ids", []).append(request_id)
            yield Out(list(prompt), {sp["allowed_token_ids"][0]: -0.1, sp["allowed_token_ids"][1]: -2.4})

        async def collective_rpc(self, method):
            return [(0, 0)]

    class AsyncLLM:
        @staticmethod
        def from_engine_args(args, usage_context):
            seen["usage"] = usage_context
            return Engine()
    mods = {"vllm": types.ModuleType("vllm"), "vllm.engine": types.ModuleType("vllm.engine"),
            "vllm.engine.arg_utils": types.ModuleType("vllm.engine.arg_utils"),
            "vllm.usage": types.ModuleType("vllm.usage"), "vllm.usage.usage_lib": types.ModuleType("vllm.usage.usage_lib"),
            "vllm.v1": types.ModuleType("vllm.v1"), "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
            "vllm.v1.engine.async_llm": types.ModuleType("vllm.v1.engine.async_llm"),
            "vllm.inputs": types.ModuleType("vllm.inputs")}
    mods["vllm"].LLM = lambda **kw: pytest.fail("--vllm-overlap on must not start LLM")
    mods["vllm"].SamplingParams = lambda **kw: kw
    mods["vllm.engine.arg_utils"].AsyncEngineArgs = AsyncEngineArgs
    mods["vllm.usage.usage_lib"].UsageContext = type("UsageContext", (), {"LLM_CLASS": "LLM_CLASS"})
    mods["vllm.v1.engine.async_llm"].AsyncLLM = AsyncLLM
    mods["vllm.inputs"].TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda *_: None)
    return seen


def test_overlap_runs_calls_together(tmp_path, fake_async_vllm):
    """--vllm-overlap on: the engine arguments of LLM (its defaults: seed 0, no stats log; the LLM_CLASS scheduler
    limits), several engine calls in flight without the lock, and each call gets its own prompts' answers in order."""
    import concurrent.futures as cf
    (tmp_path / "config.json").write_text(json.dumps(E4B), encoding="utf-8")
    be = VLLMBackend(tmp_path, max_len=8192, overlap=True)
    args = fake_async_vllm["args"]
    assert args["seed"] == 0 and args["disable_log_stats"] and args["enable_prefix_caching"]
    assert args["kv_sharing_fast_prefill"] and args["max_model_len"] == 8192 and args["logprobs_mode"] == "processed_logprobs"
    assert fake_async_vllm["usage"] == "LLM_CLASS" and be.inflight == VLLMBackend.OVERLAP_CALLS
    fake_async_vllm["wait"] = 3  # three prompts of two calls must be in the engine at once
    with cf.ThreadPoolExecutor(2) as ex:
        a = ex.submit(be.run_shared, [([[1, 2], [1, 3]], [[10, 11], [12, 13]])])
        b = ex.submit(be.run_shared, [([[4, 5]], [[20, 21]])])
        ra, rb = a.result(timeout=30), b.result(timeout=30)
    assert fake_async_vllm["peak"] >= 3 and len(set(fake_async_vllm["ids"])) == 3
    assert len(ra[0]) == 2 and len(rb[0]) == 1 and all(p[0] > 0.9 for p in ra[0] + rb[0])
    assert be.llm.collective_rpc("basal_unified_flags") == [(0, 0)]  # (the Gemma 4 Unified media check)


def test_overlap_flag():
    assert parser().parse_args([]).vllm_overlap == "off"
    assert parser().parse_args(["--mode", "vllm", "--vllm-overlap", "on"]).vllm_overlap == "on"


def test_unified_audio_counts_as_its_processor():
    from basal.vllm_unified import audio_tokens
    # one soft token per 640 samples (40 ms at 16 kHz), the last one partly padded; the encoder models' count differs
    assert [audio_tokens(n) for n in (1, 640, 641, 16000, 480000)] == [1, 1, 2, 25, 750]
    assert audio_tokens(480001, cap=750) == 750


def test_unified_patch_and_check(monkeypatch):
    import inspect
    import torch
    from types import SimpleNamespace
    import basal.vllm_unified as vu

    class Attn(torch.nn.Module):
        def __init__(self, clamp):
            super().__init__()
            self.mm_prefix_clamp_sliding_window = clamp

    class Model(torch.nn.Module):  # vLLM 0.31: image ranges cleared in the global layer, clamped in the sliding ones
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.language_model = torch.nn.Sequential(Attn(True), Attn(False), Attn(True))
            self._full_attn_layer_idxs = frozenset({1})

        def _clear_mm_prefix_for_full_attn_layers(self):
            pass

    class Info:
        def get_audio_repl(self, *, audio_len, processor=None):
            raise AssertionError("the encoder models' count")

        def get_hf_config(self):
            return SimpleNamespace(boa_token_id=1, eoa_token_id=3)

    details = SimpleNamespace(select_token_id=lambda ids, t: (ids, t))
    mods = {n: type(sys)(n) for n in ("vllm", "vllm.model_executor", "vllm.model_executor.models",
                                      "vllm.model_executor.models.gemma4_unified", "vllm.model_executor.models.gemma4_mm")}
    mods["vllm.model_executor.models.gemma4_unified"].__dict__.update(
        Gemma4UnifiedForConditionalGeneration=Model, Gemma4UnifiedProcessingInfo=Info)
    mods["vllm.model_executor.models.gemma4_mm"].PromptUpdateDetails = details
    mods["vllm.model_executor.models"].__dict__.update(gemma4_unified=mods["vllm.model_executor.models.gemma4_unified"],
                                                       gemma4_mm=mods["vllm.model_executor.models.gemma4_mm"])
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    assert vu.flags(Model(vllm_config=None)) == (1, 2)
    vu.patch()
    vu.patch()  # once
    assert {"vllm_config", "prefix"} <= set(inspect.signature(Model.__init__).parameters)  # how vLLM calls it
    m = Model(vllm_config=None)
    assert vu.flags(m) == (0, 0)  # image blocks in every layer, past the window
    proc = SimpleNamespace(feature_extractor=SimpleNamespace(audio_samples_per_token=640), audio_seq_length=750,
                           audio_token_id=2)
    assert Info().get_audio_repl(audio_len=16001, processor=proc) == ([1] + [2] * 26 + [3], 2)
    worker = type("W", (vu.UnifiedWorker,), {"get_model": lambda self: m})()
    vu.check(SimpleNamespace(collective_rpc=lambda name: [getattr(worker, name)()]))
    with pytest.raises(SystemExit, match="did not take"):
        vu.check(SimpleNamespace(collective_rpc=lambda name: [(1, 2)]))
