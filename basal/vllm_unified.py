"""Images and audio of Gemma 4 Unified models (basal-omni-1.1-max) through vLLM (`--mode vllm --media on`).

vLLM 0.31 serves Gemma4UnifiedForConditionalGeneration with two rules of the Gemma 4 models with encoders (31B) that
the unified models do not follow (transformers' Gemma4UnifiedModel and `--mode eager`):

- **Image attention.** The soft tokens of one image attend to each other in every layer, OR-ed after the sliding
  window (create_masks_for_generate). vLLM keeps that block in the sliding-window layers only, clamped to the window,
  and makes the global layers causal: through vLLM 1.1-max's images got the eager answer on 94.2% only.
- **Audio token counts.** The processor gives one soft token per audio_samples_per_token (640) samples:
  ceil(samples / 640). vLLM counts the mel frames and the two stride-2 convolutions of the encoder models, which
  differs on some clip lengths (19 of 285 clips).

patch() changes both in the process that runs it, once: the prompt processor (the process that calls vLLM) and the
model. The model is built in vLLM's worker, which imports this module as its worker extension (WORKER_EXTENSION,
vLLM's worker_extension_cls) before it builds the model and patches on import when ENV is set (vllm_args sets both);
check() reads the flags of the loaded model back, so a server whose worker missed the patch refuses to start instead of
reading images the encoder models' way.
"""
import math
import os

WORKER_EXTENSION = "basal.vllm_unified.UnifiedWorker"
ENV = "BASAL_VLLM_UNIFIED"
ARCH = "Gemma4UnifiedForConditionalGeneration"


def is_unified(cfg):
    """A Gemma 4 Unified model by its config.json."""
    return ARCH in (cfg.get("architectures") or ())


def audio_tokens(samples, samples_per_token=640, cap=None):
    """Soft tokens of an audio clip of `samples` samples, as the unified processor counts them (ceil), at most cap."""
    n = math.ceil(samples / samples_per_token)
    return n if cap is None else min(n, cap)


def _unify(model):
    """A built Gemma4UnifiedForConditionalGeneration: image blocks in the global layers too, past the window."""
    model._full_attn_layer_idxs = frozenset()  # vLLM clears the image ranges of these layers at every forward
    for m in model.language_model.modules():
        if hasattr(m, "mm_prefix_clamp_sliding_window"):
            m.mm_prefix_clamp_sliding_window = False  # (causal AND window) OR image block


def patch():
    """Patch vLLM's Gemma4 Unified model and prompt processor in this process (idempotent)."""
    try:
        from vllm.model_executor.models import gemma4_unified as gu
        from vllm.model_executor.models.gemma4_mm import PromptUpdateDetails
    except ImportError as e:
        raise SystemExit(f"--media on with --mode vllm for Gemma 4 Unified models needs vLLM's gemma4_unified ({e})")
    model, info = gu.Gemma4UnifiedForConditionalGeneration, gu.Gemma4UnifiedProcessingInfo
    if getattr(model, "_basal_unified", False):
        return
    if not hasattr(model, "_clear_mm_prefix_for_full_attn_layers") or not hasattr(info, "get_audio_repl"):
        raise SystemExit("--media on with --mode vllm for Gemma 4 Unified models: this vLLM's Gemma 4 code is not the "
                         "one checked (vLLM 0.31); use --mode eager --media on")
    init = model.__init__

    def __init__(self, *, vllm_config, prefix=""):  # vLLM builds a model by these parameter names (inspect.signature)
        init(self, vllm_config=vllm_config, prefix=prefix)
        _unify(self)

    def get_audio_repl(self, *, audio_len, processor=None):
        processor = processor or self.get_hf_processor()
        cfg = self.get_hf_config()
        n = audio_tokens(audio_len, getattr(processor.feature_extractor, "audio_samples_per_token", 640),
                         processor.audio_seq_length)
        eoa = cfg.eoa_token_id if hasattr(cfg, "eoa_token_id") else cfg.eoa_token_index
        ids = [cfg.boa_token_id] + [processor.audio_token_id] * n + [eoa]
        return PromptUpdateDetails.select_token_id(ids, processor.audio_token_id)

    model.__init__, info.get_audio_repl = __init__, get_audio_repl
    model._basal_unified = True


def flags(model):
    """(global layers whose image ranges vLLM clears, attention layers that clamp an image to the window)."""
    clamped = sum(bool(getattr(m, "mm_prefix_clamp_sliding_window", False)) for m in model.language_model.modules())
    return len(model._full_attn_layer_idxs), clamped


def check(llm):
    """Refuse a server whose worker built the model without the patch (SystemExit)."""
    got = llm.collective_rpc("basal_unified_flags")
    if any(tuple(f) != (0, 0) for f in got):
        raise SystemExit(f"--media on with --mode vllm: vLLM's worker did not take the Gemma 4 Unified attention rule "
                         f"(global layers cleared, layers clamped: {got}); use --mode eager --media on")


class UnifiedWorker:
    """vLLM worker extension: importing this module in the worker patched it; basal_unified_flags reports the model."""

    def basal_unified_flags(self):
        return flags(self.get_model())


if os.environ.get(ENV) == "1":  # vLLM's worker imports this module by WORKER_EXTENSION before it builds the model
    patch()
