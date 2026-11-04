"""What a model derived from a basal model keeps of its source's folder, and which models a derivation that reads with
the plain causal forward refuses: one rule set for basal-finetune (a merged LoRA), basal-fuse-adapter (and the merged
copy --adapter makes for vLLM / SGLang), basal-interpolate and basal-export-gguf --systemone.

- FITTED: what was measured or trained on the source weights' activations and hidden states: the NVFP4 activation
  scales, the evidence head, the early-exit heads and the thinking folder (its LoRA adapter was trained on the source
  weights). A model with other weights never takes them over (basal-serve refuses the first three under --adapter for
  the same reason).
- CALIBRATIONS: fitted to the source's answers. basal-finetune and basal-fuse-adapter drop them (a fused model takes
  the adapter's own, adapter.fused_calibration); basal-interpolate keeps --meta-from's as a starting point
  (docs/INTERPOLATION.md: fit one per merged model).
- DESCRIBES: the source checkpoint's model card, manifest and checksums.
- reading_refused: a model read two-way or with isolated options (basal.json "two_way" / "option_isolation") is served
  by the engine's own Gemma 4 forward only; a tool that reads it with the plain causal forward, one prompt per
  question (fine-tuning, llama.cpp's decision template), would train or describe another reading than the served one.

The single readers these tools share live beside: adapter.read (LoRA files in any floating dtype, bf16 included) and
gemma4_graph.load_nvfp4_amax (NVFP4_AMAX.json, wrapped or plain).
"""
from fnmatch import fnmatch
from pathlib import PurePath

from .prompt import model_option_isolation, model_two_way

# names (fnmatch patterns) of the files and folders at the top of a model folder in each group
FITTED = ("NVFP4_AMAX.json", "evidence_head.pt", "exit_heads", "think")
CALIBRATIONS = ("CALIBRATION*",)
DESCRIBES = ("README.md", "MANIFEST.md", "SHA256SUMS")


def stale(rel, calibrations=True):
    """True for a path of the source folder (relative to it) that a model with other weights does not take over: under
    a FITTED or DESCRIBES name, or a CALIBRATIONS one unless calibrations=False (basal-interpolate keeps those)."""
    top = PurePath(rel).parts[0]
    return any(fnmatch(top, p) for p in FITTED + DESCRIBES + (CALIBRATIONS if calibrations else ()))


def reading_refused(model_dir):
    """Why a model cannot be read with the plain causal forward, one prompt per question (None: it can)."""
    if model_two_way(model_dir):
        return "it is read two-way (basal.json \"two_way\")"
    if model_option_isolation(model_dir):
        return "its options are read in isolated blocks (basal.json \"option_isolation\")"
    return None
