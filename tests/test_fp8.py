"""ModelOpt FP8 checkpoints (the -FP8 repositories) load natively: decoder linears become FP8Linear with the
checkpoint's static scales, everything else stays as stored; on CPU the FP8 product is emulated."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from basal.engine import FP8Linear, load_modelopt_fp8, modelopt_fp8
from tests.test_mlx import tiny_llama

QCFG = {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": ["lm_head", "model.embed_tokens"]}


def fp8_checkpoint(path):
    """A tiny Llama (attention and MLP biases) written as ModelOpt writes FP8: per-tensor weight and input scales."""
    m = tiny_llama(path)
    sd = {}
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.startswith("model.layers."):
            w = mod.weight.detach().float()
            ws = w.abs().max() / 448.0
            sd[f"{name}.weight"] = (w / ws).to(torch.float8_e4m3fn)
            sd[f"{name}.weight_scale"] = ws.reshape(())
            sd[f"{name}.input_scale"] = torch.tensor(0.05)
            if mod.bias is not None:
                sd[f"{name}.bias"] = mod.bias.detach().clone()
    for k, v in m.state_dict().items():
        if not any(k.startswith(n) for n in {x.rsplit(".", 1)[0] for x in sd}):
            sd.setdefault(k, v.detach().clone())
    for f in path.glob("*.safetensors"):
        f.unlink()
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path / "model.safetensors"))
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    cfg["quantization_config"] = QCFG
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return m


def test_modelopt_fp8_loads_and_matches(tmp_path):
    ref = fp8_checkpoint(tmp_path)
    assert modelopt_fp8(tmp_path) and not modelopt_fp8(tmp_path / "missing")
    m = load_modelopt_fp8(tmp_path, "cpu", torch.float32)
    lin = [n for n, x in m.named_modules() if isinstance(x, FP8Linear)]
    assert len(lin) == 2 * 7 and not isinstance(m.lm_head, FP8Linear)  # 2 layers x q k v o gate up down
    ids = torch.tensor([list(range(3, 40))])
    with torch.no_grad():
        a = torch.log_softmax(ref(input_ids=ids).logits[0, -1].float(), -1)
        b = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)
    assert (a.exp() - b.exp()).abs().max() < 0.05  # FP8 weights and inputs: close, not equal


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_modelopt_fp8_loads_the_same_tensors_as_a_model_built_on_the_device(tmp_path, dtype):
    """The model is built with its parameters on the meta device and takes the checkpoint's tensors (load_state_dict
    assign): every parameter and buffer as a model built on the device and copied into (the loader before 2.5) has
    it, bit for bit, in the same dtype, and nothing is left on the meta device."""
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from transformers.initialization import no_init_weights  # transformers 5
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    fp8_checkpoint(tmp_path)
    m = load_modelopt_fp8(tmp_path, "cpu", dtype)
    cfg = AutoConfig.from_pretrained(tmp_path)
    del cfg.quantization_config
    with no_init_weights():
        old = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    sd = load_file(str(tmp_path / "model.safetensors"))
    for name, mod in list(old.named_modules()):
        if isinstance(mod, torch.nn.Linear) and f"{name}.weight_scale" in sd:
            parent, child = name.rsplit(".", 1)
            setattr(old.get_submodule(parent), child,
                    FP8Linear(sd.pop(f"{name}.weight"), sd.pop(f"{name}.weight_scale"), sd.pop(f"{name}.input_scale"),
                              sd.pop(f"{name}.bias", None)))
    old.load_state_dict(sd, strict=False)
    a, b = m.state_dict(), old.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].device == b[k].device and torch.equal(a[k], b[k]), k
    for k, v in old.named_buffers():  # buffers made at construction (rotary frequencies), not in the checkpoint
        assert torch.equal(dict(m.named_buffers())[k], v), k
    assert not any(t.is_meta for t in [*m.parameters(), *m.buffers()])


def test_fp8_checkpoint_refuses_other_quantisation(tmp_path, monkeypatch):
    fp8_checkpoint(tmp_path)
    from basal import engine
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_: type("T", (), {"pad_token": 1})())
    monkeypatch.setattr(engine.torch.backends.cuda, "enable_cudnn_sdp", lambda *_: None)
    with pytest.raises(ValueError, match="FP8 checkpoint"):
        engine.GraphBackend(tmp_path, quant="nvfp4", warm=False)
