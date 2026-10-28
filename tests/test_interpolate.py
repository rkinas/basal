"""basal-interpolate: alpha 0 / 1 reproduce the checkpoints, 0.5 is the fp32 mean stored in bf16, integer tensors come
from the new checkpoint, shards (also nested ones) are kept, mismatched or quantized checkpoints are refused before
anything is written, a tokenizer serialized differently is accepted while one that splits text differently needs
--allow-tokenizer-difference, and the non-weight files come from --meta-from (on tiny checkpoints)."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from basal.interpolate import interpolate, main

VOCAB = {"<unk>": 0, "<s>": 1, "▁": 2, "a": 3, "b": 4, "ab": 5, "▁a": 6, "▁ab": 7}
MERGES = [("a", "b"), ("▁", "a"), ("▁a", "b")]


def tokenizer_json(legacy=False, vocab=VOCAB):
    """A tiny BPE tokenizer: the transformers 5 layout (Metaspace pre-tokenizer), or with legacy=True the older one
    (Prepend / Replace normalizer, as in Bielik-*-v3.0-Instruct), which also marks a word boundary at a leading space."""
    t = Tokenizer(models.BPE(vocab=vocab, merges=MERGES, unk_token="<unk>"))
    if legacy:
        t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    else:
        t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=False)
    t.add_special_tokens(["<s>"])
    return json.loads(t.to_str())


TOKENIZER = tokenizer_json()


def make_ckpt(d, seed, sharded=False, extra=None, shape=(4, 8), config=None, tokenizer=TOKENIZER, prompt=2,
              subdir=""):
    d = Path(d)
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    t = {"a.weight": torch.randn(shape, generator=g).to(torch.bfloat16),
         "b.weight": torch.randn(3, generator=g).to(torch.bfloat16),
         "pos.ids": torch.arange(5, dtype=torch.int64) + seed}
    t.update(extra or {})
    if sharded:
        (d / subdir).mkdir(exist_ok=True)
        s1 = {"a.weight": t["a.weight"]}
        s2 = {k: v for k, v in t.items() if k != "a.weight"}
        f1, f2 = f"{subdir}model-00001-of-00002.safetensors", f"{subdir}model-00002-of-00002.safetensors"
        save_file(s1, d / f1)
        save_file(s2, d / f2)
        wm = {**dict.fromkeys(s1, f1), **dict.fromkeys(s2, f2)}
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    else:
        save_file(t, d / "model.safetensors")
    config = config or {"architectures": ["LlamaForCausalLM"], "hidden_size": 8,
                        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                        "dtype": "bfloat16", "transformers_version": "5.17.0"}
    (d / "config.json").write_text(json.dumps(config))
    (d / "tokenizer.json").write_text(json.dumps(tokenizer))
    (d / "basal.json").write_text(json.dumps({"name": f"m{seed}", "layout": "qfirst", "prompt": prompt}))
    for f in ("README.md", "SHA256SUMS", "evidence_head.pt", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        (d / f).write_text("x")
    (d / "think").mkdir()
    (d / "think" / "x.json").write_text("{}")
    return d, t


def load_dir(d):
    out = {}
    for f in Path(d).rglob("*.safetensors"):
        out.update(load_file(f))
    return out


@pytest.fixture
def pair(tmp_path):
    old, to = make_ckpt(tmp_path / "old", 1, prompt=1)
    new, tn = make_ckpt(tmp_path / "new", 2)
    return old, new, to, tn


def test_alpha0_is_old(pair, tmp_path):
    old, new, to, _ = pair
    r = load_dir(interpolate(old, new, 0.0, tmp_path / "o"))
    for k in ("a.weight", "b.weight"):
        assert torch.equal(r[k], to[k]) and r[k].dtype == torch.bfloat16


def test_alpha1_is_new(pair, tmp_path):
    old, new, _, tn = pair
    r = load_dir(interpolate(old, new, 1.0, tmp_path / "o"))
    for k in tn:
        assert torch.equal(r[k], tn[k])


def test_alpha_half(pair, tmp_path):
    old, new, to, tn = pair
    r = load_dir(interpolate(old, new, 0.5, tmp_path / "o"))
    for k in ("a.weight", "b.weight"):
        assert torch.equal(r[k], ((to[k].float() + tn[k].float()) / 2).to(torch.bfloat16))
    assert torch.equal(r["pos.ids"], tn["pos.ids"])  # not floating point: from the new checkpoint


@pytest.mark.parametrize("subdir", ["", "weights/"])
def test_sharded(tmp_path, subdir):
    old, to = make_ckpt(tmp_path / "old", 1, sharded=True, subdir=subdir)
    new, tn = make_ckpt(tmp_path / "new", 2, sharded=True, subdir=subdir)
    out = interpolate(old, new, 0.5, tmp_path / "o")
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert all((out / f).is_file() for f in index["weight_map"].values())  # the index finds every shard
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*.safetensors")) == [
        f"{subdir}model-00001-of-00002.safetensors", f"{subdir}model-00002-of-00002.safetensors"]
    r = load_dir(out)
    assert torch.equal(r["a.weight"], ((to["a.weight"].float() + tn["a.weight"].float()) / 2).to(torch.bfloat16))
    assert set(r) == set(tn)


FP8 = {"c.weight": torch.zeros(2, dtype=torch.float8_e4m3fn)}


@pytest.mark.parametrize("old_kw, new_kw", [
    ({}, {"config": {"architectures": ["Gemma4"], "hidden_size": 8}}),
    ({}, {"extra": {"c.weight": torch.zeros(2)}}),
    ({}, {"shape": (4, 9)}),
    ({}, {"tokenizer": tokenizer_json(vocab={**VOCAB, "ba": 8})}),
    ({}, {"config": {"architectures": ["LlamaForCausalLM"], "hidden_size": 8,
                     "quantization_config": {"quant_method": "modelopt"}}}),
    ({"extra": FP8}, {"extra": FP8}),
])
def test_mismatch_refused_before_writing(tmp_path, old_kw, new_kw):
    old, _ = make_ckpt(tmp_path / "old", 1, **old_kw)
    new, _ = make_ckpt(tmp_path / "new", 2, **new_kw)
    with pytest.raises(SystemExit):
        interpolate(old, new, 0.5, tmp_path / "o")
    assert not (tmp_path / "o").exists() and not (tmp_path / ".o.partial").exists()


@pytest.mark.parametrize("shard", ["../evil.safetensors", "/tmp/evil.safetensors"])
def test_index_path_outside_model_refused(tmp_path, shard):
    old, _ = make_ckpt(tmp_path / "old", 1, sharded=True)
    new, _ = make_ckpt(tmp_path / "new", 2, sharded=True)
    index = json.loads((new / "model.safetensors.index.json").read_text())
    index["weight_map"]["a.weight"] = shard
    (new / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(SystemExit, match="outside the model folder"):
        interpolate(old, new, 0.5, tmp_path / "o")
    assert not (tmp_path / "evil.safetensors").exists() and not (tmp_path / "o").exists()


def test_missing_tokenizer_refused(tmp_path):
    old, _ = make_ckpt(tmp_path / "old", 1)
    new, _ = make_ckpt(tmp_path / "new", 2)
    (old / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="tokenizer.json missing"):
        interpolate(old, new, 0.5, tmp_path / "o")


def test_nonempty_out_refused(pair, tmp_path):
    old, new, _, _ = pair
    interpolate(old, new, 0.5, tmp_path / "o")
    with pytest.raises(SystemExit, match="not empty"):
        interpolate(old, new, 0.25, tmp_path / "o")
    with pytest.raises(SystemExit, match="not empty"):  # never into a source checkpoint
        interpolate(old, new, 0.25, new)


def test_failed_write_leaves_nothing(pair, tmp_path, monkeypatch):
    import basal.interpolate

    old, new, _, _ = pair

    def boom(path, *a, **k):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(basal.interpolate, "save_streamed", boom)
    with pytest.raises(OSError):
        interpolate(old, new, 0.5, tmp_path / "o")
    assert not (tmp_path / "o").exists() and not (tmp_path / ".o.partial").exists()


def test_other_serialization_accepted(tmp_path):
    """Written by transformers 4.x: rope_theta at the top level, torch_dtype, merges as strings. Same architecture,
    vocabulary and splitting, so the pair is accepted."""
    config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 8, "rope_theta": 1000000, "rope_scaling": None,
              "torch_dtype": "bfloat16", "transformers_version": "4.44.2", "use_cache": True}
    tokenizer = json.loads(json.dumps(TOKENIZER))
    tokenizer["model"]["merges"] = [" ".join(m) for m in tokenizer["model"]["merges"]]
    old, _ = make_ckpt(tmp_path / "old", 1, config=config, tokenizer=tokenizer)
    new, _ = make_ckpt(tmp_path / "new", 2)
    out = interpolate(old, new, 0.5, tmp_path / "o")
    assert json.loads((out / "tokenizer.json").read_text()) == TOKENIZER  # from the new checkpoint


def test_tokenizer_splitting_difference(tmp_path):
    """Same vocabulary, older layout that splits a leading space differently: refused unless allowed."""
    old, _ = make_ckpt(tmp_path / "old", 1, tokenizer=tokenizer_json(legacy=True))
    new, _ = make_ckpt(tmp_path / "new", 2)
    with pytest.raises(SystemExit, match="settings differ .*for example"):
        interpolate(old, new, 0.5, tmp_path / "o")
    out = interpolate(old, new, 0.5, tmp_path / "o", allow_tokenizer_difference=True)
    assert json.loads((out / "tokenizer.json").read_text()) == TOKENIZER


def test_tokenizer_setting_difference_without_probe_effect(tmp_path):
    """A setting that changes no probe text (here the decoder) still needs --allow-tokenizer-difference."""
    tokenizer = json.loads(json.dumps(TOKENIZER))
    tokenizer["decoder"] = {"type": "ByteFallback"}
    old, _ = make_ckpt(tmp_path / "old", 1, tokenizer=tokenizer)
    new, _ = make_ckpt(tmp_path / "new", 2)
    with pytest.raises(SystemExit, match=r"settings differ \(decoder\)"):
        interpolate(old, new, 0.5, tmp_path / "o")


def test_streamed_file_matches_save_file(tmp_path):
    """save_streamed writes the bytes save_file writes: mixed dtypes, a scalar, an empty tensor, a non-ASCII name."""
    from basal.interpolate import save_streamed

    g = torch.Generator().manual_seed(0)
    t = {"w": torch.randn(4, 8, generator=g).to(torch.bfloat16), "pos": torch.arange(5), "żółw": torch.randn(3),
         "s": torch.tensor(2.5, dtype=torch.float16), "e": torch.zeros(0, 3), "m": torch.ones(2, dtype=torch.bool),
         "f8": torch.randn(4, generator=g).to(torch.float8_e4m3fn)}
    names = {torch.bfloat16: "BF16", torch.int64: "I64", torch.float32: "F32", torch.float16: "F16",
             torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3"}
    save_file(t, tmp_path / "ref.safetensors", metadata={"format": "pt"})
    save_streamed(tmp_path / "got.safetensors", {k: (names[v.dtype], tuple(v.shape)) for k, v in t.items()},
                  t.__getitem__, {"format": "pt"})
    assert (tmp_path / "got.safetensors").read_bytes() == (tmp_path / "ref.safetensors").read_bytes()


def test_meta_files_and_basal_json(pair, tmp_path):
    old, new, _, _ = pair
    out = interpolate(old, new, 0.25, tmp_path / "o")
    for f in ("README.md", "SHA256SUMS", "evidence_head.pt", "think/x.json", "pytorch_model.bin",
              "pytorch_model.bin.index.json"):
        assert not (out / f).exists(), f
    assert (out / "config.json").exists() and (out / "tokenizer.json").exists()
    info = json.loads((out / "basal.json").read_text())
    assert (info["layout"], info["prompt"], info["name"]) == ("qfirst", 2, "m2-interp-a0.25")
    assert info["interpolation"]["alpha"] == 0.25
    assert info["interpolation"]["old"] == str(old) and info["interpolation"]["new"] == str(new)


def test_cli_meta_from_old(pair, tmp_path):
    old, new, _, _ = pair
    main([str(old), str(new), "--alpha", "0.5", "1", "--out", str(tmp_path / "m"), "--meta-from", "old"])
    info = json.loads((tmp_path / "m-a0.5" / "basal.json").read_text())
    assert (info["prompt"], info["name"]) == (1, "m1-interp-a0.5")  # format of the old checkpoint
    assert (tmp_path / "m-a1" / "model.safetensors").exists()


@pytest.mark.parametrize("alphas", [["1.5"], ["0.1", "0.1000001"]])
def test_cli_alpha_refused(pair, tmp_path, alphas):
    old, new, _, _ = pair
    with pytest.raises(SystemExit):
        main([str(old), str(new), "--alpha", *alphas, "--out", str(tmp_path / "m")])
    assert not list(tmp_path.glob("m-a*"))
