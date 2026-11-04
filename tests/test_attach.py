"""basal-attach: a chat model never trained on basal decisions served as a decision model (tiny random Qwen 3 and
Gemma 4 models with a character-level tokenizer, CPU): the directory it writes, the readout check per prompt format,
basal-serve answering with it and basal-calibrate fitting its calibration."""
import asyncio
import json
import os
import string

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from basal.attach import PLAIN_TEMPLATE, attach, main
from basal.calibrate import main as calibrate
from basal.prompt import CODES
from basal.server import Server, parser

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tokenizer(two_char_codes=False, template=TEMPLATE):
    """One token per character (two-character option codes split, unless merged)."""
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for c in string.printable + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ":
        vocab.setdefault(c, len(vocab))
    merges = []
    for c in (CODES[26:] if two_char_codes else ()):
        vocab.setdefault(c, len(vocab))
        merges.append((c[0], c[1]))
    t = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
                                   chat_template=template)


def qwen(path, tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True, pad_token_id=0, eos_token_id=1)
    Qwen3ForCausalLM(cfg).eval().save_pretrained(path)
    tok.save_pretrained(path)
    return path


def gemma4(path, tok):
    from tests.test_gemma4 import text_config
    from transformers import Gemma4ForCausalLM
    torch.manual_seed(0)
    cfg = text_config(vocab_size=len(tok), vocab_size_per_layer_input=len(tok), max_position_embeddings=4096,
                      pad_token_id=0, eos_token_id=1)
    Gemma4ForCausalLM(cfg).eval().save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    """On the CPU, and the SDPA kernels as they were: the eager engine turns some off process-wide for Gemma 4."""
    monkeypatch.setenv("BASAL_DEVICE", "cpu")
    b = torch.backends.cuda
    kept = (b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.cudnn_sdp_enabled())
    yield
    for enable, on in zip((b.enable_flash_sdp, b.enable_mem_efficient_sdp, b.enable_cudnn_sdp), kept):
        enable(on)


def ask(model_dir, questions, *extra):
    srv = Server(parser().parse_args(["--mode", "eager", "--model", str(model_dir), *extra]))

    async def go():
        w = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide({"state": "Order 1240 arrived damaged.", "questions": questions})
        finally:
            w.cancel()
    return srv, asyncio.run(go())["answers"]


QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "tech": "Tech",
                                                                                   "sales": "Sales"}},
             "refund": {"type": "noul", "instructions": "Refund?"},
             "tone": {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}}


@pytest.mark.parametrize("build", [qwen, gemma4])
def test_attach_and_serve(tmp_path, build):
    base = build(tmp_path / "base", tokenizer())
    meta = attach(base, tmp_path / "out")
    out = tmp_path / "out"
    assert meta["prompt"] == 1 and meta["layout"] == "canonical" and meta["readout"]["codes"] == "ABCDEFGHIJ"
    assert meta["attached"]["chat_template"] == "the base's" and meta["attached"]["model_type"].startswith(
        "qwen3" if build is qwen else "gemma4")
    assert json.loads((out / "basal.json").read_text()) == meta
    assert os.path.samefile(out / "config.json", base / "config.json") and os.path.samefile(
        out / "model.safetensors", base / "model.safetensors")
    tok = PreTrainedTokenizerFast.from_pretrained(out)
    assert meta["readout"]["ids"] == [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJ"]
    cal = json.loads((out / "CALIBRATION.json").read_text())
    assert cal["prompt"] == 1 and cal["temperature_per_prim"] == {} and cal["thresholds"] == {}
    srv, ans = ask(out, QUESTIONS)
    assert srv.fmt == 1 and srv.cal == cal
    assert ans["team"]["choice"] in ("billing", "tech", "sales") and abs(sum(ans["team"]["probabilities"].values()) - 1) < 1e-5
    assert 0 < ans["refund"]["noul"] < 1 and set(ans["tone"]["legend"]) == {"0", "1", "2"}
    act = {"type": "act", "instructions": "Pay?", "criteria": {"true": "Yes", "false": "No"},
           "costs": {"wrong": 2, "defer": 1}, "max_error": 0.05}
    a = ask(out, {"pay": act})[1]["pay"]
    assert a["deferred"] is True and a["calibration"] == "unvalidated"   # no certificate until basal-calibrate


def test_plain_chat_template_and_refusals(tmp_path):
    base = qwen(tmp_path / "base", tokenizer(template=None))
    meta = attach(base, tmp_path / "out")
    assert meta["attached"]["chat_template"] == "chat_template.jinja (plain)"
    assert (tmp_path / "out" / "chat_template.jinja").read_text() == PLAIN_TEMPLATE
    assert PreTrainedTokenizerFast.from_pretrained(tmp_path / "out").chat_template == PLAIN_TEMPLATE
    with pytest.raises(SystemExit, match="not empty"):
        attach(base, tmp_path / "out")
    with pytest.raises(SystemExit, match="already a basal model"):
        attach(tmp_path / "out", tmp_path / "again")
    with pytest.raises(SystemExit, match=r"prompt format 2: 229 of its 255 option codes are not one token"):
        attach(base, tmp_path / "out", fmt=2, force=True)   # one token per character: "AA" is two


def test_format_2_when_every_code_is_one_token(tmp_path, capsys):
    base = qwen(tmp_path / "base", tokenizer(two_char_codes=True))
    main([str(base), "--out", str(tmp_path / "out"), "--format", "2"])
    assert "prompt format 2, 255 options at most" in capsys.readouterr().err
    meta = json.loads((tmp_path / "out" / "basal.json").read_text())
    assert meta["prompt"] == 2 and len(meta["readout"]["ids"]) == 255 and meta["readout"]["codes"][26] == "AA"
    srv, ans = ask(tmp_path / "out", {"q": {"type": "choice", "instructions": "Which?",
                                            "criteria": {f"k{i}": f"option {i}" for i in range(30)}}})
    assert srv.fmt == 2 and len(ans["q"]["probabilities"]) == 30


def test_calibrate_fits_the_attached_model(tmp_path, capsys):
    base = qwen(tmp_path / "base", tokenizer())
    out = tmp_path / "out"
    attach(base, out)
    with open(tmp_path / "labelled.jsonl", "w") as f:
        for i in range(60):
            q = dict(QUESTIONS[("team", "refund", "tone")[i % 3]])
            q["gold"] = ("billing", "yes", 1)[i % 3]
            f.write(json.dumps({"id": f"r{i}", "state": f"Case {i}: order {1000 + i} arrived late.",
                                "questions": {"q": q}}) + "\n")
    cal_path = tmp_path / "CAL.json"
    calibrate([str(tmp_path / "labelled.jsonl"), "--mode", "eager", "--model", str(out), "--out", str(cal_path),
               "--min-per-type", "5"])
    assert "max_error on CERT" in capsys.readouterr().out
    cal = json.loads(cal_path.read_text())
    assert cal["baseline"].endswith("(basal-attach defaults, uncalibrated: fit with basal-calibrate)")
    assert all(cal["temperature_source"][t].startswith("fitted") for t in ("choice", "noul", "score"))
    srv, _ = ask(out, {"refund": QUESTIONS["refund"]}, "--calibration", str(cal_path))
    assert srv.temps == cal["temperature_per_prim"]


@pytest.mark.parametrize("where", ["same", "alias", "inside", "contains"])
def test_out_that_overlaps_the_base_is_refused_before_any_write(tmp_path, where):
    """--out the base's own directory (also through a symlink), a directory inside it or one containing it: refused
    before anything is written, so the base's files stay as they were (they were replaced by links to themselves)."""
    base = tmp_path / "models" / "base"
    base.mkdir(parents=True)
    files = {"config.json": b'{"model_type": "qwen3"}', "model.safetensors": b"\x00weights\x01"}
    for name, data in files.items():
        (base / name).write_bytes(data)
    out = {"same": base, "inside": base / "attached", "contains": tmp_path / "models"}.get(where)
    if where == "alias":
        out = tmp_path / "alias"
        out.symlink_to(base, target_is_directory=True)
    with pytest.raises(SystemExit, match="base model's"):
        attach(base, out, force=True)
    assert sorted(p.name for p in base.iterdir()) == sorted(files)
    for name, data in files.items():
        assert not (base / name).is_symlink() and (base / name).read_bytes() == data
    assert not (base / "attached").exists()
