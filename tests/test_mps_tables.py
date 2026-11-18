"""Embedding tables PyTorch MPS reads in part as zeros: no row starting at element 2**31 or beyond is read (torch 2.13;
2.14.1 reads them), and Gemma 4 E4B's per-layer embedding table (262,144 x 10,752) has such rows. On MPS that table
stays in host memory (as with --ple-offload on); any other such table is refused; on every other device, from torch
2.14.1 on and for smaller tables, nothing changes. The limit is moved down for the tiny models here (a real table is
5.6 GB in bf16), and every test but test_torch_that_reads_them runs as on a torch that does not read them."""
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma4ForCausalLM

from basal import engine, finetune
from basal.engine import EagerBackend, mps_host_tables, mps_unreadable
from basal.gemma4 import HostRows

from test_gemma4 import groups_of, text_config

MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


@pytest.fixture(autouse=True)
def torch_that_reads_none(request, monkeypatch):
    if request.node.originalname != "test_torch_that_reads_them":
        monkeypatch.setattr(engine, "MPS_ROWS_READ", "99")


def table(rows, cols):
    """A (Llama) model with one embedding table of rows x cols, without its memory."""
    m = torch.nn.Module()
    m.config = SimpleNamespace(model_type="llama")
    m.emb = torch.nn.Embedding(rows, cols, device="meta")
    return m


@pytest.mark.parametrize("rows,cols,unreadable", [
    (199729, 10752, False),   # E4B's table up to id 199,728, whose row only ends past 2**31: read whole
    (199730, 10752, True),    # id 199,729 starts past 2**31: read as zeros
    (262144, 10752, True),    # E4B
    (262144, 8960, True),     # E2B: from id 239,675 on
    (262144, 8192, False),    # the last row starts at 2**31 - 8192
    (262145, 8192, True),     # the last row starts at 2**31
    (262144, 5376, False),    # Gemma 4 31B's token embeddings
])
def test_rows_from_element_2_31(rows, cols, unreadable):
    assert mps_unreadable(table(rows, cols)) == (["emb"] if unreadable else [])


@pytest.mark.parametrize("version,read", [("2.13.0", False), ("2.14.0", False), ("2.14.1", True), ("2.15.0", True)])
def test_torch_that_reads_them(monkeypatch, version, read):
    monkeypatch.setattr(torch, "__version__", version)
    assert mps_unreadable(table(262144, 10752)) == ([] if read else ["emb"])
    if read:
        assert mps_host_tables(table(262144, 10752), "mps", "x", offload=False) == 0


def tiny():
    """E4B-like: the per-layer table (64 x 6 layers x 16) is the largest table, larger than the token embeddings
    (64 x 64)."""
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(text_config(hidden_size_per_layer_input=16)).eval()
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(30.0)
    return m


PLE_ONLY = 5000  # a limit only the tiny model's per-layer table reaches (63 x 96 = 6,048; its token table: 4,032)


def test_other_devices_unchanged(monkeypatch):
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", 1)
    for dev in ("cpu", "cuda", "cuda:0", torch.device("cpu")):
        m = tiny()
        assert mps_host_tables(m, dev, "x") == 0
        assert type(m.model.embed_tokens_per_layer) is not HostRows
    assert mps_host_tables(table(262144, 10752), "cpu", "x", offload=False) == 0


def test_mps_ple_on_host(monkeypatch):
    m = tiny()
    assert mps_host_tables(m, "mps", "x") == 0  # every table read whole: nothing moves
    assert type(m.model.embed_tokens_per_layer) is not HostRows
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", PLE_ONLY)
    assert mps_unreadable(m) == ["model.embed_tokens_per_layer"]
    emb = m.model.embed_tokens_per_layer
    assert mps_host_tables(m, "mps:0", "x") == emb.weight.numel() * 4
    assert isinstance(m.model.embed_tokens_per_layer, HostRows)
    assert mps_host_tables(m, "mps", "x") == 0  # already on the host


def test_mps_refused(monkeypatch):
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", PLE_ONLY)
    with pytest.raises(ValueError, match="basal-finetune --device mps: .*model.embed_tokens_per_layer"):
        mps_host_tables(tiny(), "mps", "basal-finetune --device mps", offload=False)  # a checkpoint written from it
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", 1000)  # the token embeddings too: they cannot stay on the host
    with pytest.raises(ValueError, match=r"model\.embed_tokens, model\.embed_tokens_per_layer .*--device cpu"):
        mps_host_tables(tiny(), "mps", "x")
    with pytest.raises(ValueError, match="emb"):
        mps_host_tables(table(262145, 8192), "mps", "x")  # no Gemma 4 model


def stub_tokenizer(monkeypatch):
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())


def test_eager_cpu_keeps_its_table(tmp_path, monkeypatch):
    tiny().save_pretrained(tmp_path)
    stub_tokenizer(monkeypatch)
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", 1)
    be = EagerBackend(tmp_path, "float32", device="cpu")
    assert type(be.model.model.embed_tokens_per_layer) is not HostRows


@pytest.mark.skipif(not MPS, reason="needs PyTorch MPS")
@pytest.mark.parametrize("cls", ["EagerBackend", "MPSBackend"])
def test_mps_engines_answer_as_the_cpu(tmp_path, monkeypatch, cls):
    """--mode eager and mps on MPS with the per-layer table over the (moved) limit: it stays on the host, and the
    answers are the CPU's."""
    tiny().save_pretrained(tmp_path)
    stub_tokenizer(monkeypatch)
    groups = groups_of()
    lids = [[[10, 11, 12]] * len(g) for g in groups]
    ref = EagerBackend(tmp_path, "float32", device="cpu").run_shared(list(zip(groups, lids)))
    plain = getattr(engine, cls)(tmp_path, "float32", device="mps")
    assert type(plain.model.model.embed_tokens_per_layer) is not HostRows  # under the limit: as before
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", PLE_ONLY)
    be = getattr(engine, cls)(tmp_path, "float32", device="mps")
    assert isinstance(be.model.model.embed_tokens_per_layer, HostRows)
    assert be.model.model.embed_tokens.weight.device.type == "mps"
    for a, b in zip(be.run_shared(list(zip(groups, lids))), ref):
        for x, y in zip(a, b):
            assert max(abs(p - q) for p, q in zip(x, y)) < 1e-4


def test_eager_ple_offload_on_still_refuses_another_table(tmp_path, monkeypatch):
    """--ple-offload on moves the per-layer table, and every table left is checked as without it."""
    tiny().save_pretrained(tmp_path)
    stub_tokenizer(monkeypatch)
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", 1000)  # the token embeddings too
    with pytest.raises(ValueError, match=r"model\.embed_tokens .*--device cpu") as e:
        EagerBackend(tmp_path, "float32", device="mps", ple_offload=True)
    assert "embed_tokens_per_layer" not in str(e.value)  # on the host by then
    if MPS:  # (the model goes to the device from here on)
        monkeypatch.setattr(engine, "MPS_ROW_LIMIT", PLE_ONLY)  # only the per-layer table: offloaded, nothing to refuse
        be = EagerBackend(tmp_path, "float32", device="mps", ple_offload=True)
        assert isinstance(be.model.model.embed_tokens_per_layer, HostRows)


def test_finetune_refuses_before_writing(tmp_path, monkeypatch, capsys):
    """basal-finetune --device mps of a model with a table MPS reads in part as zeros stops before --out is created."""
    src, out = tmp_path / "model", tmp_path / "out"
    tiny().save_pretrained(src)
    (src / "basal.json").write_text("{}")
    monkeypatch.setattr(engine, "MPS_ROW_LIMIT", PLE_ONLY)
    monkeypatch.setattr(finetune, "refused", lambda _src: None)
    monkeypatch.setattr(finetune, "split", lambda *_a, **_k: (["t"], ["h"], {}))
    monkeypatch.setattr(finetune, "model_format", lambda _src: "x")
    monkeypatch.setattr(finetune, "model_layout", lambda _src: "x")
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=None, get_peft_model=None))  # (not reached)
    args = [str(tmp_path / "labelled.jsonl"), "--model", str(src), "--out", str(out), "--device", "mps"]
    with pytest.raises(SystemExit, match="basal-finetune --device mps: .*model.embed_tokens_per_layer"):
        finetune.main(args)
    assert not out.exists()
    out.mkdir()  # an existing heldout file of an earlier run survives --force too
    (out / "heldout.jsonl").write_text("kept")
    with pytest.raises(SystemExit, match="model.embed_tokens_per_layer"):
        finetune.main(args + ["--force"])
    assert (out / "heldout.jsonl").read_text() == "kept"
