"""basal-export-gguf: BPE merge ranks become llama.cpp token scores, the word-boundary prefix is switched off and the
decision metadata for Ollama is written (on a tiny GGUF; the converter itself is not run)."""
import json

import pytest

gguf = pytest.importorskip("gguf")

from basal.gguf_export import decision_metadata, fix_gguf, merge_rank_scores

TOKENIZER = {"model": {"merges": [["▁", "a"], ["b", "c"], ["▁a", "bc"], ["b", "c"]]},
             "added_tokens": [{"id": 0, "content": "<s>", "special": True}]}


def test_merge_rank_scores_follow_merge_order():
    tokens = ["<s>", "a", "▁a", "bc", "▁abc"]
    assert merge_rank_scores(TOKENIZER, tokens) == ([-5.0, -5.0, -1.0, -2.0, -3.0], 3)  # 4 merges: unranked = -5
    assert merge_rank_scores({"model": {"merges": ["▁ a", "b c"]}}, ["bc", "▁a"]) == ([-2.0, -1.0], 2)  # "a b" form


def test_decision_metadata_from_calibration():
    meta = decision_metadata("llama", {"temperature_per_prim": {"choice": 0.9, "noul": 1.3, "score": 0.9}})
    assert meta["llama.decision.type"] == ("string", "basal")
    assert meta["llama.decision.temperature.noul"] == ("float32", 1.3)


def test_fix_gguf_rewrites_tokenizer_and_adds_decision_metadata(tmp_path):
    raw = tmp_path / "raw.gguf"
    w = gguf.GGUFWriter(str(raw), arch="llama")
    w.add_tokenizer_model("llama")
    w.add_token_list(["<s>", "a", "▁a", "bc", "▁abc"])
    w.add_token_scores([-1000.0] * 5)
    w.add_token_types([1] * 5)  # converter output when tokenizer_config.json lists no special tokens
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    (tmp_path / "tokenizer.json").write_text(json.dumps(TOKENIZER))
    (tmp_path / "CALIBRATION.json").write_text(json.dumps({"temperature_per_prim": {"choice": 0.96, "noul": 1.33,
                                                                                   "score": 0.96}}))
    out = tmp_path / "fixed.gguf"
    plain = tmp_path / "plain.gguf"
    assert fix_gguf(raw, plain, tmp_path) == (3, 5)
    assert "llama.decision.type" not in gguf.GGUFReader(plain).fields  # usable for generation in any Ollama
    assert fix_gguf(raw, out, tmp_path, ollama_decision=True) == (3, 5)
    f = gguf.GGUFReader(out).fields
    assert [f["tokenizer.ggml.scores"].parts[i][0] for i in f["tokenizer.ggml.scores"].data] == [-5, -5, -1, -2, -3]
    assert f["tokenizer.ggml.add_space_prefix"].contents() is False
    assert [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data] == [3, 1, 1, 1, 1]
    assert f["llama.decision.type"].contents() == "basal"
    assert f["llama.decision.temperature.noul"].contents() == pytest.approx(1.33)


def test_cli_rejects_missing_converter_before_download(tmp_path, capsys):
    from basal.gguf_export import main
    with pytest.raises(SystemExit) as e:
        main(["Remek/never-downloaded", str(tmp_path / "x.gguf"), "--llama-cpp", str(tmp_path)])
    assert e.value.code == 2 and "convert_hf_to_gguf.py not found" in capsys.readouterr().err


def test_cli_creates_missing_output_directory(tmp_path):
    from basal.gguf_export import main
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("import sys; sys.exit(3)\n")  # stops right after the temp dir
    out = tmp_path / "new" / "dir" / "model.gguf"
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), str(out), "--llama-cpp", str(llama)])
    assert "converter failed (exit 3)" in str(e.value.code) and out.parent.is_dir()

