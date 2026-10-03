"""Export a basal checkpoint to GGUF whose tokenizer matches the Hugging Face one, with decision metadata for Ollama.

llama.cpp's converter writes basal's BPE vocabulary as a SentencePiece-style vocabulary with one constant score for
every token and with a word-boundary prefix, so llama.cpp (and Ollama, LM Studio, llama-server with text prompts)
splits text differently from the tokenizer the models were trained with. This tool runs the converter and then fixes
the file:

- tokenizer.ggml.scores: -(1 + rank of the BPE merge that creates each token), and below every merge for tokens no
  merge creates, so llama.cpp's greedy highest-score-first merging follows the BPE merge order;
- tokenizer.ggml.add_space_prefix = false: the Hugging Face tokenizer adds no word-boundary marker after special
  tokens such as <|im_start|>;
- tokenizer.ggml.token_type: every added token of tokenizer.json is CONTROL (special) or USER_DEFINED, as the Hugging
  Face tokenizer matches them whole; the converter reads special tokens from tokenizer_config.json only, and
  basal-1.0-4.5B's does not list <|im_start|> and <|im_end|>, so llama.cpp split them into characters;
- with --ollama-decision: <arch>.decision.type = "basal" and <arch>.decision.temperature.{choice,noul,score} from
  CALIBRATION.json. Ollama's /v1/systemone then renders basal's own prompt format, scores both option orders and
  calibrates. Such a file is decision-only in Ollama (/api/generate and /api/chat refuse it) and needs an Ollama with
  the basal decision encoding; other Ollama versions reject it.

The tokenizer matches the Hugging Face one for text that starts with a special token (every basal prompt starts with
<s>); for text without one, Hugging Face adds a word-boundary marker at the very start and llama.cpp does not.

    basal-export-gguf Remek/basal-1.0-1.5B basal-1.0-1.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp --ollama-decision
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DECISION_TYPES = ("choice", "noul", "score")


def merge_rank_scores(tokenizer_json, tokens):
    """Score of each GGUF token: -(1 + rank of the first BPE merge producing it). Tokens no merge produces (single
    characters, byte tokens, special tokens) get -(1 + number of merges), ordered after every merge (basal's vocabulary
    has no multi-character text token that no merge produces)."""
    merges = tokenizer_json["model"]["merges"]
    rank = {}
    for i, m in enumerate(merges):
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        rank.setdefault(a + b, i)
    return [-1.0 - rank.get(t, len(merges)) for t in tokens], sum(t in rank for t in tokens)


def added_token_types(tokenizer_json, token_types):
    """GGUF token types with the added tokens of tokenizer.json marked CONTROL (special) or USER_DEFINED."""
    import gguf

    types = list(token_types)
    for t in tokenizer_json.get("added_tokens", []):
        if 0 <= t["id"] < len(types):
            types[t["id"]] = int(gguf.TokenType.CONTROL if t.get("special") else gguf.TokenType.USER_DEFINED)
    return types


def decision_metadata(arch, calibration):
    """{key: (type, value)} for the decision encoding and the per-type temperatures of CALIBRATION.json."""
    temps = calibration["temperature_per_prim"]
    meta = {f"{arch}.decision.type": ("string", "basal")}
    for typ in DECISION_TYPES:
        meta[f"{arch}.decision.temperature.{typ}"] = ("float32", float(temps[typ]))
    return meta


def fix_gguf(raw, outfile, model_dir, ollama_decision=False):
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    model_dir = Path(model_dir)
    reader = gguf.GGUFReader(raw)
    field = reader.fields[gguf.Keys.Tokenizer.LIST]
    tokens = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    tokenizer_json = json.loads((model_dir / "tokenizer.json").read_text())
    scores, ranked = merge_rank_scores(tokenizer_json, tokens)
    type_field = reader.fields[gguf.Keys.Tokenizer.TOKEN_TYPE]
    token_types = added_token_types(tokenizer_json, [int(type_field.parts[i][0]) for i in type_field.data])
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    types = {"string": gguf.GGUFValueType.STRING, "float32": gguf.GGUFValueType.FLOAT32}
    new = {gguf.Keys.Tokenizer.ADD_PREFIX: MetadataDetails(gguf.GGUFValueType.BOOL, False),
           gguf.Keys.Tokenizer.SCORES: MetadataDetails(gguf.GGUFValueType.ARRAY, scores,
                                                       sub_type=gguf.GGUFValueType.FLOAT32),
           gguf.Keys.Tokenizer.TOKEN_TYPE: MetadataDetails(gguf.GGUFValueType.ARRAY, token_types,
                                                           sub_type=gguf.GGUFValueType.INT32)}
    calibration = model_dir / "CALIBRATION.json"
    if ollama_decision:
        for key, (typ, value) in decision_metadata(arch, json.loads(calibration.read_text())).items():
            new[key] = MetadataDetails(types[typ], value)
    writer = gguf.GGUFWriter(outfile, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new, [])
    return ranked, len(tokens)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="local directory or Hugging Face repo id")
    ap.add_argument("outfile")
    ap.add_argument("--outtype", default="f16", help="converter output type: f16, q8_0, bf16, f32")
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout containing convert_hf_to_gguf.py")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ollama-decision", action="store_true",
                    help="write decision metadata for Ollama's /v1/systemone (decision-only file; needs an Ollama with "
                         "the basal decision encoding)")
    a = ap.parse_args(argv)
    converter = Path(a.llama_cpp).expanduser() / "convert_hf_to_gguf.py"
    if not converter.is_file():
        ap.error(f"{converter} not found; clone https://github.com/ggml-org/llama.cpp and pass it as --llama-cpp")
    try:
        import gguf  # noqa: F401
    except ImportError:
        ap.error('the gguf package is missing; install it with: uv pip install -e ".[gguf-export]"')
    from huggingface_hub import snapshot_download

    model_dir = Path(a.model) if Path(a.model).exists() else Path(snapshot_download(a.model, revision=a.revision))
    out = Path(a.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw = Path(tmp) / "converted.gguf"
        done = subprocess.run([sys.executable, str(converter), str(model_dir), "--outtype", a.outtype, "--outfile", str(raw)])
        if done.returncode != 0:
            sys.exit(f"basal-export-gguf: llama.cpp's converter failed (exit {done.returncode}); see its output above")
        ranked, total = fix_gguf(raw, out, model_dir, a.ollama_decision)
    extra = ", decision.type=basal" if a.ollama_decision else ""
    print(f"wrote {out}: {ranked}/{total} token scores from BPE merges, add_space_prefix=false{extra}")


if __name__ == "__main__":
    main()
