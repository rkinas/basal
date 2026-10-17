"""Export a basal checkpoint to GGUF whose tokenizer matches the Hugging Face one.

llama.cpp's converter writes basal's BPE vocabulary as a SentencePiece-style vocabulary with one constant score for
every token and with a word-boundary prefix, so llama.cpp (and Ollama, LM Studio, llama-server with text prompts)
splits text differently from the tokenizer the models were trained with. This tool runs the converter and then fixes
the file as the -GGUF repositories are fixed (docs/PORTS.md, "Tokenizer fix"). With --gguf it fixes an existing GGUF
instead (a stock conversion, a llama-quantize output, an older basal GGUF); the model then only supplies
tokenizer.json, downloaded without the weights:

- tokenizer.ggml.scores: minus the rank of the first BPE merge that creates each token (the first merge scores 0), so
  llama.cpp's greedy highest-score-first merging follows the BPE merge order; the added tokens of tokenizer.json score
  -1000 (the converter's value), every other token (single characters, byte tokens) scores 0;
- tokenizer.ggml.add_space_prefix = false: the Hugging Face tokenizer adds no word-boundary marker after special
  tokens such as <|im_start|>;
- tokenizer.ggml.token_type: every added token of tokenizer.json is CONTROL (special) or USER_DEFINED, as the Hugging
  Face tokenizer matches them whole; the converter reads special tokens from tokenizer_config.json only, and
  basal-1.0-4.5B's and the max models' do not list <|im_start|>, so llama.cpp split it into characters.

Fixing a fixed file changes nothing (the -GGUF files of basal-1.5 and basal-2.0 come out byte-identical). The tokenizer
matches the Hugging Face one for text that starts with a special token (every basal prompt starts with <s>); for text
without one, Hugging Face adds a word-boundary marker at the very start and llama.cpp does not.

    basal-export-gguf Remek/basal-2.0-4.5B basal-2.0-4.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp
    basal-export-gguf Remek/basal-1.0-1.5B fixed/basal-1.0-1.5B-Q8_0.gguf --gguf basal-1.0-1.5B-Q8_0.gguf
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ADDED_TOKEN_SCORE = -1000.0  # llama.cpp's converter scores the added tokens so; the -GGUF files keep it


def merge_rank_scores(tokenizer_json, tokens):
    """Token scores as in the -GGUF files: -(rank of the first BPE merge producing the token); -1000 for the added
    tokens of tokenizer.json, whatever the input file had; 0 for the rest (single characters, byte tokens: llama.cpp
    only compares the scores of merge results, and basal's vocabulary has no other multi-character token). Returns
    (scores, tokens ranked)."""
    rank = {}
    for i, m in enumerate(tokenizer_json["model"]["merges"]):
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        rank.setdefault(a + b, i)
    added = {t["id"] for t in tokenizer_json.get("added_tokens", [])}
    return ([ADDED_TOKEN_SCORE if i in added else -float(rank[t]) if t in rank else 0.0 for i, t in enumerate(tokens)],
            sum(t in rank for t in tokens))


def added_token_types(tokenizer_json, token_types):
    """GGUF token types with the added tokens of tokenizer.json marked CONTROL (special) or USER_DEFINED."""
    import gguf

    types = list(token_types)
    for t in tokenizer_json.get("added_tokens", []):
        if 0 <= t["id"] < len(types):
            types[t["id"]] = int(gguf.TokenType.CONTROL if t.get("special") else gguf.TokenType.USER_DEFINED)
    return types


def fix_gguf(raw, outfile, model_dir):
    """Copy `raw` to `outfile` with the three tokenizer fields fixed from `model_dir`/tokenizer.json. Any
    <arch>.decision.* keys of an earlier draft of this tool are dropped. Returns (tokens ranked, tokens)."""
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    reader = gguf.GGUFReader(raw)
    tokens = reader.fields[gguf.Keys.Tokenizer.LIST].contents()
    tokenizer_json = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    if any(i >= len(tokens) or tokens[i] != t for t, i in tokenizer_json["model"]["vocab"].items()):
        raise ValueError(f"the vocabulary of {raw} differs from {Path(model_dir) / 'tokenizer.json'}")
    scores, ranked = merge_rank_scores(tokenizer_json, tokens)
    token_types = added_token_types(tokenizer_json, reader.fields[gguf.Keys.Tokenizer.TOKEN_TYPE].contents())
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    new = {gguf.Keys.Tokenizer.ADD_PREFIX: MetadataDetails(gguf.GGUFValueType.BOOL, False),
           gguf.Keys.Tokenizer.SCORES: MetadataDetails(gguf.GGUFValueType.ARRAY, scores,
                                                       sub_type=gguf.GGUFValueType.FLOAT32),
           gguf.Keys.Tokenizer.TOKEN_TYPE: MetadataDetails(gguf.GGUFValueType.ARRAY, token_types,
                                                           sub_type=gguf.GGUFValueType.INT32)}
    stale = [k for k in reader.fields if k.startswith(f"{arch}.decision.")]
    writer = gguf.GGUFWriter(outfile, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new, stale)
    return ranked, len(tokens)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # no docstring under python -OO
    ap.add_argument("model", help="local directory or Hugging Face repo id (with --gguf: only its tokenizer.json is "
                                  "read)")
    ap.add_argument("outfile")
    ap.add_argument("--gguf", default=None, help="fix this GGUF file instead of converting the model (a stock "
                                                "conversion, a llama-quantize output, an older basal GGUF)")
    ap.add_argument("--outtype", default="f16", help="converter output type: f16, q8_0, bf16, f32")
    ap.add_argument("--llama-cpp", default=None, help="llama.cpp checkout containing convert_hf_to_gguf.py "
                                                      "(not needed with --gguf)")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args(argv)
    out = Path(a.outfile)
    if out.is_dir():
        ap.error(f"{out} is a directory; give the name of the output file")
    if a.gguf:
        if not Path(a.gguf).is_file():
            ap.error(f"{a.gguf} not found")
        if out.exists() and out.samefile(a.gguf):
            ap.error("the output must be a new file: the input is read while the output is written")
    else:
        converter = Path(a.llama_cpp or "").expanduser() / "convert_hf_to_gguf.py"
        if not a.llama_cpp or not converter.is_file():
            ap.error(f"{converter} not found; clone https://github.com/ggml-org/llama.cpp and pass it as --llama-cpp "
                     "(or fix an existing file with --gguf)")
    try:
        import gguf  # noqa: F401
    except ImportError:
        ap.error('the gguf package is missing; install the gguf-export extra (from a clone: uv pip install -e '
                 '".[gguf-export]") or uv pip install "gguf>=0.19"')
    from huggingface_hub import snapshot_download

    model_dir = Path(a.model) if Path(a.model).exists() else Path(snapshot_download(
        a.model, revision=a.revision, allow_patterns=["tokenizer.json"] if a.gguf else None))
    if not (model_dir / "tokenizer.json").is_file():
        ap.error(f"{a.model} has no tokenizer.json; pass the basal model the GGUF was converted from")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw = Path(a.gguf or Path(tmp) / "converted.gguf")
        if not a.gguf:
            done = subprocess.run([sys.executable, str(converter), str(model_dir), "--outtype", a.outtype,
                                   "--outfile", str(raw)])
            if done.returncode != 0:
                sys.exit(f"basal-export-gguf: llama.cpp's converter failed (exit {done.returncode}); "
                         "see its output above")
        try:
            ranked, total = fix_gguf(raw, out, model_dir)
        except ValueError as e:
            sys.exit(f"basal-export-gguf: {e}")
    print(f"wrote {out}: {ranked}/{total} token scores from BPE merges, add_space_prefix=false")


if __name__ == "__main__":
    main()
