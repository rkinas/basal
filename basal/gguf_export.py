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

With --systemone the file also becomes a decision model for llama.cpp's /v1/systemone (llama-server b11908 or newer,
LM Studio 0.4.26 or newer, whose /v1/decisions reads it too); the tensors are not touched. llama.cpp knows a fixed list
of decision encodings; basal prompt format 2 is the "openjev" one up to 26 options: one prompt per question, the
logits of the option codes A-Z at the last prompt token, a softmax divided by a temperature. The file gets
<arch>.decision.type = "openjev", tokenizer.chat_template.systemone (basal's prompt as a template: the system prompt,
state, question and coded options in the model's chat template, plus the answer prefill; Polish or English as
basal-serve picks) and <arch>.decision.temperature.<type>.<bucket> from the model's calibration
(CALIBRATION.llamacpp.json, else CALIBRATION.gguf.json, else CALIBRATION.json; or --calibration). The template is
checked against basal-serve's own prompts on a fixed set of requests before anything is written. llama.cpp applies
only these temperatures: thresholds, coverage and the engine-level types (act, multi, tool) stay with basal-serve.

    basal-export-gguf Remek/basal-2.0-4.5B basal-2.0-4.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp
    basal-export-gguf Remek/basal-1.0-1.5B fixed/basal-1.0-1.5B-Q8_0.gguf --gguf basal-1.0-1.5B-Q8_0.gguf
    basal-export-gguf Remek/basal-2.0-4.5B-GGUF lmstudio/basal-2.0-4.5B-Q8_0.gguf --gguf basal-2.0-4.5B-Q8_0.gguf \\
        --systemone
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ADDED_TOKEN_SCORE = -1000.0  # llama.cpp's converter scores the added tokens so; the -GGUF files keep it

SYSTEMONE_TYPE = "openjev"  # llama.cpp's decision encoding that reads basal's prompt format 2 (codes A-Z)
SYSTEMONE_TEMPLATE = "tokenizer.chat_template.systemone"
SYSTEMONE_MAX_OPTIONS = 26  # openjev's labels after Z are a-z, basal's codes are AA, AB, ...: the template refuses more
# llama.cpp's temperature buckets by option count (server-decision.cpp) and the counts of each question type
LLAMA_BUCKETS = (("2", 2, 2), ("3_5", 3, 5), ("6_10", 6, 10), ("11", 11, SYSTEMONE_MAX_OPTIONS))
SYSTEMONE_COUNTS = {"choice": (2, SYSTEMONE_MAX_OPTIONS), "noul": (2, 2), "score": (2, 10)}
SYSTEMONE_CALIBRATIONS = ("CALIBRATION.llamacpp.json", "CALIBRATION.gguf.json", "CALIBRATION.json")
MODEL_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
               "basal.json", "CALIBRATION*.json"]  # --gguf --systemone: what the model folder supplies


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


def _lit(text):
    """`text` as a Jinja string literal (llama.cpp's template lexer reads the escapes used here)."""
    if any(ord(c) < 32 and c not in "\n\t" for c in text):
        raise ValueError(f"a control character in the chat template's text: {text!r}")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'


STRING_T = '(t ~ "")'  # the option as a string, whatever the label or description was


def _fang(var, specials):
    """Jinja expression for `var` as basal-serve shows it after prompt.defang (each special token broken by a
    zero-width space after its first character)."""
    from .prompt import _ZWSP

    for n in specials:
        var = f"{var}.replace({_lit(n)}, {_lit(n[0] + _ZWSP + n[1:])})"
    return var


def chat_frame(tok):
    """The model's chat template around one system and one user message, with the generation prompt as basal renders
    it (no thinking): (before the system text, between the two, after the user text, whether the template trims the
    user text)."""
    sys_mark, user_mark = "\ue000basal-system\ue000", "\ue000basal-user\ue000"
    padded = f"\t{user_mark} \n"
    text = tok.apply_chat_template([{"role": "system", "content": sys_mark}, {"role": "user", "content": padded}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.count(sys_mark) != 1 or text.count(user_mark) != 1 or text.index(sys_mark) > text.index(user_mark):
        raise ValueError("the chat template does not place a system and a user message once each")
    trims = padded not in text
    head, rest = text.split(sys_mark)
    mid, tail = rest.split(user_mark if trims else padded)
    if trims and (mid.endswith(padded[0]) or tail.startswith(padded[-2:])):
        raise ValueError("the chat template changes the whitespace around a message in a way basal cannot follow")
    return head, mid, tail, trims


def systemone_template(head, mid, tail, trims=False, specials=()):
    """basal prompt format 2 (canonical layout, one option order) as a template for llama.cpp's /v1/systemone. Its
    variables: state, instructions (text or JSON), type, options [{key, description}] in the order llama.cpp reads the
    labels A, B, ... (a choice in request order, noul true then false, score levels low to high). `trims`: the chat
    template strips the user text (llama.cpp's trim strips ASCII whitespace only, Python's strip all of it).
    `specials`: the special tokens of the tokenizer (prompt.special_names), broken in the state, the question and the
    options as prompt.defang does."""
    from .prompt import CODES, PL_CHARS, PREFILL, TEMPLATES_2

    def turn(lang):
        parts = re.split(r"\{(state|q|opts)\}", TEMPLATES_2[lang]["user"])
        names = {"state": "s", "q": "q", "opts": "ns.opts"}
        user = " ~ ".join(names[p] if i % 2 else _lit(p) for i, p in enumerate(parts))
        return " ~ ".join([_lit(head + TEMPLATES_2[lang]["system"] + mid), f"({user}) | trim" if trims else user,
                           _lit(tail + PREFILL)])

    def text(v):
        return f"({v} if {v} is string else {v} | tojson)"

    pl, codes = ", ".join(map(_lit, sorted(PL_CHARS))), ", ".join(map(_lit, CODES[:SYSTEMONE_MAX_OPTIONS]))
    return "\n".join([
        "{#- basal prompt format 2 for llama.cpp's /v1/systemone, written by basal-export-gguf -#}",
        "{%- if options | length < 2 or options | length > " + str(SYSTEMONE_MAX_OPTIONS) + " -%}",
        "{{- raise_exception(" + _lit(f"basal: a question has 2 to {SYSTEMONE_MAX_OPTIONS} options here") + ") -}}",
        "{%- endif -%}",
        "{%- set ns = namespace(pl=false, opts=\"\") -%}",
        f"{{%- set s = {text('state')} -%}}",
        f"{{%- set q = {text('instructions')} -%}}",
        # a request text cannot write a special token of the tokenizer (prompt.defang); Polish letters are not changed
        f"{{%- set s = {_fang('s', specials)} -%}}",
        f"{{%- set q = {_fang('q', specials)} -%}}",
        # Polish when the state has a Polish letter, else when the question has one (basal-serve's rule)
        f"{{%- for c in [{pl}] -%}}{{%- if c in s -%}}{{%- set ns.pl = true -%}}{{%- endif -%}}{{%- endfor -%}}",
        "{%- if not ns.pl -%}",
        f"{{%- for c in [{pl}] -%}}{{%- if c in q -%}}{{%- set ns.pl = true -%}}{{%- endif -%}}{{%- endfor -%}}",
        "{%- endif -%}",
        f"{{%- set codes = [{codes}] -%}}",
        "{%- for o in options -%}",
        "{%- set d = o.description -%}{%- set k = o.key -%}{%- set keyed = type == \"choice\" -%}",
        # LM Studio's /v1/decisions passes OpenAI choices as {"value", "description"} and levels as {"label",
        # "description"}: shown as the System One request {value: description} / {label: description} would be
        "{%- if d is mapping and d | length == 2 and \"description\" in d and ((type == \"choice\" and \"value\" in d)"
        " or (type == \"score\" and \"label\" in d)) -%}",
        "{%- set k = (d.value if d.value is string else d.value | tojson) if type == \"choice\" else d.label -%}",
        "{%- set d = d.description -%}{%- set keyed = true -%}",
        "{%- endif -%}",
        "{%- if type == \"noul\" -%}",
        "{%- if d -%}" f"{{%- set t = {text('d')} -%}}",
        "{%- elif k == \"true\" -%}{%- set t = \"Tak\" if ns.pl else \"Yes\" -%}",
        "{%- else -%}{%- set t = \"Nie\" if ns.pl else \"No\" -%}{%- endif -%}",
        f"{{%- elif not keyed -%}}{{%- set t = {text('d')} -%}}",
        "{%- elif d is none -%}{%- set t = k -%}",
        f"{{%- else -%}}{{%- set t = k ~ \": \" ~ {text('d')} -%}}",
        "{%- endif -%}",
        f"{{%- set t = {_fang(STRING_T, specials)} -%}}",
        "{%- set ns.opts = ns.opts ~ (\"\" if loop.first else \"\\n\") ~ codes[loop.index0] ~ \". \" ~ t -%}",
        "{%- endfor -%}",
        "{%- if ns.pl -%}{{- " + turn("pl") + " -}}{%- else -%}{{- " + turn("en") + " -}}{%- endif -%}"])


# requests the template is checked on before it is written: Polish from the state or from the question, descriptions
# or none, JSON states and descriptions, quotes and backslashes, whitespace at the ends, 26 options
SYSTEMONE_PROBES = (
    ("Customer: I was charged twice for order 1182 and nobody answered my emails.",
     {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"billing": "payments, refunds, invoices", "shipping": None, "technical": "the app or the website"}}),
    ("Klient: zamówienie nie dotarło, proszę o zwrot pieniędzy.",
     {"type": "noul", "instructions": "Is the customer asking for a refund?"}),
    ("Order 1182: delivered on 3 May.",
     {"type": "noul", "instructions": "Czy zamówienie dotarło?", "criteria": {"true": "tak, dotarło", "false": ""}}),
    ("  The patient reports mild pain since Monday.\n",
     {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", " now "]}),
    ({"ticket": 1182, "text": "Zażółć gęślą jaźń", "tags": ["a", "b"], "paid": True},
     {"type": "choice", "instructions": {"ask": "category"}, "criteria": {f"k{i}": None for i in range(26)}}),
    ('He said "stop" \\ and left.\n\nThen {{ nothing }} happened.',
     {"type": "choice", "instructions": "Who left?", "criteria": {"x": {"name": "X", "id": 7}, "y": "Y "}}),
)


def systemone_options(q):
    """The options llama.cpp gives the template, in its label order (openjev: noul true first)."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        return [{"key": k, "description": v} for k, v in crit.items()]
    if q["type"] == "score":
        return [{"key": str(i), "description": v} for i, v in enumerate(crit)]
    return [{"key": k, "description": (crit or {}).get(k)} for k in ("true", "false")]


def render_systemone(template, state, q):
    """The template's prompt for one question, rendered as llama.cpp does (trim_blocks, lstrip_blocks; tojson as
    compact JSON with ", " and ": " and no ASCII escaping)."""
    import jinja2

    def raise_exception(msg):
        raise ValueError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(id="q", type=q["type"], instructions=q["instructions"], state=state,
                                            options=systemone_options(q), images=[])


def check_systemone(tok, template):
    """The template renders basal-serve's prompt on every probe, and the option codes A-Z are the single tokens
    llama.cpp reads (each code's own token); raises ValueError otherwise."""
    from .prompt import CODES, letter_ids, render
    from .server import to_items

    for i, (state, q) in enumerate(SYSTEMONE_PROBES):
        item = to_items(state, {"q": q}, max_opts=SYSTEMONE_MAX_OPTIONS)[0]
        want = render(tok, item["state"], item["question"], item["options"], item["lang"], fmt=2)
        if render_systemone(template, state, q) != want:
            raise ValueError(f"the systemone template differs from basal's prompt on probe {i}: the model's chat "
                             "template changes the messages (e.g. trims them)")
    codes = CODES[:SYSTEMONE_MAX_OPTIONS]
    own = tok.convert_tokens_to_ids(list(codes))
    at_answer = letter_ids(tok, render_systemone(template, *SYSTEMONE_PROBES[0]), len(codes), fmt=2)
    if at_answer != own or tok.unk_token_id in own:
        raise ValueError("the option codes A-Z are not the tokens llama.cpp reads at the answer position")


def systemone_temperatures(cal):
    """{"<type>.<llama.cpp bucket>": T} of a basal calibration (calibration file 2, one option order); ValueError when a
    llama.cpp bucket would need two of basal's temperatures."""
    from .server import temperature

    if cal.get("prompt", 2) != 2 or cal.get("orders", 1) != 1:
        raise ValueError("the calibration was fitted for another prompt format or for several option orders; "
                         "llama.cpp reads prompt format 2 in one option order")
    out = {}
    for typ, (lo, hi) in SYSTEMONE_COUNTS.items():
        for name, b_lo, b_hi in LLAMA_BUCKETS:
            temps = {temperature(cal, typ, k) for k in range(max(lo, b_lo), min(hi, b_hi) + 1)}
            if len(temps) > 1:
                raise ValueError(f"{typ} options {b_lo}-{b_hi} have {len(temps)} temperatures in this calibration; "
                                 "llama.cpp keeps one per bucket (2, 3-5, 6-10, 11+)")
            if temps:
                out[f"{typ}.{name}"] = temps.pop()
    return out


def systemone_calibration(model_dir, path=None):
    """The calibration file written into the GGUF: `path`, else the first of SYSTEMONE_CALIBRATIONS in the model
    folder (the llama.cpp ports' own fits first)."""
    if path:
        return Path(path)
    for name in SYSTEMONE_CALIBRATIONS:
        if (Path(model_dir) / name).is_file():
            return Path(model_dir) / name
    raise ValueError(f"{model_dir} has no calibration file ({', '.join(SYSTEMONE_CALIBRATIONS)}); pass --calibration")


def systemone_metadata(model_dir, calibration=None):
    """(the decision keys of --systemone, without the architecture prefix, as {key: (kind, value)}; the calibration
    file used). The model must be prompt format 2 in the canonical layout, read one way, without isolated options."""
    from transformers import AutoTokenizer

    from .prompt import model_format, model_layout, model_option_isolation, model_two_way

    if model_format(model_dir) != 2 or model_layout(model_dir) != "canonical" or model_two_way(model_dir) \
            or model_option_isolation(model_dir):
        raise ValueError("--systemone needs a prompt-format-2 model in the canonical layout, read one way, without "
                         "isolated options (its basal.json)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    from .prompt import special_names

    template = systemone_template(*chat_frame(tok), specials=special_names(tok))
    check_systemone(tok, template)
    cal_path = systemone_calibration(model_dir, calibration)
    temps = systemone_temperatures(json.loads(cal_path.read_text(encoding="utf-8")))
    meta = {"decision.type": ("string", SYSTEMONE_TYPE), SYSTEMONE_TEMPLATE: ("string", template)}
    meta.update({f"decision.temperature.{k}": ("float32", t) for k, t in temps.items()})
    return meta, cal_path


def fix_gguf(raw, outfile, model_dir, decision=None):
    """Copy `raw` to `outfile` with the three tokenizer fields fixed from `model_dir`/tokenizer.json. The input's
    <arch>.decision.* keys and systemone template are dropped; `decision` (systemone_metadata) writes new ones. Returns
    (tokens ranked, tokens)."""
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    reader = gguf.GGUFReader(raw)
    tokens = reader.fields[gguf.Keys.Tokenizer.LIST].contents()
    tokenizer_json = json.loads((Path(model_dir) / "tokenizer.json").read_text(encoding="utf-8"))
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
    kinds = {"string": gguf.GGUFValueType.STRING, "float32": gguf.GGUFValueType.FLOAT32}
    for key, (kind, value) in (decision or {}).items():
        new[key if key.startswith("tokenizer.") else f"{arch}.{key}"] = MetadataDetails(kinds[kind], value)
    stale = [k for k in reader.fields if k.startswith(f"{arch}.decision.") or k == SYSTEMONE_TEMPLATE]
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
    ap.add_argument("--systemone", action="store_true",
                    help="also write the decision metadata of llama.cpp's /v1/systemone (llama-server, LM Studio "
                         "0.4.26+): basal's prompt as a template and the calibrated temperatures; prompt-format-2 "
                         "models, at most 26 options per question")
    ap.add_argument("--calibration", default=None,
                    help="--systemone: the calibration file whose temperatures are written (default: the model's "
                         + ", else ".join(SYSTEMONE_CALIBRATIONS) + ")")
    a = ap.parse_args(argv)
    if a.calibration and not a.systemone:
        ap.error("--calibration needs --systemone")
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

    files = (MODEL_FILES if a.systemone else ["tokenizer.json"]) if a.gguf else None
    model_dir = Path(a.model) if Path(a.model).exists() else Path(snapshot_download(
        a.model, revision=a.revision, allow_patterns=files))
    if not (model_dir / "tokenizer.json").is_file():
        ap.error(f"{a.model} has no tokenizer.json; pass the basal model the GGUF was converted from")
    decision = cal = None
    if a.systemone:
        try:
            decision, cal = systemone_metadata(model_dir, a.calibration)
        except ValueError as e:
            sys.exit(f"basal-export-gguf: {e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw = Path(a.gguf or Path(tmp) / "converted.gguf")
        if not a.gguf:
            done = subprocess.run([sys.executable, str(converter), str(model_dir), "--outtype", a.outtype,
                                   "--outfile", str(raw)])
            if done.returncode != 0:
                sys.exit(f"basal-export-gguf: llama.cpp's converter failed (exit {done.returncode}); "
                         "see its output above")
        part = Path(tmp) / out.name  # written in full first, then renamed: an interrupted export leaves no
        try:  # truncated file under the output's name (nor overwrites an older one)
            ranked, total = fix_gguf(raw, part, model_dir, decision)
        except ValueError as e:
            sys.exit(f"basal-export-gguf: {e}")
        os.replace(part, out)
    print(f"wrote {out}: {ranked}/{total} token scores from BPE merges, add_space_prefix=false")
    if decision:
        print(f"  decision model for /v1/systemone ({SYSTEMONE_TYPE}, at most {SYSTEMONE_MAX_OPTIONS} options), "
              f"temperatures from {cal.name}")


if __name__ == "__main__":
    main()
