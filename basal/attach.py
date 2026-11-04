"""Attach a base model: serve a chat model that was never trained on basal decisions (a Gemma 4 or Qwen model, for
example) as a decision model, without retraining, and fit its calibration on your own labelled requests.

  basal-attach Qwen/Qwen3-4B --out qwen3-4b-basal              # the model directory basal-serve reads
  basal-serve --model qwen3-4b-basal --mode eager
  basal-calibrate labelled.jsonl --model qwen3-4b-basal --mode eager --out CALIBRATION.custom.json
  basal-serve --model qwen3-4b-basal --mode eager --calibration CALIBRATION.custom.json

The output directory links (symlinks) every file of the base model and adds:
  basal.json          "prompt": the prompt format (1 by default: options A-J and an answer instruction, which a model
                      not trained on basal follows best; 2 with --format 2: up to 255 option codes, when every code is
                      one token), "layout": "canonical", "attached": the base model and its revision, "readout": the
                      option codes and their token ids at the answer position;
  CALIBRATION.json    uncalibrated defaults: no temperatures (1.0) and no certified thresholds, so max_error defers
                      every automatic action and act answers say "calibration": "unvalidated" until you serve a file
                      fitted with basal-calibrate;
  chat_template.jinja only for a tokenizer without a chat template: --chat-template FILE, else a plain one
                      ("System: ...", "User: ...", "Assistant: ").
The readout of every type is the next-token distribution over the option codes after the chat template and the
{"answer": " prefill (a noul question reads two codes, a score question one per level), so every code of the format
must be one token there. basal-attach checks that on a Polish and an English prompt and refuses a format whose codes
the tokenizer splits.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .prompt import FORMATS, letter_ids, max_options, render

PLAIN_TEMPLATE = ("{% for m in messages %}{{ m['role'] | capitalize }}: {{ m['content'] }}\n\n{% endfor %}"
                  "{% if add_generation_prompt %}Assistant: {% endif %}")
SAMPLE = {"en": ("Order 1240 arrived damaged; the customer asks for a refund.", "What should we do?"),
          "pl": ("Zamówienie 1240 dotarło uszkodzone; klient prosi o zwrot.", "Co robimy?")}
OWN = ("basal.json", "CALIBRATION.json")  # written here, never linked from the base


def split_codes(tok, prompt, fmt):
    """The option codes of a format that are not one token at the answer position of `prompt`."""
    base = tok(prompt, add_special_tokens=False).input_ids
    bad = []
    for c in FORMATS[fmt]["codes"]:
        full = tok(prompt + c, add_special_tokens=False).input_ids
        if full[: len(base)] != base or len(full) != len(base) + 1:
            bad.append(c)
    return bad


def readout(tok, fmt):
    """{lang: the token id of every option code of the format at the answer position}, on one prompt per language
    with the most options of the format; SystemExit naming the codes the tokenizer splits (or that share an id)."""
    out = {}
    for lang, (state, question) in SAMPLE.items():
        prompt = render(tok, state, question, [f"option {i}" for i in range(max_options(fmt))], lang, fmt)
        ids = letter_ids(tok, prompt, max_options(fmt), fmt)
        if ids is None:
            bad = split_codes(tok, prompt, fmt)
            why = (f"{len(bad)} of its {max_options(fmt)} option codes are not one token at the answer position "
                   f"({', '.join(bad[:12])}{' ...' if len(bad) > 12 else ''})" if bad
                   else "two option codes share a token id")
            raise SystemExit(f"prompt format {fmt}: {why} for this tokenizer ({lang} prompt)"
                             + ("; use --format 1 (options A-J)" if fmt == 2 else ""))
        out[lang] = ids
    return out


def check_out(src, out):
    """SystemExit before anything is written when the output directory is the base's own directory (also through a
    symlink), lies inside it or contains it: the links would replace the base's files with links to themselves."""
    s, o = Path(src).resolve(), Path(out).resolve()
    if o == s:
        raise SystemExit(f"--out {out}: the base model's own directory; give a new directory")
    if o.is_relative_to(s) or s.is_relative_to(o):
        raise SystemExit(f"--out {out}: {'inside' if o.is_relative_to(s) else 'contains'} the base model's directory "
                         f"{src}; give a directory outside it")


def attach(base, out, fmt=1, chat_template=None, revision=None, force=False):
    """Write the basal model directory `out` for the base model `base` (a directory or a Hugging Face repo id) ->
    the basal.json written."""
    from transformers import AutoTokenizer

    from .engine import resolve
    from .server import model_identity
    src = resolve(base, revision)
    if (src / "basal.json").exists():
        raise SystemExit(f"{base} is already a basal model (it has basal.json): serve it as it is")
    out = Path(out)
    check_out(src, out)
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"--out {out}: not empty (--force replaces the links and files basal-attach writes)")
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.name in OWN or f.name == "chat_template.jinja" and chat_template:
            continue
        link = out / f.name
        if link.is_symlink() or link.exists():
            shutil.rmtree(link) if link.is_dir() and not link.is_symlink() else link.unlink()
        target = f.resolve() if f.is_symlink() else f.absolute()
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except OSError:
            if target.is_dir():
                shutil.copytree(target, link)
            else:
                try:
                    os.link(target, link)
                except OSError:
                    shutil.copy2(target, link)
    tok = AutoTokenizer.from_pretrained(out)
    template = None
    if chat_template or not getattr(tok, "chat_template", None):
        template = Path(chat_template).read_text(encoding="utf-8") if chat_template else PLAIN_TEMPLATE
        (out / "chat_template.jinja").unlink(missing_ok=True)
        (out / "chat_template.jinja").write_text(template, encoding="utf-8")
        tok = AutoTokenizer.from_pretrained(out)
        if tok.chat_template != template:
            raise SystemExit(f"{out}: the tokenizer does not use chat_template.jinja (its tokenizer_config.json has a "
                             "chat template of its own)")
    ids = readout(tok, fmt)
    config = json.loads((src / "config.json").read_text(encoding="utf-8")) if (src / "config.json").exists() else {}
    name, rev = model_identity(base, src)
    meta = {"prompt": fmt, "layout": "canonical",
            "attached": {"base": name, "revision": rev, "model_type": config.get("model_type"),
                         "chat_template": "chat_template.jinja (plain)" if template == PLAIN_TEMPLATE
                         else "chat_template.jinja (--chat-template)" if template else "the base's",
                         "date": datetime.date.today().isoformat(), "tool": f"basal-attach {__version__}"},
            "readout": {"codes": "".join(FORMATS[fmt]["codes"]) if fmt == 1 else list(FORMATS[fmt]["codes"]),
                        "ids": ids["en"] if ids["en"] == ids["pl"] else ids}}
    (out / "basal.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    cal = {"prompt": fmt, "temperature_per_prim": {}, "thresholds": {},
           "source_tag": "basal-attach defaults, uncalibrated: fit with basal-calibrate"}
    (out / "CALIBRATION.json").write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="the base model: a local directory or a Hugging Face repo id")
    ap.add_argument("--out", required=True, help="the basal model directory to write (links to the base's files)")
    ap.add_argument("--format", type=int, choices=sorted(FORMATS), default=1,
                    help="prompt format: 1 (options A-J with an answer instruction, the default) or 2 (up to 255 codes)")
    ap.add_argument("--chat-template", dest="chat_template", default=None,
                    help="a Jinja chat template for a tokenizer without one (default: a plain one)")
    ap.add_argument("--revision", default=None, help="the base's Hugging Face revision")
    ap.add_argument("--force", action="store_true", help="write into a directory that is not empty")
    a = ap.parse_args(argv)
    meta = attach(a.base, a.out, a.format, a.chat_template, a.revision, a.force)
    print(f"wrote {a.out}: prompt format {meta['prompt']}, {max_options(meta['prompt'])} options at most, readout "
          f"checked on the base's tokenizer ({meta['attached']['chat_template']} chat template); uncalibrated until "
          f"basal-calibrate:\n  basal-serve --model {a.out} --mode eager\n"
          f"  basal-calibrate labelled.jsonl --model {a.out} --mode eager --out CALIBRATION.custom.json", file=sys.stderr)


if __name__ == "__main__":
    main()
