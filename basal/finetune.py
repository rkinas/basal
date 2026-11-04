"""Adapt a basal model to your own labelled decisions with LoRA (2.5), then calibrate it with basal-calibrate.

  pip install "basal[finetune]"
  basal-finetune labelled.jsonl --model Remek/basal-omni-pico --out my-pico    # trains, merges, writes my-pico/
  basal-calibrate my-pico/heldout.jsonl --model my-pico --mode eager --out my-pico/CALIBRATION.json
  basal-serve --model my-pico --mode eager

The input is the JSONL of basal-calibrate (docs/CALIBRATION.md): one request per line, every question with its correct
answer in "gold". The lines are split by group into a training part and a held-out part (--holdout, default 25%): the
LoRA sees the first only, basal-calibrate must fit on the second (a calibration fitted on lines the model trained on
is over-confident). Training uses the prompt basal-serve builds, in the model's format, and the loss of the served
readout: the softmax over the option codes at the answer position. Each example is shown in the given option order or
the reversed one, the two orders basal-serve averages. choice, noul, score and act questions are trained; multi
questions and task types are skipped (counted).

The base is a basal model (a directory with basal.json) or a chat model attached with basal-attach. It is not
changed: --out gets a full merged model (and the adapter in --out/lora)."""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import __version__
from .calibrate import InputError, load
from .engine import default_device, resolve
from .facts import inject as inject_facts
from .jsonl import read_lines
from .prompt import letter_ids, max_options, model_format, model_layout, model_option_isolation, model_two_way, render
from .server import to_items

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def in_holdout(group, holdout, seed):
    """Whether a group is held out: a hash of the group, so twins and one customer's lines stay on the same side."""
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
    return h / 0x100000000 < holdout


def split(path, holdout=0.25, seed=0):
    """Labelled JSONL -> (train decisions with their bodies, held-out line texts, skipped counts). Lines that share a
    group on any of their decisions (a per-question "group") form one unit, so no group is on both sides; a unit goes
    with the group of the first decision of its first line (a line that shares no group: its first decision's group)."""
    bodies, decisions = load(path)
    lines = read_lines(path)
    root, first, owner = {}, {}, {}  # line -> parent line (union-find), line -> its first group, group -> a line

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for d in decisions:
        n = d["line"]
        root.setdefault(n, n)
        first.setdefault(n, d["group"])
        other = owner.setdefault(d["group"], n)
        a, b = find(n), find(other)
        root[max(a, b)] = min(a, b)  # the unit's root is its first line
    side = {n: in_holdout(first[find(n)], holdout, seed) for n in root}
    train, skipped = [], {}
    for d in decisions:
        if side[d["line"]]:
            continue
        if d["prim"] == "multi":
            skipped["multi"] = skipped.get("multi", 0) + 1
        else:
            train.append((bodies[d["req"]], d))
    held = [lines[n - 1] for n in sorted(k for k, v in side.items() if v)]
    return train, held, skipped


def example(tok, body, d, fmt, layout, max_len):
    """(prompt ids per option order, code ids, gold index per order) for one labelled decision, or None with why."""
    state = body["state"]
    if body.get("facts") == "auto":
        state = inject_facts(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
    item = to_items(state, {d["name"]: body["questions"][d["name"]]}, max_options(fmt), force_lang=body.get("lang"))[0]
    opts, k = item["options"], len(item["options"])
    orders = []
    for perm in (list(range(k)), list(range(k))[::-1]):
        prompt = render(tok, item["state"], item["question"], [opts[i] for i in perm], item["lang"], fmt, layout)
        codes = letter_ids(tok, prompt, k, fmt)
        if codes is None:
            return None, "an option code is not one token at the answer position"
        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) > max_len:
            return None, f"longer than --max-len {max_len} tokens"
        orders.append((ids, codes, perm.index(d["gold"])))
    return orders, None


def refused(src):
    """Why basal-finetune cannot train model src, or None: it trains and scores with the plain causal forward, and a
    model the engine reads another way (basal.json "two_way", "option_isolation") would be trained under masks it is
    not served with (and the merged model, keeping its basal.json, served two-way)."""
    for key, read in (("two_way", model_two_way), ("option_isolation", model_option_isolation)):
        if read(src):
            return (f"{src}: basal.json \"{key}\": basal-finetune trains with the plain causal forward only; this model "
                    "is served with other attention masks")
    return None


def lora_targets(model):
    """The linear layers LoRA trains: those of the text decoder's layers only (layers.N.<module>, as --adapter takes
    them), not the vision / audio towers ("all-linear" would train those too, and --adapter refuses such an adapter)."""
    from .adapter import _LAYER
    return sorted(n for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and _LAYER.match(n))


def copy_side_files(src, out):
    """The base model's files the merged model keeps (tokenizer, chat template, processor ...): not the weights, the
    index, basal.json (written anew), the card, nor what belongs to the base weights only (adapter.NOT_FUSED: its
    calibration files and NVFP4 activation scales)."""
    from .adapter import NOT_FUSED
    for f in sorted(Path(src).iterdir()):
        if f.is_file() and not f.name.endswith(WEIGHT_SUFFIXES) and not f.name.endswith(".index.json") \
                and f.name not in ("basal.json", "README.md", "MANIFEST.md", "SHA256SUMS", ".gitattributes") \
                and not any(f.match(s) for s in NOT_FUSED):
            shutil.copy2(f, Path(out) / f.name)


def code_logits(model, ids, codes, softcap, device):
    """The logits of the option codes at the answer position as basal-serve reads them: the output head's input (the
    final norm's output at the last position) times the code rows in fp32, then the model's final logit soft cap once.
    (The model's own logits are capped already: a Gemma forward applies final_logit_softcapping itself.) In fp32 also
    under the training steps' autocast, which would take the product in bf16 / fp16."""
    head = model.get_output_embeddings()
    got = {}
    hook = head.register_forward_pre_hook(lambda mod, args: got.__setitem__("h", args[0]))
    try:
        model(input_ids=torch.tensor([ids], device=device), logits_to_keep=1)
    finally:
        hook.remove()
    sel = torch.tensor(codes, device=device)
    with torch.autocast(torch.device(device).type, enabled=False):  # fp32 under the training steps' autocast too
        z = got["h"][0, -1].float() @ head.weight.index_select(0, sel).float().T
        if head.bias is not None:
            z = z + head.bias.index_select(0, sel).float()
        return torch.tanh(z / softcap) * softcap if softcap else z


def accuracy(model, examples, softcap, device):
    """Top-1 accuracy of the given option order over examples."""
    hits = 0
    with torch.no_grad():
        for orders in examples:
            ids, codes, gold = orders[0]
            hits += int(code_logits(model, ids, codes, softcap, device).argmax().item() == gold)
    return hits / max(len(examples), 1)


def pick_dtype(name, device):
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # a T4 has no bfloat16
    return torch.bfloat16 if device == "mps" else torch.float32


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labelled", help="labelled JSONL, the input of basal-calibrate")
    ap.add_argument("--model", required=True, help="a basal model: local directory or Hugging Face repo id")
    ap.add_argument("--out", required=True, help="directory to write: the merged model, heldout.jsonl, lora/")
    ap.add_argument("--holdout", type=float, default=0.25, help="share of the groups kept out of training (default 0.25)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="stop after this many optimizer steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16, help="examples per optimizer step (accumulated one by one)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", dest="max_len", type=int, default=2048, help="examples with a longer prompt are skipped")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                    help="auto: bfloat16 on a GPU that has it, float16 on a T4, bfloat16 on Apple Silicon, else float32")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--grad-checkpointing", dest="grad_ckpt", action="store_true",
                    help="less memory, about a third slower (use it on a 16 GB GPU with a model of 3B or more)")
    ap.add_argument("--eval-max", dest="eval_max", type=int, default=200, help="held-out decisions scored before and after")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="write into a directory that is not empty")
    a = ap.parse_args(argv)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit('basal-finetune needs peft: pip install "basal[finetune]"') from None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"--out {out}: not empty (--force writes into it)")
    src = resolve(a.model)
    if not (src / "basal.json").exists():
        raise SystemExit(f"{a.model} has no basal.json: use a basal model, or attach a chat model first "
                         "(basal-attach, docs/SERVING.md)")
    if refused(src):
        raise SystemExit(refused(src))
    fmt, layout = model_format(src), model_layout(src)
    try:
        train, held, skipped = split(a.labelled, a.holdout, a.seed)
    except InputError as e:
        raise SystemExit(str(e)) from None
    if not train or not held:
        raise SystemExit(f"{a.labelled}: {len(train)} training decisions and {len(held)} held-out lines; need both "
                         "(more groups, or a different --holdout)")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device or default_device()
    dtype = pick_dtype(a.dtype, device)
    tok = AutoTokenizer.from_pretrained(src)

    def build(pairs):
        ex = []
        for body, d in pairs:
            orders, why = example(tok, body, d, fmt, layout, a.max_len)
            if orders:
                ex.append(orders)
            else:
                skipped[why] = skipped.get(why, 0) + 1
        return ex

    examples = build(train)
    if not examples:
        raise SystemExit(f"no trainable decision ({skipped})")
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.jsonl").write_text("\n".join(held) + "\n", encoding="utf-8")
    bodies, decisions = load(out / "heldout.jsonl")
    dev_pairs = [(bodies[d["req"]], d) for d in decisions if d["prim"] != "multi"][: a.eval_max]
    dev = build(dev_pairs)
    print(f"basal-finetune: {len(examples)} training decisions, {len(dev)} held-out scored, {len(held)} held-out lines"
          + (f", skipped {skipped}" if skipped else "") + f"; {a.model} format {fmt} on {device} {str(dtype)[6:]}",
          file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(device)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    softcap = getattr(cfg, "final_logit_softcapping", None)
    model.eval()
    before = accuracy(model, dev, softcap, device) if dev else None
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=lora_targets(model), task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"basal-finetune: LoRA rank {a.rank} on the decoder's linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M "
          "parameters", file=sys.stderr)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device) if dtype == torch.float16 and device == "cuda" else None
    steps = math.ceil(len(examples) * a.epochs / a.batch)
    steps = min(steps, a.max_steps) if a.max_steps else steps
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then cosine to zero
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    model.train()
    t0, order, pos, run = time.time(), [], 0, []
    for step in range(steps):
        for _ in range(a.batch):
            if pos >= len(order):
                order, pos = random.sample(range(len(examples)), len(examples)), 0
            ids, codes, gold = random.choice(examples[order[pos]])  # the given option order or the reversed one
            pos += 1
            with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32 and device != "mps"):
                lg = code_logits(model, ids, codes, softcap, device)
            loss = F.cross_entropy(lg[None], torch.tensor([gold], device=device)) / a.batch
            (scaler.scale(loss) if scaler else loss).backward()
            run.append(loss.item() * a.batch)
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            print(f"basal-finetune: step {step + 1}/{steps} loss {sum(run) / len(run):.4f} "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)
            run = []
    model.eval()
    after = accuracy(model, dev, softcap, device) if dev else None

    model.save_pretrained(out / "lora")
    merged = model.merge_and_unload()
    copy_side_files(src, out)
    for old in out.glob("*.safetensors"):
        old.unlink()
    merged.save_pretrained(out, safe_serialization=True)
    meta = json.loads((src / "basal.json").read_text(encoding="utf-8"))
    meta.pop("test_metrics", None)
    meta["finetuned"] = {"base": str(a.model), "method": f"LoRA rank {a.rank}", "decisions": len(examples),
                         "epochs": a.epochs, "tool": f"basal-finetune {__version__}"}
    (out / "basal.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    cal = {"prompt": fmt, "temperature_per_prim": {}, "thresholds": {},
           "source_tag": "basal-finetune defaults, uncalibrated: fit with basal-calibrate on heldout.jsonl"}
    (out / "CALIBRATION.json").write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    report = {"train_decisions": len(examples), "heldout_lines": len(held), "scored": len(dev), "skipped": skipped,
              "top1_before": before, "top1_after": after, "steps": steps, "seconds": round(time.time() - t0)}
    (out / "finetune.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"basal-finetune: held-out top-1 {before:.3f} -> {after:.3f}" if dev else "basal-finetune: done", file=sys.stderr)
    print(f"wrote {out} (uncalibrated). Next:\n"
          f"  basal-calibrate {out}/heldout.jsonl --model {out} --mode eager --out {out}/CALIBRATION.json\n"
          f"  basal-serve --model {out} --mode eager", file=sys.stderr)


if __name__ == "__main__":
    main()
