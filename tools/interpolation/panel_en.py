"""A public English decision panel: 10 tasks from the Hugging Face hub as simple basal-calibrate items.

Each line is {"id", "family", "state", "question", "options", "type", "gold": <index>}; `family` is the task, so
basal-calibrate's leave-one-family-out ECE is per task. Items are sampled with a fixed seed.

    uv run --with datasets python tools/interpolation/panel_en.py --per-task 400 --out panel_en.jsonl
"""
import argparse
import json
import random

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


def arc(config):
    def f(r):
        labels = r["choices"]["label"]
        return dict(state="", question=r["question"], options=r["choices"]["text"],
                    gold=labels.index(r["answerKey"]))
    return ("allenai/ai2_arc", config, "test", f)


def obqa(r):
    return dict(state="", question=r["question_stem"], options=r["choices"]["text"],
                gold=r["choices"]["label"].index(r["answerKey"]))


def csqa(r):
    return dict(state="", question=r["question"], options=r["choices"]["text"],
                gold=r["choices"]["label"].index(r["answerKey"]))


def boolq(r):
    q = r["question"].strip().rstrip("?") + "?"
    return dict(type="noul", state=r["passage"], question=q[0].upper() + q[1:], options=["Yes", "No"],
                gold=0 if r["answer"] else 1)


def hellaswag(r):
    return dict(state=r["ctx"], question="Which ending continues the text most plausibly?", options=r["endings"],
                gold=int(r["label"]))


def winogrande(r):
    return dict(state=r["sentence"], question="Which option correctly fills the blank (_)?",
                options=[r["option1"], r["option2"]], gold=int(r["answer"]) - 1)


def sciq(r):
    opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
    order = list(range(4))
    random.Random(r["question"]).shuffle(order)
    return dict(state=r["support"], question=r["question"], options=[opts[i] for i in order], gold=order.index(0))


def piqa(r):
    return dict(state="", question=f"Goal: {r['goal']}\nWhich solution achieves the goal?",
                options=[r["sol1"], r["sol2"]], gold=int(r["label"]))


def mmlu(r):
    return dict(state="", question=f"({r['subject'].replace('_', ' ')}) {r['question']}", options=r["choices"],
                gold=int(r["answer"]))


TASKS = {
    "arc_easy": arc("ARC-Easy"),
    "arc_challenge": arc("ARC-Challenge"),
    "openbookqa": ("allenai/openbookqa", "main", "test", obqa),
    "commonsense_qa": ("tau/commonsense_qa", None, "validation", csqa),
    "boolq": ("google/boolq", None, "validation", boolq),
    "hellaswag": ("Rowan/hellaswag", None, "validation", hellaswag),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", winogrande),
    "sciq": ("allenai/sciq", None, "test", sciq),
    "piqa": ("baber/piqa", None, "validation", piqa),
    "mmlu": ("cais/mmlu", "all", "test", mmlu),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-task", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--out", default="panel_en.jsonl")
    args = p.parse_args()
    with open(args.out, "w") as out:
        for name in args.tasks:
            repo, config, split, conv = TASKS[name]
            ds = load_dataset(repo, config, split=split)
            idx = random.Random(f"{args.seed}-{name}").sample(range(len(ds)), min(args.per_task, len(ds)))
            n = 0
            for i in idx:
                item = conv(ds[i])
                if not 2 <= len(item["options"]) <= len(LETTERS) or not all(o.strip() for o in item["options"]):
                    continue
                item.setdefault("type", "choice")
                # json.dumps escapes U+0085 / U+2028 / U+2029 only with ensure_ascii, and basal-calibrate splits on them
                line = json.dumps({"id": f"{name}-{i}", "family": name, **item}, ensure_ascii=False)
                out.write(line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                          + "\n")
                n += 1
            print(f"{name}: {n} items ({repo}, {split})")


if __name__ == "__main__":
    main()
