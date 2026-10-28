"""A public Polish decision panel: 6 tasks (Belebele pol_Latn + 5 KLEJ tasks) as simple basal-calibrate items.

Same item format as panel_en.py ({"id", "family", "state", "question", "options", "type", "gold": <index>}), so the
results can be read side by side with the English panel: does an interpolated checkpoint recover English accuracy
without losing Polish?

    uv run --with datasets python tools/interpolation/panel_pl.py --per-task 400 --out panel_pl.jsonl
"""
import argparse
import json
import random

from datasets import load_dataset

POLEMO = {"__label__meta_plus_m": 0, "__label__meta_minus_m": 1, "__label__meta_zero": 2, "__label__meta_amb": 3}
CDSC = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}


def belebele(r):
    return dict(state=r["flores_passage"], question=r["question"],
                options=[r[f"mc_answer{i}"] for i in range(1, 5)], gold=int(r["correct_answer_num"]) - 1)


def polemo(r):
    return dict(state=r["sentence"], question="Jaki jest wydźwięk tej opinii?",
                options=["pozytywny", "negatywny", "neutralny", "ambiwalentny (zarówno pozytywny, jak i negatywny)"],
                gold=POLEMO[r["target"]])


def dyk(r):
    q, a = r["question"].strip('" '), r["answer"].strip('" ')
    return dict(type="noul", state=f"Pytanie: {q}\nOdpowiedź: {a}", question="Czy ta odpowiedź odpowiada na pytanie?",
                options=["Tak", "Nie"], gold=0 if int(r["target"]) == 1 else 1)


def cdsc(r):
    return dict(state=f"Zdanie A: {r['sentence_A']}\nZdanie B: {r['sentence_B']}",
                question="Jaka jest relacja zdania B do zdania A?",
                options=["zdanie B wynika ze zdania A", "zdanie B przeczy zdaniu A", "brak związku logicznego"],
                gold=CDSC[r["entailment_judgment"]])


def cbd(r):
    return dict(type="noul", state=r["sentence"], question="Czy ten wpis jest przemocą słowną (cyberprzemocą)?",
                options=["Tak", "Nie"], gold=0 if int(r["target"]) == 1 else 1)


def psc(r):
    return dict(type="noul", state=f"Tekst: {r['extract_text']}\n\nStreszczenie: {r['summary_text']}",
                question="Czy to streszczenie dotyczy tego tekstu?", options=["Tak", "Nie"],
                gold=0 if int(r["label"]) == 1 else 1)


TASKS = {
    "belebele_pl": ("facebook/belebele", "pol_Latn", "test", belebele),
    "polemo2_in": ("allegro/klej-polemo2-in", None, "test", polemo),
    "dyk": ("allegro/klej-dyk", None, "test", dyk),
    "cdsc_e": ("allegro/klej-cdsc-e", None, "test", cdsc),
    "cbd": ("allegro/klej-cbd", None, "test", cbd),
    "psc": ("allegro/klej-psc", None, "test", psc),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-task", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="panel_pl.jsonl")
    args = p.parse_args()
    with open(args.out, "w") as out:
        for name, (repo, config, split, conv) in TASKS.items():
            ds = load_dataset(repo, config, split=split)
            idx = random.Random(f"{args.seed}-{name}").sample(range(len(ds)), min(args.per_task, len(ds)))
            for i in idx:
                item = conv(ds[i])
                item.setdefault("type", "choice")
                # json.dumps escapes U+0085 / U+2028 / U+2029 only with ensure_ascii, and basal-calibrate splits on them
                line = json.dumps({"id": f"{name}-{i}", "family": name, **item}, ensure_ascii=False)
                out.write(line.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                          + "\n")
            print(f"{name}: {len(idx)} items ({repo}, {split})")


if __name__ == "__main__":
    main()
