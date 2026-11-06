"""basal-selfcheck: does this server (this GPU, this build, this quantisation) answer like the reference?

  basal-selfcheck --url http://127.0.0.1:8000 --reference basal-2.0-mini               # a reference file we ship
  basal-selfcheck --url http://127.0.0.1:8000 --reference ref.json --note "GTX 1060 6GB, Vulkan, Q8_0"

A reference file holds a few dozen requests (short, medium and long states, Polish and English, choice / noul / score
questions) and the probabilities that basal's own PyTorch engine (fp32) gave for them. The check sends the same
requests to your server and prints, over all readouts:

  top-1 agreement     the share of questions whose most probable option is the reference's. A disagreement where the
                      server's top option was within 0.02 of the reference's top, in the reference, counts as
                      agreeing ("outside near ties"): any change of rounding can flip those
  max |dp|            the largest change of any option's probability, and the mean
  seconds             wall time per request

The exit status is 0 when the agreement outside near ties is at least --gate (default 0.98), 1 otherwise. Start the
server without calibration (basal-serve ... --no-calibration), as the reference was made: the check compares raw
probabilities. Requests the server had to cut (usage.truncated: the context is too short) are left out and counted; the check
fails (exit 1) when fewer than half of the requests were compared, when a request failed or when a question came back
unanswered.

Make a reference file for a model (a maintainer's step; run it against a server of the native engine, fp32):

  basal-serve --model Remek/basal-2.0-mini --dtype float32 --no-calibration --port 8000
  basal-selfcheck --make-reference basal-2.0-mini.json --url http://127.0.0.1:8000"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__, synth
from .textio import read_text

TIE = 0.02
SHAPES = ((12, 200), (6, 800), (2, 1500))  # (requests, state tokens)
EXAMPLES = Path(__file__).parent / "examples"

QUESTIONS = {
    "pl": {"urgent": ("noul", "Czy klient potrzebuje odpowiedzi jeszcze dziś?", None),
           "tone": ("score", "Jaki jest ton ostatniego zgłoszenia?", ["spokojny", "zniecierpliwiony", "wzburzony"])},
    "en": {"urgent": ("noul", "Does the customer need an answer today?", None),
           "tone": ("score", "What is the tone of the latest ticket?", ["calm", "impatient", "angry"])},
}


def build_requests():
    """The reference requests: one state with a choice question (the unit that should handle the latest entry, 2..10
    options), a yes/no question and a three-level score question. Deterministic (basal.synth)."""
    reqs = []
    for k, (n, tokens) in enumerate(SHAPES):
        for it in synth.items(n, tokens, seed=100 + k):
            lang = "pl" if any(c in "ąćęłńóśźż" for c in it["state"]) else "en"
            qs = {"unit": {"type": "choice", "instructions": it["question"],
                           "criteria": {f"o{i}": o for i, o in enumerate(it["options"])}}}
            for name, (kind, text, levels) in QUESTIONS[lang].items():
                qs[name] = {"type": kind, "instructions": text, **({"criteria": levels} if levels else {})}
            reqs.append({"id": f"{len(reqs):02d}-{tokens}-{lang}", "state": it["state"], "questions": qs})
    return reqs


def call(url, path, body=None, timeout=300):
    req = urllib.request.Request(url.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url}{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    except OSError as e:
        raise SystemExit(f"{url}{path}: {e} (is the server running?)") from None
    except ValueError as e:  # a 200 that is not JSON (a proxy's error page)
        raise SystemExit(f"{url}{path}: the reply is not JSON ({e})") from None


def uncalibrated(entry):
    """Whether a GET /v1/models entry serves raw probabilities: every temperature 1.0, the per-type ones and the
    per-option-count ones of a format-2 calibration (as basal-calibrate checks a server)."""
    temps = list((entry.get("temperature_per_prim") or {}).values())
    for per_bucket in (entry.get("temperature_per_bucket") or {}).values():
        temps += list((per_bucket or {}).values())
    return all(float(t) == 1.0 for t in temps)


def server_info(url):
    m = call(url, "/v1/models")["models"][0]
    return {"name": m.get("name"), "mode": m.get("mode"), "uncalibrated": uncalibrated(m)}


def find(reference):
    """A reference given by a model name -> the file shipped with basal, else a path."""
    for p in (Path(reference), EXAMPLES / f"selfcheck_{reference}.json", EXAMPLES / reference):
        if p.is_file():
            return p
    shipped = sorted(f.stem.removeprefix("selfcheck_") for f in EXAMPLES.glob("selfcheck_*.json"))
    raise SystemExit(f"reference {reference!r}: not a file; shipped references: {', '.join(shipped) or 'none'}")


def make_reference(url, out, note=""):
    info = server_info(url)
    if not info["uncalibrated"]:
        raise SystemExit("the reference needs raw probabilities: start the server with --no-calibration")
    items = []
    for req in build_requests():
        res = call(url, "/v1/systemone", {"state": req["state"], "questions": req["questions"]})
        items.append({**req, "answers": {k: {"type": a["type"], "probabilities": a["probabilities"]}
                                         for k, a in res["answers"].items()}})
    ref = {"format": 1, "model": (info["name"] or "").removesuffix("-GGUF"), "engine": info["mode"], "basal": __version__,
           "note": note or "probabilities of basal's native engine, uncalibrated", "items": items}
    Path(out).write_text(json.dumps(ref, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(items)} requests, {sum(len(i['answers']) for i in items)} readouts", file=sys.stderr)


def compare(ref, got):
    """Probability dicts of one question -> (top-1 agrees, agrees outside near ties, max |dp|, sum |dp|, options)."""
    keys = list(ref)
    r, g = [ref[k] for k in keys], [got.get(k, 0.0) for k in keys]
    if any(x != x or abs(x) == float("inf") for x in g + r):  # a NaN or infinite answer or reference is the largest possible change, never an agreement
        return False, False, 1.0, float(len(keys)), len(keys)
    a, b = max(range(len(r)), key=r.__getitem__), max(range(len(g)), key=g.__getitem__)
    dps = [abs(x - y) for x, y in zip(r, g)]
    # near tie: the server's top option is within TIE of the reference's top, in the reference's probabilities (a
    # three-way tie counts), so a flip to an option the reference ranked clearly lower is still a disagreement
    return a == b, a == b or r[a] - r[b] < TIE, max(dps), sum(dps), len(keys)


def check(url, reference, gate, note):
    ref = json.loads(read_text(find(reference)))
    info = server_info(url)
    if info["name"] and ref["model"] and info["name"].removesuffix("-GGUF") != ref["model"].removesuffix("-GGUF"):
        print(f"warning: the server reports model {info['name']!r}, the reference is for {ref['model']!r}", file=sys.stderr)
    if not info["uncalibrated"]:
        print("warning: the server applies a calibration; start it with --no-calibration, or the probabilities are "
              "tempered and the |dp| below is not the engine's", file=sys.stderr)
    rows, cut, errors, missing, times = [], 0, 0, 0, []
    for it in ref["items"]:
        t = time.perf_counter()
        try:
            res = call(url, "/v1/systemone", {"state": it["state"], "questions": it["questions"]})
        except SystemExit as e:
            errors += 1
            print(f"{it['id']}: {e}", file=sys.stderr)
            continue
        times.append(time.perf_counter() - t)
        if not isinstance(res, dict) or not isinstance(res.get("answers"), dict):
            # a 200 that is JSON but not an answer ({"error": ...}, null, a list): a failed request
            errors += 1
            print(f"{it['id']}: the reply is not an answer object ({str(res)[:100]})", file=sys.stderr)
            continue
        if (res.get("usage") or {}).get("truncated"):
            cut += 1
            continue
        for name, want in it["answers"].items():
            a = res["answers"].get(name)
            if not isinstance(a, dict) or not isinstance(a.get("probabilities"), dict):
                missing += 1  # unanswered or without probabilities: not a pass
                continue
            rows.append((it["id"], name, want["type"], *compare(want["probabilities"], a["probabilities"])))
    if not rows:
        raise SystemExit("no readouts compared (every request failed, was cut or came back without answers: "
                         f"{errors} failed, {cut} cut, {missing} answers missing)")
    n = len(rows)
    top1 = sum(r[3] for r in rows) / n
    agree = sum(r[4] for r in rows) / n
    worst = max(rows, key=lambda r: r[5])
    options = sum(r[7] for r in rows)
    times.sort()
    print(f"basal-selfcheck {__version__}  reference {ref['model']} ({ref['engine']}, basal {ref['basal']}): {ref['note']}")
    print(f"server       {info['name']} ({info['mode']})" + (f"   {note}" if note else ""))
    print(f"requests     {len(ref['items'])}: {len(ref['items']) - cut - errors} compared, {cut} cut by the context "
          f"(usage.truncated), {errors} failed, {missing} answers missing")
    print(f"readouts     {n}")
    print(f"top-1        {top1:.4f}   outside near ties (server's top within {TIE} of the reference's top) {agree:.4f}")
    print(f"|dp|         mean {sum(r[6] for r in rows) / options:.4f}   max {worst[5]:.4f} ({worst[0]} {worst[1]})")
    print(f"seconds      p50 {times[len(times) // 2]:.2f}   max {times[-1]:.2f} per request" if times else "")
    for t in ("choice", "noul", "score"):
        sub = [r for r in rows if r[2] == t]
        if sub:
            print(f"  {t:<7}    {len(sub):>4} readouts  top-1 {sum(r[3] for r in sub) / len(sub):.4f}  "
                  f"max |dp| {max(r[5] for r in sub):.4f}")
    ok = agree >= gate and not errors and not missing and len(ref["items"]) - cut - errors >= len(ref["items"]) / 2
    print(f"{'PASS' if ok else 'FAIL'}  (gate: agreement outside near ties >= {gate}, no failed request, no missing answer, "
          "at least half of the requests compared)")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the basal-serve (any --mode) to check")
    ap.add_argument("--reference", help="a reference file, or the name of one shipped with basal")
    ap.add_argument("--gate", type=float, default=0.98)
    ap.add_argument("--note", default="", help="what you test, printed with the result (GPU, driver, quantisation); "
                    "with --make-reference: how the reference was made")
    ap.add_argument("--make-reference", dest="make", metavar="FILE", help="write a reference file from this server")
    a = ap.parse_args(argv)
    if a.make:
        return make_reference(a.url, a.make, a.note)
    if not a.reference:
        ap.error("--reference is required (or --make-reference FILE)")
    sys.exit(check(a.url, a.reference, a.gate, a.note))


if __name__ == "__main__":
    main()
