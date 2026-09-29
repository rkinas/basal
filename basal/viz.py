"""Local visualiser: every step of a typed decision, from the rendered prompt to the calibrated answer.

  basal-viz --model Remek/basal-1.0-1.5B --port 8080      then open http://127.0.0.1:8080

POST /api/explain takes a /v1/systemone request body and returns, per question: the prompt of each option order (split
into template / state / question / options / prefill), the shared-prefix packing, the letter probabilities of each
order, the per-layer readout ("logit lens": each layer's hidden state at the answer position through the final norm and
the LM head, restricted to the option letters), the order average, the calibrated distribution and the decision
against the confidence thresholds of CALIBRATION.json. The answer is also computed with the packed forward the server
uses, to show that both paths agree.
"""
import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import torch

from .engine import SharedBackend, default_device, resolve, sync
from .prompt import LETTERS, PREFILL, letter_ids, render
from .run import to_request
from .server import to_items

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"


def segments(prompt, state, question, options):
    """Split a rendered prompt into labelled pieces (the template text between them is "template")."""
    marks, at = [], 0
    for kind, text in [("state", state), ("question", question)] + [
            ("option", f"{LETTERS[k]}. {o}") for k, o in enumerate(options)]:
        i = prompt.find(text, at) if text else -1
        if i < 0:
            continue
        marks.append((i, i + len(text), kind))
        at = i + len(text)
    out, at = [], 0
    for a, b, kind in marks:
        if a > at:
            out.append({"kind": "template", "text": prompt[at:a]})
        out.append({"kind": kind, "text": prompt[a:b]})
        at = b
    tail = prompt[at:]
    if tail.endswith(PREFILL):
        if tail[: -len(PREFILL)]:
            out.append({"kind": "template", "text": tail[: -len(PREFILL)]})
        out.append({"kind": "prefill", "text": PREFILL})
    elif tail:
        out.append({"kind": "template", "text": tail})
    return out


class Explainer:
    def __init__(self, a):
        md = resolve(a.model)
        self.name = a.model.rstrip("/").split("/")[-1]
        self.be = SharedBackend(md, a.dtype, a.device)
        self.tok, self.model = self.be.tok, self.be.model
        cal = md / "CALIBRATION.json"
        cal = json.loads(cal.read_text()) if cal.exists() else {}
        self.temps = cal.get("temperature_per_prim", {})
        self.thresholds = {k: v["confidence"] for k, v in cal.get("thresholds", {}).items()}
        self.lock = threading.Lock()
        self.info = {"model": self.name, "device": str(self.be.dev), "dtype": a.dtype,
                     "layers": len(self.model.model.layers), "thresholds": self.thresholds, "temperatures": self.temps}

    @torch.no_grad()
    def lens(self, ids, letters):
        """Forward with a hook on every decoder layer; letter distribution after each layer and the final readout."""
        mm, caught = self.model.model, []
        hooks = [layer.register_forward_hook(lambda _m, _i, o: caught.append((o[0] if isinstance(o, tuple) else o)[0, -1]))
                 for layer in mm.layers]
        try:
            final = mm(input_ids=torch.tensor([ids], device=self.be.dev), use_cache=False).last_hidden_state[0, -1]
        finally:
            for h in hooks:
                h.remove()
        H = torch.stack([mm.norm(h[None])[0] for h in caught[:-1]] + [final])
        lp = torch.log_softmax(self.model.lm_head(H).float(), -1)
        per_layer = torch.softmax(lp[:, letters], -1)
        top = lp[-1].topk(5)  # what the model would write without the letter restriction
        return per_layer.tolist(), [(self.tok.decode([int(i)]), float(p.exp())) for p, i in zip(top.values, top.indices)]

    def explain_question(self, q):
        k = len(q["options"])
        perms = [list(range(k)), list(range(k))[::-1]]
        orders, toks = [], []
        t0 = time.perf_counter()
        for perm in perms:
            opts = [q["options"][c] for c in perm]
            prompt = render(self.tok, q["state"], q["question"], opts, q["lang"])
            ids = self.tok(prompt, add_special_tokens=False).input_ids
            letters = letter_ids(self.tok, prompt, k)
            if letters is None:
                raise ValueError("option letters are not single tokens for this tokenizer")
            toks.append(ids)
            orders.append(dict(perm=perm, prompt=prompt, segments=segments(prompt, q["state"], q["question"], opts),
                               n_tokens=len(ids), letters=list(LETTERS[:k]), letter_ids=letters))
        t_tok = time.perf_counter() - t0

        sync(self.be.dev); t0 = time.perf_counter()
        for o, ids in zip(orders, toks):
            layers, top = self.lens(ids, o["letter_ids"])
            o["letter_probs"] = layers[-1]
            o["top_tokens"] = top
            canon = lambda p: [p[o["perm"].index(j)] for j in range(k)]  # position -> option index
            o["canon"] = canon(o["letter_probs"])
            o["layers_canon"] = [canon(p) for p in layers]
        sync(self.be.dev); t_sep = time.perf_counter() - t0

        sync(self.be.dev); t0 = time.perf_counter()
        packed = self.be.run_shared([([o["prompt"] for o in orders], [o["letter_ids"] for o in orders])])[0]
        sync(self.be.dev); t_packed = time.perf_counter() - t0
        diff = max(abs(a - b) for o, p in zip(orders, packed) for a, b in zip(o["letter_probs"], p))
        _, _, seg, _ = self.be._pack(toks)

        mean = [sum(o["canon"][j] for o in orders) / len(orders) for j in range(k)]
        T = self.temps.get(q["type"], 1.0)
        cal = torch.softmax(torch.log(torch.tensor(mean).clamp_min(1e-12)) / T, -1).tolist() if T != 1.0 else mean
        conf = max(cal)
        best = cal.index(conf)
        answer = {"index": best, "key": q["keys"][best], "confidence": conf}
        if q["type"] == "noul":
            answer["noul"] = cal[0]
        elif q["type"] == "score":
            answer["score"] = sum(i * v for i, v in enumerate(cal))
        n_layers = len(orders[0]["layers_canon"])
        return dict(
            name=q["name"], type=q["type"], lang=q["lang"], keys=q["keys"], options=q["options"],
            question=q["question"], orders=orders,
            packing=dict(prefix=seg.count(0), tails=[seg.count(g) for g in range(1, len(orders) + 1)],
                         separate=sum(len(t) for t in toks)),
            layers=[[sum(o["layers_canon"][L][j] for o in orders) / len(orders) for j in range(k)] for L in range(n_layers)],
            mean=mean, temperature=T, calibrated=cal, answer=answer, thresholds=self.thresholds,
            packed_max_diff=diff,
            timing_ms=dict(tokenize=t_tok * 1000, separate_with_lens=t_sep * 1000, packed=t_packed * 1000))

    def explain(self, body):
        with self.lock:
            t0 = time.perf_counter()
            qs = [self.explain_question(q) for q in to_items(body["state"], body["questions"])]
            return {"model": self.name, "questions": qs, "total_ms": (time.perf_counter() - t0) * 1000}


def examples():
    out = []
    for f in sorted(EXAMPLES.glob("*.jsonl")):
        for n, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            ex = {"file": f.stem, "id": item.get("id") or f"{f.stem}-{n + 1}", "request": to_request(item)}
            if "gold" in item and "questions" not in item:
                ex["gold"] = {"q": ["true", "false"][item["gold"]] if item.get("type") == "noul" else str(item["gold"])}
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser(description="basal decision visualiser (local web page)")
    ap.add_argument("--model", default="Remek/basal-1.0-1.5B", help="local directory or Hugging Face repo id")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, help=f"cuda, mps or cpu (default here: {default_device()})")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route

    t0 = time.time()
    ex = Explainer(a)
    exs = examples()

    async def explain(request):
        try:
            body = await request.json()
            return JSONResponse(await asyncio.get_running_loop().run_in_executor(None, ex.explain, body))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=422)

    app = Starlette(routes=[
        Route("/", lambda _: FileResponse(HERE / "viz.html")),
        Route("/api/info", lambda _: JSONResponse(ex.info)),
        Route("/api/examples", lambda _: JSONResponse(exs)),
        Route("/api/explain", explain, methods=["POST"]),
    ])
    print(f"basal-viz: {ex.name} on {ex.be.dev} ready in {time.time() - t0:.0f}s -> http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
