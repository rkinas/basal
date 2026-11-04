"""Local visualiser: every step of a typed decision, from the rendered prompt to the calibrated answer.

  basal-viz --model Remek/basal-2.0-mini --port 8080      then open http://127.0.0.1:8080

  basal-viz --model Remek/basal-2.0-mini-MLX-8bit --mode mlx      Apple Silicon: the MLX engine (mlx-q8 too)

The page runs a basal-serve Server in this process (--mode mps on Apple Silicon, eager elsewhere, or --mode mlx /
mlx-q8) and reads the layers of the model it serves.
POST /api/explain takes a /v1/basal request body and returns, per question (per label for multi): the prompt of each
option order the server asks (split into template / state / question / options / prefill; prompt format and option
codes from the model's basal.json, one order by default for format-2 models), the shared-prefix packing of the orders,
the code probabilities of each order, the per-layer readout ("logit lens": each layer's hidden state at the answer
position through the final norm and the output head, restricted to the option codes), the order average, the
calibrated distribution and the decision against the certified thresholds of the calibration file basal-serve loads.
The request is also answered by the server itself (Server.decide), to show that the page explains the served answer.
"""
import argparse
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import torch

from .engine import EagerBackend, default_device
from .facts import inject as inject_facts
from .gemma4 import is_gemma4, text_model
from .jsonl import read_lines
from .prompt import FORMATS, PREFILL, RequestError
from .run import to_request
from .server import Server, certified, http_status, temperature, to_items
from .server import parser as serve_parser

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"
TARGETS = ("0.01", "0.05")  # max_error targets the page compares the confidence with


def segments(prompt, state, question, options, fmt=1):
    """Split a rendered prompt into labelled pieces (the template text between them is "template")."""
    codes = FORMATS[fmt]["codes"]
    marks, at = [], 0
    for kind, text in [("state", state), ("question", question)] + [
            ("option", f"{codes[k]}. {o}") for k, o in enumerate(options)]:
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


MLX_MODES = ("mlx", "mlx-q8")


def lens_parts(model):
    """(decoder, output head, final logit softcap) the PyTorch lens reads: the Llama model, or the text decoder of a
    Gemma 4 basal-omni model with its head (bias) and softcap."""
    if not is_gemma4(model):
        return model.model, model.lm_head, None
    tm = text_model(model)
    return tm, model.get_output_embeddings(), getattr(tm.config, "final_logit_softcapping", None)


@torch.no_grad()
def torch_lens(model, dev, ids, codes, parts=None):
    """Forward with a hook on every decoder layer -> code distribution after each layer (fp32 code logits, as the
    engines read them; the head's bias and the final logit softcap where the model has them) and the top 5 tokens
    (ids, probabilities) of the full vocabulary at the last layer. parts: lens_parts(model)."""
    mm, head, softcap = parts or lens_parts(model)
    caught = []
    hooks = [layer.register_forward_hook(
        lambda _m, _i, o: caught.append((o[0] if isinstance(o, tuple) else o)[0, -1].clone()))
             for layer in mm.layers]
    try:
        final = mm(input_ids=torch.tensor([ids], device=dev), use_cache=False,
                   output_hidden_states=False).last_hidden_state[0, -1]
    finally:
        for h in hooks:
            h.remove()
    H = torch.stack([mm.norm(h[None])[0] for h in caught[:-1]] + [final])
    W = head.weight
    logits = H.float() @ W[codes].float().T
    top_logits = H[-1] @ W.T
    if head.bias is not None:
        logits = logits + head.bias[codes].float()
        top_logits = top_logits + head.bias.float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
        top_logits = torch.tanh(top_logits.float() / softcap) * softcap
    per_layer = torch.softmax(logits, -1)
    top = torch.log_softmax(top_logits.float(), -1).topk(5)  # what the model would write without the codes
    return per_layer.tolist(), [(int(i), float(p.exp())) for p, i in zip(top.values, top.indices)]


class _Catch:
    """An MLX decoder layer that also keeps its output at the last position (attributes pass through to the layer)."""

    def __init__(self, layer, out):
        self.layer, self.out = layer, out

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kw):
        h = self.layer(*args, **kw)
        self.out.append(h[0, -1])
        return h


def mlx_lens(be, ids, codes):
    """torch_lens on the MLX backend that serves (--mode mlx / mlx-q8), on its thread: each decoder layer's output at
    the answer position through the final norm, read as the backend reads the codes (MLXBackend._letters: fp32 code
    rows, or the quantised head itself)."""
    def run():
        mx, mm, caught = be.mx, be.lm.model, []
        layers = mm.layers
        mm.layers = [_Catch(layer, caught) for layer in layers]
        try:
            final = mm(mx.array([ids]))[0, -1]
        finally:
            mm.layers = layers
        H = mx.stack([mm.norm(h) for h in caught[:-1]] + [final])
        per_layer = be._letters(H, [codes] * len(caught))
        z = be._head(H[-1:])[0].astype(mx.float32)
        if be.softcap:
            z = mx.tanh(z / be.softcap) * be.softcap
        top = mx.argsort(-z)[:5]
        p = mx.exp(z[top] - mx.logsumexp(z))
        return per_layer, list(zip(top.tolist(), p.tolist()))
    return be.pool.submit(run).result()


def packing(toks):
    """Tokens of the option orders' prompts: shared prefix (computed once in the packed row), each order's own tail,
    and the tokens of separate prompts."""
    prefix = len(os.path.commonprefix(toks)) if len(toks) > 1 else 0
    return dict(prefix=prefix, tails=[len(t) - prefix for t in toks], separate=sum(map(len, toks)))


class Explainer:
    mlx = False  # --mode mlx / mlx-q8: the MLX engine's layers (mlx_lens), else the PyTorch model's (torch_lens)

    def __init__(self, a):
        mode = getattr(a, "mode", None)
        if mode in MLX_MODES:  # the MLX engine of basal-serve: the page reads its layers
            if a.device:
                raise SystemExit(f"--device is for the PyTorch modes; --mode {mode} runs on the Mac's GPU")
            args = ["--model", a.model, "--mode", mode]
        else:
            dev = a.device or default_device()
            # the in-process PyTorch engines: the page reads the model's layers
            mode = "mps" if dev == "mps" else "eager"
            args = ["--model", a.model, "--mode", mode, "--dtype", a.dtype]
            args += ["--device", dev] if mode == "eager" else []
        args += ["--orders", str(a.orders)] if a.orders else []
        args += ["--calibration", a.calibration] if a.calibration else []
        self.srv = Server(serve_parser().parse_args(args))
        be = self.srv.backend
        self.mlx = mode in MLX_MODES
        self.tok, self.fmt = be.tok, self.srv.fmt
        if self.mlx:
            q = be.quantization
            self.model, self.dev, self.gemma4 = None, "mlx", False
            arch, n_layers = be.config.get("model_type"), len(be.lm.model.layers)
            # "8-bit"; "4-bit nvfp4" for an -MLX-fp4 export (quantisation mode other than affine); bf16 weights
            dtype = (f"{q['bits']}-bit" + (f" {q['mode']}" if q.get("mode", "affine") != "affine" else "") if q
                     else str(be.lm.model.embed_tokens.weight.dtype).rsplit(".", 1)[-1])
            if arch != "llama":
                raise SystemExit(f"basal-viz --mode {mode} reads the layers of the Llama-architecture models (basal-1.0, "
                                 f"basal-1.5, basal-2.0); {a.model} is {arch!r} (Gemma 4 basal-omni: the PyTorch "
                                 "modes)")
        else:
            self.model, self.dev, dtype = be.model, be.dev, a.dtype
            self.gemma4 = is_gemma4(self.model)
            if not self.gemma4 and getattr(self.model.config, "model_type", None) != "llama":
                raise SystemExit(f"basal-viz reads the layers of the Llama-architecture models (basal-1.0, basal-1.5, "
                                 f"basal-2.0) and Gemma 4 basal-omni; {a.model} is {self.model.config.model_type!r}")
            self.text_model, self.head, self.softcap = lens_parts(self.model)
            n_layers = len(self.text_model.layers)
        self.name = self.srv.name
        self.lock = threading.Lock()
        cal = self.srv.cal
        self.thresholds = {t: {q: certified(cal, float(t), q) for q in ("choice", "noul", "score")} for t in TARGETS}
        self.info = {"model": self.name, "device": str(self.dev), "dtype": dtype, "mode": mode,
                     "layers": n_layers, "prompt_format": self.fmt, "orders": self.srv.orders,
                     "thresholds": self.thresholds, "temperatures": cal.get("temperature_per_prim", {}),
                     "temperature_per_bucket": cal.get("temperature_per_bucket", {})}

    def lens(self, ids, codes):
        """Code distribution after each layer and the top tokens of the full vocabulary at the last layer."""
        if self.mlx:
            per_layer, top = mlx_lens(self.srv.backend, ids, codes)
        else:
            # shortcut: full-sequence Gemma lens is limited to one eager chunk; use chunked capture for longer prompts.
            if self.gemma4 and len(ids) > EagerBackend.CHUNK:
                raise RequestError(f"Gemma 4 inspection supports at most {EagerBackend.CHUNK} prompt tokens; "
                                   "shorten the state or question")
            per_layer, top = torch_lens(self.model, self.dev, ids, codes, (self.text_model, self.head, self.softcap))
        return per_layer, [(self.tok.decode([i]), p) for i, p in top]

    def explain_branch(self, q, bi):
        question, options = q["branches"][bi]
        k = len(options)
        t = "noul" if q["type"] == "multi" or q.get("multi") else q.get("base", q["type"])
        t0 = time.perf_counter()
        jobs = self.srv.jobs_for(q, question, options)  # (perm, prompt, code ids) per option order, as served
        if any(ids is None for _, _, ids in jobs):
            raise ValueError("option codes are not single tokens for this tokenizer")
        codes = FORMATS[self.fmt]["codes"]
        orders, toks = [], []
        for perm, prompt, cids in jobs:
            ids = self.tok(prompt, add_special_tokens=False).input_ids
            toks.append(ids)
            opts = [options[c] for c in perm]
            orders.append(dict(perm=perm, prompt=prompt, segments=segments(prompt, q["state"], question, opts, self.fmt),
                               n_tokens=len(ids), letters=list(codes[:k]), letter_ids=cids))
        t_tok = time.perf_counter() - t0
        # the lens runs the whole prompt without truncation: a prompt over the context (the model's, or the engine's
        # own limit, as basal-serve checks it) is refused before any forward
        limit, n = getattr(self.srv, "context", None), max(map(len, toks))
        if limit and n > limit[0]:
            raise RequestError(f"basal-viz explains prompts within {limit[1]} ({limit[0]} tokens); this prompt has "
                               f"{n} (basal-serve would truncate its state)")

        t0 = time.perf_counter()
        for o, ids in zip(orders, toks):
            layers, top = self.lens(ids, o["letter_ids"])
            o["letter_probs"], o["top_tokens"] = layers[-1], top
            canon = lambda p, perm=o["perm"]: [p[perm.index(j)] for j in range(k)]  # position -> option index
            o["canon"] = canon(o["letter_probs"])
            o["layers_canon"] = [canon(p) for p in layers]
        t_sep = time.perf_counter() - t0

        t0 = time.perf_counter()
        packed = self.srv.backend.run_shared([(toks, [o["letter_ids"] for o in orders])])[0]
        t_packed = time.perf_counter() - t0
        diff = max(abs(a - b) for o, p in zip(orders, packed) for a, b in zip(o["letter_probs"], p))

        mean = [sum(o["canon"][j] for o in orders) / len(orders) for j in range(k)]
        T = temperature(self.srv.cal, t, k)
        cal = self.srv.canonical(t, [o["perm"] for o in orders], [o["letter_probs"] for o in orders])
        conf = max(cal)
        best = cal.index(conf)
        multi = q["type"] == "multi" or q.get("multi")
        keys = ["true", "false"] if multi else q["keys"]
        answer = {"index": best, "key": keys[best], "confidence": conf}
        if t == "noul":
            answer["noul"] = cal[0]
        elif t == "score":
            answer["score"] = sum(i * v for i, v in enumerate(cal))
        n_layers = len(orders[0]["layers_canon"])
        label = q["labels"][bi] if multi and q.get("labels") else None
        return dict(
            name=q["name"] + (f" · {label}" if label else ""), qname=q["name"], label_key=q["keys"][bi] if multi else None,
            type=t, kind=q["type"], lang=q["lang"], keys=keys, options=list(options), question=question, orders=orders,
            packing=packing(toks),
            layers=[[sum(o["layers_canon"][L][j] for o in orders) / len(orders) for j in range(k)] for L in range(n_layers)],
            mean=mean, temperature=T, calibrated=cal, answer=answer,
            thresholds={target: self.thresholds[target].get(t) for target in TARGETS}, packed_max_diff=diff,
            timing_ms=dict(tokenize=t_tok * 1000, separate_with_lens=t_sep * 1000, packed=t_packed * 1000))

    def explain(self, body):
        """The page's steps for every question of a request (one entry per label of a multi question); with
        "facts": "auto" the state carries the computed facts, as basal-serve reads it. Text only: a request with media
        is refused."""
        if body.get("facts", "off") not in ("off", "auto"):
            raise RequestError('facts must be "off" or "auto"')
        if body.get("media"):
            raise ValueError("basal-viz reads text only: remove the media to inspect this request")
        with self.lock:
            t0 = time.perf_counter()
            state = body["state"]
            if body.get("facts") == "auto":
                state = inject_facts(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
            items = to_items(state, body["questions"], self.srv.max_options, force_lang=body.get("lang"))
            qs = [self.explain_branch(q, bi) for q in items for bi in range(len(q["branches"]))]
            return {"model": self.name, "questions": qs, "total_ms": (time.perf_counter() - t0) * 1000}


def compare_served(res, served):
    """Largest difference between each explained distribution and the probabilities basal-serve returns."""
    for q in res["questions"]:
        a = served["answers"].get(q["qname"], {}).get("probabilities") or {}
        if q["label_key"] is not None:
            ours = {q["label_key"]: q["calibrated"][0]}  # multi: P(yes) per label
        else:
            ours = dict(zip(q["keys"], q["calibrated"]))
        q["served_max_diff"] = max((abs(v - a[k]) for k, v in ours.items() if k in a), default=None)
    res["served"] = served["answers"]
    return res


def examples():
    out = []
    for f in sorted(EXAMPLES.glob("*.jsonl")):
        for n, line in enumerate(read_lines(f)):
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
    ap.add_argument("--model", default="Remek/basal-2.0-mini", help="local directory or Hugging Face repo id")
    ap.add_argument("--mode", default=None, choices=MLX_MODES,
                    help="Apple Silicon: the MLX engine of basal-serve (the -MLX-8bit / -MLX-fp4 exports or bf16 "
                         "weights; mlx-q8: 8-bit weights made at load time); default: the PyTorch engine of --device")
    ap.add_argument("--dtype", default="bfloat16", help="the PyTorch modes")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help=f"default here: {default_device()} (mps runs --mode mps, cuda and cpu --mode eager)")
    ap.add_argument("--orders", type=int, choices=[1, 2], default=None,
                    help="option orders per question (default: the model's, 1 for prompt-format-2 models)")
    ap.add_argument("--calibration", default=None, help="a calibration file instead of the model's (as basal-serve)")
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
    # The lens hooks and served comparison share one model, including on the MPS backend.
    inspection_lock = asyncio.Lock()

    async def explain(request):
        try:
            body = await request.json()
            async with inspection_lock:
                res = await asyncio.get_running_loop().run_in_executor(None, ex.explain, body)
                return JSONResponse(compare_served(res, await ex.srv.decide(body)))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=http_status(e))

    async def lifespan(_app):
        task = asyncio.get_running_loop().create_task(ex.srv.worker())  # the batching loop of basal-serve
        yield
        task.cancel()

    app = Starlette(routes=[
        Route("/", lambda _: FileResponse(HERE / "viz.html")),
        Route("/api/info", lambda _: JSONResponse(ex.info)),
        Route("/api/examples", lambda _: JSONResponse(exs)),
        Route("/api/explain", explain, methods=["POST"]),
    ], lifespan=lifespan)
    print(f"basal-viz: {ex.name} on {ex.dev} ready in {time.time() - t0:.0f}s -> http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
