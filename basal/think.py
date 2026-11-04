"""Think mode (experimental, opt-in): a question whose calibrated confidence is below its
type's certified 5% threshold gets up to --think-tokens reasoning tokens, then a second readout over the state and the
reasoning, with its own calibration file (CALIBRATION.think.json).

Readout 1 is basal-serve's, unchanged (engine, option orders, CALIBRATION.json); with --think off (the default) the
server is basal-serve as without this module. The reasoning and readout 2 run on a vLLM OpenAI server in its own process
(its own Python and GPU memory share), which this module starts and stops, on the think weights: <model>/think/ (a
full model directory), else the model plus a LoRA adapter (<model>/think/adapter), else the model itself (zero-shot).
The reasoning is greedy, stops at </think>, and is written in option order 1, so readout 2 reads order 1 only. With
--deterministic the vLLM requests run one at a time without its prefix cache, so two runs give the same rows.

  basal-serve --model <dir> --think auto [--think-python <python with vllm>]     (or python -m basal.think ...)
  python -m basal.think rows <labelled.jsonl> --url http://127.0.0.1:8000 --out rows.jsonl [--think force]

Request: "think": "auto" (the default when the server runs --think auto), "off", or "force" (every question of a think
type, for calibration readouts); "think_text": true returns the reasoning. A question that thought answers with readout
2's probabilities and confidence plus "think": {"tokens", "ms", "calibration", "no_think": {"probabilities",
"confidence"}}; every answer of a think server carries "meta": {"think": {"used", "tokens"}} (what benchmark harnesses
read). Multi questions, tool sets and requests whose state was truncated never think.
"""
import argparse
import asyncio
import atexit
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import torch

from . import rank, server, tasks
from .engine import resolve
from .facts import inject as inject_facts
from .jsonl import read_lines
from .prompt import PREFILL, RequestError, render
from .server import (Server, act_answer, certified, check_calibration, refuse_lean_flex, temper, temperature,
                     to_items)
from .tools import answer as tool_answer

TYPES = ("choice", "noul", "score", "act", "tool")  # single-readout types
OPEN, CLOSE = "<think>\n", "</think>"
MAX_ERROR = 0.05  # a question thinks below its type's certified threshold for this max_error


class Sidecar:
    """vLLM's OpenAI server in its own process: greedy reasoning, then the readout over the shown codes (processed
    log-probabilities after allowed_token_ids, so the softmax over the codes equals the readout of the other engines)."""

    def __init__(self, model, adapter=None, python=None, mem=0.3, max_len=8192, max_options=255, deterministic=False,
                 url=None, log="basal-think-vllm.log"):
        self.name = "think" if adapter else "base"
        self.max_len, self.proc, self.http, self.log_file = max_len, None, None, None
        self.lock = asyncio.Lock() if deterministic else None  # --deterministic: one vLLM request at a time
        if url:  # a running server: --served-model-name base, and --lora-modules think=<adapter> with an adapter
            self.url = url.rstrip("/")
            return
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        cmd = [python or sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
               "--served-model-name", "base", "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
               "--gpu-memory-utilization", str(mem), "--max-model-len", str(max_len), "--seed", "0",
               "--generation-config", "vllm",  # no sampling defaults from the model's generation_config.json
               "--logprobs-mode", "processed_logprobs", "--max-logprobs", str(max(20, max_options)),
               "--uvicorn-log-level", "warning"]
        if adapter:
            cmd += ["--enable-lora", "--max-lora-rank", "32", "--max-loras", "1", "--lora-modules", f"think={adapter}"]
        if deterministic:
            cmd += ["--no-enable-prefix-caching"]
        self.log = log
        # its own Python: none of the server's PYTHONPATH (an overlay built for another Python breaks vLLM's imports);
        # same process group as the server: stopping the server's group stops vLLM too
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        self.log_file = open(log, "ab")  # (vLLM's output; closed with the sidecar)
        self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, env=env)
        atexit.register(self.close)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log_file is not None:
            self.log_file.close()

    def wait_ready(self, timeout=1200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                tail = Path(self.log).read_text(errors="replace", encoding="utf-8").splitlines()[-15:]
                raise SystemExit(f"basal think: the vLLM server exited ({self.proc.returncode}):\n" + "\n".join(tail))
            try:
                if httpx.get(self.url + "/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise SystemExit(f"basal think: the vLLM server at {self.url} was not ready in {timeout}s (log: {self.log})")

    async def complete(self, body):
        if self.http is None:
            self.http = httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=512))
        body = dict(body, model=self.name, temperature=0.0, seed=0)
        if self.lock:
            async with self.lock:
                r = await self.http.post(self.url + "/v1/completions", json=body)
        else:
            r = await self.http.post(self.url + "/v1/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"think server: {r.status_code} {r.text[:300]}")
        d = r.json()
        return d["choices"][0], d["usage"]

    async def reason(self, ids, max_tokens):
        """Prompt token ids (ending in <think>) -> (reasoning text without </think>, generated tokens)."""
        c, u = await self.complete({"prompt": ids, "max_tokens": max_tokens, "stop": [CLOSE]})
        return c["text"], int(u["completion_tokens"])

    async def readout(self, ids, codes):
        """Prompt token ids (ending in the answer prefill) -> softmax over the code ids (a float32 tensor)."""
        c, _ = await self.complete({"prompt": ids, "max_tokens": 1, "logprobs": len(codes), "allowed_token_ids": codes,
                                    "return_tokens_as_token_ids": True})
        top = c["logprobs"]["top_logprobs"][0]  # {"token_id:<id>": log-probability}
        lp = {int(k.rsplit(":", 1)[1]): v for k, v in top.items()}
        return torch.softmax(torch.tensor([lp.get(i, -1e9) for i in codes], dtype=torch.float32), -1)


def think_calibration(md, path):
    """--think-calibration: a file, "off" (readout 2 uncalibrated), else <model>/CALIBRATION.think.json if shipped ->
    (calibration dict, its name for the answers)."""
    if path == "off":
        return {}, "off"
    p = Path(path) if path else Path(md) / "CALIBRATION.think.json"
    if not p.exists():
        if path:
            raise SystemExit(f"--think-calibration {path}: no such file")
        print(f"basal think: warning: no {p.name} in the model; readout 2 is served uncalibrated", flush=True)
        return {}, "missing"
    return json.loads(p.read_text(encoding="utf-8")), p.name


def answer(q, pl, cal):
    """Readout 2's probabilities (canonical order) -> the answer of the question's type, as Server.decide builds it."""
    probs = {k: float(v) for k, v in zip(q["keys"], pl)}
    conf = float(max(pl))
    if q["type"] == "act":
        return act_answer(q, probs, conf, cal)
    if q["type"] == "tool":
        return tool_answer(q, probs, conf, cal)
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"], "probabilities": probs, "confidence": conf}
    if q["type"] == "score":
        return {"type": "score", "score": float(sum(i * v for i, v in enumerate(pl))),
                "legend": dict(zip(q["keys"], q["options"])), "probabilities": probs, "confidence": conf}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}


class ThinkServer(Server):
    def __init__(self, a):
        self.think = a.think
        if a.think == "off":
            super().__init__(a)
            return
        refuse_lean_flex(a.lean, a.tree)  # before the model is resolved and the sidecar started
        md = resolve(a.model, a.revision)
        weights = a.think_model or (md / "think" if (md / "think" / "config.json").exists() else md)
        adapter = a.think_adapter or (md / "think" / "adapter" if (md / "think" / "adapter").exists() else None)
        # started (and ready) before the engine loads, so vLLM's memory profile does not see the engine's allocations
        self.sidecar = Sidecar(weights, adapter, a.think_python, a.think_memory, a.think_max_len,
                               deterministic=a.deterministic, url=a.think_url, log=a.think_log)
        self.sidecar.wait_ready()
        super().__init__(a)
        self.think_cal, self.think_cal_name = think_calibration(md, a.think_calibration)
        if self.think_cal:
            check_calibration(self.think_cal, self.fmt, self.think_cal_name)
        self.think_tokens = a.think_tokens
        self.think_log, self.log = self.log, None  # the propensity log records the served answers, after think
        print(f"basal think: auto below the certified {MAX_ERROR:.0%} thresholds, <= {a.think_tokens} tokens, "
              f"weights {weights}{f' + adapter {adapter}' if adapter else ''}, readout 2 calibration "
              f"{self.think_cal_name}", flush=True)

    def routed(self, q, ans, mode):
        if q["type"] not in TYPES or q.get("multi") or len(q["branches"]) != 1:
            return False
        if mode == "force":
            return True
        thr = certified(self.cal, MAX_ERROR, q.get("base", q["type"]))
        return thr is None or ans["confidence"] < thr

    async def think_one(self, q):
        """One question -> (readout 2's calibrated probabilities, reasoning, tokens, ms), or (None, why)."""
        t0 = time.perf_counter()
        text = render(self.tok, q["state"], q["question"], q["options"], q["lang"], self.fmt)
        head = text[: -len(PREFILL)] + OPEN
        ids = self.tok(head, add_special_tokens=False).input_ids
        if len(ids) + self.think_tokens + 16 > self.sidecar.max_len:
            return None, "context"
        notes, n = await self.sidecar.reason(ids, self.think_tokens)
        prompt = head + notes.strip() + "\n" + CLOSE + "\n" + PREFILL
        codes = self.code_ids(prompt, len(q["options"]))
        p = await self.sidecar.readout(self.tok(prompt, add_special_tokens=False).input_ids, codes)
        pl = temper(p, temperature(self.think_cal, q.get("base", q["type"]), len(p))).tolist()
        return (pl, notes, n, round((time.perf_counter() - t0) * 1000, 2)), None

    async def decide(self, body):
        mode = body.get("think", self.think)
        if mode not in ("off", "auto", "force"):
            raise RequestError('think must be "off", "auto" or "force"')
        out = await super().decide(body)
        if self.think == "off":
            return out
        t0 = time.perf_counter()
        todo = []
        if mode != "off" and not out["usage"].get("truncated"):
            state = body["state"]
            if body.get("facts") == "auto":  # the state readout 1 saw (off the event loop, as basal-serve reads it)
                text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                state = await asyncio.to_thread(inject_facts, text)
            # the plain questions answered (not tasks or rank, which are answered from several decisions, nor a
            # question ask_if skipped: null)
            plain = {n: q for n, q in body["questions"].items()
                     if isinstance(q, dict) and not tasks.is_task(q) and not rank.is_rank(q)}
            todo = [q for q in to_items(state, plain, self.max_options, force_lang=body.get("lang"))
                    if out["answers"].get(q["name"]) is not None and self.routed(q, out["answers"][q["name"]], mode)]
        n_think = 0
        for q, (res, why) in zip(todo, await asyncio.gather(*(self.think_one(q) for q in todo))):
            old = out["answers"][q["name"]]
            if res is None:
                old["think"] = {"skipped": why}
                continue
            pl, notes, n, ms = res
            ans = answer(q, pl, self.think_cal)
            ans["think"] = {"tokens": n, "ms": ms, "calibration": self.think_cal_name,
                            "no_think": {"probabilities": old["probabilities"], "confidence": old["confidence"]}}
            if body.get("think_text"):
                ans["think"]["text"] = notes
            for k in ("evidence",):
                if k in old:
                    ans[k] = old[k]
            out["answers"][q["name"]] = ans
            out["usage"]["output_tokens"] += n
            n_think += 1
        out["usage"]["think_questions"] = n_think
        for a in out["answers"].values():  # every answer of a think server says whether it used the reasoning
            if not isinstance(a, dict):  # (null: skipped by ask_if)
                continue
            th = a.get("think") or {}
            a.setdefault("meta", {})["think"] = {"used": "tokens" in th, "tokens": th.get("tokens", 0)}
        out["usage"]["latency_ms"] = round(out["usage"]["latency_ms"] + (time.perf_counter() - t0) * 1000, 2)
        if self.think_log:
            self.log = self.think_log
            self.log_decision(body, out)
            self.log = None
        return out


def add_think_args(ap):
    g = ap.add_argument_group("think mode (experimental, opt-in)")
    g.add_argument("--think", choices=["off", "auto"], default="off",
                   help="auto: questions below their type's certified 5%% threshold reason, then are read out again")
    g.add_argument("--think-tokens", dest="think_tokens", type=int, default=256, help="most reasoning tokens")
    g.add_argument("--think-calibration", dest="think_calibration", default=None,
                   help='readout 2\'s calibration file, or "off" (default: <model>/CALIBRATION.think.json)')
    g.add_argument("--think-model", dest="think_model", default=None,
                   help="think weights (default: <model>/think/ when shipped, else --model)")
    g.add_argument("--think-adapter", dest="think_adapter", default=None,
                   help="a LoRA adapter on the think weights (default: <model>/think/adapter when shipped)")
    g.add_argument("--think-python", dest="think_python", default=None,
                   help="Python with vLLM for the reasoning server (default: this one)")
    g.add_argument("--think-url", dest="think_url", default=None,
                   help="a running vLLM OpenAI server instead of starting one (served as 'base', the adapter as 'think')")
    g.add_argument("--think-memory", dest="think_memory", type=float, default=0.3, help="vLLM GPU memory fraction")
    g.add_argument("--think-max-len", dest="think_max_len", type=int, default=8192, help="vLLM context")
    g.add_argument("--think-log", dest="think_log", default="basal-think-vllm.log", help="the vLLM server's log")


# -- labelled rows of a running server ------------------------------------------------------------------------------
def summary(rows):
    """Think share, accuracy and ECE with / without think (all rows and the rows that thought), tokens, latency."""
    import numpy as np

    from .calibrate import ece

    def acc(rs, key):
        return round(float(np.mean([int(np.argmax(r[key])) == r["gold"] for r in rs])), 4) if rs else None

    def cal(rs, key):
        e = ece([(max(r[key]), int(np.argmax(r[key])) == r["gold"]) for r in rs])
        return round(e, 4) if e is not None else None

    th = [r for r in rows if r.get("think")]
    lat = lambda rs, q: round(float(np.percentile([r["latency_ms"] for r in rs], q)), 1) if rs else None
    tok = [r["think"]["tokens"] for r in th]
    return dict(n=len(rows), think=len(th), think_share=round(len(th) / max(1, len(rows)), 4),
                acc=acc(rows, "p"), acc_no_think=acc(rows, "p0"), ece=cal(rows, "p"), ece_no_think=cal(rows, "p0"),
                routed_acc=acc(th, "p"), routed_acc_no_think=acc(th, "p0"),
                routed_conf=round(float(np.mean([max(r["p"]) for r in th])), 4) if th else None,
                routed_conf_no_think=round(float(np.mean([max(r["p0"]) for r in th])), 4) if th else None,
                tokens_mean=round(float(np.mean(tok)), 1) if tok else None,
                tokens_p95=round(float(np.percentile(tok, 95)), 1) if tok else None, tokens_max=max(tok, default=None),
                latency_p50=lat(rows, 50), latency_p95=lat(rows, 95), routed_latency_p50=lat(th, 50),
                routed_latency_p95=lat(th, 95))


def rows_main(argv):
    from .calibrate import load, over_http
    ap = argparse.ArgumentParser(prog="python -m basal.think rows", description="labelled rows of a think server")
    ap.add_argument("data", help="labelled JSONL (basal-calibrate's input; simple items need an id)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", choices=["auto", "off", "force"], default="auto")
    ap.add_argument("--text", action="store_true", help="keep the reasoning in the rows")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args(argv)
    bodies, decisions = load(a.data)
    lines = read_lines(a.data)
    for b in bodies:
        b["think"] = a.think
        if a.text:
            b["think_text"] = True
    t0 = time.time()
    responses = asyncio.run(over_http(a.url.rstrip("/") + "/v1/basal", bodies, a.concurrency))
    rows, failed = [], 0
    for d in decisions:
        r = responses[d["req"]]
        if isinstance(r, Exception) or d["prim"] == "multi":
            failed += isinstance(r, Exception)
            continue
        ans = r["answers"][d["name"]]
        th = ans.get("think") if "tokens" in ans.get("think", {}) else None
        p = [float(ans["probabilities"][k]) for k in d["keys"]]
        p0 = [float(th["no_think"]["probabilities"][k]) for k in d["keys"]] if th else p
        row = dict(id=json.loads(lines[d["line"] - 1]).get("id"), family=d["family"], group=d["group"], prim=d["prim"],
                   gold=d["gold"], k=len(p), p=p, p0=p0, latency_ms=r["usage"]["latency_ms"])
        if th:
            row["think"] = {k: v for k, v in th.items() if k != "no_think"}
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    s = dict(summary(rows), failed=failed, wall_s=round(time.time() - t0, 1), think_mode=a.think, data=a.data)
    print(json.dumps(s))
    return s


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rows":
        rows_main(sys.argv[2:])
        return
    server.main()  # basal-serve (its --think flag)


if __name__ == "__main__":
    main()
