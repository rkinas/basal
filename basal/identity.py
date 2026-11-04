"""Identity gate of the fast path: every prefix-reuse path of the basal engine against the eager per-prompt reference.

  basal-bench --identity --model Remek/basal-1.5-4.5B                       # fp32: the paths' exactness
  basal-bench --identity --model Remek/basal-1.5-4.5B --dtype bfloat16      # as served
  basal-bench --identity --model Remek/basal-1.5-4.5B --dtype bfloat16 --quant fp8

Items are synthetic (basal.synth, no real data): single questions over ~1.5k-token states, requests with five
questions over one state, and long states (4-8k tokens); every question is rendered in both option orders. The
reference runs every prompt alone in an eager forward (no padding, no mask, no graph) and reads the option codes from
the full log-softmax. The paths, each through the engine's normal entry point (run_shared):

  plain          single prompts, one call each (plain CUDA graphs, padded to the length bucket)
  plain-batch    16 single prompts of different requests per call (cross-request batching)
  orders2        both orders of a question as one packed row (masked graph), one call each
  orders2-batch  16 questions per call
  soam           every question and order of a request over one state (one packed row), one call each
  soam-batch     8 requests per call
  long-plain     long single prompts (plain graphs up to 8192 tokens, an eager forward beyond)
  long-tree      long states, both orders (two passes over the root's KV cache)

Per path: readouts, argmax agreement with the reference, max and p99 |dp| over all option probabilities, how many of
the disagreements are near ties (top two within 0.02 in the reference) and, for the single questions, the agreement of
the decisions (both orders averaged, as served; also for the --questions items alone). Gate: agreement >= 0.999.
In bfloat16 any change of tensor shapes changes the rounding, so near ties can flip on any path, the plain one
included; fp32 shows whether a path is exact, and --reference plain compares every path with each prompt computed
alone in the same engine.

--questions FILE adds your own simple items (state / question / options) to the single questions; --against MODEL
takes the reference forwards from another model with the same tokenizer (the bf16 weights of an FP8 checkpoint), so
the agreement then measures the quantisation too.

The same check for every backend against the native engine, with question independence and bootstrap intervals, is
basal-agree (basal/agree.py), which also runs this gate: basal-agree --identity.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from .agree import QUESTIONS, build, compare, decisions, orders  # noqa: F401 (shared with basal-agree)
from .engine import ATTN, GraphBackend, resolve
from .jsonl import read_lines
from .server import default_orders, lean_arg, merge_arg
from .prompt import model_format

GATE = 0.999


@torch.no_grad()
def reference(be, toks, ids):
    if getattr(be, "gemma", None) is not None:
        return reference_gemma(be, toks, ids)
    h = be.model.model(input_ids=torch.tensor([toks], device=be.dev), use_cache=False).last_hidden_state[0, -1]
    lp = torch.log_softmax(be.model.lm_head(h).float(), -1)
    return torch.softmax(lp[ids], -1).tolist()


@torch.no_grad()
def reference_gemma(be, toks, ids):
    """Gemma 4 (basal-omni): the transformers forward of the decoder on SDPA's math kernel, as --mode eager runs it, and
    the code logits as eager reads them (fp32 from the head's input, the final logit softcap in fp32; a softmax over
    the code rows equals the softmax over the codes of the full log-softmax)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.MATH]):
        h = be.gemma.lm(input_ids=torch.tensor([toks], device=be.dev), use_cache=False).last_hidden_state[0, -1]
    z = h.float() @ be.model.lm_head.weight[ids].float().T
    if be.softcap:
        z = torch.tanh(z / be.softcap) * be.softcap
    return torch.softmax(z, -1).tolist()


def run_path(be, groups, per_call):
    out = []
    for i in range(0, len(groups), per_call):
        out += be.run_shared(groups[i: i + per_call])
    torch.cuda.synchronize()
    return [p for g in out for p in g]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Remek/basal-1.5-4.5B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--quant", default=None, choices=["fp8", "nvfp4"])
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--n-single", dest="n_single", type=int, default=200)
    ap.add_argument("--n-multi", dest="n_multi", type=int, default=60)
    ap.add_argument("--n-long", dest="n_long", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", default=None,
                    help="JSONL of simple items {state, question, options} added to the single questions")
    ap.add_argument("--n-questions", dest="n_questions", type=int, default=500)
    ap.add_argument("--reference", default="eager", choices=["eager", "plain"],
                    help="eager: every prompt in an eager forward (default); plain: every prompt alone through the "
                         "engine's plain graphs, so in bf16 the gate isolates what prefix reuse and batching change")
    ap.add_argument("--against", default=None,
                    help="another model (e.g. the bf16 weights of an FP8 checkpoint): the reference forwards use it, "
                         "so the agreement measures the quantisation as well as the paths")
    ap.add_argument("--attn", choices=list(ATTN), default="auto", help="attention kernel of the engine under test")
    ap.add_argument("--reference-attn", dest="reference_attn", choices=list(ATTN), default=None,
                    help="compute the reference with this attention kernel first (e.g. fa2 against --attn fa3), in a "
                         "separate engine that is freed before the engine under test is built")
    ap.add_argument("--merge", choices=["auto", "on", "off"], default="auto",
                    help="merged q/k/v and gate/up in the engine (auto: on for bf16 weights, off for FP8 checkpoints)")
    ap.add_argument("--reference-merge", dest="reference_merge", choices=["on", "off"], default=None,
                    help="the reference engine with (on) or without (off) merged projections; like --reference-attn, "
                         "it is built and freed first")
    ap.add_argument("--lean", choices=["on", "pass2", "off"], default="off",
                    help="the lean layer function in the engine (both passes, or pass 2 only)")
    ap.add_argument("--tree", choices=["dense", "flex"], default="dense", help="pass-2 mask with --lean on")
    ap.add_argument("--reference-lean", dest="reference_lean", choices=["on", "pass2", "off"], default=None,
                    help="the reference engine with or without the lean layer (built and freed first)")
    ap.add_argument("--gemma-attn", dest="gemma_attn", choices=["auto", "math", "flex", "fa3", "fa3+math"],
                    default="auto",
                    help="Gemma 4 models: the attention kernel of the engine under test (basal-serve --gemma-attn)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if (a.reference_merge or a.reference_lean) and not a.reference_attn:
        a.reference_attn = a.attn
    md = resolve(a.model)
    fmt = model_format(md)
    early = None
    if a.reference_attn:  # the reference engine first: the attention kernel is process-wide and fixed at capture
        if a.against:
            raise SystemExit("--reference-attn and --against do not combine")
        t0 = time.time()
        early = [GraphBackend(md, a.dtype, a.quant, compile=a.compile, shared=True, warm=a.reference == "plain",
                              attn=a.reference_attn, merge=merge_arg(a.reference_merge or a.merge),
                              lean=lean_arg(a.reference_lean or a.lean))]
        early.append(round(time.time() - t0, 1))
    t0 = time.time()
    be = None if early else GraphBackend(md, a.dtype, a.quant, compile=a.compile, shared=True, attn=a.attn,
                                         merge=merge_arg(a.merge), lean=lean_arg(a.lean), tree=a.tree,
                                         gemma_attn=a.gemma_attn)
    if be is not None:
        print(json.dumps(dict(model=a.model, dtype=a.dtype, quant=a.quant, prompt_format=fmt, graphs=len(be.graphs),
                              attention=be.attn, fp8_checkpoint=be.fp8_checkpoint,
                              startup_s=round(time.time() - t0, 1))), flush=True)
    ref_be = early[0] if early else be
    if a.against:  # same tokenizer required: the prompts are tokenized once
        ref_be = GraphBackend(resolve(a.against), a.dtype, None, compile=a.compile, shared=False,
                              warm=a.reference == "plain")
        probe = "Stan: zażalenie złożono 3 marca 2026 r. State: 12.50 EUR"
        if ref_be.tok(probe).input_ids != be.tok(probe).input_ids or len(ref_be.tok) != len(be.tok):
            raise SystemExit(f"--against {a.against}: a different tokenizer")
    qs = None
    if a.questions:
        qs = [json.loads(x) for x in read_lines(a.questions) if x.strip()]
        qs = [q for q in qs if "options" in q and 2 <= len(q["options"]) <= (255 if fmt == 2 else 10)][: a.n_questions]
    single, multi, long = build(ref_be.tok, fmt, a.n_single, a.n_multi, a.n_long, a.seed, qs)
    rnd = random.Random(a.seed)
    origin = list(range(len(single)))  # index >= a.n_single: an item of --questions
    rnd.shuffle(origin)
    single = [single[i] for i in origin]
    real = [i >= a.n_single for i in origin]
    one = [([t], [x]) for toks, ids in single for t, x in zip(toks, ids)]
    one_long = [([t], [x]) for toks, ids in long for t, x in zip(toks, ids)]
    # path: (groups, groups per call, data set); the groups of a data set flatten to the same prompts in the same order
    paths = {"plain": (one, 1, "single"), "plain-batch": (one, 16, "single"), "orders2": (single, 1, "single"),
             "orders2-batch": (single, 16, "single"), "soam": (multi, 1, "multi"), "soam-batch": (multi, 8, "multi"),
             "long-plain": (one_long, 1, "long"), "long-tree": (long, 1, "long")}
    t0 = time.time()
    if a.reference == "plain":  # each prompt alone through the plain graphs (no prefix reuse, no batching)
        refs = {k: run_path(ref_be, [([t], [x]) for toks, ids in groups for t, x in zip(toks, ids)], 1)
                for k, groups in (("single", single), ("multi", multi), ("long", long))}
    else:
        refs = {k: [reference(ref_be, t, x) for toks, ids in groups for t, x in zip(toks, ids)]
                for k, groups in (("single", single), ("multi", multi), ("long", long))}
    if early:  # free the reference engine, then build the one under test with its attention kernel
        print(json.dumps(dict(reference_attention=ref_be.attn, reference_startup_s=early[1])), flush=True)
        del ref_be, early
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        be = GraphBackend(md, a.dtype, a.quant, compile=a.compile, shared=True, attn=a.attn, merge=merge_arg(a.merge),
                          lean=lean_arg(a.lean), tree=a.tree, gemma_attn=a.gemma_attn)
        print(json.dumps(dict(model=a.model, dtype=a.dtype, quant=a.quant, prompt_format=fmt, graphs=len(be.graphs),
                              attention=be.attn, fp8_checkpoint=be.fp8_checkpoint,
                              startup_s=round(time.time() - t0, 1))), flush=True)
    lens = [len(t) for toks, _ in single for t in toks]
    long_lens = [len(t) for toks, _ in long for t in toks]
    print(json.dumps(dict(reference_s=round(time.time() - t0, 1), prompt_tokens_p50=statistics.median(lens),
                          long_tokens=[min(long_lens), max(long_lens)])), flush=True)
    rows = []
    for name, (groups, per_call, data) in paths.items():
        got = run_path(be, groups, per_call)
        r = dict(path=name, per_call=per_call, **compare(refs[data], got))
        if data == "single":  # decisions in the canonical order, as the server answers by default (default_orders)
            ref_d, got_d = decisions(refs[data], default_orders(None, fmt)), decisions(got, default_orders(None, fmt))
            r["decision_agreement"] = compare(ref_d, got_d)["agreement"]
            if any(real):
                r["decision_agreement_questions"] = compare([x for x, q in zip(ref_d, real) if q],
                                                            [x for x, q in zip(got_d, real) if q])["agreement"]
        r["gate"] = "pass" if r["agreement"] >= GATE else "FAIL"
        rows.append(r)
        print(json.dumps({k: round(v, 7) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    print(f"\n{'path':<15}{'readouts':>9}{'agree':>9}{'max|dp|':>10}{'p99|dp|':>10}{'ties':>7}{'decisions':>10}  gate "
          f"({a.dtype}{', ' + a.quant if a.quant else ''})")
    for r in rows:
        d = f"{r['decision_agreement']:.4f}" if "decision_agreement" in r else "–"
        print(f"{r['path']:<15}{r['readouts']:>9}{r['agreement']:>9.4f}{r['max_dp']:>10.2e}{r['p99_dp']:>10.2e}"
              f"{str(r['near_ties']) + '/' + str(r['disagreements']):>7}{d:>10}  {r['gate']}")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(model=a.model, dtype=a.dtype, quant=a.quant, prompt_format=fmt,
                                               attention=be.attn, reference_attention=a.reference_attn,
                                               merged_layers=be.merged, reference_merge=a.reference_merge,
                                               lean=be.lean is not None, reference_lean=a.reference_lean,
                                               tree="flex" if be.flex else "dense",
                                               reference=a.reference, against=a.against,
                                               questions=a.questions and len(qs), rows=rows),
                                          indent=1), encoding="utf-8")
    return rows
