"""Inference backends. Every backend exposes the same two calls:

  run(prompts, ids_list)          -> one probability list per prompt (softmax over its option letters)
  run_shared(groups[, policy])    -> groups = [(prompts, ids_list)], all option orders of one question in one group

Backends:
  EagerBackend      plain PyTorch forward (reference, any GPU, Apple Silicon (MPS) or CPU)
  SharedBackend     shared prefix for the two option orders + token-budget batching without CUDA graphs
                    (Apple Silicon (MPS), CPU; also runs on CUDA)
  GraphBackend      static shapes + CUDA graphs, optional torch.compile, shared prefix for the two option orders,
                    token-budget batching, optional torchao FP8 / NVFP4 quantisation
  ExitGraphBackend  GraphBackend split into graph segments at trained early-exit layers; exit policy per request
  VLLMBackend       vLLM, for ModelOpt FP8 / NVFP4 checkpoints (native low-precision kernels)
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import PREFILL


def resolve(name, revision=None):
    """Local directory or Hugging Face repo id (downloaded once to the HF cache)."""
    p = Path(name)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision))


def default_device():
    """cuda if available, then Apple Silicon (mps), then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev):
    """Wait for queued kernels (timing): CUDA and MPS run asynchronously."""
    kind = torch.device(dev).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


class EagerBackend:
    def __init__(self, model_dir, dtype="bfloat16", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        dev = device or default_device()
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(dev).eval()
        self.dev = torch.device(dev)
        self.prefill = PREFILL

    @torch.no_grad()
    def run(self, prompts, ids_list):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
        logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        return [torch.softmax(lp[b, ids], -1).tolist() for b, ids in enumerate(ids_list)]

    def run_shared(self, groups, policy=None):
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


class GraphBackend(EagerBackend):
    """Fast path. Inputs are padded on the RIGHT into a small set of static shapes and run with plain causal attention
    and no mask: padding after the last real token cannot influence it, so the hidden state is read at the last real
    position. One CUDA graph per shape, all graphs in one memory pool, captured at start-up (lazy capture under load
    would stall the server). With `shared`, the option orders of one question are packed into ONE row:
    [shared prefix | options order 1 | options order 2], positions of each option block continue from the end of the
    prefix, and a block mask lets each option block see the prefix and itself only -- exactly equivalent to separate
    forwards, but the state (usually most of the tokens) is computed once."""

    LENS = [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048, 3072]
    BATCHES = [1, 2, 4, 8, 16, 32]
    TOKEN_BUDGET = 12288  # max batch x bucket length per forward

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, shared=True, warm=True):
        torch.backends.cuda.enable_cudnn_sdp(False)  # before capture: a captured graph keeps its attention kernel
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).cuda().eval()
        self.dev = next(self.model.parameters()).device
        self.prefill = PREFILL
        if quant:
            quantize(self.model, quant)
        self.fwd, self.fwd_masked = self._forward, self._forward_masked
        if compile:  # fuse normalisation / rotary / activation kernels, then capture graphs of the compiled forward
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
            self.fwd = torch.compile(self._forward, dynamic=True)
            self.fwd_masked = torch.compile(self._forward_masked, dynamic=True)
        self.shared = shared
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        if warm:
            for L in self.LENS:
                for b in self.BATCHES:
                    if b * L <= self.TOKEN_BUDGET or b == 1:
                        self._graph_shared(b, L) if shared else self._graph(b, L)

    # -- helpers -------------------------------------------------------------------------------------------------
    def _bucket(self, n, xs):
        for x in xs:
            if n <= x:
                return x
        raise ValueError(f"too long: {n} > {xs[-1]}")

    def _chunks(self, lens):
        """Indices sorted by length, cut into chunks with padded size <= TOKEN_BUDGET (short: big batches)."""
        order = sorted(range(len(lens)), key=lambda k: lens[k])
        out, cur = [], []
        for k in order:
            nxt = cur + [k]
            if cur and (len(nxt) > self.BATCHES[-1] or lens[k] > self.LENS[-1] or
                        self._bucket(len(nxt), self.BATCHES) * self._bucket(lens[k], self.LENS) > self.TOKEN_BUDGET):
                out.append(cur); nxt = [k]
            cur = nxt
        return out + ([cur] if cur else [])

    @torch.no_grad()
    def _forward(self, ids):
        return self.model.model(input_ids=ids, use_cache=False).last_hidden_state

    @torch.no_grad()
    def _forward_masked(self, ids, mask, pos):
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state

    def _capture(self, key, fn, *static):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                fn(*static)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn(*static)
        self.graphs[key] = (g, *static, out)
        return self.graphs[key]

    def _graph(self, b, L):
        key = ("p", b, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        return self._capture(key, self.fwd, ids)

    def _mask_from_seg(self, seg):
        """seg [b, L]: 0 = shared prefix, k>0 = option block k, -1 = padding -> additive 4D mask [b, 1, L, L]."""
        L = seg.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=seg.device).tril()[None]
        si, sj = seg[:, :, None], seg[:, None, :]
        allow = causal & (sj >= 0) & ((sj == 0) | (sj == si))
        allow |= torch.eye(L, dtype=torch.bool, device=seg.device)[None]
        dt = next(self.model.parameters()).dtype
        return torch.where(allow, 0.0, torch.finfo(dt).min).to(dt)[:, None]

    def _graph_shared(self, b, L):
        key = ("s", b, L)
        if key in self.graphs:
            return self.graphs[key]
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(L, device=self.dev)[None].expand(b, L).contiguous()
        mask = self._mask_from_seg(torch.ones((b, L), dtype=torch.long, device=self.dev))
        return self._capture(key, self.fwd_masked, ids, mask, pos)

    @staticmethod
    def _pack(toks):
        """Token lists of one group -> (ids, positions, segment ids, readout positions)."""
        P = min(len(t) for t in toks) - 1
        for k in range(P):
            if any(t[k] != toks[0][k] for t in toks[1:]):
                P = k
                break
        if len(toks) == 1:
            P = 0
        ids, pos, seg, last = list(toks[0][:P]), list(range(P)), [0] * P, []
        for g, t in enumerate(toks, 1):
            suf = t[P:]
            ids += suf; pos += range(P, P + len(suf)); seg += [g] * len(suf)
            last.append(len(ids) - 1)
        return ids, pos, seg, last

    def _readout(self, h, lids):
        lp = torch.log_softmax(self.model.lm_head(h).float(), -1)  # last_hidden_state is already normalised
        return [torch.softmax(lp[m, lids[m]], -1).tolist() for m in range(len(lids))]

    # -- public calls --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, prompts, ids_list):
        if self.shared:
            return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]
        enc = [self.tok(p, add_special_tokens=False).input_ids for p in prompts]
        res = [None] * len(enc)
        for idx in self._chunks([len(e) for e in enc]):
            if len(enc[idx[-1]]) > self.LENS[-1]:  # longer than the largest captured shape: plain forward
                for k in idx:
                    h = self._forward(torch.tensor([enc[k]], device=self.dev))[0, -1:]
                    res[k] = self._readout(h, [ids_list[k]])[0]
                continue
            chunk = [enc[k] for k in idx]
            L, b = self._bucket(max(map(len, chunk)), self.LENS), self._bucket(len(chunk), self.BATCHES)
            g, ids, out = self._graph(b, L)
            host = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long)
            for r, e in enumerate(chunk):
                host[r, : len(e)] = torch.tensor(e)
            ids.copy_(host.pin_memory(), non_blocking=True)
            g.replay()
            last = torch.tensor([len(e) - 1 for e in chunk], device=self.dev)
            h = out[torch.arange(len(chunk), device=self.dev), last]
            for r, p in zip(idx, self._readout(h, [ids_list[k] for k in idx])):
                res[r] = p
        return res

    def _fill(self, packs, idx, b, L, ids, mask, pos):
        h_ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long)
        h_pos = torch.arange(L)[None].repeat(b, 1)
        h_seg = torch.full((b, L), -1, dtype=torch.long)
        rows, cols = [], []
        for r, k in enumerate(idx):
            t, pp, sg, last = packs[k]
            h_ids[r, : len(t)] = torch.tensor(t); h_pos[r, : len(t)] = torch.tensor(pp)
            h_seg[r, : len(t)] = torch.tensor(sg)
            rows += [r] * len(last); cols += last
        ids.copy_(h_ids.pin_memory(), non_blocking=True)
        pos.copy_(h_pos.pin_memory(), non_blocking=True)
        mask.copy_(self._mask_from_seg(h_seg.to(self.dev, non_blocking=True)))
        return torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev)

    def _eager_shared(self, pack, lids):
        t, pp, sg, last = pack
        h = self._forward_masked(torch.tensor([t], device=self.dev),
                                 self._mask_from_seg(torch.tensor([sg], device=self.dev)),
                                 torch.tensor([pp], device=self.dev))[0, last]
        return self._readout(h, lids)

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        if not self.shared:  # plain graphs captured: one row per option order
            return EagerBackend.run_shared(self, groups)
        packs = [self._pack([self.tok(p, add_special_tokens=False).input_ids for p in prompts]) for prompts, _ in groups]
        res = [None] * len(packs)
        for idx in self._chunks([len(pk[0]) for pk in packs]):
            if len(packs[idx[-1]][0]) > self.LENS[-1]:
                for k in idx:
                    res[k] = self._eager_shared(packs[k], groups[k][1])
                continue
            L = self._bucket(max(len(packs[k][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            g, ids, mask, pos, out = self._graph_shared(b, L)
            ri, ci = self._fill(packs, idx, b, L, ids, mask, pos)
            g.replay()
            probs = self._readout(out[ri, ci], [x for k in idx for x in groups[k][1]])
            j = 0
            for k in idx:
                n = len(groups[k][1]); res[k] = probs[j: j + n]; j += n
        return res


class SharedBackend(GraphBackend):
    """The shared-prefix path of GraphBackend without CUDA graphs, for Apple Silicon (MPS) and CPU. The option orders
    of one question are packed into one row (see GraphBackend), questions are batched under the token budget, and rows
    are padded on the right to the same length buckets, so the Metal kernels see a small, repeating set of shapes."""

    TOKEN_BUDGET = 4096  # smaller than on CUDA: attention scores are materialised, and the GPU shares system memory

    def __init__(self, model_dir, dtype="bfloat16", device=None, warm=True):
        EagerBackend.__init__(self, model_dir, dtype, device)
        self.tok.padding_side = "right"
        self.shared = True
        if warm:  # the first forward of a shape is slow on MPS (kernel selection); run the common ones once
            with torch.no_grad():
                for L in self.LENS[:8]:
                    self._forward_masked(*self._inputs([([self.tok.pad_token_id] * (L - 1), [0] * (L - 1),
                                                         [1] * (L - 1), [L - 2])], [0], 1, L)[:3])
                sync(self.dev)

    def _inputs(self, packs, idx, b, L):
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long)
        pos = torch.arange(L)[None].repeat(b, 1)
        seg = torch.full((b, L), -1, dtype=torch.long)
        rows, cols = [], []
        for r, k in enumerate(idx):
            t, pp, sg, last = packs[k]
            ids[r, : len(t)] = torch.tensor(t); pos[r, : len(t)] = torch.tensor(pp); seg[r, : len(t)] = torch.tensor(sg)
            rows += [r] * len(last); cols += last
        d = self.dev
        return ids.to(d), self._mask_from_seg(seg.to(d)), pos.to(d), torch.tensor(rows, device=d), torch.tensor(cols, device=d)

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        packs = [self._pack([self.tok(p, add_special_tokens=False).input_ids for p in prompts]) for prompts, _ in groups]
        res = [None] * len(packs)
        for idx in self._chunks([len(pk[0]) for pk in packs]):
            longest = max(len(packs[k][0]) for k in idx)
            L = self._bucket(longest, self.LENS) if longest <= self.LENS[-1] else longest
            ids, mask, pos, ri, ci = self._inputs(packs, idx, len(idx), L)
            probs = self._readout(self._forward_masked(ids, mask, pos)[ri, ci], [x for k in idx for x in groups[k][1]])
            j = 0
            for k in idx:
                n = len(groups[k][1]); res[k] = probs[j: j + n]; j += n
        return res


class ExitHead(torch.nn.Module):
    """Trained early-exit head: RMSNorm (initialised from the final norm) + residual low-rank adapter, followed by the
    frozen LM head restricted to the option letters."""

    def __init__(self, final_norm, d, r=256):
        super().__init__()
        self.norm = type(final_norm)(d, eps=final_norm.variance_epsilon)
        self.a = torch.nn.Linear(d, r, bias=False)
        self.b = torch.nn.Linear(r, d, bias=False)

    def forward(self, h):
        x = self.norm(h.float())
        return x + self.b(self.a(x))


class ExitGraphBackend(GraphBackend):
    """Shared-prefix fast path split into CUDA-graph segments at the exit layers. After each segment the exit heads
    score the letters at the readout positions; if every readout in the chunk reaches the threshold of that layer the
    chunk stops there (a batch stops only as a whole). `policy` selects the thresholds per request: "off" = always the
    final layer, or an agreement level calibrated in advance ("0.999", "0.995", "0.99", "0.98")."""

    def __init__(self, model_dir, dtype="bfloat16", quant=None, compile=False, heads_dir=None, default_policy="off"):
        super().__init__(model_dir, dtype, quant, compile=False, shared=True, warm=False)
        hd = Path(heads_dir) if heads_dir else Path(model_dir) / "exit_heads"
        ck = torch.load(hd / "heads.pt", map_location="cpu")
        th = json.loads((hd / "thresholds.json").read_text())
        self.policies = {k: {int(L): t for L, t in v["taus"].items() if t <= 1.0} for k, v in th["levels"].items()}
        self.policies["off"] = {}
        self.default_policy = default_policy
        self.exits = sorted({L for pol in self.policies.values() for L in pol})
        H = self.model.config.hidden_size
        self.heads = torch.nn.ModuleDict({str(L): ExitHead(self.model.model.norm, H, ck["rank"]) for L in ck["layers"]})
        self.heads.load_state_dict(ck["state"])
        self.heads = self.heads.to(self.dev).eval()
        n = len(self.model.model.layers)
        self.bounds = [0] + self.exits + [n]
        self.segs = [self._seg_fn(a, b, b == n) for a, b in zip(self.bounds, self.bounds[1:])]
        if compile:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
            self.segs = [torch.compile(f, dynamic=True) for f in self.segs]
        self.stats = {}
        for L in self.LENS:
            for b in self.BATCHES:
                if b * L <= self.TOKEN_BUDGET or b == 1:
                    self._graph_exit(b, L)

    def _seg_fn(self, a, b, last):
        mm = self.model.model

        @torch.no_grad()
        def f(h, mask, pos, cos, sin):
            for layer in mm.layers[a:b]:
                h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
                h = h[0] if isinstance(h, tuple) else h
            return mm.norm(h) if last else h
        return f

    def _graph_exit(self, b, L):
        key = ("e", b, L)
        if key in self.graphs:
            return self.graphs[key]
        mm = self.model.model
        ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        pos = torch.arange(L, device=self.dev)[None].expand(b, L).contiguous()
        mask = self._mask_from_seg(torch.ones((b, L), dtype=torch.long, device=self.dev))

        @torch.no_grad()
        def emb():
            h = mm.embed_tokens(ids)
            cos, sin = mm.rotary_emb(h, pos)
            return h, cos, sin
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                h, cos, sin = emb()
                for f in self.segs:
                    h = f(h, mask, pos, cos, sin)
        torch.cuda.current_stream().wait_stream(st)
        graphs, outs = [], []
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            h, cos, sin = emb()
            h = self.segs[0](h, mask, pos, cos, sin)
        graphs.append(g); outs.append(h)
        for f in self.segs[1:]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                h = f(outs[-1], mask, pos, cos, sin)
            graphs.append(g); outs.append(h)
        self.graphs[key] = (graphs, ids, mask, pos, outs)
        return self.graphs[key]

    @torch.no_grad()
    def run(self, prompts, ids_list, policy=None):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)], policy)]

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        taus = self.policies[policy or self.default_policy]
        packs = [self._pack([self.tok(p, add_special_tokens=False).input_ids for p in prompts]) for prompts, _ in groups]
        res = [None] * len(packs)
        W = self.model.lm_head.weight
        for idx in self._chunks([len(pk[0]) for pk in packs]):
            if len(packs[idx[-1]][0]) > self.LENS[-1]:
                for k in idx:
                    res[k] = self._eager_shared(packs[k], groups[k][1])
                continue
            L = self._bucket(max(len(packs[k][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            graphs, ids, mask, pos, outs = self._graph_exit(b, L)
            ri, ci = self._fill(packs, idx, b, L, ids, mask, pos)
            lids = [x for k in idx for x in groups[k][1]]
            probs, depth = None, self.bounds[-1]
            for s, g in enumerate(graphs):
                g.replay()
                if s < len(self.exits) and self.exits[s] in taus:
                    Lx = self.exits[s]
                    lg = (self.heads[str(Lx)](outs[s][ri, ci]).to(W.dtype) @ W.T).float()
                    ps = [torch.softmax(lg[m, lids[m]], -1) for m in range(len(lids))]
                    if min(float(p.max()) for p in ps) >= taus[Lx]:
                        probs, depth = [p.tolist() for p in ps], Lx
                        break
            if probs is None:
                probs = self._readout(outs[-1][ri, ci], lids)
            key = (policy or self.default_policy, depth)
            self.stats[key] = self.stats.get(key, 0) + len(idx)
            j = 0
            for k in idx:
                n = len(groups[k][1]); res[k] = probs[j: j + n]; j += n
        return res


class VLLMBackend:
    """vLLM backend for ModelOpt FP8 / NVFP4 checkpoints. One generated token restricted to the option letters; with
    logprobs_mode="processed_logprobs" the log-probabilities are computed after that restriction, so the softmax over
    the letters equals the readout of the other backends. vLLM's automatic prefix caching shares the state between the
    two option orders."""

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096):
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.llm = LLM(model=str(model_dir), dtype=dtype, gpu_memory_utilization=mem, max_model_len=max_len,
                       enable_prefix_caching=True, logprobs_mode="processed_logprobs", max_logprobs=16)

    def run(self, prompts, ids_list):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        reqs = [TokensPrompt(prompt_token_ids=self.tok(p, add_special_tokens=False).input_ids) for p in prompts]
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(ids), allowed_token_ids=list(ids))
               for ids in ids_list]
        res = []
        for o, ids in zip(self.llm.generate(reqs, sps, use_tqdm=False), ids_list):
            lp = o.outputs[0].logprobs[0]
            x = torch.tensor([lp[i].logprob if i in lp else -1e9 for i in ids], dtype=torch.float32)
            res.append(torch.softmax(x, -1).tolist())
        return res

    def run_shared(self, groups, policy=None):
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


def quantize(model, quant):
    """On-the-fly torchao quantisation of the decoder layers (the LM head stays in bf16).
    fp8   -- dynamic FP8 activations + FP8 weights, per-row scales (Ada, Hopper, Blackwell)
    nvfp4 -- NVFP4 activations + weights (Blackwell); for best FP4 speed prefer the ModelOpt NVFP4 checkpoint + vLLM"""
    from torchao.quantization import quantize_
    if quant == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif quant == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        cfg = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    else:
        raise ValueError(f"unknown quantisation {quant!r} (fp8 | nvfp4)")
    quantize_(model.model, cfg)
