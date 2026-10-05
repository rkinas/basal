"""Inference backends. Every backend exposes the same two calls:

  run(prompts, ids_list)          -> one probability list per prompt (softmax over its option letters)
  run_shared(groups[, policy])    -> groups = [(prompts, ids_list)]; the prompts of one group share a prefix (all option
                                     orders of one question, or every branch of a whole request: state once, ask many)

A prompt is a string or a list of token ids (the server tokenizes once and passes ids).

Backends:
  EagerBackend      plain PyTorch forward (reference, any GPU, Apple MPS or CPU)
  GraphBackend      static shapes + CUDA graphs, optional torch.compile, shared prefix for the two option orders,
                    token-budget batching, optional torchao FP8 / NVFP4 quantisation
  ExitGraphBackend  GraphBackend split into graph segments at trained early-exit layers; exit policy per request
  MPSBackend        GraphBackend's shared prefix + token-budget batching on Apple MPS (PyTorch, no graphs)
  VLLMBackend       vLLM, for ModelOpt FP8 / NVFP4 checkpoints (native low-precision kernels)
  SGLangBackend     SGLang offline engine (1.5): the same letter readout from the log-probabilities of the option tokens
  MLXBackend        Apple Silicon, MLX / mlx-lm (1.5): MLX fp8 / fp4 exports or bf16 weights (optional 8-bit at load
                    time); shared-prefix KV cache
  GGUFBackend       a GGUF file in process through llama.cpp (llama-cpp-python; Metal, CUDA or CPU); shared prefix as
                    llama.cpp sequences
  OllamaBackend     Ollama with the GGUF exports (1.5): raw prompt, one token, top log-probabilities of the letters
  LlamaCppBackend   llama.cpp server with the GGUF exports (1.5): token-id prompt, `n_probs` log-probabilities
  TPUBackend        (basal/tpu.py) the same packing in JAX on Google TPU, one XLA executable per shape
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import PREFILL


METADATA_FILES = ["*.json", "*.jinja", "*.model", "*.txt", "*.md", "Modelfile*"]  # tokenizer, template, calibration


def resolve(name, revision=None, metadata_only=False):
    """Local directory or Hugging Face repo id (downloaded once to the HF cache). metadata_only: tokenizer, chat
    template and calibration files only (the Ollama / llama.cpp modes, where the engine serves the GGUF weights)."""
    p = Path(name)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision, allow_patterns=METADATA_FILES if metadata_only else None))


def default_device():
    """BASAL_DEVICE if set, else cuda, else Apple Silicon (mps), else cpu."""
    import os
    if os.environ.get("BASAL_DEVICE"):
        return os.environ["BASAL_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EagerBackend:
    def __init__(self, model_dir, dtype="bfloat16", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        dev = device or default_device()
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(dev).eval()
        self.dev = dev
        self.prefill = PREFILL

    def encode(self, prompts):
        """Strings -> token id lists (one batched tokenizer call); id lists pass through."""
        txt = [k for k, p in enumerate(prompts) if isinstance(p, str)]
        out = list(prompts)
        if txt:
            for k, ids in zip(txt, self.tok([prompts[k] for k in txt], add_special_tokens=False).input_ids):
                out[k] = ids
        return out

    @torch.no_grad()
    def run(self, prompts, ids_list):
        enc = self.encode(prompts)
        L = max(map(len, enc))
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(e)) + list(e) for e in enc], device=self.dev)
        att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=self.dev)
        logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :].float()
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
    forwards, but the state (usually most of the tokens) is computed once. A group may hold every branch of a request
    (SOAM: all questions x option orders over one state); a group whose packed row is longer than the largest shape is
    bisected into smaller groups, which is still exact (each branch sees the prefix and itself only)."""

    LENS = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072]
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
        self._wrows = {}
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

    def _mask_from_seg(self, seg, parents=None):
        """seg [b, L]: block id of every token, -1 = padding; parents: per row, the parent block of every block (-1 =
        root). A token sees earlier tokens of its own block and all tokens of its ancestor blocks (the prefix trie of
        the prompts), so every readout equals a separate forward of its own prompt. Without parents: block 0 is the
        shared prefix of all other blocks. -> additive 4D mask [b, 1, L, L]."""
        b, L = seg.shape
        if parents is None:
            nb = int(seg.max()) + 1
            parents = [[-1] + [0] * (nb - 1)] * b
        NB = max(map(len, parents)) + 1  # last index: padding, sees nothing
        anc = np.zeros((b, NB, NB), dtype=bool)
        for r, par in enumerate(parents):
            a = anc[r]
            for k, pk in enumerate(par):  # parents precede children (depth-first order)
                if pk >= 0:
                    a[k] = a[pk]
                a[k, k] = True
        anc = torch.from_numpy(anc).to(seg.device, non_blocking=True)
        s = torch.where(seg >= 0, seg, NB - 1)
        allow = anc[torch.arange(b, device=seg.device)[:, None, None], s[:, :, None], s[:, None, :]]
        allow &= torch.ones(L, L, dtype=torch.bool, device=seg.device).tril()[None]
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
        """Token lists of one group -> (ids, positions, block ids, readout positions, block parents): the compressed
        prefix trie of the prompts, flattened depth-first. Shared prefixes (the state for all questions, a question's
        text for its option orders) are computed once; positions are each token's index in its own prompt."""
        import os
        ids, pos, seg, parent, last = [], [], [], [], [None] * len(toks)

        def build(members, start, par):
            end = start + len(os.path.commonprefix([toks[m][start:] for m in members]))
            blk = len(parent)
            parent.append(par)
            ids.extend(toks[members[0]][start:end]); pos.extend(range(start, end)); seg.extend([blk] * (end - start))
            kids = {}
            for m in members:
                if len(toks[m]) == end:
                    last[m] = len(ids) - 1
                else:
                    kids.setdefault(toks[m][end], []).append(m)
            for ms in kids.values():
                build(ms, end, blk)

        roots = {}
        for m, tk in enumerate(toks):
            roots.setdefault(tk[0], []).append(m)
        for ms in roots.values():
            build(ms, 0, -1)
        return ids, pos, seg, last, parent

    def _rows(self, lids):
        """Answer-row slice of the LM head for the letters in lids, and each readout's column indices (padded, masked).
        softmax over the letters of the full log-softmax equals softmax over these rows' logits (row-only readout)."""
        key = tuple(sorted({i for ids in lids for i in ids}))
        if key not in self._wrows:
            self._wrows[key] = self.model.lm_head.weight[list(key)].contiguous()
        col = {t: j for j, t in enumerate(key)}
        K = max(map(len, lids))
        idx = torch.tensor([[col[i] for i in ids] + [0] * (K - len(ids)) for ids in lids], device=self.dev)
        pad = torch.tensor([[False] * len(ids) + [True] * (K - len(ids)) for ids in lids], device=self.dev)
        return self._wrows[key], idx, pad

    def _letters(self, z, idx, pad, lids):
        p = torch.softmax(z.float().gather(1, idx).masked_fill(pad, float("-inf")), -1).tolist()  # one device sync
        return [row[: len(ids)] for row, ids in zip(p, lids)]

    def _readout(self, h, lids):
        W, idx, pad = self._rows(lids)
        return self._letters(h @ W.T, idx, pad, lids)  # last_hidden_state is already normalised

    # -- public calls --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, prompts, ids_list):
        if self.shared:
            return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]
        enc = self.encode(prompts)
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

    def _host_rows(self, packs, idx, b, L):
        """Packed rows -> padded host tensors ids / positions / block ids [b, L], readout rows and columns, and the
        block parents of every row."""
        h_ids = torch.full((b, L), self.tok.pad_token_id, dtype=torch.long)
        h_pos = torch.arange(L)[None].repeat(b, 1)
        h_seg = torch.full((b, L), -1, dtype=torch.long)
        rows, cols, parents = [], [], [[] for _ in range(b)]
        for r, k in enumerate(idx):
            t, pp, sg, last, parents[r] = packs[k]
            h_ids[r, : len(t)] = torch.tensor(t); h_pos[r, : len(t)] = torch.tensor(pp)
            h_seg[r, : len(t)] = torch.tensor(sg)
            rows += [r] * len(last); cols += last
        return h_ids, h_pos, h_seg, rows, cols, parents

    def _fill(self, packs, idx, b, L, ids, mask, pos):
        h_ids, h_pos, h_seg, rows, cols, parents = self._host_rows(packs, idx, b, L)
        ids.copy_(h_ids.pin_memory(), non_blocking=True)
        pos.copy_(h_pos.pin_memory(), non_blocking=True)
        mask.copy_(self._mask_from_seg(h_seg.to(self.dev, non_blocking=True), parents))
        return torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev)

    def _eager_shared(self, pack, lids):
        t, pp, sg, last, par = pack
        h = self._forward_masked(torch.tensor([t], device=self.dev),
                                 self._mask_from_seg(torch.tensor([sg], device=self.dev), [par]),
                                 torch.tensor([pp], device=self.dev))[0, last]
        return self._readout(h, lids)

    def _oversize(self, pack, n):
        """A packed row of n readouts that must be split: longer than the largest captured length."""
        return len(pack[0]) > self.LENS[-1]

    def _units(self, groups):
        """Groups -> packed rows ("units"). A group whose packed row exceeds the largest captured length is bisected
        (exact: every branch still sees the shared prefix and itself only). Returns [(pack, lids, group, first)]."""
        units = []
        for k, (prompts, lids) in enumerate(groups):
            toks = self.encode(prompts)
            todo = [(0, len(toks))]
            while todo:
                a, b = todo.pop()
                pk = self._pack(toks[a:b])
                if self._oversize(pk, b - a) and b - a > 1:
                    m = (a + b) // 2
                    todo += [(m, b), (a, m)]
                else:
                    units.append((pk, lids[a:b], k, a))
        return units

    @staticmethod
    def _assemble(groups, units, probs):
        res = [[None] * len(g[0]) for g in groups]
        for (pk, lids, k, a), pr in zip(units, probs):
            res[k][a: a + len(lids)] = pr
        return res

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        if not self.shared:  # plain graphs captured: one row per option order
            return EagerBackend.run_shared(self, groups)
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._chunks([len(u[0][0]) for u in units]):
            if len(units[idx[-1]][0][0]) > self.LENS[-1]:
                for k in idx:
                    probs[k] = self._eager_shared(units[k][0], units[k][1])
                continue
            L = self._bucket(max(len(units[k][0][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            g, ids, mask, pos, out = self._graph_shared(b, L)
            ri, ci = self._fill([u[0] for u in units], idx, b, L, ids, mask, pos)
            g.replay()
            pr = self._readout(out[ri, ci], [x for k in idx for x in units[k][1]])
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)


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
        units = self._units(groups)
        res = [None] * len(units)
        for idx in self._chunks([len(u[0][0]) for u in units]):
            if len(units[idx[-1]][0][0]) > self.LENS[-1]:
                for k in idx:
                    res[k] = self._eager_shared(units[k][0], units[k][1])
                continue
            L = self._bucket(max(len(units[k][0][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            graphs, ids, mask, pos, outs = self._graph_exit(b, L)
            ri, ci = self._fill([u[0] for u in units], idx, b, L, ids, mask, pos)
            lids = [x for k in idx for x in units[k][1]]
            W, cols, pad = self._rows(lids)
            probs, depth = None, self.bounds[-1]
            for s, g in enumerate(graphs):
                g.replay()
                if s < len(self.exits) and self.exits[s] in taus:
                    Lx = self.exits[s]
                    z = self.heads[str(Lx)](outs[s][ri, ci]).to(W.dtype) @ W.T
                    p = torch.softmax(z.float().gather(1, cols).masked_fill(pad, float("-inf")), -1)
                    if float(p.max(-1).values.min()) >= taus[Lx]:
                        probs, depth = [row[: len(x)] for row, x in zip(p.tolist(), lids)], Lx
                        break
            if probs is None:
                probs = self._letters(outs[-1][ri, ci] @ W.T, cols, pad, lids)
            key = (policy or self.default_policy, depth)
            self.stats[key] = self.stats.get(key, 0) + len(idx)
            j = 0
            for k in idx:
                n = len(units[k][1]); res[k] = probs[j: j + n]; j += n
        return self._assemble(groups, units, res)


class MPSBackend(GraphBackend):
    """Apple GPU through PyTorch MPS: the shared-prefix packing (prefix trie, SOAM) and token-budget batching of
    GraphBackend, run as plain forwards (MPS has no CUDA graphs). Rows are padded on the right to the longest packed
    row of the chunk."""

    def __init__(self, model_dir, dtype="bfloat16", shared=True, device="mps"):
        EagerBackend.__init__(self, model_dir, dtype, device)
        self.dev = torch.device(device)
        self.shared = shared
        self._wrows = {}

    @torch.no_grad()
    def run(self, prompts, ids_list):
        if not self.shared:
            return EagerBackend.run(self, prompts, ids_list)
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        if not self.shared:
            return EagerBackend.run_shared(self, groups)
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._chunks([len(u[0][0]) for u in units]):
            if len(units[idx[-1]][0][0]) > self.LENS[-1]:
                for k in idx:
                    probs[k] = self._eager_shared(units[k][0], units[k][1])
                continue
            L = max(len(units[k][0][0]) for k in idx)
            ids, pos, seg, rows, cols, parents = self._host_rows([u[0] for u in units], idx, len(idx), L)
            h = self._forward_masked(ids.to(self.dev), self._mask_from_seg(seg.to(self.dev), parents), pos.to(self.dev))
            pr = self._readout(h[rows, cols], [x for k in idx for x in units[k][1]])
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)


class VLLMBackend:
    """vLLM backend for ModelOpt FP8 / NVFP4 checkpoints. One generated token restricted to the option letters; with
    logprobs_mode="processed_logprobs" the log-probabilities are computed after that restriction, so the softmax over
    the letters equals the readout of the other backends. Where vLLM supports it, logprob_token_ids asks for exactly
    the letter ids: vllm-metal ignores the logprobs mode and returns the top-k of the unrestricted vocabulary, which
    can miss a letter. vLLM's automatic prefix caching shares the state between the two option orders."""

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096):
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.llm = LLM(model=str(model_dir), dtype=dtype, gpu_memory_utilization=mem, max_model_len=max_len,
                       enable_prefix_caching=True, logprobs_mode="processed_logprobs", max_logprobs=16)

    def run(self, prompts, ids_list):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        reqs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in EagerBackend.encode(self, prompts)]
        import inspect
        exact = "logprob_token_ids" in inspect.signature(SamplingParams).parameters
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(ids), allowed_token_ids=list(ids),
                              **(dict(logprob_token_ids=list(ids)) if exact else {}))
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


class SGLangBackend:
    """SGLang offline engine (1.5). One generated token; the engine returns the log-probabilities of the option letters
    at the answer position (`token_ids_logprob`), and the softmax over them equals the readout of the other backends
    (a softmax over a subset of log-softmax values is the softmax over the same logits). RadixAttention shares the
    state between the option orders of one question and between the questions of one request."""

    def __init__(self, model_dir, dtype="bfloat16", mem=0.6, max_len=4096):
        import sglang
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.llm = sglang.Engine(model_path=str(model_dir), dtype=dtype, mem_fraction_static=mem,
                                 context_length=max_len, log_level="error")

    @staticmethod
    def _letters(meta, ids):
        """meta_info["output_token_ids_logprobs"][0] -> logprob per requested id ((logprob, token_id, text) tuples)."""
        got = {int(x[1]): float(x[0]) for x in meta["output_token_ids_logprobs"][0]}
        return [got.get(i, -1e9) for i in ids]

    def run(self, prompts, ids_list):
        enc = [list(ids) for ids in EagerBackend.encode(self, prompts)]
        outs = self.llm.generate(input_ids=enc, sampling_params=[{"max_new_tokens": 1, "temperature": 0.0}] * len(enc),
                                 return_logprob=True, logprob_start_len=-1,
                                 token_ids_logprob=[list(ids) for ids in ids_list])
        return [torch.softmax(torch.tensor(self._letters(o["meta_info"], ids), dtype=torch.float32), -1).tolist()
                for o, ids in zip(outs, ids_list)]

    def run_shared(self, groups, policy=None):
        flat = self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]])
        out, j = [], 0
        for g in groups:
            out.append(flat[j: j + len(g[0])]); j += len(g[0])
        return out


def _split(flat, groups):
    out, j = [], 0
    for g in groups:
        out.append(flat[j: j + len(g[0])]); j += len(g[0])
    return out


def mlx_config_overrides(cfg):
    """transformers 5 writes the rotary settings as `rope_parameters`; mlx-lm reads `rope_theta` / `rope_scaling` only
    (without this override it silently uses rope_theta = 10000)."""
    rp = cfg.get("rope_parameters") or {}
    out = {}
    if "rope_theta" not in cfg and "rope_theta" in rp:
        out["rope_theta"] = rp["rope_theta"]
    if not cfg.get("rope_scaling") and rp.get("rope_type", "default") != "default":
        out["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
    return out


class MLXBackend:
    """Apple Silicon (MLX / mlx-lm): the MLX exports (the -MLX-8bit / -MLX-fp4 repositories) or a bf16 Hugging Face
    directory. Same prompts, tokens and letter readout as the PyTorch backends. The shared prefix of a group (both
    option orders of a question; with SOAM every question and order of a request, i.e. the state) is computed once into
    a KV cache, and the remaining suffixes run as one right-padded batch continuing that cache: a suffix sees the
    prefix and its own earlier tokens only, so every readout equals a separate forward of its prompt. All MLX work
    runs on one dedicated thread (the server calls the backend from an executor)."""

    MAX_ROWS = 16      # suffix rows per forward
    TOKEN_BUDGET = 8192  # rows x padded length per forward without a shared prefix
    CACHE_BYTES = 2 << 30  # MLX buffer cache limit (and wired headroom above the weights)

    def __init__(self, model_dir, max_rows=None, quant=None):
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.prefill = PREFILL
        self.max_rows = max_rows or self.MAX_ROWS
        self.pool.submit(self._load, Path(model_dir), quant).result()

    def _load(self, md, quant=None):
        """quant: None, or "q8": MLX affine 8-bit weights (group size 64) of the decoder blocks' linear layers, made at
        load time from bf16 weights (the embeddings and the LM head stay bf16); the -MLX-8bit exports need no quant."""
        import mlx.core as mx
        from mlx_lm.utils import load_model
        self.mx = mx
        cfg = json.loads((md / "config.json").read_text())
        if quant not in (None, "q8"):
            raise ValueError(f"unknown MLX quantisation {quant!r} (q8)")
        # lazy for q8: the weights are materialised once, after quantisation (never the full bf16 copy)
        self.model, self.config = load_model(md, lazy=bool(quant), model_config=mlx_config_overrides(cfg))
        self.quantization = self.config.get("quantization")
        if quant:
            if self.quantization:
                raise ValueError(f"{md} is already quantised ({self.quantization}); use it without --quant")
            import mlx.nn as nn
            nn.quantize(self.model, group_size=64, bits=8,
                        class_predicate=lambda path, m: isinstance(m, nn.Linear) and path.startswith("model.layers."))
            mx.eval(self.model.parameters())
            self.quantization = {"group_size": 64, "bits": 8}
        # keep the weights resident (wired: under memory pressure macOS would page them out and every forward would
        # wait for them) and the buffer cache small (by default it grows towards the whole working set, one entry per
        # new shape, and on a 16-24 GB Mac pushes everything else into swap)
        from mlx.utils import tree_flatten
        weights = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        if mx.metal.is_available():
            rec = mx.device_info().get("max_recommended_working_set_size", weights)
            mx.set_wired_limit(min(rec, weights + self.CACHE_BYTES))
        mx.set_cache_limit(self.CACHE_BYTES)

    def encode(self, prompts):
        return EagerBackend.encode(self, prompts)

    def _head(self, h):
        m = self.model
        return m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)

    def _letters(self, h, lids):
        """Hidden states [n, d] at the readout positions -> softmax over each readout's letters (float32)."""
        mx = self.mx
        z = self._head(h).astype(mx.float32)
        K = max(map(len, lids))
        idx = mx.array([list(ids) + [ids[0]] * (K - len(ids)) for ids in lids])
        pad = mx.array([[False] * len(ids) + [True] * (K - len(ids)) for ids in lids])
        z = mx.where(pad, -mx.inf, mx.take_along_axis(z, idx, axis=1))
        p = mx.softmax(z, axis=-1).tolist()
        return [row[: len(ids)] for row, ids in zip(p, lids)]

    def _forward(self, rows, prefix=None):
        """Token rows (right-padded into one batch) -> hidden state at each row's last token. `prefix`: per-layer
        (keys, values) of a shared prefix that every row continues (positions start after it)."""
        mx = self.mx
        from mlx_lm.models.cache import KVCache
        B, S = len(rows), max(map(len, rows))
        pad = self.tok.pad_token_id
        ids = mx.array([list(r) + [pad] * (S - len(r)) for r in rows])
        cache = None
        if prefix is not None:
            cache = []
            for k, v in prefix:
                c = KVCache()
                c.keys, c.values, c.offset = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0), k.shape[2]
                cache.append(c)
        h = self.model.model(ids, cache=cache)
        return h[mx.arange(B), mx.array([len(r) - 1 for r in rows])]

    def _prefix(self, toks):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        self.model.model(mx.array([toks]), cache=cache)
        kv = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache]
        mx.eval(kv)
        return kv

    def _group(self, toks, lids):
        """One group: shared prefix once, then the suffixes in batches of max_rows."""
        import os
        P = min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1)
        prefix = self._prefix(toks[0][:P]) if P > 0 else None
        out = []
        for i in range(0, len(toks), self.max_rows):
            h = self._forward([t[P:] for t in toks[i: i + self.max_rows]], prefix)
            out += self._letters(h, lids[i: i + self.max_rows])
        return out

    def _singles(self, toks, lids):
        """Unrelated prompts: right-padded batches of similar length, no cache."""
        chunks, cur = [], []
        for k in sorted(range(len(toks)), key=lambda k: len(toks[k])):
            if cur and (len(cur) >= self.max_rows or (len(cur) + 1) * len(toks[k]) > self.TOKEN_BUDGET):
                chunks.append(cur); cur = []
            cur.append(k)
        res = [None] * len(toks)
        for c in chunks + [cur]:
            for j, p in zip(c, self._letters(self._forward([toks[j] for j in c]), [lids[j] for j in c])):
                res[j] = p
        return res

    def _run_shared(self, groups):
        enc = [(self.encode(p), x) for p, x in groups]
        out = [None] * len(groups)
        single = [k for k, (t, _) in enumerate(enc) if len(t) == 1]
        if single:
            for k, p in zip(single, self._singles([enc[k][0][0] for k in single], [enc[k][1][0] for k in single])):
                out[k] = [p]
        for k, (t, x) in enumerate(enc):
            if out[k] is None:
                out[k] = self._group(t, x)
        return out

    def run_shared(self, groups, policy=None):
        return self.pool.submit(self._run_shared, groups).result()

    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]


class GGUFBackend(GraphBackend):
    """A GGUF file in process through llama.cpp (llama-cpp-python: Metal on Apple Silicon, CUDA or CPU elsewhere). Only
    the weights come from the GGUF file; tokenizer, chat template and CALIBRATION.json come from the Hugging Face model
    directory, so the token ids are exactly those of the other backends. The packed prefix trie of a group (as in
    GraphBackend: the state once, every question and option order after it) becomes llama.cpp sequences: every readout
    has its own sequence and every token belongs to the sequences of all readouts below its block, so a shared prefix
    is computed once and each readout sees its own prompt only. All rows of a chunk go through one llama_decode
    (unified KV cache, cleared per chunk)."""

    N_CTX = 16384  # KV cells per chunk: room for a single long prompt
    N_SEQ = 64     # readouts (sequences) per chunk

    def __init__(self, model_dir, gguf, n_gpu_layers=-1):
        import logging

        import llama_cpp as C
        logging.getLogger("llama-cpp-python").setLevel(logging.ERROR)  # llama.cpp load / Metal info lines
        self.C = C
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.shared = True
        C.llama_backend_init()
        mp = C.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        self.cmodel = C.llama_model_load_from_file(str(gguf).encode(), mp)
        if not self.cmodel:
            raise ValueError(f"llama.cpp could not load {gguf}")
        # a full model directory: its architecture must match the GGUF file; the -GGUF repositories carry tokenizer and
        # calibration without config.json (their .gguf files belong to them)
        cp = Path(model_dir) / "config.json"
        cfg = json.loads(cp.read_text()) if cp.exists() else {}
        for name, llama_name in (("num_hidden_layers", "llama_model_n_layer"), ("hidden_size", "llama_model_n_embd")):
            if not cfg:
                break
            expected = cfg.get(name)
            if expected is None:
                raise ValueError(f"{model_dir}/config.json is missing {name}")
            actual = getattr(C, llama_name)(self.cmodel)
            if actual != expected:
                raise ValueError(f"{gguf}: {llama_name.removeprefix('llama_model_')} is {actual}, "
                                 f"but {model_dir}/config.json specifies {name}={expected}")
        self.n_vocab = C.llama_vocab_n_tokens(C.llama_model_get_vocab(self.cmodel))
        expected_vocab = cfg.get("vocab_size")
        if expected_vocab is not None and self.n_vocab != expected_vocab:
            raise ValueError(f"{gguf}: vocabulary is {self.n_vocab}, "
                             f"but {model_dir}/config.json specifies vocab_size={expected_vocab}")
        if self.n_vocab < len(self.tok):
            raise ValueError(f"{gguf}: vocabulary of {self.n_vocab} tokens, tokenizer of {model_dir} has {len(self.tok)}")
        cp = C.llama_context_default_params()
        cp.n_ctx = cp.n_batch = self.N_CTX
        cp.n_ubatch = 512
        cp.n_seq_max = self.N_SEQ
        cp.kv_unified = True  # sequences share the cells of their common prefix
        self.ctx = C.llama_init_from_model(self.cmodel, cp)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self.batch = C.llama_batch_init(self.N_CTX, 0, self.N_SEQ)  # a prefix token belongs to every readout below it

    def __del__(self):
        C = getattr(self, "C", None)
        if C is None:
            return
        if getattr(self, "batch", None) is not None:
            C.llama_batch_free(self.batch)
        if getattr(self, "ctx", None):
            C.llama_free(self.ctx)
        if getattr(self, "cmodel", None):
            C.llama_model_free(self.cmodel)

    def _oversize(self, pack, n):
        return len(pack[0]) > self.N_CTX or n > self.N_SEQ

    def _decode_chunks(self, units):
        """Units sorted by length, cut into chunks of at most N_CTX tokens and N_SEQ readouts."""
        out, cur, n_tok, n_seq = [], [], 0, 0
        for k in sorted(range(len(units)), key=lambda k: len(units[k][0][0])):
            t, r = len(units[k][0][0]), len(units[k][1])
            if t > self.N_CTX:
                raise ValueError(f"prompt too long for the GGUF backend: {t} tokens (max {self.N_CTX})")
            if cur and (n_tok + t > self.N_CTX or n_seq + r > self.N_SEQ):
                out.append(cur); cur, n_tok, n_seq = [], 0, 0
            cur.append(k); n_tok += t; n_seq += r
        return out + ([cur] if cur else [])

    def _decode(self, packs, idx):
        """One llama_decode for the packed rows idx -> readout logits [n_readouts, n_vocab] (float32)."""
        C, b = self.C, self.batch
        n, base, reads = 0, 0, []
        for k in idx:
            t, pos, seg, last, parent = packs[k]
            below = [[] for _ in parent]  # sequences (readouts) below every block of the trie
            for r, i in enumerate(last):
                blk = seg[i]
                while blk >= 0:
                    below[blk].append(base + r); blk = parent[blk]
            for tok, p, s in zip(t, pos, seg):
                b.token[n], b.pos[n], b.logits[n] = tok, p, 0
                b.n_seq_id[n] = len(below[s])
                for j, q in enumerate(below[s]):
                    b.seq_id[n][j] = q
                n += 1
            for i in last:
                b.logits[n - len(t) + i] = 1
                reads.append(n - len(t) + i)
            base += len(last)
        b.n_tokens = n
        C.llama_memory_clear(C.llama_get_memory(self.ctx), False)
        if C.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        return np.stack([np.ctypeslib.as_array(C.llama_get_logits_ith(self.ctx, r), shape=(self.n_vocab,))
                         for r in reads])

    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    def run_shared(self, groups, policy=None):
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._decode_chunks(units):
            lg = self._decode([u[0] for u in units], idx)
            pr = []
            for m, ids in enumerate(x for k in idx for x in units[k][1]):
                z = lg[m, list(ids)].astype(np.float64)
                e = np.exp(z - z.max())
                pr.append((e / e.sum()).tolist())
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)


class _HTTPLetters:
    """Shared part of the HTTP backends (Ollama, llama.cpp server): one request per prompt (one generated token with
    the top log-probabilities at the answer position), sent sequentially so that the engine's prompt cache reuses the
    state between the option orders and the questions of a request; `parallel` > 1 sends that many at once (for an
    engine started with several slots / OLLAMA_NUM_PARALLEL)."""

    TOP = 20  # top log-probabilities requested; the option letters dominate the answer position
    RETRIES = 3  # dropped connections retried per request

    def __init__(self, model_dir, url, parallel=1, timeout=120.0):
        import httpx
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefill = PREFILL
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        self.parallel = parallel
        self.pool = None
        if parallel > 1:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=parallel)

    def _post(self, path, body):
        """POST with a few retries on dropped connections (an engine restarting or closing an idle keep-alive)."""
        import time

        import httpx
        for k in range(self.RETRIES + 1):
            try:
                r = self.http.post(path, json=body)
                r.raise_for_status()
                return r.json()
            except httpx.TransportError:  # connection refused / reset, server disconnected
                if k == self.RETRIES:
                    raise
                time.sleep(0.5 * 2 ** k)

    def run(self, prompts, ids_list):
        jobs = list(zip(prompts, ids_list))
        lps = list(self.pool.map(lambda j: self._one(*j), jobs)) if self.pool else [self._one(*j) for j in jobs]
        return [torch.softmax(torch.tensor(x, dtype=torch.float32), -1).tolist() for x in lps]

    def run_shared(self, groups, policy=None):
        return _split(self.run([p for g in groups for p in g[0]], [x for g in groups for x in g[1]]), groups)

    def close(self):
        self.http.close()


class OllamaBackend(_HTTPLetters):
    """Ollama (GGUF exports: the -GGUF repositories; Apple Silicon, CPU, CUDA). The prompt is rendered and
    calibrated here (the model directory gives the tokenizer, chat template and CALIBRATION.json) and sent as a raw
    prompt (`raw: true`: no Ollama template) to /api/generate with one generated token and `logprobs` /
    `top_logprobs`; Ollama tokenizes it with the GGUF vocabulary. Letters missing from the top list get -1e9 (their
    log-probability is below the TOP-th token, i.e. practically zero next to the letters that are listed)."""

    takes_text = True  # Ollama accepts text only; the server skips its own tokenization

    def __init__(self, model_dir, url="http://127.0.0.1:11434", model=None, parallel=1, num_ctx=4096, keep_alive="30m"):
        super().__init__(model_dir, url, parallel)
        if not model:
            raise ValueError("--ollama-model is required for --mode ollama")
        self.name, self.num_ctx, self.keep_alive = model, num_ctx, keep_alive
        self.letter_text = {}
        self.checked = False

    def _raw(self, prompt):
        """The rendered prompt starts with the BOS text; the GGUF exports set add_bos_token, so Ollama adds BOS itself
        (sending it as text too would give two)."""
        bos = self.tok.bos_token
        return prompt[len(bos):] if bos and prompt.startswith(bos) else prompt

    def _text(self, i):
        if i not in self.letter_text:
            self.letter_text[i] = self.tok.decode([i])
        return self.letter_text[i]

    def body(self, prompt):
        return {"model": self.name, "prompt": prompt, "raw": True, "stream": False, "logprobs": True,
                "top_logprobs": self.TOP, "keep_alive": self.keep_alive,
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx}}

    @staticmethod
    def parse(resp, letters):
        """/api/generate response -> log-probability of each letter text at the first generated position."""
        first = resp["logprobs"][0]
        got = {x["token"]: float(x["logprob"]) for x in first.get("top_logprobs") or []}
        got.setdefault(first["token"], float(first["logprob"]))
        return [got.get(t, -1e9) for t in letters]

    def _one(self, prompt, ids):
        if not isinstance(prompt, str):
            prompt = self.tok.decode(prompt)
        out = self._post("/api/generate", self.body(self._raw(prompt)))
        if not self.checked:  # first request: Ollama's token count must equal ours (same tokenizer, one BOS)
            self.checked = True
            n, got = len(self.tok(prompt, add_special_tokens=False).input_ids), out.get("prompt_eval_count")
            if got is not None and got != n:
                import warnings
                warnings.warn(f"Ollama evaluated {got} prompt tokens, the basal tokenizer gives {n}: the GGUF vocabulary "
                              "differs from the model's tokenizer (use the GGUF files of the -GGUF repositories)")
        return self.parse(out, [self._text(i) for i in ids])


class LlamaCppBackend(_HTTPLetters):
    """llama.cpp server (`llama-server -m model.gguf`): the same readout from /completion with the prompt given as
    token ids (exactly the server's tokens) and `n_probs`; the log-probabilities are those of the raw logits
    (`post_sampling_probs` off). The prompt cache (`cache_prompt`) shares the state between requests of one slot."""

    def body(self, ids):
        return {"prompt": list(ids), "n_predict": 1, "n_probs": self.TOP, "temperature": 0.0, "cache_prompt": True,
                "post_sampling_probs": False, "return_tokens": False}

    @staticmethod
    def parse(resp, ids):
        first = resp["completion_probabilities"][0]
        got = {int(x["id"]): float(x["logprob"]) for x in first.get("top_logprobs") or []}
        got.setdefault(int(first["id"]), float(first["logprob"]))
        return [got.get(int(i), -1e9) for i in ids]

    def _one(self, prompt, ids):
        toks = prompt if not isinstance(prompt, str) else self.tok(prompt, add_special_tokens=False).input_ids
        return self.parse(self._post("/completion", self.body(toks)), ids)


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
