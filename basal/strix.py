"""`--mode strix`: a basal-1.5 / basal-2.0 forward written for AMD Strix Halo (Radeon 8060S, gfx1151, ROCm), with
Triton kernels where the stock ones fall short on this GPU (docs/benchmarks/STRIX_HALO.md, "--mode strix"). Speeds at
~6k tokens:

  attention   flash attention over the packed prefix trie, GQA-packed (the query heads of a kv head share each key /
              value tile). GraphBackend's block mask becomes an interval per key: depth-first packing puts every
              block's subtree in one contiguous range, so key k is visible to query q iff k <= q < sub_end[k]. Key
              blocks no query of a block can see are skipped, blocks every query sees run without a mask, so a packed
              row costs what a causal row of the same length costs. ~20 TFLOPS; AOTriton's flash attention ~10.
  FFN input   gate and up projections, their biases and SwiGLU in one kernel (two accumulators per output tile, one
              load of the activations for both): half the output bytes, no activation pass. ~32 TFLOPS; F.linear +
              silu * up ~17.
  qkv, o,     matmuls with the bias, and for o / down the residual add, in the epilogue; tile configs per row count.
  down        Faster than F.linear + add on every shape measured (o at 6k tokens 2.4 vs 3.7 ms per layer).
  RMSNorm,    small kernels; rotary writes q / k / v head-major for the attention kernel.
  rotary

The last layer runs attention, o and the FFN for the readout rows only. q, k and v are one projection (weights
concatenated at start-up; the Hugging Face modules keep views of them, so the evidence head still runs the stock
model). Rounding follows the Hugging Face bf16 forward step by step (RMSNorm in fp32 then the weight in bf16, rotary as
round(q cos) + round(rot(q) sin), silu(gate) and its product with up rounded to bf16, bias then residual each rounded),
so answers match the other bf16 engines. Every packed row of a chunk is one sequence (no padding, no shapes, no
graphs): rows of different requests are separate trees of the same interval mask."""
import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine import GraphBackend
from .prompt import PREFILL


# -- kernels -------------------------------------------------------------------------------------------------------
@triton.jit
def _attn_fwd(Q, K, V, O, SUB, KMIN, KMAX, TOK, L, NQ, som, scale,
              HPG: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, GATHER: tl.constexpr):
    """One program = BT query tokens x the HPG query heads of one kv head (GQA packed: each key / value tile is
    loaded once for the whole group). Q [H, L, D], K / V [KV, L, D] head-major; O [NQ, H * D]. GATHER: the query
    tokens are TOK[0:NQ] (ascending), else 0..L-1."""
    pid_t = tl.program_id(0)
    kvh = tl.program_id(1)
    R: tl.constexpr = BT * HPG
    r = tl.arange(0, R)
    head = kvh * HPG + r // BT
    qi = pid_t * BT + r % BT  # query index (row of O)
    live = qi < NQ
    if GATHER:
        tok = tl.load(TOK + qi, mask=live, other=0)
        q0 = tl.load(TOK + pid_t * BT)
        q_last = tl.load(TOK + tl.minimum(pid_t * BT + BT, NQ) - 1)
    else:
        tok = qi
        q0 = pid_t * BT
        q_last = tl.minimum(q0 + BT, L) - 1
    rd = tl.arange(0, D)
    q = tl.load(Q + (head[:, None] * L + tok[:, None]) * D + rd[None, :], mask=live[:, None], other=0.0)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    qs = scale * 1.4426950408889634  # exp2 below
    for k0 in range(0, q_last + 1, BN):  # causal: no key after the block's last query
        kb = k0 // BN
        if tl.load(KMAX + kb) > q0:  # some key of the block is still visible to some query of this block
            rn = k0 + tl.arange(0, BN)
            k = tl.load(K + (kvh * L + rn[None, :]) * D + rd[:, None], mask=rn[None, :] < L, other=0.0)
            s = tl.dot(q, k) * qs
            if ((k0 + BN - 1) > q0) | (tl.load(KMIN + kb) <= q_last) | GATHER:  # not all keys visible to all
                se = tl.load(SUB + rn, mask=rn < L, other=0)
                vis = (rn[None, :] <= tok[:, None]) & (tok[:, None] < se[None, :])
                s = tl.where(vis, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            # a row that has seen no visible key yet (rows of a later tree in the sequence) keeps -inf: subtract 0
            # instead, so its p and alpha are 0 rather than nan
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(s - m_use[:, None])
            alpha = tl.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + (kvh * L + rn[:, None]) * D + rd[None, :], mask=rn[:, None] < L, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + qi[:, None] * som + head[:, None] * D + rd[None, :], acc.to(O.dtype.element_ty), mask=live[:, None])


@triton.jit
def _swiglu_mm(x_ptr, wg_ptr, wu_ptr, bg_ptr, bu_ptr, o_ptr, M, N, K, sxm, swn, som,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = GM * npn
    first = (pid // group) * GM
    gsz = min(npm - first, GM)
    pm = first + ((pid % group) % gsz)
    pn = (pid % group) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None] * sxm + rk[None, :]
    gp = wg_ptr + rn[None, :] * swn + rk[:, None]
    up = wu_ptr + rn[None, :] * swn + rk[:, None]
    ag = tl.zeros((BM, BN), dtype=tl.float32)
    au = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        xv = tl.load(xp, mask=rm[:, None] < M, other=0.0)
        ag += tl.dot(xv, tl.load(gp))
        au += tl.dot(xv, tl.load(up))
        xp += BK
        gp += BK
        up += BK
    g = (ag + tl.load(bg_ptr + rn).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    u = (au + tl.load(bu_ptr + rn).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + rm[:, None] * som + rn[None, :], (s * u).to(tl.bfloat16), mask=rm[:, None] < M)


@triton.jit
def _linear(x_ptr, w_ptr, b_ptr, r_ptr, o_ptr, M, N, K, sxm, swn, som,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr, RES: tl.constexpr):
    """o = x @ w^T + b, rounded to bf16 as F.linear; RES: then + r in bf16 (the residual add; o may alias r)."""
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = GM * npn
    first = (pid // group) * GM
    gsz = min(npm - first, GM)
    pm = first + ((pid % group) % gsz)
    pn = (pid % group) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None] * sxm + rk[None, :]
    wp = w_ptr + rn[None, :] * swn + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(xp, mask=rm[:, None] < M, other=0.0), tl.load(wp))
        xp += BK
        wp += BK
    o = (acc + tl.load(b_ptr + rn).to(tl.float32)[None, :]).to(tl.bfloat16)
    if RES:
        r = tl.load(r_ptr + rm[:, None] * som + rn[None, :], mask=rm[:, None] < M, other=0.0)
        o = (o.to(tl.float32) + r.to(tl.float32)).to(tl.bfloat16)
    tl.store(o_ptr + rm[:, None] * som + rn[None, :], o, mask=rm[:, None] < M)


@triton.jit
def _rmsnorm(h_ptr, w_ptr, xo_ptr, eps, N: tl.constexpr):
    """xo = w * rmsnorm(h), as the stock LlamaRMSNorm: fp32 statistics, normalised value cast to bf16, then the
    weight multiplied in bf16."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    hf = tl.load(h_ptr + row * N + offs).to(tl.float32)
    y = (hf * tl.rsqrt(tl.sum(hf * hf, 0) / N + eps)).to(tl.bfloat16)
    tl.store(xo_ptr + row * N + offs, (tl.load(w_ptr + offs).to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16))


@triton.jit
def _rope(x_ptr, pos_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, L, sxm, H: tl.constexpr, KV: tl.constexpr,
          HALF: tl.constexpr):
    """Row `row` of the qkv projection -> q [H, L, D], k / v [KV, L, D] (head-major, contiguous: what the attention
    kernel reads fastest), rotary applied to q and k."""
    row = tl.program_id(0)
    head = tl.program_id(1)  # q heads, k heads, v heads: consecutive 2 * HALF blocks of the qkv row
    offs = tl.arange(0, HALF)
    base = x_ptr + row * sxm + head * 2 * HALF
    x1 = tl.load(base + offs)
    x2 = tl.load(base + HALF + offs)
    if head < H + KV:
        p = tl.load(pos_ptr + row)
        c = tl.load(cos_ptr + p * HALF + offs).to(tl.float32)
        s = tl.load(sin_ptr + p * HALF + offs).to(tl.float32)
        f1, f2 = x1.to(tl.float32), x2.to(tl.float32)
        bf = tl.bfloat16
        x1 = ((f1 * c).to(bf).to(tl.float32) + (-f2 * s).to(bf).to(tl.float32)).to(bf)
        x2 = ((f2 * c).to(bf).to(tl.float32) + (f1 * s).to(bf).to(tl.float32)).to(bf)
    if head < H:
        dst = q_ptr + (head * L + row) * 2 * HALF
    elif head < H + KV:
        dst = k_ptr + ((head - H) * L + row) * 2 * HALF
    else:
        dst = v_ptr + ((head - H - KV) * L + row) * 2 * HALF
    tl.store(dst + offs, x1)
    tl.store(dst + HALF + offs, x2)


# -- backend -------------------------------------------------------------------------------------------------------
class StrixHaloFastBackend(GraphBackend):
    """`--mode strix` (see the module docstring). Rows of one chunk are concatenated into a single sequence of at
    most TOKEN_BUDGET tokens; a packed row longer than that is bisected like in GraphBackend (exact). A single prompt
    longer than that runs alone in one forward, as GraphBackend runs one past its largest graph (the server keeps
    prompts within the model's context; activations grow linearly, attention memory does not grow with length)."""

    streams, inflight = False, 1  # its own run_shared: no early readouts, one call at a time
    TOKEN_BUDGET = 16384  # tokens per forward (the GGUF context): ~0.4 GB of FFN activations
    # tile configs measured on the 8060S (Triton 3.7, ROCm 7.2), inside the full forward where it mattered
    ATTN = dict(BT=8, BN=32, num_warps=8, num_stages=1)  # ~20 TFLOPS at 6k tokens
    # FFN input (gate / up / SwiGLU): (largest row count, config), first match wins; ~32 TFLOPS at 6k tokens
    FFN = ((512, dict(BM=128, BN=32, BK=64, GM=8, num_warps=4, num_stages=1, waves_per_eu=2)),
           (None, dict(BM=128, BN=64, BK=64, GM=16, num_warps=8, num_stages=1, waves_per_eu=1)))
    # projections: (largest row count, config) per shape, first match wins; round-robin sweeps on the 8060S
    LINEAR = {
        "qkv": ((512, dict(BM=64, BN=64, BK=64, GM=8, num_warps=8, num_stages=2)),
                (1536, dict(BM=128, BN=64, BK=64, GM=8, num_warps=8, num_stages=2)),
                (None, dict(BM=64, BN=128, BK=64, GM=8, num_warps=8, num_stages=2))),
        "o": ((512, dict(BM=128, BN=64, BK=64, GM=8, num_warps=4, num_stages=2)),
              (None, dict(BM=128, BN=64, BK=64, GM=4, num_warps=8, num_stages=2))),
        "down": ((512, dict(BM=64, BN=64, BK=32, GM=8, num_warps=4, num_stages=1)),
                 (None, dict(BM=128, BN=128, BK=64, GM=8, num_warps=4, num_stages=2)))}
    LINEAR_ROWS = 2304  # qkv and o run in row chunks of at most this many (they slow down past ~3k rows)

    @classmethod
    def unsupported(cls, cfg):
        """What in a model config the kernels cannot run (empty: supported). Every tile config must divide its
        projection (the matmuls have no column or depth masks), tl.arange needs power-of-two extents (head size, query
        heads per kv head, hidden size), and the rotary tables are computed once at start-up. The forward is Llama's
        (basal-1.5, basal-2.0): other architectures (basal-omni is Gemma 4) are refused before their shapes are read."""
        if getattr(cfg, "model_type", None) != "llama":
            return [f"model type {getattr(cfg, 'model_type', None)} (Llama only: basal-1.5, basal-2.0)"]
        H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        hidden, inter = cfg.hidden_size, cfg.intermediate_size
        pow2 = lambda n: n > 0 and n & (n - 1) == 0  # noqa: E731
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        rope = rope.get("rope_type", "default")
        out = []
        if cfg.hidden_act != "silu":
            out.append(f"activation {cfg.hidden_act} (silu only)")
        if not pow2(hidden):
            out.append(f"hidden size {hidden} (a power of two)")
        if not pow2(D) or D < 16:
            out.append(f"head size {D} (a power of two, at least 16)")
        if H % KV or not pow2(H // KV):
            out.append(f"{H} query heads over {KV} kv heads (a power-of-two ratio)")
        if "dynamic" in rope or rope == "longrope":
            out.append(f"RoPE type {rope} (its frequencies depend on the sequence length)")
        for name, n, k, cfgs in (("qkv", (H + 2 * KV) * D, hidden, cls.LINEAR["qkv"]),
                                 ("o", hidden, H * D, cls.LINEAR["o"]), ("down", hidden, inter, cls.LINEAR["down"]),
                                 ("gate / up", inter, hidden, cls.FFN)):
            bn, bk = max(c["BN"] for _, c in cfgs), max(c["BK"] for _, c in cfgs)
            if n % bn or k % bk:
                out.append(f"{name} projection {k} -> {n} (multiples of {bk} -> {bn})")
        return out

    def __init__(self, model_dir, dtype="bfloat16", warm=True):
        if dtype != "bfloat16":
            raise ValueError("--mode strix runs bf16 only")
        if not torch.version.hip:
            raise ValueError("--mode strix needs PyTorch built for ROCm (its kernels and tile configs are for AMD "
                             "Strix Halo)")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
        self.dev = next(self.model.parameters()).device
        self.prefill = PREFILL
        self.shared = True
        self._wrows = {}
        cfg = self.model.config
        if problems := self.unsupported(cfg):
            raise ValueError("--mode strix cannot run this model: " + "; ".join(problems))
        self.H, self.KV = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.hidden, self.eps = cfg.hidden_size, cfg.rms_norm_eps
        m = self.model.model
        self.layers = []
        for layer in m.layers:
            a, f = layer.self_attn, layer.mlp
            w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous()
            b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).contiguous() \
                if a.q_proj.bias is not None else None
            nq, nk = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
            for mod, lo, hi in ((a.q_proj, 0, nq), (a.k_proj, nq, nq + nk), (a.v_proj, nq + nk, w.shape[0])):
                mod.weight.data = w[lo:hi]  # views: the stock modules (evidence head) share the fused weights
                if b is not None:
                    mod.bias.data = b[lo:hi]
            zeros = lambda n: torch.zeros(n, device=self.dev, dtype=torch.bfloat16)  # noqa: E731 (no bias)
            bias = lambda mod: mod.bias if mod.bias is not None else zeros(mod.weight.shape[0])  # noqa: E731
            self.layers.append((layer.input_layernorm.weight, w, b if b is not None else zeros(w.shape[0]),
                                a.o_proj.weight, bias(a.o_proj), layer.post_attention_layernorm.weight,
                                f.gate_proj.weight.contiguous(), bias(f.gate_proj), f.up_proj.weight.contiguous(),
                                bias(f.up_proj), f.down_proj.weight.contiguous(), bias(f.down_proj)))
        torch.cuda.empty_cache()
        self.embed, self.final_norm = m.embed_tokens.weight, m.norm.weight
        # rotary tables as the stock model computes them: fp32 angles, cos / sin scaled, then cast to bf16
        rot = m.rotary_emb
        pos = torch.arange(cfg.max_position_embeddings, device=self.dev, dtype=torch.float32)
        ang = pos[:, None] * rot.inv_freq.float()[None, :]
        self.cos = (ang.cos() * rot.attention_scaling).to(torch.bfloat16).contiguous()
        self.sin = (ang.sin() * rot.attention_scaling).to(torch.bfloat16).contiguous()
        if warm:  # compile every kernel once (Triton JIT), short and multi-block rows; within vocabulary and context
            V = self.embed.shape[0]
            for n in (min(300, self.cos.shape[0] - 1), min(1100, self.cos.shape[0] - 1)):
                toks = [[1] + list(range(1000, 1000 + n)), [1] + list(range(1000, 1000 + n // 2)) + [7, 8]]
                self.run_shared([([[t % V for t in x] for x in toks], [[30 % V, 31 % V]] * 2)])
            self._warm_shapes()

    # Start-up forwards for attention and rotary (one tile config each): sequence lengths other and a multiple of 16,
    # each against readout rows 1, other and a multiple of 16 (the last layer's attention takes both counts), and a
    # sequence of one token (a request that packs to one). They run the first layer and the last only: every layer
    # between needs the copies the first one compiles.
    WARM_FORWARDS = ((1, 1), (17, 1), (32, 1), (17, 3), (32, 3), (17, 16), (32, 16))

    @staticmethod
    def warm_rows(cfgs):
        """Row counts that cover a (largest row count, config) list: 1 (the last row chunk of qkv / o, one readout
        row), then in every range one count that is not a multiple of 16 and one that is."""
        rows, lo = [1], 0
        for lim, _ in cfgs:
            rows += [lo + 17 + ((lo + 17) % 16 == 0), (lo // 16 + 2) * 16]
            lo = lim
        return rows

    def _warm_row(self, L, nq):
        """Inputs of `forward` for one packed row of L tokens and its last nq tokens read out: causal sequences of
        at most the context each (positions within the rotary tables), token ids within the vocabulary."""
        i = torch.arange(L, device=self.dev)
        P = self.cos.shape[0]
        ends = torch.clamp((i // P + 1) * P, max=L)
        return (i + 1000) % self.embed.shape[0], (i % P).to(torch.int32), ends.to(torch.int32), i[L - nq:]

    @torch.no_grad()
    def _warm_shapes(self):
        """Compile every kernel copy a request can need. Triton compiles a kernel once per tile config and per class of
        each integer argument: 1, a multiple of 16, other. The row count picks both (the config by its range in FFN /
        LINEAR, the class by its value), and a request that needs a copy not compiled yet waits for it: ~0.2-0.3 s per
        copy on the 8060S, ~1 ms when Triton's disk cache holds it (in the request matrix, the first timed pass of S
        with 1 question and of M with 2 did). The projections run on zero rows of warm_rows, as the forward calls them
        (same strides, residual for o / down); attention and rotary in the short WARM_FORWARDS, over the first and the
        last layer (all 60 layers of basal-2.0 took ~0.7 s per start)."""
        z = lambda m, n: torch.zeros(m, n, device=self.dev, dtype=torch.bfloat16)  # noqa: E731
        _, wqkv, bqkv, wo, bo, _, wg, bg, wu, bu, wd, bd = self.layers[0]
        for m in self.warm_rows(self.LINEAR["qkv"]):
            self._linear("qkv", z(m, self.hidden), wqkv, bqkv)
        for m in self.warm_rows(self.LINEAR["o"]):
            self._linear("o", z(m, wo.shape[1]), wo, bo, z(m, self.hidden))
        for m in self.warm_rows(self.LINEAR["down"]):
            self._linear("down", z(m, wd.shape[1]), wd, bd, z(m, self.hidden))
        for m in self.warm_rows(self.FFN):
            self._ffn_in(z(m, self.hidden), wg, bg, wu, bu)
        layers = self.layers
        self.layers = layers[:1] + layers[-1:] if len(layers) > 1 else layers
        try:
            for L, nq in self.WARM_FORWARDS:
                self.forward(*self._warm_row(L, nq))
            self._wait()
        finally:
            self.layers = layers

    def _oversize(self, pack, n):
        return len(pack[0]) > self.TOKEN_BUDGET

    def _chunks_tok(self, units):
        """Units sorted by length, cut into chunks of at most TOKEN_BUDGET tokens."""
        out, cur, n = [], [], 0
        for k in sorted(range(len(units)), key=lambda k: len(units[k][0][0])):
            t = len(units[k][0][0])
            if cur and n + t > self.TOKEN_BUDGET:
                out.append(cur); cur, n = [], 0
            cur.append(k); n += t
        return out + ([cur] if cur else [])

    @staticmethod
    def _sub_end(pack, off):
        """Per token of a packed row: the end of its block's subtree (global index, row starting at off)."""
        t, pos, seg, last, parent = pack
        own_end = [0] * len(parent)
        for i, s in enumerate(seg):
            own_end[s] = i + 1
        sub = own_end[:]
        for b in range(len(parent) - 1, -1, -1):  # children follow their parent (depth-first order)
            if parent[b] >= 0:
                sub[parent[b]] = max(sub[parent[b]], sub[b])
        return [off + sub[s] for s in seg]

    def _qkv_heads(self, qkv, pos):
        """qkv projection -> rotary q, k and v, head-major [heads, L, D]."""
        L = qkv.shape[0]
        H, KV, D = self.H, self.KV, self.D
        q = torch.empty(H, L, D, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(KV, L, D, device=qkv.device, dtype=qkv.dtype)
        v = torch.empty(KV, L, D, device=qkv.device, dtype=qkv.dtype)
        _rope[(L, H + 2 * KV)](qkv, pos, self.cos, self.sin, q, k, v, L, qkv.stride(0), H=H, KV=KV, HALF=D // 2,
                               num_warps=1)
        return q, k, v

    def _key_bounds(self, sub):
        """Smallest and largest sub_end of every key block of the attention kernel: blocks no query of a query block
        can see are skipped, blocks every query sees run without a mask."""
        BN = self.ATTN["BN"]
        nb = triton.cdiv(sub.shape[0], BN)
        pad = nb * BN - sub.shape[0]
        kmin = F.pad(sub, (0, pad), value=2 ** 30).view(nb, BN).amin(1).contiguous()
        kmax = F.pad(sub, (0, pad), value=0).view(nb, BN).amax(1).contiguous()
        return kmin, kmax

    def _attention(self, q, k, v, sub, kmin, kmax, tok=None):
        """Attention of every token (tok None) or of the query tokens tok (ascending) -> [n, H * D]."""
        H, L, D = q.shape
        n = L if tok is None else tok.shape[0]
        o = torch.empty(n, H * D, device=q.device, dtype=q.dtype)
        c = self.ATTN
        _attn_fwd[(triton.cdiv(n, c["BT"]), self.KV)](q, k, v, o, sub, kmin, kmax, tok if tok is not None else sub,
                                                      L, n, o.stride(0), 1.0 / math.sqrt(D), HPG=H // self.KV, D=D,
                                                      BT=c["BT"], BN=c["BN"], GATHER=tok is not None,
                                                      num_warps=c["num_warps"], num_stages=c["num_stages"])
        return o

    def _ffn_in(self, x, wg, bg, wu, bu):
        M, K = x.shape
        N = wg.shape[0]
        o = torch.empty(M, N, device=x.device, dtype=x.dtype)
        c = next(c for lim, c in self.FFN if lim is None or M <= lim)
        grid = (triton.cdiv(M, c["BM"]) * (N // c["BN"]),)
        _swiglu_mm[grid](x, wg, wu, bg, bu, o, M, N, K, x.stride(0), wg.stride(0), o.stride(0), **c)
        return o

    def _linear(self, kind, x, w, b, r=None):
        """x @ w^T + b (+ r, written into r)."""
        M, K = x.shape
        N = w.shape[0]
        o = r if r is not None else torch.empty(M, N, device=x.device, dtype=x.dtype)
        step = M if kind == "down" else self.LINEAR_ROWS
        for a in range(0, M, step):
            xa, oa = x[a: a + step], o[a: a + step]
            m = xa.shape[0]
            c = next(c for lim, c in self.LINEAR[kind] if lim is None or m <= lim)
            _linear[(triton.cdiv(m, c["BM"]) * (N // c["BN"]),)](xa, w, b, oa, oa, m, N, K, xa.stride(0), w.stride(0),
                                                                 oa.stride(0), RES=r is not None, **c)
        return o

    def _norm(self, h, w):
        x = torch.empty_like(h)
        _rmsnorm[(h.shape[0],)](h, w, x, self.eps, N=self.hidden, num_warps=4)
        return x

    @torch.no_grad()
    def forward(self, ids, pos, sub, reads):
        """One sequence of packed rows -> final-normalised hidden states at the readout rows. The last layer runs
        attention, o and the FFN for the readout rows only (nothing else reads its output)."""
        kmin, kmax = self._key_bounds(sub)
        h = F.embedding(ids, self.embed)
        x = self._norm(h, self.layers[0][0])
        n = len(self.layers)
        for i, (_, wqkv, bqkv, wo, bo, ln2, wg, bg, wu, bu, wd, bd) in enumerate(self.layers):
            q, k, v = self._qkv_heads(self._linear("qkv", x, wqkv, bqkv), pos)
            if i + 1 < n:
                a = self._attention(q, k, v, sub, kmin, kmax)
            else:  # last layer: readout rows only (sorted for the attention kernel)
                rs, order = torch.sort(reads)
                a = self._attention(q, k, v, sub, kmin, kmax, tok=rs.to(torch.int32))
                h = h[rs]
            x = self._norm(self._linear("o", a, wo, bo, h), ln2)  # h += o_proj(a)
            g = self._ffn_in(x, wg, bg, wu, bu)
            x = self._norm(self._linear("down", g, wd, bd, h),  # h += down_proj(g)
                           self.layers[i + 1][0] if i + 1 < n else self.final_norm)
        return x[torch.argsort(order)]

    @staticmethod
    def _wait():
        """Wait for the GPU by polling an event with short sleeps: ROCm's synchronize spins a CPU core at 100% for
        the whole forward (seconds on long rows), this costs at most half a millisecond."""
        ev = torch.cuda.Event()
        ev.record()
        while not ev.query():
            time.sleep(0.0005)

    @torch.no_grad()
    def run(self, prompts, ids_list):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    @torch.no_grad()
    def run_shared(self, groups, policy=None):
        units = self._units(groups)
        probs = [None] * len(units)
        for idx in self._chunks_tok(units):
            ids, pos, sub, reads, off = [], [], [], [], 0
            for k in idx:
                t, pp, sg, last, par = pk = units[k][0]
                ids += t; pos += pp; sub += self._sub_end(pk, off); reads += [off + i for i in last]
                off += len(t)
            if max(pos) >= self.cos.shape[0]:
                raise ValueError(f"prompt too long for --mode strix: position {max(pos)}, the model has "
                                 f"{self.cos.shape[0]}")
            dev = self.dev
            h = self.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev, dtype=torch.int32),
                             torch.tensor(sub, device=dev, dtype=torch.int32), torch.tensor(reads, device=dev))
            self._wait()
            pr = self._readout(h, [x for k in idx for x in units[k][1]])
            j = 0
            for k in idx:
                n = len(units[k][1]); probs[k] = pr[j: j + n]; j += n
        return self._assemble(groups, units, probs)
