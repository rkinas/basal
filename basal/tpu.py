"""Google TPU backend
"""
import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .engine import GraphBackend
from .prompt import PREFILL

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}
DTYPES = ("bfloat16", "float32")  # weight / activation dtypes of the TPU backend


def load_safetensors(path):
    """Memory-map a .safetensors file into numpy arrays without torch (the safetensors numpy API cannot read BF16)."""
    from ml_dtypes import bfloat16
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", mm[:8].tobytes())[0]
    header = json.loads(mm[8: 8 + n].tobytes())
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = bfloat16 if meta["dtype"] == "BF16" else _ST_DTYPES.get(meta["dtype"])
        if dt is None:
            raise ValueError(f"{path}: unsupported safetensors dtype {meta['dtype']} ({name})")
        a, b = meta["data_offsets"]
        out[name] = mm[8 + n + a: 8 + n + b].view(dt).reshape(meta["shape"])
    return out


def model_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if cfg.get("model_type") != "llama":
        raise ValueError(f"TPU backend supports Llama-architecture checkpoints, got {cfg.get('model_type')!r}")
    rope = cfg.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or cfg.get("rope_scaling"):
        raise ValueError(f"TPU backend supports default RoPE only, got {rope or cfg.get('rope_scaling')}")
    nh = cfg["num_attention_heads"]
    return dict(layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"], ffn=cfg["intermediate_size"], heads=nh,
                kv_heads=cfg.get("num_key_value_heads", nh), head_dim=cfg.get("head_dim") or cfg["hidden_size"] // nh,
                eps=cfg.get("rms_norm_eps", 1e-6), theta=float(rope.get("rope_theta", cfg.get("rope_theta", 10000.0))),
                max_pos=cfg.get("max_position_embeddings"))


def stack_params(w, c, dtype="bfloat16"):
    """HF-named weights -> host pytree with the decoder layers stacked on axis 0, q / k / v fused, linear weights
    [in, out]. Biases are always present (zeros when the checkpoint has none), so the compiled program does not depend
    on the checkpoint."""
    from ml_dtypes import bfloat16
    if dtype not in DTYPES:
        raise ValueError(f"--mode tpu supports --dtype {' / '.join(DTYPES)}, got {dtype!r}")
    dt = {"bfloat16": bfloat16, "float32": np.float32}[dtype]
    L, H, F = c["layers"], c["hidden"], c["ffn"]
    Q, KV = c["heads"] * c["head_dim"], c["kv_heads"] * c["head_dim"]
    shapes = dict(ln1=(H,), wqkv=(H, Q + 2 * KV), bqkv=(Q + 2 * KV,), wo=(Q, H), bo=(H,), ln2=(H,),
                  wg=(H, F), bg=(F,), wu=(H, F), bu=(F,), wd=(F, H), bd=(H,))
    lay = {k: np.zeros((L, *s), dt) for k, s in shapes.items()}

    def get(name):
        return np.asarray(w[name]).astype(dt)

    def bias(name, out, sl=slice(None)):
        if name in w:
            out[sl] = get(name)

    def fill(i):
        p = f"model.layers.{i}"
        lay["ln1"][i], lay["ln2"][i] = get(f"{p}.input_layernorm.weight"), get(f"{p}.post_attention_layernorm.weight")
        for name, sl in (("q", slice(0, Q)), ("k", slice(Q, Q + KV)), ("v", slice(Q + KV, None))):
            lay["wqkv"][i, :, sl] = get(f"{p}.self_attn.{name}_proj.weight").T
            bias(f"{p}.self_attn.{name}_proj.bias", lay["bqkv"][i], sl)
        lay["wo"][i] = get(f"{p}.self_attn.o_proj.weight").T
        bias(f"{p}.self_attn.o_proj.bias", lay["bo"][i])
        for name, wk, bk in (("gate", "wg", "bg"), ("up", "wu", "bu"), ("down", "wd", "bd")):
            lay[wk][i] = get(f"{p}.mlp.{name}_proj.weight").T
            bias(f"{p}.mlp.{name}_proj.bias", lay[bk][i])

    with ThreadPoolExecutor(min(16, os.cpu_count() or 1)) as ex:  # numpy copies release the GIL: ~5x faster load
        list(ex.map(fill, range(L)))
    embed = get("model.embed_tokens.weight")
    head = get("lm_head.weight") if "lm_head.weight" in w else embed  # tied embeddings
    return dict(embed=embed, norm=get("model.norm.weight"), lm_head=head, layers=lay)


def load_params(model_dir, dtype="bfloat16", device=None):
    """Checkpoint directory -> (config, parameter pytree on `device`, default: JAX's default device)."""
    import jax
    c = model_config(model_dir)
    w = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        w.update(load_safetensors(f))
    return c, jax.device_put(stack_params(w, c, dtype), device)


def forward(params, ids, pos, end, *, c, q_block=None, kv_block=None):
    """ids / pos / end (b, L) int32 (end: see tpu_kernels.tree_mask) -> last-layer hidden states (b, L, H), before the
    final norm. Stays on the device; `readout` reads the letter logits. q_block / kv_block: blocked attention for long
    prompts (see tpu_kernels.attention)."""
    import jax

    from . import tpu_kernels as K
    b, L = ids.shape
    nh, nkv, hd, H = c["heads"], c["kv_heads"], c["head_dim"], c["hidden"]
    Q, KV = nh * hd, nkv * hd
    cos, sin = K.rope_cache(L, hd, c["theta"])  # positions never exceed the row length
    scale = 1.0 / math.sqrt(hd)

    def layer(h, lp):
        x = K.rms_norm(h.reshape(b * L, H), lp["ln1"], c["eps"])
        qkv = K.linear(x, lp["wqkv"], lp["bqkv"])
        q = K.rope(qkv[:, :Q].reshape(b, L, nh, hd), pos, cos, sin)
        k = K.rope(qkv[:, Q: Q + KV].reshape(b, L, nkv, hd), pos, cos, sin)
        v = qkv[:, Q + KV:].reshape(b, L, nkv, hd)
        o = K.attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), end, scale,
                        q_block, kv_block)
        h = h + K.linear(o.transpose(0, 2, 1, 3).reshape(b * L, Q), lp["wo"], lp["bo"]).reshape(b, L, H)
        x = K.rms_norm(h.reshape(b * L, H), lp["ln2"], c["eps"])
        f = K.swiglu(x, lp["wg"], lp["bg"], lp["wu"], lp["bu"])
        return h + K.linear(f, lp["wd"], lp["bd"]).reshape(b, L, H), None

    h, _ = jax.lax.scan(layer, params["embed"][ids], params["layers"])
    return h


def readout(params, h, rows, cols, lids, *, c):
    """h (b, L, H), rows / cols (n,) int32 readout positions, lids (S,) int32 output-head rows -> fp32 logits (n, S)
    of the next token. Only the letter rows of the head: the full-vocabulary normaliser cancels in the letter
    softmax."""
    import jax
    import jax.numpy as jnp

    from . import tpu_kernels as K
    hr = K.rms_norm(h[rows, cols], params["norm"], c["eps"])
    return jnp.einsum("nh,sh->ns", hr.astype(jnp.float32), params["lm_head"][lids].astype(jnp.float32),
                      precision=jax.lax.Precision.HIGHEST)


def subtree_ends(seg, parent):
    """Packed row (block id per token, parent block per block; GraphBackend._pack) -> per token, the end of its block's
    subtree. _pack flattens the trie depth-first, so a block and its descendants are one contiguous run of tokens."""
    end = [0] * len(parent)
    for t, k in enumerate(seg):
        end[k] = t + 1
    for k in range(len(parent) - 1, 0, -1):  # children come after their parent
        if parent[k] >= 0:
            end[parent[k]] = max(end[parent[k]], end[k])
    return [end[k] for k in seg]


def tpu_available():
    """True when JAX is installed and sees a TPU. Only the default-mode probe: a JAX that fails to import or to
    initialise counts as no TPU (an explicit --mode tpu shows the error, TPUBackend)."""
    import importlib.util
    if importlib.util.find_spec("jax") is None:
        return False
    try:
        import jax
        return bool(jax.devices("tpu"))
    except Exception:
        return False


def _softmax(x):
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


class TPUBackend(GraphBackend):
    """Google TPU through JAX / XLA (plain-JAX kernels, bf16 or fp32), experimental and community-maintained.
    GraphBackend's units (every group -- all option orders of a question, or every branch of a request: SOAM -- is
    the compressed prefix trie of its prompts) packed as ONE row each: the attention mask comes from the trie, so a
    unit runs whole whatever its branches. Units are batched by length into forwards of at most TOKEN_BUDGET padded
    tokens. The forward and the small readout (always READOUTS positions, more in several calls) compile once per
    (rows, length) bucket; with warm=True all of them compile at start-up.

    Units longer than the largest bucket run alone in one of the coarse LONG_LENS buckets up to max_len, with a
    blocked attention (Q_BLOCK queries x KV_BLOCK keys, online softmax), so memory grows with the length instead of
    its square; a group longer than max_len is bisected, a single prompt longer than max_len is refused (the server
    truncates the state first: `context`)."""

    # a coarser grid than the CUDA graphs': every shape is an XLA compilation (about 2 s on a v5e) at start-up
    LENS = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072]
    BATCHES = [1, 2, 4, 8, 16, 32]
    LETTER_SLOTS = 256  # output-head rows read per position (format 2: up to 255 option codes); fixed, so new code ids
    #                     never recompile
    streams = True  # run_shared(..., done=callback): a group is reported as soon as its forwards are read out
    inflight = 1    # one engine call at a time (one device; the call's forwards are dispatched back to back)
    LONG_LENS = [4096, 6144, 8192, 12288, 16384, 24576, 32768]  # above LENS[-1]; multiples of Q_BLOCK and KV_BLOCK
    Q_BLOCK = 1024  # queries per attention block for lengths above LENS[-1]
    KV_BLOCK = 1024  # keys per attention block for lengths above LENS[-1]
    READOUTS = 64  # positions per readout call (padded; a forward with more readouts reads them in several calls),
    #               so the readout has one shape per bucket whatever the questions x option orders of its rows

    def __init__(self, model_dir, dtype="bfloat16", shared=True, warm=True, max_len=4096, platform="tpu"):
        import jax
        self.jax = jax
        if dtype not in DTYPES:
            raise ValueError(f"--mode tpu supports --dtype {' / '.join(DTYPES)}, got {dtype!r}")
        try:
            self.device = jax.devices(platform)[0]
        except RuntimeError as e:
            raise RuntimeError(f"--mode tpu: JAX sees no {platform} device (available: {jax.devices()}); run on a TPU "
                               "VM with jax[tpu] installed") from e
        c = model_config(model_dir)
        max_len = min(max_len, c["max_pos"] or max_len)  # never beyond the model's positions
        self.max_len = max_len
        self.context, self.context_reason = max_len, f"--max-len {max_len} of --mode tpu"
        self.long_lens = [x for x in self.LONG_LENS if self.LENS[-1] < x < max_len]
        if max_len > self.LENS[-1]:
            step = math.lcm(self.Q_BLOCK, self.KV_BLOCK)
            self.long_lens.append(-(-max_len // step) * step)
        cache = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(Path.home() / ".cache" / "basal" / "jax")
        jax.config.update("jax_compilation_cache_dir", cache)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.prefill = PREFILL
        self.shared = shared
        self.cfg, self.params = load_params(model_dir, dtype, self.device)
        full, qb, kvb = self.LENS[-1], self.Q_BLOCK, self.KV_BLOCK
        self._fwd = jax.jit(lambda p, ids, pos, end: forward(
            p, ids, pos, end, c=c, **(dict(q_block=qb, kv_block=kvb) if ids.shape[1] > full else {})))
        self._read = jax.jit(lambda p, h, rows, cols, li: readout(p, h, rows, cols, li, c=c))
        self.slots, self._lids = {}, None
        if warm:
            self._warm()

    def _warm(self):
        """Compile every bucket shape before serving: the forward and the readout per (batch, length), so no request
        waits for a compilation (the readout's shape does not depend on the number of readouts)."""
        for L in self.LENS + self.long_lens:
            for b in self.BATCHES:
                if b * L <= self.TOKEN_BUDGET or b == 1:
                    self._forward_rows([([0] * L, list(range(L)), [0] * L, [L - 1], [-1])] * b, b, L)

    def _bucket(self, n, xs):
        for x in xs:
            if n <= x:
                return x
        if xs is self.LENS:
            for x in self.long_lens:
                if n <= x:
                    return x
        raise ValueError(f"prompt of {n} tokens is longer than --max-len {self.max_len} (--mode tpu)")

    def _letter_slots(self, ids_list):
        """Column of each letter id in the logits; the device array of output-head rows changes only for new ids."""
        new = [i for ids in ids_list for i in ids if i not in self.slots]
        if new or self._lids is None:
            for i in new:
                self.slots.setdefault(i, len(self.slots))
            n = -(-max(len(self.slots), 1) // self.LETTER_SLOTS) * self.LETTER_SLOTS
            lids = np.zeros(n, np.int32)
            for i, s in self.slots.items():
                lids[s] = i
            self._lids = self.jax.device_put(lids, self.device)
        return self.slots

    def _forward_rows(self, packs, b, L):
        """Packed rows (GraphBackend._pack) padded to (b, L) -> fp32 logits (readouts of all rows in order,
        LETTER_SLOTS) on the host."""
        if self._lids is None:
            self._letter_slots([])
        ids = np.full((b, L), self.tok.pad_token_id, np.int32)
        pos = np.tile(np.arange(L, dtype=np.int32), (b, 1))
        end = np.tile(np.arange(1, L + 1, dtype=np.int32), (b, 1))  # padding sees itself only
        rows, cols = [], []
        for r, (t, pp, sg, last, parent) in enumerate(packs):
            ids[r, : len(t)], pos[r, : len(t)], end[r, : len(t)] = t, pp, subtree_ends(sg, parent)
            rows += [r] * len(last); cols += last
        n, R = len(rows), self.READOUTS
        ri, ci = np.zeros(-(-n // R) * R, np.int32), np.zeros(-(-n // R) * R, np.int32)
        ri[:n], ci[:n] = rows, cols
        h = self._fwd(self.params, ids, pos, end)
        out = [self._read(self.params, h, ri[i: i + R], ci[i: i + R], self._lids) for i in range(0, n, R)]
        return np.concatenate([np.asarray(x) for x in out])[:n]

    def _oversize(self, pack, n):
        """A unit that must be bisected: longer than max_len. The trie mask runs any unit as one row (several roots,
        any number of tokens below the root), so a request's state is computed once whenever it fits."""
        return len(pack[0]) > self.max_len

    def run(self, prompts, ids_list, policy=None):
        return [r[0] for r in self.run_shared([([p], [x]) for p, x in zip(prompts, ids_list)])]

    def run_shared(self, groups, policy=None, done=None):
        """GraphBackend.run_shared's contract: the option probabilities of every prompt of every group; done(k, res)
        (optional) is called for group k as soon as all its units are read out. Without `shared` every prompt is a
        unit of its own (GraphBackend._units)."""
        units = self._units(groups)
        longest = max(len(u[0][0]) for u in units)
        if longest > self.max_len:  # refuse before anything is compiled or run
            raise ValueError(f"prompt of {longest} tokens is longer than --max-len {self.max_len} (--mode tpu)")
        slots = self._letter_slots([x for _, ids_list, *_ in groups for x in ids_list])
        res = [[None] * len(g[0]) for g in groups]
        left = [0] * len(groups)
        for u in units:
            left[u[2]] += 1
        for idx in self._chunks([len(u[0][0]) for u in units]):
            L = self._bucket(max(len(units[k][0][0]) for k in idx), self.LENS)
            b = self._bucket(len(idx), self.BATCHES)
            lg = self._forward_rows([units[k][0] for k in idx], b, L).astype(np.float64)
            j = 0
            for k in idx:
                _, lids, g, a = units[k]
                res[g][a: a + len(lids)] = [_softmax(lg[j + o, [slots[i] for i in ids]]) for o, ids in enumerate(lids)]
                j += len(lids)
                left[g] -= 1
                if left[g] == 0 and done is not None:
                    done(g, res[g])
        return res

    def memory_gb(self):
        stats = self.device.memory_stats() or {}
        return stats.get("peak_bytes_in_use", float("nan")) / 2**30

    def device_name(self):
        return self.device.device_kind
