"""The readout of the option codes: one implementation for every torch engine (eager, the CUDA graph engines, Strix,
fast-exit) and for basal-finetune, so what a model is trained on is what it is served with.

A question's answer is the softmax over its option codes of the logits at the answer position. The softmax over the
code rows of the output head equals the softmax over the codes of the full log-softmax, so only those rows are read:
in fp32 from the normalised hidden state (bf16 logits of magnitude 16-32 step by 0.125, which clumps the
probabilities), with the head's bias and the model's final logit soft cap applied once (Gemma). The fp32 rows of each
set of codes are cached, the most recently used within WROWS_BYTES (MLX keeps its own arrays in the same cache).
"""
import torch

WROWS_BYTES = 64 << 20  # the fp32 code rows an engine keeps (one entry per option-code set, least recently used out)


def _nbytes(x):
    if x is None:
        return 0
    if isinstance(x, tuple):
        return sum(_nbytes(y) for y in x)
    return int(x.nbytes)  # (torch and MLX arrays)


def code_rows(cache, key, make):
    """cache[key], made by make() when missing: the output-head rows of one set of option codes. The cache keeps the most
    recently used sets within WROWS_BYTES (every option count and code set was kept for good: up to ~0.7 GB); an entry is
    the same rows whenever it is made again."""
    w = cache.pop(key, None)
    if w is None:
        w = make()
    cache[key] = w  # (most recently used last)
    total = sum(_nbytes(x) for x in cache.values())
    for old in list(cache):
        if total <= WROWS_BYTES or old == key:
            break
        total -= _nbytes(cache.pop(old))
    return w


def capped(z, softcap):
    """The model's final logit soft cap (Gemma: softcap x tanh(z / softcap)), applied once; none: z as it is."""
    return torch.tanh(z / softcap) * softcap if softcap else z


class CodeReadout:
    """The code readout of one output head (an nn.Linear-like module: weight [vocab, D], optional bias) and soft cap.

    rows(lids) -> (rows, idx, pad): the fp32 head rows (and bias) of the union of the prompts' code ids, each prompt's
    columns into them and the padding of the shorter ones; logits(h, rows) -> fp32 code logits [n, k] of hidden states
    h [n, D], capped once; probs(h, lids) -> each prompt's probabilities over its codes (one masked softmax, one copy to
    the host)."""

    def __init__(self, head, softcap=None, cache=None, to_dev=None):
        self.head, self.softcap = head, softcap
        self.cache = {} if cache is None else cache
        self.to_dev = to_dev or (lambda x: x.to(head.weight.device))

    def rows(self, lids):
        key = tuple(sorted({i for ids in lids for i in ids}))

        def make():
            sel = self.to_dev(torch.tensor(key))
            b = self.head.bias.index_select(0, sel).float() if getattr(self.head, "bias", None) is not None else None
            return self.head.weight.index_select(0, sel).float(), b
        rows = code_rows(self.cache, key, make)
        col = {t: j for j, t in enumerate(key)}
        K = max(map(len, lids))
        idx = self.to_dev(torch.tensor([[col[i] for i in ids] + [0] * (K - len(ids)) for ids in lids]))
        pad = self.to_dev(torch.tensor([[False] * len(ids) + [True] * (K - len(ids)) for ids in lids]))
        return rows, idx, pad

    def logits(self, h, rows):
        W, b = rows
        z = h.float() @ W.T
        return capped(z if b is None else z + b, self.softcap)

    @staticmethod
    def probs_of(z, idx, pad):
        """Code logits [n, k] -> each prompt's probabilities over its codes [n, K] (padding 0), as a tensor."""
        return torch.softmax(z.float().gather(1, idx).masked_fill(pad, float("-inf")), -1)

    def probs(self, h, lids):
        rows, idx, pad = self.rows(lids)
        p = self.probs_of(self.logits(h, rows), idx, pad).tolist()  # one device sync
        return [row[: len(ids)] for row, ids in zip(p, lids)]
