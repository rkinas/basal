"""The branch pass as a cascade on CUDA (basal.engine._varlen, _varlen_tree, _cascade_varlen): torch's varlen attention
against the attention written out (its log-sum-exp layout, causal aligned at the end, per-sequence key counts), and the
cascade against the dense masked attention of today's pass 2. Need CUDA sm80+ and torch's varlen attention."""
import importlib.util

import numpy as np
import pytest
import torch

from basal.engine import _cascade_varlen, _varlen, _varlen_tree, torch_varlen_available
from tests.test_engine16 import ref_varlen

why = torch_varlen_available(torch.bfloat16)
if why is None:  # the cascade also needs seqused_k (_cascade_why), which torch_varlen_available does not ask for
    import inspect

    from torch.nn.attention.varlen import varlen_attn
    if "seqused_k" not in inspect.signature(varlen_attn).parameters:
        why = "this torch's varlen_attn has no seqused_k"
cuda = pytest.mark.skipif(why is not None, reason=f"torch varlen attention: {why}")
H, HKV, D = 16, 2, 128  # basal-2.0-4.5B: 8 query heads per key / value head, head size 128
triton = pytest.mark.skipif(importlib.util.find_spec("triton") is None,
                            reason="the fused merge is torch.compile's (Triton); without it GraphBackend does not fuse")


def to_dev(x):
    return torch.from_numpy(np.asarray(x)).cuda()


@cuda
def test_varlen_matches_the_attention_written_out():
    g = torch.Generator(device="cuda").manual_seed(0)
    nq, nk = [3, 40, 1, 17], [9, 40, 30, 64]
    q = torch.randn(sum(nq), H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k, v = (torch.randn(sum(nk), HKV, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in "kv")
    cu_q, cu_k = (to_dev(np.r_[0, np.cumsum(x)].astype(np.int32)) for x in (nq, nk))
    for causal, used in ((True, None), (False, to_dev(np.array([9, 25, 1, 64], dtype=np.int32)))):
        o, lse = _varlen(q, k, v, cu_q, cu_k, max(nq), max(nk), D ** -0.5, causal=causal, used=used)
        ro, rl = ref_varlen(q.cpu(), k.cpu(), v.cpu(), cu_q.cpu(), cu_k.cpu(), max(nq), max(nk), D ** -0.5, causal,
                            None if used is None else used.cpu())
        assert lse.shape == (sum(nq), H) and lse.dtype == torch.float32
        assert torch.allclose(lse.cpu(), rl, atol=2e-2) and torch.allclose(o.float().cpu(), ro.float(), atol=2e-2)


@cuda
def test_cascade_equals_dense_masked_attention():
    """Two rows (different root lengths, the second padded), nested blocks: the cascade against one masked SDPA call
    over cat(root, branch) with repeated heads, as today's pass 2 runs it; bf16 within the dense call's own error."""
    g = torch.Generator(device="cuda").manual_seed(1)
    S, Lk, P = 96, 160, [150, 70]
    seg = np.full((2, S), -1)
    seg[0, :96] = [0] * 30 + [1] * 10 + [2] * 10 + [3] * 30 + [4] * 16
    seg[1, :61] = [0] * 40 + [1] * 21
    parents = [[-1, 0, 0, -1, 3], [-1, 0]]
    tree = _varlen_tree(seg, parents, P, Lk, to_dev)
    q = torch.randn(2, H, S, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k, v = (torch.randn(2, HKV, S, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in "kv")
    kp, vp = (torch.randn(2, HKV, Lk, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in "kv")
    got = _cascade_varlen(q, k, v, kp, vp, tree, D ** -0.5).float()
    allow = torch.zeros(2, S, Lk + S, dtype=torch.bool)
    for r in range(2):
        allow[r, :, : P[r]] = True
        for i in range(S):
            if seg[r, i] < 0:
                continue
            chain, b = set(), seg[r, i]
            while b >= 0:
                chain.add(b)
                b = parents[r][b]
            for j in range(i + 1):
                allow[r, i, Lk + j] = seg[r, j] in chain
    mask = torch.zeros(allow.shape).masked_fill(~allow, float("-inf"))[:, None].cuda()
    rep = H // HKV
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), torch.cat([kp, k], 2).float().repeat_interleave(rep, 1),
        torch.cat([vp, v], 2).float().repeat_interleave(rep, 1), attn_mask=mask).transpose(1, 2)
    real = torch.from_numpy(seg >= 0).cuda()
    assert torch.allclose(got[real], ref[real], atol=1e-2)


@cuda
@triton
def test_fused_merge_gives_the_merge_bit_for_bit():
    """The merge compiled into one kernel (_merge_fused: a GraphBackend that compiles) against _merge_attention op by
    op: the cascade's output bit for bit, for rows without padding (both log-sum-exps as the kernel returns them) and
    rows with padding (the branch's scattered into -inf), the two layouts a cascade passes."""
    g = torch.Generator(device="cuda").manual_seed(2)
    S, Lk = 96, 160
    full = np.array([[0] * 40 + [1] * 26 + [2] * 30, [0] * 50 + [1] * 46])
    padded = np.full((2, S), -1)
    padded[0, :96] = [0] * 30 + [1] * 10 + [2] * 10 + [3] * 30 + [4] * 16
    padded[1, :61] = [0] * 40 + [1] * 21
    for seg, parents in ((full, [[-1, 0, -1], [-1, 0]]), (padded, [[-1, 0, 0, -1, 3], [-1, 0]])):
        tree = _varlen_tree(seg, parents, [150, 70], Lk, to_dev)
        q = torch.randn(2, H, S, D, device="cuda", dtype=torch.bfloat16, generator=g)
        k, v = (torch.randn(2, HKV, S, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in "kv")
        kp, vp = (torch.randn(2, HKV, Lk, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in "kv")
        op_by_op = _cascade_varlen(q, k, v, kp, vp, tree, D ** -0.5)
        fused = _cascade_varlen(q, k, v, kp, vp, tree._replace(fused=True), D ** -0.5)
        assert tree.full == (seg is full) and torch.equal(fused, op_by_op)
