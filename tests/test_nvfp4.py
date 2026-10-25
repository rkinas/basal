"""NVFP4 (basal/nvfp4.py, --mode nvfp4 --nvfp4-amax on basal-omni): e2m1 values with e4m3 scales per 16 features and
a global scale per tensor, the scales in cuBLAS's 128 x 4 swizzle; on CPU the GEMM's maths dequantised."""
import torch

from basal import nvfp4


def test_codes_round_to_nearest_ties_to_even():
    u = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, -0.75, -5])
    assert nvfp4._values(nvfp4._codes(u)).tolist() == [0, 0, .5, 1, 1, 1, 1.5, 2, 2, 2, 3, 4, 4, 4, 6, 6, -1, -4]


def test_swizzle_round_trip():
    torch.manual_seed(0)
    s = torch.rand(300, 52).mul(400).to(nvfp4.E4M3)
    f = nvfp4.swizzle(s)
    assert f.numel() == 384 * 52
    assert torch.equal(nvfp4.unswizzle(f, 300, 52).view(torch.uint8), s.view(torch.uint8))


def test_weight_scales_by_least_squares_beat_amax_scales():
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    c, s, g = nvfp4.quantize(w)
    err = (nvfp4.dequantize(c, s, g, 256, 512) - w).norm() / w.norm()
    keep = nvfp4.CANDIDATES
    try:
        nvfp4.CANDIDATES = (1.0,)
        c1, s1, g1 = nvfp4.quantize(w)
    finally:
        nvfp4.CANDIDATES = keep
    err1 = (nvfp4.dequantize(c1, s1, g1, 256, 512) - w).norm() / w.norm()
    assert err < err1 < 0.12


def test_linear_and_parts():
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 256, bias=False)
    x = torch.randn(37, 512).to(torch.bfloat16)
    f = nvfp4.NVFP4Linear.of(lin, x.float().abs().amax())
    y = f(x).float()
    ref = x.float() @ lin.weight.float().t()
    assert (y - ref).norm() / ref.norm() < 0.2
    assert torch.equal(f.part(128, 128)(x), f(x)[:, 128:])
    # a row's output does not depend on the rows around it (the block scales are per token)
    assert torch.equal(f(x[:1]), f(x)[:1])
