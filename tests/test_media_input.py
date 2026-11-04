"""Regression tests for fix/media-input (Z-1082)."""
import base64
import io
import sys
import types

import numpy as np
import pytest

pytest.importorskip("PIL")  # pillow is in the [media] extra, not in [test]
from PIL import Image, ImageDraw

from basal import media


def fake_soundfile(monkeypatch, frames, sr, ch=2):
    reads = []
    sf = types.ModuleType("soundfile")
    sf.info = lambda f: types.SimpleNamespace(frames=frames, samplerate=sr, channels=ch)

    def read(f, dtype="float32", always_2d=True):
        reads.append(frames)
        return np.zeros((min(frames, 1000), ch), dtype="float32"), sr
    sf.read = read
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    return reads


@pytest.mark.parametrize("frames, sr", [(3 * 3600 * 48000, 48000), (31 * 16000, 16000)])
def test_i06_audio_over_the_window_is_refused_before_it_is_decoded(monkeypatch, frames, sr):
    reads = fake_soundfile(monkeypatch, frames, sr)
    with pytest.raises(ValueError):
        media._audio(b"x", 0, 30.0)
    assert reads == []


@pytest.mark.parametrize("frames, sr", [(10 * 48000, 48000), (30 * 16000, 16000)])
def test_i06_audio_inside_the_window_is_decoded_once(monkeypatch, frames, sr):
    if sr != media.SAMPLE_RATE:  # the 48000 Hz case is resampled; scipy is in the [media] extra, not in [test]
        pytest.importorskip("scipy.signal")
    reads = fake_soundfile(monkeypatch, frames, sr)
    media._audio(b"x", 0, 30.0)
    assert len(reads) == 1


def png(im, fmt="PNG", **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


def contrast(im):
    lo, hi = im.convert("L").getextrema()
    return hi - lo


def test_i07_black_on_a_transparent_png_stays_visible():
    a = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    ImageDraw.Draw(a).rectangle([8, 8, 40, 24], fill=(0, 0, 0, 255))
    assert contrast(media._image(png(a), 0)) > 100


def test_i07_black_on_a_transparent_grey_png_stays_visible():
    la = Image.new("LA", (64, 32), (0, 0))
    ImageDraw.Draw(la).rectangle([8, 8, 40, 24], fill=(0, 255))
    assert contrast(media._image(png(la), 0)) > 100


def test_i07_opaque_pixels_are_unchanged():
    assert media._image(png(Image.new("RGB", (8, 8), (10, 200, 30))), 0).getpixel((3, 3)) == (10, 200, 30)
    assert media._image(png(Image.new("RGBA", (8, 8), (0, 0, 0, 255))), 0).getpixel((3, 3)) == (0, 0, 0)


def jpeg(size, orientation=None):
    im = Image.new("RGB", size, (255, 255, 255))
    im.paste((255, 0, 0), (0, 0, size[0] // 4, size[1] // 2))  # red block at the top left of the STORED pixels
    b = io.BytesIO()
    exif = im.getexif()
    if orientation:
        exif[0x0112] = orientation
    im.save(b, "JPEG", quality=95, exif=exif.tobytes() if orientation else b"")
    return b.getvalue()


def red(px):
    return px[0] > 200 and px[1] < 80


def test_i16_exif_rotation_is_applied():
    out = media._image(jpeg((40, 20), 6), 0)
    assert out.size == (20, 40) and red(out.getpixel((out.size[0] - 2, 2)))


def test_i16_exif_upside_down_is_applied():
    out = media._image(jpeg((40, 20), 3), 0)
    assert red(out.getpixel((out.size[0] - 2, out.size[1] - 2)))


@pytest.mark.parametrize("orientation", [None, 1])
def test_i16_no_exif_or_normal_orientation_is_unchanged(orientation):
    out = media._image(jpeg((40, 20), orientation), 0)
    assert out.size == (40, 20) and red(out.getpixel((2, 2)))


def frames(fmt, n):
    ims = [Image.new("RGB", (16, 16), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)][:n]]
    b = io.BytesIO()
    ims[0].save(b, fmt, save_all=True, append_images=ims[1:], duration=100, loop=0)
    return b.getvalue()


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "PNG", "WEBP"])
def test_i17_animated_or_multi_page_image_is_refused(fmt):
    with pytest.raises(ValueError, match="frame"):
        media._image(frames(fmt, 3), 0)


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "PNG"])
def test_i17_single_frame_image_is_accepted(fmt):
    assert media._image(frames(fmt, 1), 0).size == (16, 16)


def test_i17_a_phone_photo_stored_as_mpo_is_accepted_as_its_first_frame():
    """MPO (Ultra HDR gain map, depth map, burst) keeps several frames in one JPEG file; frame 0 is the photo."""
    a, b = Image.new("RGB", (16, 16), (200, 0, 0)), Image.new("RGB", (8, 8), (0, 0, 200))
    buf = io.BytesIO()
    a.save(buf, "MPO", save_all=True, append_images=[b])
    assert Image.open(io.BytesIO(buf.getvalue())).n_frames == 2  # the file really has two frames
    im = media._image(buf.getvalue(), 0)
    assert im.size == (16, 16) and im.getpixel((0, 0))[0] > 150


def b64_png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (30, 60, 90)).save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


@pytest.mark.parametrize("w, h", [(6000, 300), (300, 6000)])
def test_i18_a_decoded_image_is_at_most_4096_a_side(w, h):
    (im,), _, _ = media.decode([{"type": "image", "data": b64_png(w, h)}], ("image",))
    assert max(im.size) == 4096 and abs(im.size[0] / im.size[1] / (w / h) - 1) < 0.05


def test_i18_a_transparent_image_is_shrunk_before_it_is_composited(monkeypatch):
    """The peak is the large copies: alpha_composite must see the 4096 px image, not the 6000 px one."""
    sizes = []
    real = Image.alpha_composite
    monkeypatch.setattr(Image, "alpha_composite", lambda a, b: sizes.append(a.size) or real(a, b))
    buf = io.BytesIO()
    Image.new("RGBA", (6000, 300), (255, 0, 0, 128)).save(buf, "PNG")
    im = media._image(buf.getvalue(), 0)
    assert sizes and max(max(s) for s in sizes) <= 4096 and im.size[0] == 4096 and im.mode == "RGB"


def test_i18_a_palette_image_with_transparency_still_lands_on_white():
    p = Image.new("P", (8, 8), 0)
    p.putpalette([0, 0, 0] + [0] * 765)
    buf = io.BytesIO()
    p.save(buf, "PNG", transparency=0)
    assert media._image(buf.getvalue(), 0).getpixel((0, 0)) == (255, 255, 255)


def test_i18_a_small_image_keeps_its_size():
    (im,), _, _ = media.decode([{"type": "image", "data": b64_png(640, 480)}], ("image",))
    assert im.size == (640, 480)


def test_s19_decoding_does_not_overwrite_the_global_pil_pixel_limit(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 123456789)
    media._image(png(Image.new("RGB", (8, 8))), 0)
    assert Image.MAX_IMAGE_PIXELS == 123456789


RAW = bytes([251, 255, 254, 250, 253, 252] * 40)  # its base64 has "+" and "/" (urlsafe: "-" and "_")


@pytest.mark.parametrize("wrap", [
    lambda s: s,
    lambda s: "\r\n".join(s[i:i + 76] for i in range(0, len(s), 76)),  # `base64` of coreutils, MIME
    lambda s: s.replace("+", "-").replace("/", "_"),
    lambda s: "DATA:image/png;base64," + s,
], ids=["plain", "wrapped", "urlsafe", "upper-case-scheme"])
def test_s20_base64_as_the_tools_write_it_is_accepted(wrap):
    import base64
    assert media._raw(wrap(base64.b64encode(RAW).decode()), 0) == RAW


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_s20_urlsafe_base64_without_its_padding_is_accepted(n):
    raw = bytes([251, 255, 254, 250, 253, 252] * 40)[: 240 + n]
    s = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert media._raw(s, 0) == raw and media._raw("data:image/png;base64," + s, 0) == raw
    assert media._raw(base64.urlsafe_b64encode(raw).decode(), 0) == raw  # with the padding, as before


def test_s20_a_broken_padding_is_still_refused():
    with pytest.raises(ValueError, match="not valid base64"):
        media._raw("QUJD=", 0)
    with pytest.raises(ValueError, match="not valid base64"):
        media._raw("QUJDR", 0)  # 5 characters can never be base64
