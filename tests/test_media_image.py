"""How an image of a request is decoded (basal/media.py _image): transparency, the EXIF orientation, animated and
multi-page files, and the largest side kept."""
import base64
import io

import pytest

pytest.importorskip("PIL")  # pillow is in the [media] extra, not in [test]
from PIL import Image, ImageDraw

from basal import media


def png(im, fmt="PNG", **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


def contrast(im):
    lo, hi = im.convert("L").getextrema()
    return hi - lo


def test_black_on_a_transparent_png_stays_visible():
    a = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    ImageDraw.Draw(a).rectangle([8, 8, 40, 24], fill=(0, 0, 0, 255))
    assert contrast(media._image(png(a), 0)) > 100


def test_black_on_a_transparent_grey_png_stays_visible():
    la = Image.new("LA", (64, 32), (0, 0))
    ImageDraw.Draw(la).rectangle([8, 8, 40, 24], fill=(0, 255))
    assert contrast(media._image(png(la), 0)) > 100


def test_opaque_pixels_are_unchanged():
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


def test_exif_rotation_is_applied():
    out = media._image(jpeg((40, 20), 6), 0)
    assert out.size == (20, 40) and red(out.getpixel((out.size[0] - 2, 2)))


def test_exif_upside_down_is_applied():
    out = media._image(jpeg((40, 20), 3), 0)
    assert red(out.getpixel((out.size[0] - 2, out.size[1] - 2)))


@pytest.mark.parametrize("orientation", [None, 1])
def test_no_exif_or_normal_orientation_is_unchanged(orientation):
    out = media._image(jpeg((40, 20), orientation), 0)
    assert out.size == (40, 20) and red(out.getpixel((2, 2)))


def frames(fmt, n):
    ims = [Image.new("RGB", (16, 16), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)][:n]]
    b = io.BytesIO()
    ims[0].save(b, fmt, save_all=True, append_images=ims[1:], duration=100, loop=0)
    return b.getvalue()


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "PNG", "WEBP"])
def test_an_animated_or_multi_page_image_is_refused(fmt):
    with pytest.raises(ValueError, match="frame"):
        media._image(frames(fmt, 3), 0)


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "PNG"])
def test_a_single_frame_image_is_accepted(fmt):
    assert media._image(frames(fmt, 1), 0).size == (16, 16)


def test_a_phone_photo_stored_as_mpo_is_accepted_as_its_first_frame():
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
def test_a_decoded_image_is_at_most_4096_a_side(w, h):
    (im,), _, _ = media.decode([{"type": "image", "data": b64_png(w, h)}], ("image",))
    assert max(im.size) == 4096 and abs(im.size[0] / im.size[1] / (w / h) - 1) < 0.05


def test_a_transparent_image_is_shrunk_before_it_is_composited(monkeypatch):
    """The peak is the large copies: alpha_composite must see the 4096 px image, not the 6000 px one."""
    sizes = []
    real = Image.alpha_composite
    monkeypatch.setattr(Image, "alpha_composite", lambda a, b: sizes.append(a.size) or real(a, b))
    buf = io.BytesIO()
    Image.new("RGBA", (6000, 300), (255, 0, 0, 128)).save(buf, "PNG")
    im = media._image(buf.getvalue(), 0)
    assert sizes and max(max(s) for s in sizes) <= 4096 and im.size[0] == 4096 and im.mode == "RGB"


def test_a_palette_image_with_transparency_still_lands_on_white():
    p = Image.new("P", (8, 8), 0)
    p.putpalette([0, 0, 0] + [0] * 765)
    buf = io.BytesIO()
    p.save(buf, "PNG", transparency=0)
    assert media._image(buf.getvalue(), 0).getpixel((0, 0)) == (255, 255, 255)


def test_a_small_image_keeps_its_size():
    (im,), _, _ = media.decode([{"type": "image", "data": b64_png(640, 480)}], ("image",))
    assert im.size == (640, 480)
