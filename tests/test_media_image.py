"""How an image of a request is decoded (basal/media.py _image): transparency, the EXIF orientation, animated and
multi-page files, and the largest side kept."""
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
