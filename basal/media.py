"""Images and audio in a request (basal-omni on Gemma 4; `basal-serve --mode eager --media on`).

A request carries its media with the state:

  "media": [{"type": "image" | "audio", "data": "<base64>"}, ...], "detail": "default" | "high" (optional)

The items go where the state text goes, before it, in the order given: the prompt the basal-omni evaluation reads them
with. Images: any format Pillow opens, converted to RGB; "detail" sets the image's soft tokens
(default 280, high 1,120). Audio: WAV, FLAC, OGG or MP3 (libsndfile), mixed to mono and resampled to 16 kHz; at most the
model's audio window (Gemma 4: 750 tokens x 40 ms = 30 s). One request holds at most MAX_ITEMS items of at most
MAX_BYTES each; an image may have at most MAX_PIXELS pixels (decompression bombs are refused before decoding).
Errors are ValueError (HTTP 422).
"""
import base64
import binascii
import io
from math import gcd

MAX_ITEMS = 8
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000
SAMPLE_RATE = 16000
DETAIL = {"default": 280, "high": 1120}


def kinds_of(config):
    """Media a model's config.json can take: ("image", "audio") for an image and an audio encoder."""
    return tuple(k for k, key in (("image", "vision_config"), ("audio", "audio_config")) if config.get(key))


def audio_window(processor_config):
    """Longest audio clip in seconds (Gemma 4: audio_seq_length tokens of audio_ms_per_token ms), None if unknown."""
    n, ms = processor_config.get("audio_seq_length"), processor_config.get("audio_ms_per_token")
    return n * ms / 1000 if n and ms else None


def detail_tokens(detail):
    if detail not in DETAIL:
        raise ValueError(f'detail must be one of {sorted(DETAIL)}, got {detail!r}')
    return DETAIL[detail]


def _raw(data, n):
    if not isinstance(data, str) or not data:
        raise ValueError(f"media {n}: data must be a base64 string")
    if data.startswith("data:") and "," in data:  # a data URL: data:<type>;base64,<payload>
        data = data.split(",", 1)[1]
    if len(data) * 3 // 4 > MAX_BYTES:
        raise ValueError(f"media {n}: more than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"media {n}: data is not valid base64") from None


def _image(raw, n):
    from PIL import Image, ImageOps, UnidentifiedImageError
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise ValueError(f"media {n}: {w}x{h} pixels, more than {MAX_PIXELS:,}")
        im.load()
        im = ImageOps.exif_transpose(im)  # a phone photo is stored sideways with an Orientation tag
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:  # onto white, not onto black
            rgba = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        return im.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
        raise ValueError(f"media {n}: not a readable image ({type(e).__name__})") from None


def _audio(raw, n, max_seconds):
    import numpy as np
    import soundfile as sf
    try:
        if max_seconds:  # the header first: a few kB of FLAC can decode to hours of float32
            info = sf.info(io.BytesIO(raw))
            if info.samplerate and info.frames / info.samplerate > max_seconds + 1e-6:
                raise ValueError(f"media {n}: {info.frames / info.samplerate:.1f} s of audio, more than the model's "
                                 f"{max_seconds:g} s")
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except (RuntimeError, TypeError) as e:  # soundfile's LibsndfileError is a RuntimeError
        raise ValueError(f"media {n}: not readable audio (WAV, FLAC, OGG or MP3; {type(e).__name__})") from None
    if x.shape[0] == 0:
        raise ValueError(f"media {n}: empty audio")
    if max_seconds and x.shape[0] / sr > max_seconds + 1e-6:
        raise ValueError(f"media {n}: {x.shape[0] / sr:.1f} s of audio, more than the model's {max_seconds:g} s")
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def decode(entries, kinds, max_audio_seconds=None):
    """The request's "media" -> (images, audio arrays, the type of every item in order). kinds: what the model takes."""
    if not isinstance(entries, list) or not entries:
        raise ValueError('media must be a non-empty list of {"type": "image" | "audio", "data": "<base64>"}')
    if len(entries) > MAX_ITEMS:
        raise ValueError(f"{len(entries)} media items, at most {MAX_ITEMS} per request")
    images, audio, order = [], [], []
    for n, m in enumerate(entries):
        t = m.get("type") if isinstance(m, dict) else None
        if t not in ("image", "audio"):
            raise ValueError(f'media {n}: type must be "image" or "audio"')
        if t not in kinds:
            raise ValueError(f"media {n}: this model does not take {t} (it takes: {', '.join(kinds) or 'no media'})")
        raw = _raw(m.get("data"), n)
        if t == "image":
            images.append(_image(raw, n))
        else:
            audio.append(_audio(raw, n, max_audio_seconds))
        order.append(t)
    return images, audio, order
