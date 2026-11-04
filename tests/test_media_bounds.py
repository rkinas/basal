"""Media work is bounded and off the event loop: an audio clip is read up to the model's window (a long clip that
compresses well, such as silence in FLAC, is refused without decoding it whole), a clip of a model without a known
window up to MAX_AUDIO_SAMPLES; images and audio are decoded in a worker thread, and the engine's media calls run in a
thread of their own, so requests waiting for media never hold the threads text batches need."""
import asyncio
import base64
import io
import threading

import pytest
from test_media import MediaServer, png, wav

from basal import media


def flac_silence(seconds, sr=16000, channels=2):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    buf = io.BytesIO()
    sf.write(buf, np.zeros((int(seconds * sr), channels), dtype="float32"), sr, format="FLAC")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def frames_read(monkeypatch):
    sf = pytest.importorskip("soundfile")
    got, read = [], sf.SoundFile.read

    def spy(self, *a, **kw):
        out = read(self, *a, **kw)
        got.append(len(out))
        return out
    monkeypatch.setattr(sf.SoundFile, "read", spy)
    return got


def test_a_long_clip_is_refused_after_reading_at_most_the_window(frames_read):
    data = flac_silence(120)
    assert len(data) < 200_000                                       # 120 s of silence: a small request
    with pytest.raises(ValueError, match="120.0 s of audio, more than the model's 30 s"):
        media.decode([{"type": "audio", "data": data}], ("audio",), 30.0)
    assert max(frames_read) <= 30 * 16000 + 1


def test_a_clip_at_the_window_is_read_whole(frames_read):
    _, (x,), _ = media.decode([{"type": "audio", "data": flac_silence(30)}], ("audio",), 30.0)
    assert len(x) == 30 * media.SAMPLE_RATE


def test_without_a_window_a_clip_is_capped_in_samples(monkeypatch, frames_read):
    monkeypatch.setattr(media, "MAX_AUDIO_SAMPLES", 2 * 16000 * 2)    # 2 s of 16 kHz stereo
    with pytest.raises(ValueError, match="more than 64,000 samples"):
        media.decode([{"type": "audio", "data": flac_silence(3)}], ("audio",), None)
    assert max(frames_read) <= 16001 * 2
    _, (x,), _ = media.decode([{"type": "audio", "data": flac_silence(1.5)}], ("audio",), None)
    assert len(x) == 24000


def test_media_is_decoded_and_read_off_the_event_loop():
    srv = MediaServer([0.8, 0.2])
    threads = {}
    decode = srv.media_of

    def media_of(body, pol):
        threads["decode"] = threading.current_thread().name
        return decode(body, pol)
    srv.media_of = media_of
    run = srv.backend.run_media

    def run_media(*a, **kw):
        threads["engine"] = threading.current_thread().name
        return run(*a, **kw)
    srv.backend.run_media = run_media
    body = {"state": "", "lang": "pl", "media": [{"type": "image", "data": png()}, {"type": "audio", "data": wav()}],
            "questions": {"q": {"type": "noul", "instructions": "Czy?"}}}
    asyncio.run(srv._run(body))
    main = threading.main_thread().name
    assert threads["decode"] != main and threads["engine"].startswith("basal-media")
