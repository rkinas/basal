"""Regression tests for fix/media-input (Z-1082)."""
import sys
import types

import numpy as np
import pytest

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
