"""
Smoke tests for the PADS library.

These tests use a synthetic waveform so they can run without trained
checkpoints. The goal is to validate that the feature-extraction pipeline
produces tensors of the expected shape and that the model class can be
instantiated and run forward.
"""
from __future__ import annotations

import numpy as np
import pytest


SR = 16000


def _synthetic_audio(seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 440 * t)
    ).astype(np.float32)
    # Add a small amount of noise so the spectrum is non-trivial.
    wav += 0.01 * np.random.RandomState(0).randn(len(wav)).astype(np.float32)
    return wav


# ---------- signal utilities ---------------------------------------------------

def test_preemphasis_shape_and_first_sample():
    from pads.signal_utils import preemphasis
    x = np.arange(10, dtype=np.float32)
    y = preemphasis(x, coeff=0.5)
    assert y.shape == x.shape
    assert y[0] == x[0]
    assert np.allclose(y[1:], x[1:] - 0.5 * x[:-1])


def test_frame_signal_dimensions():
    from pads.signal_utils import frame_signal
    sig = np.arange(1000, dtype=np.float32)
    frames = frame_signal(sig, frame_len=128, frame_step=64, window=None)
    # First frame should match the first 128 samples.
    assert frames.shape[1] == 128
    np.testing.assert_array_equal(frames[0], sig[:128])
    # Second frame starts at sample 64.
    np.testing.assert_array_equal(frames[1], sig[64:64 + 128])


def test_get_filterbank_shape_and_caching():
    from pads.signal_utils import get_filterbank
    fb1 = get_filterbank(n_filters=64, n_fft=2048, sample_rate=16000)
    fb2 = get_filterbank(n_filters=64, n_fft=2048, sample_rate=16000)
    assert fb1.shape == (64, 1025)
    assert fb1 is fb2, "lru_cache should return the same array"


# ---------- 3-channel tensor builder ------------------------------------------

def test_spectrogram2d_output_shapes():
    from pads.features import Spectrogram2DTensor
    wav = _synthetic_audio(seconds=2.5)
    builder = Spectrogram2DTensor(wav, sample_rate=SR)
    spec = builder.get_2d_spectrograms()
    assert spec.ndim == 3
    assert spec.shape[0] == 3      # 3 channels
    assert spec.shape[2] == 64     # mel filterbank size

    clips = builder.get_2d_tensor_clips()
    assert clips.ndim == 4
    assert clips.shape[1] == 3
    # default clip_seconds=1s, step=10ms -> 100 frames
    assert clips.shape[2] == 100
    assert clips.shape[3] == 64
    # 2.5s audio with 1s clips, 0.5s hop -> ~4 clips
    assert clips.shape[0] >= 2


def test_pipeline_clip_count_long_audio():
    from pads.features import Spectrogram2DTensor
    wav = _synthetic_audio(seconds=10.0)
    clips = Spectrogram2DTensor(wav, sample_rate=SR).get_2d_tensor_clips()
    # 10s audio, 1s clips, 0.5s hop -> ~19 clips
    assert 15 <= clips.shape[0] <= 21


# ---------- normalisation ------------------------------------------------------

def test_normalize_batch_per_channel_runs():
    from pads.normalize import normalize_batch_per_channel
    clips = np.random.RandomState(1).randn(4, 3, 100, 64).astype(np.float32)
    out = normalize_batch_per_channel(clips)
    assert out.shape == clips.shape
    assert np.isfinite(out).all()


def test_normalize_minmax_round_trip():
    from pads.normalize import normalize_tensor_minmax
    norms = [{"min": "0.0", "max": "1.0"} for _ in range(3)]
    x = np.random.rand(3, 100, 64).astype(np.float32)
    y = normalize_tensor_minmax(x, norms)
    np.testing.assert_allclose(x, y)  # min=0,max=1 -> identity


# ---------- model (no checkpoint required) ------------------------------------

def test_selfconv_forward_pass():
    import torch
    from pads.models import SelfConv
    model = SelfConv(nc=3, input_shape=(1, 3, 100, 64))
    model.eval()
    x = torch.randn(2, 3, 100, 64)
    with torch.inference_mode():
        scores, emb = model(x)
    assert scores.shape == (2, 2)
    assert emb.shape == (2, 256)


# ---------- emotions ----------------------------------------------------------

def test_pad_to_emotion_returns_quadrant():
    from pads.emotions import pad_to_emotion, EMOTION_QUADRANTS
    q = pad_to_emotion(0.9, 0.9, 0.9)  # exuberant
    assert q.name == "Exuberant"
    q = pad_to_emotion(0.1, 0.1, 0.1)  # bored
    assert q.name == "Bored"
    assert len(EMOTION_QUADRANTS) == 8


def test_emotion_distribution_sums_to_one():
    from pads.emotions import emotion_distribution
    v = np.array([0.8, 0.2, 0.9, 0.3])
    a = np.array([0.7, 0.7, 0.3, 0.3])
    d = np.array([0.6, 0.4, 0.8, 0.2])
    dist = emotion_distribution(v, a, d)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_pad_result_supports_partial_dimensions():
    from pads.inference import PADResult
    result = PADResult(
        arousal=np.empty(0, dtype=np.float32),
        valence=np.array([0.2, 0.8], dtype=np.float32),
        dominance=np.empty(0, dtype=np.float32),
    )
    assert result.n_clips == 2
    assert result.available_dimensions() == ["valence"]
    assert not result.has_all_dimensions()
    df = result.to_dataframe()
    assert list(df.columns) == ["time_s", "valence"]
    with pytest.raises(ValueError):
        result.dominant_emotion()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
