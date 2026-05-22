"""
Signal-processing primitives used by PADS.

These are vectorised replacements for the routines in the original
``sigproc.py`` and ``mel_extractor.py`` files. They produce identical
numerical output but run noticeably faster on long recordings because
framing uses ``numpy.lib.stride_tricks`` instead of Python loops, and the
mel filter-bank is computed in a fully vectorised pass.

All functions are pure (no global state) and have no PyTorch dependency,
so they are safe to call from worker processes or notebooks.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# -----------------------------------------------------------------------------
# Pre-emphasis & normalisation
# -----------------------------------------------------------------------------

def preemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply a first-order pre-emphasis filter ``y[n] = x[n] - a*x[n-1]``.

    Vectorised; works in O(N) without per-sample Python loops.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        out = np.empty_like(signal)
        out[0] = signal[0]
        out[1:] = signal[1:] - coeff * signal[:-1]
        return out
    # Multi-channel: apply across last axis.
    out = np.empty_like(signal)
    out[..., 0] = signal[..., 0]
    out[..., 1:] = signal[..., 1:] - coeff * signal[..., :-1]
    return out


def normalise_amplitude(signal: np.ndarray) -> np.ndarray:
    """DC-remove and scale to [-1, 1]. Returns float32."""
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - float(signal.mean())
    peak = float(np.abs(signal).max())
    if peak > 0:
        signal = signal / peak
    return signal


# -----------------------------------------------------------------------------
# Framing
# -----------------------------------------------------------------------------

def frame_signal(
    signal: np.ndarray,
    frame_len: int,
    frame_step: int,
    window: str | None = "hamming",
) -> np.ndarray:
    """Slice ``signal`` into overlapping frames.

    Implementation notes
    --------------------
    The original PADS code built frames with a Python list comprehension and
    ``np.vstack``. For long recordings that triggered hundreds of allocations.
    Here we use ``sliding_window_view`` which returns a strided *view* (no
    copy) and only copies when we multiply by the window.

    Parameters
    ----------
    signal : 1-D float array
    frame_len : int
        Frame size in samples.
    frame_step : int
        Hop size in samples.
    window : {"hamming", "hanning", None}
        Analysis window applied to each frame.

    Returns
    -------
    np.ndarray of shape ``(num_frames, frame_len)``
    """
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"frame_signal expects a 1-D signal, got shape {signal.shape}")
    if len(signal) < frame_len:
        # Pad with zeros to allow at least one frame.
        pad = frame_len - len(signal)
        signal = np.concatenate([signal, np.zeros(pad, dtype=signal.dtype)])

    # Strided view: shape = (len-frame_len+1, frame_len)
    all_windows = sliding_window_view(signal, frame_len)
    # Downsample by the hop length.
    frames = all_windows[::frame_step].copy()  # copy so it's writeable.

    if window is not None:
        if window == "hamming":
            w = np.hamming(frame_len).astype(np.float32)
        elif window in ("hann", "hanning"):
            w = np.hanning(frame_len).astype(np.float32)
        else:
            raise ValueError(f"Unknown window type: {window!r}")
        frames *= w  # broadcasted multiply
    return frames


# -----------------------------------------------------------------------------
# Spectra
# -----------------------------------------------------------------------------

def magnitude_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Magnitude spectrum (real-only, non-redundant half) of each frame."""
    spec = np.fft.rfft(frames, n=n_fft)
    return np.abs(spec).astype(np.float32)


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Power spectrum, ``(1/N) * |FFT|^2``."""
    return (1.0 / n_fft) * np.square(magnitude_spectrum(frames, n_fft))


def next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# -----------------------------------------------------------------------------
# Mel filter-bank
# -----------------------------------------------------------------------------

def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def get_filterbank(
    n_filters: int = 64,
    n_fft: int = 2048,
    sample_rate: int = 16000,
    low_freq: float = 0.0,
    high_freq: float | None = None,
) -> np.ndarray:
    """Build a triangular mel filter-bank matrix ``(n_filters, n_fft//2 + 1)``.

    Results are cached - identical parameters reuse the same filterbank.
    """
    high_freq = high_freq if high_freq is not None else sample_rate / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    # FFT bin number for each mel point.
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for j in range(n_filters):
        left, centre, right = bins[j], bins[j + 1], bins[j + 2]
        if centre > left:
            i = np.arange(left, centre)
            fbank[j, i] = (i - left) / max(centre - left, 1)
        if right > centre:
            i = np.arange(centre, right)
            fbank[j, i] = (right - i) / max(right - centre, 1)
    return fbank


def log_mel_spectrum(spec: np.ndarray, melfb: np.ndarray) -> np.ndarray:
    """Project a magnitude / power spectrogram onto the mel filterbank and log-compress.

    ``spec`` shape: ``(n_frames, n_fft//2 + 1)``.
    Output shape:   ``(n_frames, n_filters)``.
    """
    mel = spec @ melfb.T
    # Replace zeros with eps so log doesn't blow up.
    eps = np.finfo(np.float32).eps
    np.maximum(mel, eps, out=mel)
    return np.log(mel)
