"""
Signal-processing primitives used by PADS.

Vectorised replacements for the routines in the original sigproc.py and
mel_extractor.py. Identical numerical output, several times faster on long
recordings.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def preemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply a first-order pre-emphasis filter y[n] = x[n] - a*x[n-1]."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        out = np.empty_like(signal)
        out[0] = signal[0]
        out[1:] = signal[1:] - coeff * signal[:-1]
        return out
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


def frame_signal(signal, frame_len, frame_step, window="hamming"):
    """Slice signal into overlapping frames using sliding_window_view."""
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"frame_signal expects a 1-D signal, got {signal.shape}")
    if len(signal) < frame_len:
        pad = frame_len - len(signal)
        signal = np.concatenate([signal, np.zeros(pad, dtype=signal.dtype)])

    all_windows = sliding_window_view(signal, frame_len)
    frames = all_windows[::frame_step].copy()

    if window is not None:
        if window == "hamming":
            w = np.hamming(frame_len).astype(np.float32)
        elif window in ("hann", "hanning"):
            w = np.hanning(frame_len).astype(np.float32)
        else:
            raise ValueError(f"Unknown window type: {window!r}")
        frames *= w
    return frames


def magnitude_spectrum(frames, n_fft):
    spec = np.fft.rfft(frames, n=n_fft)
    return np.abs(spec).astype(np.float32)


def power_spectrum(frames, n_fft):
    return (1.0 / n_fft) * np.square(magnitude_spectrum(frames, n_fft))


def next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def get_filterbank(n_filters=64, n_fft=2048, sample_rate=16000, low_freq=0.0, high_freq=None):
    """Build a triangular mel filter-bank matrix (n_filters, n_fft//2 + 1)."""
    high_freq = high_freq if high_freq is not None else sample_rate / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
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


def log_mel_spectrum(spec, melfb):
    """Project a magnitude spectrogram onto a mel filterbank and log-compress."""
    mel = spec @ melfb.T
    eps = np.finfo(np.float32).eps
    np.maximum(mel, eps, out=mel)
    return np.log(mel)
