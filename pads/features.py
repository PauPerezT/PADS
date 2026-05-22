"""
High-level feature extraction for PADS.

Produces three-channel mel-spectrogram clips compatible with the published
checkpoints. Vectorised throughout; several times faster than the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import signal_utils as su


DEFAULT_WIN_SIZES_S: Tuple[float, float, float] = (0.016, 0.025, 0.040)
DEFAULT_STEP_S: float = 0.010
DEFAULT_N_FFT: int = 2048
DEFAULT_N_MEL_FILTERS: int = 64
DEFAULT_FMAX: int = 8000
DEFAULT_CLIP_SECONDS: float = 1.0
DEFAULT_CLIP_HOP_SECONDS: float = 0.5


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 16000
    win_sizes_s: Tuple[float, float, float] = DEFAULT_WIN_SIZES_S
    step_s: float = DEFAULT_STEP_S
    n_fft: int = DEFAULT_N_FFT
    n_mel_filters: int = DEFAULT_N_MEL_FILTERS
    fmax: int = DEFAULT_FMAX
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    clip_hop_seconds: float = DEFAULT_CLIP_HOP_SECONDS


class Spectrogram2DTensor:
    """Compute multi-resolution 3-channel mel-spectrogram tensors."""

    def __init__(self, sig, sample_rate=16000, cfg=None):
        self.cfg = cfg or SpectrogramConfig(sample_rate=sample_rate)
        if sample_rate != self.cfg.sample_rate:
            self.cfg = SpectrogramConfig(
                sample_rate=sample_rate,
                win_sizes_s=self.cfg.win_sizes_s,
                step_s=self.cfg.step_s,
                n_fft=self.cfg.n_fft,
                n_mel_filters=self.cfg.n_mel_filters,
                fmax=self.cfg.fmax,
                clip_seconds=self.cfg.clip_seconds,
                clip_hop_seconds=self.cfg.clip_hop_seconds,
            )

        self.sig = np.ascontiguousarray(sig, dtype=np.float32)
        self._melfb = su.get_filterbank(
            n_filters=self.cfg.n_mel_filters,
            n_fft=self.cfg.n_fft,
            sample_rate=self.cfg.fmax,
        )

    def _mel_spec(self, win_size_s: float) -> np.ndarray:
        fs = self.cfg.sample_rate
        win_size = int(win_size_s * fs)
        step_size = int(self.cfg.step_s * fs)
        if step_size < 1:
            raise ValueError("step_s too small for the given sample rate")

        sig = su.normalise_amplitude(self.sig)
        frames = su.frame_signal(sig, win_size, step_size, window="hamming")
        mag = su.magnitude_spectrum(frames, n_fft=self.cfg.n_fft)
        return su.log_mel_spectrum(mag, self._melfb)

    def get_2d_spectrograms(self) -> np.ndarray:
        specs = [self._mel_spec(w) for w in self.cfg.win_sizes_s]
        n_frames = min(s.shape[0] for s in specs)
        return np.stack([s[:n_frames] for s in specs], axis=0).astype(np.float32)

    def get_2d_tensor_clips(self) -> np.ndarray:
        specs = self.get_2d_spectrograms()  # (3, T, F)
        n_frames = specs.shape[1]
        clip_frames = int(round(self.cfg.clip_seconds / self.cfg.step_s))
        clip_hop = int(round(self.cfg.clip_hop_seconds / self.cfg.step_s))

        if clip_frames <= 0 or clip_hop <= 0:
            raise ValueError("clip_seconds and clip_hop_seconds must be positive")
        if n_frames < clip_frames:
            pad = clip_frames - n_frames
            specs = np.pad(specs, ((0, 0), (0, pad), (0, 0)), mode="edge")
            n_frames = specs.shape[1]

        windows = sliding_window_view(specs, clip_frames, axis=1)
        windows = windows[:, ::clip_hop]
        clips = np.transpose(windows, (1, 0, 3, 2)).astype(np.float32, copy=True)
        return np.ascontiguousarray(clips)


def extract_pad_tensors(sig, sample_rate=16000, cfg=None):
    return Spectrogram2DTensor(sig, sample_rate, cfg).get_2d_tensor_clips()


def mel_filterbank(n_filters=64, n_fft=2048, sample_rate=16000):
    return su.get_filterbank(n_filters=n_filters, n_fft=n_fft, sample_rate=sample_rate)


def log_mel_spectrum(spec, melfb):
    return su.log_mel_spectrum(spec, melfb)
