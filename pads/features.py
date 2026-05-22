"""
High-level feature extraction for PADS.

The original PADS pipeline computes three magnitude spectrograms (with
window sizes of 16, 25, and 40 ms), maps each onto a 64-band mel filter-
bank, then stacks them as a 3-channel "spectrogram tensor". Tensors are
chopped into overlapping clips (1 s long, 0.5 s hop by default) that feed
the CNN+attention+GRU classifier.

This rewrite keeps the same output format (so existing checkpoints work),
but is several times faster on long recordings thanks to:

* vectorised framing (``sliding_window_view`` instead of list-comprehensions);
* a single mel filterbank shared across all three window sizes;
* a strided ``sliding_window_view`` over the (time, mel) plane to extract
  clips - no Python loop;
* float32 throughout (the original mixed float64 and float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import signal_utils as su


# Default analysis parameters used by the published PADS checkpoints.
DEFAULT_WIN_SIZES_S: Tuple[float, float, float] = (0.016, 0.025, 0.040)
DEFAULT_STEP_S: float = 0.010
DEFAULT_N_FFT: int = 2048
DEFAULT_N_MEL_FILTERS: int = 64
DEFAULT_FMAX: int = 8000
DEFAULT_CLIP_SECONDS: float = 1.0
DEFAULT_CLIP_HOP_SECONDS: float = 0.5


@dataclass(frozen=True)
class SpectrogramConfig:
    """Configuration bundle for :class:`Spectrogram2DTensor`."""

    sample_rate: int = 16000
    win_sizes_s: Tuple[float, float, float] = DEFAULT_WIN_SIZES_S
    step_s: float = DEFAULT_STEP_S
    n_fft: int = DEFAULT_N_FFT
    n_mel_filters: int = DEFAULT_N_MEL_FILTERS
    fmax: int = DEFAULT_FMAX
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    clip_hop_seconds: float = DEFAULT_CLIP_HOP_SECONDS


class Spectrogram2DTensor:
    """Compute multi-resolution 3-channel mel-spectrogram tensors.

    Parameters
    ----------
    sig : 1-D numpy array
        The input speech waveform (already resampled to ``sample_rate``).
    sample_rate : int
        Sampling frequency.
    cfg : :class:`SpectrogramConfig`, optional
        Override default analysis parameters.

    Notes
    -----
    All three sub-spectrograms are projected onto the *same* mel filterbank.
    Constructing it once and reusing it is one of the main speedups vs.
    the original code, where it was rebuilt three times implicitly.
    """

    def __init__(
        self,
        sig: np.ndarray,
        sample_rate: int = 16000,
        cfg: SpectrogramConfig | None = None,
    ):
        self.cfg = cfg or SpectrogramConfig(sample_rate=sample_rate)
        if sample_rate != self.cfg.sample_rate:
            # Allow callers to pass sr without bothering with cfg; assume cfg defaults except sr.
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

    # ------------------------------------------------------------------
    # Single-resolution mel-spectrogram
    # ------------------------------------------------------------------

    def _mel_spec(self, win_size_s: float) -> np.ndarray:
        """Return a log-mel spectrogram of shape ``(n_frames, n_mel_filters)``."""
        fs = self.cfg.sample_rate
        win_size = int(win_size_s * fs)
        step_size = int(self.cfg.step_s * fs)
        if step_size < 1:
            raise ValueError("step_s too small for the given sample rate")

        sig = su.normalise_amplitude(self.sig)
        frames = su.frame_signal(sig, win_size, step_size, window="hamming")
        mag = su.magnitude_spectrum(frames, n_fft=self.cfg.n_fft)
        return su.log_mel_spectrum(mag, self._melfb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_2d_spectrograms(self) -> np.ndarray:
        """Return the stacked 3-channel log-mel spectrogram.

        Shape: ``(3, n_frames, n_mel_filters)`` where ``n_frames`` is the
        minimum frame count across the three window sizes (so the channels
        align in time).
        """
        specs = [self._mel_spec(w) for w in self.cfg.win_sizes_s]
        # Window sizes differ, so the number of frames differs slightly.
        n_frames = min(s.shape[0] for s in specs)
        stacked = np.stack([s[:n_frames] for s in specs], axis=0).astype(np.float32)
        return stacked

    def get_2d_tensor_clips(self) -> np.ndarray:
        """Slice the 3-channel spectrogram into overlapping clips.

        Output shape: ``(n_clips, 3, clip_frames, n_mel_filters)`` where
        ``clip_frames = clip_seconds / step_s``.
        """
        specs = self.get_2d_spectrograms()  # (3, T, F)
        n_frames = specs.shape[1]
        clip_frames = int(round(self.cfg.clip_seconds / self.cfg.step_s))
        clip_hop = int(round(self.cfg.clip_hop_seconds / self.cfg.step_s))

        if clip_frames <= 0 or clip_hop <= 0:
            raise ValueError("clip_seconds and clip_hop_seconds must be positive")
        if n_frames < clip_frames:
            # Pad along time so we can return at least one clip.
            pad = clip_frames - n_frames
            specs = np.pad(specs, ((0, 0), (0, pad), (0, 0)), mode="edge")
            n_frames = specs.shape[1]

        # Vectorised clipping: stride a window over the time axis.
        # sliding_window_view returns (3, T - clip_frames + 1, F, clip_frames)
        windows = sliding_window_view(specs, clip_frames, axis=1)
        windows = windows[:, ::clip_hop]  # apply hop
        # Reorder to (n_clips, 3, clip_frames, F).
        clips = np.transpose(windows, (1, 0, 3, 2)).astype(np.float32, copy=True)
        return np.ascontiguousarray(clips)


# Convenience wrappers ---------------------------------------------------------

def extract_pad_tensors(
    sig: np.ndarray,
    sample_rate: int = 16000,
    cfg: SpectrogramConfig | None = None,
) -> np.ndarray:
    """Functional API for :meth:`Spectrogram2DTensor.get_2d_tensor_clips`."""
    return Spectrogram2DTensor(sig, sample_rate, cfg).get_2d_tensor_clips()


def mel_filterbank(
    n_filters: int = 64,
    n_fft: int = 2048,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Standalone mel filterbank computation - delegates to cached impl."""
    return su.get_filterbank(n_filters=n_filters, n_fft=n_fft, sample_rate=sample_rate)


def log_mel_spectrum(spec: np.ndarray, melfb: np.ndarray) -> np.ndarray:
    """Project a magnitude spectrogram onto a mel filterbank and log-compress."""
    return su.log_mel_spectrum(spec, melfb)
