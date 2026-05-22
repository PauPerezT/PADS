"""
Normalisation utilities for PADS tensors.

Two strategies supported: per-spectrum min/max (Lightning checkpoints)
and pre-fit per-channel min/max from data_norm/*.json (legacy checkpoints).
"""

from __future__ import annotations

import json
import os

import numpy as np


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Per-spectrum min/max normalisation (Lightning-compatible)."""
    spectrum = np.asarray(spectrum, dtype=np.float32)
    mn = spectrum.min()
    mx = spectrum.max()
    denom = mx - mn
    if denom == 0:
        return spectrum.copy()
    return spectrum - (mn / denom)


def normalize_clip_per_channel(clip: np.ndarray) -> np.ndarray:
    """Apply normalize_spectrum to each of the 3 channels (shape: 3,T,F)."""
    out = np.empty_like(clip, dtype=np.float32)
    for c in range(clip.shape[0]):
        out[c] = normalize_spectrum(clip[c])
    return out


def normalize_batch_per_channel(clips: np.ndarray) -> np.ndarray:
    """Vectorised per-clip per-channel normalisation (N,3,T,F)."""
    clips = np.asarray(clips, dtype=np.float32)
    if clips.ndim != 4:
        raise ValueError(f"Expected 4-D batch (N,C,T,F), got {clips.shape}")
    mn = clips.min(axis=(2, 3), keepdims=True)
    mx = clips.max(axis=(2, 3), keepdims=True)
    denom = mx - mn
    safe = np.where(denom == 0, 1.0, denom)
    return clips - (mn / safe)


def normalize_tensor_minmax(tensor, norm_params):
    """Apply pre-fit min/max normalisation per channel (legacy checkpoints)."""
    if len(norm_params) != 3:
        raise ValueError("Expected exactly 3 sets of normalisation parameters")

    mins = np.array([float(p["min"]) for p in norm_params], dtype=np.float32)
    maxs = np.array([float(p["max"]) for p in norm_params], dtype=np.float32)
    rng = maxs - mins
    rng = np.where(rng == 0, 1.0, rng)

    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim == 3:
        c_mins = mins[:, None, None]
        c_rng = rng[:, None, None]
    elif tensor.ndim == 4:
        c_mins = mins[None, :, None, None]
        c_rng = rng[None, :, None, None]
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}")

    return (tensor - c_mins) / c_rng


def load_norm_params(data_norm_dir):
    params = []
    for i in (1, 2, 3):
        path = os.path.join(data_norm_dir, f"param_c{i}.json")
        with open(path, "r") as fp:
            params.append(json.load(fp))
    return params
