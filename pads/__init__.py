"""
PADS - Pleasure-Arousal-Dominance representations from Speech.

A lightweight Python library to extract PAD (Pleasure-Arousal-Dominance)
emotional features from speech signals.

Quickstart
----------
    >>> from pads import extract_features
    >>> features = extract_features("audio.wav", outputs=("posteriors", "emotion"))
    >>> print(features["posteriors"].head())

Importing :mod:`pads` does NOT import PyTorch. The torch-requiring symbols
(``PADExtractor``, ``PADResult``, ``extract_features``, ``SelfConv``) are
loaded lazily on first access.
"""

from __future__ import annotations

from .features import (
    Spectrogram2DTensor,
    SpectrogramConfig,
    mel_filterbank,
    log_mel_spectrum,
    extract_pad_tensors,
)
from .normalize import normalize_spectrum, normalize_tensor_minmax
from .emotions import pad_to_emotion, EMOTION_QUADRANTS

__version__ = "1.0.0"

# Names exported lazily (require torch).
_LAZY = {
    "PADExtractor":     ("pads.inference", "PADExtractor"),
    "PADResult":        ("pads.inference", "PADResult"),
    "extract_features": ("pads.inference", "extract_features"),
    "SelfConv":         ("pads.models",    "SelfConv"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module 'pads' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "PADExtractor",
    "PADResult",
    "extract_features",
    "SelfConv",
    "Spectrogram2DTensor",
    "SpectrogramConfig",
    "mel_filterbank",
    "log_mel_spectrum",
    "extract_pad_tensors",
    "normalize_spectrum",
    "normalize_tensor_minmax",
    "pad_to_emotion",
    "EMOTION_QUADRANTS",
    "__version__",
]
