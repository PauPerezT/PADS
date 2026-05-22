"""
High-level PADS inference.

The :class:`PADExtractor` wraps the entire pipeline:

    audio file -> waveform (16 kHz) -> 3-channel mel-spectrogram clips
    -> normalised batch -> per-dimension classifier -> PAD posteriors

A functional shortcut is also provided:

    >>> from pads import extract_features
    >>> feats = extract_features("speech.wav", outputs=("posteriors", "emotion"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .features import Spectrogram2DTensor, SpectrogramConfig
from .normalize import normalize_batch_per_channel, normalize_tensor_minmax, load_norm_params
from .models import load_lightning_model, load_legacy_model
from .emotions import pad_to_emotion, emotion_distribution


PAD_DIMENSIONS = ("arousal", "valence", "dominance")
PAD_OUTPUTS = ("posteriors", "embeddings", "emotion")


@dataclass
class PADResult:
    """Container for the output of PADExtractor.extract."""

    arousal: np.ndarray
    valence: np.ndarray
    dominance: np.ndarray
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    clip_hop_seconds: float = 0.5

    @property
    def n_clips(self) -> int:
        for arr in (self.arousal, self.valence, self.dominance):
            if arr.size:
                return int(arr.shape[0])
        return 0

    def available_dimensions(self) -> List[str]:
        return [
            name for name, arr in (
                ("arousal", self.arousal),
                ("valence", self.valence),
                ("dominance", self.dominance),
            )
            if arr.size
        ]

    def has_all_dimensions(self) -> bool:
        return len(self.available_dimensions()) == 3

    @staticmethod
    def _mean_or_nan(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")

    @property
    def arousal_mean(self) -> float:
        return self._mean_or_nan(self.arousal)

    @property
    def valence_mean(self) -> float:
        return self._mean_or_nan(self.valence)

    @property
    def dominance_mean(self) -> float:
        return self._mean_or_nan(self.dominance)

    def dominant_emotion(self):
        if not self.has_all_dimensions():
            raise ValueError(
                "Need arousal, valence, AND dominance posteriors to map to an emotion octant. "
                f"Available: {self.available_dimensions()}"
            )
        return pad_to_emotion(self.valence_mean, self.arousal_mean, self.dominance_mean)

    def emotion_distribution(self) -> Dict[str, float]:
        if not self.has_all_dimensions():
            raise ValueError(
                "Need arousal, valence, AND dominance posteriors. "
                f"Available: {self.available_dimensions()}"
            )
        return emotion_distribution(self.valence, self.arousal, self.dominance)

    def timestamps(self) -> np.ndarray:
        starts = np.arange(self.n_clips, dtype=np.float32) * self.clip_hop_seconds
        return starts + self.clip_seconds / 2

    def summary(self) -> str:
        parts = []
        if self.arousal.size:
            parts.append(f"A={self.arousal_mean:.2f}")
        if self.valence.size:
            parts.append(f"V={self.valence_mean:.2f}")
        if self.dominance.size:
            parts.append(f"D={self.dominance_mean:.2f}")
        out = f"PADResult: {self.n_clips} clips, " + " ".join(parts)
        if self.has_all_dimensions():
            em = self.dominant_emotion()
            out += f" -> {em.name} {em.emoji}"
        return out

    def to_dataframe(self):
        import pandas as pd
        data = {"time_s": self.timestamps()}
        if self.arousal.size:
            data["arousal"] = self.arousal
        if self.valence.size:
            data["valence"] = self.valence
        if self.dominance.size:
            data["dominance"] = self.dominance
        return pd.DataFrame(data)


class PADExtractor:
    """End-to-end PAD posterior extractor."""

    def __init__(
        self,
        checkpoints_dir="checkpoints/one_sec",
        data_norm_dir=None,
        device: str = "cpu",
        legacy: bool = False,
        cfg=None,
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device
        self.legacy = legacy
        self.cfg = cfg or SpectrogramConfig(
            clip_seconds=(0.5 if legacy else 1.0),
            clip_hop_seconds=(0.25 if legacy else 0.5),
        )
        self._models: Dict[str, torch.nn.Module] = {}
        self._norm_params = None
        if legacy:
            if data_norm_dir is None:
                raise ValueError("legacy=True requires data_norm_dir")
            self._norm_params = load_norm_params(data_norm_dir)

    def _get_model(self, dim: str):
        if dim not in self._models:
            suffix = ".ckp" if self.legacy else ".ckpt"
            path = self.checkpoints_dir / f"{dim}_checkpoint{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            if self.legacy:
                model = load_legacy_model(path, device=self.device)
            else:
                model = load_lightning_model(path, device=self.device)
            self._models[dim] = model
        return self._models[dim]

    def extract(
        self,
        audio,
        sample_rate=None,
        dims: Sequence[str] = PAD_DIMENSIONS,
        batch_size: int = 32,
        return_embeddings: bool = False,
        apply_sigmoid: bool = False,
    ) -> PADResult:
        wav, sr = self._load_audio(audio, sample_rate)
        spec_builder = Spectrogram2DTensor(wav, sample_rate=sr, cfg=self.cfg)
        clips = spec_builder.get_2d_tensor_clips()

        if self.legacy:
            clips = normalize_tensor_minmax(clips, self._norm_params)
        else:
            clips = normalize_batch_per_channel(clips)

        clips_t = torch.from_numpy(clips).to(self.device)

        per_dim_scores: Dict[str, np.ndarray] = {}
        per_dim_emb: Dict[str, np.ndarray] = {}
        for dim in dims:
            if dim not in PAD_DIMENSIONS:
                raise ValueError(
                    f"Unknown dimension {dim!r}; expected one of {PAD_DIMENSIONS}"
                )
            model = self._get_model(dim)
            scores, emb = self._infer_in_batches(model, clips_t, batch_size)
            if apply_sigmoid:
                probs = torch.sigmoid(scores)
            else:
                probs = torch.softmax(scores, dim=1)
            per_dim_scores[dim] = probs[:, 1].cpu().numpy().astype(np.float32)
            if return_embeddings:
                per_dim_emb[dim] = emb.cpu().numpy().astype(np.float32)

        return PADResult(
            arousal=per_dim_scores.get("arousal", np.empty(0, dtype=np.float32)),
            valence=per_dim_scores.get("valence", np.empty(0, dtype=np.float32)),
            dominance=per_dim_scores.get("dominance", np.empty(0, dtype=np.float32)),
            embeddings=per_dim_emb,
            sample_rate=sr,
            clip_seconds=self.cfg.clip_seconds,
            clip_hop_seconds=self.cfg.clip_hop_seconds,
        )

    def _load_audio(self, audio, sample_rate):
        target_sr = self.cfg.sample_rate
        if isinstance(audio, (str, Path)):
            import librosa
            wav, sr = librosa.load(str(audio), sr=target_sr, mono=True)
            return wav.astype(np.float32), target_sr
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sample_rate is None:
            raise ValueError("sample_rate must be given when audio is a numpy array")
        if sample_rate != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)
        return wav, target_sr

    @torch.inference_mode()
    def _infer_in_batches(self, model, clips, batch_size):
        n = clips.shape[0]
        scores_chunks: List[torch.Tensor] = []
        emb_chunks: List[torch.Tensor] = []
        for start in range(0, n, batch_size):
            chunk = clips[start:start + batch_size]
            s, e = model(chunk)
            scores_chunks.append(s)
            emb_chunks.append(e)
        return torch.cat(scores_chunks, dim=0), torch.cat(emb_chunks, dim=0)


def extract_features(
    audio,
    *,
    outputs: Sequence[str] = ("posteriors",),
    dims: Sequence[str] = PAD_DIMENSIONS,
    dimensions: Optional[Sequence[str]] = None,
    sample_rate=None,
    checkpoints_dir="checkpoints/one_sec",
    data_norm_dir=None,
    device: str = "cpu",
    legacy: bool = False,
    batch_size: int = 32,
    apply_sigmoid: bool = False,
) -> Dict[str, "object"]:
    """One-shot helper that wraps PADExtractor.

    Example
    -------
        >>> from pads import extract_features
        >>> feats = extract_features("speech.wav", outputs=("posteriors", "emotion"))
        >>> feats["posteriors"].head()
        >>> feats["emotion"].name

    outputs may include "posteriors", "embeddings", "emotion", "result".
    Use dims=... or dimensions=... to choose arousal, valence, dominance, or
    any subset.
    """
    selected_dims = tuple(dimensions if dimensions is not None else dims)
    bad_dims = sorted(set(selected_dims) - set(PAD_DIMENSIONS))
    if bad_dims:
        raise ValueError(
            f"Unknown dimension(s) {bad_dims!r}; "
            f"expected a subset of {list(PAD_DIMENSIONS)}"
        )

    allowed = set(PAD_OUTPUTS) | {"result"}
    unknown = set(outputs) - allowed
    if unknown:
        raise ValueError(
            f"Unknown output(s) {sorted(unknown)!r}; "
            f"expected a subset of {sorted(allowed)}"
        )

    extractor = PADExtractor(
        checkpoints_dir=checkpoints_dir,
        data_norm_dir=data_norm_dir,
        device=device,
        legacy=legacy,
    )

    result = extractor.extract(
        audio,
        sample_rate=sample_rate,
        dims=selected_dims,
        batch_size=batch_size,
        return_embeddings="embeddings" in outputs,
        apply_sigmoid=apply_sigmoid,
    )

    out: Dict[str, "object"] = {}
    if "posteriors" in outputs:
        out["posteriors"] = result.to_dataframe()
    if "embeddings" in outputs:
        out["embeddings"] = result.embeddings
    if "emotion" in outputs:
        out["emotion"] = result.dominant_emotion() if result.has_all_dimensions() else None
    if "result" in outputs:
        out["result"] = result
    return out
