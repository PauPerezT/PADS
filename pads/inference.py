"""
High-level PADS inference.

The :class:`PADExtractor` wraps the entire pipeline:

    audio file -> waveform (16 kHz) -> 3-channel mel-spectrogram clips
    -> normalised batch -> per-dimension classifier -> PAD posteriors

Once instantiated it caches the three models (arousal, valence, dominance)
so subsequent calls only re-run the cheap feature extractor.

Example
-------
    >>> from pads import PADExtractor
    >>> ext = PADExtractor(checkpoints_dir="checkpoints/one_sec")
    >>> result = ext.extract("speech.wav")
    >>> print(result.summary())
    >>> df = result.to_dataframe()  # one row per 1s clip
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
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


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class PADResult:
    """Container for the output of :meth:`PADExtractor.extract`.

    Attributes
    ----------
    arousal, valence, dominance : np.ndarray
        Posterior probabilities of the *positive* class (active arousal,
        positive valence, dominant) per 1 s clip. Shape ``(n_clips,)``.
    embeddings : dict[str, np.ndarray]
        Per-dimension embedding vectors per clip, shape ``(n_clips, 256)``.
    sample_rate : int
        Sampling rate the audio was resampled to (always 16000).
    clip_seconds, clip_hop_seconds : float
        Clip length and hop used for inference.
    """

    arousal: np.ndarray
    valence: np.ndarray
    dominance: np.ndarray
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    clip_hop_seconds: float = 0.5

    # --- summary helpers ----------------------------------------------------

    @property
    def n_clips(self) -> int:
        for values in (self.arousal, self.valence, self.dominance):
            if values.size:
                return int(values.shape[0])
        return 0

    def available_dimensions(self) -> List[str]:
        """Return the PAD dimensions present in this result."""
        return [
            dim for dim, values in (
                ("arousal", self.arousal),
                ("valence", self.valence),
                ("dominance", self.dominance),
            )
            if values.size
        ]

    def has_all_dimensions(self) -> bool:
        """Return True when arousal, valence, and dominance are all present."""
        return all(values.size for values in (self.arousal, self.valence, self.dominance))

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
            raise ValueError("dominant_emotion requires arousal, valence, and dominance")
        return pad_to_emotion(self.valence_mean, self.arousal_mean, self.dominance_mean)

    def emotion_distribution(self) -> Dict[str, float]:
        if not self.has_all_dimensions():
            raise ValueError("emotion_distribution requires arousal, valence, and dominance")
        return emotion_distribution(self.valence, self.arousal, self.dominance)

    def timestamps(self) -> np.ndarray:
        """Return the centre timestamp (in seconds) of each clip."""
        starts = np.arange(self.n_clips, dtype=np.float32) * self.clip_hop_seconds
        return starts + self.clip_seconds / 2

    def summary(self) -> str:
        """Human-readable summary string."""
        parts = []
        if self.arousal.size:
            parts.append(f"A={self.arousal_mean:.2f}")
        if self.valence.size:
            parts.append(f"V={self.valence_mean:.2f}")
        if self.dominance.size:
            parts.append(f"D={self.dominance_mean:.2f}")
        summary = f"PADResult: {self.n_clips} clips, " + " ".join(parts)
        if self.has_all_dimensions():
            em = self.dominant_emotion()
            summary += f" -> {em.name} {em.emoji}"
        return summary

    def to_dataframe(self):
        """Return a pandas DataFrame (one row per clip)."""
        import pandas as pd
        data = {"time_s": self.timestamps()}
        if self.arousal.size:
            data["arousal"] = self.arousal
        if self.valence.size:
            data["valence"] = self.valence
        if self.dominance.size:
            data["dominance"] = self.dominance
        return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Extractor
# -----------------------------------------------------------------------------

class PADExtractor:
    """End-to-end PAD posterior extractor.

    Parameters
    ----------
    checkpoints_dir : str or Path
        Folder containing ``arousal_checkpoint.ckpt``, ``valence_checkpoint.ckpt``,
        ``dominance_checkpoint.ckpt`` (Lightning) or ``*_checkpoint.ckp`` (legacy).
    data_norm_dir : str or Path, optional
        Folder with ``param_c{1,2,3}.json`` for legacy checkpoints. Ignored in
        Lightning mode.
    device : {"cpu", "cuda"}
    legacy : bool
        If ``True`` use the older ``.ckp`` checkpoints (input 50x64) and
        pre-fit per-channel normalisation. If ``False`` (default) use the
        Lightning ``.ckpt`` checkpoints (input 100x64) with per-spectrum
        normalisation.
    cfg : SpectrogramConfig, optional
        Override the analysis parameters. Defaults match the published
        checkpoints.
    """

    def __init__(
        self,
        checkpoints_dir: str | Path = "checkpoints/one_sec",
        data_norm_dir: Optional[str | Path] = None,
        device: str = "cpu",
        legacy: bool = False,
        cfg: Optional[SpectrogramConfig] = None,
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

    # ------------------------------------------------------------------
    # Model loading (lazy + cached per instance)
    # ------------------------------------------------------------------

    def _get_model(self, dim: str) -> torch.nn.Module:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: Optional[int] = None,
        dims: Sequence[str] = PAD_DIMENSIONS,
        batch_size: int = 32,
        return_embeddings: bool = False,
        apply_sigmoid: bool = False,
    ) -> PADResult:
        """Run the full PADS pipeline on an audio file or waveform.

        Parameters
        ----------
        audio : path-like or 1-D array
            Either a path to an audio file (anything librosa can read) or
            a numpy waveform (mono).
        sample_rate : int, optional
            Required when ``audio`` is a numpy array. Ignored for files.
        dims : sequence of str
            Subset of ``("arousal", "valence", "dominance")`` to predict.
        batch_size : int
            Inference batch size. The default fits comfortably on a CPU.
        return_embeddings : bool
            If True, populate ``PADResult.embeddings`` with the
            pre-classifier feature vectors.
        apply_sigmoid : bool
            If True, apply sigmoid to scores. Otherwise softmax (default).
        """
        # 1. Load waveform.
        wav, sr = self._load_audio(audio, sample_rate)

        # 2. Build 3-channel spectrogram clips.
        spec_builder = Spectrogram2DTensor(wav, sample_rate=sr, cfg=self.cfg)
        clips = spec_builder.get_2d_tensor_clips()  # (N, 3, T, F)

        # 3. Normalise.
        if self.legacy:
            clips = normalize_tensor_minmax(clips, self._norm_params)
        else:
            clips = normalize_batch_per_channel(clips)

        clips_t = torch.from_numpy(clips).to(self.device)

        # 4. Run each dimension.
        per_dim_scores: Dict[str, np.ndarray] = {}
        per_dim_emb: Dict[str, np.ndarray] = {}
        for dim in dims:
            if dim not in PAD_DIMENSIONS:
                raise ValueError(f"Unknown dimension {dim!r}; expected one of {PAD_DIMENSIONS}")
            model = self._get_model(dim)
            scores, emb = self._infer_in_batches(model, clips_t, batch_size)
            if apply_sigmoid:
                probs = torch.sigmoid(scores)
            else:
                probs = torch.softmax(scores, dim=1)
            # Positive-class posterior (index 1).
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_audio(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: Optional[int],
    ):
        target_sr = self.cfg.sample_rate
        if isinstance(audio, (str, Path)):
            import librosa
            wav, sr = librosa.load(str(audio), sr=target_sr, mono=True)
            return wav.astype(np.float32), target_sr
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)  # downmix to mono
        if sample_rate is None:
            raise ValueError("sample_rate must be given when audio is a numpy array")
        if sample_rate != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)
        return wav, target_sr

    @torch.inference_mode()
    def _infer_in_batches(
        self,
        model: torch.nn.Module,
        clips: torch.Tensor,
        batch_size: int,
    ):
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
    audio: str | Path | np.ndarray,
    sample_rate: Optional[int] = None,
    checkpoints_dir: str | Path = "checkpoints/one_sec",
    dimensions: Sequence[str] = PAD_DIMENSIONS,
    outputs: Sequence[str] = ("posteriors",),
    device: str = "cpu",
    batch_size: int = 32,
    legacy: bool = False,
    data_norm_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Convenience feature-extractor API.

    This is the easiest entry point for users who want PADS as a feature
    extractor rather than as a class-based pipeline.

    Examples
    --------
    >>> from pads import extract_features
    >>> out = extract_features(
    ...     "speech.wav",
    ...     dimensions=("arousal", "valence", "dominance"),
    ...     outputs=("posteriors", "emotion"),
    ... )
    >>> out["posteriors"].head()
    """
    bad_dims = sorted(set(dimensions) - set(PAD_DIMENSIONS))
    if bad_dims:
        raise ValueError(f"Unknown dimensions {bad_dims}; expected values in {PAD_DIMENSIONS}")

    bad_outputs = sorted(set(outputs) - set(PAD_OUTPUTS))
    if bad_outputs:
        raise ValueError(f"Unknown outputs {bad_outputs}; expected values in {PAD_OUTPUTS}")

    extractor = PADExtractor(
        checkpoints_dir=checkpoints_dir,
        data_norm_dir=data_norm_dir,
        device=device,
        legacy=legacy,
    )
    result = extractor.extract(
        audio,
        sample_rate=sample_rate,
        dims=dimensions,
        batch_size=batch_size,
        return_embeddings=("embeddings" in outputs),
    )

    features: Dict[str, object] = {"result": result}
    if "posteriors" in outputs:
        features["posteriors"] = result.to_dataframe()
    if "embeddings" in outputs:
        features["embeddings"] = result.embeddings
    if "emotion" in outputs:
        if result.has_all_dimensions():
            features["emotion"] = result.dominant_emotion()
            features["emotion_distribution"] = result.emotion_distribution()
        else:
            features["emotion"] = None
            features["emotion_distribution"] = None
    return features
