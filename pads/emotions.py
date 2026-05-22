"""
Map PAD posteriors to qualitative emotion labels.

The PAD (Pleasure / Arousal / Dominance) representation places affective
states in a 3-D continuous space. To make results more readable we map
the three posteriors to one of eight Mehrabian-style emotion quadrants,
following Russell (1980) and Mehrabian (1996).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class EmotionQuadrant:
    name: str
    valence: int    # +1 positive, -1 negative
    arousal: int    # +1 active, -1 passive
    dominance: int  # +1 dominant, -1 submissive
    emoji: str      # short label glyph


# Mehrabian's PAD octants. Names follow the standard literature.
EMOTION_QUADRANTS: Tuple[EmotionQuadrant, ...] = (
    EmotionQuadrant("Exuberant",   +1, +1, +1, ":D"),
    EmotionQuadrant("Dependent",   +1, +1, -1, ":)"),
    EmotionQuadrant("Relaxed",     +1, -1, +1, ":>"),
    EmotionQuadrant("Docile",      +1, -1, -1, ":."),
    EmotionQuadrant("Hostile",     -1, +1, +1, ">:("),
    EmotionQuadrant("Anxious",     -1, +1, -1, ":S"),
    EmotionQuadrant("Disdainful",  -1, -1, +1, ":/"),
    EmotionQuadrant("Bored",       -1, -1, -1, ":|"),
)


def _sign(prob_positive: float) -> int:
    """Convert a posterior into a +1/-1 sign vote."""
    return 1 if prob_positive >= 0.5 else -1


def pad_to_emotion(
    valence: float,
    arousal: float,
    dominance: float,
) -> EmotionQuadrant:
    """Pick the closest PAD octant for the given posteriors.

    Each posterior should be the probability of the *positive* class
    (i.e. positive valence, active arousal, dominant). They are reduced
    to +/-1 signs and the matching quadrant is returned.
    """
    sv = _sign(float(valence))
    sa = _sign(float(arousal))
    sd = _sign(float(dominance))
    for q in EMOTION_QUADRANTS:
        if q.valence == sv and q.arousal == sa and q.dominance == sd:
            return q
    raise RuntimeError("no quadrant matched - this should be unreachable")


def emotion_distribution(
    valence: Iterable[float],
    arousal: Iterable[float],
    dominance: Iterable[float],
) -> Dict[str, float]:
    """Return the share of time each emotion octant occupies.

    Useful for whole-utterance summaries: pass per-clip posteriors for
    each dimension and the function returns a dict mapping octant name
    to its relative frequency in ``[0, 1]``.
    """
    valence = np.asarray(list(valence), dtype=np.float32)
    arousal = np.asarray(list(arousal), dtype=np.float32)
    dominance = np.asarray(list(dominance), dtype=np.float32)
    n = len(valence)
    if not (len(arousal) == n == len(dominance)):
        raise ValueError("valence, arousal, dominance must have the same length")

    counts: Dict[str, int] = {q.name: 0 for q in EMOTION_QUADRANTS}
    for v, a, d in zip(valence, arousal, dominance):
        q = pad_to_emotion(v, a, d)
        counts[q.name] += 1
    return {name: c / n if n else 0.0 for name, c in counts.items()}
