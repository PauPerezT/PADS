"""
Plotly visualisations for PADResult objects.

Pure helpers - they take a :class:`PADResult` (or arrays) and return Plotly
``Figure`` objects ready to render in Streamlit, Jupyter, or any web app.

We use Plotly rather than matplotlib (the original PADS dependency) so
that figures render interactively in browsers without an Agg backend.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _ensure_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "plotly is required for pads.visualization. Install with `pip install plotly`."
        ) from e


def plot_pad_timeline(result, height: int = 380):
    """Line plot of per-clip PAD posteriors over time."""
    _ensure_plotly()
    import plotly.graph_objects as go

    t = result.timestamps()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=result.arousal,   mode="lines+markers",
                             name="Arousal",   line=dict(color="#E45756", width=2)))
    fig.add_trace(go.Scatter(x=t, y=result.valence,   mode="lines+markers",
                             name="Valence",   line=dict(color="#4C78A8", width=2)))
    fig.add_trace(go.Scatter(x=t, y=result.dominance, mode="lines+markers",
                             name="Dominance", line=dict(color="#54A24B", width=2)))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="PAD posteriors over time",
        xaxis_title="Time (s)",
        yaxis_title="P(positive class)",
        yaxis=dict(range=[0, 1]),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_pad_radar(result, height: int = 380):
    """Radar plot of mean PAD posteriors."""
    _ensure_plotly()
    import plotly.graph_objects as go

    means = [result.arousal_mean, result.valence_mean, result.dominance_mean]
    labels = ["Arousal", "Valence", "Dominance"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=means + [means[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Mean posteriors",
        line=dict(color="#7B61FF"),
        fillcolor="rgba(123, 97, 255, 0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        title="Mean posteriors",
    )
    return fig


def plot_emotion_distribution(result, height: int = 380):
    """Horizontal bar chart of time spent in each emotion octant."""
    _ensure_plotly()
    import plotly.graph_objects as go

    dist = result.emotion_distribution()
    names = list(dist.keys())
    vals = [dist[n] * 100 for n in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color="#7B61FF"),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="Time spent per emotion (PAD octants)",
        xaxis_title="% of clips",
        height=height,
        margin=dict(l=80, r=40, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_mel_spectrogram(spec: np.ndarray, sample_rate: int = 16000,
                         step_s: float = 0.01, height: int = 320,
                         title: str = "Log-mel spectrogram"):
    """Heatmap of a single-channel log-mel spectrogram (shape: T x F)."""
    _ensure_plotly()
    import plotly.graph_objects as go

    if spec.ndim == 3:  # take channel 0 if a tensor was passed
        spec = spec[0]
    spec_t = spec.T  # frequency on y axis
    times = np.arange(spec.shape[0]) * step_s
    fig = go.Figure(go.Heatmap(
        z=spec_t,
        x=times,
        colorscale="Inferno",
        colorbar=dict(title="log energy"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Mel band",
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def plot_waveform(wav: np.ndarray, sample_rate: int = 16000, height: int = 200):
    """Time-domain waveform plot (downsampled for fast rendering)."""
    _ensure_plotly()
    import plotly.graph_objects as go

    wav = np.asarray(wav).astype(np.float32)
    # Downsample for plotting if waveform is very long.
    max_points = 4000
    if len(wav) > max_points:
        idx = np.linspace(0, len(wav) - 1, max_points).astype(int)
        ys = wav[idx]
        xs = idx / sample_rate
    else:
        ys = wav
        xs = np.arange(len(wav)) / sample_rate
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines",
                               line=dict(color="#4C78A8", width=1)))
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
