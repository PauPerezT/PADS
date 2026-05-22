"""
PADS Streamlit web interface.

Run with::

    streamlit run web/app.py

The app lets you upload an audio file (or record live, where supported),
extracts mel-spectrogram features with :mod:`pads.features`, runs the
PAD classifiers, and shows interactive Plotly charts of:

* the waveform,
* a log-mel spectrogram,
* per-clip PAD posteriors over time,
* the radar of mean posteriors,
* the dominant emotion + an octant distribution bar chart.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st


# --- bootstrap ------------------------------------------------------------------

# Add project root to sys.path so the local `pads` package is importable when
# Streamlit is launched from the project directory.
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pads import PADExtractor, Spectrogram2DTensor
from pads.visualization import (
    plot_emotion_distribution,
    plot_mel_spectrogram,
    plot_pad_radar,
    plot_pad_timeline,
    plot_waveform,
)
from pads.emotions import EMOTION_QUADRANTS


# --- page config ----------------------------------------------------------------

st.set_page_config(
    page_title="PADS - Speech Emotion Explorer",
    page_icon="[*]",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.metric-big {
    background: linear-gradient(135deg, #7B61FF 0%, #4C78A8 100%);
    border-radius: 14px;
    padding: 16px 18px;
    color: white;
    margin-bottom: 10px;
}
.metric-big h3 { margin: 0; font-size: 14px; opacity: 0.85; font-weight: 500; }
.metric-big p  { margin: 6px 0 0; font-size: 28px; font-weight: 700; }
.emotion-card {
    background: #1f1f29;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    border: 1px solid #2c2c3c;
}
.emotion-card .name { font-size: 32px; font-weight: 700; margin: 6px 0; }
.emotion-card .glyph { font-size: 56px; }
.small-note { color: #8a8a99; font-size: 13px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- sidebar --------------------------------------------------------------------

with st.sidebar:
    st.title("PADS")
    st.caption("Pleasure-Arousal-Dominance from speech")

    st.markdown("### Model")
    ckpt_dir = st.text_input(
        "Checkpoints directory",
        value=str(_ROOT / "checkpoints" / "one_sec"),
        help="Folder containing arousal/valence/dominance .ckpt files",
    )
    legacy = st.checkbox(
        "Legacy mode (.ckp + 50x64 input)",
        value=False,
        help="Tick this if you only have the older non-Lightning checkpoints.",
    )
    data_norm_dir = None
    if legacy:
        data_norm_dir = st.text_input(
            "Data norm directory",
            value=str(_ROOT / "data_norm"),
            help="Folder containing param_c{1,2,3}.json (legacy normalisation).",
        )

    st.markdown("### Inference")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    batch_size = st.slider("Batch size", 1, 64, 16)
    return_embeddings = st.checkbox("Return embeddings", value=False)

    st.markdown("---")
    st.caption(
        "Built with [Streamlit](https://streamlit.io) and the optimised "
        "[pads](https://github.com/PauPerezT/PADS) library."
    )


# --- helpers --------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PAD models...")
def get_extractor(ckpt_dir: str, legacy: bool, data_norm_dir: str | None, device: str):
    return PADExtractor(
        checkpoints_dir=ckpt_dir,
        data_norm_dir=data_norm_dir,
        device=device,
        legacy=legacy,
    )


@st.cache_data(show_spinner=False)
def load_audio_bytes(buf: bytes, suffix: str):
    """Persist uploaded bytes to a temp file and load as a 16kHz waveform."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(buf)
        path = tmp.name
    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32), 16000, path


# --- main UI --------------------------------------------------------------------

st.title("PADS - Speech Emotion Explorer")
st.markdown(
    "Upload an audio recording, and PADS will predict the speaker's "
    "**Pleasure (Valence)**, **Arousal**, and **Dominance** over time."
)

uploaded = st.file_uploader(
    "Choose an audio file (wav, mp3, flac, ogg, m4a)",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    accept_multiple_files=False,
)

col_demo, col_rec = st.columns([1, 2])
with col_demo:
    use_demo = st.button("Use synthetic demo audio", use_container_width=True)
with col_rec:
    st.caption("Tip: live recording is supported on Chrome via the browser's mic API.")

# Demo audio generation (for users without a file handy).
if use_demo and uploaded is None:
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Synthetic vowel-like signal with vibrato.
    f0 = 180 + 6 * np.sin(2 * np.pi * 5 * t)
    wav = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)
    wav *= np.exp(-0.5 * t)  # decay envelope
    st.session_state["_demo_wav"] = wav
    st.session_state["_demo_sr"] = sr
    st.success("Synthetic demo audio generated.")

# Resolve waveform.
wav, sr = None, 16000
if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    wav, sr, _path = load_audio_bytes(uploaded.getvalue(), suffix)
elif "_demo_wav" in st.session_state:
    wav = st.session_state["_demo_wav"]
    sr = st.session_state["_demo_sr"]

if wav is None:
    st.info(
        "Upload a file above, or click *Use synthetic demo audio* to try the pipeline "
        "with a tone-based stand-in."
    )
    st.stop()

# Show the waveform + an audio player.
st.markdown("#### Input audio")
st.plotly_chart(plot_waveform(wav, sample_rate=sr), use_container_width=True)

buf = io.BytesIO()
import soundfile as sf
sf.write(buf, wav, sr, format="WAV")
buf.seek(0)
st.audio(buf.read(), format="audio/wav")

# Always show a mel spectrogram (uses just the feature extractor, no model).
with st.spinner("Computing mel-spectrogram..."):
    spec_builder = Spectrogram2DTensor(wav, sample_rate=sr)
    spec_3ch = spec_builder.get_2d_spectrograms()

st.markdown("#### Multi-resolution log-mel spectrogram")
tab1, tab2, tab3 = st.tabs(["16 ms window", "25 ms window", "40 ms window"])
with tab1:
    st.plotly_chart(plot_mel_spectrogram(spec_3ch[0], title="Channel 1 - 16 ms"),
                    use_container_width=True)
with tab2:
    st.plotly_chart(plot_mel_spectrogram(spec_3ch[1], title="Channel 2 - 25 ms"),
                    use_container_width=True)
with tab3:
    st.plotly_chart(plot_mel_spectrogram(spec_3ch[2], title="Channel 3 - 40 ms"),
                    use_container_width=True)

# --- run PAD models -----------------------------------------------------------

ckpt_ok = Path(ckpt_dir).exists() and any(Path(ckpt_dir).glob("*.ckpt" if not legacy else "*.ckp"))
if not ckpt_ok:
    st.warning(
        f"No checkpoints found in `{ckpt_dir}`. "
        f"Feature extraction is working - drop the trained {'`*.ckpt`' if not legacy else '`*.ckp`'} files "
        "in the directory above to enable PAD inference."
    )
    st.stop()

if st.button("Run PAD inference", type="primary"):
    try:
        extractor = get_extractor(ckpt_dir, legacy, data_norm_dir, device)
    except Exception as e:
        st.error(f"Could not initialise extractor: {e}")
        st.stop()

    with st.spinner("Running models..."):
        try:
            result = extractor.extract(
                wav, sample_rate=sr,
                batch_size=batch_size,
                return_embeddings=return_embeddings,
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    st.success(f"Done. {result.n_clips} clips analysed.")

    # --- summary cards ---
    top_em = result.dominant_emotion()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    for col, name, val in zip(
        (c1, c2, c3),
        ("Arousal", "Valence", "Dominance"),
        (result.arousal_mean, result.valence_mean, result.dominance_mean),
    ):
        col.markdown(
            f"<div class='metric-big'><h3>{name}</h3><p>{val:.2f}</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='emotion-card'>"
            f"<div class='glyph'>{top_em.emoji}</div>"
            f"<div class='name'>{top_em.name}</div>"
            f"<div class='small-note'>dominant PAD octant</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(plot_pad_timeline(result), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_pad_radar(result), use_container_width=True)

    st.plotly_chart(plot_emotion_distribution(result), use_container_width=True)

    with st.expander("Per-clip results table"):
        df = result.to_dataframe()
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="pad_posteriors.csv",
            mime="text/csv",
        )

    with st.expander("PAD octant reference"):
        ref = [
            {"Octant": q.name, "Valence": "+" if q.valence > 0 else "-",
             "Arousal": "+" if q.arousal > 0 else "-",
             "Dominance": "+" if q.dominance > 0 else "-"}
            for q in EMOTION_QUADRANTS
        ]
        st.table(ref)
