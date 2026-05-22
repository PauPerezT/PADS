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
    page_title="PADS - Speech PAD Analysis",
    page_icon="PAD",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.metric-big {
    background: #f6f8fb;
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    padding: 16px 18px;
    color: #17202a;
    margin-bottom: 10px;
}
.metric-big h3 { margin: 0; font-size: 14px; color: #52616b; font-weight: 600; }
.metric-big p  { margin: 6px 0 0; font-size: 28px; font-weight: 700; }
.emotion-card {
    background: #17202a;
    border-radius: 8px;
    padding: 22px;
    text-align: center;
    border: 1px solid #263445;
    color: #ffffff;
}
.emotion-card .name { font-size: 32px; font-weight: 700; margin: 6px 0; }
.emotion-card .glyph { font-size: 56px; }
.small-note { color: #c7d0d9; font-size: 13px; }
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

    st.markdown("### Dimensions")
    dimension_options = {
        "Arousal": "arousal",
        "Valence / pleasure": "valence",
        "Dominance": "dominance",
    }
    selected_dimension_labels = st.multiselect(
        "PAD dimensions",
        list(dimension_options.keys()),
        default=list(dimension_options.keys()),
    )
    selected_dims = tuple(dimension_options[label] for label in selected_dimension_labels)

    st.markdown("### Outputs")
    show_posteriors = st.checkbox("Posterior probabilities", value=True)
    return_embeddings = st.checkbox("Embeddings", value=False)
    show_emotions = st.checkbox("PAD-to-emotion conversion", value=True)

    st.markdown("### Inference")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    batch_size = st.slider("Batch size", 1, 64, 16)

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

st.title("PADS")
st.subheader("Pleasure-Arousal-Dominance analysis from speech")

tab_upload, tab_record, tab_demo = st.tabs(["Upload audio", "Record audio", "Demo"])

uploaded = None
recorded = None
use_demo = False

with tab_upload:
    uploaded = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

with tab_record:
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Microphone recording", label_visibility="collapsed")
    else:
        st.info("Microphone recording is available in newer Streamlit versions.")

with tab_demo:
    use_demo = st.button("Load demo audio", use_container_width=True)

# Demo audio generation (for users without a file handy).
if use_demo and uploaded is None and recorded is None:
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
input_file = uploaded or recorded
if input_file is not None:
    suffix = os.path.splitext(input_file.name)[1] or ".wav"
    wav, sr, _path = load_audio_bytes(input_file.getvalue(), suffix)
elif "_demo_wav" in st.session_state:
    wav = st.session_state["_demo_wav"]
    sr = st.session_state["_demo_sr"]

if wav is None:
    st.info("Add an audio file, record from the browser, or load the demo audio.")
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

if not selected_dims:
    st.warning("Select at least one PAD dimension in the sidebar.")
    st.stop()

if st.button("Analyze audio", type="primary", use_container_width=True):
    try:
        extractor = get_extractor(ckpt_dir, legacy, data_norm_dir, device)
    except Exception as e:
        st.error(f"Could not initialise extractor: {e}")
        st.stop()

    with st.spinner("Running models..."):
        try:
            result = extractor.extract(
                wav, sample_rate=sr,
                dims=selected_dims,
                batch_size=batch_size,
                return_embeddings=return_embeddings,
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    st.success(f"Done. {result.n_clips} clips analysed.")

    # --- summary cards ---
    metric_values = [
        ("Arousal", result.arousal_mean, result.arousal.size),
        ("Valence", result.valence_mean, result.valence.size),
        ("Dominance", result.dominance_mean, result.dominance.size),
    ]
    visible_metrics = [metric for metric in metric_values if metric[2]]
    metric_cols = st.columns(max(1, len(visible_metrics)))
    for col, (name, val, _size) in zip(metric_cols, visible_metrics):
        col.markdown(
            f"<div class='metric-big'><h3>{name}</h3><p>{val:.2f}</p></div>",
            unsafe_allow_html=True,
        )

    if show_emotions and result.has_all_dimensions():
        top_em = result.dominant_emotion()
        st.markdown(
            f"<div class='emotion-card'>"
            f"<div class='glyph'>{top_em.emoji}</div>"
            f"<div class='name'>{top_em.name}</div>"
            f"<div class='small-note'>dominant PAD octant</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif show_emotions:
        st.info("PAD-to-emotion conversion needs arousal, valence, and dominance.")

    if show_posteriors:
        st.markdown("---")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.plotly_chart(plot_pad_timeline(result), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_pad_radar(result), use_container_width=True)

    if show_emotions and result.has_all_dimensions():
        st.plotly_chart(plot_emotion_distribution(result), use_container_width=True)

    if show_posteriors:
        st.markdown("#### Per-clip posteriors")
        df = result.to_dataframe()
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="pad_posteriors.csv",
            mime="text/csv",
        )

    if return_embeddings and result.embeddings:
        st.markdown("#### Embeddings")
        emb_summary = {
            dim: f"{values.shape[0]} clips x {values.shape[1]} features"
            for dim, values in result.embeddings.items()
        }
        st.json(emb_summary)
        emb_buf = io.BytesIO()
        np.savez(emb_buf, **result.embeddings)
        st.download_button(
            "Download embeddings",
            emb_buf.getvalue(),
            file_name="pad_embeddings.npz",
            mime="application/octet-stream",
        )

    with st.expander("PAD octant reference"):
        ref = [
            {"Octant": q.name, "Valence": "+" if q.valence > 0 else "-",
             "Arousal": "+" if q.arousal > 0 else "-",
             "Dominance": "+" if q.dominance > 0 else "-"}
            for q in EMOTION_QUADRANTS
        ]
        st.table(ref)
