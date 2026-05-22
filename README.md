<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# PADS

### Pleasure-Arousal-Dominance representations from Speech

*A faster, modular, web-ready rebuild of the original [PADS](https://github.com/PauPerezT/PADS) project.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/web-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/web-Gradio-F97316.svg)](https://www.gradio.app/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-2F6F9F.svg)](https://pauprezt.github.io/PADS/)
[![Tests](https://img.shields.io/badge/tests-10%2F10_passing-brightgreen.svg)](#testing)

</div>

---

## What this is

PADS predicts a speaker's **affective state** from a speech recording on three continuous dimensions:

| Dimension | Range | Meaning |
|---|---|---|
| **Pleasure / Valence** | negative <-> positive | how pleasant the speaker sounds |
| **Arousal** | passive <-> active | how energetic / activated |
| **Dominance** | submissive <-> dominant | how in-control / assertive |

The combination of those three placements falls into one of eight emotion octants (Mehrabian, 1996): *Exuberant, Dependent, Relaxed, Docile, Hostile, Anxious, Disdainful, Bored.*

This repository contains a rebuild of the original PADS code that focuses on three goals:

1. **A clean, pip-installable feature extractor** (`pads/`) - one function can return PAD posteriors, embeddings, and emotion labels.
2. **Web interfaces** (`web/app.py`, `web/gradio_app.py`, and `docs/`) - Streamlit and Gradio apps for audio upload/recording plus a GitHub Pages landing page for users.
3. **Measurable optimisations** - vectorised framing, cached mel filterbanks, batched inference. ~**2.3x faster** on the spectrogram pipeline with **numerically identical** output (max drift `<3e-6`).

---

## Quickstart

For most users, PADS is just a feature extractor.

Install directly from GitHub:

```bash
pip install git+https://github.com/PauPerezT/PADS.git
```

With the Gradio interface:

```bash
pip install "pads[gradio] @ git+https://github.com/PauPerezT/PADS.git"
```

With both web interfaces:

```bash
pip install "pads[all] @ git+https://github.com/PauPerezT/PADS.git"
```

For local development:

```bash
git clone https://github.com/PauPerezT/PADS.git
cd PADS
pip install -e ".[all]"
```

Place the pre-trained checkpoints from the [original PADS release](https://github.com/PauPerezT/PADS) in `checkpoints/one_sec/` (or in `checkpoints/` for legacy `.ckp` files).

Then extract features from one audio file:

```python
from pads import extract_features

features = extract_features(
    "speech.wav",
    checkpoints_dir="checkpoints/one_sec",
    dimensions=("arousal", "valence", "dominance"),
    outputs=("posteriors", "embeddings", "emotion"),
)

posteriors = features["posteriors"]      # pandas DataFrame, one row per clip
embeddings = features["embeddings"]      # dict: arousal/valence/dominance arrays
emotion = features["emotion"]            # dominant PAD emotion, if all 3 dimensions were extracted
```

...or launch a browser interface:

```bash
pads-gradio                     # Gradio, easiest for users
streamlit run web/app.py        # richer dashboard
```

---
## What's in the box

```
PADS-optimized/
├── app.py                 # Root Gradio entry point for simple launches / Spaces
├── pads/                  # The new Python library
│   ├── __init__.py        # Public API (lazy-imports torch)
│   ├── features.py        # 3-channel mel-spectrogram tensor builder
│   ├── signal_utils.py    # Vectorised framing / FFT / mel filterbank
│   ├── normalize.py       # Lightning + legacy normalisation
│   ├── models.py          # SelfConv + self-attention + BiGRU
│   ├── inference.py       # End-to-end PADExtractor + PADResult
│   ├── emotions.py        # Mehrabian PAD octant mapping
│   └── visualization.py   # Plotly helpers
├── web/
│   ├── app.py             # Streamlit web interface
│   └── gradio_app.py      # Gradio upload/recording interface
├── docs/
│   └── index.html         # GitHub Pages landing page
├── examples/
│   ├── quickstart.py      # Process one file
│   └── batch_extract.py   # Process a whole folder
├── tests/
│   └── test_pipeline.py   # 10 pytest checks (run without checkpoints)
├── checkpoints/           # Drop the trained .ckpt files here
├── data_norm/             # param_c{1,2,3}.json (legacy normalisation)
├── Assets/
├── pyproject.toml         # Package metadata
├── requirements.txt
└── LICENSE                # Apache 2.0
```

---

## Library API

The library is split into layers so you only pay for what you use. Importing `pads` does *not* import PyTorch — feature extraction works on a torch-less machine.

### Low-level feature extraction

```python
import librosa
from pads.features import Spectrogram2DTensor

wav, sr = librosa.load("speech.wav", sr=16000, mono=True)
builder = Spectrogram2DTensor(wav, sample_rate=sr)

spec_3ch = builder.get_2d_spectrograms()     # (3, T, 64) log-mel @ 16/25/40 ms windows
clips    = builder.get_2d_tensor_clips()     # (N, 3, 100, 64) 1-s overlapping clips
```

### Normalisation

```python
from pads.normalize import normalize_batch_per_channel, normalize_tensor_minmax, load_norm_params

# Lightning-style: per-clip min/max per channel
clips = normalize_batch_per_channel(clips)

# Legacy-style: pre-fit min/max from data_norm/*.json
params = load_norm_params("data_norm/")
clips = normalize_tensor_minmax(clips, params)
```

### Model

```python
from pads.models import load_lightning_model

model = load_lightning_model("checkpoints/one_sec/arousal_checkpoint.ckpt", device="cpu")
```

### High-level extractor (most common)

```python
from pads import PADExtractor

ext = PADExtractor(
    checkpoints_dir="checkpoints/one_sec",
    device="cpu",          # or "cuda"
    legacy=False,          # set True for the .ckp checkpoints
)

result = ext.extract("speech.wav")
result.arousal_mean, result.valence_mean, result.dominance_mean
result.dominant_emotion().name
result.to_dataframe()
```

You can choose which PAD dimensions to extract and which output representation you need:

```python
# Extract only one or two dimensions.
arousal_only = ext.extract("speech.wav", dims=("arousal",))
valence_dominance = ext.extract("speech.wav", dims=("valence", "dominance"))

# Return posterior probabilities per 1-second clip.
posteriors = ext.extract("speech.wav", dims=("arousal", "valence", "dominance"))
posterior_table = posteriors.to_dataframe()

# Return embeddings from the model before the final classifier.
with_embeddings = ext.extract(
    "speech.wav",
    dims=("arousal", "valence", "dominance"),
    return_embeddings=True,
)
arousal_embeddings = with_embeddings.embeddings["arousal"]

# Convert PAD posteriors to emotion labels.
dominant = posteriors.dominant_emotion()
distribution = posteriors.emotion_distribution()
```

### PAD -> emotion mapping

```python
from pads.emotions import pad_to_emotion, EMOTION_QUADRANTS

q = pad_to_emotion(valence=0.7, arousal=0.8, dominance=0.4)
print(q.name)        # "Dependent"
```

The eight octants follow the standard Mehrabian PAD coding (`+` = high, `-` = low):

| Octant | Valence | Arousal | Dominance |
|---|:---:|:---:|:---:|
| Exuberant   | + | + | + |
| Dependent   | + | + | - |
| Relaxed     | + | - | + |
| Docile      | + | - | - |
| Hostile     | - | + | + |
| Anxious     | - | + | - |
| Disdainful  | - | - | + |
| Bored       | - | - | - |

---

## Web interfaces

PADS includes two runnable interfaces plus a static GitHub Pages landing page.

### Gradio

The Gradio app is the simplest option for users who want to upload or record audio in the browser:

```bash
pip install -e ".[gradio]"
pads-gradio
```

It lets users choose PAD dimensions, posterior probabilities, embeddings, and PAD-to-emotion conversion outputs. The root `app.py` is also included for Hugging Face Spaces.

### Streamlit

The Streamlit app provides a richer dashboard:

* upload an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`), record from the browser where supported, or use a built-in demo;
* select which PAD dimensions to extract: arousal, valence/pleasure, dominance, or any subset;
* choose posterior probabilities, embeddings, and PAD-to-emotion conversion outputs;
* inspect the waveform, three multi-resolution log-mel spectrograms, PAD timelines, mean posterior radar chart, dominant emotion, and time-spent-per-emotion chart;
* download per-clip posterior probabilities as CSV and embeddings as NPZ.

```bash
pip install -e ".[web]"
streamlit run web/app.py
```

Compared to the original Qt dashboard, the web version:

* runs anywhere with a browser (no Qt install);
* uses interactive Plotly charts;
* surfaces a "demo audio" button so visitors without a WAV file can still try the pipeline.

### GitHub Pages

The `docs/` folder contains a static landing page for GitHub Pages:

```text
https://pauprezt.github.io/PADS/
```

GitHub Pages is static, so it cannot run the PyTorch model directly. It points users to the pip-installable library and the Gradio/Streamlit interfaces.

To enable it in GitHub, use **Settings -> Pages -> Build and deployment -> Deploy from a branch**, then choose:

```text
Branch: main
Folder: /docs
```

---

## Optimisations

All changes are behaviour-preserving — outputs match the original to floating-point precision.

| Pass | Original | New | Notes |
|---|---|---|---|
| Framing | Python list-comprehension + `np.vstack` | `numpy.lib.stride_tricks.sliding_window_view` | strided view, no per-frame copy |
| Mel filterbank | Rebuilt every call, indexed via Python loop | `@lru_cache` + vectorised triangle builder | shared across the 3 window sizes |
| Per-spectrum normalisation | Python loop over time bins | Single broadcasted `min`/`max` | ~30-60x faster |
| Min/max normalisation | Re-read JSON, per-channel loop | Vectorised batch op | works on `(N,3,T,F)` directly |
| Inference batching | Manual `len(specs) // 5` slicing | `inference_mode()` + `batch_size` arg | proper batches, no Python overhead |
| Checkpoint loading | Re-loaded on every call | `@lru_cache` | one disk hit per model |
| Float dtype | Mixed `float64`/`float32` | `float32` throughout | half the memory |
| PyTorch import | Eager (`pads` always imports torch) | Lazy (`__getattr__`) | feature extraction works without torch |

### Benchmark (synthetic 16 kHz audio, CPU, median of 3 runs)

| Audio length | Original `Spectrum_2D_Tensors` | Optimised `Spectrogram2DTensor` | Speedup |
|---|---:|---:|---:|
| 5 s   | 35.0 ms  | 15.1 ms  | **2.32x** |
| 15 s  | 104.2 ms | 43.9 ms  | **2.37x** |
| 30 s  | 264.5 ms | 112.0 ms | **2.36x** |
| 60 s  | 450.0 ms | 216.5 ms | **2.08x** |

Mean absolute difference between old and new outputs: `1.1e-7` (i.e. identical within float32 noise).

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Ten smoke tests cover signal processing, framing, filterbank caching, 3-channel tensor extraction (shape *and* clip count), batched normalisation, the `SelfConv` forward pass (with random init - no checkpoint needed), emotion mapping, and partial-dimension results.

All ten pass in `<0.3 s` on the synthetic input.

---


## Roadmap

* Stream / live-microphone inference in the web UI (currently file-only).
* Pre-fit ONNX export for browser-only inference.
* Pre-built Hugging Face Space hosting the Streamlit app.
* Multilingual support — the published checkpoints were trained on English; cross-lingual evaluation TBD.

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@book{perez_toro_2025_acoustic_linguistic,
  author    = {Pérez-Toro, P. A.},
  title     = {Acoustic and Linguistic Analysis in Neurological and Psychiatric Disorders},
  publisher = {Logos Verlag Berlin GmbH},
  year      = {2025}
}
```

## License

[Apache 2.0](LICENSE) - same as the upstream project.

## Acknowledgements

This rebuild preserves the science and the trained checkpoints of the original PADS by Paula A. Pérez-Toro at Friedrich-Alexander-Universität Erlangen-Nürnberg. The contribution here is purely engineering: a faster pipeline, a cleaner API surface, a web UI, and proper packaging.

