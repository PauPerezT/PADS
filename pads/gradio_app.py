"""Gradio interface for the installed PADS package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import PADExtractor


DIMENSION_LABELS = {
    "Arousal": "arousal",
    "Valence / pleasure": "valence",
    "Dominance": "dominance",
}


def _empty_outputs(message: str):
    return message, pd.DataFrame(), "Not available", None


def analyze_audio(
    audio_path: str | None,
    dimensions: list[str] | None,
    output_posteriors: bool,
    output_embeddings: bool,
    output_emotions: bool,
    checkpoints_dir: str,
    device: str,
    batch_size: int,
):
    """Run PADS on an uploaded audio file and return Gradio outputs."""
    if audio_path is None:
        return _empty_outputs("Please upload or record an audio file first.")

    dimensions = dimensions or []
    dims = tuple(DIMENSION_LABELS[label] for label in dimensions if label in DIMENSION_LABELS)
    if not dims:
        return _empty_outputs("Please select at least one PAD dimension.")

    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists() or not any(ckpt_path.glob("*.ckpt")):
        return _empty_outputs(
            f"No .ckpt checkpoints found in {ckpt_path}. "
            "Add arousal, valence, and/or dominance checkpoints before running inference."
        )

    try:
        extractor = PADExtractor(checkpoints_dir=ckpt_path, device=device)
        result = extractor.extract(
            audio_path,
            dims=dims,
            batch_size=int(batch_size),
            return_embeddings=output_embeddings,
        )
    except Exception as exc:
        return _empty_outputs(f"Inference failed: {exc}")

    posterior_table = result.to_dataframe() if output_posteriors else pd.DataFrame()

    emotion_text = "Not requested"
    if output_emotions:
        if result.has_all_dimensions():
            dominant = result.dominant_emotion()
            distribution = result.emotion_distribution()
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
            lines = [f"Dominant emotion: {dominant.name} {dominant.emoji}", ""]
            lines.extend(f"{name}: {share:.1%}" for name, share in ranked)
            emotion_text = "\n".join(lines)
        else:
            emotion_text = "PAD-to-emotion conversion requires arousal, valence, and dominance."

    embeddings_file = None
    if output_embeddings and result.embeddings:
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        np.savez(tmp.name, **result.embeddings)
        embeddings_file = tmp.name

    return result.summary(), posterior_table, emotion_text, embeddings_file


def build_app():
    import gradio as gr

    def audio_component():
        try:
            return gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        except TypeError:
            return gr.Audio(source="upload", type="filepath", label="Audio")

    with gr.Blocks(title="PADS Speech Analysis") as demo:
        gr.Markdown(
            """
            # PADS
            Upload or record speech audio and extract Pleasure/Valence, Arousal,
            and Dominance posteriors. You can also export model embeddings and
            convert the full PAD representation into emotion octants.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio = audio_component()
                dimensions = gr.CheckboxGroup(
                    choices=list(DIMENSION_LABELS.keys()),
                    value=list(DIMENSION_LABELS.keys()),
                    label="PAD dimensions",
                )
                output_posteriors = gr.Checkbox(value=True, label="Posterior probabilities")
                output_embeddings = gr.Checkbox(value=False, label="Embeddings")
                output_emotions = gr.Checkbox(value=True, label="PAD-to-emotion conversion")
                checkpoints_dir = gr.Textbox(value="checkpoints/one_sec", label="Checkpoints directory")
                device = gr.Radio(choices=["cpu", "cuda"], value="cpu", label="Device")
                batch_size = gr.Slider(1, 64, value=16, step=1, label="Batch size")
                run_button = gr.Button("Analyze audio", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Textbox(label="Summary", lines=2)
                posteriors = gr.Dataframe(label="Per-clip posteriors")
                emotions = gr.Textbox(label="Emotion conversion", lines=10)
                embeddings = gr.File(label="Download embeddings (.npz)")

        run_button.click(
            analyze_audio,
            inputs=[
                audio,
                dimensions,
                output_posteriors,
                output_embeddings,
                output_emotions,
                checkpoints_dir,
                device,
                batch_size,
            ],
            outputs=[summary, posteriors, emotions, embeddings],
        )

    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
