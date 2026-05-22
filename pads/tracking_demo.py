"""Gradio emotion-tracking demo for PADS.

Lets the user compare multiple recordings and see how PAD / emotion
estimates evolve over time within a recording.
"""

from __future__ import annotations

from pathlib import Path

from .inference import PADExtractor


DIMENSION_LABELS = {
    "Arousal": "arousal",
    "Valence / pleasure": "valence",
    "Dominance": "dominance",
}


def track_emotions(audio_path, checkpoints_dir, device):
    """Run PADExtractor.extract on one file and format the timeline + dominant emotion."""
    if not audio_path:
        return "Please upload or record audio first.", None

    ckpt = Path(checkpoints_dir)
    if not ckpt.exists() or not any(ckpt.glob("*.ckpt")):
        return f"No .ckpt files in {ckpt}.", None

    try:
        extractor = PADExtractor(checkpoints_dir=ckpt, device=device)
        result = extractor.extract(audio_path)
    except Exception as exc:
        return f"Inference failed: {exc}", None

    df = result.to_dataframe()
    if result.has_all_dimensions():
        dom = result.dominant_emotion()
        summary = f"Dominant emotion: {dom.name} {dom.emoji} ({result.summary()})"
    else:
        summary = result.summary()
    return summary, df


def build_app():
    import gradio as gr

    def audio_component():
        try:
            return gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        except TypeError:
            return gr.Audio(source="upload", type="filepath", label="Audio")

    with gr.Blocks(title="PADS Emotion Tracking") as demo:
        gr.Markdown(
            """
            # PADS - Emotion tracking
            Upload or record speech audio and inspect how PAD posteriors and
            the dominant emotion change second-by-second across the recording.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = audio_component()
                checkpoints_dir = gr.Textbox(
                    value="checkpoints/one_sec",
                    label="Checkpoints directory",
                )
                device = gr.Radio(choices=["cpu", "cuda"], value="cpu", label="Device")
                run_button = gr.Button("Track emotions", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Textbox(label="Summary", lines=2)
                table = gr.Dataframe(label="Per-clip posteriors")

        run_button.click(
            track_emotions,
            inputs=[audio, checkpoints_dir, device],
            outputs=[summary, table],
        )

    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
