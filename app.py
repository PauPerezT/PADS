"""Root Gradio entry point for Hugging Face Spaces and simple local runs."""

from pads.gradio_app import build_app


demo = build_app()


if __name__ == "__main__":
    demo.launch()
