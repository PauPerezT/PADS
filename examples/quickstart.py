"""
Minimal quickstart - extract PAD features from a single audio file.

Usage:
    python examples/quickstart.py path/to/audio.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

from pads import PADExtractor


def main(audio_path: str):
    # Build extractor. Use legacy=True for the older .ckp checkpoints.
    extractor = PADExtractor(
        checkpoints_dir="checkpoints/one_sec",
        device="cpu",
    )
    result = extractor.extract(audio_path)

    print(result.summary())
    print()
    print("Per-clip table:")
    print(result.to_dataframe().to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
