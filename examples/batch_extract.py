"""
Batch extraction example.

Process every audio file in a folder and save the per-clip PAD posteriors
to a single CSV.

Usage:
    python examples/batch_extract.py path/to/audio_dir output.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pads import PADExtractor


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main(audio_dir: str, csv_path: str):
    audio_dir = Path(audio_dir)
    extractor = PADExtractor(checkpoints_dir="checkpoints/one_sec", device="cpu")

    rows = []
    files = [f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise SystemExit(f"No audio files found in {audio_dir}")

    for f in files:
        print(f"Processing {f.name} ...")
        try:
            result = extractor.extract(str(f))
        except Exception as e:
            print(f"  skipped: {e}")
            continue
        df = result.to_dataframe()
        df.insert(0, "file", f.name)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(csv_path, index=False)
    print(f"\nSaved {len(out)} rows to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
