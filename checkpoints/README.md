# Checkpoints

Drop the pre-trained model files in this folder.

## Lightning checkpoints (default)

Place these in `checkpoints/one_sec/`:

* `arousal_checkpoint.ckpt`
* `valence_checkpoint.ckpt`
* `dominance_checkpoint.ckpt`

Used with `PADExtractor(checkpoints_dir="checkpoints/one_sec", legacy=False)`.

Input shape: `(N, 3, 100, 64)` — 1-second clips at 10 ms hop on a 64-band mel filterbank.

## Legacy checkpoints

Place these directly in `checkpoints/`:

* `arousal_checkpoint.ckp`
* `valence_checkpoint.ckp`
* `dominance_checkpoint.ckp`

Used with `PADExtractor(checkpoints_dir="checkpoints", data_norm_dir="data_norm", legacy=True)`.

Input shape: `(N, 3, 50, 64)` — 0.5-second clips. Requires `data_norm/param_c{1,2,3}.json`.

## Where to get them

Both sets are published as part of the upstream PADS repository:
<https://github.com/PauPerezT/PADS>. Clone it and copy the `checkpoints/`
directory in here.
