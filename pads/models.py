"""
PyTorch model definitions for PADS.

The architecture is a small CNN with a self-attention block followed by a
bidirectional GRU and two linear heads (embedding + classifier). It is
unchanged from the original code - we keep the same parameter names and
layer order so the published checkpoints continue to load.

Two factory helpers are provided:

* :func:`load_lightning_model` - load a ``.ckpt`` file produced by the
  PyTorch Lightning trainer (input shape 100x64).
* :func:`load_legacy_model`    - load a ``.ckp`` file produced by the old
  training script (input shape 50x64).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Self-attention block
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Spatial self-attention over a 2-D feature map (B, C, W, H)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, w, h = x.size()
        q = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, w * h)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(b, -1, w * h)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, w, h)
        # The original code uses ``gamma * out * x`` (multiplicative gating);
        # we keep it identical so checkpoints transfer.
        out = self.gamma * out * x
        return out, attn


# -----------------------------------------------------------------------------
# SelfConv classifier
# -----------------------------------------------------------------------------

class SelfConv(nn.Module):
    """CNN + self-attention + BiGRU classifier used by PADS."""

    def __init__(
        self,
        nc: int = 3,
        input_shape: Tuple[int, int, int, int] = (1, 3, 100, 64),
        dim: int = 256,
        n_classes: int = 2,
        hidden_size: int = 128,
        bidirectional: bool = True,
        nlayers_bigru: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.nlayers_bigru = nlayers_bigru
        self.n_classes = n_classes

        # CNN: 1 conv layer + max-pool + batchnorm.
        self.conv1 = nn.Conv2d(nc, 8, kernel_size=(1, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.attention_cnn = SelfAttention(8)

        # Derive GRU input size from a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            n_freq, n_time = self._probe_conv_shape(dummy)
        gru_input = n_freq * 8

        # Time dimension after pooling (== n_time of the conv output) is kept
        # at 100 to match the published Lightning checkpoints' linear layer
        # input size. For 50-frame legacy checkpoints, callers pass dim=2 via
        # input_shape and the linear layer uses num_feats below.
        self.num_feats = n_time
        self.GRU = nn.GRU(
            input_size=gru_input,
            hidden_size=hidden_size,
            num_layers=nlayers_bigru,
            bidirectional=bidirectional,
            batch_first=True,
        )

        idx_bi = 2 if bidirectional else 1
        output_gru = hidden_size * idx_bi * self.num_feats

        self.BatchNorm_Gru = nn.BatchNorm1d(self.num_feats)
        self.linear = nn.Linear(output_gru, dim)
        self.linear_class = nn.Linear(dim, n_classes)

    # ------------------------------------------------------------------
    def _probe_conv_shape(self, x: torch.Tensor) -> Tuple[int, int]:
        out = self._forward_conv(x)
        # out shape: (B, 8, T, F') where T is unchanged time dim, F' is freq after pool.
        return out.size(3), out.size(2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.pool1(self.conv1(x))))
        x, _ = self.attention_cnn(x)
        return x

    def _forward_gru(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, T, F')
        out = x.permute(0, 2, 1, 3)            # (B, T, 8, F')
        out = out.contiguous().view(out.shape[0], out.shape[1], -1)
        out, _ = self.GRU(out)
        out = self.BatchNorm_Gru(out)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_conv(x)
        x = self._forward_gru(x)
        x = F.gelu(self.linear(x.view(x.size(0), -1)))
        emb = x
        logits = self.linear_class(x)
        return logits, emb


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def _strip_lightning_prefix(state_dict: dict) -> dict:
    """Lightning sometimes stores keys prefixed with ``model.`` - strip them."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new[k[len("model."):]] = v
        else:
            new[k] = v
    return new


@lru_cache(maxsize=8)
def _load_state_dict(checkpoint_path: str, map_location: str = "cpu") -> dict:
    """Cached state-dict loader - same file is read at most once."""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return _strip_lightning_prefix(sd)


def load_lightning_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    input_shape: Tuple[int, int, int, int] = (1, 3, 100, 64),
) -> SelfConv:
    """Load a PADS model from a Lightning ``.ckpt`` file.

    Parameters
    ----------
    checkpoint_path : path-like
        Path to ``arousal_checkpoint.ckpt`` / ``valence_checkpoint.ckpt`` /
        ``dominance_checkpoint.ckpt``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    input_shape : 4-tuple
        Used to derive layer sizes. Default is the published 1 s clip shape.
    """
    model = SelfConv(nc=input_shape[1], input_shape=input_shape)
    sd = _load_state_dict(str(checkpoint_path), map_location=device)
    # Drop keys that don't exist in our cleaner model (e.g. extra metrics).
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if len(filtered) == 0:
        raise RuntimeError(
            f"No matching keys found in {checkpoint_path!r}; checkpoint may be incompatible."
        )
    model = model.to(device).eval()
    return model


def load_legacy_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    input_shape: Tuple[int, int, int, int] = (1, 3, 50, 64),
) -> SelfConv:
    """Load a PADS model from a legacy ``.ckp`` file (pre-Lightning)."""
    return load_lightning_model(checkpoint_path, device=device, input_shape=input_shape)
