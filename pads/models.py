"""
PyTorch model definitions for PADS.

Architecture matches the original code so the published checkpoints continue
to load. Provides helper factories for both Lightning .ckpt and legacy .ckp
checkpoints.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Spatial self-attention over a 2-D feature map (B, C, W, H)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.size()
        q = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, w * h)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(b, -1, w * h)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, w, h)
        out = self.gamma * out * x
        return out, attn


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

        self.conv1 = nn.Conv2d(nc, 8, kernel_size=(1, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.attention_cnn = SelfAttention(8)

        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            n_freq, n_time = self._probe_conv_shape(dummy)
        gru_input = n_freq * 8

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

    def _probe_conv_shape(self, x):
        out = self._forward_conv(x)
        return out.size(3), out.size(2)

    def _forward_conv(self, x):
        x = F.leaky_relu(self.bn1(self.pool1(self.conv1(x))))
        x, _ = self.attention_cnn(x)
        return x

    def _forward_gru(self, x):
        out = x.permute(0, 2, 1, 3)
        out = out.contiguous().view(out.shape[0], out.shape[1], -1)
        out, _ = self.GRU(out)
        out = self.BatchNorm_Gru(out)
        return out

    def forward(self, x):
        x = self._forward_conv(x)
        x = self._forward_gru(x)
        x = F.gelu(self.linear(x.view(x.size(0), -1)))
        emb = x
        logits = self.linear_class(x)
        return logits, emb


def _strip_lightning_prefix(state_dict: dict) -> dict:
    new = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new[k[len("model."):]] = v
        else:
            new[k] = v
    return new


@lru_cache(maxsize=8)
def _load_state_dict(checkpoint_path: str, map_location: str = "cpu") -> dict:
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return _strip_lightning_prefix(sd)


def load_lightning_model(checkpoint_path, device: str = "cpu",
                         input_shape: Tuple[int, int, int, int] = (1, 3, 100, 64)):
    """Load a PADS model from a Lightning .ckpt file."""
    model = SelfConv(nc=input_shape[1], input_shape=input_shape)
    sd = _load_state_dict(str(checkpoint_path), map_location=device)
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    if len(filtered) == 0:
        raise RuntimeError(
            f"No matching keys in {checkpoint_path!r}; checkpoint may be incompatible."
        )
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    return model


def load_legacy_model(checkpoint_path, device: str = "cpu",
                      input_shape: Tuple[int, int, int, int] = (1, 3, 50, 64)):
    """Load a PADS model from a legacy .ckp file."""
    return load_lightning_model(checkpoint_path, device=device, input_shape=input_shape)
