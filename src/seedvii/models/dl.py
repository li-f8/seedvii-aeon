"""PyTorch models for Phase 2 (no-leak deep learning baseline).

Input layout: (B, 310, T) — DE-flat features, 310 = 62 channels × 5 bands,
T = fixed length after resample (e.g. 90 for the SEED-VII LOVO pipeline).

All models output logits over 7 emotion classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DECNN(nn.Module):
    """Compact 1D CNN for DE-flat clip features (310 ch × T timepoints).

    Design notes
    ------------
    Only 1200 training cases in LOVO → must keep params small.
    - Projection 310 → 32 with 1×1 conv first (reduces redundancy across
      bands/channels before temporal modelling).
    - Two small temporal blocks, heavy dropout, label smoothing in the loss.
    """

    def __init__(self, in_ch: int = 310, n_classes: int = 7,
                 width: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        # GroupNorm instead of BatchNorm: no running stats → no train/eval
        # drift on MPS (previous BN version collapsed to majority class mid-
        # training on some folds). 8 groups works with width=32, width*2=64.
        g1 = min(8, width)
        g2 = min(8, width * 2)
        self.features = nn.Sequential(
            # 1×1 conv: spatial-spectral projection
            nn.Conv1d(in_ch, width, kernel_size=1),
            nn.GroupNorm(g1, width), nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.GroupNorm(g2, width * 2), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.GroupNorm(g2, width * 2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class DEEEGNet(nn.Module):
    """EEGNet-style model operating on DE features reshaped as (5, 62, T).

    Respects the physical structure of SEED-VII DE:
      - 5 frequency bands  (delta, theta, alpha, beta, gamma)
      - 62 EEG electrodes
      - T band-power timepoints (~4 s each after LDS)

    The Conv2d sequence mimics EEGNet's recipe:
      1. Temporal conv (within each band × electrode)
      2. Depthwise conv across electrodes (spatial filtering per band)
      3. Separable conv (temporal + pointwise) + pooling
      4. GAP + small linear head

    Expected input to forward: (B, 310, T) — we reshape to (B, 5, 62, T) here.
    """

    def __init__(self, n_bands: int = 5, n_electrodes: int = 62,
                 n_classes: int = 7, F1: int = 8, D: int = 2, F2: int = 16,
                 kern_t1: int = 7, kern_t2: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes

        # Using GroupNorm instead of BatchNorm2d: previous BN version produced
        # NaN on MPS (thin spatial dims after depthwise conv, 1×T/2 feature
        # maps). GroupNorm normalises per-sample → no running stats, no MPS
        # reduction-order sensitivity.
        g1 = max(1, min(4, F1))          # F1=8   → 4 groups
        g2 = max(1, min(4, F1 * D))      # F1*D=16 → 4 groups
        gF2 = max(1, min(4, F2))         # F2=16  → 4 groups

        # Block 1: temporal conv on each (band, electrode) row
        self.temporal = nn.Sequential(
            nn.Conv2d(n_bands, F1, kernel_size=(1, kern_t1),
                      padding=(0, kern_t1 // 2), bias=False),
            nn.GroupNorm(g1, F1),
        )
        # Block 2: depthwise spatial conv across electrodes per filter
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_electrodes, 1),
                      groups=F1, bias=False),
            nn.GroupNorm(g2, F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        # Block 3: separable temporal conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, kern_t2),
                      padding=(0, kern_t2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.GroupNorm(gF2, F2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(F2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 310, T)  →  (B, 5, 62, T)
        B, C, T = x.shape
        assert C == self.n_bands * self.n_electrodes, (
            f"expected {self.n_bands*self.n_electrodes} channels, got {C}"
        )
        # DE-flat layout from to_ts.py: f.transpose(2, 1, 0).reshape(62*5, -1)
        #   → first axis = electrode (outer), band (inner)
        # So view as (B, 62, 5, T) then permute bands to front
        x = x.view(B, self.n_electrodes, self.n_bands, T).permute(0, 2, 1, 3)
        # x now: (B, 5, 62, T)
        h = self.temporal(x)
        h = self.depthwise(h)
        h = self.separable(h)
        h = h.flatten(1)
        return self.head(h)


def build_dl(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"decnn", "cnn"}:
        return DECNN(**kwargs)
    if name in {"eegnet", "de_eegnet"}:
        # Strip args meant for DECNN
        kwargs.pop("in_ch", None)
        return DEEEGNet(**kwargs)
    raise ValueError(f"unknown dl model '{name}'")
