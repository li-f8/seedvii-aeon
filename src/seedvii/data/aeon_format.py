"""Convert SEED-VII arrays into aeon's 3D multivariate TSC format.

aeon convention:  X shape  (n_cases, n_channels, n_timepoints)
                  y shape  (n_cases,)
"""
from __future__ import annotations

import numpy as np


def de_to_aeon(X_eeg_band_first: np.ndarray) -> np.ndarray:
    """DE-LDS features → aeon 3D.

    SEED-VII raw layout:   (N, 5_bands, 62_channels)  — only 5 "timepoints" per case
    aeon expected layout:  (N, n_channels, n_timepoints)
    We treat the 5 bands as the "time" axis and 62 electrodes as channels.

    Parameters
    ----------
    X_eeg_band_first : (N, 5, 62)

    Returns
    -------
    X_aeon : (N, 62, 5) float32
    """
    if X_eeg_band_first.ndim != 3 or X_eeg_band_first.shape[1] != 5:
        raise ValueError(f"Expected (N, 5, 62), got {X_eeg_band_first.shape}")
    return np.ascontiguousarray(
        X_eeg_band_first.transpose(0, 2, 1).astype(np.float32)
    )


def eye_to_aeon(X_eye: np.ndarray) -> np.ndarray:
    """Eye features (N, 33) → (N, 33, 1) pseudo-time-series for aeon."""
    if X_eye.ndim != 2:
        raise ValueError(f"Expected (N, eye_dim), got {X_eye.shape}")
    return np.ascontiguousarray(X_eye[:, :, None].astype(np.float32))
