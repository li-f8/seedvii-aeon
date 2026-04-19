"""Multimodal fusion stubs (EEG + Eye).

Placeholder for future work: cross-modal attention, late-fusion ensembles, etc.
For the initial benchmark we use sklearn-style early fusion (concatenate
features) — see scripts/run_single.py.
"""
from __future__ import annotations

import numpy as np


def early_fusion_concat(X_eeg_aeon: np.ndarray, X_eye_aeon: np.ndarray) -> np.ndarray:
    """Concat along the channel axis.

    Parameters
    ----------
    X_eeg_aeon : (N, 62, 5)
    X_eye_aeon : (N, 33, 5)   must be time-aligned (e.g. broadcast 33-dim static across 5)

    Returns
    -------
    (N, 95, 5)
    """
    if X_eeg_aeon.shape[0] != X_eye_aeon.shape[0]:
        raise ValueError("case count mismatch between EEG and Eye")
    if X_eeg_aeon.shape[2] != X_eye_aeon.shape[2]:
        raise ValueError("time length mismatch — broadcast eye features first")
    return np.concatenate([X_eeg_aeon, X_eye_aeon], axis=1)


def broadcast_static_to_series(X_static: np.ndarray, n_timepoints: int) -> np.ndarray:
    """Turn (N, D) static features into (N, D, T) by repeating along time."""
    return np.broadcast_to(X_static[:, :, None], (*X_static.shape, n_timepoints)).copy()
