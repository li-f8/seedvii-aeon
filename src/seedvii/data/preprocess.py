"""Preprocessing for raw EEG (not used by the DE-feature pipeline).

Stubs for the future raw-signal TSC pipeline: bandpass, downsample, windowing.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Integer-factor downsampling along the last axis (naive decimation)."""
    if factor <= 1:
        return x
    return np.ascontiguousarray(x[..., ::factor])


def window(x: np.ndarray, win_len: int, stride: int | None = None) -> np.ndarray:
    """Slice a long recording into fixed-length windows.

    Parameters
    ----------
    x : (n_channels, n_timepoints)
    win_len : length of each window in samples
    stride : hop length; defaults to win_len (non-overlapping)

    Returns
    -------
    windows : (n_windows, n_channels, win_len)
    """
    stride = stride or win_len
    n_ch, T = x.shape
    n_win = (T - win_len) // stride + 1
    if n_win <= 0:
        return np.empty((0, n_ch, win_len), dtype=x.dtype)
    out = np.stack(
        [x[:, i * stride : i * stride + win_len] for i in range(n_win)],
        axis=0,
    )
    return out
