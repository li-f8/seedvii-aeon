"""SEED-VII data loading.

Loads DE-LDS EEG features and 33-dim eye features from the SEED-VII .mat files.
Per-subject z-score normalisation is applied.

Directory layout expected (symlinked under ./data/seed-VII):
    seed-VII/
        EEG_features/{sub}.mat     # keys 'de_LDS_{vid}' for vid in 1..80
        EYE_features/{sub}.mat     # keys '{vid}' for vid in 1..80
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio

EMOTIONS = ["Disgust", "Fear", "Sad", "Neutral", "Happy", "Anger", "Surprise"]
_EMOTION_MAP = {name: i for i, name in enumerate(EMOTIONS)}

# 80-video emotion order per SEED-VII stimulation schedule
# (4 sessions × 20 videos; each session repeats its 10-clip block twice).
_VIDEO_EMOTIONS: list[str] = (
    # Session 1 (videos 1-20)
    ["Happy", "Neutral", "Disgust", "Sad", "Anger",
     "Anger", "Sad", "Disgust", "Neutral", "Happy"] * 2
    # Session 2 (videos 21-40)
    + ["Anger", "Sad", "Fear", "Neutral", "Surprise",
       "Surprise", "Neutral", "Fear", "Sad", "Anger"] * 2
    # Session 3 (videos 41-60)
    + ["Happy", "Surprise", "Disgust", "Fear", "Anger",
       "Anger", "Fear", "Disgust", "Surprise", "Happy"] * 2
    # Session 4 (videos 61-80)
    + ["Disgust", "Sad", "Fear", "Surprise", "Happy",
       "Happy", "Surprise", "Fear", "Sad", "Disgust"] * 2
)
VIDEO_LABELS: list[int] = [_EMOTION_MAP[e] for e in _VIDEO_EMOTIONS]
assert len(VIDEO_LABELS) == 80, f"Expected 80 video labels, got {len(VIDEO_LABELS)}"


def _default_data_root() -> Path:
    """Project-root-relative default: <repo>/data/seed-VII."""
    return Path(__file__).resolve().parents[3] / "data" / "seed-VII"


def load_eeg_de_features(
    subject: int,
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DE-LDS features for a single subject.

    Returns
    -------
    X : (N, 5, 62) float32    band-first layout (matches SEED-VII convention)
    y : (N,) int64            emotion labels 0..6
    video_ids : (N,) int64    video index 1..80 per window
    """
    root = Path(data_root) if data_root else _default_data_root()
    mat = sio.loadmat(root / "EEG_features" / f"{subject}.mat")

    feats, labels, vids = [], [], []
    for vid in range(1, 81):
        key = f"de_LDS_{vid}"
        if key not in mat:
            continue
        f = mat[key]  # (T, 5, 62) per SEED-VII convention
        T = f.shape[0]
        feats.append(f)
        labels.append(np.full(T, VIDEO_LABELS[vid - 1], dtype=np.int64))
        vids.append(np.full(T, vid, dtype=np.int64))

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0)
    v = np.concatenate(vids, axis=0)
    return X, y, v


def load_de_sequence_windows(
    subject: int,
    win_sec: int = 10,
    stride_sec: int | None = None,
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over the DE time axis *within each video*.

    SEED-VII's ``de_LDS_{vid}`` has shape ``(T_seconds, 5_bands, 62_channels)``.
    Each second is one DE estimate per band per channel. We slide a
    ``win_sec``-length window along the second axis to produce fixed-length
    time series suitable for aeon TSC methods.

    Output layout: ``(N, 62*5, win_sec)``  — each channel × band combination
    becomes an aeon "channel", and ``win_sec`` is the aeon "time" axis.
    This keeps 310 rich spectrotemporal channels and a window long enough
    (≥ 9 samples) for Hydra / MultiRocket kernels.

    Parameters
    ----------
    subject : 1..20
    win_sec : number of consecutive seconds per case (default 10)
    stride_sec : hop in seconds (default = win_sec, i.e. non-overlapping)

    Returns
    -------
    X : (N, 310, win_sec) float32
    y : (N,) int64
    video_ids : (N,) int64
    """
    root = Path(data_root) if data_root else _default_data_root()
    mat = sio.loadmat(root / "EEG_features" / f"{subject}.mat")

    stride = stride_sec or win_sec
    Xs, ys, vs = [], [], []
    for vid in range(1, 81):
        key = f"de_LDS_{vid}"
        if key not in mat:
            continue
        f = mat[key]  # (T, 5, 62)
        T = f.shape[0]
        # merge band × channel → 310 feature channels, move time to axis=-1
        # resulting per-second vector shape: (310,)
        f2 = f.transpose(0, 2, 1).reshape(T, -1)  # (T, 310)
        f2 = f2.T  # (310, T)
        n_win = (T - win_sec) // stride + 1
        if n_win <= 0:
            continue
        windows = np.stack(
            [f2[:, i * stride : i * stride + win_sec] for i in range(n_win)],
            axis=0,
        ).astype(np.float32)  # (n_win, 310, win_sec)
        Xs.append(windows)
        ys.append(np.full(n_win, VIDEO_LABELS[vid - 1], dtype=np.int64))
        vs.append(np.full(n_win, vid, dtype=np.int64))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    v = np.concatenate(vs, axis=0)
    return X, y, v


def load_de_sequence_multimodal(
    subject_ids: Iterable[int] = range(1, 21),
    win_sec: int = 10,
    stride_sec: int | None = None,
    data_root: Path | str | None = None,
    normalise: bool = True,
) -> dict:
    """DE-sequence loader for multiple subjects.

    Returns
    -------
    dict with keys X_eeg (N, 310, win_sec), y, subjects, video_ids.
    """
    Xs, ys, subs, vids = [], [], [], []
    for sub in subject_ids:
        X, y, v = load_de_sequence_windows(sub, win_sec, stride_sec, data_root)
        Xs.append(X)
        ys.append(y)
        subs.append(np.full(X.shape[0], sub, dtype=np.int64))
        vids.append(v)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(subs, axis=0)
    video_ids = np.concatenate(vids, axis=0)

    if normalise:
        for sub in np.unique(subjects):
            mask = subjects == sub
            m = X[mask].mean(axis=(0, 2), keepdims=True)
            s = X[mask].std(axis=(0, 2), keepdims=True) + 1e-8
            X[mask] = (X[mask] - m) / s

    return {"X_eeg": X, "y": y, "subjects": subjects, "video_ids": video_ids}


def load_raw_eeg_windows(
    subject: int,
    win_sec: float = 4.0,
    fs: int = 200,
    stride_sec: float | None = None,
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed raw EEG and slice into fixed-length windows.

    The SEED-VII ``EEG_preprocessed/{sub}.mat`` file contains 80 keys named
    ``"1"`` .. ``"80"`` (one per video), each a ``(62, T)`` float64 array
    sampled at 200 Hz. We slide a non-overlapping ``win_sec`` window over
    each video and label every window with the video's emotion.

    Parameters
    ----------
    subject : 1..20
    win_sec : window length in seconds (default 4 s → 800 samples)
    fs : sampling rate (default 200 Hz — matches SEED-VII preprocessing)
    stride_sec : hop in seconds (default = win_sec, i.e. non-overlapping)

    Returns
    -------
    X : (N, 62, win_len) float32
    y : (N,) int64            emotion label 0..6
    video_ids : (N,) int64    video index 1..80 per window
    """
    root = Path(data_root) if data_root else _default_data_root()
    mat = sio.loadmat(root / "EEG_preprocessed" / f"{subject}.mat")

    win_len = int(round(win_sec * fs))
    stride = int(round((stride_sec or win_sec) * fs))

    Xs, ys, vs = [], [], []
    for vid in range(1, 81):
        key = str(vid)
        if key not in mat:
            continue
        sig = mat[key]  # (62, T)
        if sig.ndim != 2 or sig.shape[0] != 62:
            raise ValueError(
                f"Subject {subject} video {vid}: unexpected shape {sig.shape}"
            )
        T = sig.shape[1]
        n_win = (T - win_len) // stride + 1
        if n_win <= 0:
            continue
        # slice: (n_win, 62, win_len)
        windows = np.stack(
            [sig[:, i * stride : i * stride + win_len] for i in range(n_win)],
            axis=0,
        ).astype(np.float32)
        Xs.append(windows)
        ys.append(np.full(n_win, VIDEO_LABELS[vid - 1], dtype=np.int64))
        vs.append(np.full(n_win, vid, dtype=np.int64))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    v = np.concatenate(vs, axis=0)
    return X, y, v


def load_raw_multimodal(
    subject_ids: Iterable[int] = range(1, 21),
    win_sec: float = 4.0,
    fs: int = 200,
    stride_sec: float | None = None,
    data_root: Path | str | None = None,
    normalise: bool = True,
) -> dict:
    """Load raw EEG windows for multiple subjects (no eye, for now).

    Returns
    -------
    dict with keys
        X_eeg     (N, 62, win_len) float32
        y         (N,)
        subjects  (N,)
        video_ids (N,)
    """
    Xs, ys, subs, vids = [], [], [], []
    for sub in subject_ids:
        X, y, v = load_raw_eeg_windows(sub, win_sec, fs, stride_sec, data_root)
        Xs.append(X)
        ys.append(y)
        subs.append(np.full(X.shape[0], sub, dtype=np.int64))
        vids.append(v)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(subs, axis=0)
    video_ids = np.concatenate(vids, axis=0)

    if normalise:
        # per-subject, per-channel z-score across all windows+timepoints
        for sub in np.unique(subjects):
            mask = subjects == sub
            # compute stats along (N, T) for each channel
            m = X[mask].mean(axis=(0, 2), keepdims=True)  # (1, 62, 1)
            s = X[mask].std(axis=(0, 2), keepdims=True) + 1e-8
            X[mask] = (X[mask] - m) / s

    return {"X_eeg": X, "y": y, "subjects": subjects, "video_ids": video_ids}


def load_eye_features(
    subject: int,
    data_root: Path | str | None = None,
) -> np.ndarray:
    """Load 33-dim eye movement features for one subject (concatenated across videos)."""
    root = Path(data_root) if data_root else _default_data_root()
    mat = sio.loadmat(root / "EYE_features" / f"{subject}.mat")

    feats = []
    for vid in range(1, 81):
        key = str(vid)
        if key not in mat:
            continue
        feats.append(mat[key])
    return np.concatenate(feats, axis=0).astype(np.float32)


def _zscore_per_subject(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per-subject z-score. Works on any trailing shape; normalises along axis-0 slices."""
    out = np.zeros_like(X)
    for sub in np.unique(subjects):
        mask = subjects == sub
        flat = X[mask].reshape(mask.sum(), -1)
        m, s = flat.mean(0), flat.std(0) + 1e-8
        out[mask] = ((flat - m) / s).reshape(X[mask].shape)
    return out.astype(np.float32)


def load_multimodal(
    subject_ids: Iterable[int] = range(1, 21),
    data_root: Path | str | None = None,
    normalise: bool = True,
) -> dict:
    """Load EEG + Eye + labels for a list of subjects.

    Returns a dict with keys:
        X_eeg     (N, 5, 62)   band-first EEG DE features
        X_eye     (N, 33)      eye movement features
        y         (N,)         emotion label 0..6
        subjects  (N,)         subject id 1..20
        video_ids (N,)         video index 1..80
    """
    Xe_all, Xy_all, y_all, sub_all, vid_all = [], [], [], [], []
    for sub in subject_ids:
        Xe, y, vids = load_eeg_de_features(sub, data_root)
        Xy = load_eye_features(sub, data_root)
        if Xy.shape[0] != Xe.shape[0]:
            raise ValueError(
                f"Subject {sub}: EEG N={Xe.shape[0]} but Eye N={Xy.shape[0]}"
            )
        Xe_all.append(Xe)
        Xy_all.append(Xy)
        y_all.append(y)
        vid_all.append(vids)
        sub_all.append(np.full(Xe.shape[0], sub, dtype=np.int64))

    X_eeg = np.concatenate(Xe_all, axis=0)
    X_eye = np.concatenate(Xy_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(sub_all, axis=0)
    video_ids = np.concatenate(vid_all, axis=0)

    if normalise:
        X_eeg = _zscore_per_subject(X_eeg, subjects)
        X_eye = _zscore_per_subject(X_eye, subjects)

    return {
        "X_eeg": X_eeg,
        "X_eye": X_eye,
        "y": y,
        "subjects": subjects,
        "video_ids": video_ids,
    }

