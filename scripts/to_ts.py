"""Export SEED-VII to aeon .ts format (one file per subject).

Per Prof. Bagnall's request: ``seedVIIsubject{N}.ts`` holds all 80 clips
for subject N, each case is one clip (variable length, 1-6 min),
labelled with its emotion class. Stratified train / test splits are
also written as ``seedVIIsubject{N}_TRAIN.ts`` / ``_TEST.ts``.

Usage
-----
    python scripts/to_ts.py --subjects 1 --out data/ts_raw
    python scripts/to_ts.py --subjects 1 2 3 --features raw
    python scripts/to_ts.py --subjects 1 --features de_flat
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedShuffleSplit

from aeon.datasets import save_to_ts_file

from seedvii.data.loader import (
    VIDEO_LABELS,
    _default_data_root,
)
from seedvii.utils import get_logger

log = get_logger()


def _centre_crop(sig: np.ndarray, want_len: int) -> np.ndarray:
    """Return the central ``want_len`` samples of ``sig`` (shape (C, T))."""
    T = sig.shape[-1]
    if T <= want_len:
        return sig  # too short — keep as-is
    start = (T - want_len) // 2
    return sig[..., start : start + want_len]


def load_subject_as_clips_raw(
    subject: int, data_root: Path, middle_sec: float | None = None,
    fs: int = 200,
) -> tuple[list, np.ndarray]:
    """Each case = one clip of raw EEG.

    If ``middle_sec`` is given, extract the central ``middle_sec`` seconds
    of each clip so all cases are equal length (``middle_sec * fs`` samples).
    """
    mat = sio.loadmat(data_root / "EEG_preprocessed" / f"{subject}.mat")
    want_len = int(round(middle_sec * fs)) if middle_sec else None
    X, y = [], []
    for vid in range(1, 81):
        if str(vid) not in mat:
            continue
        sig = mat[str(vid)].astype(np.float32)  # (62, T)
        if want_len is not None:
            sig = _centre_crop(sig, want_len)
        X.append(sig)
        y.append(VIDEO_LABELS[vid - 1])
    return X, np.asarray(y, dtype=np.int64)


def load_subject_as_clips_de_flat(
    subject: int, data_root: Path, middle_sec: float | None = None,
) -> tuple[list, np.ndarray]:
    """Each case = one clip of DE-LDS features.

    If ``middle_sec`` is given, extract the central ``middle_sec`` samples
    (DE is at 1 Hz, so 1 sample = 1 second) so all cases are equal length.
    """
    mat = sio.loadmat(data_root / "EEG_features" / f"{subject}.mat")
    want_len = int(round(middle_sec)) if middle_sec else None
    X, y = [], []
    for vid in range(1, 81):
        key = f"de_LDS_{vid}"
        if key not in mat:
            continue
        f = mat[key]  # (T, 5, 62)
        # reshape -> (310, T) so channels-first, time-last per aeon convention
        f2 = f.transpose(2, 1, 0).reshape(62 * 5, -1).astype(np.float32)
        if want_len is not None:
            f2 = _centre_crop(f2, want_len)
        X.append(f2)
        y.append(VIDEO_LABELS[vid - 1])
    return X, np.asarray(y, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="subject IDs 1..20")
    p.add_argument("--features", choices=["raw", "de_flat"], default="raw",
                   help="raw = preprocessed EEG @200Hz (62 ch); "
                        "de_flat = DE-LDS (310 ch, 1Hz)")
    p.add_argument("--out", default="data/ts",
                   help="output directory (one subdir per feature type)")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="fraction of clips for stratified test split")
    p.add_argument("--middle-sec", type=float, default=None,
                   help="if given, take the central N seconds of each clip "
                        "so all cases are equal length (fair comparison "
                        "across clips); e.g. 60 -> 12000 samples at 200 Hz")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _default_data_root()
    out_dir = Path(args.out) / args.features
    out_dir.mkdir(parents=True, exist_ok=True)

    load_fn = (load_subject_as_clips_raw
               if args.features == "raw"
               else load_subject_as_clips_de_flat)

    for sub in args.subjects:
        log.info("subject %d  (features=%s, middle_sec=%s)",
                 sub, args.features, args.middle_sec)
        X, y = load_fn(sub, root, middle_sec=args.middle_sec)
        lengths = [x.shape[1] for x in X]
        log.info("  n_cases=%d  n_channels=%d  length min/mean/max=%d/%d/%d",
                 len(X), X[0].shape[0],
                 min(lengths), int(np.mean(lengths)), max(lengths))
        log.info("  class distribution=%s", np.bincount(y, minlength=7).tolist())

        problem = f"seedVIIsubject{sub}"

        # Full file: all 80 clips
        save_to_ts_file(X, y, label_type="classification",
                        path=str(out_dir), problem_name=problem)
        log.info("  wrote %s/%s.ts  (all clips)", out_dir, problem)

        # Stratified train/test split for first single-subject experiments
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(np.zeros(len(y)), y))
        X_tr = [X[i] for i in tr_idx]; y_tr = y[tr_idx]
        X_te = [X[i] for i in te_idx]; y_te = y[te_idx]

        save_to_ts_file(X_tr, y_tr, label_type="classification",
                        path=str(out_dir), problem_name=problem,
                        file_suffix="_TRAIN")
        save_to_ts_file(X_te, y_te, label_type="classification",
                        path=str(out_dir), problem_name=problem,
                        file_suffix="_TEST")
        log.info("  wrote %s/%s_TRAIN.ts  (n=%d)", out_dir, problem, len(y_tr))
        log.info("  wrote %s/%s_TEST.ts   (n=%d)", out_dir, problem, len(y_te))


if __name__ == "__main__":
    main()
