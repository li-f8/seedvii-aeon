"""Cross-validation protocols for SEED-VII.

Two settings:
    * Cross-Subject (CS) — 5-fold group split over 20 subjects
    * Within-Subject (WS) — stratified 5-fold by video, inside each subject
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def cross_subject_splits(
    subjects: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per fold, grouped by subject id."""
    unique_subs = np.unique(subjects)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_sub_idx, te_sub_idx in kf.split(unique_subs):
        tr_subs = unique_subs[tr_sub_idx]
        te_subs = unique_subs[te_sub_idx]
        tr_mask = np.isin(subjects, tr_subs)
        te_mask = np.isin(subjects, te_subs)
        yield np.where(tr_mask)[0], np.where(te_mask)[0]


def within_subject_splits(
    video_ids: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per fold, stratified by video so that windows
    from the same video never appear in both splits.

    Call once per subject (after masking to that subject's rows).
    """
    unique_v = np.unique(video_ids)
    v_label = np.array([y[video_ids == v][0] for v in unique_v])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_v_idx, te_v_idx in skf.split(unique_v, v_label):
        tr_vids = unique_v[tr_v_idx]
        te_vids = unique_v[te_v_idx]
        tr_mask = np.isin(video_ids, tr_vids)
        te_mask = np.isin(video_ids, te_vids)
        yield np.where(tr_mask)[0], np.where(te_mask)[0]
