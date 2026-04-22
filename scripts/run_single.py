"""Smoke-test entry point: run one aeon classifier on a subset of subjects.

Examples
--------
    python scripts/run_single.py --classifier multirocket_hydra --subjects 1 2 3
    python scripts/run_single.py --classifier multirocket_hydra --subjects 1 2 3 \
        --modality eeg+eye
    python scripts/run_single.py --classifier drcif --subjects 1 2 3 4 5 \
        --protocol cross_subject
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from seedvii.data import load_multimodal, load_raw_multimodal, load_de_sequence_multimodal
from seedvii.data.aeon_format import de_to_aeon, eye_to_aeon
from seedvii.eval.metrics import FoldResult, evaluate, summarise
from seedvii.eval.protocols import cross_subject_splits, within_subject_splits
from seedvii.models import build_classifier
from seedvii.utils import get_logger, set_seed

log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="multirocket_hydra")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--protocol", choices=["cross_subject", "within_subject", "loso"],
                   default="cross_subject",
                   help="loso = cross_subject with n_folds = n_subjects")
    p.add_argument("--modality", choices=["eeg", "eye", "eeg+eye"], default="eeg")
    p.add_argument("--features", choices=["de", "de_seq", "raw"], default="de_seq",
                   help="de = flat DE per second; de_seq = windowed DE time series "
                        "(recommended); raw = windowed raw EEG")
    p.add_argument("--win-sec", type=float, default=10.0,
                   help="window length in seconds (de_seq: integer seconds; raw: any)")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalise", action="store_true",
                   help="disable per-subject z-score (default: normalise=True)")
    p.add_argument("--out", default="results/logs/run_single.json")
    return p.parse_args()


def assemble_features(data: dict, modality: str) -> np.ndarray:
    X_eeg = de_to_aeon(data["X_eeg"])      # (N, 62, 5)
    if modality == "eeg":
        return X_eeg
    X_eye = eye_to_aeon(data["X_eye"])     # (N, 33, 1)
    if modality == "eye":
        return X_eye
    # eeg+eye: early concat on channel axis — broadcast eye to 5 timepoints
    X_eye_b = np.broadcast_to(X_eye, (X_eye.shape[0], X_eye.shape[1], 5)).copy()
    return np.concatenate([X_eeg, X_eye_b], axis=1)  # (N, 95, 5)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # LOSO = cross_subject with n_folds = n_subjects
    if args.protocol == "loso":
        args.protocol = "cross_subject"
        args.n_folds = len(args.subjects)
        log.info("LOSO mode: protocol=cross_subject, n_folds=%d", args.n_folds)
    log.info("loading subjects %s  (features=%s, modality=%s) …",
             args.subjects, args.features, args.modality)

    normalise = not args.no_normalise
    log.info("normalise (per-subject z-score): %s", normalise)

    if args.features == "raw":
        if args.modality != "eeg":
            raise ValueError("raw features currently support EEG-only; use --modality eeg")
        data = load_raw_multimodal(
            subject_ids=args.subjects, win_sec=args.win_sec, normalise=normalise)
        X = data["X_eeg"]  # (N, 62, win_len)
    elif args.features == "de_seq":
        if args.modality != "eeg":
            raise ValueError("de_seq currently supports EEG-only; use --modality eeg")
        data = load_de_sequence_multimodal(
            subject_ids=args.subjects, win_sec=int(args.win_sec), normalise=normalise)
        X = data["X_eeg"]  # (N, 310, win_sec)
    else:
        data = load_multimodal(subject_ids=args.subjects, normalise=normalise)
        X = assemble_features(data, args.modality)
    y = data["y"]
    log.info("X shape=%s  y shape=%s  class counts=%s",
             X.shape, y.shape, np.bincount(y).tolist())

    fold_results: list[FoldResult] = []

    if args.protocol == "cross_subject":
        splits = list(cross_subject_splits(data["subjects"],
                                           n_splits=args.n_folds, seed=args.seed))
        for fold, (tr, te) in enumerate(splits, 1):
            log.info("=== Fold %d/%d  train=%d  test=%d ===",
                     fold, len(splits), len(tr), len(te))
            clf = build_classifier(args.classifier, n_jobs=args.n_jobs,
                                   random_state=args.seed)
            t0 = time.time(); clf.fit(X[tr], y[tr]); t_fit = time.time() - t0
            t0 = time.time(); y_pred = clf.predict(X[te]); t_pred = time.time() - t0
            acc, f1 = evaluate(y[te], y_pred)
            log.info("  acc=%.4f  macro_f1=%.4f  fit=%.1fs  pred=%.1fs",
                     acc, f1, t_fit, t_pred)
            fold_results.append(FoldResult(fold=fold, acc=acc, macro_f1=f1,
                                           y_true=y[te], y_pred=y_pred))

    else:  # within_subject: loop per subject, each subject has its own CV
        subjects_arr = data["subjects"]
        video_ids = data["video_ids"]
        for sub in np.unique(subjects_arr):
            sub_mask = subjects_arr == sub
            sub_idx = np.where(sub_mask)[0]
            sub_v = video_ids[sub_mask]
            sub_y = y[sub_mask]
            splits = list(within_subject_splits(
                sub_v, sub_y, n_splits=args.n_folds, seed=args.seed))
            for fold, (tr_local, te_local) in enumerate(splits, 1):
                tr = sub_idx[tr_local]
                te = sub_idx[te_local]
                clf = build_classifier(args.classifier, n_jobs=args.n_jobs,
                                       random_state=args.seed)
                t0 = time.time(); clf.fit(X[tr], y[tr]); t_fit = time.time() - t0
                t0 = time.time(); y_pred = clf.predict(X[te]); t_pred = time.time() - t0
                acc, f1 = evaluate(y[te], y_pred)
                log.info("Sub%02d Fold %d/%d  tr=%d te=%d  acc=%.4f  f1=%.4f  "
                         "fit=%.1fs  pred=%.1fs",
                         sub, fold, len(splits), len(tr), len(te),
                         acc, f1, t_fit, t_pred)
                fold_results.append(FoldResult(
                    fold=int(sub) * 100 + fold, acc=acc, macro_f1=f1,
                    y_true=y[te], y_pred=y_pred))

    summary = summarise(fold_results)
    summary["config"] = vars(args)
    log.info("=" * 60)
    log.info("%s  acc=%.4f±%.4f  f1=%.4f±%.4f",
             args.classifier,
             summary["acc_mean"], summary["acc_std"],
             summary["f1_mean"], summary["f1_std"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
