"""Run one aeon classifier on the clip-level .ts files (Tony's setup).

Each case = one clip (middle-60 truncation), label = clip emotion.
Protocol:
  - ``loso``: train on 19 subjects' .ts files, test on 1 held-out subject,
    loop over all 20 subjects.
  - ``ws`` : use the provided TRAIN/TEST split per subject
    (``seedVIIsubject{N}_TRAIN.ts`` / ``_TEST.ts``), average over subjects.

Examples
--------
    python scripts/run_ts.py --classifier multirocket_hydra \\
        --ts-dir data/ts_mid60/de_flat --subjects {1..20} --protocol loso

    python scripts/run_ts.py --classifier multirocket_hydra \\
        --ts-dir data/ts_mid60/de_flat --subjects {1..20} --protocol ws
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from aeon.datasets import load_from_ts_file

from seedvii.eval.metrics import FoldResult, evaluate, summarise
from seedvii.models import build_classifier
from seedvii.utils import get_logger, set_seed

log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="multirocket_hydra")
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--ts-dir", required=True,
                   help="directory containing seedVIIsubject{N}.ts "
                        "(and _TRAIN/_TEST variants)")
    p.add_argument("--protocol",
                   choices=["loso", "ws", "cs5", "lovo"], default="loso",
                   help="loso = leave-one-subject-out (T leak + video leak); "
                        "cs5  = 5-fold cross-subject GroupKFold (T leak + video leak); "
                        "ws   = within-subject using provided TRAIN/TEST splits; "
                        "lovo = SEED-VII official 4-fold cross-video "
                        "(clean: test videos never seen in train, no T leak, "
                        "no video content leak)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/logs/run_ts.json")
    p.add_argument("--fixed-length", type=int, default=None,
                   help="force all cases to this length (centre-crop longer, "
                        "zero-pad shorter). Required for classifiers that "
                        "can't handle unequal length (MultiRocket, Hydra, etc.)")
    p.add_argument("--length-mode",
                   choices=["truncate", "resample", "hybrid"],
                   default="truncate",
                   help="how to enforce fixed_length: 'truncate' = centre-crop + "
                        "zero-pad (WARNING: zero-pad leaks clip-length → label); "
                        "'resample' = linear-interpolation for everyone (no leak); "
                        "'hybrid' = centre-crop if T>=target else resample up "
                        "(Tony's middle-N spirit)")
    return p.parse_args()


def load_subject(ts_dir: Path, sub: int, suffix: str = "") -> tuple[list, np.ndarray]:
    """Load a subject's .ts file, returning X as list of arrays, y as int64."""
    fpath = ts_dir / f"seedVIIsubject{sub}{suffix}.ts"
    X, y = load_from_ts_file(str(fpath))
    # y comes back as string labels ("0".."6"); cast to int
    y = np.asarray([int(v) for v in y], dtype=np.int64)
    return X, y


def _resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    """Per-channel linear interpolation to target_len along the time axis."""
    c, t = x.shape
    xp_src = np.linspace(0, 1, t)
    xp_tgt = np.linspace(0, 1, target_len)
    return np.stack(
        [np.interp(xp_tgt, xp_src, x[ch]) for ch in range(c)], axis=0
    ).astype(np.float32)


def pad_truncate_to_length(X: list[np.ndarray], target_len: int,
                            mode: str = "truncate") -> np.ndarray:
    """Force all cases in X to have the same n_timepoints = target_len.

    mode:
      - 'truncate': centre-crop longer, zero-pad shorter (WARNING: length leak).
      - 'resample': linear interpolation to target_len for everyone.
      - 'hybrid'  : centre-crop if T>=target, resample up if T<target
                    (Tony's "middle-N" spirit: preserve long clips as-is,
                    only stretch short clips).
    Returns 3D array (n_cases, n_channels, target_len).
    """
    out = []
    for x in X:
        c, t = x.shape
        if t == target_len:
            out.append(x.astype(np.float32))
            continue
        if mode == "resample":
            out.append(_resample_linear(x, target_len))
        elif mode == "hybrid":
            if t >= target_len:
                start = (t - target_len) // 2
                out.append(x[:, start:start + target_len].astype(np.float32))
            else:
                out.append(_resample_linear(x, target_len))
        else:  # truncate / zero-pad
            if t > target_len:
                start = (t - target_len) // 2
                out.append(x[:, start:start + target_len].astype(np.float32))
            else:
                pad = np.zeros((c, target_len - t), dtype=np.float32)
                out.append(np.concatenate([x.astype(np.float32), pad], axis=1))
    return np.stack(out, axis=0)


def stack_cases(X_list: list[list], y_list: list[np.ndarray]) -> tuple[np.ndarray | list, np.ndarray]:
    """Concatenate per-subject case lists. Returns 3D ndarray if shapes match,
    else keeps as list (for variable-length series)."""
    all_X = [x for Xs in X_list for x in Xs]
    all_y = np.concatenate(y_list)
    # If all cases share the same (channels, timepoints), stack to 3D ndarray
    shapes = {x.shape for x in all_X}
    if len(shapes) == 1:
        return np.stack(all_X, axis=0), all_y
    return all_X, all_y


def run_loso(args: argparse.Namespace) -> list[FoldResult]:
    ts_dir = Path(args.ts_dir)
    log.info("loading %d subjects from %s …", len(args.subjects), ts_dir)

    # Preload all subjects once
    per_sub: dict[int, tuple[list, np.ndarray]] = {}
    for sub in args.subjects:
        X, y = load_subject(ts_dir, sub)
        per_sub[sub] = (X, y)
        log.info("  subject %d: n=%d  shape=%s  classes=%s",
                 sub, len(X), X[0].shape, np.bincount(y, minlength=7).tolist())

    fold_results: list[FoldResult] = []
    for fold, held_out in enumerate(args.subjects, 1):
        train_subs = [s for s in args.subjects if s != held_out]
        X_tr_list = [x for s in train_subs for x in per_sub[s][0]]
        y_tr = np.concatenate([per_sub[s][1] for s in train_subs])
        X_te_list, y_te = per_sub[held_out]

        if args.fixed_length is not None:
            X_tr = pad_truncate_to_length(X_tr_list, args.fixed_length, mode=args.length_mode)
            X_te = pad_truncate_to_length(X_te_list, args.fixed_length, mode=args.length_mode)
        else:
            X_tr, _ = stack_cases([[x for x in X_tr_list]], [y_tr])
            X_te = (np.stack(X_te_list, axis=0)
                    if len({x.shape for x in X_te_list}) == 1 else X_te_list)

        log.info("=== Fold %d/%d  hold-out=sub%02d  train=%d  test=%d ===",
                 fold, len(args.subjects), held_out,
                 len(X_tr) if hasattr(X_tr, "__len__") else X_tr.shape[0],
                 len(X_te) if hasattr(X_te, "__len__") else X_te.shape[0])
        clf = build_classifier(args.classifier, n_jobs=args.n_jobs,
                               random_state=args.seed)
        t0 = time.time(); clf.fit(X_tr, y_tr); t_fit = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(X_te); t_pred = time.time() - t0
        acc, f1 = evaluate(y_te, y_pred)
        log.info("  acc=%.4f  macro_f1=%.4f  fit=%.1fs  pred=%.1fs",
                 acc, f1, t_fit, t_pred)
        fold_results.append(FoldResult(fold=fold, acc=acc, macro_f1=f1,
                                       y_true=y_te, y_pred=y_pred))
    return fold_results


def run_lovo(args: argparse.Namespace) -> list[FoldResult]:
    """SEED-VII official 4-fold cross-video protocol.

    fold_of_video(v) = ((v - 1) % 20) // 5      (v in 1..80)
    Each fold: 20 videos × n_subjects test cases, 60 videos × n_subjects train.
    Cases inside seedVIIsubject{N}.ts are stored in video_id order 1..80, so
    case index i (0-based) -> video_id = i + 1.
    """
    ts_dir = Path(args.ts_dir)
    log.info("loading %d subjects from %s …", len(args.subjects), ts_dir)

    per_sub: dict[int, tuple[list, np.ndarray]] = {}
    for sub in args.subjects:
        X, y = load_subject(ts_dir, sub)
        assert len(X) == 80, (
            f"LOVO expects 80 cases per subject (video_id 1..80); "
            f"subject {sub} has {len(X)} — probably middle-N truncated .ts"
        )
        per_sub[sub] = (X, y)

    # video_id (1..80) -> fold_id (0..3)
    fold_of_vid = np.array([((v - 1) % 20) // 5 for v in range(1, 81)])

    fold_results: list[FoldResult] = []
    for fold in range(4):
        tr_mask = fold_of_vid != fold
        te_mask = fold_of_vid == fold

        X_tr_list, y_tr_list = [], []
        X_te_list, y_te_list = [], []
        for sub in args.subjects:
            X_s, y_s = per_sub[sub]
            for i, (x, yi) in enumerate(zip(X_s, y_s)):
                if tr_mask[i]:
                    X_tr_list.append(x); y_tr_list.append(yi)
                else:
                    X_te_list.append(x); y_te_list.append(yi)
        y_tr = np.asarray(y_tr_list, dtype=np.int64)
        y_te = np.asarray(y_te_list, dtype=np.int64)

        if args.fixed_length is not None:
            X_tr = pad_truncate_to_length(X_tr_list, args.fixed_length,
                                          mode=args.length_mode)
            X_te = pad_truncate_to_length(X_te_list, args.fixed_length,
                                          mode=args.length_mode)
        else:
            X_tr = (np.stack(X_tr_list, axis=0)
                    if len({x.shape for x in X_tr_list}) == 1 else X_tr_list)
            X_te = (np.stack(X_te_list, axis=0)
                    if len({x.shape for x in X_te_list}) == 1 else X_te_list)

        log.info("=== LOVO Fold %d/4  test videos=%d  train videos=%d  "
                 "n_train=%d  n_test=%d ===",
                 fold + 1, te_mask.sum(), tr_mask.sum(),
                 len(X_tr) if hasattr(X_tr, "__len__") else X_tr.shape[0],
                 len(X_te) if hasattr(X_te, "__len__") else X_te.shape[0])
        clf = build_classifier(args.classifier, n_jobs=args.n_jobs,
                               random_state=args.seed)
        t0 = time.time(); clf.fit(X_tr, y_tr); t_fit = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(X_te); t_pred = time.time() - t0
        acc, f1 = evaluate(y_te, y_pred)
        log.info("  acc=%.4f  macro_f1=%.4f  fit=%.1fs  pred=%.1fs",
                 acc, f1, t_fit, t_pred)
        fold_results.append(FoldResult(fold=fold + 1, acc=acc, macro_f1=f1,
                                       y_true=y_te, y_pred=y_pred))
    return fold_results


def run_ws(args: argparse.Namespace) -> list[FoldResult]:
    ts_dir = Path(args.ts_dir)
    fold_results: list[FoldResult] = []
    for sub in args.subjects:
        X_tr, y_tr = load_subject(ts_dir, sub, suffix="_TRAIN")
        X_te, y_te = load_subject(ts_dir, sub, suffix="_TEST")
        if args.fixed_length is not None:
            X_tr = pad_truncate_to_length(X_tr, args.fixed_length, mode=args.length_mode)
            X_te = pad_truncate_to_length(X_te, args.fixed_length, mode=args.length_mode)
        else:
            X_tr = np.stack(X_tr, axis=0) if len({x.shape for x in X_tr}) == 1 else X_tr
            X_te = np.stack(X_te, axis=0) if len({x.shape for x in X_te}) == 1 else X_te
        log.info("=== Sub%02d  train=%d  test=%d ===",
                 sub, len(X_tr) if hasattr(X_tr, "__len__") else X_tr.shape[0],
                 len(X_te) if hasattr(X_te, "__len__") else X_te.shape[0])
        clf = build_classifier(args.classifier, n_jobs=args.n_jobs,
                               random_state=args.seed)
        t0 = time.time(); clf.fit(X_tr, y_tr); t_fit = time.time() - t0
        t0 = time.time(); y_pred = clf.predict(X_te); t_pred = time.time() - t0
        acc, f1 = evaluate(y_te, y_pred)
        log.info("  acc=%.4f  macro_f1=%.4f  fit=%.1fs  pred=%.1fs",
                 acc, f1, t_fit, t_pred)
        fold_results.append(FoldResult(fold=sub, acc=acc, macro_f1=f1,
                                       y_true=y_te, y_pred=y_pred))
    return fold_results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    log.info("classifier=%s  protocol=%s  ts_dir=%s  subjects=%s",
             args.classifier, args.protocol, args.ts_dir, args.subjects)

    if args.protocol == "loso":
        results = run_loso(args)
    elif args.protocol == "lovo":
        results = run_lovo(args)
    else:
        results = run_ws(args)

    summary = summarise(results)
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
