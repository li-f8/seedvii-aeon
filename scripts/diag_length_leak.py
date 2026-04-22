"""Diagnose whether clip length T alone encodes emotion label.

For each subject we build a 1-feature table (T = n_timepoints of each .ts case)
and run LOSO with a simple classifier. If acc >> 1/7 (= 14.3%) the dataset
leaks label through clip duration, and any aligner that preserves T (zero-pad
truncate, resample, hybrid) will implicitly exploit it.

Usage:
    python scripts/diag_length_leak.py --ts-dir data/ts_full/de_flat \\
        --subjects {1..20}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from aeon.datasets import load_from_ts_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ts-dir", required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_lengths(ts_dir: Path, sub: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_from_ts_file(str(ts_dir / f"seedVIIsubject{sub}.ts"))
    T = np.asarray([x.shape[1] for x in X], dtype=np.int64)
    y = np.asarray([int(v) for v in y], dtype=np.int64)
    return T, y


def main() -> None:
    args = parse_args()
    ts_dir = Path(args.ts_dir)

    per_sub: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sub in args.subjects:
        T, y = load_lengths(ts_dir, sub)
        per_sub[sub] = (T, y)

    # Global length-label contingency on pooled data
    T_all = np.concatenate([per_sub[s][0] for s in args.subjects])
    y_all = np.concatenate([per_sub[s][1] for s in args.subjects])
    print(f"\n=== Pooled stats (n={len(y_all)}) ===")
    print(f"T: min={T_all.min()}  median={int(np.median(T_all))}  "
          f"max={T_all.max()}  unique={len(np.unique(T_all))}")
    print("Per-class mean T:")
    for c in range(7):
        mask = y_all == c
        if mask.any():
            print(f"  class {c}: n={mask.sum():4d}  "
                  f"T mean={T_all[mask].mean():.2f}  "
                  f"median={int(np.median(T_all[mask]))}  "
                  f"std={T_all[mask].std():.2f}")

    # LOSO with T-only feature
    print("\n=== LOSO using ONLY T as feature ===")
    accs, f1s = [], []
    dummy_accs = []
    for held in args.subjects:
        T_tr = np.concatenate([per_sub[s][0] for s in args.subjects if s != held])
        y_tr = np.concatenate([per_sub[s][1] for s in args.subjects if s != held])
        T_te, y_te = per_sub[held]

        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed,
                                     n_jobs=-1)
        clf.fit(T_tr.reshape(-1, 1), y_tr)
        y_pred = clf.predict(T_te.reshape(-1, 1))
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")

        dummy = DummyClassifier(strategy="most_frequent").fit(
            T_tr.reshape(-1, 1), y_tr)
        d_acc = accuracy_score(y_te, dummy.predict(T_te.reshape(-1, 1)))

        accs.append(acc); f1s.append(f1); dummy_accs.append(d_acc)
        print(f"  sub{held:02d}  T-only acc={acc:.4f}  f1={f1:.4f}  "
              f"dummy={d_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"T-only  acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"T-only  f1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Dummy   acc = {np.mean(dummy_accs):.4f} ± {np.std(dummy_accs):.4f}")
    print(f"Chance (1/7) = {1/7:.4f}")
    print("=" * 60)
    print("Interpretation:")
    print("  - if T-only acc ≈ chance: length does NOT leak label")
    print("  - if T-only acc >> chance: any aligner preserving T leaks label")


if __name__ == "__main__":
    main()
