"""Evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class FoldResult:
    fold: int
    acc: float
    macro_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def summarise(results: list[FoldResult]) -> dict:
    """Aggregate fold-level metrics."""
    accs = np.array([r.acc for r in results])
    f1s = np.array([r.macro_f1 for r in results])
    y_true = np.concatenate([r.y_true for r in results])
    y_pred = np.concatenate([r.y_pred for r in results])
    return {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "acc_per_fold": accs.tolist(),
        "f1_per_fold": f1s.tolist(),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }
