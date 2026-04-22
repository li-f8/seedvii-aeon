"""Phase 2: no-leak deep learning baseline on SEED-VII.

Reads the *same* .ts files and the *same* 4-fold video partition
(`data/seedvii_folds.json`) that Phase 1 aeon classifiers use, so results
are directly comparable on a per-fold basis.

Protocol: SEED-VII official 4-fold cross-video (LOVO).
  - each fold's test set = 20 videos × 20 subjects = 400 cases
  - each fold's train set = 60 videos × 20 subjects = 1200 cases
  - test videos never appear in train → no T leak, no video content leak

Example
-------
    python scripts/run_dl_lovo.py --model decnn \\
        --ts-dir data/ts_full/de_flat --subjects {1..20} \\
        --fixed-length 90 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aeon.datasets import load_from_ts_file

from seedvii.eval.metrics import FoldResult, evaluate, summarise
from seedvii.models.dl import build_dl
from seedvii.utils import get_logger, set_seed

log = get_logger()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_subject(ts_dir: Path, sub: int) -> tuple[list, np.ndarray]:
    X, y = load_from_ts_file(str(ts_dir / f"seedVIIsubject{sub}.ts"))
    y = np.asarray([int(v) for v in y], dtype=np.int64)
    return X, y


def _resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    c, t = x.shape
    xp_src = np.linspace(0, 1, t)
    xp_tgt = np.linspace(0, 1, target_len)
    return np.stack(
        [np.interp(xp_tgt, xp_src, x[ch]) for ch in range(c)], axis=0
    ).astype(np.float32)


def resample_all(X: list[np.ndarray], target_len: int) -> np.ndarray:
    """Linear-interpolate every case to target_len (matches Phase 1 resample)."""
    out = []
    for x in X:
        if x.shape[1] == target_len:
            out.append(x.astype(np.float32))
        else:
            out.append(_resample_linear(x, target_len))
    return np.stack(out, axis=0)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *, model_name: str, epochs: int, batch_size: int,
    lr: float, weight_decay: float, device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)

    # Per-feature standardisation fitted on train only (avoid leakage)
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    ds_tr = TensorDataset(torch.from_numpy(X_tr_n).float(),
                          torch.from_numpy(y_tr).long())
    ds_te = TensorDataset(torch.from_numpy(X_te_n).float(),
                          torch.from_numpy(y_te).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    model = build_dl(model_name, in_ch=X_tr.shape[1], n_classes=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = -1.0
    best_y_pred = None
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0; n = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        tr_loss /= max(n, 1)

        # Eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                trues.append(yb.numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        acc, f1 = evaluate(y_true, y_pred)
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "test_acc": acc, "test_f1": f1})
        if acc > best_acc:
            best_acc = acc; best_y_pred = y_pred.copy()
        if ep % 10 == 0 or ep == 1:
            log.info("  ep %3d  tr_loss=%.4f  te_acc=%.4f  te_f1=%.4f  "
                     "best_acc=%.4f", ep, tr_loss, acc, f1, best_acc)

    return best_y_pred, {"history": history, "best_acc": best_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="decnn")
    p.add_argument("--ts-dir", required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--folds-json", default="data/seedvii_folds.json")
    p.add_argument("--fixed-length", type=int, default=90,
                   help="resample every clip to this length (Phase 1 winner: 90)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--subject-zscore", action="store_true", default=True,
                   help="per-subject per-channel z-score at load time "
                        "(default on; --no-subject-zscore to disable)")
    p.add_argument("--no-subject-zscore", dest="subject_zscore",
                   action="store_false")
    p.add_argument("--out", default="results/logs/dl_lovo.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (pick_device() if args.device == "auto"
              else torch.device(args.device))
    log.info("model=%s  device=%s  epochs=%d  bs=%d  lr=%.1e  L=%d",
             args.model, device, args.epochs, args.batch_size, args.lr,
             args.fixed_length)

    # Load fold partition
    folds = json.loads(Path(args.folds_json).read_text())
    fold_of_vid = {v["video_id"]: v["fold"] for v in folds["videos"]}
    log.info("loaded %d videos across %d folds from %s",
             folds["n_videos"], folds["n_folds"], args.folds_json)

    # Load all subjects once, resample, optionally per-subject z-score,
    # keep per-(subject, video) indexing
    ts_dir = Path(args.ts_dir)
    all_X, all_y, all_vid = [], [], []
    for sub in args.subjects:
        X_list, y = load_subject(ts_dir, sub)
        assert len(X_list) == 80, f"subject {sub} has {len(X_list)} cases, expected 80"
        X = resample_all(X_list, args.fixed_length)   # (80, 310, L)
        if args.subject_zscore:
            mu = X.mean(axis=(0, 2), keepdims=True)      # per-channel
            sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
            X = ((X - mu) / sd).astype(np.float32)
        all_X.append(X)
        all_y.append(y)
        all_vid.append(np.arange(1, 81))
    X_all = np.concatenate(all_X, axis=0)        # (n_sub * 80, 310, L)
    y_all = np.concatenate(all_y, axis=0)
    vid_all = np.concatenate(all_vid, axis=0)
    log.info("assembled X=%s  y=%s  classes=%s",
             X_all.shape, y_all.shape, np.bincount(y_all, minlength=7).tolist())

    fold_of_case = np.array([fold_of_vid[int(v)] for v in vid_all])

    fold_results: list[FoldResult] = []
    histories: list[dict] = []
    for fold in range(folds["n_folds"]):
        tr_mask = fold_of_case != fold
        te_mask = fold_of_case == fold
        log.info("=== LOVO Fold %d/%d  train=%d  test=%d ===",
                 fold + 1, folds["n_folds"], tr_mask.sum(), te_mask.sum())
        t0 = time.time()
        y_pred, info = train_one_fold(
            X_all[tr_mask], y_all[tr_mask],
            X_all[te_mask], y_all[te_mask],
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, device=device,
            seed=args.seed + fold,
        )
        dt = time.time() - t0
        acc, f1 = evaluate(y_all[te_mask], y_pred)
        log.info("  fold %d BEST  acc=%.4f  f1=%.4f  time=%.1fs",
                 fold + 1, acc, f1, dt)
        fold_results.append(FoldResult(fold=fold + 1, acc=acc, macro_f1=f1,
                                       y_true=y_all[te_mask], y_pred=y_pred))
        histories.append({"fold": fold + 1, **info})

    summary = summarise(fold_results)
    summary["config"] = vars(args)
    summary["config"]["device"] = str(device)
    summary["histories"] = histories
    log.info("=" * 60)
    log.info("%s  acc=%.4f±%.4f  f1=%.4f±%.4f", args.model,
             summary["acc_mean"], summary["acc_std"],
             summary["f1_mean"], summary["f1_std"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
