"""Phase 3: EEG + Eye late-fusion baseline on SEED-VII.

For each fold (under lovo / loso / random), trains two separate DECNN
models — one on EEG DE-flat (310 ch), one on Eye features (33 ch) —
using the same segment-level + clip-level soft-vote recipe as
run_dl_lovo_segments.py. At evaluation time the per-clip softmax
probabilities of the two models are averaged and argmax gives the
clip-level fused prediction.

Assumes the EEG and Eye .ts files have been produced by scripts/to_ts.py
and share the same per-video length (T_vid), so segments align 1:1
between modalities by (subject, video) pair and by segment index within
a clip.

Example
-------
    python scripts/run_dl_fusion.py \\
        --ts-dir-eeg data/ts_full/de_flat \\
        --ts-dir-eye data/ts_full/eye \\
        --subjects {1..20} --protocol lovo \\
        --seg-len 5 --stride 3 --epochs 40 --lr 3e-4
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
# Data (re-implemented minimally; run_dl_lovo_segments.py has the same)
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


def clip_to_segments(x: np.ndarray, seg_len: int, stride: int,
                     ) -> list[np.ndarray]:
    c, t = x.shape
    if t < seg_len:
        return [_resample_linear(x, seg_len)]
    segs: list[np.ndarray] = []
    starts = list(range(0, t - seg_len + 1, stride))
    if starts and starts[-1] + seg_len < t:
        starts.append(t - seg_len)
    if not starts:
        starts = [0]
    for s in starts:
        segs.append(x[:, s:s + seg_len].astype(np.float32))
    return segs


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Train-one-fold variant that also returns best-epoch clip-level probabilities
# ---------------------------------------------------------------------------

def train_one_fold_with_probs(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te_seg: np.ndarray, clip_id_te: np.ndarray,
    y_te_clip: np.ndarray,
    *, model_name: str, epochs: int, batch_size: int,
    lr: float, weight_decay: float, device: torch.device, seed: int,
    tag: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same as run_dl_lovo_segments.train_one_fold but also returns the
    per-clip softmax probabilities at the best epoch (for fusion)."""
    set_seed(seed)

    ds_tr = TensorDataset(torch.from_numpy(X_tr).float(),
                          torch.from_numpy(y_tr).long())
    ds_te = TensorDataset(torch.from_numpy(X_te).float(),
                          torch.from_numpy(y_te_seg).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    model = build_dl(model_name, in_ch=X_tr.shape[1], n_classes=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    n_test_clips = len(y_te_clip)
    best_clip_acc = -1.0
    best_clip_pred: np.ndarray | None = None
    best_clip_probs: np.ndarray | None = None
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0; n = 0
        skipped = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        tr_loss /= max(n, 1)

        model.eval()
        probs_list = []
        with torch.no_grad():
            for xb, _ in dl_te:
                xb = xb.to(device)
                p = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs_list.append(p)
        probs = np.concatenate(probs_list, axis=0)

        seg_acc = float((probs.argmax(1) == y_te_seg).mean())

        clip_probs = np.zeros((n_test_clips, 7), dtype=np.float32)
        counts = np.zeros(n_test_clips, dtype=np.int64)
        for i, cid in enumerate(clip_id_te):
            clip_probs[cid] += probs[i]; counts[cid] += 1
        clip_probs /= counts[:, None]
        clip_pred = clip_probs.argmax(1)
        clip_acc, clip_f1 = evaluate(y_te_clip, clip_pred)

        history.append({"epoch": ep, "train_loss": tr_loss,
                        "seg_acc": seg_acc, "clip_acc": clip_acc,
                        "clip_f1": clip_f1,
                        "skipped_batches": skipped})
        if clip_acc > best_clip_acc:
            best_clip_acc = clip_acc
            best_clip_pred = clip_pred.copy()
            best_clip_probs = clip_probs.copy()
        if ep % 10 == 0 or ep == 1:
            log.info("  [%s] ep %3d  tr_loss=%.4f  seg_acc=%.4f  "
                     "clip_acc=%.4f  clip_f1=%.4f  best=%.4f  skip=%d",
                     tag, ep, tr_loss, seg_acc, clip_acc, clip_f1,
                     best_clip_acc, skipped)

    return best_clip_pred, best_clip_probs, {
        "history": history, "best_clip_acc": best_clip_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ts-dir-eeg", required=True)
    p.add_argument("--ts-dir-eye", required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--folds-json", default="data/seedvii_folds.json")
    p.add_argument("--model", default="decnn")
    p.add_argument("--seg-len", type=int, default=5)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--subject-zscore", action="store_true", default=True)
    p.add_argument("--no-subject-zscore", dest="subject_zscore",
                   action="store_false")
    p.add_argument("--protocol", default="lovo",
                   choices=["lovo", "loso", "random"])
    p.add_argument("--fusion-weight-eeg", type=float, default=0.5,
                   help="primary weight for headline FUSED line; the script "
                        "also sweeps w_eeg ∈ {0.0, 0.1, ..., 1.0} and records "
                        "all 11 fused accuracies in the output JSON")
    p.add_argument("--out", default="results/logs/dl_fusion.json")
    return p.parse_args()


def build_modality_arrays(ts_dir: Path, subjects: list[int],
                          seg_len: int, stride: int, subject_zscore: bool,
                          fold_of_vid: dict[int, int]):
    """Return X_seg, y_seg, clip_of_seg, y_clip, clip_video, clip_subject,
    clip_fold_lovo (for parity with run_dl_lovo_segments)."""
    seg_list, seg_label, seg_clip = [], [], []
    clip_label, clip_fold, clip_meta = [], [], []
    clip_counter = 0
    for sub in subjects:
        X_list, y = load_subject(ts_dir, sub)
        assert len(X_list) == 80, f"subject {sub} has {len(X_list)} cases"
        if subject_zscore:
            stacked = np.concatenate(X_list, axis=1)
            mu = stacked.mean(axis=1, keepdims=True)
            sd = stacked.std(axis=1, keepdims=True) + 1e-6
            X_list = [((x - mu) / sd).astype(np.float32) for x in X_list]
        for i, (x, yi) in enumerate(zip(X_list, y)):
            vid = i + 1
            segs = clip_to_segments(x, seg_len, stride)
            for s in segs:
                seg_list.append(s); seg_label.append(int(yi))
                seg_clip.append(clip_counter)
            clip_label.append(int(yi))
            clip_fold.append(fold_of_vid[vid])
            clip_meta.append((int(sub), vid))
            clip_counter += 1
    X_seg = np.stack(seg_list, axis=0)
    y_seg = np.asarray(seg_label, dtype=np.int64)
    clip_of_seg = np.asarray(seg_clip, dtype=np.int64)
    y_clip = np.asarray(clip_label, dtype=np.int64)
    clip_video = np.asarray([m[1] for m in clip_meta], dtype=np.int64)
    clip_subject = np.asarray([m[0] for m in clip_meta], dtype=np.int64)
    clip_fold_lovo = np.asarray(clip_fold, dtype=np.int64)
    return (X_seg, y_seg, clip_of_seg, y_clip, clip_video, clip_subject,
            clip_fold_lovo)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (pick_device() if args.device == "auto"
              else torch.device(args.device))
    log.info("model=%s  protocol=%s  device=%s  epochs=%d  bs=%d  lr=%.1e  "
             "seg_len=%d  stride=%d  w_eeg=%.2f",
             args.model, args.protocol, device, args.epochs,
             args.batch_size, args.lr, args.seg_len, args.stride,
             args.fusion_weight_eeg)

    folds = json.loads(Path(args.folds_json).read_text())
    fold_of_vid = {v["video_id"]: v["fold"] for v in folds["videos"]}

    # Build both modality streams (identical segment layout: same T_vid,
    # seg_len, stride → same segment count per (subject, video))
    log.info("loading EEG  from %s", args.ts_dir_eeg)
    (X_eeg, y_seg_eeg, clip_of_seg_eeg, y_clip, clip_video, clip_subject,
     clip_fold_lovo) = build_modality_arrays(
        Path(args.ts_dir_eeg), args.subjects, args.seg_len, args.stride,
        args.subject_zscore, fold_of_vid)
    log.info("loading Eye  from %s", args.ts_dir_eye)
    (X_eye, y_seg_eye, clip_of_seg_eye, y_clip2, _, _, _) = \
        build_modality_arrays(
        Path(args.ts_dir_eye), args.subjects, args.seg_len, args.stride,
        args.subject_zscore, fold_of_vid)

    # Sanity: the two modalities must produce identical segment layout so
    # clip ids align. (Same T_vid per clip, same seg_len/stride.)
    assert np.array_equal(y_seg_eeg, y_seg_eye), "segment labels differ"
    assert np.array_equal(clip_of_seg_eeg, clip_of_seg_eye), \
        "segment→clip mapping differs — modality T_vid mismatch"
    assert np.array_equal(y_clip, y_clip2)
    log.info("X_eeg=%s  X_eye=%s  n_segs=%d  n_clips=%d",
             X_eeg.shape, X_eye.shape, X_eeg.shape[0], len(y_clip))

    # Protocol → fold_of_clip
    n_folds = folds["n_folds"]
    if args.protocol == "lovo":
        fold_of_clip = clip_fold_lovo
    elif args.protocol == "loso":
        subs = sorted(set(int(s) for s in args.subjects))
        assert len(subs) % n_folds == 0
        per = len(subs) // n_folds
        sub_to_fold = {s: i // per for i, s in enumerate(subs)}
        fold_of_clip = np.asarray(
            [sub_to_fold[int(s)] for s in clip_subject], dtype=np.int64)
    elif args.protocol == "random":
        rng = np.random.default_rng(args.seed)
        fold_of_clip = np.full(len(y_clip), -1, dtype=np.int64)
        for lbl in np.unique(y_clip):
            idx = np.where(y_clip == lbl)[0]
            rng.shuffle(idx)
            for j, k in enumerate(idx):
                fold_of_clip[k] = j % n_folds
    else:
        raise ValueError(args.protocol)

    # --- Per-fold training ---
    clip_of_seg = clip_of_seg_eeg   # identical
    fold_results_fused: list[FoldResult] = []
    fold_results_eeg: list[FoldResult] = []
    fold_results_eye: list[FoldResult] = []
    histories = []
    w_e, w_y = args.fusion_weight_eeg, 1.0 - args.fusion_weight_eeg

    # Weight sweep: for each fold we store per-weight clip-level probs so
    # we can aggregate at the end. Weights at 0.0, 0.1, ..., 1.0 (EEG side).
    sweep_weights = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2).tolist()
    # per-fold per-weight predictions: list over folds of dict[w -> (y_true, y_pred)]
    sweep_fold_records: list[dict] = []

    for fold in range(n_folds):
        tr_clip_mask = fold_of_clip != fold
        te_clip_mask = fold_of_clip == fold
        tr_seg_mask = tr_clip_mask[clip_of_seg]
        te_seg_mask = te_clip_mask[clip_of_seg]

        te_clip_ids_global = np.where(te_clip_mask)[0]
        remap = {int(g): j for j, g in enumerate(te_clip_ids_global)}
        te_seg_clip_local = np.array(
            [remap[int(c)] for c in clip_of_seg[te_seg_mask]], dtype=np.int64)
        y_te_clip = y_clip[te_clip_mask]

        log.info("=== %s Fold %d/%d  train clips=%d  test clips=%d  "
                 "train segs=%d  test segs=%d ===",
                 args.protocol.upper(), fold + 1, n_folds,
                 tr_clip_mask.sum(), te_clip_mask.sum(),
                 tr_seg_mask.sum(), te_seg_mask.sum())

        t0 = time.time()
        # -- EEG branch --
        pred_eeg, probs_eeg, info_eeg = train_one_fold_with_probs(
            X_eeg[tr_seg_mask], y_seg_eeg[tr_seg_mask],
            X_eeg[te_seg_mask], y_seg_eeg[te_seg_mask], te_seg_clip_local,
            y_te_clip,
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, device=device,
            seed=args.seed + fold, tag="EEG")
        # -- Eye branch --
        pred_eye, probs_eye, info_eye = train_one_fold_with_probs(
            X_eye[tr_seg_mask], y_seg_eye[tr_seg_mask],
            X_eye[te_seg_mask], y_seg_eye[te_seg_mask], te_seg_clip_local,
            y_te_clip,
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, device=device,
            seed=args.seed + fold + 1000, tag="EYE")

        # -- Primary fusion (headline w_eeg) --
        probs_fused = w_e * probs_eeg + w_y * probs_eye
        pred_fused = probs_fused.argmax(1)

        acc_eeg, f1_eeg = evaluate(y_te_clip, pred_eeg)
        acc_eye, f1_eye = evaluate(y_te_clip, pred_eye)
        acc_fused, f1_fused = evaluate(y_te_clip, pred_fused)
        dt = time.time() - t0
        log.info("  fold %d  EEG=%.4f/%.4f  EYE=%.4f/%.4f  "
                 "FUSED(w=%.2f)=%.4f/%.4f  time=%.1fs",
                 fold + 1, acc_eeg, f1_eeg, acc_eye, f1_eye,
                 w_e, acc_fused, f1_fused, dt)

        # -- Weight sweep (post-hoc, same probs) --
        per_weight = {}
        for w in sweep_weights:
            pf = w * probs_eeg + (1.0 - w) * probs_eye
            pr = pf.argmax(1)
            acc_w, f1_w = evaluate(y_te_clip, pr)
            per_weight[f"{w:.2f}"] = {"acc": acc_w, "f1": f1_w,
                                      "y_pred": pr.tolist()}
        sweep_fold_records.append({"y_true": y_te_clip.tolist(),
                                   "per_weight": per_weight})

        fold_results_eeg.append(FoldResult(fold + 1, acc_eeg, f1_eeg,
                                           y_te_clip, pred_eeg))
        fold_results_eye.append(FoldResult(fold + 1, acc_eye, f1_eye,
                                           y_te_clip, pred_eye))
        fold_results_fused.append(FoldResult(fold + 1, acc_fused, f1_fused,
                                             y_te_clip, pred_fused))
        histories.append({
            "fold": fold + 1,
            "eeg": info_eeg, "eye": info_eye,
            "eeg_acc": acc_eeg, "eye_acc": acc_eye,
            "fused_acc": acc_fused,
        })

    sum_eeg = summarise(fold_results_eeg)
    sum_eye = summarise(fold_results_eye)
    sum_fus = summarise(fold_results_fused)

    # Aggregate sweep: for each weight, build per-fold FoldResult and summarise
    sweep_summary: dict[str, dict] = {}
    for w in sweep_weights:
        key = f"{w:.2f}"
        fold_rs: list[FoldResult] = []
        for fi, rec in enumerate(sweep_fold_records):
            y_true = np.asarray(rec["y_true"], dtype=np.int64)
            y_pred = np.asarray(rec["per_weight"][key]["y_pred"],
                                dtype=np.int64)
            a, f = evaluate(y_true, y_pred)
            fold_rs.append(FoldResult(fi + 1, a, f, y_true, y_pred))
        sweep_summary[key] = summarise(fold_rs)

    log.info("=" * 70)
    log.info("EEG    acc=%.4f±%.4f  f1=%.4f±%.4f",
             sum_eeg["acc_mean"], sum_eeg["acc_std"],
             sum_eeg["f1_mean"], sum_eeg["f1_std"])
    log.info("EYE    acc=%.4f±%.4f  f1=%.4f±%.4f",
             sum_eye["acc_mean"], sum_eye["acc_std"],
             sum_eye["f1_mean"], sum_eye["f1_std"])
    log.info("FUSED  acc=%.4f±%.4f  f1=%.4f±%.4f  (w_eeg=%.2f, headline)",
             sum_fus["acc_mean"], sum_fus["acc_std"],
             sum_fus["f1_mean"], sum_fus["f1_std"], w_e)
    log.info("-" * 70)
    log.info("Weight sweep  (w_eeg  ->  fused acc ± std / f1)")
    best_w, best_acc = None, -1.0
    for w in sweep_weights:
        s = sweep_summary[f"{w:.2f}"]
        log.info("  w_eeg=%.2f  acc=%.4f±%.4f  f1=%.4f",
                 w, s["acc_mean"], s["acc_std"], s["f1_mean"])
        if s["acc_mean"] > best_acc:
            best_acc = s["acc_mean"]; best_w = w
    log.info("BEST   w_eeg=%.2f  acc=%.4f", best_w, best_acc)

    summary = {
        "config": {**vars(args), "device": str(device)},
        "eeg": sum_eeg, "eye": sum_eye, "fused_headline": sum_fus,
        "weight_sweep": sweep_summary,
        "best_weight": {"w_eeg": best_w, "acc": best_acc},
        "histories": histories,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
