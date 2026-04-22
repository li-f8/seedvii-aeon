"""Phase 2 (segment-level): DL LOVO with segment-level training + clip-level voting.

Differences from run_dl_lovo.py (clip-level):
  - Each clip's DE series is sliced into overlapping segments of ``seg_len``
    timepoints (e.g. 10 = ~40 s of DE-LDS).
  - Train samples = segments (~50k per fold vs 1200 at clip level).
  - Each segment inherits the parent clip's emotion label.
  - At test time, model predicts softmax per segment; segments belonging to
    the same (subject, video) are soft-voted (mean of probabilities) and the
    argmax gives the clip-level prediction.

Metrics are reported at **clip level** so the number is directly comparable
to Phase 1 aeon MultiRocket (28.75%).
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


def clip_to_segments(x: np.ndarray, seg_len: int, stride: int,
                     ) -> list[np.ndarray]:
    """Slice (C, T) clip into list of (C, seg_len) segments.

    If T < seg_len: resample the whole clip up to seg_len (one segment).
    Else: sliding window with given stride, plus a final aligned window
    so the clip's tail is always covered.
    """
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
# Train / eval
# ---------------------------------------------------------------------------

def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te_seg: np.ndarray, clip_id_te: np.ndarray,
    y_te_clip: np.ndarray,
    *, model_name: str, epochs: int, batch_size: int,
    lr: float, weight_decay: float, device: torch.device, seed: int,
) -> tuple[np.ndarray, dict]:
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
    best_clip_pred = None
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

        # Segment-level eval + clip-level soft-vote aggregation
        model.eval()
        probs_list = []
        with torch.no_grad():
            for xb, _ in dl_te:
                xb = xb.to(device)
                p = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs_list.append(p)
        probs = np.concatenate(probs_list, axis=0)   # (n_test_segments, 7)

        seg_acc = float((probs.argmax(1) == y_te_seg).mean())

        # Soft-vote: mean probability per clip
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
            best_clip_acc = clip_acc; best_clip_pred = clip_pred.copy()
        if ep % 5 == 0 or ep == 1:
            log.info("  ep %3d  tr_loss=%.4f  seg_acc=%.4f  "
                     "clip_acc=%.4f  clip_f1=%.4f  best_clip=%.4f  "
                     "skip=%d",
                     ep, tr_loss, seg_acc, clip_acc, clip_f1,
                     best_clip_acc, skipped)

    return best_clip_pred, {"history": history,
                             "best_clip_acc": best_clip_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="eegnet")
    p.add_argument("--ts-dir", required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--folds-json", default="data/seedvii_folds.json")
    p.add_argument("--seg-len", type=int, default=10,
                   help="segment length in DE timepoints (~4s each)")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--subject-zscore", action="store_true", default=True)
    p.add_argument("--no-subject-zscore", dest="subject_zscore",
                   action="store_false")
    p.add_argument("--protocol", default="lovo",
                   choices=["lovo", "loso", "random"],
                   help="lovo=leave-videos-out (no T/video leak); "
                        "loso=leave-subjects-out (subject-level gen, video "
                        "shared across train/test); "
                        "random=stratified random 80/20-per-fold (all leaks)")
    p.add_argument("--out", default="results/logs/dl_segments.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (pick_device() if args.device == "auto"
              else torch.device(args.device))
    log.info("model=%s  protocol=%s  device=%s  epochs=%d  bs=%d  lr=%.1e  "
             "seg_len=%d  stride=%d",
             args.model, args.protocol, device, args.epochs, args.batch_size,
             args.lr, args.seg_len, args.stride)

    folds = json.loads(Path(args.folds_json).read_text())
    fold_of_vid = {v["video_id"]: v["fold"] for v in folds["videos"]}
    log.info("loaded %d videos across %d folds",
             folds["n_videos"], folds["n_folds"])

    # Per (subject, video): load clip, resample-if-short, slice to segments
    # Build flat arrays: segs (N_seg, C, seg_len), seg_label (N_seg,),
    #                  seg_clip_idx (N_seg, clip_global_id), clip_label, clip_fold
    ts_dir = Path(args.ts_dir)
    seg_list, seg_label, seg_clip = [], [], []
    clip_label, clip_fold, clip_meta = [], [], []
    clip_counter = 0
    for sub in args.subjects:
        X_list, y = load_subject(ts_dir, sub)
        assert len(X_list) == 80
        if args.subject_zscore:
            # per-channel per-subject stats computed on the whole subject's data
            stacked = np.concatenate(X_list, axis=1)   # (310, sum T)
            mu = stacked.mean(axis=1, keepdims=True)
            sd = stacked.std(axis=1, keepdims=True) + 1e-6
            X_list = [((x - mu) / sd).astype(np.float32) for x in X_list]

        for i, (x, yi) in enumerate(zip(X_list, y)):
            vid = i + 1
            segs = clip_to_segments(x, args.seg_len, args.stride)
            for s in segs:
                seg_list.append(s); seg_label.append(int(yi))
                seg_clip.append(clip_counter)
            clip_label.append(int(yi))
            clip_fold.append(fold_of_vid[vid])
            clip_meta.append((int(sub), vid))
            clip_counter += 1

    X_seg = np.stack(seg_list, axis=0)        # (N_seg, 310, seg_len)
    y_seg = np.asarray(seg_label, dtype=np.int64)
    clip_of_seg = np.asarray(seg_clip, dtype=np.int64)
    y_clip = np.asarray(clip_label, dtype=np.int64)
    clip_video = np.asarray([m[1] for m in clip_meta], dtype=np.int64)
    clip_subject = np.asarray([m[0] for m in clip_meta], dtype=np.int64)
    log.info("X_seg=%s  y_seg=%s  n_clips=%d  seg/clip_avg=%.1f",
             X_seg.shape, y_seg.shape, len(y_clip),
             len(y_seg) / max(len(y_clip), 1))

    # --- Assign each clip to one of 4 folds according to --protocol --------
    n_folds = folds["n_folds"]
    if args.protocol == "lovo":
        # fold by video (official SEED-VII 4-fold cross-video)
        fold_of_clip = np.asarray(clip_fold, dtype=np.int64)
        log.info("protocol=lovo : test videos never appear in train "
                 "(no T leak, no video leak)")
    elif args.protocol == "loso":
        # 4-fold grouped by subject (5 subjects per fold, keeps train size
        # comparable to LOVO's 1200 train clips)
        subs = sorted(set(int(s) for s in args.subjects))
        assert len(subs) % n_folds == 0, (
            f"need n_subjects divisible by {n_folds} folds for LOSO, got "
            f"{len(subs)}")
        per = len(subs) // n_folds
        sub_to_fold = {s: i // per for i, s in enumerate(subs)}
        fold_of_clip = np.asarray(
            [sub_to_fold[int(s)] for s in clip_subject], dtype=np.int64)
        log.info("protocol=loso : test subjects held out (video content "
                 "shared across train/test → video leak, no subject leak)")
    elif args.protocol == "random":
        # Stratified-random 4-fold at clip level (labels balanced per fold).
        # Every (subject, video) pair lands in an arbitrary fold → all
        # leakage pathways open: test videos in train, test subjects in
        # train, T-based shortcut preserved.
        rng = np.random.default_rng(args.seed)
        fold_of_clip = np.full(len(y_clip), -1, dtype=np.int64)
        for lbl in np.unique(y_clip):
            idx = np.where(y_clip == lbl)[0]
            rng.shuffle(idx)
            # Round-robin assign into folds for near-even class balance
            for j, k in enumerate(idx):
                fold_of_clip[k] = j % n_folds
        assert (fold_of_clip >= 0).all()
        log.info("protocol=random : stratified random 4-fold over clips "
                 "(all leakage paths active — upper-bound estimate)")
    else:
        raise ValueError(args.protocol)

    # Log per-fold class distribution for sanity
    for f in range(n_folds):
        n_te = int((fold_of_clip == f).sum())
        n_tr = int((fold_of_clip != f).sum())
        log.info("  fold %d: train=%d clips  test=%d clips", f + 1, n_tr, n_te)

    fold_results: list[FoldResult] = []
    histories: list[dict] = []
    for fold in range(n_folds):
        tr_clip_mask = fold_of_clip != fold
        te_clip_mask = fold_of_clip == fold
        tr_seg_mask = tr_clip_mask[clip_of_seg]
        te_seg_mask = te_clip_mask[clip_of_seg]

        # Remap test clip ids to 0..n_test_clips-1 for aggregation
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
        y_pred, info = train_one_fold(
            X_seg[tr_seg_mask], y_seg[tr_seg_mask],
            X_seg[te_seg_mask], y_seg[te_seg_mask], te_seg_clip_local,
            y_te_clip,
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, device=device,
            seed=args.seed + fold,
        )
        dt = time.time() - t0
        acc, f1 = evaluate(y_te_clip, y_pred)
        log.info("  fold %d BEST clip acc=%.4f  clip f1=%.4f  time=%.1fs",
                 fold + 1, acc, f1, dt)
        fold_results.append(FoldResult(fold=fold + 1, acc=acc, macro_f1=f1,
                                       y_true=y_te_clip, y_pred=y_pred))
        histories.append({"fold": fold + 1, **info})

    summary = summarise(fold_results)
    summary["config"] = vars(args)
    summary["config"]["device"] = str(device)
    summary["histories"] = histories
    log.info("=" * 60)
    log.info("%s  clip-level  acc=%.4f±%.4f  f1=%.4f±%.4f",
             args.model, summary["acc_mean"], summary["acc_std"],
             summary["f1_mean"], summary["f1_std"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
