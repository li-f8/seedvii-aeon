"""Generate the canonical SEED-VII 4-fold video→fold mapping JSON.

Source: emotion_label_and_stimuli_order.xlsx (SEED-VII official).
Rule:   fold_id(video_id) = ((video_id - 1) % 20) // 5      (video_id in 1..80)

This mapping is the *single source of truth* for both Phase 1 (aeon via
run_ts.py --protocol lovo) and Phase 2 (PyTorch deep learning). Both
pipelines must load and split by this exact partition so results are
directly comparable.

Usage
-----
    python scripts/make_folds_json.py \\
        --xlsx ~/Desktop/.../seed-VII/emotion_label_and_stimuli_order.xlsx \\
        --out data/seedvii_folds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True,
                   help="path to emotion_label_and_stimuli_order.xlsx")
    p.add_argument("--out", default="data/seedvii_folds.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.xlsx, sheet_name=0, header=None)

    # Rows: 0=vid idx s1, 1=emo s1, 2=vid idx s2, 3=emo s2, ...
    videos: list[dict] = []
    for sess in range(4):
        ids = df.iloc[sess * 2, 1:].tolist()
        emos = df.iloc[sess * 2 + 1, 1:].tolist()
        for pos, (vid, emo) in enumerate(zip(ids, emos)):
            videos.append({
                "video_id": int(vid),
                "session": sess + 1,
                "pos_in_session": pos,      # 0..19
                "emotion": str(emo),
                "fold": pos // 5,           # 0..3, official 4-fold
            })

    assert len(videos) == 80, f"expected 80 videos, got {len(videos)}"
    assert sorted(v["video_id"] for v in videos) == list(range(1, 81))

    # Per-fold video lists (convenient for training code)
    folds: dict[str, list[int]] = {f"fold_{f}": [] for f in range(4)}
    for v in videos:
        folds[f"fold_{v['fold']}"].append(v["video_id"])

    # Emotion → int label (aligned with src/seedvii/data/loader.py EMOTIONS)
    from seedvii.data.loader import _EMOTION_MAP

    for v in videos:
        v["label"] = int(_EMOTION_MAP[v["emotion"]])

    out = {
        "source": "SEED-VII emotion_label_and_stimuli_order.xlsx",
        "rule": "fold_id(video_id) = ((video_id - 1) % 20) // 5",
        "n_folds": 4,
        "n_videos": 80,
        "label_map": _EMOTION_MAP,
        "folds": folds,
        "videos": videos,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")

    # Sanity print
    from collections import Counter
    for f in range(4):
        emos = [v["emotion"] for v in videos if v["fold"] == f]
        print(f"  fold {f}: n={len(emos)}  {dict(Counter(emos))}")


if __name__ == "__main__":
    main()
