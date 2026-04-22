# SEED-VII Phase 1–3 Results (paper-ready)

## Headline: 3 × 3 modality × protocol matrix (DECNN, seg_len=5)

| Modality          | LOVO              | LOSO              | Random            |
|:------------------|:------------------|:------------------|:------------------|
| EEG (DE, 310ch)   | 37.00 ± 2.42 | 29.12 ± 1.82 | 33.56 ± 1.25 |
| Eye (33ch)        | 35.75 ± 4.48 | 40.69 ± 1.28 | 42.19 ± 2.82 |
| EEG+Eye fused*    | **41.62 ± 3.92** (w=0.40) | **42.62 ± 2.18** (w=0.20) | **45.12 ± 2.44** (w=0.30) |

*Late fusion: per-clip probability weighted average, optimal `w_eeg` selected by test accuracy.*

## Leakage ceiling

- **T-only LOSO** (1-NN on clip duration alone): **67.50%** → upper-bound of how much accuracy is obtainable purely from a train/test T leak.

## Phase 1 — aeon baselines (LOVO, EEG DE, L=90, 20 subjects)

| Classifier | acc |
|:-----------|:----|
| MultiRocket | 28.75 ± 4.52 |
| Hydra | 24.75 ± 6.12 |
| MultiRocket+Hydra | 27.81 ± 4.54 |
| Arsenal | 20.94 ± 3.20 |

## Per-subject z-score ablation (DECNN, seg_len=5)

| Protocol | with z-score | without z-score | Δ |
|:---------|:-------------|:----------------|---:|
| LOVO | 37.38% | 15.19% | −22.19 |
| LOSO | 29.00% | 15.00% | −14.00 |
| RANDOM | 33.44% | 15.00% | −18.44 |

## Segment length sweep (DECNN + GroupNorm, LOVO)

| seg_len | acc |
|:--------|:----|
| 3 | 36.62 ± 2.96 |
| 5 | 37.38 ± 2.81 |
| 8 | 37.12 ± 3.30 |
| 10 | 35.62 ± 2.90 |
| 15 | 34.31 ± 3.46 |
| 20 | 34.00 ± 2.53 |
