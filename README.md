# seedvii-aeon

Multimodal emotion recognition on **SEED-VII** — classical time-series
classifiers from [**aeon**](https://github.com/aeon-toolkit/aeon),
deep-learning baselines (DECNN / EEGNet), and EEG + eye-movement late
fusion, all evaluated under three protocols with a shared 4-fold
cross-video partition.

MSc dissertation project, University of Southampton — supervisor
Prof. Tony Bagnall.

## What this repo contains

- **Phase 1 — Classical TSC baselines**: MultiRocket, Hydra, Arsenal,
  DrCIF, MR-Hydra on DE-flat EEG features under LOVO.
- **Phase 2 — Deep learning baselines**: compact 1-D CNN on DE-flat
  (`DECNN`) and a DE-reshaped EEGNet (`DEEEGNet`). Segment-level
  training with clip-level soft-vote aggregation. GroupNorm throughout
  for MPS stability.
- **Phase 3 — Multi-modal late fusion**: DECNN on EEG (310ch DE-LDS) +
  DECNN on eye-tracking (33ch), per-clip probability weighted average
  with a post-hoc weight sweep.
- **Three evaluation protocols**: LOVO (cross-video, default),
  LOSO (cross-subject), and stratified random split.
- **Diagnostics**: z-score ablation, segment-length sweep, and a
  clip-duration-only 1-NN baseline that exposes a stimulus-duration
  confound under LOSO.

## Headline results

### Modality × protocol (DECNN, seg_len = 5, clip-level)

| Modality            | LOVO                         | LOSO                         | Random                       |
|:--------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| EEG (DE, 310ch)     | 37.00 ± 2.42                 | 29.12 ± 1.82                 | 33.56 ± 1.25                 |
| Eye (33ch)          | 35.75 ± 4.48                 | 40.69 ± 1.28                 | 42.19 ± 2.82                 |
| EEG + Eye fused     | **41.62 ± 3.92** (w = 0.40)  | **42.62 ± 2.18** (w = 0.20)  | **45.12 ± 2.44** (w = 0.30)  |

*Fusion: per-clip probability weighted average, `w_eeg` selected by
test accuracy (sweep over `{0.0, 0.1, …, 1.0}`).*

### Stimulus-duration confound (LOSO)

- A 1-nearest-neighbour classifier using **clip duration alone** reaches
  **67.50%** under LOSO — an indicator that SEED-VII's fixed 80-video
  stimulus design, combined with label assignment per video, leaks
  label information via clip length when stimuli are shared between
  train and test subjects.

### Phase 1 — classical aeon baselines (LOVO, EEG DE, L = 90)

| Classifier          | Accuracy        |
|:--------------------|:----------------|
| MultiRocket         | 28.75 ± 4.52    |
| MultiRocket + Hydra | 27.81 ± 4.54    |
| Hydra               | 24.75 ± 6.12    |
| Arsenal             | 20.94 ± 3.20    |

### Per-subject z-score ablation (DECNN)

| Protocol | with z-score | without z-score |      Δ |
|:---------|:-------------|:----------------|-------:|
| LOVO     | 37.38%       | 15.19%          | −22.19 |
| LOSO     | 29.00%       | 15.00%          | −14.00 |
| Random   | 33.44%       | 15.00%          | −18.44 |

Per-subject z-score normalisation contributes ~15–22 pp across protocols
— the dominant learnable signal in this pipeline.

### Segment-length sweep (DECNN + GroupNorm, LOVO)

| seg_len | Accuracy     |
|:--------|:-------------|
| 3       | 36.62 ± 2.96 |
| **5**   | **37.38 ± 2.81** |
| 8       | 37.12 ± 3.30 |
| 10      | 35.62 ± 2.90 |
| 15      | 34.31 ± 3.46 |
| 20      | 34.00 ± 2.53 |

Full tables and publication figures live in `results/summary/`.

## Quick start

```bash
# 1. create venv (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. install
pip install -e ".[dev]"

# 3. put SEED-VII at ./data/seed-VII (symlink is fine)
#    expected layout:
#      data/seed-VII/EEG_preprocessed/{1..20}.mat
#      data/seed-VII/EEG_features/{1..20}.mat
#      data/seed-VII/EYE_features/{1..20}.mat
ls data/seed-VII

# 4. build the shared 4-fold video partition once
python scripts/make_folds_json.py

# 5. export SEED-VII into aeon .ts format
python scripts/to_ts.py --subjects {1..20} --features de_flat
python scripts/to_ts.py --subjects {1..20} --features eye
```

### Running experiments

```bash
# Phase 1 — classical TSC baseline (LOVO)
python scripts/run_ts.py \
    --classifier multirocket_hydra \
    --features de_flat --resample 90

# Phase 2 — DECNN segment-level (LOVO / LOSO / Random)
python scripts/run_dl_lovo_segments.py \
    --model decnn --seg-len 5 --protocol lovo \
    --subjects {1..20}

# Phase 3 — EEG + Eye late fusion with weight sweep
python scripts/run_dl_fusion.py \
    --protocol random --seg-len 5 --subjects {1..20}

# Diagnostics
python scripts/diag_length_leak.py           # T-only 1-NN leakage ceiling
python scripts/summarize_results.py          # rebuild tables + figures
```

## Project layout

```
src/seedvii/
  data/             # loaders (DE / DE-seq / raw EEG / eye), aeon adapters
  models/
    tsc_wrappers.py # aeon classifier factory
    dl.py           # DECNN + DEEEGNet (PyTorch, GroupNorm)
  eval/             # CS / WS / cross-video splitters, metrics
  utils/            # logging, seeding
scripts/
  to_ts.py                     # export to aeon .ts format (raw / de_flat / eye)
  make_folds_json.py           # 4-fold video partition (shared by all phases)
  run_ts.py                    # Phase 1: classical TSC runner
  run_single.py                # legacy single-classifier runner
  run_dl_lovo.py               # Phase 2: DL clip-level
  run_dl_lovo_segments.py      # Phase 2: DL segment + clip-vote
  run_dl_fusion.py             # Phase 3: EEG + Eye late fusion + weight sweep
  diag_length_leak.py          # T-only 1-NN duration confound diagnostic
  summarize_results.py         # aggregates results/logs/*.json → tables + figures
  inspect_raw.py               # one-off .mat structure inspector
results/
  logs/         # per-experiment JSON + text logs (gitignored)
  figures/      # publication figures (gitignored)
  summary/      # main_table.csv / .md (tracked)
data/           # SEED-VII — gitignored, obtain from BCMI
```

## Evaluation protocols

- **LOVO** (cross-video, default): shared 4-fold partition in
  `data/seedvii_folds.json`; `fold_id = ((video_id − 1) % 20) // 5`.
  Test videos never appear in train. All subjects are pooled.
- **LOSO** (cross-subject): 19 training subjects, 1 test subject.
  Stimuli are shared between train and test — see the duration-confound
  note above.
- **Random**: stratified random 4-fold split over clips, ignoring both
  subject and video identity.

## Notes on implementation

- **GroupNorm over BatchNorm**: both `DECNN` and `DEEEGNet` use
  `GroupNorm` rather than `BatchNorm2d/1d`. On Apple Silicon MPS, the
  depthwise `Conv2d(F1, F1*D, kernel_size=(62, 1), groups=F1)` inside
  EEGNet produces NaNs in a large fraction of batches under BN; switching
  to GN eliminates the training collapse.
- **Segment-level training, clip-level evaluation**: each clip's DE
  sequence is sliced into `seg_len`-timepoint windows (overlap = 0 by
  default); segments inherit the parent clip's label. At test time,
  softmax probabilities of all segments belonging to the same
  `(subject, video)` are averaged before `argmax`.
- **Per-subject z-score normalisation** is applied to every feature type
  before any split — removing it collapses accuracy to chance on all
  protocols (see table above).

## Data

SEED-VII is released by the BCMI Lab, SJTU under their own EULA and is
**not redistributed in this repository**. Request access at
<https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html> and place the
extracted folders under `data/seed-VII/`.

The dataset is introduced in:
Jiang W-B, Liu X-H, Zheng W-L, Lu B-L.
*SEED-VII: A Multimodal Dataset of Six Basic Emotions With Continuous
Labels for Emotion Recognition.* IEEE Transactions on Affective
Computing, 16(2): 969–985, April–June 2025.

## Known issues

- `aeon.classification.convolution_based.HydraClassifier` crashes on
  short series: when `n_timepoints < 9`, the internal dilations list is
  empty and `torch.cat([])` fails. Reproducible with DE-as-timeseries at
  `n_timepoints = 5`. Fix candidate: raise a clear `ValueError` in
  `_HydraInternal.__init__`.

## Acknowledgements

- aeon-toolkit developers
- BCMI Lab, SJTU, for releasing SEED-VII
- University of Southampton
