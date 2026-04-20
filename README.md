# seedvii-aeon

Multimodal EEG-based emotion recognition on **SEED-VII** using time series
classification (TSC) methods from the [**aeon**](https://github.com/aeon-toolkit/aeon)
toolkit.

MSc dissertation project, University of Southampton — supervisor
Prof. Tony Bagnall.

## Motivation

Most published EEG emotion-recognition pipelines hand-craft features
(Differential Entropy, PSD, etc.) and feed them to an MLP / SVM.
This project asks: **how well do general-purpose TSC methods perform on
raw / lightly-processed EEG, compared to the DE + MLP baseline?**

SEED-VII is a natural testbed — 20 subjects, 80 video stimuli, 7 emotions
(Disgust / Fear / Sad / Neutral / Happy / Anger / Surprise), 62-channel EEG
plus 33-dim eye-tracking features.

## Current status

| Setting | Method | Features | Protocol | Acc (mean ± std) |
|---|---|---|---|---|
| CW2 baseline (PyTorch, external) | MLP | DE (flat) | WS, 1 subject | **47.96%** |
| This repo | MultiRocket + Hydra | raw EEG, 4 s window | WS, 1 subject | 25.48% ± 3.39% |
| This repo | MultiRocket + Hydra | DE sequence, 10 s | CS, 3 subjects | 20.11% |
| This repo | MultiRocket + Hydra | DE sequence, 10 s | WS, 3 subjects | 21.15% |

**Working hypothesis:** naïve raw-TSC underperforms DE-based pipelines on
EEG emotion because the informative structure lives in band-power
statistics, not in the raw waveform — a potentially useful negative
result for the TSC community.

Runtime reference: ~2 min / fold for MultiRocket+Hydra on 1 subject
(Mac M-series, CPU). Extrapolated HIVE-COTE V2 on 20 subjects CS
≈ 1–2 days single-machine.

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

# 4. smoke test
python scripts/run_single.py \
    --classifier multirocket_hydra \
    --subjects 1 2 3 \
    --features de_seq \
    --protocol cross_subject
```

### Common CLI flags

| Flag | Values | Notes |
|---|---|---|
| `--classifier` | `multirocket_hydra`, `multirocket`, `hydra`, `arsenal`, `drcif`, `hivecote_v2`, `inception_time` | see `src/seedvii/models/tsc_wrappers.py` |
| `--features` | `de`, `de_seq`, `raw` | `de_seq` recommended |
| `--modality` | `eeg`, `eye`, `eeg+eye` | eye / fusion only for `--features de` |
| `--protocol` | `cross_subject`, `within_subject` | WS loops per subject internally |
| `--win-sec` | float (raw) / int (de_seq) | window length |
| `--n-folds` | int | CV folds |

## Project layout

```
src/seedvii/        # importable package
  data/             # loaders (DE / DE-seq / raw EEG / eye), aeon adapters
  models/           # aeon classifier factory
  eval/             # CS grouped + WS video-stratified splitters, metrics
  utils/            # logging, seeding
scripts/
  run_single.py     # single-classifier experiment runner
  inspect_raw.py    # one-off .mat structure inspector
configs/            # YAML experiment templates
tests/              # smoke tests (skip if data absent)
data/               # SEED-VII — gitignored, obtain from BCMI Lab
results/            # logs / figures — gitignored
```

## Evaluation protocols

- **Cross-subject (CS):** `GroupKFold` by subject ID — test set never
  sees training subjects. Default 5 folds.
- **Within-subject (WS):** for each subject independently, stratified
  split by video ID (no video leak between train/test), then
  `StratifiedKFold` over videos. Mean over subjects.

## Known issues (upstream aeon)

- `aeon.classification.convolution_based.HydraClassifier` crashes on
  short series: when `n_timepoints < 9`, the internal dilations list is
  empty and `torch.cat([])` fails. Reproducible with DE-as-timeseries
  (`n_timepoints=5`). Fix candidate: raise a clear `ValueError` in
  `_HydraInternal.__init__` — happy to open a PR.

## Data

SEED-VII is released by the BCMI Lab, SJTU under their own EULA and is
**not redistributed in this repository**. Request access at
<https://bcmi.sjtu.edu.cn/home/seed/> and place the extracted folders
under `data/seed-VII/`.

## Acknowledgements

- aeon-toolkit developers
- BCMI Lab, SJTU, for SEED-VII
- University of Southampton (IRIDIS HPC, if used in final experiments)

