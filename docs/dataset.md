# SEED-VII Dataset

Background material for the MSc thesis. This document describes the raw
SEED-VII release and summarises the reduced versions used in this project.

## Source and licence

- **Name:** SEED-VII
- **Provider:** BCMI Lab, Shanghai Jiao Tong University
- **Web:** <https://bcmi.sjtu.edu.cn/home/seed/>
- **Access:** End-User Licence Agreement (EULA); not redistributed in this
  repository.
- **Reference paper:** Wei-Bang Jiang, Xuan-Hao Liu, Wei-Long Zheng and
  Bao-Liang Lu. *SEED-VII: A Multimodal Dataset of Six Basic Emotions
  with Continuous Labels for Emotion Recognition.* IEEE Transactions on
  Affective Computing, vol. 16, no. 2, pp. 969–985, April–June 2025.
  doi: [10.1109/TAFFC.2024.3485057](https://doi.org/10.1109/TAFFC.2024.3485057)

## Subjects

- **Total subjects:** 20 healthy adults.
- **Sex balance:** 10 male, 10 female.
- **Age range:** 19 – 26 years (mean ≈ 22.5).
- Each subject has accompanying Big-Five-style personality scores
  (E / P / N / L values) in `subject info.xlsx`; not used in this project.
- All 20 subjects completed all four experimental sessions.

## Stimuli and sessions

Each subject watched **80 video clips** distributed over **4 sessions**
(20 clips per session), drawn from Chinese cinema / TV / documentaries /
YouTube magic-show footage. Each clip is labelled with one of seven
discrete emotions.

### Class labels

| Index | Emotion |
|-------|---------|
| 0 | Disgust |
| 1 | Fear |
| 2 | Sad |
| 3 | Neutral |
| 4 | Happy |
| 5 | Anger |
| 6 | Surprise |

### Session composition

Each session uses a fixed 5-emotion subset, with each emotion appearing
in 4 clips (2 unique clips × 2 repetitions within the session):

| Session | Emotions present |
|---------|------------------|
| 1 | Happy, Neutral, Disgust, Sad, Anger |
| 2 | Anger, Sad, Fear, Neutral, Surprise |
| 3 | Happy, Surprise, Disgust, Fear, Anger |
| 4 | Disgust, Sad, Fear, Surprise, Happy |

This design means **every emotion appears in at least 2 sessions**, so
within-subject evaluation can still see all 7 classes even when one
session is held out.

### Clip durations

Clip lengths vary from ~1 minute to ~6 minutes, averaging
≈ 3 minutes per clip. Per subject this yields roughly
**80 clips × 3 min ≈ 4 hours of emotional EEG recording**,
giving ≈ 77 hours of EEG across all 20 subjects.

## Recording hardware and preprocessing

### EEG

- **Channels:** 62 (extended international 10-20 layout;
  channel order in `Channel Order.xlsx`).
- **Original sampling rate:** 1000 Hz (ESI NeuroScan).
- **Released sampling rate:** 200 Hz (downsampled).
- **Provided preprocessing** (`EEG_preprocessed/`):
  - 1 – 75 Hz band-pass filter, 50 Hz notch removed
  - EOG artefact rejection
  - Stored as `(62, T)` `double` arrays per clip, one MATLAB `.mat`
    file per subject with keys `"1"` … `"80"`.

### Eye tracking

- **Device:** SMI ETG 2w head-mounted eye tracker.
- **Features provided** (`EYE_features/`): 33 hand-crafted features
  (pupil diameter, blink / fixation / saccade / event statistics),
  one value per second, aligned to the EEG timeline.

### Pre-computed DE features

- **Location:** `EEG_features/`, per-subject `.mat` files with keys
  `de_LDS_{vid}` for vid = 1 … 80.
- **Shape:** `(T_seconds, 5_bands, 62_channels)` per clip.
- **Bands:** δ (1 – 4 Hz), θ (4 – 8 Hz), α (8 – 14 Hz),
  β (14 – 31 Hz), γ (31 – 50 Hz).
- **Smoothing:** Linear Dynamic System (LDS) applied per band and
  channel — this is the "DE-LDS" variant used as the standard
  baseline feature in the SEED dataset family.

## Reduced / derived datasets used in this project

The loader (`src/seedvii/data/loader.py`) produces three alternative
aeon-formatted views of the data, all in the shape convention
`(n_cases, n_channels, n_timepoints)`:

### 1. DE (flat, one second per case)

- Case = one second from one clip.
- Shape: `(62, 5)` — 62 EEG channels × 5 frequency bands.
- Per-subject z-score normalisation applied by default.
- Matches the standard "DE + MLP" baseline used in prior SEED work.

### 2. DE sequence (windowed DE over several seconds)

- Case = a `win_sec`-second window sliding over the DE sequence of
  one clip (non-overlapping by default).
- Shape: `(310, win_sec)` — each of the 62 × 5 = 310 channel × band
  combinations becomes an aeon "channel"; `win_sec` is the time axis.
- Typical `win_sec = 10`.

### 3. Raw EEG (windowed raw signal)

- Case = a `win_sec`-second window sliding over the preprocessed raw
  EEG of one clip.
- Shape: `(62, win_sec × 200)` — 62 channels, `200 Hz` sampling.
- `win_sec = 4` → series length 800. This is the view closest to a
  generic TSC problem.

### Concrete example: 5-subject reduced set, raw 4 s windows

| Quantity | Value |
|----------|-------|
| Subjects | 5 (IDs 1 – 5) |
| Channels | 62 |
| Sampling rate | 200 Hz |
| Window length | 4 s (`n_timepoints = 800`) |
| Total cases | 17 435 |
| Cases per class | Disgust 2310, Fear 2770, Sad 3085, Neutral 1660, Happy 2025, Anger 2740, Surprise 2845 |
| Class balance | min/max ≈ 0.54 (Neutral under-represented) |

### Concrete example: 5-subject reduced set, DE sequence (10 s)

| Quantity | Value |
|----------|-------|
| Subjects | 5 (IDs 1 – 5) |
| Channels | 310 (62 × 5 band) |
| Window length | 10 s (`n_timepoints = 10`) |
| Total cases | ≈ 1 400 (40% fewer than raw because each "case" covers 10 s instead of 4 s) |

## Evaluation protocols

Two evaluation protocols are implemented in `src/seedvii/eval/protocols.py`:

1. **Cross-subject (CS).** `GroupKFold` with subject ID as the group
   — the test fold never contains any subject seen in training.
   This is the setting of primary interest for the project: **performance
   on unseen subjects**. Default 5 folds.

2. **Within-subject (WS).** For each subject independently, stratified
   5-fold split **by video ID** (so no clip appears in both train and
   test for that subject). Results are averaged across subjects. Useful
   as a loose upper bound but not the target of the project.

## Preliminary numbers (reduced set)

First-pass results on 5 subjects, within-subject protocol, 5-fold CV,
per-subject z-score applied.

| Method | Feature view | Acc (mean ± std) | F1 (mean ± std) |
|--------|--------------|------------------|-----------------|
| MultiRocket + Hydra | raw EEG, 4 s window | 0.287 ± 0.096 | 0.272 ± 0.084 |
| MultiRocket + Hydra | DE sequence, 10 s | ~0.21 (3-subject smoke test) | — |

Chance level for 7 balanced classes is 1 / 7 ≈ 0.143, so all runs are
comfortably above chance, but the large standard deviation (±0.096 on
5 subjects × 5 folds) highlights the per-subject variance problem and
motivates moving to the cross-subject protocol on the full 20-subject
cohort as the main experimental setting.
