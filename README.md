# seedvii-aeon

Multimodal EEG-based emotion recognition on **SEED-VII** using time series
classification methods from the [**aeon**](https://github.com/aeon-toolkit/aeon)
toolkit.

MSc dissertation project, University of Southampton.
Supervisor: Prof. Tony Bagnall.

## Quick start

```bash
# 1. create venv (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. install
pip install -e ".[dev]"

# 3. make sure SEED-VII data is at ./data/seed-VII (symlink ok)
ls data/seed-VII

# 4. run a minimal smoke test
python scripts/run_single.py --classifier multirocket --subjects 1 2 3
```

## Project layout

```
src/seedvii/        # importable package
  data/             # loading, preprocessing, aeon format conversion
  models/           # TSC classifier wrappers, fusion heads
  eval/             # CS / WS protocols, metrics
  utils/            # logging, seeding
scripts/            # experiment entry points
configs/            # YAML experiment configs
tests/              # unit tests
notebooks/          # exploration (not for production)
data/               # SEED-VII (gitignored, symlink ok)
results/            # logs / figures / checkpoints (gitignored)
```

## Classifier suite

- HIVE-COTE V2
- MultiROCKET + Hydra
- DrCIF
- Arsenal
- InceptionTime
- MAET Transformer (deep-learning baseline)

## Citation

Please acknowledge the aeon toolkit and Iridis HPC if publishing.
