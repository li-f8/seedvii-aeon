"""Smoke test for the data loader. Runs only if SEED-VII is present."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seedvii.data import load_multimodal
from seedvii.data.aeon_format import de_to_aeon, eye_to_aeon

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "seed-VII"
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="SEED-VII data not found under data/seed-VII"
)


def test_load_single_subject():
    d = load_multimodal(subject_ids=[1])
    assert d["X_eeg"].ndim == 3 and d["X_eeg"].shape[1:] == (5, 62)
    assert d["X_eye"].ndim == 2 and d["X_eye"].shape[1] == 33
    assert d["X_eeg"].shape[0] == d["X_eye"].shape[0] == d["y"].shape[0]
    assert set(np.unique(d["y"])).issubset(set(range(7)))


def test_aeon_shape_conversion():
    d = load_multimodal(subject_ids=[1])
    Xe = de_to_aeon(d["X_eeg"])
    Xy = eye_to_aeon(d["X_eye"])
    assert Xe.shape[1:] == (62, 5)
    assert Xy.shape[1:] == (33, 1)
