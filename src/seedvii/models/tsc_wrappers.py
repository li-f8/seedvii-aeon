"""Factory for aeon time series classifiers used in the benchmark.

Imports are done lazily inside build_classifier so the package is import-safe
even if some heavy deep-learning deps (e.g. tensorflow for InceptionTime) are
not installed in the current environment.
"""
from __future__ import annotations

from typing import Any

CLASSIFIERS = [
    "multirocket_hydra",
    "multirocket",
    "hydra",
    "arsenal",
    "drcif",
    "hivecote_v2",
    "inception_time",
]


def build_classifier(name: str, n_jobs: int = -1, random_state: int = 42, **kwargs: Any):
    """Instantiate an aeon classifier by short name.

    Parameters
    ----------
    name : one of CLASSIFIERS
    n_jobs : parallelism for kernel/ensemble methods
    random_state : seed for reproducibility
    kwargs : forwarded to the underlying aeon class
    """
    name = name.lower()

    if name == "multirocket_hydra":
        from aeon.classification.convolution_based import MultiRocketHydraClassifier

        return MultiRocketHydraClassifier(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "multirocket":
        from aeon.classification.convolution_based import MultiRocketClassifier

        return MultiRocketClassifier(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "hydra":
        from aeon.classification.convolution_based import HydraClassifier

        return HydraClassifier(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "drcif":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "hivecote_v2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(n_jobs=n_jobs, random_state=random_state, **kwargs)

    if name == "inception_time":
        from aeon.classification.deep_learning import InceptionTimeClassifier

        return InceptionTimeClassifier(random_state=random_state, **kwargs)

    raise ValueError(f"Unknown classifier '{name}'. Choose from {CLASSIFIERS}.")
