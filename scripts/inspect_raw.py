"""One-off: inspect the structure of an EEG_preprocessed/*.mat file.

Run from project root:
    python scripts/inspect_raw.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = REPO_ROOT / "data" / "seed-VII" / "EEG_preprocessed" / "1.mat"


def main() -> None:
    print(f"Inspecting: {MAT_PATH}")
    print(f"Exists: {MAT_PATH.exists()}")
    if not MAT_PATH.exists():
        # fall back: list the directory so the user can see what IS there
        parent = MAT_PATH.parent
        print(f"Contents of {parent}:")
        if parent.exists():
            for p in sorted(parent.iterdir())[:20]:
                print(" ", p.name)
        else:
            print("  (directory does not exist)")
        return

    mat = sio.loadmat(str(MAT_PATH))
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"# keys (non-underscore): {len(keys)}")
    for k in keys[:15]:
        v = mat[k]
        shape = v.shape if isinstance(v, np.ndarray) else type(v).__name__
        dtype = v.dtype if isinstance(v, np.ndarray) else "-"
        print(f"  {k:30s}  shape={shape}  dtype={dtype}")
    if len(keys) > 15:
        print(f"  ... and {len(keys) - 15} more")


if __name__ == "__main__":
    main()
