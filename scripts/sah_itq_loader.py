#!/usr/bin/env python3
"""SAH Stage 9: ITQ Rotation Loader — loads real 512-bit ITQ model."""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List


ROOT = Path("/Users/mac/yellow_phoenix")
ITQ_PATH = Path(os.environ.get("SAH_ITQ_PATH", ROOT / "itq_model_512.npz"))


def load_rotation_matrix(path: Optional[Path] = None) -> List[List[float]]:
    """Load ITQ rotation matrix from npz. Returns 512×512 nested list."""
    p = path or ITQ_PATH
    if not p.exists():
        # Fallback to data/ directory
        p = ROOT / "data" / "itq_model_512.npz"
        if not p.exists():
            raise FileNotFoundError(f"ITQ model not found: {path or ITQ_PATH}")

    data = np.load(p)
    if "R" in data:
        rot = data["R"]
    elif "rotation" in data:
        rot = data["rotation"]
    else:
        keys = list(data.keys())
        raise KeyError(f"No 'R' or 'rotation' key in {p}. Keys: {keys}")

    rot = rot.astype(np.float32)
    if rot.shape != (512, 512):
        raise ValueError(f"Expected (512, 512), got {rot.shape}")

    return rot.tolist()


def has_itq_model(path: Optional[Path] = None) -> bool:
    """Check if ITQ model file exists."""
    p = path or ITQ_PATH
    if p.exists():
        return True
    return (ROOT / "data" / "itq_model_512.npz").exists()


def get_fallback_rotation(dim: int = 512) -> List[List[float]]:
    """Return identity matrix as fallback if ITQ not available."""
    return np.eye(dim, dtype=np.float32).tolist()
