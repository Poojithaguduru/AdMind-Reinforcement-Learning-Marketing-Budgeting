from __future__ import annotations
import numpy as np

def fixed_split(n_channels: int) -> np.ndarray:
    return np.ones(n_channels) / n_channels
