from __future__ import annotations
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    s = e / e.sum()
    return s

def clip_proportions(p: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    p = np.clip(p, mins, maxs)
    total_min = mins.sum()
    if total_min > 1.0 + 1e-6:
        return mins / total_min
    rem = max(0.0, 1.0 - total_min)
    headroom = maxs - mins
    headroom_sum = headroom.sum()
    if headroom_sum <= 1e-9:
        return mins
    frac = headroom / headroom_sum
    return mins + rem * frac
