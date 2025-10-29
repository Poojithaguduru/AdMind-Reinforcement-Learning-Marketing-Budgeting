from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ChannelParams:
    name: str
    min_pct: float
    max_pct: float
    hill_k: float
    hill_n: float
    carryover: float
    cpa_floor: float

class MarketingSimulator:
    """Synthetic but realistic multi-channel response with diminishing returns,
    carryover, weekly seasonality, and noise."""
    def __init__(
        self,
        channels: List[ChannelParams],
        price_per_conv: float = 60.0,
        noise_std: float = 0.10,
        seasonality_amp: float = 0.15,
        seed: int = 42,
    ):
        self.channels = channels
        self.price_per_conv = price_per_conv
        self.noise_std = noise_std
        self.seasonality_amp = seasonality_amp
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._state_effect = np.zeros(len(channels))

    def reset(self):
        self._t = 0
        self._state_effect = np.zeros(len(self.channels))

    def weekly_seasonality(self, t:int) -> float:
        return 1.0 + self.seasonality_amp * np.sin(2*np.pi * (t % 7) / 7.0)

    def hill_response(self, spend: np.ndarray, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        return (spend ** n) / (k ** n + spend ** n + 1e-9)

    def step(self, spend: np.ndarray, budget: float) -> Dict[str, float]:
        assert np.isclose(spend.sum(), budget, atol=1e-3), "Spend must equal budget"
        spend = np.asarray(spend, dtype=float)

        k = np.array([c.hill_k for c in self.channels])
        n = np.array([c.hill_n for c in self.channels])
        cpa_floor = np.array([c.cpa_floor for c in self.channels])

        self._state_effect = 0.7*self._state_effect + 0.3*spend

        base_resp = self.hill_response(spend + 0.2*self._state_effect, k, n)
        seas = self.weekly_seasonality(self._t)
        noise = self.rng.normal(1.0, self.noise_std, size=spend.shape[0])
        eff = base_resp * seas * noise.clip(0.7, 1.3)

        effective_cpa = cpa_floor / (0.5 + eff)
        conversions = (spend / np.maximum(1e-6, effective_cpa))

        revenue = conversions.sum() * self.price_per_conv
        cost = spend.sum()
        profit = revenue - cost
        roas = revenue / max(1.0, cost)

        self._t += 1
        return {
            "conversions": float(conversions.sum()),
            "revenue": float(revenue),
            "cost": float(cost),
            "profit": float(profit),
            "roas": float(roas),
        }

    def bounds(self, budget: float):
        mins = np.array([c.min_pct for c in self.channels]) * budget
        maxs = np.array([c.max_pct for c in self.channels]) * budget
        return mins, maxs
