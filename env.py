from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List
from .simulator import MarketingSimulator, ChannelParams
from .utils import clip_proportions

@dataclass
class EnvConfig:
    budget_daily: float
    window: int
    price_per_conv: float
    volatility_penalty: float
    noise_std: float
    seasonality_amp: float
    channels: List[ChannelParams]

class AdSpendEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.sim = MarketingSimulator(
            channels=cfg.channels,
            price_per_conv=cfg.price_per_conv,
            noise_std=cfg.noise_std,
            seasonality_amp=cfg.seasonality_amp,
            seed=seed,
        )
        self.n_channels = len(cfg.channels)
        self.window = cfg.window
        self.obs_dim = self.window * (self.n_channels + 2)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(self.n_channels,), dtype=np.float32)

        self._hist_spend = np.zeros((self.window, self.n_channels), dtype=float)
        self._hist_roas = np.zeros(self.window, dtype=float)
        self._hist_conv = np.zeros(self.window, dtype=float)
        self._prev_proportions = np.ones(self.n_channels) / self.n_channels

    def _get_obs(self):
        x = np.concatenate([self._hist_spend.flatten(), self._hist_roas, self._hist_conv])
        return x.astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset()
        self._hist_spend[:] = 0.0
        self._hist_roas[:] = 1.0
        self._hist_conv[:] = 0.0
        self._prev_proportions = np.ones(self.n_channels) / self.n_channels
        return self._get_obs(), {}

    def step(self, action):
        raw = np.array(action, dtype=float)
        exps = np.exp(raw - raw.max())
        props = exps / exps.sum()
        mins = np.array([c.min_pct for c in self.cfg.channels])
        maxs = np.array([c.max_pct for c in self.cfg.channels])
        props = clip_proportions(props, mins, maxs)

        spend = props * self.cfg.budget_daily
        out = self.sim.step(spend, budget=self.cfg.budget_daily)

        self._hist_spend = np.roll(self._hist_spend, -1, axis=0)
        self._hist_spend[-1] = props
        self._hist_roas = np.roll(self._hist_roas, -1); self._hist_roas[-1] = out["roas"]
        self._hist_conv = np.roll(self._hist_conv, -1); self._hist_conv[-1] = out["conversions"]

        vol_pen = self.cfg.volatility_penalty * np.square(props - self._prev_proportions).sum()
        reward = (out["revenue"] - out["cost"]) / 1000.0 - vol_pen
        self._prev_proportions = props

        terminated = False
        truncated = False
        info = {"roas": out["roas"], "profit": out["profit"], "conversions": out["conversions"], "spend_props": props}
        return self._get_obs(), float(reward), terminated, truncated, info
