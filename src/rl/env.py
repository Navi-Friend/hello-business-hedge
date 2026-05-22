from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    transaction_cost_bps: float
    turnover_penalty: float
    drawdown_penalty: float
    action_reward_weight: float


class PairTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame, config: EnvConfig):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.config = config
        self.position = 0.0
        self.nav = 1.0
        self.max_nav = 1.0
        self.index = 0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        row = self.data.iloc[self.index]
        zscore = float(row["zscore"])
        zone = float(row["zone"])
        # Protect against NaN in observations
        if not np.isfinite(zscore):
            zscore = 0.0
        if not np.isfinite(zone):
            zone = 0.0
        return np.array([self.position, zscore, zone], dtype=np.float32)

    @staticmethod
    def _action_reward(action_value: float, zone: float) -> float:
        if zone >= 1.5:
            target = 1.0
        elif zone >= 0.5:
            target = 0.5
        elif zone <= -1.5:
            target = -1.0
        elif zone <= -0.5:
            target = -0.5
        else:
            target = 0.0
        return 1.0 - abs(action_value - target)

    def reset(self, *, seed: int | None = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.position = 0.0
        self.nav = 1.0
        self.max_nav = 1.0
        self.index = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action_value = float(np.clip(action[0], -1.0, 1.0))
        row = self.data.iloc[self.index]
        spread_ret = float(row["spread_return"])
        zone = float(row["zone"])

        # Protect against NaN/inf in data
        if not np.isfinite(spread_ret):
            spread_ret = 0.0
        if not np.isfinite(zone):
            zone = 0.0

        # Clip spread return to prevent overflow (e.g., huge single-day moves)
        spread_ret = np.clip(spread_ret, -0.05, 0.05)

        turnover = abs(action_value - self.position)
        transaction_cost = turnover * (self.config.transaction_cost_bps / 10000.0)
        pnl = self.position * spread_ret - transaction_cost

        # Clip PnL to prevent overflow in NAV
        pnl = np.clip(pnl, -0.05, 0.05)

        new_nav = self.nav * (1.0 + pnl)
        # Prevent NAV from becoming 0 or negative
        if new_nav <= 0 or not np.isfinite(new_nav):
            new_nav = self.nav * 0.99
        self.nav = new_nav

        self.max_nav = max(self.max_nav, self.nav)
        drawdown = (self.max_nav - self.nav) / max(self.max_nav, 1e-8)
        drawdown = np.clip(drawdown, 0.0, 1.0)

        reward = pnl
        reward += self.config.action_reward_weight * self._action_reward(action_value, zone)
        reward -= self.config.turnover_penalty * turnover
        reward -= self.config.drawdown_penalty * drawdown

        # Ensure reward is finite
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))

        self.position = action_value
        self.index += 1
        terminated = self.index >= len(self.data) - 1
        obs = self._get_obs() if not terminated else self._get_obs()
        info = {"nav": float(self.nav), "drawdown": float(drawdown)}
        return obs, reward, terminated, False, info
