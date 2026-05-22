from __future__ import annotations

from stable_baselines3 import A2C, PPO

from src.rl.env import PairTradingEnv


def train_agent(env: PairTradingEnv, algo: str, total_timesteps: int):
    algo_upper = algo.upper()
    if algo_upper == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    return model
