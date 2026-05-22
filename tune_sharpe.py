#!/usr/bin/env python3
"""
Auto-tune pipeline parameters until RL Sharpe reaches target.

Usage:
  python tune_sharpe.py --target 1.0 --max-runs 12 --timesteps 10000

Docker:
  docker compose exec -e LOG_LEVEL=INFO -e STOOQ_API_KEY=... app python tune_sharpe.py --target 1.0
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.config import load_config
from src.pipeline.run import run_pipeline
from src.backtest.backtester import build_pair_signal
from src.rl.env import EnvConfig, PairTradingEnv


logger = logging.getLogger(__name__)


def compute_sharpe(nav_values: np.ndarray) -> float:
    if len(nav_values) < 2:
        return 0.0
    returns = np.diff(nav_values) / (nav_values[:-1] + 1e-12)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))


def evaluate_rl_sharpe(cfg) -> float:
    model_zip = Path("data/rl_model.zip")
    if not model_zip.exists():
        logger.warning("RL model not found: %s", model_zip)
        return 0.0

    from stable_baselines3 import A2C, PPO

    prices = pd.read_parquet("data/prices.parquet")[["date", "ticker", "close"]]
    pairs = pd.read_csv("data/pairs.csv")
    if pairs.empty:
        logger.warning("pairs.csv is empty")
        return 0.0

    pair = pairs.iloc[0]
    signal = build_pair_signal(
        prices,
        long_ticker=pair["long_ticker"],
        short_ticker=pair["short_ticker"],
        hedge_lookback=cfg.pairs.hedge_lookback,
        zscore_lookback=cfg.pairs.zscore_lookback,
        entry_z=cfg.pairs.entry_z,
        exit_z=cfg.pairs.exit_z,
    )

    env_config = EnvConfig(
        transaction_cost_bps=cfg.rl.transaction_cost_bps,
        turnover_penalty=cfg.rl.turnover_penalty,
        drawdown_penalty=cfg.rl.drawdown_penalty,
        action_reward_weight=cfg.rl.action_reward_weight,
    )
    env = PairTradingEnv(signal, env_config)

    algo = cfg.rl.algo.upper()
    if algo == "PPO":
        model = PPO.load(str(model_zip))
    else:
        model = A2C.load(str(model_zip))

    obs, _ = env.reset()
    nav_values = [1.0]
    for _ in range(len(signal) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        nav_values.append(info["nav"])
        if terminated or truncated:
            break

    return compute_sharpe(np.array(nav_values))


def build_search_space() -> List[Dict[str, object]]:
    distance_methods = ["pc_distance", "pca_distance", "ssd_distance"]
    optics_xi = [0.05, 0.1, 0.2]
    dispersion_threshold = [0.01, 0.02, 0.03]
    entry_exit = [(1.5, 0.3), (1.8, 0.4), (2.0, 0.5)]
    lookbacks = [(60, 60), (90, 90), (120, 120)]
    turnover_penalty = [0.001, 0.005, 0.01]
    drawdown_penalty = [0.1, 0.2, 0.3]
    action_reward_weight = [0.05, 0.1, 0.2]
    algos = ["A2C", "PPO"]

    grid = []
    for dm in distance_methods:
        for xi in optics_xi:
            for disp in dispersion_threshold:
                for (entry_z, exit_z) in entry_exit:
                    for (hedge_lb, z_lb) in lookbacks:
                        for tp in turnover_penalty:
                            for dp in drawdown_penalty:
                                for arw in action_reward_weight:
                                    for algo in algos:
                                        grid.append(
                                            {
                                                "distance_method": dm,
                                                "optics_xi": xi,
                                                "dispersion_threshold": disp,
                                                "entry_z": entry_z,
                                                "exit_z": exit_z,
                                                "hedge_lookback": hedge_lb,
                                                "zscore_lookback": z_lb,
                                                "turnover_penalty": tp,
                                                "drawdown_penalty": dp,
                                                "action_reward_weight": arw,
                                                "algo": algo,
                                            }
                                        )
    random.shuffle(grid)
    return grid


def apply_params(cfg, params: Dict[str, object], timesteps: int) -> None:
    cfg.clustering.distance_method = params["distance_method"]
    cfg.clustering.optics_xi = params["optics_xi"]
    cfg.clustering.dispersion_threshold = params["dispersion_threshold"]
    cfg.pairs.entry_z = params["entry_z"]
    cfg.pairs.exit_z = params["exit_z"]
    cfg.pairs.hedge_lookback = params["hedge_lookback"]
    cfg.pairs.zscore_lookback = params["zscore_lookback"]
    cfg.rl.turnover_penalty = params["turnover_penalty"]
    cfg.rl.drawdown_penalty = params["drawdown_penalty"]
    cfg.rl.action_reward_weight = params["action_reward_weight"]
    cfg.rl.algo = params["algo"]
    cfg.rl.total_timesteps = timesteps
    cfg.rl.enabled = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=1.0, help="Sharpe target to stop")
    parser.add_argument("--max-runs", type=int, default=12, help="Maximum number of trials")
    parser.add_argument("--timesteps", type=int, default=10000, help="RL timesteps per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for search order")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    random.seed(args.seed)

    results: List[Dict[str, object]] = []
    search_space = build_search_space()
    max_runs = min(args.max_runs, len(search_space))

    logger.info("Starting auto-tune: target=%.2f, max_runs=%d, timesteps=%d", args.target, max_runs, args.timesteps)

    for idx in range(max_runs):
        params = search_space[idx]
        cfg = load_config("config/base.yaml")
        apply_params(cfg, params, args.timesteps)

        logger.info("Run %d/%d | params=%s", idx + 1, max_runs, params)
        run_pipeline(cfg)

        sharpe = evaluate_rl_sharpe(cfg)
        logger.info("Run %d result: RL Sharpe=%.3f", idx + 1, sharpe)

        row = dict(params)
        row["sharpe"] = sharpe
        row["timesteps"] = args.timesteps
        results.append(row)

        if sharpe >= args.target:
            logger.info("Target reached: Sharpe=%.3f >= %.2f", sharpe, args.target)
            break

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_path = Path("data/tuning_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path.resolve())

    if not results_df.empty:
        best = results_df.iloc[0].to_dict()
        best_path = Path("data/best_params.yaml")
        with open(best_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(best, handle, sort_keys=False, allow_unicode=True)
        logger.info("Best params saved to %s", best_path.resolve())


if __name__ == "__main__":
    main()
