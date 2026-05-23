#!/usr/bin/env python3
"""Tune rule-based spread trading parameters without training RL.

This script reuses the current `data/pairs.csv` and evaluates only the
out-of-sample monthly trading layer. It is intentionally RL-free: if this
baseline is weak, the RL sizing model should not be trained.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.portfolio_backtest import BacktestConfig, build_pair_signal_for_month
from src.config import load_config


def compute_sharpe(nav: pd.Series) -> float:
    nav_values = nav.to_numpy(dtype=float)
    if len(nav_values) < 2:
        return 0.0
    returns = np.diff(nav_values) / (nav_values[:-1] + 1e-12)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    values = nav.to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def build_signal_cache(
    prices: pd.DataFrame,
    pairs: pd.DataFrame,
    base_config: BacktestConfig,
    hedge_lookback: int,
    zscore_lookback: int,
) -> dict[tuple[pd.Timestamp, str, str], pd.DataFrame]:
    cache: dict[tuple[pd.Timestamp, str, str], pd.DataFrame] = {}
    for month, month_df in pairs.groupby("month"):
        for row in month_df.itertuples(index=False):
            key = (month, row.long_ticker, row.short_ticker)
            signal = build_pair_signal_for_month(
                prices,
                long_ticker=row.long_ticker,
                short_ticker=row.short_ticker,
                month=month,
                formation_months=base_config.formation_months,
                hedge_lookback=hedge_lookback,
                zscore_lookback=zscore_lookback,
                entry_z=base_config.entry_z,
                exit_z=base_config.exit_z,
            )
            if not signal.empty:
                cache[key] = signal.set_index("date")
    return cache


def target_position(zscore: float, current_position: float, is_last_date: bool, config: BacktestConfig) -> float:
    if is_last_date:
        return 0.0
    if config.signal_direction == "mean_reversion":
        high_target = -1.0
        low_target = 1.0
    elif config.signal_direction == "trend_following":
        high_target = 1.0
        low_target = -1.0
    else:
        raise ValueError(f"Unsupported signal_direction: {config.signal_direction}")

    if zscore > config.entry_z:
        return high_target
    if zscore < -config.entry_z:
        return low_target
    if abs(zscore) < config.exit_z:
        return 0.0
    return current_position


def simulate_cached(
    pairs: pd.DataFrame,
    signal_cache: dict[tuple[pd.Timestamp, str, str], pd.DataFrame],
    config: BacktestConfig,
    allowed_months: set[pd.Period] | None = None,
) -> pd.DataFrame:
    nav = 1.0
    records: list[dict[str, object]] = []

    for month, month_df in pairs.groupby("month"):
        month_period = pd.Timestamp(month).to_period("M")
        if allowed_months is not None and month_period not in allowed_months:
            continue

        month_df = month_df.sort_values("mom1_diff", ascending=False).head(config.max_pairs)
        signals: dict[str, pd.DataFrame] = {}
        for idx, row in month_df.reset_index(drop=True).iterrows():
            signal = signal_cache.get((month, row["long_ticker"], row["short_ticker"]))
            if signal is not None:
                signals[f"pair_{idx}"] = signal
        if not signals:
            continue

        positions = {pair_name: 0.0 for pair_name in signals}
        all_dates = sorted(set().union(*(signal.index for signal in signals.values())))
        last_date = all_dates[-1]

        for date in all_dates:
            portfolio_pnl = 0.0
            active_pairs = 0
            for pair_name, signal in signals.items():
                if date not in signal.index:
                    continue
                row = signal.loc[date]
                zscore = float(np.clip(row["zscore"], -10, 10))
                spread_return = float(np.clip(row["spread_return"], -0.05, 0.05))
                active_pairs += 1

                target = target_position(zscore, positions[pair_name], date == last_date, config)
                turnover = abs(target - positions[pair_name])
                cost = turnover * (config.transaction_cost_bps / 10000.0)
                pair_pnl = np.clip(positions[pair_name] * spread_return - cost, -0.05, 0.05)
                portfolio_pnl += pair_pnl
                positions[pair_name] = target

            avg_pnl = float(np.clip(portfolio_pnl / max(active_pairs, 1), -0.1, 0.1))
            nav *= 1.0 + avg_pnl
            records.append({"date": date, "month": month, "nav": nav, "pnl": avg_pnl})

    return pd.DataFrame(records)


def metrics(nav_df: pd.DataFrame) -> dict[str, float]:
    if nav_df.empty:
        return {"sharpe": 0.0, "final_nav": 1.0, "max_drawdown": 0.0}
    return {
        "sharpe": compute_sharpe(nav_df["nav"]),
        "final_nav": float(nav_df["nav"].iloc[-1]),
        "max_drawdown": max_drawdown(nav_df["nav"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/prices.parquet")
    parser.add_argument("--pairs", default="data/pairs.csv")
    parser.add_argument("--out", default="data/rule_tuning_results.csv")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fast", action="store_true", help="Run a smaller grid for quick iteration")
    args = parser.parse_args()

    cfg = load_config("config/base.yaml")
    base_config = BacktestConfig(
        entry_z=cfg.pairs.entry_z,
        exit_z=cfg.pairs.exit_z,
        transaction_cost_bps=cfg.rl.transaction_cost_bps,
        max_pairs=cfg.pairs.max_portfolio_pairs,
        formation_months=cfg.clustering.formation_months,
        hedge_lookback=cfg.pairs.hedge_lookback,
        zscore_lookback=cfg.pairs.zscore_lookback,
        signal_direction=cfg.pairs.signal_direction,
        min_formation_score=cfg.pairs.min_formation_score,
    )

    prices = pd.read_parquet(args.prices)
    pairs = pd.read_csv(args.pairs)
    pairs["month"] = pd.to_datetime(pairs["month"])

    months = sorted(pairs["month"].dt.to_period("M").unique())
    split_idx = min(max(1, int(len(months) * args.train_fraction)), len(months) - 1)
    train_months = set(months[:split_idx])
    test_months = set(months[split_idx:])

    if args.fast:
        entries = [1.5, 2.0, 2.5]
        exits = [0.5]
        max_pairs_values = [10, 20]
        lookbacks = [(60, 60), (120, 120)]
        directions = ["mean_reversion", "trend_following"]
    else:
        entries = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        exits = [0.25, 0.5, 0.75, 1.0]
        max_pairs_values = [5, 10, 15, 20, 30]
        lookbacks = [(60, 60), (90, 90), (120, 120), (180, 120)]
        directions = ["mean_reversion", "trend_following"]

    rows: list[dict[str, object]] = []
    for hedge_lookback, zscore_lookback in lookbacks:
        print(f"Building signals: hedge={hedge_lookback}, zscore={zscore_lookback}", flush=True)
        signal_cache = build_signal_cache(prices, pairs, base_config, hedge_lookback, zscore_lookback)
        for entry_z, exit_z, max_pairs, direction in product(entries, exits, max_pairs_values, directions):
            if exit_z >= entry_z:
                continue
            test_config = replace(
                base_config,
                entry_z=entry_z,
                exit_z=exit_z,
                max_pairs=max_pairs,
                hedge_lookback=hedge_lookback,
                zscore_lookback=zscore_lookback,
                signal_direction=direction,
            )
            train_nav = simulate_cached(pairs, signal_cache, test_config, train_months)
            test_nav = simulate_cached(pairs, signal_cache, test_config, test_months)
            full_nav = simulate_cached(pairs, signal_cache, test_config)
            train_metrics = metrics(train_nav)
            test_metrics = metrics(test_nav)
            full_metrics = metrics(full_nav)
            rows.append(
                {
                    "signal_direction": direction,
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "max_pairs": max_pairs,
                    "hedge_lookback": hedge_lookback,
                    "zscore_lookback": zscore_lookback,
                    "train_sharpe": train_metrics["sharpe"],
                    "train_final_nav": train_metrics["final_nav"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_final_nav": test_metrics["final_nav"],
                    "test_max_drawdown": test_metrics["max_drawdown"],
                    "full_sharpe": full_metrics["sharpe"],
                    "full_final_nav": full_metrics["final_nav"],
                    "full_max_drawdown": full_metrics["max_drawdown"],
                }
            )

    results = pd.DataFrame(rows).sort_values(["test_sharpe", "train_sharpe"], ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(results.head(20).to_string(index=False))
    print(f"Saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
