#!/usr/bin/env python3
"""
Portfolio backtest: simulates trading a portfolio of multiple pairs.
Each pair is independent; positions are aggregated at portfolio level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Setup logging
logger = logging.getLogger(__name__)

# Suppress warnings
np.seterr(over='ignore', invalid='ignore')


@dataclass
class BacktestConfig:
    entry_z: float
    exit_z: float
    transaction_cost_bps: float
    max_pairs: int = 20  # Max pairs in portfolio
    formation_months: int = 36
    hedge_lookback: int = 60
    zscore_lookback: int = 60
    signal_direction: str = "mean_reversion"
    min_formation_score: float = -1e9


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS hedge ratio."""
    betas = [np.nan] * len(y)
    for idx in range(window, len(y) + 1):
        y_win = y.iloc[idx - window : idx]
        x_win = x.iloc[idx - window : idx]
        try:
            model = sm.OLS(y_win, x_win).fit()
            betas[idx - 1] = model.params.iloc[0]
        except Exception:
            betas[idx - 1] = np.nan
    return pd.Series(betas, index=y.index)


def build_pair_signal(
    prices: pd.DataFrame,
    long_ticker: str,
    short_ticker: str,
    hedge_lookback: int,
    zscore_lookback: int,
    entry_z: float,
    exit_z: float,
) -> pd.DataFrame:
    """Build trading signals for a single pair."""
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    wide = wide[[long_ticker, short_ticker]].dropna()

    price_long = wide[long_ticker]
    price_short = wide[short_ticker]
    beta = rolling_hedge_ratio(price_long, price_short, hedge_lookback)
    
    # Protect beta
    beta = beta.fillna(1.0)
    beta = beta.replace([np.inf, -np.inf], 1.0)
    beta = beta.clip(-100, 100)
    
    spread = price_long - beta * price_short
    spread_mean = spread.rolling(zscore_lookback).mean()
    spread_std = spread.rolling(zscore_lookback).std()
    
    # Protect zscore
    spread_std = spread_std.replace(0, np.nan).fillna(spread_std.max())
    spread_std = spread_std.clip(lower=1e-6)
    
    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(-10, 10)
    
    # Normalized percentage returns
    ret_long = price_long.pct_change().fillna(0.0)
    ret_short = price_short.pct_change().fillna(0.0)
    hedge_scale = 1.0 + beta.abs()
    spread_return = (ret_long - beta * ret_short) / hedge_scale
    spread_return = (
        spread_return.replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .clip(-0.05, 0.05)
    )

    vol = spread_return.rolling(21).std().fillna(0.0).clip(lower=1e-8)
    
    zone = pd.Series(0, index=zscore.index, dtype=float)
    zone[zscore <= -entry_z] = 2
    zone[(zscore > -entry_z) & (zscore <= -exit_z)] = 1
    zone[(zscore >= exit_z) & (zscore < entry_z)] = -1
    zone[zscore >= entry_z] = -2

    return pd.DataFrame(
        {
            "date": wide.index,
            "pair": f"{long_ticker}_{short_ticker}",
            "long_ticker": long_ticker,
            "short_ticker": short_ticker,
            "spread": spread,
            "zscore": zscore,
            "spread_return": spread_return,
            "vol": vol,
            "zone": zone,
        }
    ).reset_index(drop=True)


def _fixed_hedge_ratio(price_long: pd.Series, price_short: pd.Series) -> float:
    try:
        model = sm.OLS(price_long, price_short).fit()
        beta = float(model.params.iloc[0])
    except Exception:
        beta = 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, -100.0, 100.0))


def _resolve_direction_target(
    zscore: float,
    current_position: float,
    is_last_date: bool,
    entry_z: float,
    exit_z: float,
    signal_direction: str,
) -> float:
    if is_last_date:
        return 0.0

    if signal_direction == "mean_reversion":
        high_target = -1.0
        low_target = 1.0
    elif signal_direction == "trend_following":
        high_target = 1.0
        low_target = -1.0
    else:
        raise ValueError(f"Unsupported signal_direction: {signal_direction}")

    if zscore > entry_z:
        return high_target
    if zscore < -entry_z:
        return low_target
    if abs(zscore) < exit_z:
        return 0.0
    return current_position


def _score_direction(
    zscore: pd.Series,
    spread_return: pd.Series,
    entry_z: float,
    exit_z: float,
    transaction_cost_bps: float,
    signal_direction: str,
) -> float:
    position = 0.0
    nav = 1.0
    nav_values = []
    trade_count = 0
    last_date = zscore.index[-1] if len(zscore) else None

    for date in zscore.index:
        z_value = float(np.clip(zscore.loc[date], -10, 10))
        ret_value = float(np.clip(spread_return.loc[date], -0.05, 0.05))
        target = _resolve_direction_target(
            z_value,
            position,
            date == last_date,
            entry_z,
            exit_z,
            signal_direction,
        )
        turnover = abs(target - position)
        if turnover > 0:
            trade_count += 1
        cost = turnover * (transaction_cost_bps / 10000.0)
        pnl = float(np.clip(position * ret_value - cost, -0.05, 0.05))
        nav *= 1.0 + pnl
        nav_values.append(nav)
        position = target

    if trade_count == 0 or len(nav_values) < 2:
        return -float("inf")
    nav_array = np.array(nav_values, dtype=float)
    returns = np.diff(nav_array) / (nav_array[:-1] + 1e-12)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return -float("inf")
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))


def _choose_signal_direction(
    formation_spread: pd.Series,
    formation_return: pd.Series,
    entry_z: float,
    exit_z: float,
    zscore_lookback: int,
    transaction_cost_bps: float,
) -> tuple[str, float]:
    rolling_mean = formation_spread.rolling(zscore_lookback).mean()
    rolling_std = formation_spread.rolling(zscore_lookback).std().replace(0, np.nan)
    zscore = ((formation_spread - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
    formation_eval = pd.DataFrame({"zscore": zscore, "spread_return": formation_return}).dropna()
    if len(formation_eval) < max(20, zscore_lookback // 2):
        return "mean_reversion", -float("inf")

    mean_reversion_score = _score_direction(
        formation_eval["zscore"],
        formation_eval["spread_return"],
        entry_z,
        exit_z,
        transaction_cost_bps,
        "mean_reversion",
    )
    trend_score = _score_direction(
        formation_eval["zscore"],
        formation_eval["spread_return"],
        entry_z,
        exit_z,
        transaction_cost_bps,
        "trend_following",
    )
    if trend_score > mean_reversion_score:
        return "trend_following", trend_score
    return "mean_reversion", mean_reversion_score


def build_pair_signal_for_month(
    prices: pd.DataFrame,
    long_ticker: str,
    short_ticker: str,
    month: pd.Timestamp | str,
    formation_months: int,
    hedge_lookback: int,
    zscore_lookback: int,
    entry_z: float,
    exit_z: float,
    transaction_cost_bps: float = 0.0,
    signal_direction: str = "mean_reversion",
) -> pd.DataFrame:
    """Build an out-of-sample signal for one pair in one trading month.

    Hedge ratio and z-score statistics are estimated only on the formation
    window ending before the trading month starts.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if long_ticker not in wide.columns or short_ticker not in wide.columns:
        return pd.DataFrame()

    wide = wide[[long_ticker, short_ticker]].dropna()
    month_start = pd.Timestamp(month).to_period("M").to_timestamp()
    month_end = month_start + pd.offsets.MonthEnd(0)
    formation_end = month_start - pd.Timedelta(days=1)
    formation_start = formation_end - pd.DateOffset(months=formation_months)

    formation = wide.loc[(wide.index >= formation_start) & (wide.index <= formation_end)]
    trading = wide.loc[(wide.index >= month_start) & (wide.index <= month_end)]
    min_formation_rows = max(
        hedge_lookback,
        zscore_lookback,
        int(252 * formation_months / 12 * 0.8),
    )
    if len(formation) < min_formation_rows or len(trading) < 2:
        return pd.DataFrame()

    hedge_window = formation.tail(hedge_lookback)
    beta = _fixed_hedge_ratio(hedge_window[long_ticker], hedge_window[short_ticker])

    formation_spread = formation[long_ticker] - beta * formation[short_ticker]
    spread_ref = formation_spread.tail(zscore_lookback)
    spread_mean = float(spread_ref.mean())
    spread_std = float(spread_ref.std())
    if not np.isfinite(spread_std) or spread_std < 1e-6:
        spread_std = 1e-6

    spread = trading[long_ticker] - beta * trading[short_ticker]
    zscore = ((spread - spread_mean) / spread_std).replace([np.inf, -np.inf], 0.0)
    zscore = zscore.fillna(0.0).clip(-10, 10)

    prev_close = wide.loc[wide.index < month_start].tail(1)
    return_base = pd.concat([prev_close, trading])
    ret_long = return_base[long_ticker].pct_change().reindex(trading.index).fillna(0.0)
    ret_short = return_base[short_ticker].pct_change().reindex(trading.index).fillna(0.0)
    hedge_scale = 1.0 + abs(beta)
    spread_return = ((ret_long - beta * ret_short) / hedge_scale).replace([np.inf, -np.inf], 0.0)
    spread_return = spread_return.fillna(0.0).clip(-0.05, 0.05)
    vol = spread_return.rolling(21).std().fillna(0.0).clip(lower=1e-8)

    formation_ret_long = formation[long_ticker].pct_change().fillna(0.0)
    formation_ret_short = formation[short_ticker].pct_change().fillna(0.0)
    formation_return = ((formation_ret_long - beta * formation_ret_short) / hedge_scale).replace([np.inf, -np.inf], 0.0)
    formation_return = formation_return.fillna(0.0).clip(-0.05, 0.05)
    selected_direction = signal_direction
    formation_score = _score_direction(
        zscore=pd.Series(dtype=float),
        spread_return=pd.Series(dtype=float),
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_cost_bps=transaction_cost_bps,
        signal_direction="mean_reversion",
    )
    if signal_direction == "adaptive":
        selected_direction, formation_score = _choose_signal_direction(
            formation_spread,
            formation_return,
            entry_z,
            exit_z,
            zscore_lookback,
            transaction_cost_bps,
        )
    else:
        rolling_mean = formation_spread.rolling(zscore_lookback).mean()
        rolling_std = formation_spread.rolling(zscore_lookback).std().replace(0, np.nan)
        formation_zscore = ((formation_spread - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
        formation_eval = pd.DataFrame({"zscore": formation_zscore, "spread_return": formation_return}).dropna()
        if len(formation_eval) >= max(20, zscore_lookback // 2):
            formation_score = _score_direction(
                formation_eval["zscore"],
                formation_eval["spread_return"],
                entry_z,
                exit_z,
                transaction_cost_bps,
                selected_direction,
            )

    zone = pd.Series(0, index=zscore.index, dtype=float)
    zone[zscore <= -entry_z] = 2
    zone[(zscore > -entry_z) & (zscore <= -exit_z)] = 1
    zone[(zscore >= exit_z) & (zscore < entry_z)] = -1
    zone[zscore >= entry_z] = -2

    return pd.DataFrame(
        {
            "date": trading.index,
            "month": month_start,
            "pair": f"{long_ticker}_{short_ticker}",
            "long_ticker": long_ticker,
            "short_ticker": short_ticker,
            "beta": beta,
            "spread": spread,
            "zscore": zscore,
            "spread_return": spread_return,
            "vol": vol,
            "zone": zone,
            "signal_direction": selected_direction,
            "formation_score": formation_score,
        }
    ).reset_index(drop=True)


def build_rolling_signal_dataset(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    pairs_df = pairs_df.copy()
    if pairs_df.empty:
        return pd.DataFrame()

    pairs_df["month"] = pd.to_datetime(pairs_df["month"])
    signals: list[pd.DataFrame] = []
    for month, month_df in pairs_df.groupby("month"):
        candidates: list[tuple[float, float, int, pd.Series, pd.DataFrame]] = []
        for idx, row in month_df.sort_values("mom1_diff", ascending=False).reset_index(drop=True).iterrows():
            sig = build_pair_signal_for_month(
                prices,
                long_ticker=row["long_ticker"],
                short_ticker=row["short_ticker"],
                month=month,
                formation_months=config.formation_months,
                hedge_lookback=config.hedge_lookback,
                zscore_lookback=config.zscore_lookback,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
                transaction_cost_bps=config.transaction_cost_bps,
                signal_direction=config.signal_direction,
            )
            if sig.empty:
                continue
            formation_score = float(sig["formation_score"].iloc[0])
            if not np.isfinite(formation_score):
                formation_score = -1e9
            if formation_score < config.min_formation_score:
                continue
            candidates.append((formation_score, float(row["mom1_diff"]), idx, row, sig))

        candidates = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[: config.max_pairs]
        for rank, (_, _, idx, row, sig) in enumerate(candidates):
            sig = sig.copy()
            sig["episode_id"] = f"{pd.Timestamp(month).date()}_{rank}_{row['long_ticker']}_{row['short_ticker']}"
            sig["episode_end"] = False
            sig.loc[sig.index[-1], "episode_end"] = True
            signals.append(sig)

    if not signals:
        return pd.DataFrame()
    return pd.concat(signals, ignore_index=True).sort_values(["month", "episode_id", "date"]).reset_index(drop=True)


def _target_position(zscore: float, current_position: float, is_last_date: bool, config: BacktestConfig) -> float:
    return _resolve_direction_target(
        zscore,
        current_position,
        is_last_date,
        config.entry_z,
        config.exit_z,
        config.signal_direction,
    )


def simulate_portfolio(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Simulate a rolling 36-month formation / 1-month trading portfolio."""
    pairs_df = pairs_df.copy()
    pairs_df["month"] = pd.to_datetime(pairs_df["month"])
    nav = 1.0
    records = []

    for month, month_df in pairs_df.groupby("month"):
        candidates: list[tuple[float, float, int, pd.DataFrame]] = []
        for idx, row in month_df.sort_values("mom1_diff", ascending=False).reset_index(drop=True).iterrows():
            sig = build_pair_signal_for_month(
                prices,
                long_ticker=row["long_ticker"],
                short_ticker=row["short_ticker"],
                month=month,
                formation_months=config.formation_months,
                hedge_lookback=config.hedge_lookback,
                zscore_lookback=config.zscore_lookback,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
                transaction_cost_bps=config.transaction_cost_bps,
                signal_direction=config.signal_direction,
            )
            if not sig.empty:
                formation_score = float(sig["formation_score"].iloc[0])
                if not np.isfinite(formation_score):
                    formation_score = -1e9
                if formation_score < config.min_formation_score:
                    continue
                candidates.append((formation_score, float(row["mom1_diff"]), idx, sig.set_index("date")))

        candidates = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[: config.max_pairs]
        signals: dict[str, pd.DataFrame] = {
            f"pair_{rank}": sig for rank, (_, _, _, sig) in enumerate(candidates)
        }

        if not signals:
            continue

        positions: dict[str, float] = {k: 0.0 for k in signals.keys()}
        all_dates = sorted(set().union(*(sig.index for sig in signals.values())))
        last_date = all_dates[-1]

        for date in all_dates:
            portfolio_pnl = 0.0
            total_turnover = 0.0
            active_pairs = 0

            for pair_name, sig_df in signals.items():
                if date not in sig_df.index:
                    continue

                sig_row = sig_df.loc[date]
                active_pairs += 1
                zscore = float(sig_row["zscore"])
                spread_ret = float(sig_row["spread_return"])
                if not np.isfinite(zscore):
                    zscore = 0.0
                if not np.isfinite(spread_ret):
                    spread_ret = 0.0
                zscore = np.clip(zscore, -10, 10)
                spread_ret = np.clip(spread_ret, -0.05, 0.05)

                signal_direction = str(sig_row.get("signal_direction", config.signal_direction))
                target = _resolve_direction_target(
                    zscore,
                    positions[pair_name],
                    date == last_date,
                    config.entry_z,
                    config.exit_z,
                    signal_direction,
                )

                turnover = abs(target - positions[pair_name])
                cost = turnover * (config.transaction_cost_bps / 10000.0)
                pair_pnl = positions[pair_name] * spread_ret - cost
                pair_pnl = np.clip(pair_pnl, -0.05, 0.05)
                if not np.isfinite(pair_pnl):
                    pair_pnl = 0.0

                portfolio_pnl += pair_pnl
                total_turnover += turnover
                positions[pair_name] = target

            avg_pnl = portfolio_pnl / max(active_pairs, 1)
            if not np.isfinite(avg_pnl):
                avg_pnl = 0.0
            avg_pnl = np.clip(avg_pnl, -0.1, 0.1)
            new_nav = nav * (1.0 + avg_pnl)
            if new_nav <= 0 or not np.isfinite(new_nav):
                new_nav = nav * 0.99
            nav = new_nav

            records.append({
                "date": date,
                "month": month,
                "nav": nav,
                "pnl": avg_pnl,
                "turnover": total_turnover / max(active_pairs, 1),
                "num_pairs": active_pairs,
                "selected_pairs": len(signals),
            })

    if not records:
        raise RuntimeError("No rolling monthly pair signals could be built")
    return pd.DataFrame(records)
