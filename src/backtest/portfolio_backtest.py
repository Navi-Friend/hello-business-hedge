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


def simulate_portfolio(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Simulate trading multiple pairs as a portfolio."""
    
    # Limit pairs
    pairs_df = pairs_df.head(config.max_pairs).reset_index(drop=True)
    logger.info(f"Building signals for {len(pairs_df)} pairs...")
    
    # Build signals for all pairs
    signals: dict[str, pd.DataFrame] = {}
    for idx, row in pairs_df.iterrows():
        try:
            sig = build_pair_signal(
                prices,
                long_ticker=row["long_ticker"],
                short_ticker=row["short_ticker"],
                hedge_lookback=60,
                zscore_lookback=60,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
            )
            if not sig.empty and len(sig) > 100:  # Require min 100 observations
                signals[f"pair_{idx}"] = sig.set_index("date")
                logger.debug(f"Built signal for {row['long_ticker']} vs {row['short_ticker']}: {len(sig)} obs")
        except Exception as e:
            logger.warning(f"Failed to build signal for {row['long_ticker']} vs {row['short_ticker']}: {e}")
            continue
    
    if not signals:
        raise RuntimeError("No pairs could be built")
    
    logger.info(f"Successfully built {len(signals)} pair signals")
    
    # Get union of all dates (not intersection!)
    all_dates = set()
    for sig_df in signals.values():
        all_dates.update(sig_df.index)
    all_dates = sorted(list(all_dates))
    
    logger.info(f"Common date range: {len(all_dates)} dates from {all_dates[0]} to {all_dates[-1]}")
    
    if len(all_dates) < 100:
        raise RuntimeError(f"Insufficient overlapping dates: {len(all_dates)}")
    
    # Initialize portfolio
    nav = 1.0
    positions: dict[str, float] = {k: 0.0 for k in signals.keys()}
    records = []
    
    # Replay each date
    for date in all_dates:
        portfolio_pnl = 0.0
        total_turnover = 0.0
        active_pairs = 0
        
        # Process each pair
        for pair_name, sig_df in signals.items():
            if date not in sig_df.index:
                continue  # Skip if this pair doesn't have data for this date
            
            sig_row = sig_df.loc[date]
            active_pairs += 1
            
            # Get current signal
            zscore = float(sig_row["zscore"])
            if not np.isfinite(zscore):
                zscore = 0.0
            zscore = np.clip(zscore, -10, 10)
            
            spread_ret = float(sig_row["spread_return"])
            if not np.isfinite(spread_ret):
                spread_ret = 0.0
            spread_ret = np.clip(spread_ret, -0.05, 0.05)
            
            # Determine target position
            if zscore > config.entry_z:
                target = -1.0
            elif zscore < -config.entry_z:
                target = 1.0
            elif abs(zscore) < config.exit_z:
                target = 0.0
            else:
                target = positions[pair_name]
            
            # Compute PnL for this pair
            turnover = abs(target - positions[pair_name])
            cost = turnover * (config.transaction_cost_bps / 10000.0)
            pair_pnl = positions[pair_name] * spread_ret - cost
            pair_pnl = np.clip(pair_pnl, -0.05, 0.05)
            
            if not np.isfinite(pair_pnl):
                pair_pnl = 0.0
            
            portfolio_pnl += pair_pnl
            total_turnover += turnover
            positions[pair_name] = target
        
        # Update NAV (normalize by active pairs on this date)
        if active_pairs > 0:
            avg_pnl = portfolio_pnl / active_pairs
            if not np.isfinite(avg_pnl):
                avg_pnl = 0.0
            avg_pnl = np.clip(avg_pnl, -0.1, 0.1)  # Extra safety
            
            new_nav = nav * (1.0 + avg_pnl)
            if new_nav <= 0 or not np.isfinite(new_nav):
                new_nav = nav * 0.99
            nav = new_nav
        
        records.append({
            "date": date,
            "nav": nav,
            "pnl": portfolio_pnl / max(active_pairs, 1) if active_pairs > 0 else 0.0,
            "turnover": total_turnover / max(active_pairs, 1) if active_pairs > 0 else 0.0,
            "num_pairs": active_pairs,
        })
    
    return pd.DataFrame(records)
