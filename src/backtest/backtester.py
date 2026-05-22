from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Ensure numpy operations don't produce warnings
np.seterr(over='ignore', invalid='ignore')


@dataclass
class BacktestConfig:
    entry_z: float
    exit_z: float
    transaction_cost_bps: float
    max_pairs: int = 20  # Max pairs in portfolio


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    betas = [np.nan] * len(y)
    for idx in range(window, len(y) + 1):
        y_win = y.iloc[idx - window : idx]
        x_win = x.iloc[idx - window : idx]
        model = sm.OLS(y_win, x_win).fit()
        betas[idx - 1] = model.params.iloc[0]
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
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    wide = wide[[long_ticker, short_ticker]].dropna()

    price_long = wide[long_ticker]
    price_short = wide[short_ticker]
    beta = rolling_hedge_ratio(price_long, price_short, hedge_lookback)
    
    # Protect beta from NaN/inf
    beta = beta.fillna(1.0)
    beta = beta.replace([np.inf, -np.inf], 1.0)
    beta = beta.clip(-100, 100)
    
    spread = price_long - beta * price_short
    spread_mean = spread.rolling(zscore_lookback).mean()
    spread_std = spread.rolling(zscore_lookback).std()
    
    # Protect zscore calculation
    spread_std = spread_std.replace(0, np.nan).fillna(spread_std.max())
    spread_std = spread_std.clip(lower=1e-6)
    
    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(-10, 10)
    
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
            "spread": spread,
            "zscore": zscore,
            "spread_return": spread_return,
            "vol": vol,
            "borrow_cost": 0.0,
            "zone": zone,
        }
    ).reset_index(drop=True)


def simulate_rule_based(data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    position = 0.0
    nav = 1.0
    records = []

    for row in data.itertuples(index=False):
        # Protect against NaN/inf in data
        zscore = float(row.zscore) if hasattr(row, 'zscore') else 0.0
        spread_ret = float(row.spread_return) if hasattr(row, 'spread_return') else 0.0
        
        if not np.isfinite(zscore):
            zscore = 0.0
        if not np.isfinite(spread_ret):
            spread_ret = 0.0
        
        # Clip extreme values
        zscore = np.clip(zscore, -10, 10)
        spread_ret = np.clip(spread_ret, -0.05, 0.05)
        
        # Determine target position
        if zscore > config.entry_z:
            target = -1.0
        elif zscore < -config.entry_z:
            target = 1.0
        elif abs(zscore) < config.exit_z:
            target = 0.0
        else:
            target = position

        turnover = abs(target - position)
        cost = turnover * (config.transaction_cost_bps / 10000.0)
        pnl = position * spread_ret - cost
        
        # Protect PnL and NAV
        pnl = np.clip(pnl, -0.05, 0.05)
        new_nav = nav * (1.0 + pnl)
        
        # Prevent NAV from going negative or NaN
        if new_nav <= 0 or not np.isfinite(new_nav):
            new_nav = nav * 0.99
        
        nav = new_nav
        position = target
        
        records.append(
            {
                "date": row.date,
                "nav": nav,
                "position": position,
                "pnl": pnl,
                "zscore": zscore,
            }
        )

    return pd.DataFrame(records)
