#!/usr/bin/env python3
"""Diagnostic script to check data quality and identify issues."""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("  DATA QUALITY DIAGNOSTIC")
print("="*70)

# Check rule backtest
print("\n1. RULE BACKTEST DATA:")
backtest_file = Path("data/rule_backtest.csv")
if backtest_file.exists():
    df = pd.read_csv(backtest_file)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n   NAV stats:")
    print(f"     Min: {df['nav'].min():.6f}")
    print(f"     Max: {df['nav'].max():.6f}")
    print(f"     Mean: {df['nav'].mean():.6f}")
    print(f"\n   PnL stats:")
    print(f"     Min: {df['pnl'].min():.6f}")
    print(f"     Max: {df['pnl'].max():.6f}")
    print(f"     Mean: {df['pnl'].mean():.6f}")
    print(f"     Std: {df['pnl'].std():.6f}")
    print(f"\n   ZScore stats:")
    print(f"     Min: {df['zscore'].min():.6f}")
    print(f"     Max: {df['zscore'].max():.6f}")
    print(f"\n   First 5 rows:")
    print(df.head())
    print(f"\n   Last 5 rows:")
    print(df.tail())
else:
    print("   ⚠ File not found")

# Check prices
print("\n2. PRICES DATA:")
prices_file = Path("data/prices.parquet")
if prices_file.exists():
    prices = pd.read_parquet(prices_file)
    print(f"   Rows: {len(prices)}")
    print(f"   Columns: {prices.columns.tolist()}")
    print(f"   Date range: {prices['date'].min()} to {prices['date'].max()}")
    print(f"\n   Close price stats:")
    print(f"     Min: {prices['close'].min():.2f}")
    print(f"     Max: {prices['close'].max():.2f}")
    print(f"     Mean: {prices['close'].mean():.2f}")
    print(f"\n   Tickers: {prices['ticker'].unique().tolist()}")
    print(f"\n   Sample prices:")
    print(prices.head(10))
else:
    print("   ⚠ File not found")

# Check pairs
print("\n3. PAIRS DATA:")
pairs_file = Path("data/pairs.csv")
if pairs_file.exists():
    pairs = pd.read_csv(pairs_file)
    print(f"   Rows: {len(pairs)}")
    print(f"   Columns: {pairs.columns.tolist()}")
    if not pairs.empty:
        first = pairs.iloc[0]
        print(f"\n   First pair: {first['long_ticker']} vs {first['short_ticker']}")
        print(f"\n   Sample pairs:")
        print(pairs.head())
else:
    print("   ⚠ File not found")

print("\n" + "="*70)
print("  DIAGNOSIS")
print("="*70)

# Check for issues
if backtest_file.exists():
    df = pd.read_csv(backtest_file)
    
    # Check PnL values
    max_pnl = df['pnl'].abs().max()
    if max_pnl > 1.0:
        print(f"⚠ WARNING: Max |PnL| = {max_pnl:.2f} (should be < 0.1)")
        print("  This means single-day changes are unrealistic.")
        print("  Likely issue: spread_return is not normalized (raw price diff, not %)")
    
    # Check NAV growth
    nav_growth = df['nav'].iloc[-1] / df['nav'].iloc[0]
    if nav_growth > 2.0:
        print(f"⚠ WARNING: NAV growth = {nav_growth:.1f}x (unrealistic)")
        print("  This suggests data issue in spread calculation")
    
    # Check for NaN
    nans = df.isna().sum()
    if nans.sum() > 0:
        print(f"⚠ WARNING: Found NaN values: {nans[nans > 0].to_dict()}")
    
    if max_pnl < 0.2 and nav_growth < 2.0:
        print("✓ Data looks reasonable")

print("\n" + "="*70 + "\n")
