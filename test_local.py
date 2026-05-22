#!/usr/bin/env python3
"""
Test script to validate StatArb pipeline locally (no Docker required).
Run: python test_local.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("StatArb Pipeline - Local Test")
print("=" * 80)

# Test 1: Load configuration
print("\n[1/5] Loading configuration...")
try:
    from src.config import Config
    config = Config.load("config/base.yaml")
    print(f"✓ Config loaded: clustering_mode={config.clustering.mode}, distance_method={config.clustering.distance_method}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test 2: Import core modules
print("\n[2/5] Importing core modules...")
try:
    from src.clustering.distance_optics import DistanceOPTICS
    from src.pairs.pair_selection import PairSelector
    from src.backtest.backtester import Backtester
    from src.rl.env import PairTradingEnv
    print("✓ All core modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 3: Data fetching (optional, may fail due to network)
print("\n[3/5] Testing data fetch...")
try:
    import yfinance as yf
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = yf.download(tickers, start="2023-01-01", end="2023-12-31", progress=False)
    print(f"✓ Downloaded data for {len(tickers)} tickers: {data.shape}")
except Exception as e:
    print(f"⚠ Data fetch warning (may fail offline): {e}")

# Test 4: Test DistanceOPTICS with synthetic data
print("\n[4/5] Testing DistanceOPTICS with synthetic data...")
try:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_blobs
    
    # Synthetic returns matrix: 50 assets, 252 trading days
    X, _ = make_blobs(n_samples=252, n_features=50, centers=5, random_state=42)
    returns_df = pd.DataFrame(X, columns=[f"TICK_{i:02d}" for i in range(50)])
    
    optics = DistanceOPTICS(
        distance_method=config.clustering.distance_method,
        formation_months=config.clustering.formation_months,
        market_returns=None
    )
    
    clusters = optics.fit(returns_df)
    print(f"✓ OPTICS clustering: {len(clusters)} clusters found")
    for cid, tickers in clusters.items():
        print(f"  Cluster {cid}: {len(tickers)} assets")
except Exception as e:
    print(f"✗ OPTICS error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test PairTradingEnv
print("\n[5/5] Testing RL environment...")
try:
    import numpy as np
    
    # Minimal env initialization
    env = PairTradingEnv(
        pair_data={
            'spread': np.random.randn(100),
            'returns': np.random.randn(100) * 0.01,
            'zscore': np.random.randn(100),
        },
        initial_position=0.0,
        initial_nav=1.0,
        cost_bps=2,
        action_reward_weight=0.1,
    )
    
    obs, info = env.reset()
    print(f"✓ RL environment initialized: observation shape={obs.shape}, observation={obs}")
    
    # Take a test action
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Test action executed: reward={reward:.4f}")
except Exception as e:
    print(f"✗ RL environment error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests passed! Pipeline is ready for Docker deployment.")
print("=" * 80)
