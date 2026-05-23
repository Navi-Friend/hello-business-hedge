#!/usr/bin/env python3
"""
Comprehensive backtest evaluation script.
Computes performance metrics for both rule-based and RL strategies.

Usage:
  python scripts/evaluate_backtest.py
  
Or in Docker:
  docker compose exec app python scripts/evaluate_backtest.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_metrics(nav_values: np.ndarray, returns: np.ndarray = None) -> dict:
    """
    Compute comprehensive performance metrics.
    
    Args:
        nav_values: Array of portfolio NAV values
        returns: Daily returns (auto-computed if not provided)
    Returns:
        Dictionary with metrics
    """
    if returns is None:
        returns = np.diff(nav_values) / nav_values[:-1]
    
    # Filter out NaN/inf
    returns = returns[np.isfinite(returns)]
    
    nav_values = nav_values[np.isfinite(nav_values)]
    
    # Basic metrics
    final_nav = nav_values[-1] if len(nav_values) > 0 else 1.0
    total_return = (final_nav - 1.0) * 100
    
    # Annualized metrics (assume 252 trading days per year)
    annual_return = (np.mean(returns) * 252) * 100 if len(returns) > 0 else 0
    
    # Volatility
    daily_vol = np.std(returns) if len(returns) > 0 else 0
    annual_vol = daily_vol * np.sqrt(252) * 100
    
    # Sharpe ratio
    sharpe = (np.mean(returns) / (daily_vol + 1e-8)) * np.sqrt(252) if len(returns) > 0 else 0
    
    # Max drawdown
    cummax = np.maximum.accumulate(nav_values)
    drawdowns = (1 - nav_values / cummax) * 100
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Calmar ratio
    calmar = annual_return / (max_drawdown + 1e-8) if max_drawdown > 0 else 0
    
    # Win rate (% positive days)
    win_rate = (np.sum(returns > 0) / len(returns) * 100) if len(returns) > 0 else 0
    
    # Profit factor
    gains = np.sum(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0
    losses = np.abs(np.sum(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0
    profit_factor = gains / (losses + 1e-8) if losses > 0 else (np.inf if gains > 0 else 0)
    
    # Cumulative return distribution
    cumulative = np.cumprod(1 + returns)
    best_day = np.max(returns) * 100
    worst_day = np.min(returns) * 100
    
    return {
        'final_nav': final_nav,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'best_day': best_day,
        'worst_day': worst_day,
        'num_days': len(returns),
    }


def print_metrics_table(title: str, metrics: dict) -> None:
    """Pretty print metrics table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    metrics_to_show = [
        ('Final NAV', 'final_nav', '.4f'),
        ('Total Return', 'total_return', '.2f', '%'),
        ('Annual Return', 'annual_return', '.2f', '%'),
        ('Annual Volatility', 'annual_volatility', '.2f', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
        ('Calmar Ratio', 'calmar_ratio', '.2f'),
        ('Max Drawdown', 'max_drawdown', '.2f', '%'),
        ('Win Rate', 'win_rate', '.2f', '%'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('Best Day', 'best_day', '.2f', '%'),
        ('Worst Day', 'worst_day', '.2f', '%'),
        ('Trading Days', 'num_days', 'd'),
    ]
    
    for row in metrics_to_show:
        label = row[0]
        key = row[1]
        fmt = row[2]
        suffix = row[3] if len(row) > 3 else ''
        
        value = metrics.get(key, 0)
        if np.isinf(value):
            value_str = "∞"
        elif np.isnan(value):
            value_str = "NaN"
        else:
            value_str = f"{value:{fmt}}{suffix}"
        
        print(f"  {label:<25} {value_str:>15}")


def evaluate_rule_backtest() -> dict:
    """Evaluate rule-based portfolio backtest results."""
    logger.info("Loading rule-based backtest results...")
    
    backtest_file = Path("data/rule_backtest.csv")
    if not backtest_file.exists():
        logger.warning(f"Backtest file not found: {backtest_file}")
        return {}
    
    try:
        df = pd.read_csv(backtest_file)
        nav = df['nav'].values
        metrics = compute_metrics(nav)
        if "selected_pairs" in df.columns:
            avg_pairs = float(df["selected_pairs"].mean())
            max_pairs = int(df["selected_pairs"].max())
            title = f"RULE-BASED BACKTEST (avg {avg_pairs:.1f}, max {max_pairs} pairs)"
        elif "num_pairs" in df.columns:
            avg_pairs = float(df["num_pairs"].mean())
            max_pairs = int(df["num_pairs"].max())
            title = f"RULE-BASED BACKTEST (avg {avg_pairs:.1f}, max {max_pairs} pairs)"
        else:
            title = "RULE-BASED BACKTEST"
        print_metrics_table(title, metrics)
        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate rule backtest: {e}")
        return {}


def evaluate_rl_model() -> dict:
    """Evaluate RL model performance."""
    logger.info("Loading RL model and evaluating...")

    rl_nav_file = Path("data/rl_test_nav.csv")
    if rl_nav_file.exists():
        try:
            rl_nav = pd.read_csv(rl_nav_file)
            if "nav" in rl_nav.columns and len(rl_nav) > 2:
                metrics = compute_metrics(rl_nav["nav"].to_numpy())
                print_metrics_table("RL-OPTIMIZED MODEL (out-of-sample)", metrics)
                return metrics
            logger.warning("RL out-of-sample file is empty; model evaluation skipped")
            return {}
        except Exception as e:
            logger.warning(f"Could not read RL out-of-sample file: {e}")
    
    logger.warning(
        "No RL out-of-sample NAV available. Run `docker compose exec app python run_rl.py` "
        "to train/evaluate RL, or use VERIFY_ONLY for rule-based checks."
    )
    return {}


def compare_strategies(rule_metrics: dict, rl_metrics: dict) -> None:
    """Compare rule-based vs RL strategies."""
    if not rule_metrics or not rl_metrics:
        logger.warning("Cannot compare: missing metrics for one or both strategies")
        return
    
    print(f"\n{'='*60}")
    print("  COMPARISON: RL vs RULE-BASED")
    print(f"{'='*60}")
    
    comparisons = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annual_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Calmar Ratio', 'calmar_ratio', ''),
        ('Win Rate', 'win_rate', '%'),
    ]
    
    for label, key, suffix in comparisons:
        rule_val = rule_metrics.get(key, 0)
        rl_val = rl_metrics.get(key, 0)
        
        if np.isnan(rule_val) or np.isinf(rule_val):
            rule_val = 0
        if np.isnan(rl_val) or np.isinf(rl_val):
            rl_val = 0
        
        # For drawdown, lower is better; for others, higher is better
        if 'Drawdown' in label:
            diff = rule_val - rl_val
            better = '↓' if diff > 0 else '↑'
        else:
            diff = rl_val - rule_val
            better = '↑' if diff > 0 else '↓'
        
        print(f"  {label:<20} Rule: {rule_val:>8.2f}{suffix}  |  "
              f"RL: {rl_val:>8.2f}{suffix}  |  "
              f"Δ {diff:>+7.2f}{suffix} {better}")


def main():
    """Main evaluation entry point."""
    print("\n" + "="*60)
    print("  BACKTEST EVALUATION - Rule-Based vs RL")
    print("="*60)
    
    # Evaluate both strategies
    rule_metrics = evaluate_rule_backtest()
    rl_metrics = evaluate_rl_model()
    
    # Compare
    compare_strategies(rule_metrics, rl_metrics)
    
    print("\n" + "="*60)
    print("  ✓ Evaluation complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
