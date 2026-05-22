#!/usr/bin/env python3
"""
Environment Diagnostics and Dependency Check
Verifies all required components are installed and working
"""

import sys
import os
from pathlib import Path

print("="*80)
print("  ENVIRONMENT DIAGNOSTICS & DEPENDENCY CHECK")
print("="*80)
print()

# Check Python version
print(f"✓ Python version: {sys.version}")
print(f"✓ Python executable: {sys.executable}")
print()

# Check required packages
required_packages = {
    'numpy': 'NumPy for numerical operations',
    'pandas': 'Pandas for data manipulation',
    'scipy': 'SciPy for scientific computing',
    'sklearn': 'scikit-learn for machine learning',
    'statsmodels': 'Statsmodels for statistics',
    'gymnasium': 'Gymnasium for RL environments',
    'stable_baselines3': 'Stable Baselines3 for RL algorithms',
    'torch': 'PyTorch for neural networks',
    'pyspark': 'PySpark for distributed computing',
    'pyarrow': 'PyArrow for data I/O',
}

print("Checking required packages:")
missing_packages = []

for package, description in required_packages.items():
    try:
        if package == 'sklearn':
            import sklearn
        elif package == 'pyspark':
            import pyspark
        elif package == 'stable_baselines3':
            import stable_baselines3
        else:
            __import__(package)
        print(f"  ✓ {package:25} - {description}")
    except ImportError as e:
        print(f"  ✗ {package:25} - MISSING ({e})")
        missing_packages.append(package)

print()

# Check data files
print("Checking data files:")
data_dir = Path("data")
data_files = {
    "prices.parquet": "Historical price data",
    "fundamentals.parquet": "Fundamental data",
    "market.parquet": "Market index data",
    "pairs.csv": "Selected trading pairs",
    "rule_backtest.csv": "Rule-based backtest results",
}

for filename, description in data_files.items():
    filepath = data_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {filename:30} - {size:8.2f} MB - {description}")
    else:
        print(f"  ✗ {filename:30} - MISSING - {description}")

print()

# Check configuration
print("Checking configuration:")
config_file = Path("config/base.yaml")
if config_file.exists():
    print(f"  ✓ {config_file} exists")
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"    - RL enabled: {config.get('rl', {}).get('enabled', False)}")
        print(f"    - RL algorithm: {config.get('rl', {}).get('algo', 'N/A')}")
        print(f"    - Total timesteps: {config.get('rl', {}).get('total_timesteps', 'N/A')}")
    except Exception as e:
        print(f"  ⚠ Could not parse config: {e}")
else:
    print(f"  ✗ {config_file} not found")

print()

# Check source files
print("Checking source code structure:")
src_modules = {
    "src/pipeline/run.py": "Main pipeline orchestrator",
    "src/rl/env.py": "RL trading environment",
    "src/rl/train.py": "RL training utilities",
    "src/backtest/backtester.py": "Backtesting engine",
    "src/clustering/distance_optics.py": "Clustering algorithm",
    "src/pairs/pair_selection.py": "Pair selection logic",
}

for filepath, description in src_modules.items():
    fpath = Path(filepath)
    if fpath.exists():
        lines = len(fpath.read_text().split('\n'))
        print(f"  ✓ {filepath:35} - {lines:5} lines - {description}")
    else:
        print(f"  ✗ {filepath:35} - MISSING - {description}")

print()

# Check environment variables
print("Checking environment variables:")
env_vars = {
    'STOOQ_API_KEY': 'API key for STOOQ data',
    'LOG_LEVEL': 'Logging level',
    'SPARK_MASTER_URL': 'Spark master URL (optional)',
}

for var, description in env_vars.items():
    value = os.getenv(var)
    if value:
        display_value = value[:20] + "..." if len(value) > 20 else value
        print(f"  ✓ {var:25} = {display_value:25} - {description}")
    else:
        if var == 'SPARK_MASTER_URL':
            print(f"  ~ {var:25} (not set, will use local) - {description}")
        else:
            print(f"  ⚠ {var:25} not set - {description}")

print()

# Test imports
print("Testing critical imports:")
try:
    from src.config import load_config
    print("  ✓ Can import src.config")
except Exception as e:
    print(f"  ✗ Cannot import src.config: {e}")

try:
    from src.rl.env import PairTradingEnv, EnvConfig
    print("  ✓ Can import src.rl.env")
except Exception as e:
    print(f"  ✗ Cannot import src.rl.env: {e}")

try:
    from src.rl.train import train_agent
    print("  ✓ Can import src.rl.train")
except Exception as e:
    print(f"  ✗ Cannot import src.rl.train: {e}")

try:
    from stable_baselines3 import A2C, PPO
    print("  ✓ Can import stable_baselines3")
except Exception as e:
    print(f"  ✗ Cannot import stable_baselines3: {e}")

print()

# Summary
print("="*80)
if missing_packages:
    print(f"✗ ISSUES FOUND: {len(missing_packages)} missing package(s)")
    print("  Install with: pip install " + " ".join(missing_packages))
    sys.exit(1)
else:
    print("✓ ALL CHECKS PASSED")
    print()
    print("Ready to run pipeline. Try one of these:")
    print("  1. python run_pipeline_local.py     (full pipeline with clustering)")
    print("  2. python train_rl_direct.py        (direct RL training)")
    print("  3. python launch_local.py           (standard launcher)")
    print()
    sys.exit(0)
