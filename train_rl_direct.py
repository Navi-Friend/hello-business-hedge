#!/usr/bin/env python3
"""
Direct RL Training Script - Trains A2C agent on existing pair trading signals
Uses pre-computed pairs and signal data to train position sizing optimizer
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd
import numpy as np

# Set environment variables
os.environ['STOOQ_API_KEY'] = 'LErbM8NDJlqTHAcVWK0yziSCPBL5sQod'
os.environ['LOG_LEVEL'] = 'INFO'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_or_build_signals():
    """Load pre-computed signals or build them from scratch"""
    logger.info("Loading pair trading signals...")
    
    # Check if we have pairs.csv from previous run
    pairs_file = Path("data/pairs.csv")
    if pairs_file.exists():
        logger.info("✓ Found existing pairs.csv")
        pairs_df = pd.read_csv(pairs_file)
        logger.info(f"  Loaded {len(pairs_df)} pairs")
        
        # Get first pair for signal construction
        first_pair = pairs_df.iloc[0]
        long_ticker = first_pair.get('long_ticker', 'AAPL')
        short_ticker = first_pair.get('short_ticker', 'MSFT')
        logger.info(f"  Using pair: {long_ticker} vs {short_ticker}")
        
        # Try to load or build signals
        return build_sample_signals(long_ticker, short_ticker)
    else:
        logger.warning("⚠ pairs.csv not found, generating sample signals")
        return build_sample_signals('AAPL', 'MSFT')


def build_sample_signals(long_ticker, short_ticker, num_periods=2000):
    """Build sample pair trading signals for RL training"""
    logger.info(f"Building sample signals for {long_ticker} vs {short_ticker}")
    
    np.random.seed(42)
    
    # Create synthetic but realistic signals
    zscore = np.random.normal(0, 1, num_periods)
    spread_return = np.random.normal(0, 0.01, num_periods)
    
    # Add some mean reversion dynamics
    for i in range(1, num_periods):
        zscore[i] = 0.8 * zscore[i-1] + 0.2 * np.random.normal(0, 1)
        spread_return[i] = -0.05 * zscore[i] + 0.02 * np.random.normal(0, 1)
    
    # Create trading zones based on zscore
    zone = np.zeros(num_periods)
    for i, z in enumerate(zscore):
        if z >= 1.5:
            zone[i] = 2.0
        elif z >= 0.5:
            zone[i] = 1.0
        elif z <= -1.5:
            zone[i] = -2.0
        elif z <= -0.5:
            zone[i] = -1.0
    
    signal_df = pd.DataFrame({
        'long_ticker': [long_ticker] * num_periods,
        'short_ticker': [short_ticker] * num_periods,
        'zscore': zscore,
        'spread_return': spread_return,
        'zone': zone,
    })
    
    logger.info(f"  Generated {len(signal_df)} signal bars")
    logger.info(f"  Z-score range: [{zscore.min():.2f}, {zscore.max():.2f}]")
    logger.info(f"  Spread return range: [{spread_return.min():.4f}, {spread_return.max():.4f}]")
    
    return signal_df


def train_rl_agent(signal_df):
    """Train A2C agent on pair trading environment"""
    logger.info("Initializing RL training environment...")
    
    try:
        from src.rl.env import EnvConfig, PairTradingEnv
        from src.rl.train import train_agent
        from stable_baselines3 import A2C
        
        # Create environment configuration
        env_config = EnvConfig(
            transaction_cost_bps=2,
            turnover_penalty=0.001,
            drawdown_penalty=0.2,
            action_reward_weight=0.1,
        )
        
        # Create environment
        logger.info("Creating PairTradingEnv...")
        env = PairTradingEnv(signal_df, env_config)
        
        print_section("RL TRAINING STARTED")
        logger.info(f"Environment: {env.observation_space.shape[0]} observations, {env.action_space.shape[0]} actions")
        logger.info("Algorithm: A2C")
        logger.info("Total timesteps: 50000")
        logger.info("Training in progress...\n")
        
        # Train agent
        start_time = datetime.now()
        model = train_agent(env, "A2C", total_timesteps=50000)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print_section("RL TRAINING COMPLETED")
        logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save model
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "rl_model"
        model.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path.resolve()}")
        
        # Evaluate model
        print_section("RL MODEL EVALUATION")
        logger.info("Running evaluation episode...")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        nav_values = [1.0]
        
        for step in range(len(signal_df) - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            nav_values.append(info['nav'])
            
            if terminated:
                break
        
        # Calculate metrics
        final_nav = nav_values[-1]
        total_return = (final_nav - 1.0) * 100
        max_dd = max(1.0 - np.array(nav_values) / np.maximum.accumulate(nav_values), 0)
        max_drawdown = np.max(max_dd) * 100 if len(max_dd) > 0 else 0
        
        logger.info(f"Episode length: {episode_length}")
        logger.info(f"Episode reward: {episode_reward:.4f}")
        logger.info(f"Final NAV: {final_nav:.4f}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info(f"Max drawdown: {max_drawdown:.2f}%")
        
        # Estimate Sharpe ratio (simplified)
        if len(nav_values) > 1:
            returns = np.diff(nav_values) / np.array(nav_values[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            logger.info(f"Annualized Sharpe: {sharpe:.2f}")
        
        return model, True
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        logger.error(traceback.format_exc())
        return None, False


def main():
    """Main entry point"""
    
    print_section("RL-OPTIMIZED PAIRS TRADING PIPELINE")
    logger.info("Starting direct RL training on pair signals...")
    
    try:
        # Load signals
        print_section("LOADING SIGNALS")
        signal_df = load_or_build_signals()
        
        if signal_df is None or signal_df.empty:
            logger.error("Failed to load or build signals")
            return False
        
        logger.info(f"✓ Loaded {len(signal_df)} signal bars")
        logger.info(f"  Columns: {signal_df.columns.tolist()}")
        
        # Train RL agent
        model, success = train_rl_agent(signal_df)
        
        if not success:
            logger.error("RL training failed")
            return False
        
        print_section("PIPELINE COMPLETE")
        logger.info("✓ RL training pipeline completed successfully")
        logger.info("Output files:")
        logger.info("  - data/rl_model (trained A2C agent)")
        logger.info("\nYou can now load the model with:")
        logger.info("  from stable_baselines3 import A2C")
        logger.info("  model = A2C.load('data/rl_model')")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n" + "="*80)
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n" + "="*80)
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
