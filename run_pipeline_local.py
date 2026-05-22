#!/usr/bin/env python3
"""
Complete RL Pipeline Runner with Full Monitoring
Trains A2C agent for pairs trading position sizing optimization
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Set environment variables
os.environ['STOOQ_API_KEY'] = 'LErbM8NDJlqTHAcVWK0yziSCPBL5sQod'
os.environ['LOG_LEVEL'] = 'INFO'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_output.log')
    ]
)

logger = logging.getLogger(__name__)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_pipeline():
    """Run the complete pipeline with RL training"""
    
    print_section("STATARB PAIRS TRADING PIPELINE - RL ENABLED")
    
    try:
        # Import pipeline components
        logger.info("Loading pipeline components...")
        from src.pipeline.run import run_pipeline, main
        from src.config import load_config
        import pandas as pd
        
        # Load configuration
        logger.info("Loading configuration from config/base.yaml...")
        config = load_config("config/base.yaml")
        
        # Print config details
        print_section("CONFIGURATION DETAILS")
        logger.info(f"Data start: {config.data.start}")
        logger.info(f"Data end: {config.data.end}")
        logger.info(f"Tickers: {config.data.tickers}")
        logger.info(f"Clustering mode: {config.clustering.mode}")
        logger.info(f"RL enabled: {config.rl.enabled}")
        logger.info(f"RL algorithm: {config.rl.algo}")
        logger.info(f"RL timesteps: {config.rl.total_timesteps}")
        logger.info(f"Transaction cost (bps): {config.rl.transaction_cost_bps}")
        logger.info(f"Turnover penalty: {config.rl.turnover_penalty}")
        logger.info(f"Drawdown penalty: {config.rl.drawdown_penalty}")
        
        # Run pipeline
        print_section("RUNNING PIPELINE")
        logger.info("Starting pipeline execution...")
        start_time = datetime.now()
        
        run_pipeline(config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("PIPELINE COMPLETED")
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Check outputs
        print_section("OUTPUT FILES VERIFICATION")
        data_dir = Path("data")
        
        output_files = {
            "pairs.csv": "Selected pairs with signals",
            "rule_backtest.csv": "Baseline rule-based backtest",
            "rl_model": "Trained A2C agent (directory/zip)"
        }
        
        for filename, description in output_files.items():
            filepath = data_dir / filename
            if filepath.exists():
                if filepath.is_file():
                    size = filepath.stat().st_size / 1024
                    logger.info(f"✓ {filename:25} | {size:10.2f} KB | {description}")
                else:
                    logger.info(f"✓ {filename:25} | (directory)      | {description}")
            else:
                logger.warning(f"✗ {filename:25} | NOT FOUND | {description}")
        
        # Load and display pairs data
        print_section("PAIRS DATA SUMMARY")
        if (data_dir / "pairs.csv").exists():
            pairs_df = pd.read_csv(data_dir / "pairs.csv")
            logger.info(f"Total pairs: {len(pairs_df)}")
            logger.info(f"Columns: {', '.join(pairs_df.columns.tolist())}")
            logger.info("\nFirst 5 pairs:")
            logger.info(pairs_df.head().to_string())
        
        # Load and display rule backtest results
        print_section("RULE-BASED BACKTEST RESULTS")
        if (data_dir / "rule_backtest.csv").exists():
            rule_df = pd.read_csv(data_dir / "rule_backtest.csv")
            logger.info(f"Backtest rows: {len(rule_df)}")
            logger.info(f"Columns: {', '.join(rule_df.columns.tolist())}")
            
            if len(rule_df) > 0:
                logger.info("\nPerformance metrics:")
                if 'nav' in rule_df.columns:
                    final_nav = rule_df['nav'].iloc[-1]
                    total_return = (final_nav - 1.0) * 100
                    logger.info(f"  Final NAV: {final_nav:.4f}")
                    logger.info(f"  Total Return: {total_return:.2f}%")
                
                if 'drawdown' in rule_df.columns:
                    max_dd = rule_df['drawdown'].max()
                    logger.info(f"  Max Drawdown: {max_dd:.4f}")
                
                logger.info("\nLast 5 rows:")
                logger.info(rule_df.tail().to_string())
        
        # Check for RL model
        print_section("RL MODEL TRAINING")
        if (data_dir / "rl_model").exists():
            logger.info("✓ RL model successfully saved")
            try:
                from stable_baselines3 import A2C
                model = A2C.load(str(data_dir / "rl_model"))
                logger.info(f"  Model type: {type(model).__name__}")
                logger.info(f"  Policy: {model.policy}")
            except Exception as e:
                logger.warning(f"  Could not load model details: {e}")
        else:
            logger.warning("✗ RL model not found - check logs for training errors")
        
        print_section("SUCCESS")
        logger.info("Pipeline execution completed successfully!")
        logger.info(f"Check 'pipeline_output.log' for detailed logs")
        logger.info(f"Outputs available in: {data_dir.resolve()}")
        
        return True
        
    except Exception as e:
        print_section("ERROR - PIPELINE FAILED")
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
