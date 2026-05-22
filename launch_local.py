#!/usr/bin/env python3
"""
Standalone launcher for StatArb pipeline (no Docker/Spark cluster needed).
For local testing, use this instead of Docker.

Run: python launch_local.py
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run the full StatArb pipeline locally"""
    logger.info("=" * 80)
    logger.info("StatArb Pairs Trading Pipeline - Local Mode (No Docker)")
    logger.info("=" * 80)
    
    try:
        from src.pipeline.run import main as run_pipeline
        
        logger.info("Starting pipeline...")
        run_pipeline()
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("Outputs saved to: ./output/")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
