#!/usr/bin/env python3
"""Direct runner for RL pipeline with env vars set"""
import os
import sys
from pathlib import Path

# Set env vars
os.environ['STOOQ_API_KEY'] = 'LErbM8NDJlqTHAcVWK0yziSCPBL5sQod'
os.environ['LOG_LEVEL'] = 'INFO'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Run pipeline
if __name__ == "__main__":
    from src.pipeline.run import main
    main()
