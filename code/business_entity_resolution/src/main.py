"""
Amazon ML Challenge 2026: Business Entity Resolution
=====================================================
Main entry point for training, validation, testing, and full pipeline execution.

Usage:
  python code/business_entity_resolution/src/main.py --mode all
  python code/business_entity_resolution/src/main.py --mode train
  python code/business_entity_resolution/src/main.py --mode validate
  python code/business_entity_resolution/src/main.py --mode test
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", default="all", choices=["train", "validate", "test", "all"],
                        help="Pipeline mode: train, validate, test, or all")
    args = parser.parse_args()
    success = run_pipeline(args.mode)
    sys.exit(0 if success else 1)
