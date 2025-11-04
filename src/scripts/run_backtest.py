#!/usr/bin/env python3
"""
Simple backtest runner for the most recent trained model.
"""

import sys
import os
sys.path.append('/workspace')

import pandas as pd
import numpy as np
from datetime import datetime

from src.ml.feature_engineering import FeatureEngineer
from src.ml.backtest_integration import MLBacktestIntegrator
from src.analysis.backtest import BacktestEngine
from src.utils.logging import get_logger

def main():
    """Run backtest on the most recent trained model."""

    logger = get_logger(__name__)
    logger.info("🚀 Starting backtest execution...")

    try:
        # Step 1: Load the data
        logger.info("📁 Loading backtest data...")
        data_path = '/workspace/DATA/MERGED/merged_es_vix_test.csv'
        data = pd.read_csv(data_path)
        logger.info(f"✅ Loaded {len(data)} rows of data")

        # Step 2: Find most recent model and initialize integrator
        logger.info("🔧 Finding most recent trained model...")
        model_path = "/workspace/TRAINING/train_20251031_135340/model.pt"

        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return 1

        logger.info(f"✅ Using model: {model_path}")
        integrator = MLBacktestIntegrator(model_path=model_path)
        logger.info("✅ ML integrator initialized")

        # Step 3: Run ML backtest
        logger.info("🏃‍♂️ Running ML-enhanced backtest...")
        results = integrator.run_backtest(data)

        # Step 4: Display results
        logger.info("📊 Backtest Results:")
        if results:
            for metric, value in results.items():
                if isinstance(value, float):
                    logger.info(f"   • {metric}: {value:.4f}")
                else:
                    logger.info(f"   • {metric}: {value}")
        else:
            logger.warning("No backtest results generated")

        # Step 5: Check for any generated files
        backtest_dir = '/workspace/BACKTEST'
        if os.path.exists(backtest_dir):
            files = os.listdir(backtest_dir)
            if files:
                logger.info(f"📁 Backtest files generated in {backtest_dir}:")
                for file in files:
                    logger.info(f"   • {file}")

        logger.info("✅ Backtest execution completed successfully!")

    except Exception as e:
        logger.error(f"❌ Backtest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())