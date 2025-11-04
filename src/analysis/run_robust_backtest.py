#!/usr/bin/env python3
"""
Script to run robust ML-enhanced backtests with trained robust models.

This script integrates with the robust training pipeline and provides:
- Automatic robust model discovery
- Robust feature engineering (top 5 features)
- Rank-transformed target handling (0-1 bounded)
- ROCm 7 GPU optimization
- Comprehensive backtest performance metrics

Usage:
    python src/analysis/run_robust_backtest.py --model_path TRAINING_ROBUST/best_model.pth
    python src/analysis/run_robust_backtest.py --data_fraction 0.1 --verbose
"""

import os
import sys
import argparse
import glob
from datetime import datetime

# Add project root to path
sys.path.append('/workspace')

from src.ml.backtest_integration_robust import RobustMLBacktestIntegrator, find_latest_robust_model
from src.utils.logging import get_logger, setup_logging
from src.config.settings import settings

# Initialize logger
logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments for robust backtesting."""
    parser = argparse.ArgumentParser(
        description="Run robust ML-enhanced backtest with trained robust model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with latest robust model (auto-discovery)
  python src/analysis/run_robust_backtest.py

  # Specify model path explicitly
  python src/analysis/run_robust_backtest.py --model_path TRAINING_ROBUST/best_model.pth

  # Use subset of data for testing
  python src/analysis/run_robust_backtest.py --data_fraction 0.1 --verbose

  # Custom prediction thresholds
  python src/analysis/run_robust_backtest.py --prediction_threshold 0.3 --confidence_threshold 60
        """
    )

    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default=None,
        help="Path to trained robust model file (auto-discovery if not specified)"
    )

    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="DATA/MERGED/merged_es_vix_test.csv",
        help="Path to historical data for backtesting (default: DATA/MERGED/merged_es_vix_test.csv)"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Directory to save backtest results (default: BACKTEST/robust_ml_TIMESTAMP)"
    )

    parser.add_argument(
        "--prediction_threshold", "-pt",
        type=float,
        default=0.5,
        help="Threshold for model predictions (0-1 range, default: 0.5)"
    )

    parser.add_argument(
        "--confidence_threshold", "-ct",
        type=float,
        default=70.0,
        help="Minimum confidence for signal generation (percentage, default: 70.0)"
    )

    parser.add_argument(
        "--data_fraction", "-df",
        type=float,
        default=1.0,
        help="Fraction of data to use for backtesting (0.01-1.0, default: 1.0)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

def load_historical_data(data_path: str, data_fraction: float = 1.0):
    """Load and prepare historical data for robust backtesting."""
    import pandas as pd

    try:
        logger.info(f"📁 Loading historical data from {data_path}")

        if not os.path.exists(data_path):
            logger.error(f"❌ Data file not found: {data_path}")
            return None

        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"✅ Loaded {len(data):,} rows of data")

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None

        # Handle date column
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        elif not isinstance(data.index, pd.DatetimeIndex):
            logger.error("❌ Data must have a 'date' column or datetime index")
            return None

        # Sort by date
        data = data.sort_index()

        # Apply data sampling if requested
        if data_fraction < 1.0:
            if not (0.01 <= data_fraction <= 1.0):
                logger.error(f"❌ Invalid data_fraction: {data_fraction}. Must be between 0.01 and 1.0")
                return None

            logger.info(f"📊 Sampling {data_fraction*100:.1f}% of data")
            sample_size = max(1, int(len(data) * data_fraction))
            data = data.iloc[-sample_size:]  # Take most recent data
            logger.info(f"✅ Sampled {len(data):,} rows ({data_fraction*100:.1f}% of original)")

        # Log data range
        logger.info(f"📅 Data range: {data.index.min()} to {data.index.max()}")

        return data

    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def validate_arguments(args):
    """Validate command-line arguments."""
    errors = []

    # Validate prediction threshold
    if not (0.0 <= args.prediction_threshold <= 1.0):
        errors.append("prediction_threshold must be between 0.0 and 1.0")

    # Validate confidence threshold
    if not (0.0 <= args.confidence_threshold <= 100.0):
        errors.append("confidence_threshold must be between 0.0 and 100.0")

    # Validate data fraction
    if not (0.01 <= args.data_fraction <= 1.0):
        errors.append("data_fraction must be between 0.01 and 1.0")

    if errors:
        for error in errors:
            logger.error(f"❌ {error}")
        return False

    return True

def main():
    """Main execution function for robust backtesting."""
    # Parse arguments
    args = parse_arguments()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger.info("🚀 Starting Robust ML Backtest Engine")

    # Validate arguments
    if not validate_arguments(args):
        return 1

    # Log configuration
    logger.info("⚙️ Configuration:")
    logger.info(f"  • Model path: {args.model_path or 'Auto-discovery'}")
    logger.info(f"  • Data path: {args.data_path}")
    logger.info(f"  • Data fraction: {args.data_fraction*100:.1f}%")
    logger.info(f"  • Prediction threshold: {args.prediction_threshold}")
    logger.info(f"  • Confidence threshold: {args.confidence_threshold}%")

    # Find model path if not specified
    model_path = args.model_path
    if model_path is None:
        logger.info("🔍 Searching for latest robust model...")
        model_path = find_latest_robust_model()
        if model_path is None:
            logger.error("❌ Could not find a trained robust model")
            logger.info("💡 Train a robust model first using: python src/ml/train_robust.py")
            return 1

    # Verify model exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return 1

    logger.info(f"✅ Using model: {model_path}")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(settings.BACKTEST_DIR, f"robust_backtest_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"📁 Output directory: {args.output_dir}")

    # Load historical data
    data = load_historical_data(args.data_path, args.data_fraction)
    if data is None or data.empty:
        logger.error("❌ Failed to load historical data")
        return 1

    # Initialize robust ML backtest integrator
    logger.info("🔧 Initializing Robust ML Backtest Integrator...")
    integrator = RobustMLBacktestIntegrator(
        model_path=model_path,
        output_dir=args.output_dir,
        prediction_threshold=args.prediction_threshold,
        confidence_threshold=args.confidence_threshold
    )

    if integrator.model is None:
        logger.error("❌ Failed to initialize robust model")
        return 1

    # Run robust backtest
    logger.info("🏃‍♂️ Running robust ML backtest...")
    logger.info("  Using robust features: top 5 most predictive features")
    logger.info("  Using robust model: Huber loss + LayerNorm + Rank-transformed targets")

    results = integrator.run_backtest(data)

    # Check for errors
    if "error" in results:
        logger.error(f"❌ Robust backtest failed: {results['error']}")
        return 1

    # Display results summary
    logger.info("📊 Robust Backtest Results Summary:")

    performance = results.get("performance", {})
    if performance:
        logger.info("  🎯 Performance Metrics:")
        for metric, value in performance.items():
            if isinstance(value, float):
                if metric in ["win_rate", "total_return", "annualized_return", "max_drawdown"]:
                    logger.info(f"    • {metric.replace('_', ' ').title()}: {value:.2f}%")
                elif metric in ["sharpe_ratio", "sortino_ratio"]:
                    logger.info(f"    • {metric.replace('_', ' ').title()}: {value:.3f}")
                elif metric in ["total_profit"]:
                    logger.info(f"    • {metric.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    logger.info(f"    • {metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                logger.info(f"    • {metric.replace('_', ' ').title()}: {value}")

    # Signal statistics
    signals = results.get("signals", pd.DataFrame())
    if not signals.empty:
        logger.info("  📈 Signal Statistics:")
        logger.info(f"    • Total signals: {len(signals)}")
        buy_signals = len(signals[signals['signal'] == 'BUY'])
        sell_signals = len(signals[signals['signal'] == 'SELL'])
        logger.info(f"    • BUY signals: {buy_signals} ({buy_signals/len(signals)*100:.1f}%)")
        logger.info(f"    • SELL signals: {sell_signals} ({sell_signals/len(signals)*100:.1f}%)")
        if 'confidence' in signals.columns:
            logger.info(f"    • Average confidence: {signals['confidence'].mean():.1f}%")

    # Model information
    model_info = results.get("model_info", {})
    if model_info:
        logger.info("  🤖 Model Information:")
        logger.info(f"    • Type: {model_info.get('type', 'Unknown')}")
        logger.info(f"    • Features: {len(model_info.get('features', []))}")
        params = model_info.get('parameters', {})
        if params:
            logger.info(f"    • Total parameters: {params.get('total', 'Unknown'):,}")

    # List generated files
    logger.info("📁 Generated files:")
    for file in os.listdir(args.output_dir):
        file_path = os.path.join(args.output_dir, file)
        file_size = os.path.getsize(file_path)
        logger.info(f"  • {file} ({file_size:,} bytes)")

    logger.info(f"✅ Robust backtest completed successfully!")
    logger.info(f"📂 Results saved to: {args.output_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())