"""
Script to run backtests using trained ML models.
"""

import os
import sys
import argparse
import glob
from datetime import datetime

from src.ml.backtest_integration import MLBacktestIntegrator
from src.utils.logging import get_logger, setup_logging
from src.config.settings import settings

# Initialize logger
logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments for backtesting."""
    parser = argparse.ArgumentParser(
        description="Run backtest with trained ML model"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default=None,
        help="Path to trained model file or directory (will use best_model.pt from latest training)"
    )
    
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default=settings.DATA_DIR,
        help="Path to historical data for backtesting"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=settings.BACKTEST_DIR,
        help="Directory to save backtest results"
    )
    
    parser.add_argument(
        "--prediction_threshold", "-pt",
        type=float,
        default=0.5,
        help="Threshold for model predictions"
    )
    
    parser.add_argument(
        "--confidence_threshold", "-ct",
        type=float,
        default=70.0,
        help="Minimum confidence for signal generation (percentage)"
    )
    
    parser.add_argument(
        "--contract", "-c",
        type=str,
        choices=['ES', 'VIX', 'ALL'],
        default="ALL",
        help="Contract to backtest (ES, VIX, or ALL)"
    )
    
    parser.add_argument(
        "--compare_baseline", "-cb",
        action="store_true",
        help="Compare ML strategy with baseline strategy"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def find_latest_model(base_dir=settings.TRAINING_DIR):
    """Find the latest trained model in the training directory."""
    try:
        # Find all training directories
        train_dirs = glob.glob(os.path.join(base_dir, "train_*"))
        
        if not train_dirs:
            logger.error(f"No training directories found in {base_dir}")
            return None
            
        # Sort by creation time (newest first)
        train_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # Find best model in the newest directory
        latest_dir = train_dirs[0]
        model_path = os.path.join(latest_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            logger.error(f"No best_model.pt found in {latest_dir}")
            return None
            
        logger.info(f"Found latest model: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None

def load_historical_data(data_path, contract_filter=None):
    """Load historical data for backtesting."""
    import pandas as pd
    
    try:
        # Check if data_path is a file or directory
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data from file: {data_path}, {len(data)} rows")
            
        elif os.path.isdir(data_path):
            # Find all CSV files in the directory
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            
            if not csv_files:
                logger.error(f"No CSV files found in {data_path}")
                return None
                
            # Load and concatenate all files
            dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                
                # Filter by contract if specified
                if contract_filter and contract_filter != "ALL":
                    if 'contract' in df.columns:
                        df = df[df['contract'] == contract_filter]
                
                dfs.append(df)
                
            data = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(data)} rows from {len(csv_files)} files")
            
        else:
            logger.error(f"Invalid data path: {data_path}")
            return None
            
        # Sort by date
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger.info("Starting backtest script")
    
    # Find model path if not specified
    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            logger.error("Could not find a trained model")
            return 1
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"backtest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Backtest results will be saved to: {output_dir}")
    
    # Load historical data
    data = load_historical_data(
        args.data_path,
        contract_filter=None if args.contract == "ALL" else args.contract
    )
    
    if data is None or data.empty:
        logger.error("Failed to load historical data")
        return 1
    
    # Initialize ML backtest integrator
    integrator = MLBacktestIntegrator(
        model_path=model_path,
        output_dir=output_dir,
        prediction_threshold=args.prediction_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    # Run backtest
    logger.info("Running backtest with ML model")
    results = integrator.run_backtest(data)
    
    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return 1
    
    # Compare with baseline if requested
    if args.compare_baseline and hasattr(integrator, 'compare_strategies'):
        logger.info("Running baseline strategy for comparison")
        
        # Use existing backtest_integration functionality for baseline comparison
        baseline_results = results.get('baseline', {})
        if baseline_results:
            # Compare strategies using the built-in method
            comparison = integrator.compare_strategies(results, baseline_results)
            
            # Log comparison results
            logger.info("Strategy Comparison:")
            for metric, value in comparison.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.2f}")
                else:
                    logger.info(f"  {metric}: {value}")
    
    logger.info(f"Backtest completed successfully. Results in {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
