"""
Command-line runner for ML backtesting.
Provides a simple interface for running ML backtests from the command line.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

from ..utils.logging import get_logger, setup_logging
from ..core.data import FuturesDataManager
from .backtest_integration import MLBacktestIntegrator
from ..config.settings import settings

logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    from .arguments import get_common_parser, add_model_args
    parser = get_common_parser("Run ML-enhanced backtest for futures trading")
    add_model_args(parser)
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        choices=["RTH", "ETH", ""],
        default="RTH",
        help="Session type to analyze (RTH, ETH, or empty for all)"
    )
    
    parser.add_argument(
        "--contract", "-c",
        type=str,
        default="ES",
        help="Contract filter (e.g., 'ES' for E-mini S&P)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Prediction threshold for signal generation"
    )
    
    parser.add_argument(
        "--confidence", "-cf",
        type=float,
        default=70.0,
        help="Confidence threshold for signal filtering"
    )
    
    parser.add_argument(
        "--compare", "-cp",
        type=str,
        default="",
        help="Path to baseline strategy results for comparison"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output directory with timestamp
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting ML backtest with model: {args.model}")
    logger.info(f"Data source: {args.data}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load data
        logger.info("Loading futures data")
        data_manager = FuturesDataManager()
        data = data_manager.load_futures_data(args.data)
        
        if data is None:
            logger.error("Failed to load data")
            return 1
            
        # Filter by session and contract
        if args.session:
            data = data[data['session'] == args.session]
        
        if args.contract:
            data = data[data['contract'].str.startswith(args.contract)]
            
        logger.info(f"Filtered data: {len(data)} rows")
        
        # Initialize backtest integrator
        integrator = MLBacktestIntegrator(
            model_path=args.model,
            output_dir=output_dir,
            prediction_threshold=args.threshold,
            confidence_threshold=args.confidence
        )
        
        # Run backtest
        results = integrator.run_backtest(data)
        
        # Check for errors
        if "error" in results and "features" not in results:
            logger.error(f"Backtest failed: {results['error']}")
            return 1
            
        # Print performance summary
        if "performance" in results:
            perf = results["performance"]
            logger.info("=" * 50)
            logger.info("ML BACKTEST RESULTS")
            logger.info("=" * 50)
            logger.info(f"Total trades: {perf.get('total_trades', 0)}")
            logger.info(f"Win rate: {perf.get('win_rate', 0):.2f}%")
            logger.info(f"Total profit: ${perf.get('total_profit', 0):.2f}")
            logger.info(f"Sharpe ratio: {perf.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max drawdown: {perf.get('max_drawdown', 0):.2f}%")
            logger.info("=" * 50)
            
        # Compare with baseline if provided
        if args.compare and os.path.exists(args.compare):
            try:
                # Load baseline results
                logger.info(f"Comparing with baseline strategy: {args.compare}")
                
                # This is a simplified approach - in real world we would need to
                # load the actual baseline backtest results properly
                baseline_file = os.path.join(args.compare, "performance.csv")
                if os.path.exists(baseline_file):
                    baseline_perf = pd.read_csv(baseline_file).to_dict(orient='records')[0]
                    baseline_results = {"performance": baseline_perf}
                    
                    # Compare strategies
                    comparison = integrator.compare_strategies(results, baseline_results)
                    
                    # Print comparison
                    logger.info("=" * 50)
                    logger.info("STRATEGY COMPARISON")
                    logger.info("=" * 50)
                    logger.info(f"ML vs Baseline win rate: {comparison.get('win_rate_change', 0):.2f}%")
                    logger.info(f"ML vs Baseline profit: {comparison.get('profit_change', 0):.2f}%")
                    logger.info(f"ML avg profit per trade: ${comparison.get('ml_avg_profit_per_trade', 0):.2f}")
                    logger.info(f"Baseline avg profit per trade: ${comparison.get('baseline_avg_profit_per_trade', 0):.2f}")
                    logger.info("=" * 50)
            except Exception as e:
                logger.error(f"Error in strategy comparison: {e}")
        
        logger.info(f"Backtest completed successfully. Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Critical error in ML backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
