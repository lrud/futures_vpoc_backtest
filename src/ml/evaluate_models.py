"""
Model evaluation script for comparing model performance across contracts.
Helps validate that models work correctly on both ES and VIX data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logging import setup_logging, get_logger
from src.ml.model import ModelManager
from src.ml.backtest_integration import MLBacktestIntegrator
from src.core.data import FuturesDataManager
from src.ml.feature_engineering import FeatureEngineer

# Initialize logger
logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ML models on ES and VIX data"
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs='+',
        required=True,
        help="Paths to model files to evaluate"
    )
    
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default=None,
        help="Path to test data (if different from training data)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Directory for evaluation results"
    )
    
    parser.add_argument(
        "--contracts", "-c",
        type=str,
        nargs='+',
        default=["ES", "VIX"],
        help="Contracts to evaluate on"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def evaluate_model(model_path, data, contract=None, output_dir=None):
    """
    Evaluate a model on specific data.
    
    Parameters:
    -----------
    model_path: str
        Path to model file
    data: pd.DataFrame
        Data to evaluate on
    contract: str
        Contract type for filtering
    output_dir: str
        Output directory for results
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    try:
        # Filter data by contract if specified
        test_data = data
        if contract:
            test_data = data[data['contract'].str.startswith(contract)]
            if len(test_data) == 0:
                logger.error(f"No data found for contract: {contract}")
                return None
            
        # Create output directory
        if output_dir:
            eval_dir = os.path.join(output_dir, f"{contract}_eval")
            os.makedirs(eval_dir, exist_ok=True)
        else:
            eval_dir = os.path.dirname(model_path)
        
        logger.info(f"Evaluating model {os.path.basename(model_path)} on {contract} data")
        logger.info(f"Output directory: {eval_dir}")
        
        # Create ML backtest integrator
        integrator = MLBacktestIntegrator(
            model_path=model_path,
            output_dir=eval_dir,
            confidence_threshold=60  # Lower for more signals
        )
        
        # Run backtest
        results = integrator.run_backtest(test_data)
        
        if "error" in results and "features" not in results:
            logger.error(f"Backtest failed: {results['error']}")
            return None
        
        # Log performance summary
        if "performance" in results:
            perf = results["performance"]
            logger.info("=" * 50)
            logger.info(f"MODEL EVALUATION ON {contract}")
            logger.info("=" * 50)
            logger.info(f"Total trades: {perf.get('total_trades', 0)}")
            logger.info(f"Win rate: {perf.get('win_rate', 0):.2f}%")
            logger.info(f"Total profit: ${perf.get('total_profit', 0):.2f}")
            logger.info(f"Sharpe ratio: {perf.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max drawdown: {perf.get('max_drawdown', 0):.2f}%")
            logger.info("=" * 50)
            
            # Save performance metrics
            performance_path = os.path.join(eval_dir, f"{contract}_performance.csv")
            pd.DataFrame([perf]).to_csv(performance_path, index=False)
            
        return results
            
    except Exception as e:
        logger.error(f"Error evaluating model on {contract} data: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(evaluation_results, output_dir):
    """
    Compare performance of models across contracts.
    
    Parameters:
    -----------
    evaluation_results: dict
        Dictionary of evaluation results
    output_dir: str
        Output directory for comparison
        
    Returns:
    --------
    pd.DataFrame
        Comparison DataFrame
    """
    # Extract key metrics
    comparison_data = []
    
    for model_name, contract_results in evaluation_results.items():
        for contract, result in contract_results.items():
            if result is None or "performance" not in result:
                continue
                
            perf = result["performance"]
            
            comparison_data.append({
                'model_name': model_name,
                'contract': contract,
                'trades': perf.get('total_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'profit': perf.get('total_profit', 0),
                'sharpe': perf.get('sharpe_ratio', 0),
                'drawdown': perf.get('max_drawdown', 0),
                'profit_factor': perf.get('profit_factor', 0)
            })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) == 0:
        logger.error("No valid evaluation results for comparison")
        return None
        
    # Save comparison
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot metrics by contract and model
    metrics = [
        ('win_rate', 'Win Rate (%)', axes[0, 0]),
        ('profit', 'Total Profit ($)', axes[0, 1]),
        ('sharpe', 'Sharpe Ratio', axes[1, 0]),
        ('drawdown', 'Max Drawdown (%)', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        for contract in comparison_df['contract'].unique():
            contract_data = comparison_df[comparison_df['contract'] == contract]
            ax.bar(
                [f"{row['model_name']}_{contract}" for _, row in contract_data.iterrows()],
                contract_data[metric],
                alpha=0.7,
                label=contract
            )
            
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, alpha=0.3)
    
    axes[0, 0].legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Model comparison saved to {comparison_path}")
    logger.info(f"Comparison plot saved to {plot_path}")
    
    return comparison_df

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to subdirectory in first model directory
        output_dir = os.path.join(os.path.dirname(args.models[0]), "evaluation_results")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    logger.info("Loading test data")
    data_manager = FuturesDataManager()
    
    if args.data_path:
        data = data_manager.load_futures_data(args.data_path)
    else:
        from src.config.settings import settings
        data = data_manager.load_futures_data(settings.DATA_DIR)
    
    if data is None:
        logger.error("Failed to load data")
        return 1
    
    # Evaluate each model on each contract
    evaluation_results = {}
    
    for model_path in args.models:
        model_name = os.path.basename(model_path).replace(".pt", "")
        evaluation_results[model_name] = {}
        
        for contract in args.contracts:
            result = evaluate_model(
                model_path=model_path,
                data=data,
                contract=contract,
                output_dir=output_dir
            )
            
            evaluation_results[model_name][contract] = result
    
    # Compare models
    comparison = compare_models(evaluation_results, output_dir)
    
    if comparison is None:
        logger.error("Model comparison failed")
        return 1
        
    logger.info("Evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
