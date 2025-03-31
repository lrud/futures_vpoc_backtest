"""
ML-enhanced backtesting integration module.
Brings together ML models and backtest engine.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple

from src.utils.logging import get_logger
from src.core.signals import SignalGenerator
from src.ml.feature_engineering import FeatureEngineer
from src.analysis.backtest import BacktestEngine
from src.config.settings import settings
from src.ml.model import AMDOptimizedFuturesModel

class MLBacktestIntegrator:
    """
    Integrates ML models with the backtesting engine.
    """
    
    def __init__(self, 
                model_path: str,
                output_dir: Optional[str] = None,
                prediction_threshold: float = 0.3, # Lowered threshold
                confidence_threshold: float = 60): # Lowered threshold
        """
        Initialize ML backtest integrator.
        
        Parameters:
        -----------
        model_path: str
            Path to saved ML model
        output_dir: Optional[str]
            Directory for output files
        prediction_threshold: float
            Threshold for model predictions
        confidence_threshold: float
            Minimum confidence for signal generation
        """
        self.logger = get_logger(__name__)
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join(settings.BACKTEST_DIR, 'ml')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold
        
        # Create feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Create signal generator
        self.signal_generator = SignalGenerator(output_dir=self.output_dir)
        
        # Load model
        self.model = self._load_model()
        
        self.logger.info(f"Initialized MLBacktestIntegrator with model from {model_path}")
    
    def _load_model(self):
        """Load the ML model from saved checkpoint."""
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=device)
            
            def _get_hidden_dims(hidden_layers):
                """Extract dimensions from hidden layers."""
                if isinstance(hidden_layers, (list, tuple)):
                    return hidden_layers
                elif hasattr(hidden_layers, 'hidden_dims'):
                    return hidden_layers.hidden_dims
                else:
                    return [64, 32]  # Default fallback
            
            # Initialize model with correct architecture
            if isinstance(checkpoint, AMDOptimizedFuturesModel):
                # Direct model instance
                self.model = checkpoint
            elif "model_state_dict" in checkpoint:
                # State dict case - initialize model first
                architecture = checkpoint.get("architecture", {})
                input_dim = architecture.get("input_dim", len(checkpoint.get("feature_columns", [])))
                if not input_dim:
                    raise ValueError("Could not determine model input dimensions")
                
                # Get architecture from checkpoint or use defaults
                hidden_layers = architecture.get("hidden_layers", [64, 32])
                dropout_rate = architecture.get("dropout_rate", 0.4)
                
                self.model = AMDOptimizedFuturesModel(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    dropout_rate=dropout_rate
                )
                
                # Handle distributed-trained models by stripping "module." prefix
                state_dict = checkpoint["model_state_dict"]
                if all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict)
                
                # --- Load feature columns from checkpoint ---
                if 'feature_columns' in checkpoint and checkpoint['feature_columns']:
                    self.model.feature_columns = checkpoint['feature_columns']
                    self.logger.info(f"Loaded {len(self.model.feature_columns)} feature columns from checkpoint.")
                else:
                    self.logger.warning("Feature columns not found or empty in checkpoint.")
                    # Attempt to infer from state_dict keys if possible (less reliable)
                    # This part might need adjustment based on how features relate to layer names
                    try:
                        # Example: Infer based on input layer size if named predictably
                        input_layer_weight_shape = state_dict.get('input_layer.weight', state_dict.get('module.input_layer.weight', None))
                        if input_layer_weight_shape is not None and len(input_layer_weight_shape.shape) == 2:
                             inferred_dim = input_layer_weight_shape.shape[1]
                             self.logger.warning(f"Attempting to infer feature columns based on input layer dim: {inferred_dim}")
                             # Cannot reliably get names, but can set expected dim
                             # self.model.feature_columns = [f'feature_{i}' for i in range(inferred_dim)] # Placeholder names
                        else:
                             self.logger.error("Could not load or infer feature columns from checkpoint.")
                             # Raise error or return None depending on desired behavior
                             # raise ValueError("Feature columns missing from model checkpoint")
                    except Exception as infer_e:
                         self.logger.error(f"Error attempting to infer feature columns: {infer_e}")

            else:
                # Unknown format - try to load directly
                self.model = checkpoint
                # Check if feature_columns exist directly on the loaded object
                if not hasattr(self.model, 'feature_columns') or not self.model.feature_columns:
                     self.logger.warning("Loaded model directly, but feature_columns attribute is missing or empty.")


            self.model = self.model.to(device)
            self.model.eval()
            
            # Final check after loading and moving to device
            if not hasattr(self.model, 'feature_columns') or not self.model.feature_columns:
                 self.logger.error("Model loaded, but feature_columns are still missing or empty.")
                 # Optionally return None or raise error if feature columns are critical
                 # return None 
            else:
                 self.logger.info(f"Model ready with {len(getattr(self.model, 'feature_columns', []))} feature columns.")

            return self.model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run complete ML-enhanced backtest pipeline.
        
        Parameters:
        -----------
        data: pd.DataFrame
            DataFrame with raw OHLCV data
            
        Returns:
        --------
        Dict
            Dictionary with backtest results
        """
        if self.model is None:
            self.logger.error("No model loaded, cannot run backtest")
            return {"error": "No model loaded"}
            
        # Validate input data format
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns in input data: {missing_cols}")
            return {"error": f"Missing required columns: {missing_cols}"}
            
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("Input data must have datetime index")
            return {"error": "Data must have datetime index"}
        
        try:
            # Generate features - ensure working with a copy
            self.logger.info("Generating features from raw data")
            working_data = data.copy()

            # Log data characteristics
            self.logger.info(f"Input data stats - "
                          f"Contracts: {working_data.get('contract', 'N/A')}, "
                          f"Sessions: {working_data.get('session', 'N/A')}, "
                          f"Range: {working_data.index.min()} to {working_data.index.max()}")

            # Ensure derived columns exist
            if 'bar_range' not in working_data.columns and all(c in working_data for c in ['high', 'low']):
                working_data['bar_range'] = working_data['high'] - working_data['low']
                self.logger.info("Calculated 'bar_range'")

            if 'bar_return' not in working_data.columns and all(c in working_data for c in ['open', 'close']):
                working_data['bar_return'] = (working_data['close'] - working_data['open']) / working_data['open'].replace(0, np.nan)
                self.logger.info("Calculated 'bar_return'")

            # Generate numeric features (now returns with DatetimeIndex)
            features_df = self.feature_engineer.prepare_features(working_data)

            if features_df.empty:
                self.logger.error("No features generated")
                return {"error": "Feature generation failed"}

            # features_df now has DatetimeIndex from feature_engineering

            # --- Add back contract/session info to features_df (needed for merging later) ---
            if 'contract' in working_data.columns:
                 features_df = features_df.merge(working_data[['contract']], left_index=True, right_index=True, how='left')
            if 'session' in working_data.columns:
                 features_df = features_df.merge(working_data[['session']], left_index=True, right_index=True, how='left')

            # --- Get expected features from model ---
            if not hasattr(self.model, 'feature_columns') or not self.model.feature_columns:
                 self.logger.error("Model does not have feature_columns attribute or it's empty.")
                 return {"error": "Model missing feature column information"}
            model_feature_columns = self.model.feature_columns

            # Ensure all expected columns are in features_df
            missing_model_cols = [col for col in model_feature_columns if col not in features_df.columns]
            if missing_model_cols:
                self.logger.error(f"Features DataFrame missing columns expected by model: {missing_model_cols}")
                return {"error": f"Missing model features: {missing_model_cols}"}

            # Create DataFrame with only the features the model expects
            features_for_prediction = features_df[model_feature_columns].copy()
            self.logger.info(f"Prepared feature matrix for prediction with shape: {features_for_prediction.shape}")

            # Validate features before signal generation
            if features_for_prediction.empty:
                self.logger.error("Empty features DataFrame for prediction")
                return {"error": "Empty features DataFrame for prediction"}

            # Log prediction feature matrix details
            self.logger.info(f"Prediction feature matrix shape: {features_for_prediction.shape}")
            self.logger.debug(f"Prediction feature columns: {features_for_prediction.columns.tolist()}")

            # Validate prediction feature matrix meets StandardScaler requirements
            if features_for_prediction.shape[1] < 1:
                self.logger.error("Prediction feature matrix must contain >= 1 feature for StandardScaler")
                return {"error": "Insufficient features for scaling"}

            # Check if prediction features are all zero (NaN filling happens inside signal generator if needed)
            if (features_for_prediction == 0).all().all():
                 self.logger.warning("All features sent for prediction are zero - check feature generation")

            # --- Generate ML signals using only expected features ---
            self.logger.info("Generating ML signals")
            signals_df = self.signal_generator.generate_ml_signals(
                features_for_prediction, # Pass the correctly shaped DataFrame
                self.model,
                prediction_threshold=self.prediction_threshold,
                confidence_threshold=self.confidence_threshold
            )

            # Merge contract/session info back if signals were generated
            if not signals_df.empty:
                 # signals_df returned by generate_ml_signals has the same DatetimeIndex

                 merge_cols = []
                 if 'contract' in features_df.columns: merge_cols.append('contract')
                 if 'session' in features_df.columns: merge_cols.append('session')
                 if merge_cols:
                     # Merge contract/session first using the shared DatetimeIndex
                     signals_df = signals_df.merge(features_df[merge_cols], left_index=True, right_index=True, how='left')
                     self.logger.info(f"Merged {merge_cols} info into signals")

                 # --- Reset index to create 'date' column ---
                 signals_df.reset_index(inplace=True) # Converts 'date' index to column
                 self.logger.info("Reset index to create 'date' column in signals_df")

            else:
                self.logger.warning("No signals generated")

            if signals_df.empty:
                self.logger.warning("No signals generated")
                return {
                    "features": features_df,
                    "signals": pd.DataFrame(),
                    "backtest": {},
                    "error": "No signals generated"
                }
                
            # Initialize backtest engine
            self.logger.info("Initializing backtest engine")
            backtest_engine = BacktestEngine(
                initial_capital=settings.INITIAL_CAPITAL,
                commission=settings.COMMISSION_PER_TRADE,
                slippage=settings.SLIPPAGE,
                risk_per_trade=settings.RISK_PER_TRADE
            )
            
            # Run backtest
            self.logger.info("Running backtest")
            backtest_engine.run_backtest(signals_df)
            
            # Generate performance metrics
            performance_metrics = backtest_engine.performance_metrics
            
            # Save results
            self._save_backtest_results(
                features_df, signals_df, backtest_engine
            )
            
            return {
                "features": features_df,
                "signals": signals_df,
                "backtest": backtest_engine,
                "performance": performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest pipeline: {e}")
            return {"error": str(e)}
    
    def compare_strategies(self, ml_results: Dict, baseline_results: Dict) -> Dict:
        """
        Compare ML strategy with baseline strategy.
        
        Parameters:
        -----------
        ml_results: Dict
            ML strategy backtest results
        baseline_results: Dict
            Baseline strategy backtest results
            
        Returns:
        --------
        Dict
            Comparison metrics
        """
        try:
            ml_perf = ml_results.get("performance", {})
            baseline_perf = baseline_results.get("performance", {})
            
            if not ml_perf or not baseline_perf:
                self.logger.error("Missing performance metrics for comparison")
                return {}
                
            # Calculate comparison metrics
            comparison = {
                "ml_trade_count": ml_perf.get("total_trades", 0),
                "baseline_trade_count": baseline_perf.get("total_trades", 0),
                "ml_win_rate": ml_perf.get("win_rate", 0),
                "baseline_win_rate": baseline_perf.get("win_rate", 0),
                "ml_profit": ml_perf.get("total_profit", 0),
                "baseline_profit": baseline_perf.get("total_profit", 0),
                "ml_max_drawdown": ml_perf.get("max_drawdown", 0),
                "baseline_max_drawdown": baseline_perf.get("max_drawdown", 0),
                "ml_sharpe": ml_perf.get("sharpe_ratio", 0),
                "baseline_sharpe": baseline_perf.get("sharpe_ratio", 0),
                "ml_avg_profit_per_trade": (
                    ml_perf.get("total_profit", 0) / ml_perf.get("total_trades", 1)
                ),
                "baseline_avg_profit_per_trade": (
                    baseline_perf.get("total_profit", 0) / baseline_perf.get("total_trades", 1)
                )
            }
            
            # Calculate improvement percentages
            if baseline_perf.get("win_rate", 0) > 0:
                comparison["win_rate_change"] = (
                    (ml_perf.get("win_rate", 0) / baseline_perf.get("win_rate", 1) - 1) * 100
                )
            
            if baseline_perf.get("total_profit", 0) > 0:
                comparison["profit_change"] = (
                    (ml_perf.get("total_profit", 0) / baseline_perf.get("total_profit", 1) - 1) * 100
                )
            
            # Generate visualization
            self._plot_strategy_comparison(
                ml_results.get("backtest"), 
                baseline_results.get("backtest")
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {"error": str(e)}
    
    def _save_backtest_results(self, 
                              features_df: pd.DataFrame,
                              signals_df: pd.DataFrame,
                              backtest_engine: BacktestEngine) -> None:
        """Save backtest results to output directory."""
        try:
            # Save features
            features_path = os.path.join(self.output_dir, "ml_features.csv")
            features_df.to_csv(features_path, index=False)
            
            # Save signals
            signals_path = os.path.join(self.output_dir, "ml_signals.csv")
            signals_df.to_csv(signals_path, index=False)
            
            # Save backtest trades
            trades_path = os.path.join(self.output_dir, "ml_trades.csv")
            backtest_engine.save_trades(trades_path)
            
            # Save performance report
            report_path = os.path.join(self.output_dir, "ml_performance_report.txt")
            backtest_engine.generate_report(report_path)
            
            # Save performance plot
            plot_path = os.path.join(self.output_dir, "ml_performance_plot.png")
            backtest_engine.plot_performance(plot_path)
            
            self.logger.info(f"Saved backtest results to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
    
    def _plot_strategy_comparison(self, 
                                ml_backtest: BacktestEngine, 
                                baseline_backtest: BacktestEngine) -> None:
        """Generate and save strategy comparison plot."""
        if ml_backtest is None or baseline_backtest is None:
            self.logger.warning("Missing backtest data for comparison plot")
            return
            
        try:
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. Equity Curves
            ax1 = axes[0]
            ml_equity = ml_backtest.equity_curve
            baseline_equity = baseline_backtest.equity_curve
            
            # Normalize to same length if needed
            min_len = min(len(ml_equity), len(baseline_equity))
            ml_equity = ml_equity[:min_len]
            baseline_equity = baseline_equity[:min_len]
            
            ax1.plot(ml_equity, color='blue', linewidth=2, label='ML Strategy')
            ax1.plot(baseline_equity, color='green', linewidth=2, label='Baseline Strategy')
            ax1.set_title('Strategy Comparison: Equity Curves', fontsize=14)
            ax1.set_ylabel('Account Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdowns
            ax2 = axes[1]
            
            # Calculate drawdowns
            def calc_drawdown(equity):
                peaks = pd.Series(equity).cummax()
                return (pd.Series(equity) - peaks) / peaks * 100
            
            ml_dd = calc_drawdown(ml_equity)
            baseline_dd = calc_drawdown(baseline_equity)
            
            ax2.plot(ml_dd, color='blue', linewidth=2, label='ML Strategy')
            ax2.plot(baseline_dd, color='green', linewidth=2, label='Baseline Strategy')
            ax2.set_title('Strategy Comparison: Drawdowns', fontsize=14)
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            comparison_path = os.path.join(self.output_dir, "strategy_comparison.png")
            plt.savefig(comparison_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Saved strategy comparison to {comparison_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating comparison plot: {e}")
