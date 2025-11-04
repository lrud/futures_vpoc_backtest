"""
Robust ML-enhanced backtesting integration module.
Integrates robust ML models with the backtesting engine for financial time series.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger
from src.analysis.backtest import BacktestEngine
from src.config.settings import settings
from src.ml.feature_engineering_robust import RobustFeatureEngineer
from src.ml.model_robust import RobustFinancialNet, HuberLoss

logger = get_logger(__name__)

class RobustMLBacktestIntegrator:
    """
    Integrates Robust ML models with the backtesting engine.

    Key Features:
    - Uses robust feature engineering (top 5 important features)
    - Loads robust neural network models with Huber loss
    - Handles rank-transformed targets (0-1 bounded)
    - ROCm 7 optimized for consumer GPUs
    """

    def __init__(self,
                model_path: str,
                output_dir: Optional[str] = None,
                prediction_threshold: float = 0.5,
                confidence_threshold: float = 70.0):
        """
        Initialize robust ML backtest integrator.

        Args:
            model_path: Path to saved robust ML model
            output_dir: Directory for output files
            prediction_threshold: Threshold for model predictions (0-1 range)
            confidence_threshold: Minimum confidence for signal generation
        """
        self.logger = get_logger(__name__)
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join(settings.BACKTEST_DIR, 'robust_ml')
        os.makedirs(self.output_dir, exist_ok=True)

        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold

        # Create robust feature engineer
        self.feature_engineer = RobustFeatureEngineer()

        # Load model and scaling parameters
        self.model = self._load_robust_model()
        self.scaling_params = None
        self.feature_columns = None

        self.logger.info(f"✅ Initialized RobustMLBacktestIntegrator with model from {model_path}")

    def _load_robust_model(self) -> Optional[RobustFinancialNet]:
        """Load the robust ML model from saved checkpoint."""
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"🔄 Loading robust model on {device}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, RobustFinancialNet):
                # Direct model instance
                model = checkpoint
                self.logger.info("✅ Loaded direct model instance")
            elif "model_state_dict" in checkpoint:
                # State dict case - reconstruct model
                config = checkpoint.get("model_config", {})
                input_dim = config.get("input_dim", 5)  # Default for robust model
                hidden_dims = config.get("hidden_dims", [16, 8])  # Default robust architecture
                dropout_rate = config.get("dropout_rate", 0.1)

                # Create robust model
                model = RobustFinancialNet(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate,
                    use_residual=True
                )

                # Load state dict
                state_dict = checkpoint["model_state_dict"]

                # Handle distributed training
                if all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)

                # Load additional parameters
                self.feature_columns = checkpoint.get("feature_columns", self.feature_engineer.TOP_5_FEATURES)
                self.scaling_params = checkpoint.get("feature_statistics", {})

                self.logger.info(f"✅ Reconstructed robust model with {len(self.feature_columns)} features")
            else:
                # Unknown format
                self.logger.error("❌ Unknown checkpoint format")
                return None

            model = model.to(device)
            model.eval()

            # Log model info
            param_counts = model.count_parameters()
            self.logger.info(f"✅ Robust model loaded:")
            self.logger.info(f"  • Total parameters: {param_counts['total']:,}")
            self.logger.info(f"  • Feature columns: {self.feature_columns}")

            return model

        except Exception as e:
            self.logger.error(f"❌ Error loading robust model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _generate_signals_robust(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using robust model predictions.

        Args:
            features_df: DataFrame with engineered features (DatetimeIndex)
            data: Original OHLCV data (DatetimeIndex)

        Returns:
            DataFrame with trading signals
        """
        self.logger.info("🎯 Generating robust ML signals...")

        if self.model is None:
            self.logger.error("❌ No model loaded for signal generation")
            return pd.DataFrame()

        try:
            # Ensure we have the right features
            if self.feature_columns is None:
                self.feature_columns = self.feature_engineer.TOP_5_FEATURES

            # Select features expected by the model
            available_features = [col for col in self.feature_columns if col in features_df.columns]

            if len(available_features) != len(self.feature_columns):
                self.logger.error(f"❌ Feature mismatch: expected {self.feature_columns}, available {available_features}")
                return pd.DataFrame()

            # Prepare feature matrix
            X = features_df[available_features].values

            # Handle scaling if parameters are available
            if self.scaling_params:
                X_scaled = self._apply_scaling(X, available_features)
            else:
                # Use robust scaling from feature engineer
                X_scaled, _ = self.feature_engineer.scale_features(features_df[available_features])

            # Convert to tensor
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()

            # Convert predictions to signals
            # Robust model outputs are in 0-1 range (rank-transformed targets)
            pred_values = predictions.flatten()

            # Generate signals based on prediction thresholds
            signals = []
            for i, (idx, pred) in enumerate(zip(features_df.index, pred_values)):
                # High prediction (expected positive return percentile)
                if pred > (1 - self.prediction_threshold):
                    signal_type = "BUY"
                # Low prediction (expected negative return percentile)
                elif pred < self.prediction_threshold:
                    signal_type = "SELL"
                else:
                    continue  # No signal

                # Get price data for this timestamp
                if idx in data.index:
                    # Get the first occurrence of this timestamp
                    matching_rows = data[data.index == idx]
                    if len(matching_rows) > 0:
                        current_price = float(matching_rows.iloc[0]['close'])
                    else:
                        continue

                    # Calculate stop loss and target based on ATR or percentage
                    atr_period = 14
                    if i >= atr_period:
                        recent_data = data.iloc[max(0, i-atr_period):i+1]
                        high_low_range = recent_data['high'] - recent_data['low']
                        atr = float(high_low_range.rolling(window=atr_period).mean().iloc[-1])
                        if pd.isna(atr):
                            atr = current_price * 0.02  # Default 2%
                    else:
                        atr = current_price * 0.02  # Default 2%

                    # Set stop loss and target
                    if signal_type == "BUY":
                        stop_loss = current_price - (2 * atr)
                        target = current_price + (3 * atr)
                    else:  # SELL
                        stop_loss = current_price + (2 * atr)
                        target = current_price - (3 * atr)

                    signals.append({
                        'date': idx,
                        'signal': signal_type,
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'prediction': pred,
                        'confidence': abs(pred - 0.5) * 200  # Convert to percentage
                    })

            signals_df = pd.DataFrame(signals)

            if not signals_df.empty:
                # Filter by confidence threshold
                signals_df = signals_df[signals_df['confidence'] >= self.confidence_threshold]

                # Sort by date
                signals_df = signals_df.sort_values('date')

                self.logger.info(f"✅ Generated {len(signals_df)} trading signals")
                self.logger.info(f"  • BUY signals: {len(signals_df[signals_df['signal'] == 'BUY'])}")
                self.logger.info(f"  • SELL signals: {len(signals_df[signals_df['signal'] == 'SELL'])}")
                self.logger.info(f"  • Average confidence: {signals_df['confidence'].mean():.1f}%")
            else:
                self.logger.warning("⚠️ No signals generated")

            return signals_df

        except Exception as e:
            self.logger.error(f"❌ Error generating signals: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _apply_scaling(self, X: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """Apply saved scaling parameters to features."""
        if self.scaling_params is None:
            return X

        X_scaled = X.copy()
        for i, col in enumerate(feature_columns):
            if col in self.scaling_params:
                params = self.scaling_params[col]
                median = params.get('median', 0)
                scale = params.get('scale', 1)
                if scale > 0:
                    X_scaled[:, i] = (X[:, i] - median) / scale

        return X_scaled

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run complete robust ML-enhanced backtest pipeline.

        Args:
            data: DataFrame with raw OHLCV data

        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            self.logger.error("❌ No model loaded, cannot run backtest")
            return {"error": "No model loaded"}

        try:
            self.logger.info("🚀 Starting robust ML backtest pipeline...")

            # Validate input data
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.error(f"❌ Missing required columns: {missing_cols}")
                return {"error": f"Missing columns: {missing_cols}"}

            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                else:
                    self.logger.error("❌ Data must have datetime index or 'date' column")
                    return {"error": "Missing datetime information"}

            self.logger.info(f"📊 Processing {len(data):,} data points from {data.index.min()} to {data.index.max()}")

            # Generate robust features
            self.logger.info("🔧 Creating robust features...")
            features_df = self.feature_engineer.create_features_robust(data)

            if features_df.empty:
                self.logger.error("❌ No features generated")
                return {"error": "Feature generation failed"}

            # Align features and data
            common_index = features_df.index.intersection(data.index)
            features_df = features_df.loc[common_index]
            aligned_data = data.loc[common_index]

            self.logger.info(f"✅ Created {len(features_df.columns)} features for {len(features_df):,} data points")

            # Generate signals
            signals_df = self._generate_signals_robust(features_df, aligned_data)

            if signals_df.empty:
                self.logger.warning("⚠️ No signals generated for backtest")
                return {
                    "features": features_df,
                    "signals": pd.DataFrame(),
                    "backtest": {},
                    "performance": {},
                    "error": "No signals generated"
                }

            # Initialize backtest engine
            self.logger.info("⚙️ Initializing backtest engine...")
            backtest_engine = BacktestEngine(
                initial_capital=getattr(settings, 'INITIAL_CAPITAL', 100000),
                commission=getattr(settings, 'COMMISSION_PER_TRADE', 10),
                slippage=getattr(settings, 'SLIPPAGE', 0.25),
                risk_per_trade=getattr(settings, 'RISK_PER_TRADE', 0.01)
            )

            # Run backtest
            self.logger.info("🏃‍♂️ Running backtest...")
            backtest_engine.run_backtest(signals_df)

            # Get performance metrics
            performance_metrics = backtest_engine.performance_metrics

            # Save results
            self._save_backtest_results(features_df, signals_df, backtest_engine)

            self.logger.info("✅ Robust ML backtest completed successfully!")

            return {
                "features": features_df,
                "signals": signals_df,
                "backtest": backtest_engine,
                "performance": performance_metrics,
                "model_info": {
                    "type": "RobustFinancialNet",
                    "features": self.feature_columns,
                    "parameters": self.model.count_parameters() if self.model else {}
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Error in robust backtest pipeline: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _save_backtest_results(self,
                              features_df: pd.DataFrame,
                              signals_df: pd.DataFrame,
                              backtest_engine: BacktestEngine) -> None:
        """Save backtest results to output directory."""
        try:
            self.logger.info(f"💾 Saving backtest results to {self.output_dir}")

            # Save features
            features_path = os.path.join(self.output_dir, "robust_features.csv")
            features_df.to_csv(features_path)
            self.logger.info(f"✅ Saved features: {features_path}")

            # Save signals
            signals_path = os.path.join(self.output_dir, "robust_signals.csv")
            signals_df.to_csv(signals_path, index=False)
            self.logger.info(f"✅ Saved signals: {signals_path}")

            # Save trades
            if not backtest_engine.trades.empty:
                trades_path = os.path.join(self.output_dir, "robust_trades.csv")
                backtest_engine.save_trades(trades_path)
                self.logger.info(f"✅ Saved trades: {trades_path}")

            # Save performance report
            report_path = os.path.join(self.output_dir, "robust_performance_report.txt")
            report = backtest_engine.generate_report(report_path)
            self.logger.info(f"✅ Saved performance report: {report_path}")

            # Save performance plot
            plot_path = os.path.join(self.output_dir, "robust_performance_plot.png")
            backtest_engine.plot_performance(plot_path)
            self.logger.info(f"✅ Saved performance plot: {plot_path}")

            # Save model configuration
            config_path = os.path.join(self.output_dir, "robust_model_config.txt")
            with open(config_path, 'w') as f:
                f.write("Robust Model Configuration:\n\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Feature Columns: {self.feature_columns}\n")
                f.write(f"Prediction Threshold: {self.prediction_threshold}\n")
                f.write(f"Confidence Threshold: {self.confidence_threshold}\n")
                if self.model:
                    param_counts = self.model.count_parameters()
                    f.write(f"Total Parameters: {param_counts['total']:,}\n")
                    f.write(f"Linear Parameters: {param_counts['linear']:,}\n")
                    f.write(f"LayerNorm Parameters: {param_counts['layer_norm']:,}\n")
            self.logger.info(f"✅ Saved model config: {config_path}")

        except Exception as e:
            self.logger.error(f"❌ Error saving backtest results: {e}")

def find_latest_robust_model(base_dir: str = "TRAINING_ROBUST") -> Optional[str]:
    """Find the latest trained robust model."""
    try:
        # Check for robust training directory
        if not os.path.exists(base_dir):
            logger.error(f"Robust training directory not found: {base_dir}")
            return None

        # Look for model files
        model_files = []
        for file in os.listdir(base_dir):
            if file.endswith('.pth'):
                model_path = os.path.join(base_dir, file)
                model_files.append((model_path, os.path.getctime(model_path)))

        if not model_files:
            logger.error(f"No model files found in {base_dir}")
            return None

        # Sort by creation time (newest first)
        model_files.sort(key=lambda x: x[1], reverse=True)
        latest_model = model_files[0][0]

        logger.info(f"Found latest robust model: {latest_model}")
        return latest_model

    except Exception as e:
        logger.error(f"Error finding latest robust model: {e}")
        return None

def test_robust_backtest_integration():
    """Test the robust backtest integration."""
    logger.info("🧪 Testing Robust ML Backtest Integration...")

    # Find latest robust model
    model_path = find_latest_robust_model()
    if model_path is None:
        logger.error("❌ No robust model found for testing")
        return False

    # Load small test dataset
    data_path = "/workspace/DATA/MERGED/merged_es_vix_test.csv"
    if not os.path.exists(data_path):
        logger.error("❌ Test data not found")
        return False

    # Load sample data
    data = pd.read_csv(data_path, nrows=1000)  # Small sample for testing
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

    # Initialize integrator
    integrator = RobustMLBacktestIntegrator(
        model_path=model_path,
        output_dir="/workspace/BACKTEST_ROBUST_TEST"
    )

    # Run backtest
    results = integrator.run_backtest(data)

    # Check results
    if "error" in results:
        logger.error(f"❌ Robust backtest test failed: {results['error']}")
        return False

    logger.info("✅ Robust backtest integration test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_robust_backtest_integration()
    if success:
        print("✅ Robust ML Backtest Integration is ready for use!")
    else:
        print("❌ Robust backtest integration test failed!")